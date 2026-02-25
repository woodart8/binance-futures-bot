"""실거래 전용 전략/청산/진입 로직. live_trader_agent.py에서만 사용.

- 잔고: get_balance_usdt는 total 기준. 청산은 포지션 매칭 후 contracts>0일 때만 API 호출.
- 포지션 동기화: sync_state_from_exchange로 시작 시·매 루프 거래소 실제 포지션과 state 일치(로그와 실제 포지션 불일치 방지).
- 거래소 TP/SL: USE_EXCHANGE_TP_SL=True면 진입 직후 TAKE_PROFIT_LIMIT(익절 지정가), STOP_MARKET(손절 시장가) 등록. 익절/손절은 거래소가 처리, 박스 이탈만 봇이 TP/SL 취소 후 시장가 청산.
"""

import time
from typing import Any, Dict, Tuple

import pandas as pd

from config import (
    CONSECUTIVE_LOSS_LIMIT,
    DAILY_LOSS_LIMIT_PCT,
    LEVERAGE,
    POSITION_SIZE_PERCENT,
    SIDEWAYS_BOX_PERIOD,
    SIDEWAYS_PROFIT_TARGET,
    SIDEWAYS_STOP_LOSS_PRICE,
    SYMBOL,
    TIMEFRAME,
    REGIME_LOOKBACK_15M,
    TREND_PROFIT_TARGET,
    TREND_STOP_LOSS_PRICE,
    TREND_SLOPE_BARS,
    TREND_SLOPE_MIN_PCT,
    SIDEWAYS_BOX_EXIT_MARGIN_PCT,
    MA_LONG_PERIOD,
    MA_MID_PERIOD,
    MA_LONGEST_PERIOD,
    SIDEWAYS_BOX_RANGE_PCT_MIN,
    SIDEWAYS_BOX_RANGE_MIN,
    USE_EXCHANGE_TP_SL,
)
from data import compute_regime_15m
from strategy_core import REGIME_KR, swing_strategy_signal, get_sideways_box_bounds
from exit_logic import check_long_exit, check_short_exit, reason_to_display_message
from trade_logger import log_trade
from logger import log

def _place_exchange_tp_sl(
    exchange, symbol: str, is_long: bool, amount: float, entry_price: float, tp_pct: float, sl_pct: float
) -> Tuple[str, str]:
    """진입 후 거래소에 익절/손절 주문 등록. 반환: (tp_order_id, sl_order_id). 실패 시 (None, None)."""
    pct_per_leverage = 1.0 / LEVERAGE
    if is_long:
        tp_price = entry_price * (1 + tp_pct / 100 * pct_per_leverage)
        sl_price = entry_price * (1 - sl_pct / 100 * pct_per_leverage)
        close_side = "sell"
    else:
        tp_price = entry_price * (1 - tp_pct / 100 * pct_per_leverage)
        sl_price = entry_price * (1 + sl_pct / 100 * pct_per_leverage)
        close_side = "buy"
    tp_id, sl_id = None, None
    try:
        # 익절: 지정가(TAKE_PROFIT_LIMIT) — 목표가 도달 시 해당 가격에 limit 체결 (슬리피지 감소)
        tp_order = exchange.create_order(
            symbol, "TAKE_PROFIT_LIMIT", close_side, amount, tp_price, {"stopPrice": tp_price, "reduceOnly": True}
        )
        tp_id = tp_order.get("id") if isinstance(tp_order, dict) else None
    except Exception as e:
        log(f"익절 주문 등록 실패: {e}", "WARNING")
    try:
        sl_order = exchange.create_order(
            symbol, "STOP_MARKET", close_side, amount, None, {"stopPrice": sl_price, "reduceOnly": True}
        )
        sl_id = sl_order.get("id") if isinstance(sl_order, dict) else None
    except Exception as e:
        log(f"손절 주문 등록 실패: {e}", "WARNING")
    return (tp_id or "", sl_id or "")


def _cancel_tp_sl_orders(exchange, symbol: str, state: Dict[str, Any]) -> None:
    """박스 이탈 등으로 시장가 청산 전, 걸어둔 익절/손절 주문 취소."""
    for key in ("tp_order_id", "sl_order_id"):
        oid = state.get(key)
        if not oid:
            continue
        try:
            exchange.cancel_order(oid, symbol)
        except Exception:
            pass


# 바이낸스 선물 fetch_positions 반환 symbol이 "BTC/USDT" 또는 "BTC/USDT:USDT" 등으로 올 수 있음
def _position_matches(pos: dict, symbol: str, side: str) -> bool:
    ps = (pos.get("symbol") or "").strip()
    if not ps:
        return False
    sym_norm = symbol.replace("/", "")
    ps_norm = ps.replace("/", "").replace(":USDT", "")
    return (ps == symbol or ps_norm == sym_norm) and (pos.get("side") == side)


def get_balance_usdt(exchange) -> float:
    """거래소 USDT 총 잔고(total) 반환. 손익·진입 한도는 total 기준. 00/08/16 UTC 펀딩은 거래소가 자동 정산."""
    balance = exchange.fetch_balance()
    usdt_info = balance.get("USDT", {})
    total = float(usdt_info.get("total", 0) or 0)
    if total == 0:
        total = float(usdt_info.get("free", 0) or 0)
    return total


def init_live_state() -> Dict[str, Any]:
    return {
        "has_position": False,
        "is_long": False,
        "entry_price": 0.0,
        "entry_balance": 0.0,
        "entry_regime": "",
        "box_high_entry": 0.0,
        "box_low_entry": 0.0,
        "highest_price": 0.0,
        "lowest_price": float("inf"),
        "trailing_stop_active": False,
        "best_pnl_pct": 0.0,
        "daily_start_balance": 0.0,
        "daily_start_date": "",
        "consecutive_loss_count": 0,
        "last_candle_time": None,
        "tp_order_id": None,
        "sl_order_id": None,
    }


def sync_state_from_exchange(exchange, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    거래소 실제 포지션을 조회해 state를 맞춤. 재시작·수동 청산 등으로 로그와 실제가 어긋나는 것을 방지.
    반환: 업데이트된 state (포지션 없으면 has_position=False, 있으면 entry_price 등 채움).
    """
    try:
        positions = exchange.fetch_positions([SYMBOL])
    except Exception as e:
        log(f"포지션 동기화 조회 실패: {e}", "WARNING")
        return state

    for pos in positions:
        if not _position_matches(pos, SYMBOL, "long") and not _position_matches(pos, SYMBOL, "short"):
            continue
        contracts = float(pos.get("contracts", 0) or 0)
        if contracts <= 0:
            contracts = abs(float(pos.get("positionAmt", 0) or pos.get("info", {}).get("positionAmt", 0) or 0))
        if contracts <= 0:
            continue
        # 실제 포지션 있음
        is_long = pos.get("side", "").lower() == "long"
        entry_price = float(pos.get("entryPrice", 0) or pos.get("info", {}).get("entryPrice", 0) or 0)
        if entry_price <= 0:
            entry_price = float(pos.get("markPrice", 0) or pos.get("info", {}).get("markPrice", 0) or 0)
        try:
            balance = get_balance_usdt(exchange)
        except Exception:
            balance = state.get("entry_balance", 0.0)
        # 실제 거래소 기준으로 state 덮어씀 (재시작·수동 진입 시 진입가만 알 수 있음)
        highest = entry_price if is_long else (state.get("highest_price", 0) or 0)
        if is_long and state.get("highest_price"):
            highest = max(entry_price, state["highest_price"])
        lowest = entry_price if not is_long else (state.get("lowest_price", float("inf")) if state.get("lowest_price") != float("inf") else float("inf"))
        if not is_long and state.get("lowest_price") is not None and state.get("lowest_price") != float("inf"):
            lowest = min(entry_price, state["lowest_price"])
        state = {
            **state,
            "has_position": True,
            "is_long": is_long,
            "entry_price": entry_price,
            "entry_balance": state.get("entry_balance") or balance,
            "entry_regime": state.get("entry_regime") or "trend",
            "box_high_entry": state.get("box_high_entry") or 0.0,
            "box_low_entry": state.get("box_low_entry") or 0.0,
            "highest_price": highest,
            "lowest_price": lowest,
            "best_pnl_pct": state.get("best_pnl_pct", 0.0),
        }
        return state

    # 포지션 없음 → state도 포지션 없음으로
    if state.get("has_position"):
        log("[동기화] 거래소에 포지션 없음 — state를 포지션 없음으로 맞춤", "WARNING")
    return {
        **state,
        "has_position": False,
        "is_long": False,
        "entry_price": 0.0,
        "entry_regime": "",
        "box_high_entry": 0.0,
        "box_low_entry": 0.0,
        "highest_price": 0.0,
        "lowest_price": float("inf"),
        "best_pnl_pct": 0.0,
        "tp_order_id": None,
        "sl_order_id": None,
    }


def check_tp_sl_and_close(exchange, state: Dict[str, Any], current_price: float, df: pd.DataFrame) -> Tuple[Dict[str, Any], bool]:
    """
    새 5분봉이 뜬 뒤, 전봉 종가 기준 익절/손절 조건 만족 시 청산.
    포지션은 _position_matches로 심볼/사이드 매칭(바이낸스 BTC/USDT:USDT 등), 수량은 contracts 또는 positionAmt.
    contracts<=0이면 주문 없이 return(청산생략 로그만). 반환: (업데이트된 state, did_close).
    """
    if not state.get("has_position"):
        return (state, False)

    latest = df.iloc[-1]
    rsi = float(latest["rsi"])
    is_long = state["is_long"]
    entry_price = state["entry_price"]
    entry_balance = state["entry_balance"]
    entry_regime = state["entry_regime"]
    box_high_entry = state["box_high_entry"] or 0.0
    box_low_entry = state["box_low_entry"] or 0.0
    highest_price = state["highest_price"]
    lowest_price = state["lowest_price"]
    best_pnl_pct = state["best_pnl_pct"]
    daily_start_balance = state["daily_start_balance"]
    daily_start_date = state["daily_start_date"]
    consecutive_loss_count = state["consecutive_loss_count"]

    if len(df) >= REGIME_LOOKBACK_15M * 3:
        regime, _, _, _, _, price_history_15m, _, _, _ = compute_regime_15m(df, current_price)
    else:
        regime = "neutral"

    pnl_pct = (current_price - entry_price) / entry_price * LEVERAGE * 100 if is_long else (entry_price - current_price) / entry_price * LEVERAGE * 100
    if is_long:
        if current_price > highest_price:
            highest_price = current_price
        if pnl_pct > best_pnl_pct:
            best_pnl_pct = pnl_pct
    else:
        if current_price < lowest_price:
            lowest_price = current_price
        if pnl_pct > best_pnl_pct:
            best_pnl_pct = pnl_pct

    signal = None
    close_reason = ""
    if is_long:
        reg = entry_regime or ""
        reason = check_long_exit(
            regime=reg, pnl_pct=pnl_pct, price=current_price,
            entry_price=entry_price, best_pnl_pct=best_pnl_pct,
            box_high=box_high_entry, box_low=box_low_entry,
        )
        if reason:
            signal = "flat"
            close_reason = reason_to_display_message(reason, is_long=True)

    if signal is None and not is_long:
        reg = entry_regime or ""
        reason = check_short_exit(
            regime=reg, pnl_pct=pnl_pct, price=current_price,
            entry_price=entry_price, best_pnl_pct=best_pnl_pct,
            box_high=box_high_entry, box_low=box_low_entry,
        )
        if reason:
            signal = "flat"
            close_reason = reason_to_display_message(reason, is_long=False)

    # 거래소에 TP/SL 걸어둔 경우 익절·손절은 거래소가 처리, 봇은 박스 이탈만 시장가 청산
    if signal == "flat" and USE_EXCHANGE_TP_SL and (state.get("tp_order_id") or state.get("sl_order_id")):
        if "박스권" not in close_reason:
            signal = None
            close_reason = ""

    if signal != "flat":
        out = {**state, "highest_price": highest_price, "lowest_price": lowest_price, "best_pnl_pct": best_pnl_pct}
        return (out, False)

    if not close_reason:
        close_reason = "전략청산"
    _cancel_tp_sl_orders(exchange, SYMBOL, state)
    try:
        balance = get_balance_usdt(exchange)
    except Exception as e:
        log(f"잔고 조회 실패: {e}", "ERROR")
        return (state, False)
    try:
        positions = exchange.fetch_positions([SYMBOL])
    except Exception as e:
        log(f"포지션 조회 실패: {e}", "ERROR")
        return (state, False)

    contracts = 0.0
    side_to_close = "long" if is_long else "short"
    for pos in positions:
        if _position_matches(pos, SYMBOL, side_to_close):
            contracts = float(pos.get("contracts", 0) or 0)
            if contracts <= 0:
                contracts = abs(float(pos.get("positionAmt", 0) or pos.get("info", {}).get("positionAmt", 0) or 0))
            break

    if contracts <= 0:
        log(f"[청산생략] 포지션 미조회 contracts=0 (심볼/사이드: {SYMBOL} {side_to_close})", "WARNING")
        return (state, False)

    try:
        if is_long:
            exchange.create_market_sell_order(SYMBOL, contracts, {"reduceOnly": True})
        else:
            exchange.create_market_buy_order(SYMBOL, contracts, {"reduceOnly": True})
        side_str = "LONG" if is_long else "SHORT"
        regime_kr = REGIME_KR.get(entry_regime, entry_regime)
    except Exception as e:
        log(f"청산 주문 실패: {e}", "ERROR")
        return (state, False)

    time.sleep(1)  # 청산 정산 반영 대기 후 잔고 조회
    try:
        new_balance = get_balance_usdt(exchange)
    except Exception as e:
        new_balance = balance
    pnl = new_balance - entry_balance
    if pnl < 0:
        consecutive_loss_count += 1
    else:
        consecutive_loss_count = 0
    pnl_pct_final = (current_price - entry_price) / entry_price * LEVERAGE * 100 if is_long else (entry_price - current_price) / entry_price * LEVERAGE * 100
    side_str = "LONG" if is_long else "SHORT"
    regime_kr = REGIME_KR.get(entry_regime, entry_regime)
    log(f"{side_str} 청산 | {regime_kr} | {close_reason} | 진입={entry_price:.2f} 청산={current_price:.2f} 수익률={pnl_pct_final:+.2f}% 손익={pnl:+.2f} 잔고={new_balance:.2f}")
    log_trade(
        side="LONG" if is_long else "SHORT",
        entry_price=entry_price,
        exit_price=current_price,
        pnl=pnl,
        balance_after=new_balance,
        meta={
            "timeframe": TIMEFRAME,
            "symbol": SYMBOL,
            "mode": "live",
            "regime": entry_regime,
            "pnl_pct": round(pnl_pct_final, 2),
            "consecutive_loss": consecutive_loss_count,
            "reason": close_reason,
        },
    )
    new_state = {
        **state,
        "has_position": False,
        "is_long": False,
        "entry_price": 0.0,
        "entry_regime": "",
        "box_high_entry": 0.0,
        "box_low_entry": 0.0,
        "highest_price": 0.0,
        "lowest_price": float("inf"),
        "best_pnl_pct": 0.0,
        "entry_balance": new_balance,
        "consecutive_loss_count": consecutive_loss_count,
        "daily_start_balance": daily_start_balance,
        "daily_start_date": daily_start_date,
        "tp_order_id": None,
        "sl_order_id": None,
    }
    return (new_state, True)


def try_live_entry(
    exchange,
    state: Dict[str, Any],
    df: pd.DataFrame,
    current_price: float,
    *,
    log_hold_info: bool = True,
) -> Tuple[Dict[str, Any], bool]:
    """
    새 5분봉이 뜬 뒤, 전봉 종가 기준 진입 조건 만족 시 시장가 진입(수량은 전봉 종가 기준).
    log_hold_info=False면 [진입생략] 로그 생략(5분봉 시점에만 정보 출력할 때 사용).
    반환: (업데이트된 state, did_enter).
    """
    if state.get("has_position"):
        return (state, False)

    latest = df.iloc[-1]
    latest_time = latest["timestamp"]
    rsi = float(latest["rsi"])
    short_ma = float(latest["ma_short"])
    long_ma = float(latest["ma_long"])
    rsi_prev = float(df["rsi"].iloc[-2]) if len(df) >= 2 else None
    open_prev = float(df["open"].iloc[-2]) if len(df) >= 2 else None
    close_prev = float(df["close"].iloc[-2]) if len(df) >= 2 else None
    open_curr = float(latest["open"])

    if len(df) >= REGIME_LOOKBACK_15M * 3:
        regime, short_ma_15m, long_ma_15m, ma_50_15m, ma_100_15m, price_history_15m, rsi_15m, rsi_15m_prev, ma_long_history = compute_regime_15m(df, current_price)
    else:
        regime, short_ma_15m, long_ma_15m, ma_50_15m, ma_100_15m, price_history_15m, rsi_15m, rsi_15m_prev, ma_long_history = "neutral", 0.0, 0.0, 0.0, 0.0, [], 0.0, None, []
    if len(df) >= SIDEWAYS_BOX_PERIOD:
        tail = df.tail(SIDEWAYS_BOX_PERIOD + 1)
        price_history = list(zip(tail["high"].tolist(), tail["low"].tolist(), tail["close"].tolist()))
    else:
        price_history = df["close"].tolist()
    use_15m = len(price_history_15m) >= REGIME_LOOKBACK_15M
    regime_price_hist = price_history_15m if (use_15m and regime == "sideways") else None
    rsi_value = float(rsi_15m) if (use_15m and rsi_15m is not None) else rsi
    rsi_prev_use = rsi_15m_prev if (use_15m and rsi_15m_prev is not None) else rsi_prev

    signal = swing_strategy_signal(
        rsi_value=rsi_value,
        price=current_price,
        rsi_prev=rsi_prev_use,
        open_prev=open_prev,
        close_prev=close_prev,
        open_curr=open_curr,
        short_ma=short_ma,
        long_ma=long_ma,
        has_position=False,
        is_long=False,
        regime=regime,
        price_history=price_history,
        regime_short_ma=short_ma_15m if use_15m else None,
        regime_long_ma=long_ma_15m if use_15m else None,
        regime_ma_50=ma_50_15m if use_15m else None,
        regime_ma_100=ma_100_15m if use_15m else None,
        regime_price_history=regime_price_hist,
        regime_ma_long_history=ma_long_history if use_15m else None,
    )
    if signal not in ("long", "short"):
        return (state, False)

    try:
        balance = get_balance_usdt(exchange)
    except Exception as e:
        log(f"잔고 조회 실패: {e}", "ERROR")
        return (state, False)

    daily_start_balance = state["daily_start_balance"]
    daily_start_date = state["daily_start_date"]
    consecutive_loss_count = state["consecutive_loss_count"]
    current_date = latest_time.strftime("%Y-%m-%d") if hasattr(latest_time, "strftime") else str(latest_time)[:10]
    if daily_start_date != current_date:
        daily_start_balance = balance
        daily_start_date = current_date
        consecutive_loss_count = 0
    daily_loss_pct = 0.0
    if daily_start_balance > 0:
        daily_loss_pct = (daily_start_balance - balance) / daily_start_balance * 100
    daily_limit_hit = daily_loss_pct >= DAILY_LOSS_LIMIT_PCT
    consecutive_limit_hit = consecutive_loss_count >= CONSECUTIVE_LOSS_LIMIT
    if daily_limit_hit or consecutive_limit_hit:
        if log_hold_info:
            if daily_limit_hit:
                log(f"[진입생략] 일일손실한도 {daily_loss_pct:.1f}% | 잔고={balance:.2f}")
            else:
                log(f"[진입생략] 연속손실 {consecutive_loss_count}회 당일 중단 | 시그널={signal}")
        return (state, False)

    order_usdt = balance * POSITION_SIZE_PERCENT  # 마진(투입 담보)
    if order_usdt < 5.0:
        return (state, False)

    # 포지션 규모(노션널) = 마진 × 레버리지 → 주문 수량(BTC) = 노션널 / 가격
    amount = (order_usdt * LEVERAGE) / current_price

    try:
        if signal == "long":
            exchange.create_market_buy_order(SYMBOL, amount)
            regime_kr = REGIME_KR.get(regime, regime)
            is_trend = regime == "trend"
            tp_pct = TREND_PROFIT_TARGET if is_trend else SIDEWAYS_PROFIT_TARGET
            sl_pct = TREND_STOP_LOSS_PRICE if is_trend else SIDEWAYS_STOP_LOSS_PRICE
            pct_per_leverage = 1.0 / LEVERAGE
            tg = current_price * (1 + tp_pct / 100 * pct_per_leverage)
            st = current_price * (1 - sl_pct / 100 * pct_per_leverage)
            tp_oid, sl_oid = "", ""
            if USE_EXCHANGE_TP_SL:
                tp_oid, sl_oid = _place_exchange_tp_sl(exchange, SYMBOL, True, amount, current_price, tp_pct, sl_pct)
            entry_tp_sl = f" 목표가={tg:.2f} 손절가={st:.2f}"
            if USE_EXCHANGE_TP_SL and (tp_oid or sl_oid):
                entry_tp_sl += " (거래소 TP/SL 등록)"
            box_str = ""
            bounds = get_sideways_box_bounds(price_history_15m, REGIME_LOOKBACK_15M) if regime == "sideways" and use_15m and len(price_history_15m) >= REGIME_LOOKBACK_15M else None
            if bounds:
                bh, bl = bounds
                box_str = f" 박스 하단={bl:.2f} 상단={bh:.2f}"
                box_high_entry, box_low_entry = bh, bl
            else:
                box_high_entry, box_low_entry = 0.0, 0.0
            log(f"LONG 진입 | {regime_kr} | 가격={current_price:.2f} 잔고={balance:.2f} 투입={order_usdt:.2f} USDT{entry_tp_sl}{box_str}")
            new_state = {
                **state,
                "has_position": True,
                "is_long": True,
                "entry_price": current_price,
                "highest_price": current_price,
                "entry_regime": regime,
                "entry_balance": balance,
                "best_pnl_pct": 0.0,
                "box_high_entry": box_high_entry,
                "box_low_entry": box_low_entry,
                "daily_start_balance": daily_start_balance,
                "daily_start_date": daily_start_date,
                "consecutive_loss_count": consecutive_loss_count,
                "last_candle_time": latest_time,
                "tp_order_id": tp_oid,
                "sl_order_id": sl_oid,
            }
        else:
            exchange.create_market_sell_order(SYMBOL, amount)
            regime_kr = REGIME_KR.get(regime, regime)
            is_trend = regime == "trend"
            tp_pct = TREND_PROFIT_TARGET if is_trend else SIDEWAYS_PROFIT_TARGET
            sl_pct = TREND_STOP_LOSS_PRICE if is_trend else SIDEWAYS_STOP_LOSS_PRICE
            pct_per_leverage = 1.0 / LEVERAGE
            tg = current_price * (1 - tp_pct / 100 * pct_per_leverage)
            st = current_price * (1 + sl_pct / 100 * pct_per_leverage)
            tp_oid, sl_oid = "", ""
            if USE_EXCHANGE_TP_SL:
                tp_oid, sl_oid = _place_exchange_tp_sl(exchange, SYMBOL, False, amount, current_price, tp_pct, sl_pct)
            entry_tp_sl = f" 목표가={tg:.2f} 손절가={st:.2f}"
            if USE_EXCHANGE_TP_SL and (tp_oid or sl_oid):
                entry_tp_sl += " (거래소 TP/SL 등록)"
            box_str = ""
            bounds = get_sideways_box_bounds(price_history_15m, REGIME_LOOKBACK_15M) if regime == "sideways" and use_15m and len(price_history_15m) >= REGIME_LOOKBACK_15M else None
            if bounds:
                bh, bl = bounds
                box_str = f" 박스 하단={bl:.2f} 상단={bh:.2f}"
                box_high_entry, box_low_entry = bh, bl
            else:
                box_high_entry, box_low_entry = 0.0, 0.0
            log(f"SHORT 진입 | {regime_kr} | 가격={current_price:.2f} 잔고={balance:.2f} 투입={order_usdt:.2f} USDT{entry_tp_sl}{box_str}")
            new_state = {
                **state,
                "has_position": True,
                "is_long": False,
                "entry_price": current_price,
                "lowest_price": current_price,
                "entry_regime": regime,
                "entry_balance": balance,
                "best_pnl_pct": 0.0,
                "box_high_entry": box_high_entry,
                "box_low_entry": box_low_entry,
                "daily_start_balance": daily_start_balance,
                "daily_start_date": daily_start_date,
                "consecutive_loss_count": consecutive_loss_count,
                "last_candle_time": latest_time,
                "tp_order_id": tp_oid,
                "sl_order_id": sl_oid,
            }
        return (new_state, True)
    except Exception as e:
        log(f"진입 주문 실패: {e}", "ERROR")
        return (state, False)


def process_live_candle(exchange, state: Dict[str, Any], df: pd.DataFrame) -> Tuple[Dict[str, Any], bool, bool]:
    """
    새 5분봉이 뜬 뒤, 전봉 종가 기준 익절/손절/박스이탈 판단 후 청산 주문 실행.
    반환: (업데이트된 state, skip, did_close)
    skip=True 이면 60초 대기 후 재조회. did_close=True 이면 해당 턴에 5분 로그 생략.
    """
    latest = df.iloc[-1]
    latest_time = latest["timestamp"]
    price = float(latest["close"])
    rsi = float(latest["rsi"])
    short_ma = float(latest["ma_short"])
    long_ma = float(latest["ma_long"])

    has_position = state["has_position"]
    is_long = state["is_long"]
    entry_price = state["entry_price"]
    entry_balance = state["entry_balance"]
    entry_regime = state["entry_regime"]
    box_high_entry = state["box_high_entry"]
    box_low_entry = state["box_low_entry"]
    highest_price = state["highest_price"]
    lowest_price = state["lowest_price"]
    best_pnl_pct = state["best_pnl_pct"]
    daily_start_balance = state["daily_start_balance"]
    daily_start_date = state["daily_start_date"]
    consecutive_loss_count = state["consecutive_loss_count"]

    if len(df) >= REGIME_LOOKBACK_15M * 3:
        regime, short_ma_15m, long_ma_15m, ma_50_15m, ma_100_15m, price_history_15m, rsi_15m, rsi_15m_prev, ma_long_history = compute_regime_15m(df, price)
    else:
        regime, short_ma_15m, long_ma_15m, ma_50_15m, ma_100_15m, price_history_15m, rsi_15m, rsi_15m_prev, ma_long_history = "neutral", 0.0, 0.0, 0.0, 0.0, [], 0.0, None, []

    if len(df) >= SIDEWAYS_BOX_PERIOD:
        tail = df.tail(SIDEWAYS_BOX_PERIOD + 1)
        price_history = list(zip(tail["high"].tolist(), tail["low"].tolist(), tail["close"].tolist()))
    else:
        price_history = df["close"].tolist()
    use_15m = len(price_history_15m) >= REGIME_LOOKBACK_15M

    signal = None

    if has_position:
        pnl_pct = (price - entry_price) / entry_price * LEVERAGE * 100 if is_long else (entry_price - price) / entry_price * LEVERAGE * 100

        if signal is None and is_long:
            if price > highest_price:
                highest_price = price
            if pnl_pct > best_pnl_pct:
                best_pnl_pct = pnl_pct
            reg = entry_regime or ""
            reason = check_long_exit(
                regime=reg, pnl_pct=pnl_pct, price=price,
                entry_price=entry_price, best_pnl_pct=best_pnl_pct,
                box_high=box_high_entry or 0, box_low=box_low_entry or 0,
            )
            if reason:
                signal = "flat"

        if signal is None and not is_long:
            if price < lowest_price:
                lowest_price = price
            if pnl_pct > best_pnl_pct:
                best_pnl_pct = pnl_pct
            reg = entry_regime or ""
            reason = check_short_exit(
                regime=reg, pnl_pct=pnl_pct, price=price,
                entry_price=entry_price, best_pnl_pct=best_pnl_pct,
                box_high=box_high_entry or 0, box_low=box_low_entry or 0,
            )
            if reason:
                signal = "flat"

    try:
        balance = get_balance_usdt(exchange)
    except Exception as e:
        log(f"잔고 조회 실패: {e}", "ERROR")
        time.sleep(60)
        return (state, True, False)

    current_date = latest_time.strftime("%Y-%m-%d") if hasattr(latest_time, "strftime") else str(latest_time)[:10]
    if daily_start_date != current_date:
        daily_start_balance = balance
        daily_start_date = current_date
        consecutive_loss_count = 0

    daily_loss_pct = 0.0
    if daily_start_balance > 0:
        daily_loss_pct = (daily_start_balance - balance) / daily_start_balance * 100
    daily_limit_hit = daily_loss_pct >= DAILY_LOSS_LIMIT_PCT
    consecutive_limit_hit = consecutive_loss_count >= CONSECUTIVE_LOSS_LIMIT

    if has_position and signal == "flat":
        # 청산: 포지션 조회 → contracts>0일 때만 API 주문 후 잔고/손익 로그. contracts=0이면 [청산실패] 로그 후 return(API 안 감).
        close_reason = reason_to_display_message(reason, is_long) if reason else "전략청산"
        try:
            positions = exchange.fetch_positions([SYMBOL])
        except Exception as e:
            log(f"포지션 조회 실패: {e}", "ERROR")
            time.sleep(60)
            return (state, True, False)

        contracts = 0.0
        side_to_close = "long" if is_long else "short"
        for pos in positions:
            if _position_matches(pos, SYMBOL, side_to_close):
                contracts = float(pos.get("contracts", 0) or 0)
                if contracts <= 0:
                    contracts = abs(float(pos.get("positionAmt", 0) or pos.get("info", {}).get("positionAmt", 0) or 0))
                break

        if contracts <= 0:
            log(f"[청산실패] 포지션 미조회 contracts=0 (심볼/사이드: {SYMBOL} {side_to_close}) — API 청산 요청 안 감", "WARNING")
            return (state, False, False)

        _cancel_tp_sl_orders(exchange, SYMBOL, state)
        try:
            if is_long:
                exchange.create_market_sell_order(SYMBOL, contracts, {"reduceOnly": True})
            else:
                exchange.create_market_buy_order(SYMBOL, contracts, {"reduceOnly": True})
        except Exception as e:
            log(f"청산 주문 실패: {e}", "ERROR")
            return (state, True, False)

        time.sleep(1)  # 청산 정산 반영 대기 후 잔고 조회
        try:
            new_balance = get_balance_usdt(exchange)
        except Exception as e:
            log(f"청산 후 잔고 조회 실패: {e}", "ERROR")
            new_balance = balance
        pnl = new_balance - entry_balance
        if pnl < 0:
            consecutive_loss_count += 1
        else:
            consecutive_loss_count = 0
        pnl_pct = (price - entry_price) / entry_price * LEVERAGE * 100 if is_long else (entry_price - price) / entry_price * LEVERAGE * 100
        side_str = "LONG" if is_long else "SHORT"
        regime_kr = REGIME_KR.get(entry_regime, entry_regime)
        log(f"{side_str} 청산 | {regime_kr} | {close_reason} | 진입={entry_price:.2f} 청산={price:.2f} 수익률={pnl_pct:+.2f}% 손익={pnl:+.2f} 잔고={new_balance:.2f}")
        log_trade(
            side="LONG" if is_long else "SHORT",
            entry_price=entry_price,
            exit_price=price,
            pnl=pnl,
            balance_after=new_balance,
            meta={
                "timeframe": TIMEFRAME,
                "symbol": SYMBOL,
                "mode": "live",
                "regime": entry_regime,
                "pnl_pct": round(pnl_pct, 2),
                "consecutive_loss": consecutive_loss_count,
                "reason": close_reason,
            },
        )
        new_state = {
            **state,
            "has_position": False,
            "is_long": False,
            "entry_price": 0.0,
            "entry_regime": "",
            "box_high_entry": 0.0,
            "box_low_entry": 0.0,
            "highest_price": 0.0,
            "lowest_price": float("inf"),
            "best_pnl_pct": 0.0,
            "entry_balance": new_balance,
            "consecutive_loss_count": consecutive_loss_count,
            "daily_start_balance": daily_start_balance,
            "daily_start_date": daily_start_date,
            "last_candle_time": latest_time,
            "tp_order_id": None,
            "sl_order_id": None,
        }
        return (new_state, False, True)

    # 5분 상태 로그용: 갱신된 state 반영 (청산/진입 없을 때도 highest/lowest/best_pnl 등 반영)
    out = {
        **state,
        "highest_price": highest_price,
        "lowest_price": lowest_price,
        "best_pnl_pct": best_pnl_pct,
        "daily_start_balance": daily_start_balance,
        "daily_start_date": daily_start_date,
        "consecutive_loss_count": consecutive_loss_count,
        "last_candle_time": latest_time,
    }
    return (out, False, False)


def log_5m_status(exchange, state: Dict[str, Any], df: pd.DataFrame) -> None:
    """1분봉 생길 때마다 상태 로그 (pos_status, regime, box, 가격, RSI, 잔고, 미실현)."""
    latest = df.iloc[-1]
    price = float(latest["close"])
    rsi_5m = float(latest["rsi"])
    has_position = state["has_position"]
    is_long = state["is_long"]
    entry_price = state["entry_price"]
    entry_regime = state["entry_regime"]
    box_high_entry = state["box_high_entry"]
    box_low_entry = state["box_low_entry"]

    try:
        bal = get_balance_usdt(exchange)
    except Exception:
        bal = 0.0
    pos_status = "LONG" if has_position and is_long else ("SHORT" if has_position else "NONE")
    unrealized = ""
    if has_position:
        pnl_pct = (price - entry_price) / entry_price * LEVERAGE * 100 if is_long else (entry_price - price) / entry_price * LEVERAGE * 100
        unrealized = f" 미실현={pnl_pct:+.2f}%"
    regime_str = f" | {REGIME_KR.get(entry_regime, entry_regime)}" if has_position and entry_regime else ""
    box_str = ""
    if has_position and entry_regime == "sideways" and box_high_entry > 0 and box_low_entry > 0:
        box_str = f" | 박스 하단={box_low_entry:.2f} 상단={box_high_entry:.2f}"
    
    # 현재 장세 판단 및 상세 정보 계산
    regime_detail = ""
    rsi_15m = None
    if len(df) >= REGIME_LOOKBACK_15M * 3:
        regime, short_ma_15m, long_ma_15m, ma_50_15m, ma_100_15m, price_history_15m, rsi_15m, rsi_15m_prev, ma_long_history = compute_regime_15m(df, price)
        
        if regime == "trend":
            # 추세장: MA20 기울기 정보
            if ma_long_history and len(ma_long_history) >= TREND_SLOPE_BARS:
                recent_ma20 = ma_long_history[-TREND_SLOPE_BARS:]
                ma20_start = recent_ma20[0]
                ma20_end = recent_ma20[-1]
                if ma20_start and ma20_start > 0:
                    slope_pct = (ma20_end - ma20_start) / ma20_start * 100.0
                    trend_dir = "상승" if slope_pct > 0 else "하락"
                    regime_detail = f" | 추세장({trend_dir}): MA20 기울기 {slope_pct:+.2f}% (기준 ±{TREND_SLOPE_MIN_PCT}%)"
        elif regime == "sideways":
            # 횡보장: 박스 정보
            bounds = get_sideways_box_bounds(price_history_15m, REGIME_LOOKBACK_15M)
            if bounds:
                box_high, box_low = bounds
                box_range = box_high - box_low
                box_range_pct = box_range / box_low * 100 if box_low > 0 else 0
                price_pos = (price - box_low) / box_range * 100 if box_range > 0 else 0
                regime_detail = f" | 횡보장: 박스 하단={box_low:.2f} 상단={box_high:.2f} 폭={box_range_pct:.2f}% 가격위치={price_pos:.1f}%"
        elif regime == "neutral":
            regime_detail = " | 중립"
    else:
        regime_detail = " | 중립: 데이터 부족"

    # RSI 표시는 15분봉 기준(rsi_15m)이 있으면 그 값을, 없으면 5분봉 RSI를 사용
    rsi_display = float(rsi_15m) if rsi_15m is not None else rsi_5m
    log(f"[1분] {pos_status}{regime_str}{box_str}{regime_detail} | 가격={price:.2f} RSI={rsi_display:.0f} 잔고={bal:.2f}{unrealized}")
