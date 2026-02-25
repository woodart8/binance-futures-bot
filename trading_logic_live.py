"""실거래 전용 전략/청산/진입 로직. live_trader_agent.py에서만 사용.

- 잔고: get_balance_usdt는 total 기준. 청산은 포지션 매칭 후 contracts>0일 때만 API 호출.
- 포지션 동기화: sync_state_from_exchange로 시작 시·매 루프 거래소 실제 포지션과 state 일치.
- 거래소 익절/손절: USE_EXCHANGE_TP_SL=True면 진입 직후 익절=지정가(LIMIT) reduceOnly, 손절=STOP_MARKET. 가격은 체결가 기준. 거래소에서 체결되면 동기화 시 청산 로그·CSV 기록.
- 잔여 SL 정리: 포지션이 전혀 없는데 SL 주문만 남은 경우 동기화 시 SL 주문을 자동 취소해 수동 청산 후 남은 주문을 정리.
- 로그 기준: 진입·청산·[1분] 상태 로그의 진입가/목표가/손절가/PnL은 거래소 실제 포지션(또는 체결가) 기준.
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

def _symbol_matches_order(symbol: str, order_symbol: str) -> bool:
    """주문의 symbol이 우리 심볼과 같은지 (BTC/USDT vs BTCUSDT vs BTC/USDT:USDT)."""
    if not order_symbol:
        return False
    a = symbol.replace("/", "").replace(":USDT", "").upper()
    b = (order_symbol or "").replace("/", "").replace(":USDT", "").upper()
    return a == b


def _fetch_open_algo_orders(exchange, symbol: str) -> list:
    """바이낸스 조건부(알고) 미체결 주문 조회. GET /fapi/v1/openAlgoOrders. ccxt에 없어 request로 fapi private 호출."""
    sym = symbol.replace("/", "").replace(":USDT", "")  # BTCUSDT
    params = {"symbol": sym}
    if not hasattr(exchange, "request"):
        return []
    for path in ("openAlgoOrders", "v1/openAlgoOrders"):
        try:
            data = exchange.request(path, "fapiPrivate", "GET", params)
            return data if isinstance(data, list) else []
        except Exception:
            continue
    return []


def _get_existing_tp_sl_order_ids(exchange, symbol: str) -> Tuple[str, str]:
    """거래소 미체결 주문에서 익절(limit reduceOnly)·손절(STOP_MARKET) 주문 ID 반환. openOrders + openAlgoOrders 확인. (익절_id, 손절_id)."""
    tp_id, sl_id = "", ""
    sym_norm = symbol.replace("/", "").replace(":USDT", "").upper()

    # 1) 일반 미체결 주문 (openOrders)
    try:
        try:
            orders = exchange.fetch_open_orders()
        except Exception:
            orders = exchange.fetch_open_orders(symbol)
        for o in orders:
            osym = o.get("symbol") or o.get("info", {}).get("symbol") or ""
            if not _symbol_matches_order(symbol, osym):
                continue
            info_type = (o.get("info", {}).get("type") or "").upper()
            t = (o.get("type") or info_type or "").upper()
            oid = o.get("id") or o.get("info", {}).get("orderId")
            if not oid:
                continue
            reduce_only = o.get("reduceOnly") or o.get("info", {}).get("reduceOnly")
            # 익절 = limit reduceOnly (TAKE_PROFIT 타입은 사용 안 함. 과거 주문 호환용으로만 검사)
            if (t == "LIMIT" or info_type == "LIMIT") and reduce_only and not tp_id:
                tp_id = str(oid)
            elif "TAKE_PROFIT" in t or "TAKE_PROFIT" in info_type:
                tp_id = str(oid)
            elif "STOP_MARKET" in t or info_type == "STOP_MARKET":
                sl_id = str(oid)
    except Exception as e:
        log(f"미체결 주문 조회 실패: {e}", "WARNING")

    # 2) 조건부(알고) 미체결 주문 (openAlgoOrders) — 손절(STOP_MARKET)이 여기 있을 수 있음. 익절은 limit이라 openOrders에 있음.
    if not tp_id or not sl_id:
        algo_list = _fetch_open_algo_orders(exchange, symbol)
        for a in algo_list:
            if not isinstance(a, dict):
                continue
            asym = (a.get("symbol") or "").upper()
            if asym != sym_norm:
                continue
            otype = (a.get("orderType") or a.get("type") or "").upper()
            oid = a.get("algoId") or a.get("orderId") or a.get("id")
            if not oid:
                continue
            if "TAKE_PROFIT" in otype and not tp_id:
                tp_id = str(oid)
            elif otype == "STOP_MARKET" and not sl_id:
                sl_id = str(oid)
    return (tp_id, sl_id)


def _place_tp_order(exchange, symbol: str, is_long: bool, amount: float, entry_price: float, tp_pct: float) -> str:
    """익절: 지정가(LIMIT) + reduceOnly. 목표가에 지정가 주문으로 체결."""
    pct_per_leverage = 1.0 / LEVERAGE
    if is_long:
        tp_price = entry_price * (1 + tp_pct / 100 * pct_per_leverage)
        close_side = "sell"
    else:
        tp_price = entry_price * (1 - tp_pct / 100 * pct_per_leverage)
        close_side = "buy"
    try:
        tp_order = exchange.create_order(
            symbol, "limit", close_side, amount, tp_price, {"reduceOnly": True}
        )
        return (tp_order.get("id") or "") if isinstance(tp_order, dict) else ""
    except Exception as e:
        log(f"익절 주문 등록 실패: {e}", "WARNING")
        return ""


def _place_sl_order(exchange, symbol: str, is_long: bool, amount: float, entry_price: float, sl_pct: float) -> str:
    """손절 주문 1개만 등록. 반환: order_id 또는 ""."""
    pct_per_leverage = 1.0 / LEVERAGE
    if is_long:
        sl_price = entry_price * (1 - sl_pct / 100 * pct_per_leverage)
        close_side = "sell"
    else:
        sl_price = entry_price * (1 + sl_pct / 100 * pct_per_leverage)
        close_side = "buy"
    try:
        sl_order = exchange.create_order(
            symbol, "STOP_MARKET", close_side, amount, None, {"stopPrice": sl_price, "reduceOnly": True}
        )
        return (sl_order.get("id") or "") if isinstance(sl_order, dict) else ""
    except Exception as e:
        log(f"손절 주문 등록 실패: {e}", "WARNING")
        return ""


def _place_exchange_tp_sl(
    exchange, symbol: str, is_long: bool, amount: float, entry_price: float, tp_pct: float, sl_pct: float
) -> Tuple[str, str]:
    """진입 후 거래소에 익절/손절 주문 등록. 반환: (tp_order_id, sl_order_id). 실패 시 (None, None)."""
    tp_id = _place_tp_order(exchange, symbol, is_long, amount, entry_price, tp_pct)
    sl_id = _place_sl_order(exchange, symbol, is_long, amount, entry_price, sl_pct)
    return (tp_id or "", sl_id or "")


def _get_entry_price_after_order(
    exchange, symbol: str, is_long: bool, order_response: dict, fallback_price: float
) -> float:
    """진입 시장가 주문 후 실제 체결가 반환. 주문 응답의 average/price 우선, 없으면 포지션 조회, 둘 다 없으면 fallback."""
    if isinstance(order_response, dict):
        avg = order_response.get("average")
        if avg is not None and float(avg) > 0:
            return float(avg)
        price = order_response.get("price")
        if price is not None and float(price) > 0:
            return float(price)
    try:
        positions = exchange.fetch_positions([symbol])
        side = "long" if is_long else "short"
        for pos in positions:
            if not _position_matches(pos, symbol, side):
                continue
            contracts = float(pos.get("contracts", 0) or 0) or abs(
                float(pos.get("positionAmt", 0) or pos.get("info", {}).get("positionAmt", 0) or 0)
            )
            if contracts <= 0:
                continue
            entry_price = float(pos.get("entryPrice", 0) or pos.get("info", {}).get("entryPrice", 0) or 0)
            if entry_price > 0:
                return entry_price
            return float(pos.get("markPrice", 0) or pos.get("info", {}).get("markPrice", 0) or 0)
    except Exception:
        pass
    return fallback_price


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


def _get_current_position_for_log(exchange) -> Tuple[float, bool]:
    """로그용: 거래소 현재 포지션의 진입가, 방향. 포지션 없거나 조회 실패 시 (0.0, False)."""
    try:
        positions = exchange.fetch_positions([SYMBOL])
        for pos in positions:
            if not _position_matches(pos, SYMBOL, "long") and not _position_matches(pos, SYMBOL, "short"):
                continue
            contracts = float(pos.get("contracts", 0) or 0) or abs(
                float(pos.get("positionAmt", 0) or pos.get("info", {}).get("positionAmt", 0) or 0)
            )
            if contracts <= 0:
                continue
            entry_price = float(pos.get("entryPrice", 0) or pos.get("info", {}).get("entryPrice", 0) or 0)
            if entry_price <= 0:
                entry_price = float(pos.get("markPrice", 0) or pos.get("info", {}).get("markPrice", 0) or 0)
            is_long = pos.get("side", "").lower() == "long"
            return (entry_price, is_long)
    except Exception:
        pass
    return (0.0, False)


def _get_last_close_trade_price(exchange, symbol: str, is_long: bool) -> float:
    """방금 청산된 포지션의 체결가(평균). 최근 체결 내역에서 청산 방향(sell=롱청산, buy=숏청산) 거래 1건 가격 반환. 없으면 0."""
    try:
        trades = exchange.fetch_my_trades(symbol, limit=30)
        if not trades:
            return 0.0
        # 최신이 마지막일 수 있음. 청산 = 롱이면 sell, 숏이면 buy
        close_side = "sell" if is_long else "buy"
        for t in reversed(trades):
            side = (t.get("side") or "").lower()
            if side != close_side:
                continue
            price = float(t.get("price") or t.get("average") or 0)
            if price > 0:
                return price
        return 0.0
    except Exception:
        return 0.0


def _log_exchange_tp_sl_close(exchange, state: Dict[str, Any], new_state: Dict[str, Any]) -> None:
    """거래소에서 익절/손절이 체결된 경우 청산 로그 및 log_trade(CSV) 기록."""
    entry_price = state.get("entry_price") or 0.0
    entry_balance = state.get("entry_balance") or 0.0
    entry_regime = state.get("entry_regime") or "trend"
    is_long = state.get("is_long", True)
    try:
        new_balance = get_balance_usdt(exchange)
    except Exception:
        new_balance = entry_balance
    pnl = new_balance - entry_balance
    exit_price = _get_last_close_trade_price(exchange, SYMBOL, is_long)
    if exit_price <= 0:
        exit_price = entry_price
    pnl_pct = (exit_price - entry_price) / entry_price * LEVERAGE * 100 if is_long else (entry_price - exit_price) / entry_price * LEVERAGE * 100
    side_str = "LONG" if is_long else "SHORT"
    regime_kr = REGIME_KR.get(entry_regime, entry_regime)
    close_reason = "거래소 익절/손절 체결"
    log(f"{side_str} 청산 | {regime_kr} | {close_reason} | 진입={entry_price:.2f} 청산={exit_price:.2f} 수익률={pnl_pct:+.2f}% 손익={pnl:+.2f} 잔고={new_balance:.2f}")
    log_trade(
        side=side_str,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=pnl,
        balance_after=new_balance,
        meta={
            "timeframe": TIMEFRAME,
            "symbol": SYMBOL,
            "mode": "live",
            "regime": entry_regime,
            "pnl_pct": round(pnl_pct, 2),
            "reason": close_reason,
        },
    )
    new_state["consecutive_loss_count"] = (state.get("consecutive_loss_count") or 0) + 1 if pnl < 0 else 0


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
    포지션을 처음 발견할 때: USE_EXCHANGE_TP_SL이면 익절(limit reduceOnly)·손절(STOP_MARKET) 미체결 조회 후 없는 것만 추가 등록.
    포지션이 사라졌을 때(익절/손절 체결): 청산 로그 및 trades_log.csv 기록.
    반환: 업데이트된 state.
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
        entry_regime = state.get("entry_regime") or "trend"
        tp_oid = state.get("tp_order_id") or ""
        sl_oid = state.get("sl_order_id") or ""
        # 포지션을 처음 발견할 때만: 거래소에 이미 있는 TP/SL은 유지하고, 없는 것만 추가 등록.
        if USE_EXCHANGE_TP_SL and not state.get("has_position"):
            existing_tp, existing_sl = _get_existing_tp_sl_order_ids(exchange, SYMBOL)
            if existing_tp:
                tp_oid = existing_tp
            if existing_sl:
                sl_oid = existing_sl
            is_trend = entry_regime == "trend"
            tp_pct = TREND_PROFIT_TARGET if is_trend else SIDEWAYS_PROFIT_TARGET
            sl_pct = TREND_STOP_LOSS_PRICE if is_trend else SIDEWAYS_STOP_LOSS_PRICE
            added = []
            if not tp_oid:
                tp_oid = _place_tp_order(exchange, SYMBOL, is_long, contracts, entry_price, tp_pct)
                if tp_oid:
                    added.append("익절")
            if not sl_oid:
                sl_oid = _place_sl_order(exchange, SYMBOL, is_long, contracts, entry_price, sl_pct)
                if sl_oid:
                    added.append("손절")
            if added:
                log(f"[동기화] 기존 포지션 TP/SL 추가 등록 ({', '.join(added)}) | 진입가={entry_price:.2f} 수량={contracts}")
        state = {
            **state,
            "has_position": True,
            "is_long": is_long,
            "entry_price": entry_price,
            "entry_balance": state.get("entry_balance") or balance,
            "entry_regime": entry_regime,
            "box_high_entry": state.get("box_high_entry") or 0.0,
            "box_low_entry": state.get("box_low_entry") or 0.0,
            "highest_price": highest,
            "lowest_price": lowest,
            "best_pnl_pct": state.get("best_pnl_pct", 0.0),
            "tp_order_id": tp_oid,
            "sl_order_id": sl_oid,
        }
        return state

    # 포지션 없음 → state도 포지션 없음으로. 거래소에서 익절/손절 체결됐으면 청산 로그 + CSV 기록.
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
        "tp_order_id": None,
        "sl_order_id": None,
    }
    # 이전에는 포지션이 있었는데 지금은 없고, TP/SL 주문 ID가 남아 있다면 익절/손절 체결로 간주하고 로그 기록
    if state.get("has_position") and (state.get("tp_order_id") or state.get("sl_order_id")):
        _log_exchange_tp_sl_close(exchange, state, new_state)
    elif state.get("has_position"):
        log("[동기화] 거래소에 포지션 없음 — state를 포지션 없음으로 맞춤", "WARNING")

    # 현재 포지션이 전혀 없는 상태라면, 거래소에 남아 있는 SL 주문이 있더라도 모두 취소
    # (수동 청산 등으로 포지션은 없는데 SL 주문만 남아 있는 상황 방지)
    if USE_EXCHANGE_TP_SL:
        try:
            _, existing_sl = _get_existing_tp_sl_order_ids(exchange, SYMBOL)
            if existing_sl:
                try:
                    exchange.cancel_order(existing_sl, SYMBOL)
                    log(f"[동기화] 포지션 없음 상태에서 잔여 SL 주문 취소 (order_id={existing_sl})")
                except Exception as e:
                    log(f"[동기화] 잔여 SL 주문 취소 실패: {e}", "WARNING")
        except Exception:
            # 조회 실패 시에는 그냥 넘어감 (기존 동작 유지)
            pass

    return new_state


def check_tp_sl_and_close(exchange, state: Dict[str, Any], current_price: float, df: pd.DataFrame) -> Tuple[Dict[str, Any], bool]:
    """
    새 5분봉이 뜬 뒤, 전봉 종가 기준 익절/손절 조건 만족 시 청산.
    포지션은 _position_matches로 심볼/사이드 매칭(바이낸스 BTC/USDT:USDT 등), 수량은 contracts 또는 positionAmt.
    contracts<=0이면 주문 없이 return(청산생략 로그만). 청산 로그의 진입가·수익률은 fetch한 포지션 entryPrice 기준.
    반환: (업데이트된 state, did_close).
    """
    if not state.get("has_position"):
        return (state, False)

    latest = df.iloc[-1]
    rsi = float(latest["rsi"])
    is_long = state["is_long"]
    entry_price = state["entry_price"]
    entry_balance = state["entry_balance"]
    entry_regime = state["entry_regime"]
    entry_trend_direction = state.get("entry_trend_direction", "")
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
            trend_direction=entry_trend_direction,
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
            trend_direction=entry_trend_direction,
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
    entry_price_actual = entry_price  # 로그는 실제 포지션 기준
    side_to_close = "long" if is_long else "short"
    for pos in positions:
        if _position_matches(pos, SYMBOL, side_to_close):
            contracts = float(pos.get("contracts", 0) or 0)
            if contracts <= 0:
                contracts = abs(float(pos.get("positionAmt", 0) or pos.get("info", {}).get("positionAmt", 0) or 0))
            ep = float(pos.get("entryPrice", 0) or pos.get("info", {}).get("entryPrice", 0) or 0)
            if ep > 0:
                entry_price_actual = ep
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
    pnl_pct_final = (current_price - entry_price_actual) / entry_price_actual * LEVERAGE * 100 if is_long else (entry_price_actual - current_price) / entry_price_actual * LEVERAGE * 100
    side_str = "LONG" if is_long else "SHORT"
    regime_kr = REGIME_KR.get(entry_regime, entry_regime)
    log(f"{side_str} 청산 | {regime_kr} | {close_reason} | 진입={entry_price_actual:.2f} 청산={current_price:.2f} 수익률={pnl_pct_final:+.2f}% 손익={pnl:+.2f} 잔고={new_balance:.2f}")
    log_trade(
        side="LONG" if is_long else "SHORT",
        entry_price=entry_price_actual,
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
        rsi_history = tail["rsi"].tolist()
    else:
        price_history = df["close"].tolist()
        rsi_history = df["rsi"].tolist()
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
        rsi_history=rsi_history,
    )
    if signal not in ("long", "short"):
        return (state, False)

    # 추세 방향(상승/하락) 계산: 추세장일 때만 사용
    entry_trend_direction = ""
    if regime == "trend" and ma_long_history and len(ma_long_history) >= TREND_SLOPE_BARS:
        recent_ma20 = ma_long_history[-TREND_SLOPE_BARS:]
        ma20_start = recent_ma20[0]
        ma20_end = recent_ma20[-1]
        if ma20_start and ma20_start > 0:
            slope_pct = (ma20_end - ma20_start) / ma20_start * 100.0
            if slope_pct > TREND_SLOPE_MIN_PCT:
                entry_trend_direction = "up"
            elif slope_pct < -TREND_SLOPE_MIN_PCT:
                entry_trend_direction = "down"

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
            order = exchange.create_market_buy_order(SYMBOL, amount)
            entry_price_actual = _get_entry_price_after_order(exchange, SYMBOL, True, order, current_price)
            regime_kr = REGIME_KR.get(regime, regime)
            is_trend = regime == "trend"
            if is_trend and entry_trend_direction in ("up", "down"):
                is_counter_trend = entry_trend_direction == "down"  # 하락장 롱 = 역추세
                if is_counter_trend:
                    tp_pct = COUNTER_TREND_PROFIT_TARGET
                    sl_pct = COUNTER_TREND_STOP_LOSS_PRICE
                else:
                    tp_pct = TREND_PROFIT_TARGET
                    sl_pct = TREND_STOP_LOSS_PRICE
            else:
                tp_pct = TREND_PROFIT_TARGET if is_trend else SIDEWAYS_PROFIT_TARGET
                sl_pct = TREND_STOP_LOSS_PRICE if is_trend else SIDEWAYS_STOP_LOSS_PRICE
            pct_per_leverage = 1.0 / LEVERAGE
            tg = entry_price_actual * (1 + tp_pct / 100 * pct_per_leverage)
            st = entry_price_actual * (1 - sl_pct / 100 * pct_per_leverage)
            tp_oid, sl_oid = "", ""
            if USE_EXCHANGE_TP_SL:
                tp_oid, sl_oid = _place_exchange_tp_sl(exchange, SYMBOL, True, amount, entry_price_actual, tp_pct, sl_pct)
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
            log(f"LONG 진입 | {regime_kr} | 가격={entry_price_actual:.2f} 잔고={balance:.2f} 투입={order_usdt:.2f} USDT{entry_tp_sl}{box_str}")
            new_state = {
                **state,
                "has_position": True,
                "is_long": True,
                "entry_price": entry_price_actual,
                "highest_price": entry_price_actual,
                "entry_regime": regime,
                "entry_trend_direction": entry_trend_direction,
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
            order = exchange.create_market_sell_order(SYMBOL, amount)
            entry_price_actual = _get_entry_price_after_order(exchange, SYMBOL, False, order, current_price)
            regime_kr = REGIME_KR.get(regime, regime)
            is_trend = regime == "trend"
            if is_trend and entry_trend_direction in ("up", "down"):
                is_counter_trend = entry_trend_direction == "up"  # 상승장 숏 = 역추세
                if is_counter_trend:
                    tp_pct = COUNTER_TREND_PROFIT_TARGET
                    sl_pct = COUNTER_TREND_STOP_LOSS_PRICE
                else:
                    tp_pct = TREND_PROFIT_TARGET
                    sl_pct = TREND_STOP_LOSS_PRICE
            else:
                tp_pct = TREND_PROFIT_TARGET if is_trend else SIDEWAYS_PROFIT_TARGET
                sl_pct = TREND_STOP_LOSS_PRICE if is_trend else SIDEWAYS_STOP_LOSS_PRICE
            pct_per_leverage = 1.0 / LEVERAGE
            tg = entry_price_actual * (1 - tp_pct / 100 * pct_per_leverage)
            st = entry_price_actual * (1 + sl_pct / 100 * pct_per_leverage)
            tp_oid, sl_oid = "", ""
            if USE_EXCHANGE_TP_SL:
                tp_oid, sl_oid = _place_exchange_tp_sl(exchange, SYMBOL, False, amount, entry_price_actual, tp_pct, sl_pct)
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
            log(f"SHORT 진입 | {regime_kr} | 가격={entry_price_actual:.2f} 잔고={balance:.2f} 투입={order_usdt:.2f} USDT{entry_tp_sl}{box_str}")
            new_state = {
                **state,
                "has_position": True,
                "is_long": False,
                "entry_price": entry_price_actual,
                "lowest_price": entry_price_actual,
                "entry_regime": regime,
                "entry_trend_direction": entry_trend_direction,
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
    청산 로그의 진입가·수익률은 fetch한 포지션 entryPrice 기준.
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
        entry_price_actual = entry_price  # 로그/CSV는 실제 포지션 기준
        side_to_close = "long" if is_long else "short"
        for pos in positions:
            if _position_matches(pos, SYMBOL, side_to_close):
                contracts = float(pos.get("contracts", 0) or 0)
                if contracts <= 0:
                    contracts = abs(float(pos.get("positionAmt", 0) or pos.get("info", {}).get("positionAmt", 0) or 0))
                ep = float(pos.get("entryPrice", 0) or pos.get("info", {}).get("entryPrice", 0) or 0)
                if ep > 0:
                    entry_price_actual = ep
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
        pnl_pct = (price - entry_price_actual) / entry_price_actual * LEVERAGE * 100 if is_long else (entry_price_actual - price) / entry_price_actual * LEVERAGE * 100
        side_str = "LONG" if is_long else "SHORT"
        regime_kr = REGIME_KR.get(entry_regime, entry_regime)
        log(f"{side_str} 청산 | {regime_kr} | {close_reason} | 진입={entry_price_actual:.2f} 청산={price:.2f} 수익률={pnl_pct:+.2f}% 손익={pnl:+.2f} 잔고={new_balance:.2f}")
        log_trade(
            side="LONG" if is_long else "SHORT",
            entry_price=entry_price_actual,
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
    """1분봉 생길 때마다 상태 로그. 포지션/진입가/목표가/손절가/미실현 PnL은 거래소 현재 포지션 기준."""
    latest = df.iloc[-1]
    price = float(latest["close"])
    rsi_5m = float(latest["rsi"])
    has_position = state["has_position"]
    entry_regime = state["entry_regime"]
    box_high_entry = state["box_high_entry"]
    box_low_entry = state["box_low_entry"]

    # 포지션 있으면 거래소에서 현재 진입가/방향 조회(로그는 항상 현재 포지션 기준)
    entry_price = state["entry_price"]
    is_long = state["is_long"]
    if has_position:
        ex_entry, ex_long = _get_current_position_for_log(exchange)
        if ex_entry > 0:
            entry_price, is_long = ex_entry, ex_long

    try:
        bal = get_balance_usdt(exchange)
    except Exception:
        bal = 0.0
    pos_status = "LONG" if has_position and is_long else ("SHORT" if has_position else "NONE")
    unrealized = ""
    pos_detail = ""  # 진입가·목표가·손절가
    if has_position and entry_price > 0:
        pnl_pct = (price - entry_price) / entry_price * LEVERAGE * 100 if is_long else (entry_price - price) / entry_price * LEVERAGE * 100
        unrealized = f" 미실현={pnl_pct:+.2f}%"
        is_trend = entry_regime == "trend"
        tp_pct = TREND_PROFIT_TARGET if is_trend else SIDEWAYS_PROFIT_TARGET
        sl_pct = TREND_STOP_LOSS_PRICE if is_trend else SIDEWAYS_STOP_LOSS_PRICE
        pct_per_leverage = 1.0 / LEVERAGE
        if is_long:
            tg = entry_price * (1 + tp_pct / 100 * pct_per_leverage)
            st = entry_price * (1 - sl_pct / 100 * pct_per_leverage)
        else:
            tg = entry_price * (1 - tp_pct / 100 * pct_per_leverage)
            st = entry_price * (1 + sl_pct / 100 * pct_per_leverage)
        pos_detail = f" 진입가={entry_price:.2f} 목표가={tg:.2f} 손절가={st:.2f}"
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
    log(f"[1분] {pos_status}{regime_str}{box_str}{pos_detail}{regime_detail} | 가격={price:.2f} RSI={rsi_display:.0f} 잔고={bal:.2f}{unrealized}")
