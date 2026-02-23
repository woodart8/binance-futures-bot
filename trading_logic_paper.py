"""페이퍼 트레이딩 전용 전략/청산/진입 로직. paper_trading.py에서만 사용."""

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

import config as _config
from config import (
    CONSECUTIVE_LOSS_LIMIT,
    DAILY_LOSS_LIMIT_PCT,
    FEE_RATE,
    INITIAL_BALANCE,
    LEVERAGE,
    POSITION_SIZE_PERCENT,
    SLIPPAGE_PCT,
    SIDEWAYS_BOX_PERIOD,
    SIDEWAYS_PROFIT_TARGET,
    SIDEWAYS_STOP_LOSS_PRICE,
    TREND_PROFIT_TARGET,
    TREND_STOP_LOSS_PRICE,
    REGIME_LOOKBACK_15M,
    SYMBOL,
    TIMEFRAME,
)
from data import compute_regime_15m
from strategy_core import REGIME_KR, swing_strategy_signal, get_hold_reason, get_entry_reason, get_sideways_box_bounds
from exit_logic import check_long_exit, check_short_exit, reason_to_display_message
from trade_logger import log_trade, log_funding
from logger import log
from funding import (
    fetch_current_funding_rate,
    get_funding_utc_key,
    compute_funding_pnl,
    FUNDING_HOURS_UTC,
)

RISK_PER_TRADE = POSITION_SIZE_PERCENT
REASON_LOG_INTERVAL = 300
_daily_limit_logged_date: str = ""
_last_reason_log_time: float = 0.0


@dataclass
class PaperState:
    balance: float
    equity: float
    has_long_position: bool
    has_short_position: bool
    entry_price: float
    position_size: float
    peak_equity: float
    max_drawdown: float
    highest_price: float
    lowest_price: float
    trailing_stop_active: bool
    best_pnl_pct: float
    entry_regime: str
    position_entry_time: float
    box_high: float
    box_low: float
    daily_start_balance: float
    daily_start_date: str
    consecutive_loss_count: int
    last_funding_utc_key: str  # "YYYY-MM-DD-HH" 적용한 마지막 펀딩 시각


def _paper_balance_path() -> str:
    fname = getattr(_config, "PAPER_BALANCE_FILE", "paper_balance.json")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)


def load_paper_balance() -> tuple:
    """저장된 모의투자 잔고를 읽는다. (balance, last_funding_utc_key). 없거나 유효하지 않으면 (INITIAL_BALANCE, "")."""
    path = _paper_balance_path()
    if not os.path.isfile(path):
        return (INITIAL_BALANCE, "")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        balance = float(data.get("balance", 0))
        if balance <= 0:
            return (INITIAL_BALANCE, "")
        last_funding = str(data.get("last_funding_utc_key", ""))
        return (balance, last_funding)
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return (INITIAL_BALANCE, "")


def save_paper_balance(balance: float, last_funding_utc_key: str = "") -> None:
    """모의투자 잔고 및 마지막 펀딩 적용 구간을 파일에 저장."""
    path = _paper_balance_path()
    try:
        from datetime import datetime
        data = {
            "balance": round(balance, 2),
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "last_funding_utc_key": last_funding_utc_key,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError as e:
        log(f"잔고 저장 실패: {path} - {e}", "ERROR")


def apply_funding_if_needed(
    state: PaperState,
    exchange,
    current_utc_datetime: "datetime",
    current_price: float,
) -> bool:
    """
    00/08/16 UTC 펀딩 정산 시각이면 펀딩비를 적용하고 잔고를 갱신.
    포지션 보유 시에만 적용하며, 같은 정산 구간에는 한 번만 적용.
    :return: 펀딩을 적용했으면 True.
    """
    from datetime import datetime, timezone

    if not (state.has_long_position or state.has_short_position):
        return False
    key = get_funding_utc_key(current_utc_datetime)
    if key is None or (state.last_funding_utc_key and key <= state.last_funding_utc_key):
        return False
    try:
        rate = fetch_current_funding_rate(exchange, SYMBOL)
    except Exception as e:
        log(f"펀딩 레이트 조회 실패: {e}", "ERROR")
        return False
    position_value = state.position_size * current_price
    is_long = state.has_long_position
    side = "LONG" if is_long else "SHORT"
    funding_pnl = compute_funding_pnl(position_value, rate, is_long)
    state.balance += funding_pnl
    state.last_funding_utc_key = key
    log_funding(
        funding_utc_key=key,
        side=side,
        position_value=position_value,
        funding_rate=rate,
        funding_pnl=funding_pnl,
        balance_after=state.balance,
        meta={"symbol": SYMBOL, "mode": "paper"},
    )
    rate_pct = rate * 100
    log(f"펀딩비 적용 | {key} UTC | {side} | 레이트={rate_pct:.4f}% 명목={position_value:.2f} USDT | 펀딩손익={funding_pnl:+.2f} 잔고={state.balance:.2f}")
    save_paper_balance(state.balance, state.last_funding_utc_key)
    return True


def _log_daily_limit_once(current_date: str, daily_loss_pct: float, regime_kr: str = "") -> None:
    global _daily_limit_logged_date
    if _daily_limit_logged_date != current_date:
        _daily_limit_logged_date = current_date
        kr = f"{regime_kr} | " if regime_kr else ""
        log(f"[진입안함] {kr}일일손실한도 {daily_loss_pct:.1f}% 도달")


def _should_log_reason() -> bool:
    global _last_reason_log_time
    if time.time() - _last_reason_log_time >= REASON_LOG_INTERVAL:
        _last_reason_log_time = time.time()
        return True
    return False


def init_state() -> PaperState:
    balance, last_funding_utc_key = load_paper_balance()
    return PaperState(
        balance=balance,
        equity=balance,
        has_long_position=False,
        has_short_position=False,
        entry_price=0.0,
        position_size=0.0,
        peak_equity=balance,
        max_drawdown=0.0,
        highest_price=0.0,
        lowest_price=float("inf"),
        trailing_stop_active=False,
        best_pnl_pct=0.0,
        entry_regime="",
        position_entry_time=0.0,
        box_high=0.0,
        box_low=0.0,
        daily_start_balance=balance,
        daily_start_date="",
        consecutive_loss_count=0,
        last_funding_utc_key=last_funding_utc_key,
    )


def close_position(state: PaperState, candle: pd.Series, side: str, reason: str, raw_reason: Optional[str] = None) -> None:
    price = float(candle["close"])
    entry_price = state.entry_price
    entry_regime = state.entry_regime

    if side == "LONG":
        pnl_pct = (price - entry_price) / entry_price * LEVERAGE
    else:
        pnl_pct = (entry_price - price) / entry_price * LEVERAGE

    slippage_cost = state.balance * RISK_PER_TRADE * SLIPPAGE_PCT / 100
    gross_pnl = pnl_pct * state.balance * RISK_PER_TRADE
    fee = state.balance * RISK_PER_TRADE * FEE_RATE
    net_pnl = gross_pnl - fee - slippage_cost
    state.balance += net_pnl

    if net_pnl < 0:
        state.consecutive_loss_count += 1
    else:
        state.consecutive_loss_count = 0

    state.has_long_position = False
    state.has_short_position = False
    state.position_size = 0.0
    state.entry_price = 0.0
    state.highest_price = 0.0
    state.lowest_price = float("inf")
    state.trailing_stop_active = False
    state.best_pnl_pct = 0.0
    state.entry_regime = ""
    state.position_entry_time = 0.0
    state.box_high = 0.0
    state.box_low = 0.0

    log_trade(
        side=f"PAPER_{side}",
        entry_price=entry_price,
        exit_price=price,
        pnl=net_pnl,
        balance_after=state.balance,
        meta={
            "timeframe": TIMEFRAME,
            "symbol": SYMBOL,
            "mode": "paper",
            "regime": entry_regime,
            "reason": reason,
            "pnl_pct": round(pnl_pct * 100, 2),
        },
    )
    kr = REGIME_KR.get(entry_regime, entry_regime or "?")
    log(f"{side} 청산 | {kr} | {reason} | 진입={entry_price:.2f} 청산={price:.2f} 수익률={pnl_pct*100:+.2f}% 손익={net_pnl:+.2f} 잔고={state.balance:.2f}")
    save_paper_balance(state.balance, state.last_funding_utc_key)


def open_position(
    state: PaperState,
    price: float,
    side: str,
    reason: str,
    regime: str = "",
    price_history: Optional[list] = None,
    price_history_15m: Optional[list] = None,
) -> None:
    trade_capital = state.balance * RISK_PER_TRADE  # 마진(투입 담보)
    if trade_capital <= 0:
        return

    fee = trade_capital * FEE_RATE
    slippage_cost = trade_capital * SLIPPAGE_PCT / 100
    trade_capital_after_fee = trade_capital - fee - slippage_cost
    # 포지션 규모(노션널) = 마진 × 레버리지 → 수량(BTC) = 노션널 / 가격 (라이브와 동일)
    position_size = (trade_capital_after_fee * LEVERAGE) / price

    box_high_entry = 0.0
    box_low_entry = 0.0
    if regime == "sideways" and price_history_15m and len(price_history_15m) >= REGIME_LOOKBACK_15M:
        bounds = get_sideways_box_bounds(price_history_15m, REGIME_LOOKBACK_15M)
        if bounds:
            box_high_entry, box_low_entry = bounds

    if side == "LONG":
        state.has_long_position = True
    else:
        state.has_short_position = True

    state.entry_price = price
    state.position_size = position_size
    state.position_entry_time = time.time()
    state.trailing_stop_active = False
    state.best_pnl_pct = 0.0
    state.entry_regime = regime
    state.box_high = box_high_entry
    state.box_low = box_low_entry

    if side == "LONG":
        state.highest_price = price
    else:
        state.lowest_price = price

    kr = REGIME_KR.get(regime, regime or "?")
    box_str = ""
    if regime == "sideways" and state.box_high > 0 and state.box_low > 0:
        box_str = f" | 박스 하단={state.box_low:.2f} 상단={state.box_high:.2f}"
    is_trend = regime == "trend"
    tp_pct = TREND_PROFIT_TARGET if is_trend else SIDEWAYS_PROFIT_TARGET
    sl_pct = TREND_STOP_LOSS_PRICE if is_trend else SIDEWAYS_STOP_LOSS_PRICE
    pct_per_leverage = 1.0 / LEVERAGE
    if side == "LONG":
        target_price = price * (1 + tp_pct / 100 * pct_per_leverage)
        stop_price = price * (1 - sl_pct / 100 * pct_per_leverage)
    else:
        target_price = price * (1 - tp_pct / 100 * pct_per_leverage)
        stop_price = price * (1 + sl_pct / 100 * pct_per_leverage)
    log(f"{side} 진입 | {kr} | {reason} | 가격={price:.2f} 잔고={state.balance:.2f} 투입={trade_capital:.2f} | 목표가={target_price:.2f} 손절가={stop_price:.2f}{box_str}")


def _update_equity_and_drawdown(state: PaperState, price: float) -> None:
    if state.has_long_position:
        unrealized = (price - state.entry_price) / state.entry_price * LEVERAGE * state.balance * RISK_PER_TRADE
        equity = state.balance + unrealized
    elif state.has_short_position:
        unrealized = (state.entry_price - price) / state.entry_price * LEVERAGE * state.balance * RISK_PER_TRADE
        equity = state.balance + unrealized
    else:
        equity = state.balance
    state.equity = equity
    if equity > state.peak_equity:
        state.peak_equity = equity
    drawdown = (state.peak_equity - equity) / state.peak_equity if state.peak_equity > 0 else 0.0
    state.max_drawdown = max(state.max_drawdown, drawdown)


def _reason_no_exit_strategy(regime: str, is_long: bool, rsi: float, price: float, short_ma: float) -> str:
    return "청산 조건 미충족"


def check_scalp_stop_loss_and_profit(state: PaperState, current_price: float, candle: pd.Series, df: Optional[pd.DataFrame] = None) -> bool:
    if state.has_long_position:
        if current_price > state.highest_price:
            state.highest_price = current_price
        pnl_pct = (current_price - state.entry_price) / state.entry_price * LEVERAGE * 100
        if pnl_pct > state.best_pnl_pct:
            state.best_pnl_pct = pnl_pct

        regime = state.entry_regime or ""
        rsi = float(candle.get("rsi", 50.0))
        reason = check_long_exit(
            regime=regime,
            pnl_pct=pnl_pct,
            price=current_price,
            entry_price=state.entry_price,
            best_pnl_pct=state.best_pnl_pct,
            box_high=state.box_high or 0,
            box_low=state.box_low or 0,
        )
        if reason:
            msg = reason_to_display_message(reason, is_long=True)
            close_position(state, candle, "LONG", f"{msg} ({pnl_pct:.2f}%)", raw_reason=reason)
            return True

    elif state.has_short_position:
        if current_price < state.lowest_price:
            state.lowest_price = current_price
        pnl_pct = (state.entry_price - current_price) / state.entry_price * LEVERAGE * 100
        if pnl_pct > state.best_pnl_pct:
            state.best_pnl_pct = pnl_pct

        regime = state.entry_regime or ""
        rsi = float(candle.get("rsi", 50.0))
        reason = check_short_exit(
            regime=regime,
            pnl_pct=pnl_pct,
            price=current_price,
            entry_price=state.entry_price,
            best_pnl_pct=state.best_pnl_pct,
            box_high=state.box_high or 0,
            box_low=state.box_low or 0,
        )
        if reason:
            msg = reason_to_display_message(reason, is_long=False)
            close_position(state, candle, "SHORT", f"{msg} ({pnl_pct:.2f}%)", raw_reason=reason)
            return True

    # 청산대기 로그는 5분봉 단위 apply_strategy_on_candle에서만 출력 (중복 방지)
    return False


def try_paper_entry(state: PaperState, df: pd.DataFrame, current_price: float) -> bool:
    """
    새 5분봉이 뜬 뒤, 전봉 종가 기준 진입 조건 만족 시 모의 진입(실제 주문 없음).
    실거래와 동일 로직, 주문만 시뮬레이션.
    반환: 진입했으면 True.
    """
    if state.has_long_position or state.has_short_position:
        return False

    latest = df.iloc[-1]
    rsi = float(latest["rsi"])
    short_ma = float(latest["ma_short"])
    long_ma = float(latest["ma_long"])
    rsi_prev = float(df["rsi"].iloc[-2]) if len(df) >= 2 else None
    open_prev = float(df["open"].iloc[-2]) if len(df) >= 2 else None
    close_prev = float(df["close"].iloc[-2]) if len(df) >= 2 else None
    open_curr = float(latest["open"])

    if len(df) >= REGIME_LOOKBACK_15M * 3:
        regime, short_ma_15m, long_ma_15m, ma_50_15m, ma_100_15m, price_history_15m, _, ma_long_history = compute_regime_15m(df, current_price)
    else:
        regime, short_ma_15m, long_ma_15m, ma_50_15m, ma_100_15m, price_history_15m, rsi_15m, ma_long_history = "neutral", 0.0, 0.0, 0.0, 0.0, [], 0.0, []
    if len(df) >= SIDEWAYS_BOX_PERIOD:
        tail = df.tail(SIDEWAYS_BOX_PERIOD + 1)
        price_history = list(zip(tail["high"].tolist(), tail["low"].tolist(), tail["close"].tolist()))
    else:
        price_history = None
    use_15m = len(price_history_15m) >= REGIME_LOOKBACK_15M
    regime_price_hist = price_history_15m if (use_15m and regime == "sideways") else None

    signal = swing_strategy_signal(
        rsi_value=rsi,
        price=current_price,
        rsi_prev=rsi_prev,
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
        return False

    ts = latest.get("timestamp")
    current_date = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
    if state.daily_start_date != current_date:
        state.daily_start_balance = state.balance
        state.daily_start_date = current_date
        state.consecutive_loss_count = 0
    daily_loss_pct = 0.0
    if state.daily_start_balance > 0:
        daily_loss_pct = (state.daily_start_balance - state.balance) / state.daily_start_balance * 100
    if daily_loss_pct >= DAILY_LOSS_LIMIT_PCT or state.consecutive_loss_count >= CONSECUTIVE_LOSS_LIMIT:
        if daily_loss_pct >= DAILY_LOSS_LIMIT_PCT:
            _log_daily_limit_once(current_date, daily_loss_pct, REGIME_KR.get(regime, regime))
        return False

    if state.balance * RISK_PER_TRADE < 5.0:
        return False

    if signal == "long":
        reason = get_entry_reason(
            regime, "LONG", rsi, current_price, short_ma, long_ma,
            regime_short_ma=short_ma_15m if use_15m else None,
            regime_long_ma=long_ma_15m if use_15m else None,
            regime_ma_50=ma_50_15m if use_15m else None,
            regime_ma_100=ma_100_15m if use_15m else None,
            regime_price_history=regime_price_hist,
            regime_ma_long_history=ma_long_history if use_15m else None,
            price_history=price_history,
        )
        open_position(
            state, current_price, "LONG", reason.strip(),
            regime, price_history, price_history_15m=price_history_15m if regime == "sideways" else None,
        )
        return True
    else:
        reason = get_entry_reason(
            regime, "SHORT", rsi, current_price, short_ma, long_ma,
            regime_short_ma=short_ma_15m if use_15m else None,
            regime_long_ma=long_ma_15m if use_15m else None,
            regime_ma_50=ma_50_15m if use_15m else None,
            regime_ma_100=ma_100_15m if use_15m else None,
            regime_price_history=regime_price_hist,
            regime_ma_long_history=ma_long_history if use_15m else None,
            price_history=price_history,
        )
        open_position(
            state, current_price, "SHORT", reason.strip(),
            regime, price_history, price_history_15m=price_history_15m if regime == "sideways" else None,
        )
        return True


def apply_strategy_on_candle(
    state: PaperState, candle: pd.Series, df: Optional[pd.DataFrame] = None, regime: Optional[str] = None
) -> bool:
    """새 5분봉 시 상태/청산만 처리. 진입은 try_paper_entry(새 봉 시)에서만. 반환: 이번에 청산했으면 True."""
    price = float(candle["close"])
    rsi = float(candle["rsi"])
    short_ma = float(candle["ma_short"])
    long_ma = float(candle["ma_long"])

    short_ma_15m, long_ma_15m, ma_50_15m, ma_100_15m, price_history_15m = 0.0, 0.0, 0.0, 0.0, []
    rsi_15m = None
    if regime is None and df is not None and len(df) >= REGIME_LOOKBACK_15M * 3:
        regime, short_ma_15m, long_ma_15m, ma_50_15m, ma_100_15m, price_history_15m, rsi_15m, ma_long_history = compute_regime_15m(df, price)
    if regime is None:
        regime = "neutral"

    has_position = state.has_long_position or state.has_short_position
    is_long = state.has_long_position

    if df is not None and len(df) >= SIDEWAYS_BOX_PERIOD:
        tail = df.tail(SIDEWAYS_BOX_PERIOD + 1)
        price_history = list(zip(tail["high"].tolist(), tail["low"].tolist(), tail["close"].tolist()))
    else:
        price_history = None

    use_15m = len(price_history_15m) >= REGIME_LOOKBACK_15M
    rsi_prev = float(df["rsi"].iloc[-2]) if df is not None and len(df) >= 2 else None
    open_prev = float(df["open"].iloc[-2]) if df is not None and len(df) >= 2 else None
    close_prev = float(df["close"].iloc[-2]) if df is not None and len(df) >= 2 else None
    open_curr = float(candle["open"])
    rsi_use = rsi
    regime_price_hist = price_history_15m if (use_15m and regime == "sideways") else None
    signal = swing_strategy_signal(
        rsi_value=rsi_use,
        price=price,
        rsi_prev=rsi_prev,
        open_prev=open_prev,
        close_prev=close_prev,
        open_curr=open_curr,
        short_ma=short_ma,
        long_ma=long_ma,
        has_position=has_position,
        is_long=is_long,
        regime=regime,
        price_history=price_history,
        regime_short_ma=short_ma_15m if use_15m else None,
        regime_long_ma=long_ma_15m if use_15m else None,
        regime_ma_50=ma_50_15m if use_15m else None,
        regime_ma_100=ma_100_15m if use_15m else None,
        regime_price_history=regime_price_hist,
        regime_ma_long_history=ma_long_history if use_15m else None,
    )

    regime_kr = REGIME_KR.get(regime, regime)

    ts = candle.get("timestamp")
    current_date = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
    if state.daily_start_date != current_date:
        state.daily_start_balance = state.balance
        state.daily_start_date = current_date
        state.consecutive_loss_count = 0

    daily_loss_pct = 0.0
    if state.daily_start_balance > 0:
        daily_loss_pct = (state.daily_start_balance - state.balance) / state.daily_start_balance * 100
    daily_limit_hit = daily_loss_pct >= DAILY_LOSS_LIMIT_PCT
    consecutive_limit_hit = state.consecutive_loss_count >= CONSECUTIVE_LOSS_LIMIT

    if signal == "flat" and has_position:
        side = "LONG" if is_long else "SHORT"
        close_position(state, candle, side, f"단타청산({regime_kr})")
        _update_equity_and_drawdown(state, price)
        return True
    else:
        if not has_position:
            if daily_limit_hit or consecutive_limit_hit:
                if daily_limit_hit:
                    _log_daily_limit_once(current_date, daily_loss_pct, regime_kr)
                elif consecutive_limit_hit:
                    log(f"[진입안함] {regime_kr} | 연속손실 {state.consecutive_loss_count}회 당일 중단")
            else:
                reason = get_hold_reason(
                    regime, rsi_use, price, short_ma, long_ma,
                    regime_short_ma=short_ma_15m if use_15m else None,
                    regime_long_ma=long_ma_15m if use_15m else None,
                    regime_ma_50=ma_50_15m if use_15m else None,
                    regime_ma_100=ma_100_15m if use_15m else None,
                    regime_price_history=regime_price_hist,
                    regime_ma_long_history=ma_long_history if use_15m else None,
                    price_history=price_history,
                )
                log(f"[진입안함] {regime_kr} | {reason}")
        elif has_position and signal == "hold":
            log(f"[청산대기] {regime_kr} | {_reason_no_exit_strategy(regime, is_long, rsi, price, short_ma)}")

    _update_equity_and_drawdown(state, price)
    return False
