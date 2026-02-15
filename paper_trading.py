"""페이퍼 트레이딩. python paper_trading.py"""

import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import (
    CONSECUTIVE_LOSS_LIMIT,
    DAILY_LOSS_LIMIT_PCT,
    FEE_RATE,
    INITIAL_BALANCE,
    LEVERAGE,
    POSITION_SIZE_PERCENT,
    RSI_PERIOD,
    SLIPPAGE_PCT,
    SIDEWAYS_BOX_PERIOD,
    SIDEWAYS_PROFIT_TARGET,
    SIDEWAYS_STOP_LOSS_PRICE,
    TREND_PROFIT_TARGET,
    TREND_STOP_LOSS_PRICE,
    MA_SHORT_PERIOD,
    MA_LONG_PERIOD,
    MA_MID_PERIOD,
    MA_LONGEST_PERIOD,
    REGIME_LOOKBACK_15M,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    SYMBOL,
    TIMEFRAME,
)
from exchange_client import get_public_exchange
from data import fetch_ohlcv, compute_regime_15m
from indicators import calculate_rsi, calculate_ma, calculate_macd
from chart_patterns import detect_chart_pattern, PATTERN_LOOKBACK
from strategy_core import REGIME_KR, swing_strategy_signal, get_hold_reason, get_entry_reason
from exit_logic import check_long_exit, check_short_exit, reason_to_display_message
from trade_logger import log_trade
from logger import log

RISK_PER_TRADE = POSITION_SIZE_PERCENT
CHECK_INTERVAL = 30
REASON_LOG_INTERVAL = 300
_last_reason_log_time: float = 0.0
_daily_limit_logged_date: str = ""


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
    pattern_type: str
    pattern_target: float
    pattern_stop: float
    daily_start_balance: float
    daily_start_date: str
    consecutive_loss_count: int


def init_state() -> PaperState:
    balance = INITIAL_BALANCE
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
        pattern_type="",
        pattern_target=0.0,
        pattern_stop=0.0,
        daily_start_balance=INITIAL_BALANCE,
        daily_start_date="",
        consecutive_loss_count=0,
    )


def close_position(state: PaperState, candle: pd.Series, side: str, reason: str) -> None:
    price = float(candle["close"])
    entry_price = state.entry_price
    entry_regime = state.entry_regime
    
    if side == "LONG":
        pnl_pct = (price - entry_price) / entry_price * LEVERAGE
    else:  # SHORT
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
    state.pattern_type = ""
    state.pattern_target = 0.0
    state.pattern_stop = 0.0
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
    
    # PNL 및 ROE 계산
    total_pnl = state.equity - INITIAL_BALANCE
    roe = (total_pnl / INITIAL_BALANCE * 100) if INITIAL_BALANCE > 0 else 0.0
    
    kr = REGIME_KR.get(entry_regime, entry_regime or "?")
    log(f"{side} 청산 | {kr} | {reason} | 진입={entry_price:.2f} 청산={price:.2f} 수익률={pnl_pct*100:+.2f}% 손익={net_pnl:+.2f} 잔고={state.balance:.2f}")


def open_position(
    state: PaperState,
    price: float,
    side: str,
    reason: str,
    regime: str = "",
    price_history: Optional[list] = None,
    price_history_15m: Optional[list] = None,
    pattern_type: str = "",
    pattern_target: float = 0.0,
    pattern_stop: float = 0.0,
) -> None:
    current_time = time.time()
    trade_capital = state.balance * RISK_PER_TRADE
    if trade_capital <= 0:
        return

    fee = trade_capital * FEE_RATE
    slippage_cost = trade_capital * SLIPPAGE_PCT / 100
    trade_capital_after_fee = trade_capital - fee - slippage_cost
    position_size = trade_capital_after_fee / price

    box_high_entry = 0.0
    box_low_entry = 0.0
    if regime == "sideways" and price_history_15m and len(price_history_15m) >= REGIME_LOOKBACK_15M:
        box_high_entry = max(price_history_15m[-REGIME_LOOKBACK_15M:])
        box_low_entry = min(price_history_15m[-REGIME_LOOKBACK_15M:])
    
    if side == "LONG":
        state.has_long_position = True
    else:  # SHORT
        state.has_short_position = True
    
    state.entry_price = price
    state.position_size = position_size
    state.position_entry_time = current_time
    state.trailing_stop_active = False
    state.best_pnl_pct = 0.0
    state.entry_regime = regime
    state.box_high = box_high_entry
    state.box_low = box_low_entry
    state.pattern_type = pattern_type
    state.pattern_target = pattern_target
    state.pattern_stop = pattern_stop
    
    if side == "LONG":
        state.highest_price = price
    else:  # SHORT
        state.lowest_price = price
    
    kr = REGIME_KR.get(regime, regime or "?")
    box_str = ""
    if regime == "sideways" and state.box_high > 0 and state.box_low > 0:
        box_str = f" | 박스 상단={state.box_high:.2f} 하단={state.box_low:.2f}"
    if pattern_target > 0 and pattern_stop > 0:
        log(f"{side} 진입 | {kr} | {reason} | 가격={price:.2f} 잔고={state.balance:.2f} 투입={trade_capital:.2f} | 패턴={pattern_type} 목표가={pattern_target:.2f} 손절가={pattern_stop:.2f}{box_str}")
    else:
        is_trend = regime == "neutral"
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


def _reason_no_exit_strategy(regime: str, is_long: bool, rsi: float, price: float, short_ma: float) -> str:
    return "청산 조건 미충족"


def _reason_no_exit_scalp(state: PaperState, pnl_pct: float, side: str, current_price: float) -> str:
    return "손절/익절 조건 미도달"


def check_scalp_stop_loss_and_profit(state: PaperState, current_price: float, candle: pd.Series) -> bool:
    if state.has_long_position:
        if state.pattern_target > 0 and state.pattern_stop > 0:
            if current_price >= state.pattern_target:
                close_position(state, candle, "LONG", f"패턴익절({state.pattern_type})")
                return True
            if current_price <= state.pattern_stop:
                close_position(state, candle, "LONG", f"패턴손절({state.pattern_type})")
                return True
            if _should_log_reason():
                regime_kr = REGIME_KR.get(state.entry_regime or "", state.entry_regime or "?")
                log(f"[손절/익절청산안함] {regime_kr} | 패턴 익절/손절 대기 중 ({state.pattern_type})")
            return False  # 패턴 대기 중, config 기반 로직 스킵
        
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
            rsi=rsi,
            price=current_price,
            entry_price=state.entry_price,
            best_pnl_pct=state.best_pnl_pct,
            box_high=state.box_high or 0,
            box_low=state.box_low or 0,
        )
        if reason:
            msg = reason_to_display_message(reason, is_long=True)
            close_position(state, candle, "LONG", f"{msg} ({pnl_pct:.2f}%)")
            return True

    elif state.has_short_position:
        if state.pattern_target > 0 and state.pattern_stop > 0:
            if current_price <= state.pattern_target:
                close_position(state, candle, "SHORT", f"패턴익절({state.pattern_type})")
                return True
            if current_price >= state.pattern_stop:
                close_position(state, candle, "SHORT", f"패턴손절({state.pattern_type})")
                return True
            if _should_log_reason():
                log(f"[청산대기] 패턴 익절/손절 대기 ({state.pattern_type})")
            return False

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
            rsi=rsi,
            price=current_price,
            entry_price=state.entry_price,
            best_pnl_pct=state.best_pnl_pct,
            box_high=state.box_high or 0,
            box_low=state.box_low or 0,
        )
        if reason:
            msg = reason_to_display_message(reason, is_long=False)
            close_position(state, candle, "SHORT", f"{msg} ({pnl_pct:.2f}%)")
            return True

    if state.has_long_position or state.has_short_position:
        side = "LONG" if state.has_long_position else "SHORT"
        if side == "LONG":
            pnl_pct = (current_price - state.entry_price) / state.entry_price * LEVERAGE * 100
        else:
            pnl_pct = (state.entry_price - current_price) / state.entry_price * LEVERAGE * 100
        if _should_log_reason():
            log(f"[청산대기] {_reason_no_exit_scalp(state, pnl_pct, side, current_price)}")
    return False


def apply_strategy_on_candle(
    state: PaperState, candle: pd.Series, df: Optional[pd.DataFrame] = None, regime: Optional[str] = None
) -> None:
    price = float(candle["close"])
    rsi = float(candle["rsi"])
    short_ma = float(candle["ma_short"])
    long_ma = float(candle["ma_long"])
    
    short_ma_15m, long_ma_15m, ma_50_15m, ma_100_15m, price_history_15m = 0.0, 0.0, 0.0, 0.0, []
    rsi_15m, macd_line_15m, macd_signal_15m = None, None, None
    if regime is None and df is not None and len(df) >= REGIME_LOOKBACK_15M * 3:
        regime, short_ma_15m, long_ma_15m, ma_50_15m, ma_100_15m, price_history_15m, rsi_15m, macd_line_15m, macd_signal_15m = compute_regime_15m(df, price)
    if regime is None:
        regime = "neutral"
    
    has_position = state.has_long_position or state.has_short_position
    is_long = state.has_long_position
    
    price_history = df["close"].tail(SIDEWAYS_BOX_PERIOD + 1).tolist() if df is not None and len(df) >= SIDEWAYS_BOX_PERIOD else None

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
        macd_line=None,
        macd_signal=None,
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

    # 패턴 감지: 15분봉 72시간(288봉) 데이터 사용
    pattern_info = None
    if df is not None and len(df) >= PATTERN_LOOKBACK * 3 and not has_position:
        df_tmp = df.copy()
        df_tmp["timestamp"] = pd.to_datetime(df_tmp["timestamp"])
        df_15m = df_tmp.set_index("timestamp").resample("15min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        if len(df_15m) >= PATTERN_LOOKBACK:
            highs = df_15m["high"].iloc[-PATTERN_LOOKBACK:].tolist()
            lows = df_15m["low"].iloc[-PATTERN_LOOKBACK:].tolist()
            closes = df_15m["close"].iloc[-PATTERN_LOOKBACK:].tolist()
            pattern_info = detect_chart_pattern(highs, lows, closes, price)
    
    if signal == "long" and not has_position:
        if daily_limit_hit or consecutive_limit_hit:
            if daily_limit_hit:
                _log_daily_limit_once(current_date, daily_loss_pct, regime_kr)
            elif consecutive_limit_hit and _should_log_reason():
                log(f"[진입안함] {regime_kr} | 연속손실 {state.consecutive_loss_count}회 당일 중단")
        else:
            pt, ptg, pst = "", 0.0, 0.0
            if pattern_info and pattern_info.side == "LONG":
                pt, ptg, pst = pattern_info.name, pattern_info.target_price, pattern_info.stop_price
                reason_suffix = f" + 패턴익절/손절 ({pt})"
            else:
                reason_suffix = ""
            entry_reason = get_entry_reason(
                regime, "LONG", rsi_use, price, short_ma, long_ma,
                regime_short_ma=short_ma_15m if use_15m else None,
                regime_long_ma=long_ma_15m if use_15m else None,
                regime_ma_50=ma_50_15m if use_15m else None,
                regime_ma_100=ma_100_15m if use_15m else None,
                regime_price_history=regime_price_hist,
                price_history=price_history,
            ) + reason_suffix
            open_position(
                state, price, "LONG", entry_reason.strip(),
                regime, price_history, price_history_15m=price_history_15m if regime == "sideways" else None,
                pattern_type=pt, pattern_target=ptg, pattern_stop=pst,
            )
    elif signal == "short" and not has_position:
        if daily_limit_hit or consecutive_limit_hit:
            if daily_limit_hit:
                _log_daily_limit_once(current_date, daily_loss_pct, regime_kr)
            elif consecutive_limit_hit and _should_log_reason():
                log(f"[진입안함] {regime_kr} | 연속손실 {state.consecutive_loss_count}회 당일 중단")
        else:
            pt, ptg, pst = "", 0.0, 0.0
            if pattern_info and pattern_info.side == "SHORT":
                pt, ptg, pst = pattern_info.name, pattern_info.target_price, pattern_info.stop_price
                reason_suffix = f" + 패턴익절/손절 ({pt})"
            else:
                reason_suffix = ""
            entry_reason = get_entry_reason(
                regime, "SHORT", rsi_use, price, short_ma, long_ma,
                regime_short_ma=short_ma_15m if use_15m else None,
                regime_long_ma=long_ma_15m if use_15m else None,
                regime_ma_50=ma_50_15m if use_15m else None,
                regime_ma_100=ma_100_15m if use_15m else None,
                regime_price_history=regime_price_hist,
                price_history=price_history,
            ) + reason_suffix
            open_position(
                state, price, "SHORT", entry_reason.strip(),
                regime, price_history, price_history_15m=price_history_15m if regime == "sideways" else None,
                pattern_type=pt, pattern_target=ptg, pattern_stop=pst,
            )
    elif signal == "flat" and has_position:
        # 패턴 진입(패턴 익절/손절 대기 중)이면 추세 반전으로 청산하지 않음. 패턴 TP/SL만 적용.
        if state.pattern_target > 0 and state.pattern_stop > 0:
            pass  # 청산 안 함, check_scalp_stop_loss_and_profit에서 패턴 익절/손절만 처리
        else:
            side = "LONG" if is_long else "SHORT"
            close_position(state, candle, side, f"단타청산({regime_kr})")
    else:
        # 5분봉마다 진입안함/청산대기 사유 로그
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
                    price_history=price_history,
                )
                log(f"[진입안함] {regime_kr} | {reason}")
        elif has_position and signal == "hold":
            log(f"[청산대기] {regime_kr} | {_reason_no_exit_strategy(regime, is_long, rsi, price, short_ma)}")

    if state.has_long_position:
        unrealized = (
            (price - state.entry_price)
            / state.entry_price
            * LEVERAGE
            * state.balance
            * RISK_PER_TRADE
        )
        equity = state.balance + unrealized
    elif state.has_short_position:
        unrealized = (
            (state.entry_price - price)
            / state.entry_price
            * LEVERAGE
            * state.balance
            * RISK_PER_TRADE
        )
        equity = state.balance + unrealized
    else:
        equity = state.balance
    
    state.equity = equity
    if equity > state.peak_equity:
        state.peak_equity = equity
    
    drawdown = (
        (state.peak_equity - equity) / state.peak_equity if state.peak_equity > 0 else 0.0
    )
    state.max_drawdown = max(state.max_drawdown, drawdown)


def main() -> None:
    exchange = get_public_exchange()
    state = init_state()

    log(f"시작 잔고={INITIAL_BALANCE:.2f}")
    
    last_candle_time = None

    try:
        while True:
            # 충분한 데이터 확보 (15분봉 72시간 패턴 = 288*3 = 864개 5분봉 필요)
            limit = max(RSI_PERIOD, MA_LONGEST_PERIOD, REGIME_LOOKBACK_15M * 3, PATTERN_LOOKBACK * 3) + 100
            df = fetch_ohlcv(exchange, limit=limit)
            
            # 지표 계산
            df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
            macd_line, signal_line, _ = calculate_macd(df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            df["macd_line"] = macd_line
            df["macd_signal"] = signal_line
            df["ma_short"] = calculate_ma(df["close"], MA_SHORT_PERIOD)
            df["ma_long"] = calculate_ma(df["close"], MA_LONG_PERIOD)
            df["ma_50"] = calculate_ma(df["close"], MA_MID_PERIOD)
            df["ma_100"] = calculate_ma(df["close"], MA_LONGEST_PERIOD)
            
            # 볼륨 평균 계산 (20기간 이동평균)
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            
            df = df.dropna().reset_index(drop=True)

            latest = df.iloc[-1]
            latest_time = latest["timestamp"]
            current_price = float(latest["close"])

            # 청산 우선 체크
            if state.has_long_position or state.has_short_position:
                if check_scalp_stop_loss_and_profit(state, current_price, latest):
                    pos_status = "NONE"
                    total_pnl = state.equity - INITIAL_BALANCE
                    roe = (total_pnl / INITIAL_BALANCE * 100) if INITIAL_BALANCE > 0 else 0.0
                    
                    time.sleep(CHECK_INTERVAL)
                    continue

            # 첫 루프
            if last_candle_time is None:
                last_candle_time = latest_time
                pos_status = (
                    "LONG" if state.has_long_position
                    else ("SHORT" if state.has_short_position else "NONE")
                )
                total_pnl = state.equity - INITIAL_BALANCE
                roe = (total_pnl / INITIAL_BALANCE * 100) if INITIAL_BALANCE > 0 else 0.0
                log(f"시작 포지션={pos_status} 가격={current_price:.2f} 잔고={state.balance:.2f} PNL={total_pnl:+.2f}")
            # 새 캔들 완성 시 전략 적용
            elif latest_time > last_candle_time:
                price = float(latest["close"])
                rsi = float(latest["rsi"])
                had_position_before = state.has_long_position or state.has_short_position

                apply_strategy_on_candle(state, latest, df)
                last_candle_time = latest_time
                just_entered = not had_position_before and (state.has_long_position or state.has_short_position)

                pos_status = (
                    "LONG" if state.has_long_position
                    else ("SHORT" if state.has_short_position else "NONE")
                )
                total_pnl = state.equity - INITIAL_BALANCE
                roe = (total_pnl / INITIAL_BALANCE * 100) if INITIAL_BALANCE > 0 else 0.0
                unrealized_pnl_pct = 0.0
                if state.has_long_position:
                    unrealized_pnl_pct = (price - state.entry_price) / state.entry_price * LEVERAGE * 100
                elif state.has_short_position:
                    unrealized_pnl_pct = (state.entry_price - price) / state.entry_price * LEVERAGE * 100
                extra = f" 미실현={unrealized_pnl_pct:+.2f}%" if unrealized_pnl_pct != 0 else ""
                regime_str = f" | {REGIME_KR.get(state.entry_regime, state.entry_regime)}" if (state.has_long_position or state.has_short_position) and state.entry_regime else ""
                box_str = ""
                if state.entry_regime == "sideways" and state.box_high > 0 and state.box_low > 0:
                    box_str = f" | 박스 상단={state.box_high:.2f} 하단={state.box_low:.2f}"
                # 진입한 봉에서는 진입 로그만 1번 남기고 [5m] 상태 로그는 생략
                if not just_entered:
                    log(f"[5m] {pos_status}{regime_str}{box_str} 가격={price:.2f} RSI={rsi:.0f} 잔고={state.balance:.2f} PNL={total_pnl:+.2f}{extra}")

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        total_pnl = state.equity - INITIAL_BALANCE
        roe = (total_pnl / INITIAL_BALANCE * 100) if INITIAL_BALANCE > 0 else 0.0
        log(f"종료 잔고={state.balance:.2f} PNL={total_pnl:+.2f}")
    except Exception as e:
        log(f"오류: {e}", "ERROR")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
