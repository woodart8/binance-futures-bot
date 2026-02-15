"""페이퍼 트레이딩. python paper_trading.py"""

import sys
import config_paper
sys.modules["config"] = config_paper
import exit_logic_paper
sys.modules["exit_logic"] = exit_logic_paper
import strategy_core_paper
sys.modules["strategy_core"] = strategy_core_paper

import time
import pandas as pd

from config import (
    INITIAL_BALANCE,
    LEVERAGE,
    RSI_PERIOD,
    MA_LONGEST_PERIOD,
    REGIME_LOOKBACK_15M,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    MA_SHORT_PERIOD,
    MA_LONG_PERIOD,
    MA_MID_PERIOD,
)
from exchange_client import get_public_exchange
from data import fetch_ohlcv
from indicators import calculate_rsi, calculate_ma, calculate_macd
from chart_patterns import PATTERN_LOOKBACK
from strategy_core import REGIME_KR
from logger import log

from trading_logic_paper import (
    PaperState,
    init_state,
    check_scalp_stop_loss_and_profit,
    apply_strategy_on_candle,
)

CHECK_INTERVAL = 30


def main() -> None:
    exchange = get_public_exchange()
    state = init_state()

    log(f"시작 잔고={INITIAL_BALANCE:.2f}")

    last_candle_time = None
    limit = max(RSI_PERIOD, MA_LONGEST_PERIOD, REGIME_LOOKBACK_15M * 3, PATTERN_LOOKBACK * 3) + 100

    try:
        while True:
            df = fetch_ohlcv(exchange, limit=limit)

            df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
            macd_line, signal_line, _ = calculate_macd(df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            df["macd_line"] = macd_line
            df["macd_signal"] = signal_line
            df["ma_short"] = calculate_ma(df["close"], MA_SHORT_PERIOD)
            df["ma_long"] = calculate_ma(df["close"], MA_LONG_PERIOD)
            df["ma_50"] = calculate_ma(df["close"], MA_MID_PERIOD)
            df["ma_100"] = calculate_ma(df["close"], MA_LONGEST_PERIOD)
            df["volume_ma"] = df["volume"].rolling(window=20).mean()

            df = df.dropna().reset_index(drop=True)

            latest = df.iloc[-1]
            latest_time = latest["timestamp"]
            current_price = float(latest["close"])

            if state.has_long_position or state.has_short_position:
                if check_scalp_stop_loss_and_profit(state, current_price, latest):
                    time.sleep(CHECK_INTERVAL)
                    continue

            if last_candle_time is None:
                last_candle_time = latest_time
                pos_status = (
                    "LONG" if state.has_long_position
                    else ("SHORT" if state.has_short_position else "NONE")
                )
                total_pnl = state.equity - INITIAL_BALANCE
                log(f"시작 포지션={pos_status} 가격={current_price:.2f} 잔고={state.balance:.2f} PNL={total_pnl:+.2f}")
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
                unrealized_pnl_pct = 0.0
                if state.has_long_position:
                    unrealized_pnl_pct = (price - state.entry_price) / state.entry_price * LEVERAGE * 100
                elif state.has_short_position:
                    unrealized_pnl_pct = (state.entry_price - price) / state.entry_price * LEVERAGE * 100
                extra = f" 미실현={unrealized_pnl_pct:+.2f}%" if unrealized_pnl_pct != 0 else ""
                regime_str = f" | {REGIME_KR.get(state.entry_regime, state.entry_regime)}" if (state.has_long_position or state.has_short_position) and state.entry_regime else ""
                box_str = ""
                if state.entry_regime == "sideways" and state.box_high > 0 and state.box_low > 0:
                    box_str = f" | 박스 하단={state.box_low:.2f} 상단={state.box_high:.2f}"
                if not just_entered:
                    log(f"[5m] {pos_status}{regime_str}{box_str} 가격={price:.2f} RSI={rsi:.0f} 잔고={state.balance:.2f} PNL={total_pnl:+.2f}{extra}")

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        total_pnl = state.equity - INITIAL_BALANCE
        log(f"종료 잔고={state.balance:.2f} PNL={total_pnl:+.2f}")
    except Exception as e:
        log(f"오류: {e}", "ERROR")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
