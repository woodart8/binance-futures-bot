"""실거래 에이전트. .env에 API_KEY, SECRET_KEY 필요."""

import sys
import config_live
sys.modules["config"] = config_live
import exit_logic_live
sys.modules["exit_logic"] = exit_logic_live
import strategy_core_live
sys.modules["strategy_core"] = strategy_core_live

import time
import pandas as pd

from config import (
    LEVERAGE,
    RSI_PERIOD,
    MA_SHORT_PERIOD,
    MA_LONG_PERIOD,
    MA_MID_PERIOD,
    MA_LONGEST_PERIOD,
    REGIME_LOOKBACK_15M,
    SYMBOL,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
)
from exchange_client import get_private_exchange
from data import fetch_ohlcv
from indicators import calculate_rsi, calculate_ma, calculate_macd
from chart_patterns import PATTERN_LOOKBACK
from logger import log

from trading_logic_live import (
    get_balance_usdt,
    init_live_state,
    process_live_candle,
    log_5m_status,
)

OHLCV_CONSECUTIVE_FAILURE_LIMIT = 30


def set_leverage(exchange) -> None:
    market = exchange.market(SYMBOL)
    exchange.set_leverage(LEVERAGE, market["id"])


def main() -> None:
    exchange = get_private_exchange()
    set_leverage(exchange)

    state = init_live_state()
    ohlcv_failure_count = 0
    limit = max(RSI_PERIOD, MA_LONGEST_PERIOD, REGIME_LOOKBACK_15M * 3, PATTERN_LOOKBACK * 3) + 100

    try:
        while True:
            try:
                df = fetch_ohlcv(exchange, limit=limit)
                ohlcv_failure_count = 0
            except Exception as e:
                ohlcv_failure_count += 1
                log(f"OHLCV 조회 실패 ({ohlcv_failure_count}/{OHLCV_CONSECUTIVE_FAILURE_LIMIT}): {e}", "ERROR")
                if ohlcv_failure_count >= OHLCV_CONSECUTIVE_FAILURE_LIMIT:
                    log(f"OHLCV 조회가 {OHLCV_CONSECUTIVE_FAILURE_LIMIT}회 연속 실패하여 프로세스를 종료합니다. 네트워크/API를 확인 후 재실행하세요.", "ERROR")
                    sys.exit(1)
                time.sleep(60)
                continue

            df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
            macd_line, signal_line, _ = calculate_macd(df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            df["macd_line"] = macd_line
            df["macd_signal"] = signal_line
            df["ma_short"] = calculate_ma(df["close"], MA_SHORT_PERIOD)
            df["ma_long"] = calculate_ma(df["close"], MA_LONG_PERIOD)
            df["ma_50"] = calculate_ma(df["close"], MA_MID_PERIOD)
            df["ma_100"] = calculate_ma(df["close"], MA_LONGEST_PERIOD)
            df = df.dropna().reset_index(drop=True)

            latest = df.iloc[-1]
            latest_time = latest["timestamp"]
            price = float(latest["close"])

            if state["last_candle_time"] is None:
                state["last_candle_time"] = latest_time
                try:
                    bal = get_balance_usdt(exchange)
                except Exception:
                    bal = 0.0
                log(f"[시작] 가격={price:.2f} 잔고={bal:.2f}")
            elif latest_time > state["last_candle_time"]:
                state, skip = process_live_candle(exchange, state, df)
                if skip:
                    time.sleep(60)
                    continue
                log_5m_status(exchange, state, df)

            time.sleep(300)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
