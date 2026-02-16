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
    LIVE_CHECK_INTERVAL,
)
from exchange_client import get_private_exchange
from data import fetch_ohlcv
from indicators import calculate_rsi, calculate_ma
from logger import log

from trading_logic_live import (
    get_balance_usdt,
    init_live_state,
    process_live_candle,
    log_5m_status,
    check_tp_sl_and_close,
    try_live_entry,
)

OHLCV_CONSECUTIVE_FAILURE_LIMIT = 30


def set_leverage_and_margin(exchange) -> None:
    market = exchange.market(SYMBOL)
    exchange.set_leverage(LEVERAGE, market["id"])
    exchange.set_margin_mode("isolated", market["id"])


def main() -> None:
    exchange = get_private_exchange()
    set_leverage_and_margin(exchange)

    state = init_live_state()
    ohlcv_failure_count = 0
    # MA100은 15분봉 기준 100개 필요 = 5분봉 기준 300개
    # REGIME_LOOKBACK_15M은 15분봉 기준 96개 = 5분봉 기준 288개
    # 충분한 데이터 확보를 위해 여유분 추가
    # MA100 계산 후 최근 96개 사용을 위해 최소 196개 15분봉 필요 = 588개 5분봉
    # 충분한 데이터 확보를 위해 여유분 추가 (resample 후 데이터 손실 고려)
    limit = (MA_LONGEST_PERIOD + REGIME_LOOKBACK_15M) * 3 + 200

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
            df["ma_short"] = calculate_ma(df["close"], MA_SHORT_PERIOD)
            df["ma_long"] = calculate_ma(df["close"], MA_LONG_PERIOD)
            df["ma_50"] = calculate_ma(df["close"], MA_MID_PERIOD)
            df["ma_100"] = calculate_ma(df["close"], MA_LONGEST_PERIOD)
            df = df.dropna().reset_index(drop=True)

            latest = df.iloc[-1]
            latest_time = latest["timestamp"]
            price = float(latest["close"])

            try:
                ticker = exchange.fetch_ticker(SYMBOL)
                current_price = float(ticker["last"])
            except Exception as e:
                log(f"현재가 조회 실패: {e}", "ERROR")
                current_price = price

            # 30초 단위: 포지션 보유 시 현재가로 익절/손절 체크
            if state.get("has_position"):
                state, did_close = check_tp_sl_and_close(exchange, state, current_price, df)
                if did_close:
                    time.sleep(LIVE_CHECK_INTERVAL)
                    continue
            else:
                # 30초 단위: 진입 조건 만족 시 현재가로 진입
                state, did_enter = try_live_entry(exchange, state, df, current_price)
                if did_enter:
                    time.sleep(LIVE_CHECK_INTERVAL)
                    continue

            if state["last_candle_time"] is None:
                state["last_candle_time"] = latest_time
                try:
                    bal = get_balance_usdt(exchange)  # 거래소 잔고(펀딩 포함)
                except Exception:
                    bal = 0.0
                log(f"[시작] 가격={price:.2f} 잔고={bal:.2f}")
            elif latest_time > state["last_candle_time"]:
                state, skip, did_close = process_live_candle(exchange, state, df)
                if skip:
                    time.sleep(60)
                    continue
                # 진입/청산 시에는 해당 로그만 남기고 5분 로그는 생략
                if not did_close:
                    log_5m_status(exchange, state, df)

            time.sleep(LIVE_CHECK_INTERVAL)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
