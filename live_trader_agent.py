"""실거래 에이전트. .env에 API_KEY, SECRET_KEY 필요.

진입·청산은 백테스트와 동일: 새 5분봉이 데이터에 뜬 뒤, 방금 종료된 봉(전봉)의 종가로만 판단.
같은 봉에서 청산한 경우 해당 봉에서는 재진입하지 않음.
"""

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
    # set_leverage/set_margin_mode는 심볼 id 필요. 바이낸스 선물 id는 "BTCUSDT" 형식(SYMBOL에서 "/" 제거).
    symbol_id = SYMBOL.replace("/", "")
    try:
        exchange.set_leverage(LEVERAGE, symbol_id)
        exchange.set_margin_mode("isolated", symbol_id)
    except Exception as e:
        err = str(e).lower()
        if "2015" in err or "invalid api-key" in err or "permissions" in err:
            log(
                "레버리지/마진 설정 실패 (-2015). 확인: 1) API 키 권한에 '선물' 체크 2) IP 제한 시 서버 IP 허용 3) 키/시크릿 재확인",
                "ERROR",
            )
        raise


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

            if state["last_candle_time"] is None:
                state["last_candle_time"] = latest_time
                try:
                    bal = get_balance_usdt(exchange)  # 거래소 잔고(펀딩 포함)
                except Exception:
                    bal = 0.0
                log(f"[시작] 가격={price:.2f} 잔고={bal:.2f}")
            elif latest_time > state["last_candle_time"] and len(df) >= 2:
                # 백테스트와 동일: 새 봉이 뜬 뒤에는 방금 종료된 봉(전봉)의 종가로 진입·청산 판단
                df_closed = df.iloc[:-1]  # 진행 중인 봉 제외, 전봉까지
                bar_closed = df_closed.iloc[-1]  # 방금 종료된 봉
                close_price = float(bar_closed["close"])
                closed_this_candle = False  # 같은 봉에서 청산 시 해당 봉에서는 재진입 안 함 (백테스트와 동일)
                # 1) 익절/손절/박스이탈 체크 (전봉 종가 기준)
                if state.get("has_position"):
                    state, did_close = check_tp_sl_and_close(exchange, state, close_price, df_closed)
                    if did_close:
                        closed_this_candle = True
                        state["last_candle_time"] = latest_time
                        time.sleep(LIVE_CHECK_INTERVAL)
                        continue
                # 2) 전략 신호 청산(flat) 및 5분봉 상태 처리 (전봉 기준)
                if not closed_this_candle:
                    state, skip, did_close = process_live_candle(exchange, state, df_closed)
                    state["last_candle_time"] = latest_time
                    if skip:
                        time.sleep(60)
                        continue
                    if did_close:
                        closed_this_candle = True
                        time.sleep(LIVE_CHECK_INTERVAL)
                        continue
                # 3) 진입 (전봉 종가 기준) — 같은 봉에서 청산한 경우 이 봉에서는 진입하지 않음
                if not closed_this_candle and not state.get("has_position"):
                    state, did_enter = try_live_entry(exchange, state, df_closed, close_price)
                    if did_enter:
                        time.sleep(LIVE_CHECK_INTERVAL)
                        continue
                # 5분 로그 (전봉 기준)
                log_5m_status(exchange, state, df_closed)

            time.sleep(LIVE_CHECK_INTERVAL)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
