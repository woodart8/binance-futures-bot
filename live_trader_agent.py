"""실거래 에이전트. .env에 API_KEY, SECRET_KEY 필요.

진입·청산: 새 1분봉이 생길 때마다 방금 종료된 1분봉 종가로 판단(백테스트 1m·페이퍼와 동일).
장세/RSI/MA: 1분봉을 5분봉으로 리샘플한 뒤 15분봉으로 다시 리샘플해 계산(15m RSI 기간 12, MA7/20/50/100).
같은 1분봉에서 청산한 경우 해당 봉에서는 재진입하지 않음. 상태 로그(`[1분] ...`)는 새 1분봉마다, 내용은 15분봉 장세/RSI/MA 기준.
포지션 동기화: 시작 시·매 루프 거래소 실제 포지션과 state 일치. USE_EXCHANGE_TP_SL=True면 진입 시 거래소에 익절/손절 주문 등록.
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
from data import fetch_ohlcv_1m_min_bars, resample_1m_to_5m
from indicators import calculate_rsi, calculate_ma
from logger import log

from trading_logic_live import (
    get_balance_usdt,
    init_live_state,
    sync_state_from_exchange,
    process_live_candle,
    log_5m_status,
    check_tp_sl_and_close,
    try_live_entry,
)

OHLCV_CONSECUTIVE_FAILURE_LIMIT = 30
# 1분봉 개수: 15분봉 100개(MA100) 확보용 1500봉 이상
MIN_1M_BARS = 100 * 15
# 15분봉 96개+MA100 확보: 15m 196개 이상 → 5m 588개 → 1m 2940개 이상
LIMIT_1M = 3000


def set_leverage_and_margin(exchange) -> None:
    # set_leverage/set_margin_mode는 심볼 id 필요. 바이낸스 선물 id는 "BTCUSDT" 형식(SYMBOL에서 "/" 제거).
    symbol_id = SYMBOL.replace("/", "")
    try:
        exchange.set_leverage(LEVERAGE, symbol_id)
        exchange.set_margin_mode("isolated", symbol_id)
    except Exception as e:
        err = str(e)
        err_lower = err.lower()
        # -4161: 오픈 포지션 있을 때 격리마진 모드에서 레버리지 축소 불가. 경고만 남기고 진행.
        if "4161" in err or "leverage reduction" in err_lower or "open positions" in err_lower:
            log(
                f"[레버리지] 오픈 포지션으로 인해 설정 생략 (현재 레버 유지). 포지션 청산 후 재시작하면 {LEVERAGE}배 적용. 원문: {e!r}",
                "WARNING",
            )
            return
        if "2015" in err_lower or "invalid api-key" in err_lower or "permissions" in err_lower:
            log(
                "레버리지/마진 설정 실패 (-2015). 확인: 1) API 키 권한에 '선물' 체크 2) IP 제한 시 서버 IP 허용 3) 키/시크릿 재확인",
                "ERROR",
            )
        raise


def main() -> None:
    exchange = get_private_exchange()
    set_leverage_and_margin(exchange)

    state = init_live_state()
    state = sync_state_from_exchange(exchange, state)  # 재시작 시 거래소 실제 포지션과 로그 상태 맞춤
    ohlcv_failure_count = 0

    try:
        while True:
            try:
                df_1m = fetch_ohlcv_1m_min_bars(exchange, min_bars=LIMIT_1M)
                ohlcv_failure_count = 0
            except Exception as e:
                ohlcv_failure_count += 1
                log(f"OHLCV(1m) 조회 실패 ({ohlcv_failure_count}/{OHLCV_CONSECUTIVE_FAILURE_LIMIT}): {e}", "ERROR")
                if ohlcv_failure_count >= OHLCV_CONSECUTIVE_FAILURE_LIMIT:
                    log(f"OHLCV 조회가 {OHLCV_CONSECUTIVE_FAILURE_LIMIT}회 연속 실패하여 프로세스를 종료합니다. 네트워크/API를 확인 후 재실행하세요.", "ERROR")
                    sys.exit(1)
                time.sleep(60)
                continue

            if df_1m.empty or len(df_1m) < 2:
                time.sleep(LIVE_CHECK_INTERVAL)
                continue

            state = sync_state_from_exchange(exchange, state)  # 매 루프마다 로그 포지션 = 실제 포지션 유지
            df_5m = resample_1m_to_5m(df_1m)
            df_5m["rsi"] = calculate_rsi(df_5m["close"], RSI_PERIOD)
            df_5m["ma_short"] = calculate_ma(df_5m["close"], MA_SHORT_PERIOD)
            df_5m["ma_long"] = calculate_ma(df_5m["close"], MA_LONG_PERIOD)
            df_5m["ma_50"] = calculate_ma(df_5m["close"], MA_MID_PERIOD)
            df_5m["ma_100"] = calculate_ma(df_5m["close"], MA_LONGEST_PERIOD)
            # 장세 계산용 5분봉 히스토리는 analyze_regime와 동일하게 전체 사용 (dropna로 앞부분을 날리지 않음)
            df_5m = df_5m.reset_index(drop=True)

            latest_1m = df_1m.iloc[-1]
            latest_time = latest_1m["timestamp"]
            price = float(latest_1m["close"])

            try:
                ticker = exchange.fetch_ticker(SYMBOL)
                current_price = float(ticker["last"])
            except Exception as e:
                log(f"현재가 조회 실패: {e}", "ERROR")
                current_price = price

            if state["last_candle_time"] is None:
                state["last_candle_time"] = latest_time
                try:
                    bal = get_balance_usdt(exchange)
                except Exception:
                    bal = 0.0
                log(f"[시작] 가격={price:.2f} 잔고={bal:.2f}")
            elif latest_time > state["last_candle_time"] and len(df_1m) >= 2:
                # 1분봉 기준: 방금 종료된 1분봉 종가로 진입·청산 판단. 장세/RSI는 5m 리샘플 데이터 사용
                if len(df_5m) < REGIME_LOOKBACK_15M * 3 or len(df_1m) < MIN_1M_BARS:
                    state["last_candle_time"] = latest_time
                    time.sleep(LIVE_CHECK_INTERVAL)
                    continue

                df_closed_1m = df_1m.iloc[:-1]
                bar_closed_1m = df_closed_1m.iloc[-1]
                close_price = float(bar_closed_1m["close"])
                df_closed = df_5m  # 전략/장세는 5분봉(리샘플) 기준

                closed_this_candle = False
                if state.get("has_position"):
                    state, did_close = check_tp_sl_and_close(exchange, state, close_price, df_closed)
                    if did_close:
                        closed_this_candle = True
                        state["last_candle_time"] = latest_time
                        time.sleep(LIVE_CHECK_INTERVAL)
                        continue
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
                if not closed_this_candle and not state.get("has_position"):
                    state, did_enter = try_live_entry(exchange, state, df_closed, close_price, log_hold_info=False)
                    if did_enter:
                        state["last_candle_time"] = latest_time
                        time.sleep(LIVE_CHECK_INTERVAL)
                        continue
                # 상태 로그: 1분봉 생길 때마다 출력 (페이퍼와 동일)
                log_5m_status(exchange, state, df_closed)

                state["last_candle_time"] = latest_time
            time.sleep(LIVE_CHECK_INTERVAL)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
