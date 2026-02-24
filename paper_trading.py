"""페이퍼 트레이딩. python paper_trading.py

진입·청산 판단: 새 1분봉이 생길 때마다, 방금 종료된 1분봉 종가로 판단.
장세/RSI/MA: 1분봉을 5분봉으로 리샘플한 뒤 15분봉으로 다시 리샘플해 계산(15m RSI 기간 12, MA7/20/50/100).
같은 1분봉에서 청산한 경우 해당 봉에서는 재진입하지 않음. 상태 로그(`[1분] ...`)는 새 1분봉마다, 내용은 15분봉 장세/RSI/MA 기준.
"""

import sys
import config_paper
sys.modules["config"] = config_paper
import exit_logic_paper
sys.modules["exit_logic"] = exit_logic_paper
import strategy_core_paper
sys.modules["strategy_core"] = strategy_core_paper

import time
from datetime import datetime, timezone
import pandas as pd

from config import (
    INITIAL_BALANCE,
    LEVERAGE,
    RSI_PERIOD,
    MA_LONGEST_PERIOD,
    REGIME_LOOKBACK_15M,
    MA_SHORT_PERIOD,
    MA_LONG_PERIOD,
    MA_MID_PERIOD,
    SYMBOL,
    TREND_SLOPE_BARS,
    TREND_SLOPE_MIN_PCT,
    SIDEWAYS_BOX_EXIT_MARGIN_PCT,
    SIDEWAYS_BOX_SLOPE_MAX,
    SIDEWAYS_BOX_SLOPE_DIFF_MAX,
    SIDEWAYS_BOX_RANGE_PCT_MIN,
    SIDEWAYS_BOX_RANGE_MIN,
)
from exchange_client import get_public_exchange
from data import fetch_ohlcv_1m_min_bars, compute_regime_15m, resample_1m_to_5m
from indicators import calculate_rsi, calculate_ma
from strategy_core import REGIME_KR, get_sideways_box_bounds
from logger import log

from trading_logic_paper import (
    PaperState,
    init_state,
    check_scalp_stop_loss_and_profit,
    apply_strategy_on_candle,
    try_paper_entry,
    apply_funding_if_needed,
)

# 루프 주기(초). 이 간격마다 OHLCV 조회; 진입·청산 판단은 새 1분봉이 뜰 때만 수행.
CHECK_INTERVAL = 10


def main() -> None:
    exchange = get_public_exchange()
    state = init_state()

    log(f"시작 잔고={INITIAL_BALANCE:.2f}")

    last_candle_time = None
    # 15분봉 96개+MA100 확보: 15m 196개 이상 → 5m 588개 → 1m 2940개 이상 (2000이면 유효 15m 부족해 항상 중립)
    limit_1m = 3000

    try:
        while True:
            df_1m = fetch_ohlcv_1m_min_bars(exchange, min_bars=limit_1m)
            if df_1m.empty or len(df_1m) < 2:
                time.sleep(CHECK_INTERVAL)
                continue

            df_5m = resample_1m_to_5m(df_1m)
            df_5m["rsi"] = calculate_rsi(df_5m["close"], RSI_PERIOD)
            df_5m["ma_short"] = calculate_ma(df_5m["close"], MA_SHORT_PERIOD)
            df_5m["ma_long"] = calculate_ma(df_5m["close"], MA_LONG_PERIOD)
            df_5m["ma_50"] = calculate_ma(df_5m["close"], MA_MID_PERIOD)
            df_5m["ma_100"] = calculate_ma(df_5m["close"], MA_LONGEST_PERIOD)
            df_5m["volume_ma"] = df_5m["volume"].rolling(window=20).mean()
            # analyze_regime와 동일하게 5분봉 전체를 유지 (dropna로 앞부분 삭제하지 않음)
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

            # 00/08/16 UTC 펀딩비 적용 (포지션 보유 시)
            if state.has_long_position or state.has_short_position:
                now_utc = datetime.now(timezone.utc)
                if apply_funding_if_needed(state, exchange, now_utc, current_price):
                    time.sleep(CHECK_INTERVAL)
                    continue

            if last_candle_time is None:
                last_candle_time = latest_time
                pos_status = (
                    "LONG" if state.has_long_position
                    else ("SHORT" if state.has_short_position else "NONE")
                )
                total_pnl = state.equity - INITIAL_BALANCE  # 매매+펀딩 포함
                log(f"시작 포지션={pos_status} 가격={current_price:.2f} 잔고={state.balance:.2f} PNL={total_pnl:+.2f}")
            elif latest_time > last_candle_time and len(df_1m) >= 2:
                # 1분봉 기준: 방금 종료된 1분봉 종가로 진입·청산 판단. 장세/RSI는 5분봉(리샘플) 기준
                df_closed_1m = df_1m.iloc[:-1]
                bar_closed_1m = df_closed_1m.iloc[-1]
                price = float(bar_closed_1m["close"])
                df_closed_5m = df_5m.iloc[:-1] if len(df_5m) >= 2 else df_5m
                candle_5m_closed = df_closed_5m.iloc[-1]  # 전략용 5분봉 (rsi/MA 포함)

                if len(df_closed_5m) < REGIME_LOOKBACK_15M * 3:
                    last_candle_time = latest_time
                    time.sleep(CHECK_INTERVAL)
                    continue

                closed_this_candle = False  # 같은 봉에서 청산 시 해당 봉에서는 재진입 안 함
                # 1) 익절/손절/박스이탈 체크 (전봉 종가 기준) — 청산 시 로그는 즉시 출력
                if state.has_long_position or state.has_short_position:
                    if check_scalp_stop_loss_and_profit(state, price, bar_closed_1m, df_closed_5m):
                        closed_this_candle = True
                        last_candle_time = latest_time
                        time.sleep(CHECK_INTERVAL)
                        continue
                # 2) 전략 신호 청산(flat) 및 로그 (전봉 기준) — 청산 시 로그는 즉시 출력
                #    전략 로직은 5분봉(rsi/MA 포함)을 사용하므로 candle_5m_closed를 넘긴다.
                if not closed_this_candle:
                    did_close = apply_strategy_on_candle(state, candle_5m_closed, df_closed_5m, log_hold_info=False)
                    last_candle_time = latest_time
                    if did_close:
                        closed_this_candle = True
                        time.sleep(CHECK_INTERVAL)
                        continue
                # 3) 진입 (전봉 종가 기준) — 진입 시 로그는 즉시 출력
                if not closed_this_candle and not (state.has_long_position or state.has_short_position):
                    if try_paper_entry(state, df_closed_5m, price):
                        last_candle_time = latest_time
                        time.sleep(CHECK_INTERVAL)
                        continue

                # 상태 로그: 1분봉 생길 때마다 출력 (장세/RSI는 15분봉 기준으로 표시)
                regime_detail = ""
                rsi_15m = None
                if len(df_closed_5m) >= REGIME_LOOKBACK_15M * 3:
                    regime, short_ma_15m, long_ma_15m, ma_50_15m, ma_100_15m, price_history_15m, rsi_15m, _, ma_long_history = compute_regime_15m(df_closed_5m, price)
                    if regime == "trend":
                        if ma_long_history and len(ma_long_history) >= TREND_SLOPE_BARS:
                            recent_ma20 = ma_long_history[-TREND_SLOPE_BARS:]
                            ma20_start = recent_ma20[0]
                            ma20_end = recent_ma20[-1]
                            if ma20_start and ma20_start > 0:
                                slope_pct = (ma20_end - ma20_start) / ma20_start * 100.0
                                trend_dir = "상승" if slope_pct > 0 else "하락"
                                regime_detail = f" | 추세장({trend_dir}): MA20 기울기 {slope_pct:+.2f}% (기준 ±{TREND_SLOPE_MIN_PCT}%)"
                    elif regime == "sideways":
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

                pos_status = (
                    "LONG" if state.has_long_position
                    else ("SHORT" if state.has_short_position else "NONE")
                )
                total_pnl = state.equity - INITIAL_BALANCE  # 매매+펀딩 포함
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
                # RSI 표시는 15분봉 기준(rsi_15m)이 있으면 그 값을, 없으면 5분봉 RSI를 사용
                rsi_5m = float(df_closed_5m.iloc[-1]["rsi"]) if len(df_closed_5m) > 0 else 0
                rsi_display = float(rsi_15m) if rsi_15m is not None else rsi_5m
                log(f"[1분] {pos_status}{regime_str}{box_str}{regime_detail} | 가격={price:.2f} RSI={rsi_display:.0f} 잔고={state.balance:.2f} PNL={total_pnl:+.2f}{extra}")

                last_candle_time = latest_time
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        total_pnl = state.equity - INITIAL_BALANCE  # 매매+펀딩 포함
        log(f"종료 잔고={state.balance:.2f} PNL={total_pnl:+.2f}")
    except Exception as e:
        log(f"오류: {e}", "ERROR")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
