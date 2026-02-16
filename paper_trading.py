"""페이퍼 트레이딩. python paper_trading.py"""

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
)
from exchange_client import get_public_exchange
from data import fetch_ohlcv
from indicators import calculate_rsi, calculate_ma
from strategy_core import REGIME_KR
from logger import log

from trading_logic_paper import (
    PaperState,
    init_state,
    check_scalp_stop_loss_and_profit,
    apply_strategy_on_candle,
    try_paper_entry,
    apply_funding_if_needed,
)

CHECK_INTERVAL = 10


def main() -> None:
    exchange = get_public_exchange()
    state = init_state()

    log(f"시작 잔고={INITIAL_BALANCE:.2f}")

    last_candle_time = None
    limit = max(RSI_PERIOD, MA_LONGEST_PERIOD, REGIME_LOOKBACK_15M * 3) + 100

    try:
        while True:
            df = fetch_ohlcv(exchange, limit=limit)

            df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
            df["ma_short"] = calculate_ma(df["close"], MA_SHORT_PERIOD)
            df["ma_long"] = calculate_ma(df["close"], MA_LONG_PERIOD)
            df["ma_50"] = calculate_ma(df["close"], MA_MID_PERIOD)
            df["ma_100"] = calculate_ma(df["close"], MA_LONGEST_PERIOD)
            df["volume_ma"] = df["volume"].rolling(window=20).mean()

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

            # 30초 단위: 포지션 보유 시 00/08/16 UTC 펀딩비 적용
            if state.has_long_position or state.has_short_position:
                now_utc = datetime.now(timezone.utc)
                if apply_funding_if_needed(state, exchange, now_utc, current_price):
                    time.sleep(CHECK_INTERVAL)
                    continue
            # 30초 단위: 포지션 보유 시 현재가로 익절/손절 체크 (모의 청산)
            if state.has_long_position or state.has_short_position:
                candle_for_close = pd.Series({**latest.to_dict(), "close": current_price})
                if check_scalp_stop_loss_and_profit(state, current_price, candle_for_close, df):
                    time.sleep(CHECK_INTERVAL)
                    continue
            else:
                # 30초 단위: 진입 조건 만족 시 현재가로 모의 진입(실제 주문 없음)
                if try_paper_entry(state, df, current_price):
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
            elif latest_time > last_candle_time:
                did_close = apply_strategy_on_candle(state, latest, df)
                last_candle_time = latest_time
                # 진입/청산 시에는 해당 로그만 남기고 5분 로그는 생략
                if did_close:
                    time.sleep(CHECK_INTERVAL)
                    continue

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
                rsi = float(latest["rsi"])
                log(f"[5m] {pos_status}{regime_str}{box_str} 가격={price:.2f} RSI={rsi:.0f} 잔고={state.balance:.2f} PNL={total_pnl:+.2f}{extra}")

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
