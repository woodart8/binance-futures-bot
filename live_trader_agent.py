"""
실거래 에이전트.

- 5분마다 바이낸스 선물 USDS-M 5분봉을 조회하고,
- backtest.py와 동일한 swing_strategy_signal 전략에 따라 진입/청산 신호가 나면 실제 주문을 전송한다.
- 각 트레이드는 `trades_log.csv` 에 기록된다.

주의:
- 이 파일은 실제 주문을 발생시킨다.
- .env 에 API_KEY, SECRET_KEY 가 올바르게 설정되어 있어야 한다.
"""

import time
from datetime import datetime
from typing import Optional

import pandas as pd

from config import (
    DAILY_LOSS_LIMIT_PCT,
    LEVERAGE,
    POSITION_SIZE_PERCENT,
    RSI_PERIOD,
    MA_SHORT_PERIOD,
    MA_LONG_PERIOD,
    MA_MID_PERIOD,
    MA_LONGEST_PERIOD,
    BULLISH_PROFIT_TARGET,
    BULLISH_STOP_LOSS,
    BULLISH_STOP_LOSS_PRICE,
    BULLISH_EARLY_EXIT_RSI,
    BULLISH_TRAILING_STOP_ACTIVATION,
    BULLISH_TRAILING_STOP_PCT,
    BEARISH_PROFIT_TARGET,
    BEARISH_STOP_LOSS,
    BEARISH_STOP_LOSS_PRICE,
    BEARISH_EARLY_EXIT_RSI,
    BEARISH_TRAILING_STOP_ACTIVATION,
    BEARISH_TRAILING_STOP_PCT,
    SIDEWAYS_BOX_PERIOD,
    SIDEWAYS_BOX_TOP_MARGIN,
    SIDEWAYS_BOX_BOTTOM_MARGIN,
    SIDEWAYS_STOP_LOSS,
    SYMBOL,
    TIMEFRAME,
)
from exchange_client import get_private_exchange
from indicators import calculate_rsi, calculate_ma
from chart_patterns import detect_chart_pattern, PATTERN_LOOKBACK
from strategy_core import (
    MovingAverageParams,
    swing_strategy_signal,
    detect_market_regime,
)
from trade_logger import log_trade


def log(message: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {message}")


def fetch_ohlcv(exchange, limit: int = 200) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def get_balance_usdt(exchange) -> float:
    balance = exchange.fetch_balance()
    usdt_info = balance.get("USDT", {})
    free = float(usdt_info.get("free", 0) or 0)
    return free


def set_leverage(exchange) -> None:
    market = exchange.market(SYMBOL)
    exchange.set_leverage(LEVERAGE, market["id"])


def main() -> None:
    exchange = get_private_exchange()
    set_leverage(exchange)

    has_position = False
    is_long = False
    entry_price = 0.0
    entry_balance = 0.0
    entry_regime = ""
    box_high_entry = 0.0
    box_low_entry = 0.0
    highest_price = 0.0
    lowest_price = float("inf")
    partial_profit_taken = False
    trailing_stop_active = False
    best_pnl_pct = 0.0
    pattern_type = ""
    pattern_target = 0.0
    pattern_stop = 0.0

    # 일일 손실 한도 추적
    daily_start_balance = 0.0
    daily_start_date = ""

    last_candle_time = None
    ma_params = MovingAverageParams(
        short_period=MA_SHORT_PERIOD,
        long_period=MA_LONG_PERIOD,
        trend_threshold=0.005,
    )

    try:
        while True:
            try:
                # 충분한 데이터 확보
                limit = max(RSI_PERIOD, MA_LONGEST_PERIOD) + 100
                df = fetch_ohlcv(exchange, limit=limit)
            except Exception as e:
                log(f"OHLCV 조회 실패: {e}", "ERROR")
                time.sleep(60)
                continue
            
            # 지표 계산
            df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
            df["ma_short"] = calculate_ma(df["close"], MA_SHORT_PERIOD)
            df["ma_long"] = calculate_ma(df["close"], MA_LONG_PERIOD)
            df["ma_50"] = calculate_ma(df["close"], MA_MID_PERIOD)
            df["ma_100"] = calculate_ma(df["close"], MA_LONGEST_PERIOD)
            
            df = df.dropna().reset_index(drop=True)

            latest = df.iloc[-1]
            latest_time = latest["timestamp"]
            price = float(latest["close"])
            rsi = float(latest["rsi"])
            short_ma = float(latest["ma_short"])
            long_ma = float(latest["ma_long"])
            ma_50 = float(latest["ma_50"])
            ma_100 = float(latest["ma_100"])

            if last_candle_time is None:
                last_candle_time = latest_time
                try:
                    bal = get_balance_usdt(exchange)
                except Exception:
                    bal = 0.0
                log(f"[시작] 가격={price:.2f} 잔고={bal:.2f}")
            elif latest_time > last_candle_time:

                # regime: 5분봉 기반 판단
                price_history = df["close"].tail(SIDEWAYS_BOX_PERIOD + 1).tolist() if len(df) >= SIDEWAYS_BOX_PERIOD else df["close"].tolist()
                regime = detect_market_regime(short_ma, long_ma, price, ma_50, ma_100, ma_params, price_history)

                # 손절/수익 실현 체크 (backtest.py와 동일한 로직)
                signal = None
                if has_position:
                    pnl_pct = (price - entry_price) / entry_price * LEVERAGE * 100 if is_long else (entry_price - price) / entry_price * LEVERAGE * 100
                    
                    # 차트 패턴 진입 시 패턴 정석 익절/손절만 적용 (config 퍼센트 무시)
                    if pattern_target > 0 and pattern_stop > 0:
                        if is_long:
                            if price >= pattern_target:
                                signal = "flat"
                            elif price <= pattern_stop:
                                signal = "flat"
                        else:  # SHORT
                            if price <= pattern_target:
                                signal = "flat"
                            elif price >= pattern_stop:
                                signal = "flat"
                        # 패턴 대기 중이면 signal 그대로, regime 기반 로직 스킵
                    
                    if signal is None and not (pattern_target > 0 and pattern_stop > 0) and is_long:
                        if price > highest_price:
                            highest_price = price
                        
                        # backtest.py와 동일한 청산 로직
                        if entry_regime == "bullish":
                            # RSI 기반 조기 청산
                            if rsi <= BULLISH_EARLY_EXIT_RSI and pnl_pct < 0:
                                signal = "flat"
                            # 트레일링 스톱
                            elif pnl_pct > best_pnl_pct:
                                best_pnl_pct = pnl_pct
                            elif pnl_pct >= BULLISH_TRAILING_STOP_ACTIVATION:
                                trailing_stop_active = True
                                if trailing_stop_active and best_pnl_pct - pnl_pct >= BULLISH_TRAILING_STOP_PCT:
                                    signal = "flat"
                            # 익절/손절
                            elif pnl_pct >= BULLISH_PROFIT_TARGET:
                                signal = "flat"
                            elif pnl_pct <= -BULLISH_STOP_LOSS:
                                signal = "flat"
                            else:
                                stop_loss_price = entry_price * (1 - BULLISH_STOP_LOSS_PRICE / 100 / LEVERAGE)
                                if price <= stop_loss_price:
                                    signal = "flat"
                        elif entry_regime == "sideways":
                            # 횡보장 롱: 고점 근처 익절, 하단 이탈 손절, 기본 손절 2%
                            if box_high_entry > 0 and box_low_entry > 0:
                                box_high_threshold = box_high_entry * (1 - SIDEWAYS_BOX_TOP_MARGIN)
                                if price >= box_high_threshold:
                                    signal = "flat"
                                else:
                                    box_low_threshold = box_low_entry * (1 - 0.01)
                                    if price < box_low_threshold:
                                        signal = "flat"
                            if signal is None and pnl_pct <= -SIDEWAYS_STOP_LOSS:
                                signal = "flat"
                    elif signal is None and not (pattern_target > 0 and pattern_stop > 0) and not is_long:  # SHORT
                        if price < lowest_price:
                            lowest_price = price
                        
                        # backtest.py와 동일한 청산 로직
                        if entry_regime == "bearish":
                            # RSI 기반 조기 청산
                            if rsi >= BEARISH_EARLY_EXIT_RSI and pnl_pct < 0:
                                signal = "flat"
                            # 트레일링 스톱
                            elif pnl_pct > best_pnl_pct:
                                best_pnl_pct = pnl_pct
                            elif pnl_pct >= BEARISH_TRAILING_STOP_ACTIVATION:
                                trailing_stop_active = True
                                if trailing_stop_active and best_pnl_pct - pnl_pct >= BEARISH_TRAILING_STOP_PCT:
                                    signal = "flat"
                            # 익절/손절
                            elif pnl_pct >= BEARISH_PROFIT_TARGET:
                                signal = "flat"
                            elif pnl_pct <= -BEARISH_STOP_LOSS:
                                signal = "flat"
                            else:
                                stop_loss_price = entry_price * (1 + BEARISH_STOP_LOSS_PRICE / 100 / LEVERAGE)
                                if price >= stop_loss_price:
                                    signal = "flat"
                        elif entry_regime == "sideways":
                            # 횡보장 숏: 저점 근처 익절, 상단 이탈 손절, 기본 손절 2%
                            if box_high_entry > 0 and box_low_entry > 0:
                                box_low_threshold = box_low_entry * (1 + SIDEWAYS_BOX_BOTTOM_MARGIN)
                                if price <= box_low_threshold:
                                    signal = "flat"
                                else:
                                    box_high_threshold = box_high_entry * (1 + 0.01)
                                    if price > box_high_threshold:
                                        signal = "flat"
                            if signal is None and pnl_pct <= -SIDEWAYS_STOP_LOSS:
                                signal = "flat"
                else:
                    # 진입 신호 생성 (backtest.py와 동일)
                    signal = swing_strategy_signal(
                        rsi_value=rsi,
                        price=price,
                        short_ma=short_ma,
                        long_ma=long_ma,
                        has_position=has_position,
                        is_long=is_long,
                        regime=regime,
                        price_history=price_history,
                    )


                # 잔고 조회
                try:
                    balance = get_balance_usdt(exchange)
                except Exception as e:
                    log(f"잔고 조회 실패: {e}", "ERROR")
                    time.sleep(60)
                    continue

                # 일일 손실 한도: 날짜 변경 시 daily_start_balance 갱신
                current_date = latest_time.strftime("%Y-%m-%d") if hasattr(latest_time, "strftime") else str(latest_time)[:10]
                if daily_start_date != current_date:
                    daily_start_balance = balance
                    daily_start_date = current_date

                # 일일 손실 한도: 초과 시 진입 불가
                daily_loss_pct = 0.0
                if daily_start_balance > 0:
                    daily_loss_pct = (daily_start_balance - balance) / daily_start_balance * 100
                daily_limit_hit = daily_loss_pct >= DAILY_LOSS_LIMIT_PCT

                if has_position and signal == "flat":
                    # 포지션 종료
                    try:
                        positions = exchange.fetch_positions([SYMBOL])
                    except Exception as e:
                        log(f"포지션 조회 실패: {e}", "ERROR")
                        time.sleep(60)
                        continue
                    contracts = 0.0
                    side_to_close = "long" if is_long else "short"
                    for pos in positions:
                        if pos.get("symbol") == SYMBOL and pos.get("side") == side_to_close:
                            contracts = float(pos.get("contracts", 0) or 0)
                            break

                    if contracts > 0:
                        try:
                            if is_long:
                                order = exchange.create_market_sell_order(
                                    SYMBOL, contracts, {"reduceOnly": True}
                                )
                            else:
                                order = exchange.create_market_buy_order(
                                    SYMBOL, contracts, {"reduceOnly": True}
                                )
                            # 주문 실행 검증
                            order_status = order.get("status", "") if order else ""
                            side_str = "LONG" if is_long else "SHORT"
                            log(f"{side_str} 청산 | 진입={entry_price:.2f} 청산={price:.2f}")
                        except Exception as e:
                            log(f"청산 주문 실패: {e}", "ERROR")
                            continue  # 상태 유지, 다음 루프에서 재시도

                    # 손익 계산
                    try:
                        new_balance = get_balance_usdt(exchange)
                    except Exception as e:
                        log(f"청산 후 잔고 조회 실패: {e}", "ERROR")
                        new_balance = balance
                    pnl = new_balance - entry_balance
                    log_trade(
                        side="LONG" if is_long else "SHORT",
                        entry_price=entry_price,
                        exit_price=price,
                        pnl=pnl,
                        balance_after=new_balance,
                        meta={"timeframe": TIMEFRAME, "symbol": SYMBOL, "regime": entry_regime},
                    )
                    has_position = False
                    is_long = False
                    entry_price = 0.0
                    entry_regime = ""
                    box_high_entry = 0.0
                    box_low_entry = 0.0
                    highest_price = 0.0
                    lowest_price = float("inf")
                    partial_profit_taken = False
                    trailing_stop_active = False
                    best_pnl_pct = 0.0
                    pattern_type = ""
                    pattern_target = 0.0
                    pattern_stop = 0.0
                    entry_balance = new_balance

                elif (not has_position) and (signal == "long" or signal == "short"):
                    if daily_limit_hit:
                        log(f"[진입생략] 일일손실한도 {daily_loss_pct:.1f}%")
                    else:
                        # 진입
                        order_usdt = balance * POSITION_SIZE_PERCENT
                        if order_usdt >= 5.0:
                            try:
                                ticker = exchange.fetch_ticker(SYMBOL)
                                mkt_price = float(ticker["last"])
                                amount = order_usdt / mkt_price
                                
                                # 차트 패턴 감지 (같은 방향 시그널일 때만)
                                pattern_info = None
                                if len(df) >= PATTERN_LOOKBACK:
                                    highs = df["high"].tolist()
                                    lows = df["low"].tolist()
                                    closes = df["close"].tolist()
                                    pattern_info = detect_chart_pattern(highs, lows, closes, mkt_price)
                                
                                if signal == "long":
                                    order = exchange.create_market_buy_order(SYMBOL, amount)
                                    pt, ptg, pst = ("", 0.0, 0.0) if not (pattern_info and pattern_info.side == "LONG") else (pattern_info.name, pattern_info.target_price, pattern_info.stop_price)
                                    log(f"LONG 진입 | 가격={mkt_price:.2f} 잔고={balance:.2f}" + (f" 패턴={pt}" if pt else ""))
                                    has_position, is_long = True, True
                                    entry_price, highest_price = mkt_price, mkt_price
                                    pattern_type, pattern_target, pattern_stop = pt, ptg, pst
                                else:
                                    order = exchange.create_market_sell_order(SYMBOL, amount)
                                    pt, ptg, pst = ("", 0.0, 0.0) if not (pattern_info and pattern_info.side == "SHORT") else (pattern_info.name, pattern_info.target_price, pattern_info.stop_price)
                                    log(f"SHORT 진입 | 가격={mkt_price:.2f} 잔고={balance:.2f}" + (f" 패턴={pt}" if pt else ""))
                                    has_position, is_long = True, False
                                    entry_price, lowest_price = mkt_price, mkt_price
                                    pattern_type, pattern_target, pattern_stop = pt, ptg, pst
                                
                                entry_regime = regime
                                entry_balance = balance
                                partial_profit_taken = False
                                trailing_stop_active = False
                                best_pnl_pct = 0.0
                                # 횡보장 진입 시 박스권 저장
                                box_high_entry = 0.0
                                box_low_entry = 0.0
                                if regime == "sideways" and len(price_history) >= SIDEWAYS_BOX_PERIOD:
                                    box_high_entry = max(price_history[-SIDEWAYS_BOX_PERIOD:])
                                    box_low_entry = min(price_history[-SIDEWAYS_BOX_PERIOD:])
                            except Exception as e:
                                log(f"진입 주문 실패: {e}", "ERROR")

                # 5분 간격 상태 로그
                pos_status = "LONG" if has_position and is_long else ("SHORT" if has_position else "NONE")
                try:
                    bal = get_balance_usdt(exchange) if has_position else balance
                except Exception:
                    bal = balance
                unrealized = ""
                if has_position:
                    pnl_pct = (price - entry_price) / entry_price * LEVERAGE * 100 if is_long else (entry_price - price) / entry_price * LEVERAGE * 100
                    unrealized = f" 미실현={pnl_pct:+.2f}%"
                log(f"[5분] {pos_status} | 가격={price:.2f} 잔고={bal:.2f}{unrealized}")

                last_candle_time = latest_time

            time.sleep(300)  # 5분봉이므로 5분(300초) 대기

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
