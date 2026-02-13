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
    BEARISH_PROFIT_TARGET,
    BEARISH_STOP_LOSS,
    BEARISH_STOP_LOSS_PRICE,
    BEARISH_EARLY_EXIT_RSI,
    SIDEWAYS_BOX_PERIOD,
    SYMBOL,
    TIMEFRAME,
)
from exchange_client import get_private_exchange
from indicators import calculate_rsi, calculate_ma
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
    highest_price = 0.0
    lowest_price = float("inf")
    partial_profit_taken = False
    trailing_stop_active = False
    best_pnl_pct = 0.0

    last_candle_time = None
    ma_params = MovingAverageParams(
        short_period=MA_SHORT_PERIOD,
        long_period=MA_LONG_PERIOD,
        trend_threshold=0.005,
    )

    try:
        while True:
            # 충분한 데이터 확보
            limit = max(RSI_PERIOD, MA_LONGEST_PERIOD) + 100
            df = fetch_ohlcv(exchange, limit=limit)
            
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
            elif latest_time > last_candle_time:

                # 시장 상태 판단
                price_history = df["close"].tail(SIDEWAYS_BOX_PERIOD + 1).tolist() if len(df) >= SIDEWAYS_BOX_PERIOD else df["close"].tolist()
                regime = detect_market_regime(short_ma, long_ma, price, ma_50, ma_100, ma_params, price_history)

                # 손절/수익 실현 체크 (backtest.py와 동일한 로직)
                signal = None
                if has_position:
                    pnl_pct = (price - entry_price) / entry_price * LEVERAGE * 100 if is_long else (entry_price - price) / entry_price * LEVERAGE * 100
                    
                    if is_long:
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
                            elif pnl_pct >= 8.0:
                                trailing_stop_active = True
                                if trailing_stop_active and best_pnl_pct - pnl_pct >= 4.0:
                                    signal = "flat"
                            # 부분 익절 (실거래에서는 부분 청산 복잡하므로 생략)
                            # 익절/손절
                            elif pnl_pct >= BULLISH_PROFIT_TARGET:
                                signal = "flat"
                            elif pnl_pct <= -BULLISH_STOP_LOSS:
                                signal = "flat"
                            else:
                                stop_loss_price = entry_price * (1 - BULLISH_STOP_LOSS_PRICE / 100 / LEVERAGE)
                                if price <= stop_loss_price:
                                    signal = "flat"
                    else:  # SHORT
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
                            elif pnl_pct >= 8.0:
                                trailing_stop_active = True
                                if trailing_stop_active and best_pnl_pct - pnl_pct >= 4.0:
                                    signal = "flat"
                            # 부분 익절 (실거래에서는 부분 청산 복잡하므로 생략)
                            # 익절/손절
                            elif pnl_pct >= BEARISH_PROFIT_TARGET:
                                signal = "flat"
                            elif pnl_pct <= -BEARISH_STOP_LOSS:
                                signal = "flat"
                            else:
                                stop_loss_price = entry_price * (1 + BEARISH_STOP_LOSS_PRICE / 100 / LEVERAGE)
                                if price >= stop_loss_price:
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
                balance = get_balance_usdt(exchange)

                if has_position and signal == "flat":
                    # 포지션 종료
                    positions = exchange.fetch_positions([SYMBOL])
                    contracts = 0.0
                    side_to_close = "long" if is_long else "short"
                    for pos in positions:
                        if pos.get("symbol") == SYMBOL and pos.get("side") == side_to_close:
                            contracts = float(pos.get("contracts", 0) or 0)
                            break

                    if contracts > 0:
                        if is_long:
                            order = exchange.create_market_sell_order(
                                SYMBOL, contracts, {"reduceOnly": True}
                            )
                        else:
                            order = exchange.create_market_buy_order(
                                SYMBOL, contracts, {"reduceOnly": True}
                            )
                            log(f"{'LONG' if is_long else 'SHORT'} 종료")

                    # 손익 계산
                    new_balance = get_balance_usdt(exchange)
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
                    highest_price = 0.0
                    lowest_price = float("inf")
                    partial_profit_taken = False
                    trailing_stop_active = False
                    best_pnl_pct = 0.0
                    entry_balance = new_balance

                elif (not has_position) and (signal == "long" or signal == "short"):
                    # 진입
                    order_usdt = balance * POSITION_SIZE_PERCENT
                    if order_usdt >= 5.0:
                        ticker = exchange.fetch_ticker(SYMBOL)
                        mkt_price = float(ticker["last"])
                        amount = order_usdt / mkt_price
                        
                        if signal == "long":
                            order = exchange.create_market_buy_order(SYMBOL, amount)
                            log("LONG 진입")
                            has_position = True
                            is_long = True
                            entry_price = mkt_price
                            highest_price = mkt_price
                        else:  # short
                            order = exchange.create_market_sell_order(SYMBOL, amount)
                            log("SHORT 진입")
                            has_position = True
                            is_long = False
                            entry_price = mkt_price
                            lowest_price = mkt_price
                        
                        entry_regime = regime
                        entry_balance = balance
                        partial_profit_taken = False
                        trailing_stop_active = False
                        best_pnl_pct = 0.0

                last_candle_time = latest_time

            time.sleep(300)  # 5분봉이므로 5분(300초) 대기

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
