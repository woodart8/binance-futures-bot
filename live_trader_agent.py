"""
실거래 에이전트.

- 5분마다 바이낸스 선물 USDS-M 5분봉을 조회하고,
- RSI_SWING 전략에 따라 진입/청산 신호가 나면 실제 주문을 전송한다.
- 각 트레이드는 `trades_log.csv` 에 기록된다.

주의:
- 이 파일은 실제 주문을 발생시킨다.
- .env 에 API_KEY, SECRET_KEY 가 올바르게 설정되어 있어야 한다.
"""

import time
from datetime import datetime

import pandas as pd

from backtest_core import BacktestResult  # 타입 재사용용
from config import (
    LEVERAGE,
    POSITION_SIZE_PERCENT,
    RSI_ENTRY,
    RSI_EXIT,
    SYMBOL,
    TIMEFRAME,
)
from exchange_client import get_private_exchange
from indicators import calculate_rsi
from strategy_core import RsiSwingParams, rsi_swing_signal
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
    log(f"레버리지 {LEVERAGE}배 설정 완료")


def main() -> None:
    log("실거래 에이전트 시작 (RSI_SWING)")
    exchange = get_private_exchange()
    set_leverage(exchange)

    params = RsiSwingParams(rsi_entry=RSI_ENTRY, rsi_exit=RSI_EXIT)

    has_position = False
    entry_price = 0.0
    entry_balance = 0.0

    last_candle_time = None

    try:
        while True:
            df = fetch_ohlcv(exchange, limit=200)
            df["rsi"] = calculate_rsi(df["close"])
            df = df.dropna().reset_index(drop=True)

            latest = df.iloc[-1]
            latest_time = latest["timestamp"]
            rsi_value = float(latest["rsi"])
            price = float(latest["close"])

            if last_candle_time is None:
                last_candle_time = latest_time
                log(
                    f"초기화 완료. 마지막 캔들 시간={latest_time}, "
                    f"가격={price:.2f}, RSI={rsi_value:.2f}"
                )
            elif latest_time > last_candle_time:
                # 새로운 5분봉 확정
                log(
                    f"새 5분봉 확정: {latest_time}, 가격={price:.2f}, RSI={rsi_value:.2f}"
                )

                # 전략 시그널
                signal = rsi_swing_signal(rsi_value, has_position, params)
                log(f"전략 시그널: {signal}")

                # 잔고 조회
                balance = get_balance_usdt(exchange)

                if has_position and signal == "flat":
                    # 포지션 종료 (마켓 SELL)
                    positions = exchange.fetch_positions([SYMBOL])
                    contracts = 0.0
                    for pos in positions:
                        if pos.get("symbol") == SYMBOL and pos.get("side") == "long":
                            contracts = float(pos.get("contracts", 0) or 0)
                            break

                    if contracts > 0:
                        order = exchange.create_market_sell_order(
                            SYMBOL, contracts, {"reduceOnly": True}
                        )
                        log(f"LONG 포지션 종료: {order}")
                    else:
                        log("종료할 LONG 포지션이 없음", "WARNING")

                    # 손익 계산 (단순 잔고 차이 기반)
                    new_balance = get_balance_usdt(exchange)
                    pnl = new_balance - entry_balance
                    log_trade(
                        side="LONG",
                        entry_price=entry_price,
                        exit_price=price,
                        pnl=pnl,
                        balance_after=new_balance,
                        meta={"timeframe": TIMEFRAME, "symbol": SYMBOL},
                    )
                    log(f"실현손익={pnl:.4f} USDT, 잔고={new_balance:.4f} USDT")
                    has_position = False
                    entry_price = 0.0
                    entry_balance = new_balance

                elif (not has_position) and signal == "long":
                    # 진입 (마켓 BUY)
                    order_usdt = balance * POSITION_SIZE_PERCENT
                    if order_usdt < 5.0:
                        log("주문 금액이 최소 주문 금액보다 작아 진입 생략", "WARNING")
                    else:
                        ticker = exchange.fetch_ticker(SYMBOL)
                        mkt_price = float(ticker["last"])
                        amount = order_usdt / mkt_price
                        order = exchange.create_market_buy_order(SYMBOL, amount)
                        log(f"LONG 진입: {order}")
                        has_position = True
                        entry_price = mkt_price
                        entry_balance = balance

                last_candle_time = latest_time

            time.sleep(60)

    except KeyboardInterrupt:
        log("사용자에 의해 종료됨", "WARNING")


if __name__ == "__main__":
    main()

