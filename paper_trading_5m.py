"""
바이낸스 선물(USDS-M) 5분봉 RSI_SWING 모의선물(페이퍼 트레이딩) 봇

- 실제 주문은 전혀 보내지 않고, 가상의 USDT 잔고로만 손익을 계산한다.
- 실거래 봇과 동일하게:
  - 5분봉, BTC/USDT
  - RSI 14
  - RSI <= 30: LONG 진입
  - RSI >= 60: 포지션 전량 청산
  - 레버리지 5배, 트레이드당 잔고의 10% 사용

실행 예시:
    python paper_trading_5m.py
"""

import time
from dataclasses import dataclass
from datetime import datetime

import ccxt
import pandas as pd

from config import SYMBOL, TIMEFRAME
from trade_logger import log_trade


RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 60
INITIAL_BALANCE = 1000.0  # 모의 USDT 시작 잔고
RISK_PER_TRADE = 0.1  # 잔고의 10% 사용
FEE_RATE = 0.0004  # 왕복 수수료 0.04% 가정
LEVERAGE = 5  # 실거래와 동일하게 5배 레버리지 가정


def log(message: str, level: str = "INFO") -> None:
    """간단 로깅 함수."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {message}")


def get_public_exchange() -> ccxt.binanceusdm:
    """
    퍼블릭 데이터 전용 바이낸스 선물 인스턴스.
    (API 키 없이 OHLCV, 티커 조회만 사용)
    """
    return ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})


def fetch_ohlcv(exchange: ccxt.binanceusdm, limit: int = 300) -> pd.DataFrame:
    """5분봉 OHLCV 조회."""
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def calculate_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.Series:
    """RSI 14 계산."""
    close = df["close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("inf"))
    rsi = 100 - (100 / (1 + rs))
    return rsi


@dataclass
class PaperState:
    balance: float
    equity: float
    has_position: bool
    entry_price: float
    position_size: float
    peak_equity: float
    max_drawdown: float


def init_state() -> PaperState:
    balance = INITIAL_BALANCE
    return PaperState(
        balance=balance,
        equity=balance,
        has_position=False,
        entry_price=0.0,
        position_size=0.0,
        peak_equity=balance,
        max_drawdown=0.0,
    )


def apply_strategy_on_candle(state: PaperState, candle: pd.Series) -> None:
    """
    한 개의 종가(완성된 5분봉)에 대해 RSI_SWING 전략 적용.
    - RSI <= 30: 롱 진입
    - RSI >= 60: 롱 청산
    """
    price = float(candle["close"])
    rsi = float(candle["rsi"])

    # 청산 조건 먼저
    if state.has_position and rsi >= RSI_OVERBOUGHT:
        # 레버리지 반영된 수익률
        pnl_pct = (price - state.entry_price) / state.entry_price * LEVERAGE
        gross_pnl = pnl_pct * state.balance * RISK_PER_TRADE
        fee = state.balance * RISK_PER_TRADE * FEE_RATE
        net_pnl = gross_pnl - fee
        state.balance += net_pnl
        state.has_position = False
        state.position_size = 0.0
        state.entry_price = 0.0
        # 모의 트레이드도 실거래와 동일 포맷으로 기록
        log_trade(
            side="PAPER_LONG",
            entry_price=float(candle["open"]),  # 단순히 해당 봉 시가를 진입가로 사용
            exit_price=price,
            pnl=net_pnl,
            balance_after=state.balance,
            meta={"timeframe": TIMEFRAME, "symbol": SYMBOL, "mode": "paper"},
        )
        log(
            f"청산: 가격={price:.2f}, RSI={rsi:.2f}, 실현손익={net_pnl:.4f} USDT, "
            f"잔고={state.balance:.4f} USDT"
        )

    # 진입 조건
    elif (not state.has_position) and rsi <= RSI_OVERSOLD:
        trade_capital = state.balance * RISK_PER_TRADE
        if trade_capital <= 0:
            return
        fee = trade_capital * FEE_RATE
        trade_capital_after_fee = trade_capital - fee
        position_size = trade_capital_after_fee / price
        state.has_position = True
        state.entry_price = price
        state.position_size = position_size
        log(
            f"진입: 가격={price:.2f}, RSI={rsi:.2f}, "
            f"사용자금={trade_capital_after_fee:.4f} USDT, 수량={position_size:.6f} BTC"
        )

    # 평가손익 및 MDD 업데이트
    if state.has_position:
        unrealized = (
            (price - state.entry_price)
            / state.entry_price
            * LEVERAGE
            * state.balance
            * RISK_PER_TRADE
        )
        equity = state.balance + unrealized
    else:
        equity = state.balance

    state.equity = equity
    if equity > state.peak_equity:
        state.peak_equity = equity
    drawdown = (
        (state.peak_equity - equity) / state.peak_equity if state.peak_equity > 0 else 0.0
    )
    state.max_drawdown = max(state.max_drawdown, drawdown)


def main() -> None:
    log("5분봉 RSI_SWING 모의선물 봇 시작 (실제 주문 없음)")
    exchange = get_public_exchange()
    state = init_state()

    last_candle_time = None

    try:
        while True:
            df = fetch_ohlcv(exchange, limit=RSI_PERIOD + 100)
            df["rsi"] = calculate_rsi(df)
            df = df.dropna().reset_index(drop=True)

            latest = df.iloc[-1]
            latest_time = latest["timestamp"]

            # 아직 첫 루프라면, 마지막 캔들을 기준으로만 시작
            if last_candle_time is None:
                last_candle_time = latest_time
                log(
                    f"초기화 완료. 시작 잔고={state.balance:.2f} USDT, "
                    f"마지막 캔들 시간={latest_time}"
                )
            # 새로운 5분봉이 완성되었을 때만 전략 적용
            elif latest_time > last_candle_time:
                log(
                    f"새로운 5분봉 확인: {latest_time}, "
                    f"가격={latest['close']:.2f}, RSI={latest['rsi']:.2f}"
                )
                apply_strategy_on_candle(state, latest)
                last_candle_time = latest_time

                log(
                    f"현재 Equity={state.equity:.4f} USDT, "
                    f"잔고={state.balance:.4f} USDT, "
                    f"MDD={state.max_drawdown * 100:.2f}%, "
                    f"포지션 보유여부={state.has_position}"
                )

            # 60초마다 한 번씩 체크 (5분봉이므로 여유있게)
            time.sleep(60)

    except KeyboardInterrupt:
        log(
            f"사용자 종료. 최종 Equity={state.equity:.4f} USDT, "
            f"잔고={state.balance:.4f} USDT, "
            f"최대 낙폭={state.max_drawdown * 100:.2f}%"
        )


if __name__ == "__main__":
    main()

