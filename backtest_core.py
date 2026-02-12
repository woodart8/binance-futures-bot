"""
공통 백테스트 엔진.

여러 전략/파라미터를 테스트할 때 재사용한다.
"""

from dataclasses import dataclass
from typing import Callable, List

import pandas as pd

from config import FEE_RATE, INITIAL_BALANCE, LEVERAGE, POSITION_SIZE_PERCENT


@dataclass
class BacktestResult:
    pnl: float
    pnl_pct: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    equity_curve: List[float]


def run_long_only_backtest(
    df: pd.DataFrame,
    signal_fn: Callable[[pd.Series, bool], str],
) -> BacktestResult:
    """
    롱 전용 전략 백테스트.

    signal_fn(row, has_position) -> "long" | "flat" | "hold"
    """
    balance = INITIAL_BALANCE
    equity = INITIAL_BALANCE
    equity_curve: List[float] = []
    has_position = False
    entry_price = 0.0

    wins = 0
    losses = 0
    peak_equity = equity
    max_dd = 0.0

    for _, row in df.iterrows():
        price = float(row["close"])
        signal = signal_fn(row, has_position)

        # 청산
        if has_position and signal == "flat":
            pnl_pct = (price - entry_price) / entry_price * LEVERAGE
            gross_pnl = pnl_pct * balance * POSITION_SIZE_PERCENT
            fee = balance * POSITION_SIZE_PERCENT * FEE_RATE
            net_pnl = gross_pnl - fee
            balance += net_pnl
            has_position = False

            if net_pnl > 0:
                wins += 1
            else:
                losses += 1

        # 진입
        elif (not has_position) and signal == "long":
            trade_capital = balance * POSITION_SIZE_PERCENT
            if trade_capital > 0:
                fee = trade_capital * FEE_RATE
                trade_capital_after_fee = trade_capital - fee
                entry_price = price
                has_position = True

        # 평가손익
        if has_position:
            unrealized = (
                (price - entry_price)
                / entry_price
                * LEVERAGE
                * balance
                * POSITION_SIZE_PERCENT
            )
            equity = balance + unrealized
        else:
            equity = balance

        equity_curve.append(equity)

        if equity > peak_equity:
            peak_equity = equity
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        max_dd = max(max_dd, drawdown)

    num_trades = wins + losses
    final_pnl = equity - INITIAL_BALANCE
    final_pnl_pct = final_pnl / INITIAL_BALANCE * 100 if INITIAL_BALANCE > 0 else 0.0
    win_rate = wins / num_trades * 100 if num_trades > 0 else 0.0

    return BacktestResult(
        pnl=final_pnl,
        pnl_pct=final_pnl_pct,
        max_drawdown=max_dd * 100,
        win_rate=win_rate,
        num_trades=num_trades,
        equity_curve=equity_curve,
    )

