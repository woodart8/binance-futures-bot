"""
전략 리서치 에이전트.

- `trades_log.csv` 를 읽어 최근 성과를 분석하고,
- 간단한 통계를 출력해 사람이 전략을 튜닝하는 데 도움을 준다.

지금은 자동으로 파라미터를 바꾸지는 않고, 리포트만 제공한다.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List


LOG_FILE = Path("trades_log.csv")


@dataclass
class Trade:
    time_utc: datetime
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    balance_after: float


def load_trades() -> List[Trade]:
    if not LOG_FILE.exists():
        print("trades_log.csv 가 아직 없습니다. 실거래 에이전트가 트레이드를 남겨야 합니다.")
        return []

    trades: List[Trade] = []
    with LOG_FILE.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            trades.append(
                Trade(
                    time_utc=datetime.fromisoformat(row["time_utc"]),
                    side=row["side"],
                    entry_price=float(row["entry_price"]),
                    exit_price=float(row["exit_price"]),
                    pnl=float(row["pnl"]),
                    balance_after=float(row["balance_after"]),
                )
            )
    return trades


def summarize_trades(trades: List[Trade]) -> None:
    if not trades:
        print("트레이드 기록이 없습니다.")
        return

    total_pnl = sum(t.pnl for t in trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0.0

    balances = [t.balance_after for t in trades]
    peak = balances[0]
    max_dd = 0.0
    for b in balances:
        if b > peak:
            peak = b
        dd = (peak - b) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    print("=== 실거래 트레이드 요약 ===")
    print(f"총 트레이드 수: {len(trades)}")
    print(f"총 PnL: {total_pnl:.4f} USDT")
    print(f"승률: {win_rate:.2f}%")
    print(f"최대 낙폭(MDD): {max_dd * 100:.2f}%")
    print(f"마지막 잔고: {balances[-1]:.4f} USDT")


def main() -> None:
    trades = load_trades()
    summarize_trades(trades)


if __name__ == "__main__":
    main()

