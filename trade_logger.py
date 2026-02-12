"""
매매 기록 로깅 유틸리티.

처음에는 CSV 파일(`trades_log.csv`)에 기록하고,
나중에 필요하면 DB로 교체할 수 있다.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


LOG_FILE = Path("trades_log.csv")


def log_trade(
    side: str,
    entry_price: float,
    exit_price: float,
    pnl: float,
    balance_after: float,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    단일 트레이드(진입~청산)를 CSV에 기록한다.
    """
    is_new = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(
                [
                    "time_utc",
                    "side",
                    "entry_price",
                    "exit_price",
                    "pnl",
                    "balance_after",
                    "meta",
                ]
            )
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                side,
                round(entry_price, 6),
                round(exit_price, 6),
                round(pnl, 6),
                round(balance_after, 6),
                meta or {},
            ]
        )

