"""매매 기록 CSV 로깅."""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

LOG_FILE = Path("trades_log.csv")
HEADERS = ("time_utc", "side", "entry_price", "exit_price", "pnl", "balance_after", "meta")


def log_trade(
    side: str,
    entry_price: float,
    exit_price: float,
    pnl: float,
    balance_after: float,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    row = [
        datetime.now(timezone.utc).isoformat(),
        side,
        round(entry_price, 6),
        round(exit_price, 6),
        round(pnl, 6),
        round(balance_after, 6),
        meta or {},
    ]
    is_new = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(HEADERS)
        w.writerow(row)
