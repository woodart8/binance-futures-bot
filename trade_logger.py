"""매매·펀딩 기록 CSV 로깅.

- trades_log.csv: 진입가, 청산가, 손익, 잔고(매매 손익 반영).
- funding_log.csv: 00/08/16 UTC 펀딩 적용 시 구간, 레이트, 펀딩 손익, 잔고(페이퍼 전용).
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

LOG_FILE = Path("trades_log.csv")
HEADERS = ("time_utc", "side", "entry_price", "exit_price", "pnl", "balance_after", "meta")
FUNDING_LOG_FILE = Path("funding_log.csv")
FUNDING_HEADERS = ("time_utc", "funding_utc_key", "side", "position_value", "funding_rate", "funding_pnl", "balance_after", "meta")


def log_trade(
    side: str,
    entry_price: float,
    exit_price: float,
    pnl: float,
    balance_after: float,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """매매 기록을 CSV에 저장. meta는 JSON 문자열로 저장."""
    row = [
        datetime.now(timezone.utc).isoformat(),
        side,
        round(entry_price, 6),
        round(exit_price, 6),
        round(pnl, 6),
        round(balance_after, 6),
        json.dumps(meta or {}, ensure_ascii=False),
    ]
    is_new = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(HEADERS)
        w.writerow(row)


def log_funding(
    funding_utc_key: str,
    side: str,
    position_value: float,
    funding_rate: float,
    funding_pnl: float,
    balance_after: float,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """펀딩비 발생 시 기록 (funding_log.csv). meta는 JSON 문자열로 저장."""
    row = [
        datetime.now(timezone.utc).isoformat(),
        funding_utc_key,
        side,
        round(position_value, 6),
        round(funding_rate, 8),
        round(funding_pnl, 6),
        round(balance_after, 6),
        json.dumps(meta or {}, ensure_ascii=False),
    ]
    is_new = not FUNDING_LOG_FILE.exists()
    with FUNDING_LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(FUNDING_HEADERS)
        w.writerow(row)
