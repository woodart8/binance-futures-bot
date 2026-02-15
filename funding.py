"""
바이낸스 선물 펀딩비(펀딩 레이트) 조회.

펀딩 정산 시각: 00:00, 08:00, 16:00 UTC (8시간 간격).
- 레이트 > 0: 롱이 숏에게 지불. 레이트 < 0: 숏이 롱에게 지불.
- 펀딩금액 = 포지션 명목가치(USDT) × 펀딩 레이트.
"""

import time
from datetime import datetime, timezone
from typing import Optional, List, Tuple

import pandas as pd
import requests
import ccxt.base.errors


# Binance USDT-M 펀딩 정산 시간(UTC 기준 시각)
FUNDING_HOURS_UTC = (0, 8, 16)
FETCH_RETRIES = 3
FETCH_RETRY_DELAY = 2


def fetch_current_funding_rate(exchange, symbol: str) -> float:
    """
    현재(다음 정산에 적용될) 펀딩 레이트를 조회.
    :return: 레이트 (소수, 예: 0.0001 = 0.01%). 실패 시 0.0 반환.
    """
    last_error = None
    for attempt in range(FETCH_RETRIES):
        try:
            data = exchange.fetch_funding_rate(symbol)
            rate = float(data.get("fundingRate", 0.0))
            return rate
        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectTimeout,
            ccxt.base.errors.RequestTimeout,
            ccxt.base.errors.NetworkError,
            Exception,
        ) as e:
            last_error = e
            if attempt < FETCH_RETRIES - 1:
                time.sleep(FETCH_RETRY_DELAY)
    if last_error:
        raise last_error
    return 0.0


def fetch_funding_history(
    exchange,
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
) -> List[Tuple[int, float]]:
    """
    과거 펀딩 레이트 히스토리 조회 (백테스트용).
    :return: [(funding_timestamp_ms, rate), ...] 오름차순.
    """
    result: List[Tuple[int, float]] = []
    current_since = start_ms
    while current_since <= end_ms:
        try:
            history = exchange.fetch_funding_history(
                symbol,
                since=current_since,
                limit=limit,
                params={"endTime": end_ms},
            )
        except Exception:
            break
        if not history:
            break
        for h in history:
            ts = int(h.get("fundingTimestamp", h.get("timestamp", 0)))
            if ts < start_ms or ts > end_ms:
                continue
            rate = float(h.get("fundingRate", 0.0))
            result.append((ts, rate))
        if len(history) < limit:
            break
        last_ts = max(r[0] for r in result)
        current_since = last_ts + 1
        if current_since > end_ms:
            break
        time.sleep(0.2)
    result.sort(key=lambda x: x[0])
    return result


def get_funding_utc_key(dt: datetime) -> Optional[str]:
    """
    해당 시각이 펀딩 정산 시각(00/08/16 UTC)이면 해당 구간 키 반환, 아니면 None.
    키 형식: "YYYY-MM-DD-HH" (UTC). 같은 구간에서는 한 번만 적용하려면 last_funding_utc_key와 비교.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    utc = dt.astimezone(timezone.utc)
    if utc.hour in FUNDING_HOURS_UTC:
        return f"{utc.year:04d}-{utc.month:02d}-{utc.day:02d}-{utc.hour:02d}"
    return None


def is_funding_candle_utc(ts) -> bool:
    """
    봉 시각이 펀딩 정산 구간(00:00, 08:00, 16:00 UTC)에 해당하는지.
    pandas Timestamp 또는 datetime.
    """
    if hasattr(ts, "tz_localize"):
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
    else:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
    return ts.hour in FUNDING_HOURS_UTC and ts.minute < 15


def compute_funding_pnl(
    position_value_usdt: float,
    funding_rate: float,
    is_long: bool,
) -> float:
    """
    펀딩으로 인한 손익(USDT). 잔고에 더할 값 (받으면 +, 내면 -).
    - 레이트 > 0: 롱이 지불 → 롱은 음수.
    - 레이트 < 0: 숏이 지불 → 롱은 양수(받음).
    """
    # amount = position_value * rate; long pays when rate > 0 → pnl = - amount for long.
    amount = position_value_usdt * funding_rate
    if is_long:
        return -amount
    return amount
