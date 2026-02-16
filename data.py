"""OHLCV 조회 및 15분봉 regime 계산."""

import time
import pandas as pd
import requests
import ccxt.base.errors

from config import (
    SYMBOL,
    TIMEFRAME,
    REGIME_LOOKBACK_15M,
    MA_SHORT_PERIOD,
    MA_LONG_PERIOD,
    MA_MID_PERIOD,
    MA_LONGEST_PERIOD,
    RSI_PERIOD,
)
from indicators import calculate_ma, calculate_rsi
from strategy_core import detect_market_regime

# 타임아웃/일시 오류 시 재시도 횟수 및 대기(초)
FETCH_RETRIES = 3
FETCH_RETRY_DELAY = 2


def _fetch_ohlcv_with_retry(exchange, symbol: str, timeframe: str, limit: int, since: int = None):
    """fetch_ohlcv 호출을 타임아웃/네트워크 오류 시 최대 FETCH_RETRIES회 재시도.
    재시도가 모두 실패하면 예외를 그대로 raise (호출자가 처리)."""
    last_error = None
    kwargs = {"limit": limit}
    if since is not None:
        kwargs["since"] = since
    for attempt in range(FETCH_RETRIES):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe, **kwargs)
        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectTimeout,
            ccxt.base.errors.RequestTimeout,
            ccxt.base.errors.NetworkError,
        ) as e:
            last_error = e
            if attempt < FETCH_RETRIES - 1:
                time.sleep(FETCH_RETRY_DELAY)
            else:
                raise last_error
    raise last_error


def fetch_ohlcv(exchange, limit: int = 400, symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> pd.DataFrame:
    ohlcv = _fetch_ohlcv_with_retry(exchange, symbol, timeframe, limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def compute_regime_15m(df: pd.DataFrame, current_price: float) -> tuple:
    """
    15분봉 기준 장세 계산.
    
    MA100 계산을 위해 충분한 데이터(약 33시간)를 가져온 후, 최근 24시간(96개 15분봉)만 사용.
    
    Returns:
        (regime, short_ma_15m, long_ma_15m, ma_50_15m, ma_100_15m, price_history_15m, rsi_15m, ma_long_history)
    """
    df_tmp = df.copy()
    df_tmp["timestamp"] = pd.to_datetime(df_tmp["timestamp"])
    df_15m = df_tmp.set_index("timestamp").resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    # MA100 계산을 위해 충분한 데이터 확보 (최소 100개 + 여유분)
    # 그 중 최근 24시간(REGIME_LOOKBACK_15M)만 사용
    df_15m["ma_short"] = calculate_ma(df_15m["close"], MA_SHORT_PERIOD)
    df_15m["ma_long"] = calculate_ma(df_15m["close"], MA_LONG_PERIOD)
    df_15m["ma_50"] = calculate_ma(df_15m["close"], MA_MID_PERIOD)
    df_15m["ma_100"] = calculate_ma(df_15m["close"], MA_LONGEST_PERIOD)
    df_15m["rsi"] = calculate_rsi(df_15m["close"], RSI_PERIOD)
    # dropna()를 하면 MA100 계산 전의 행들이 제거되어 데이터가 부족해짐
    # MA100 계산을 위해서는 전체 데이터가 필요하므로 dropna() 제거
    df_15m = df_15m.reset_index()
    # MA100 계산을 위해 최소 100개 필요
    if len(df_15m) < MA_LONGEST_PERIOD:
        return ("neutral", 0.0, 0.0, 0.0, 0.0, [], None, [])
    # MA100이 계산된 마지막 행 확인
    if df_15m["ma_100"].isna().all():
        return ("neutral", 0.0, 0.0, 0.0, 0.0, [], None, [])
    # 최근 REGIME_LOOKBACK_15M개 사용 (MA100이 계산된 행만)
    # 전체 데이터에서 MA100 계산 후, 최근 96개만 사용
    df_15m_valid = df_15m[df_15m["ma_100"].notna()]
    if len(df_15m_valid) < REGIME_LOOKBACK_15M:
        return ("neutral", 0.0, 0.0, 0.0, 0.0, [], None, [])
    # 최근 24시간(REGIME_LOOKBACK_15M)만 사용
    df_15m_recent = df_15m_valid.tail(REGIME_LOOKBACK_15M).copy()
    last = df_15m_recent.iloc[-1]
    tail = df_15m_recent.tail(REGIME_LOOKBACK_15M)
    price_history_15m = list(zip(
        tail["high"].tolist(),
        tail["low"].tolist(),
        tail["close"].tolist(),
    ))
    # MA 값은 최근 유효한 데이터에서 사용 (MA100이 정상 계산되도록)
    short_ma_15m = float(df_15m_recent.iloc[-1]["ma_short"])
    long_ma_15m = float(df_15m_recent.iloc[-1]["ma_long"])
    ma_50_15m = float(df_15m_recent.iloc[-1]["ma_50"])
    ma_100_15m = float(df_15m_recent.iloc[-1]["ma_100"])
    rsi_15m = float(df_15m_recent.iloc[-1]["rsi"])
    # MA20 히스토리 (추세장 판단용) - 유효한 데이터에서 계산하되 최근 96개만 사용
    ma_long_history = df_15m_valid["ma_long"].tail(TREND_SLOPE_BARS).tolist()
    regime = detect_market_regime(
        short_ma_15m, long_ma_15m, current_price,
        ma_50_15m, ma_100_15m,
        price_history_15m, box_period=REGIME_LOOKBACK_15M,
        ma_long_history=ma_long_history,
    )
    return (regime, short_ma_15m, long_ma_15m, ma_50_15m, ma_100_15m, price_history_15m, rsi_15m, ma_long_history)


def fetch_ohlcv_history(exchange, days: int = 365, batch_size: int = 1500) -> pd.DataFrame:
    """과거 N일 5분봉 배치 수집."""
    target_candles = days * 24 * 12
    all_ohlcv = []
    current_end = int(time.time() * 1000)
    num_batches = (target_candles + batch_size - 1) // batch_size

    for _ in range(num_batches):
        try:
            batch_start = current_end - (batch_size * 5 * 60 * 1000)
            ohlcv = _fetch_ohlcv_with_retry(exchange, SYMBOL, TIMEFRAME, batch_size, since=batch_start)
            if not ohlcv:
                break
            if all_ohlcv:
                existing = {c[0] for c in all_ohlcv}
                ohlcv = [c for c in ohlcv if c[0] not in existing]
            if ohlcv:
                all_ohlcv.extend(ohlcv)
                all_ohlcv.sort(key=lambda x: x[0])
                current_end = all_ohlcv[0][0] - (5 * 60 * 1000)
            time.sleep(0.1)
            if len(all_ohlcv) >= target_candles:
                break
        except Exception:
            break

    if not all_ohlcv:
        return pd.DataFrame()
    # 요청한 일수만큼만 사용 (최근 N일)
    if len(all_ohlcv) > target_candles:
        all_ohlcv = all_ohlcv[-target_candles:]
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
