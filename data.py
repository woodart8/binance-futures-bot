"""OHLCV 조회 및 15분봉 regime 계산."""

import time
import pandas as pd

from config import (
    SYMBOL,
    TIMEFRAME,
    REGIME_LOOKBACK_15M,
    MA_SHORT_PERIOD,
    MA_LONG_PERIOD,
    MA_MID_PERIOD,
    MA_LONGEST_PERIOD,
    RSI_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
)
from indicators import calculate_ma, calculate_rsi, calculate_macd
from strategy_core import detect_market_regime


def fetch_ohlcv(exchange, limit: int = 400, symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def compute_regime_15m(df: pd.DataFrame, current_price: float) -> tuple:
    """(regime, short_ma_15m, long_ma_15m, ma_50_15m, ma_100_15m, price_history_15m, rsi_15m, macd_line_15m, macd_signal_15m)"""
    df_tmp = df.copy()
    df_tmp["timestamp"] = pd.to_datetime(df_tmp["timestamp"])
    df_15m = df_tmp.set_index("timestamp").resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    df_15m["ma_short"] = calculate_ma(df_15m["close"], MA_SHORT_PERIOD)
    df_15m["ma_long"] = calculate_ma(df_15m["close"], MA_LONG_PERIOD)
    df_15m["ma_50"] = calculate_ma(df_15m["close"], MA_MID_PERIOD)
    df_15m["ma_100"] = calculate_ma(df_15m["close"], MA_LONGEST_PERIOD)
    df_15m["rsi"] = calculate_rsi(df_15m["close"], RSI_PERIOD)
    macd_ln, macd_sig, _ = calculate_macd(df_15m["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    df_15m["macd_line"] = macd_ln
    df_15m["macd_signal"] = macd_sig
    df_15m = df_15m.dropna().reset_index()
    if len(df_15m) < REGIME_LOOKBACK_15M:
        return ("neutral", 0.0, 0.0, 0.0, 0.0, [], None, None, None)
    last = df_15m.iloc[-1]
    price_history_15m = df_15m["close"].tail(REGIME_LOOKBACK_15M).tolist()
    short_ma_15m = float(last["ma_short"])
    long_ma_15m = float(last["ma_long"])
    ma_50_15m = float(last["ma_50"])
    ma_100_15m = float(last["ma_100"])
    rsi_15m = float(last["rsi"])
    macd_line_15m = float(last["macd_line"])
    macd_signal_15m = float(last["macd_signal"])
    regime = detect_market_regime(
        short_ma_15m, long_ma_15m, current_price,
        ma_50_15m, ma_100_15m,
        price_history_15m, box_period=REGIME_LOOKBACK_15M,
    )
    return (regime, short_ma_15m, long_ma_15m, ma_50_15m, ma_100_15m, price_history_15m, rsi_15m, macd_line_15m, macd_signal_15m)


def fetch_ohlcv_history(exchange, days: int = 365, batch_size: int = 1500) -> pd.DataFrame:
    """과거 N일 5분봉 배치 수집."""
    target_candles = days * 24 * 12
    all_ohlcv = []
    current_end = int(time.time() * 1000)
    num_batches = (target_candles + batch_size - 1) // batch_size

    for _ in range(num_batches):
        try:
            batch_start = current_end - (batch_size * 5 * 60 * 1000)
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=batch_size, since=batch_start)
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
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
