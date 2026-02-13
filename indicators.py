"""
기술적 지표 계산 모듈.
"""

from typing import Tuple

import pandas as pd

from config import RSI_PERIOD


def calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    RSI 지표 계산.

    :param series: 종가 시계열
    :param period: RSI 기간
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("inf"))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_ma(series: pd.Series, period: int) -> pd.Series:
    """
    이동평균선(MA) 계산.

    :param series: 종가 시계열
    :param period: 이동평균 기간
    """
    return series.rolling(window=period, min_periods=period).mean()


def calculate_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD 지표 계산.

    :param series: 종가 시계열
    :param fast_period: 빠른 EMA 기간 (기본값: 12)
    :param slow_period: 느린 EMA 기간 (기본값: 26)
    :param signal_period: Signal 라인 기간 (기본값: 9)
    :return: (macd, signal, histogram) 튜플
    """
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram
