"""
기술적 지표 계산 모듈.
"""

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
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("inf"))
    rsi = 100 - (100 / (1 + rs))
    return rsi

