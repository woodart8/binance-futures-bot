"""
전략 로직 모듈.

"""

from dataclasses import dataclass
from typing import Literal

from config import RSI_ENTRY, RSI_EXIT


Signal = Literal["long", "short", "flat", "hold"]


@dataclass
class RsiSwingParams:
    rsi_entry: float = RSI_ENTRY
    rsi_exit: float = RSI_EXIT


def rsi_swing_signal(rsi_value: float, has_position: bool, params: RsiSwingParams) -> Signal:
    """
    RSI_SWING 전략 시그널 생성.

    - 포지션 없음 + RSI < rsi_entry  -> "long" (진입)
    - 포지션 없음 + RSI > rsi_exit -> "short" (진입)
    - 포지션 보유 + RSI >= rsi_exit   -> "flat" (청산)
    - 포지션 보유 + RSI <= rsi_entry -> "flat" (청산)
    - 그 외                             -> "hold"
    """
    if not has_position and rsi_value < params.rsi_entry:
        return "long"
    if not has_position and rsi_value > params.rsi_exit:
        return "short"
    if has_position and rsi_value >= params.rsi_exit:
        return "flat"
    if has_position and rsi_value <= params.rsi_entry:
        return "flat"
    return "hold"

