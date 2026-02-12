"""
전략 로직 모듈.

현재는 단일 RSI_SWING 롱 전략만 구현한다.
"""

from dataclasses import dataclass
from typing import Literal

from config import RSI_ENTRY, RSI_EXIT


Signal = Literal["long", "flat", "hold"]


@dataclass
class RsiSwingParams:
    rsi_entry: float = RSI_ENTRY
    rsi_exit: float = RSI_EXIT


def rsi_swing_signal(rsi_value: float, has_position: bool, params: RsiSwingParams) -> Signal:
    """
    RSI_SWING 전략 시그널 생성.

    - 포지션 없음 + RSI <= rsi_entry  -> "long" (진입)
    - 포지션 보유 + RSI >= rsi_exit   -> "flat" (청산)
    - 그 외                             -> "hold"
    """
    if not has_position and rsi_value <= params.rsi_entry:
        return "long"
    if has_position and rsi_value >= params.rsi_exit:
        return "flat"
    return "hold"

