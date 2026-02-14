"""차트 패턴 감지 (더블바텀/탑, 삼각형). 패턴 정석 TP/SL 적용."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

PATTERN_LOOKBACK = 288
SWING_LOOKBACK = 12
SIMILARITY_PCT = 0.008
MIN_PATTERN_HEIGHT_PCT = 0.5


@dataclass
class ChartPattern:
    name: str
    side: str
    entry_price: float
    target_price: float
    stop_price: float
    neckline: float
    pattern_height: float


def _find_swing_highs(highs: List[float], lookback: int = SWING_LOOKBACK) -> List[Tuple[int, float]]:
    swings = []
    n = len(highs)
    for i in range(lookback, n - lookback):
        is_high = True
        for j in range(1, lookback + 1):
            if highs[i] < highs[i - j] or highs[i] < highs[i + j]:
                is_high = False
                break
        if is_high:
            swings.append((i, highs[i]))
    return swings


def _find_swing_lows(lows: List[float], lookback: int = SWING_LOOKBACK) -> List[Tuple[int, float]]:
    swings = []
    n = len(lows)
    for i in range(lookback, n - lookback):
        is_low = True
        for j in range(1, lookback + 1):
            if lows[i] > lows[i - j] or lows[i] > lows[i + j]:
                is_low = False
                break
        if is_low:
            swings.append((i, lows[i]))
    return swings


def _similar(a: float, b: float, pct: float = SIMILARITY_PCT) -> bool:
    if a <= 0:
        return False
    return abs(a - b) / a <= pct


def detect_double_bottom(
    highs: List[float], lows: List[float], closes: List[float], current_price: float
) -> Optional[ChartPattern]:
    if len(lows) < PATTERN_LOOKBACK:
        return None
    recent_lows = lows[-PATTERN_LOOKBACK:]
    recent_highs = highs[-PATTERN_LOOKBACK:]
    swing_lows = _find_swing_lows(recent_lows)
    if len(swing_lows) < 2:
        return None
    low1_idx, low1 = swing_lows[-2]
    low2_idx, low2 = swing_lows[-1]
    if low1_idx >= low2_idx:
        return None
    if not _similar(low1, low2):
        return None
    neckline = max(recent_highs[low1_idx:low2_idx + 1]) if low2_idx > low1_idx else recent_highs[low1_idx]
    bottom = (low1 + low2) / 2
    pattern_height = neckline - bottom
    if pattern_height <= 0:
        return None
    if pattern_height / bottom < MIN_PATTERN_HEIGHT_PCT / 100:
        return None
    if current_price <= neckline:
        return None
    target = neckline + pattern_height
    stop = bottom * 0.995
    return ChartPattern(
        name="double_bottom",
        side="LONG",
        entry_price=current_price,
        target_price=target,
        stop_price=stop,
        neckline=neckline,
        pattern_height=pattern_height,
    )


def detect_double_top(
    highs: List[float], lows: List[float], closes: List[float], current_price: float
) -> Optional[ChartPattern]:
    if len(highs) < PATTERN_LOOKBACK:
        return None
    recent_highs = highs[-PATTERN_LOOKBACK:]
    recent_lows = lows[-PATTERN_LOOKBACK:]
    swing_highs = _find_swing_highs(recent_highs)
    if len(swing_highs) < 2:
        return None
    high1_idx, high1 = swing_highs[-2]
    high2_idx, high2 = swing_highs[-1]
    if high1_idx >= high2_idx:
        return None
    if not _similar(high1, high2):
        return None
    neckline = min(recent_lows[high1_idx:high2_idx + 1]) if high2_idx > high1_idx else recent_lows[high1_idx]
    top = (high1 + high2) / 2
    pattern_height = top - neckline
    if pattern_height <= 0 or neckline <= 0:
        return None
    if pattern_height / top < MIN_PATTERN_HEIGHT_PCT / 100:
        return None
    if current_price >= neckline:
        return None
    target = neckline - pattern_height
    stop = top * 1.005
    return ChartPattern(
        name="double_top",
        side="SHORT",
        entry_price=current_price,
        target_price=target,
        stop_price=stop,
        neckline=neckline,
        pattern_height=pattern_height,
    )


def detect_ascending_triangle(
    highs: List[float], lows: List[float], closes: List[float], current_price: float
) -> Optional[ChartPattern]:
    if len(highs) < PATTERN_LOOKBACK:
        return None
    recent = list(zip(highs[-PATTERN_LOOKBACK:], lows[-PATTERN_LOOKBACK:]))
    rh = [h for h, l in recent]
    rl = [l for h, l in recent]
    swing_highs = _find_swing_highs(rh)
    swing_lows = _find_swing_lows(rl)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None
    resistance = sum(h for _, h in swing_highs[-2:]) / 2
    lows_vals = [l for _, l in swing_lows[-3:]]
    if len(lows_vals) >= 2 and lows_vals[-1] <= lows_vals[-2]:
        return None
    support_low = min(rl[-20:]) if len(rl) >= 20 else min(rl)
    pattern_height = resistance - support_low
    if pattern_height <= 0 or resistance <= 0:
        return None
    if pattern_height / resistance < MIN_PATTERN_HEIGHT_PCT / 100:
        return None
    if current_price <= resistance:
        return None
    target = resistance + pattern_height
    stop = support_low * 0.995
    return ChartPattern(
        name="ascending_triangle",
        side="LONG",
        entry_price=current_price,
        target_price=target,
        stop_price=stop,
        neckline=resistance,
        pattern_height=pattern_height,
    )


def detect_descending_triangle(
    highs: List[float], lows: List[float], closes: List[float], current_price: float
) -> Optional[ChartPattern]:
    if len(highs) < PATTERN_LOOKBACK:
        return None
    recent = list(zip(highs[-PATTERN_LOOKBACK:], lows[-PATTERN_LOOKBACK:]))
    rh = [h for h, l in recent]
    rl = [l for h, l in recent]
    swing_highs = _find_swing_highs(rh)
    swing_lows = _find_swing_lows(rl)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None
    support = sum(l for _, l in swing_lows[-2:]) / 2
    highs_vals = [h for _, h in swing_highs[-3:]]
    if len(highs_vals) >= 2 and highs_vals[-1] >= highs_vals[-2]:
        return None
    resistance_high = max(rh[-20:]) if len(rh) >= 20 else max(rh)
    pattern_height = resistance_high - support
    if pattern_height <= 0 or support <= 0:
        return None
    if pattern_height / support < MIN_PATTERN_HEIGHT_PCT / 100:
        return None
    if current_price >= support:
        return None
    target = support - pattern_height
    stop = resistance_high * 1.005
    return ChartPattern(
        name="descending_triangle",
        side="SHORT",
        entry_price=current_price,
        target_price=target,
        stop_price=stop,
        neckline=support,
        pattern_height=pattern_height,
    )


def detect_symmetrical_triangle(
    highs: List[float], lows: List[float], closes: List[float], current_price: float
) -> Optional[ChartPattern]:
    return None


def detect_chart_pattern(
    highs: List[float], lows: List[float], closes: List[float], current_price: float
) -> Optional[ChartPattern]:
    detectors = [
        detect_double_bottom,
        detect_double_top,
        detect_ascending_triangle,
        detect_descending_triangle,
        detect_symmetrical_triangle,
    ]
    for det in detectors:
        pattern = det(highs, lows, closes, current_price)
        if pattern is not None:
            return pattern
    return None
