"""
차트 패턴 감지 모듈.

트라이앵글, 더블 바텀, 더블 탑 등 주요 차트 패턴을 감지하고,
패턴별 정석 익절/손절 가격을 반환합니다.
config.py의 익절/손절 퍼센트는 무시하고 패턴 정석대로 적용합니다.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

# 패턴 감지 기간 (5분봉 기준: 144 = 12시간)
PATTERN_LOOKBACK = 144
SWING_LOOKBACK = 12  # 스윙 고점/저점 판별용 좌우 캔들 수 (더 큰 스윙만)
SIMILARITY_PCT = 0.008  # 유사 가격 판별 (0.8% 이내 - 엄격)
MIN_PATTERN_HEIGHT_PCT = 0.5  # 최소 패턴 높이 (가격의 0.5%)


@dataclass
class ChartPattern:
    """감지된 차트 패턴 정보"""
    name: str  # 패턴명
    side: str  # "LONG" or "SHORT"
    entry_price: float
    target_price: float  # 패턴 정석 익절가
    stop_price: float  # 패턴 정석 손절가
    neckline: float  # 목선 (진입 트리거)
    pattern_height: float  # 패턴 높이 (타겟 계산용)


def _find_swing_highs(highs: List[float], lookback: int = SWING_LOOKBACK) -> List[Tuple[int, float]]:
    """스윙 고점 인덱스와 가격 리스트 반환"""
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
    """스윙 저점 인덱스와 가격 리스트 반환"""
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
    """두 가격이 pct 이내로 유사한지"""
    if a <= 0:
        return False
    return abs(a - b) / a <= pct


def detect_double_bottom(
    highs: List[float], lows: List[float], closes: List[float], current_price: float
) -> Optional[ChartPattern]:
    """
    더블 바텀 패턴 감지 (상승 반전).
    두 저점이 유사하고, 그 사이 고점(목선)을 상향 돌파 시 롱.
    익절: 목선 + (목선 - 저점)
    손절: 저점 아래
    """
    if len(lows) < PATTERN_LOOKBACK:
        return None
    recent_lows = lows[-PATTERN_LOOKBACK:]
    recent_highs = highs[-PATTERN_LOOKBACK:]
    swing_lows = _find_swing_lows(recent_lows)
    if len(swing_lows) < 2:
        return None
    # 가장 최근 두 저점
    low1_idx, low1 = swing_lows[-2]
    low2_idx, low2 = swing_lows[-1]
    if low1_idx >= low2_idx:
        return None
    if not _similar(low1, low2):
        return None
    # 목선: 두 저점 사이의 고점
    neckline = max(recent_highs[low1_idx:low2_idx + 1]) if low2_idx > low1_idx else recent_highs[low1_idx]
    # 패턴 높이
    bottom = (low1 + low2) / 2
    pattern_height = neckline - bottom
    if pattern_height <= 0:
        return None
    if pattern_height / bottom < MIN_PATTERN_HEIGHT_PCT / 100:
        return None
    # 돌파 확인: 현재가가 목선 위
    if current_price <= neckline:
        return None
    target = neckline + pattern_height
    stop = bottom * 0.995  # 저점 아래 0.5%
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
    """
    더블 탑 패턴 감지 (하락 반전).
    두 고점이 유사하고, 그 사이 저점(목선)을 하향 돌파 시 숏.
    익절: 목선 - (고점 - 목선)
    손절: 고점 위
    """
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
    stop = top * 1.005  # 고점 위 0.5%
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
    """
    상승 삼각형 (Ascending Triangle) - 상승 지속.
    수평 저항선, 상승하는 지지선. 저항선 상향 돌파 시 롱.
    익절: 저항 + 패턴 높이
    손절: 최근 저점 아래
    """
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
    # 저점이 상승하는지 확인
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
    """
    하락 삼각형 (Descending Triangle) - 하락 지속.
    수평 지지선, 하락하는 저항선. 지지선 하향 돌파 시 숏.
    익절: 지지 - 패턴 높이
    손절: 최근 고점 위
    """
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
    """
    대칭 삼각형 - 엄격한 조건으로 거의 사용 안 함 (false signal 많음).
    """
    return None  # 대칭 삼각형은 false signal이 많아 비활성화


def detect_chart_pattern(
    highs: List[float], lows: List[float], closes: List[float], current_price: float
) -> Optional[ChartPattern]:
    """
    모든 패턴을 순차 검사하고, 먼저 감지된 패턴 반환.
    우선순위: Double Bottom/Top > Ascending/Descending Triangle > Symmetrical
    """
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
