"""차트 패턴 감지 (더블바텀/탑, 삼각형). 패턴 정석 TP/SL 적용. 15분봉 72시간(288봉) 기준."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

# 15분봉 288개 = 72시간
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


# 하모닉 패턴: 피보나치 비율 허용 오차
HARMONIC_RATIO_TOL = 0.08


def _ratio_ok(actual: float, expected: float, tol: float = HARMONIC_RATIO_TOL) -> bool:
    if expected == 0:
        return False
    return abs(actual - expected) <= tol


def _get_xabcd_swings(
    highs: List[float], lows: List[float]
) -> Optional[Tuple[float, float, float, float, float, str]]:
    """스윙 고/저를 시간순으로 합쳐 마지막 5개가 교대로 나오면 (X,A,B,C,D), (x,a,b,c,d,'bullish'|'bearish') 반환."""
    recent_highs = highs[-PATTERN_LOOKBACK:] if len(highs) >= PATTERN_LOOKBACK else highs
    recent_lows = lows[-PATTERN_LOOKBACK:] if len(lows) >= PATTERN_LOOKBACK else lows
    if len(recent_highs) < 50 or len(recent_lows) < 50:
        return None
    swing_highs = _find_swing_highs(recent_highs)
    swing_lows = _find_swing_lows(recent_lows)
    points = [(i, "H", h) for i, h in swing_highs] + [(i, "L", l) for i, l in swing_lows]
    points.sort(key=lambda t: t[0])
    if len(points) < 5:
        return None
    seq = []
    current_type = None
    for i in range(len(points) - 1, -1, -1):
        idx, typ, val = points[i]
        if current_type is None or typ != current_type:
            seq.append((idx, typ, val))
            current_type = typ
        if len(seq) == 5:
            break
    if len(seq) != 5:
        return None
    seq.reverse()
    types = [t[1] for t in seq]
    x_val, a_val, b_val, c_val, d_val = [t[2] for t in seq]
    if types == ["L", "H", "L", "H", "L"]:
        return (x_val, a_val, b_val, c_val, d_val, "bullish")
    if types == ["H", "L", "H", "L", "H"]:
        return (x_val, a_val, b_val, c_val, d_val, "bearish")
    return None


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


def detect_head_and_shoulders_top(
    highs: List[float], lows: List[float], closes: List[float], current_price: float
) -> Optional[ChartPattern]:
    """헤드앤숄더 (천정): 좌숄더·헤드·우숄더 3봉, 헤드가 최고. 목선 이탈 시 SHORT."""
    if len(highs) < PATTERN_LOOKBACK:
        return None
    recent_highs = highs[-PATTERN_LOOKBACK:]
    recent_lows = lows[-PATTERN_LOOKBACK:]
    swing_highs = _find_swing_highs(recent_highs)
    if len(swing_highs) < 3:
        return None
    # 최근 3개: 좌숄더, 헤드, 우숄더 (시간순)
    (i0, h0), (i1, h1), (i2, h2) = swing_highs[-3:]
    if not (i0 < i1 < i2):
        return None
    if not (h1 > h0 and h1 > h2):
        return None
    trough1 = min(recent_lows[i0 : i1 + 1]) if i1 > i0 else recent_lows[i0]
    trough2 = min(recent_lows[i1 : i2 + 1]) if i2 > i1 else recent_lows[i1]
    neckline = (trough1 + trough2) / 2
    if neckline <= 0:
        return None
    pattern_height = h1 - neckline
    if pattern_height / neckline < MIN_PATTERN_HEIGHT_PCT / 100:
        return None
    if current_price >= neckline:
        return None
    target = neckline - pattern_height
    stop = h1 * 1.005
    return ChartPattern(
        name="head_and_shoulders_top",
        side="SHORT",
        entry_price=current_price,
        target_price=target,
        stop_price=stop,
        neckline=neckline,
        pattern_height=pattern_height,
    )


def detect_inverse_head_and_shoulders(
    highs: List[float], lows: List[float], closes: List[float], current_price: float
) -> Optional[ChartPattern]:
    """역 헤드앤숄더 (바닥): 3개 저점 중 가운데가 최저. 목선 돌파 시 LONG."""
    if len(lows) < PATTERN_LOOKBACK:
        return None
    recent_highs = highs[-PATTERN_LOOKBACK:]
    recent_lows = lows[-PATTERN_LOOKBACK:]
    swing_lows = _find_swing_lows(recent_lows)
    if len(swing_lows) < 3:
        return None
    (i0, l0), (i1, l1), (i2, l2) = swing_lows[-3:]
    if not (i0 < i1 < i2):
        return None
    if not (l1 < l0 and l1 < l2):
        return None
    peak1 = max(recent_highs[i0 : i1 + 1]) if i1 > i0 else recent_highs[i0]
    peak2 = max(recent_highs[i1 : i2 + 1]) if i2 > i1 else recent_highs[i1]
    neckline = (peak1 + peak2) / 2
    if neckline <= 0:
        return None
    pattern_height = neckline - l1
    if pattern_height / neckline < MIN_PATTERN_HEIGHT_PCT / 100:
        return None
    if current_price <= neckline:
        return None
    target = neckline + pattern_height
    stop = l1 * 0.995
    return ChartPattern(
        name="inverse_head_and_shoulders",
        side="LONG",
        entry_price=current_price,
        target_price=target,
        stop_price=stop,
        neckline=neckline,
        pattern_height=pattern_height,
    )


def detect_symmetrical_triangle(
    highs: List[float], lows: List[float], closes: List[float], current_price: float
) -> Optional[ChartPattern]:
    """대칭 삼각형: 위쪽은 낮은 고점, 아래쪽은 높은 저점. 돌파 방향으로 진입."""
    if len(highs) < PATTERN_LOOKBACK:
        return None
    recent_highs = highs[-PATTERN_LOOKBACK:]
    recent_lows = lows[-PATTERN_LOOKBACK:]
    swing_highs = _find_swing_highs(recent_highs)
    swing_lows = _find_swing_lows(recent_lows)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None
    # 위쪽: 최근 스윙 고점들이 하락 추세인지 (lower highs)
    h_vals = [h for _, h in swing_highs[-3:]]
    if len(h_vals) >= 2 and h_vals[-1] >= h_vals[-2]:
        return None
    # 아래쪽: 최근 스윙 저점들이 상승 추세인지 (higher lows)
    l_vals = [l for _, l in swing_lows[-3:]]
    if len(l_vals) >= 2 and l_vals[-1] <= l_vals[-2]:
        return None
    resistance = sum(h for _, h in swing_highs[-2:]) / 2
    support = sum(l for _, l in swing_lows[-2:]) / 2
    pattern_height = resistance - support
    if pattern_height <= 0 or support <= 0:
        return None
    if pattern_height / support < MIN_PATTERN_HEIGHT_PCT / 100:
        return None
    # 상단 돌파 → LONG, 하단 이탈 → SHORT (한 번에 하나만 반환, 상단 돌파 우선)
    if current_price > resistance:
        target = resistance + pattern_height
        stop = support * 0.995
        return ChartPattern(
            name="symmetrical_triangle",
            side="LONG",
            entry_price=current_price,
            target_price=target,
            stop_price=stop,
            neckline=resistance,
            pattern_height=pattern_height,
        )
    if current_price < support:
        target = support - pattern_height
        stop = resistance * 1.005
        return ChartPattern(
            name="symmetrical_triangle",
            side="SHORT",
            entry_price=current_price,
            target_price=target,
            stop_price=stop,
            neckline=support,
            pattern_height=pattern_height,
        )
    return None


def detect_falling_wedge(
    highs: List[float], lows: List[float], closes: List[float], current_price: float
) -> Optional[ChartPattern]:
    """하락 쐐기: 낮은 고점·낮은 저점으로 수렴 후 상단 돌파 시 LONG."""
    if len(highs) < PATTERN_LOOKBACK:
        return None
    recent_highs = highs[-PATTERN_LOOKBACK:]
    recent_lows = lows[-PATTERN_LOOKBACK:]
    swing_highs = _find_swing_highs(recent_highs)
    swing_lows = _find_swing_lows(recent_lows)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None
    h_vals = [h for _, h in swing_highs[-3:]]
    l_vals = [l for _, l in swing_lows[-3:]]
    if len(h_vals) >= 2 and h_vals[-1] >= h_vals[-2]:
        return None
    if len(l_vals) >= 2 and l_vals[-1] >= l_vals[-2]:
        return None
    resistance = sum(h for _, h in swing_highs[-2:]) / 2
    support = sum(l for _, l in swing_lows[-2:]) / 2
    pattern_height = resistance - support
    if pattern_height <= 0 or support <= 0:
        return None
    if pattern_height / support < MIN_PATTERN_HEIGHT_PCT / 100:
        return None
    if current_price <= resistance:
        return None
    target = resistance + pattern_height
    stop = support * 0.995
    return ChartPattern(
        name="falling_wedge",
        side="LONG",
        entry_price=current_price,
        target_price=target,
        stop_price=stop,
        neckline=resistance,
        pattern_height=pattern_height,
    )


def detect_rising_wedge(
    highs: List[float], lows: List[float], closes: List[float], current_price: float
) -> Optional[ChartPattern]:
    """상승 쐐기: 높은 고점·높은 저점으로 수렴 후 하단 이탈 시 SHORT."""
    if len(highs) < PATTERN_LOOKBACK:
        return None
    recent_highs = highs[-PATTERN_LOOKBACK:]
    recent_lows = lows[-PATTERN_LOOKBACK:]
    swing_highs = _find_swing_highs(recent_highs)
    swing_lows = _find_swing_lows(recent_lows)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None
    h_vals = [h for _, h in swing_highs[-3:]]
    l_vals = [l for _, l in swing_lows[-3:]]
    if len(h_vals) >= 2 and h_vals[-1] <= h_vals[-2]:
        return None
    if len(l_vals) >= 2 and l_vals[-1] <= l_vals[-2]:
        return None
    resistance = sum(h for _, h in swing_highs[-2:]) / 2
    support = sum(l for _, l in swing_lows[-2:]) / 2
    pattern_height = resistance - support
    if pattern_height <= 0 or support <= 0:
        return None
    if pattern_height / support < MIN_PATTERN_HEIGHT_PCT / 100:
        return None
    if current_price >= support:
        return None
    target = support - pattern_height
    stop = resistance * 1.005
    return ChartPattern(
        name="rising_wedge",
        side="SHORT",
        entry_price=current_price,
        target_price=target,
        stop_price=stop,
        neckline=support,
        pattern_height=pattern_height,
    )


def detect_bull_flag(
    highs: List[float], lows: List[float], closes: List[float], current_price: float
) -> Optional[ChartPattern]:
    """불 플래그: 상승 폴대 후 하락 채널(숏타임) 돌파 시 LONG. 목표=폴 높이만큼 위로."""
    if len(highs) < PATTERN_LOOKBACK:
        return None
    recent_highs = highs[-PATTERN_LOOKBACK:]
    recent_lows = lows[-PATTERN_LOOKBACK:]
    swing_highs = _find_swing_highs(recent_highs)
    swing_lows = _find_swing_lows(recent_lows)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None
    # 폴: 이전 저점에서 최근 고점까지 상승
    pole_start = min(l for _, l in swing_lows[-3:]) if len(swing_lows) >= 2 else min(recent_lows)
    pole_end = max(h for _, h in swing_highs[-2:])
    if pole_end <= pole_start:
        return None
    pole_height = pole_end - pole_start
    if pole_height / pole_start < MIN_PATTERN_HEIGHT_PCT / 100:
        return None
    # 플래그: 최근 구간에서 약한 하락 채널(낮은 고점) 후 저항선 = 최근 고점대
    flag_high = max(recent_highs[-20:]) if len(recent_highs) >= 20 else max(recent_highs)
    if current_price <= flag_high:
        return None
    target = current_price + pole_height
    stop = min(recent_lows[-15:]) * 0.995 if len(recent_lows) >= 15 else min(recent_lows) * 0.995
    return ChartPattern(
        name="bull_flag",
        side="LONG",
        entry_price=current_price,
        target_price=target,
        stop_price=stop,
        neckline=flag_high,
        pattern_height=pole_height,
    )


def detect_bear_flag(
    highs: List[float], lows: List[float], closes: List[float], current_price: float
) -> Optional[ChartPattern]:
    """베어 플래그: 하락 폴대 후 상승 채널(숏타임) 이탈 시 SHORT. 목표=폴 높이만큼 아래."""
    if len(highs) < PATTERN_LOOKBACK:
        return None
    recent_highs = highs[-PATTERN_LOOKBACK:]
    recent_lows = lows[-PATTERN_LOOKBACK:]
    swing_highs = _find_swing_highs(recent_highs)
    swing_lows = _find_swing_lows(recent_lows)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None
    pole_start = max(h for _, h in swing_highs[-3:]) if len(swing_highs) >= 2 else max(recent_highs)
    pole_end = min(l for _, l in swing_lows[-2:])
    if pole_end >= pole_start:
        return None
    pole_height = pole_start - pole_end
    if pole_height / pole_start < MIN_PATTERN_HEIGHT_PCT / 100:
        return None
    flag_low = min(recent_lows[-20:]) if len(recent_lows) >= 20 else min(recent_lows)
    if current_price >= flag_low:
        return None
    target = current_price - pole_height
    stop = max(recent_highs[-15:]) * 1.005 if len(recent_highs) >= 15 else max(recent_highs) * 1.005
    return ChartPattern(
        name="bear_flag",
        side="SHORT",
        entry_price=current_price,
        target_price=target,
        stop_price=stop,
        neckline=flag_low,
        pattern_height=pole_height,
    )


# ---------- 하모닉 패턴 (XABCD, 피보나치 비율) ----------


def _check_gartley_bullish(
    x: float, a: float, b: float, c: float, d: float, tol: float
) -> bool:
    """Gartley 불리시: B=0.618 XA, BC=0.382~0.886 AB, CD=1.272~1.618 BC, D=0.786 XA."""
    xa = a - x
    if xa <= 0:
        return False
    ab_ret = (a - b) / xa
    if not _ratio_ok(ab_ret, 0.618, tol):
        return False
    ab_len = a - b
    if ab_len <= 0:
        return False
    bc_len = c - b
    if bc_len <= 0:
        return False
    bc_ratio = bc_len / ab_len
    if bc_ratio < 0.382 - tol or bc_ratio > 0.886 + tol:
        return False
    cd_len = c - d
    if cd_len <= 0:
        return False
    cd_ratio = cd_len / bc_len
    if cd_ratio < 1.272 - tol or cd_ratio > 1.618 + tol:
        return False
    d_ret = (a - d) / xa
    if not _ratio_ok(d_ret, 0.786, tol):
        return False
    return True


def _check_gartley_bearish(
    x: float, a: float, b: float, c: float, d: float, tol: float
) -> bool:
    """Gartley 베어시: X고점 A저점, B=0.618 XA, D=0.786 XA."""
    xa = x - a
    if xa <= 0:
        return False
    ab_ret = (b - a) / xa
    if not _ratio_ok(ab_ret, 0.618, tol):
        return False
    ab_len = b - a
    if ab_len <= 0:
        return False
    bc_len = b - c
    if bc_len <= 0:
        return False
    bc_ratio = bc_len / ab_len
    if bc_ratio < 0.382 - tol or bc_ratio > 0.886 + tol:
        return False
    cd_len = d - c
    if cd_len <= 0:
        return False
    cd_ratio = cd_len / bc_len
    if cd_ratio < 1.272 - tol or cd_ratio > 1.618 + tol:
        return False
    d_ret = (d - a) / xa
    if not _ratio_ok(d_ret, 0.786, tol):
        return False
    return True


def _check_butterfly_bullish(x: float, a: float, b: float, c: float, d: float, tol: float) -> bool:
    """Butterfly 불리시: AB=0.786 XA, BC=0.382~0.886 AB, CD=1.272~1.618 BC."""
    xa = a - x
    if xa <= 0:
        return False
    ab_ret = (a - b) / xa
    if not _ratio_ok(ab_ret, 0.786, tol):
        return False
    ab_len = a - b
    if ab_len <= 0:
        return False
    bc_len = c - b
    if bc_len <= 0:
        return False
    bc_ratio = bc_len / ab_len
    if bc_ratio < 0.382 - tol or bc_ratio > 0.886 + tol:
        return False
    cd_len = c - d
    if cd_len <= 0:
        return False
    cd_ratio = cd_len / bc_len
    if cd_ratio < 1.272 - tol or cd_ratio > 1.618 + tol:
        return False
    return True


def _check_butterfly_bearish(x: float, a: float, b: float, c: float, d: float, tol: float) -> bool:
    xa = x - a
    if xa <= 0:
        return False
    ab_ret = (b - a) / xa
    if not _ratio_ok(ab_ret, 0.786, tol):
        return False
    ab_len = b - a
    if ab_len <= 0:
        return False
    bc_len = b - c
    if bc_len <= 0:
        return False
    bc_ratio = bc_len / ab_len
    if bc_ratio < 0.382 - tol or bc_ratio > 0.886 + tol:
        return False
    cd_len = d - c
    if cd_len <= 0:
        return False
    cd_ratio = cd_len / bc_len
    if cd_ratio < 1.272 - tol or cd_ratio > 1.618 + tol:
        return False
    return True


def _check_bat_bullish(x: float, a: float, b: float, c: float, d: float, tol: float) -> bool:
    """Bat 불리시: AB=0.382~0.5 XA, CD=1.618~2.618 BC."""
    xa = a - x
    if xa <= 0:
        return False
    ab_ret = (a - b) / xa
    if ab_ret < 0.382 - tol or ab_ret > 0.5 + tol:
        return False
    ab_len = a - b
    if ab_len <= 0:
        return False
    bc_len = c - b
    if bc_len <= 0:
        return False
    bc_ratio = bc_len / ab_len
    if bc_ratio < 0.382 - tol or bc_ratio > 0.886 + tol:
        return False
    cd_len = c - d
    if cd_len <= 0:
        return False
    cd_ratio = cd_len / bc_len
    if cd_ratio < 1.618 - tol or cd_ratio > 2.618 + tol:
        return False
    return True


def _check_bat_bearish(x: float, a: float, b: float, c: float, d: float, tol: float) -> bool:
    xa = x - a
    if xa <= 0:
        return False
    ab_ret = (b - a) / xa
    if ab_ret < 0.382 - tol or ab_ret > 0.5 + tol:
        return False
    ab_len = b - a
    if ab_len <= 0:
        return False
    bc_len = b - c
    if bc_len <= 0:
        return False
    bc_ratio = bc_len / ab_len
    if bc_ratio < 0.382 - tol or bc_ratio > 0.886 + tol:
        return False
    cd_len = d - c
    if cd_len <= 0:
        return False
    cd_ratio = cd_len / bc_len
    if cd_ratio < 1.618 - tol or cd_ratio > 2.618 + tol:
        return False
    return True


def _check_crab_bullish(x: float, a: float, b: float, c: float, d: float, tol: float) -> bool:
    """Crab 불리시: AB=0.382~0.618 XA, CD=2.618~3.618 BC."""
    xa = a - x
    if xa <= 0:
        return False
    ab_ret = (a - b) / xa
    if ab_ret < 0.382 - tol or ab_ret > 0.618 + tol:
        return False
    ab_len = a - b
    if ab_len <= 0:
        return False
    bc_len = c - b
    if bc_len <= 0:
        return False
    bc_ratio = bc_len / ab_len
    if bc_ratio < 0.382 - tol or bc_ratio > 0.886 + tol:
        return False
    cd_len = c - d
    if cd_len <= 0:
        return False
    cd_ratio = cd_len / bc_len
    if cd_ratio < 2.618 - tol or cd_ratio > 3.618 + tol:
        return False
    return True


def _check_crab_bearish(x: float, a: float, b: float, c: float, d: float, tol: float) -> bool:
    xa = x - a
    if xa <= 0:
        return False
    ab_ret = (b - a) / xa
    if ab_ret < 0.382 - tol or ab_ret > 0.618 + tol:
        return False
    ab_len = b - a
    if ab_len <= 0:
        return False
    bc_len = b - c
    if bc_len <= 0:
        return False
    bc_ratio = bc_len / ab_len
    if bc_ratio < 0.382 - tol or bc_ratio > 0.886 + tol:
        return False
    cd_len = d - c
    if cd_len <= 0:
        return False
    cd_ratio = cd_len / bc_len
    if cd_ratio < 2.618 - tol or cd_ratio > 3.618 + tol:
        return False
    return True


def detect_harmonic_gartley(
    highs: List[float], lows: List[float], closes: List[float], current_price: float
) -> Optional[ChartPattern]:
    """하모닉 Gartley: B=0.618 XA, D=0.786 XA. PRZ(D) 근처 반등/하락 진입."""
    t = _get_xabcd_swings(highs, lows)
    if t is None:
        return None
    x, a, b, c, d = t[0], t[1], t[2], t[3], t[4]
    kind = t[5]
    tol = HARMONIC_RATIO_TOL
    if kind == "bullish" and _check_gartley_bullish(x, a, b, c, d, tol):
        if (a - x) / x < MIN_PATTERN_HEIGHT_PCT / 100:
            return None
        if current_price < d * 0.997 or current_price > a:
            return None
        target = d + 0.618 * (a - d)
        stop = x * 0.995
        return ChartPattern(
            name="harmonic_gartley",
            side="LONG",
            entry_price=current_price,
            target_price=target,
            stop_price=stop,
            neckline=d,
            pattern_height=a - d,
        )
    if kind == "bearish" and _check_gartley_bearish(x, a, b, c, d, tol):
        if (x - a) / x < MIN_PATTERN_HEIGHT_PCT / 100:
            return None
        if current_price > d * 1.003 or current_price < a:
            return None
        target = d - 0.618 * (d - a)
        stop = x * 1.005
        return ChartPattern(
            name="harmonic_gartley",
            side="SHORT",
            entry_price=current_price,
            target_price=target,
            stop_price=stop,
            neckline=d,
            pattern_height=d - a,
        )
    return None


def detect_harmonic_butterfly(
    highs: List[float], lows: List[float], closes: List[float], current_price: float
) -> Optional[ChartPattern]:
    """하모닉 Butterfly: AB=0.786 XA. PRZ(D)에서 반전 진입."""
    t = _get_xabcd_swings(highs, lows)
    if t is None:
        return None
    x, a, b, c, d = t[0], t[1], t[2], t[3], t[4]
    kind = t[5]
    tol = HARMONIC_RATIO_TOL
    if kind == "bullish" and _check_butterfly_bullish(x, a, b, c, d, tol):
        if (a - x) / x < MIN_PATTERN_HEIGHT_PCT / 100:
            return None
        if current_price < d * 0.997 or current_price > a:
            return None
        target = d + 0.618 * (a - d)
        stop = x * 0.995
        return ChartPattern(
            name="harmonic_butterfly",
            side="LONG",
            entry_price=current_price,
            target_price=target,
            stop_price=stop,
            neckline=d,
            pattern_height=a - d,
        )
    if kind == "bearish" and _check_butterfly_bearish(x, a, b, c, d, tol):
        if (x - a) / x < MIN_PATTERN_HEIGHT_PCT / 100:
            return None
        if current_price > d * 1.003 or current_price < a:
            return None
        target = d - 0.618 * (d - a)
        stop = x * 1.005
        return ChartPattern(
            name="harmonic_butterfly",
            side="SHORT",
            entry_price=current_price,
            target_price=target,
            stop_price=stop,
            neckline=d,
            pattern_height=d - a,
        )
    return None


def detect_harmonic_bat(
    highs: List[float], lows: List[float], closes: List[float], current_price: float
) -> Optional[ChartPattern]:
    """하모닉 Bat: AB=0.382~0.5 XA, CD=1.618~2.618 BC."""
    t = _get_xabcd_swings(highs, lows)
    if t is None:
        return None
    x, a, b, c, d = t[0], t[1], t[2], t[3], t[4]
    kind = t[5]
    tol = HARMONIC_RATIO_TOL
    if kind == "bullish" and _check_bat_bullish(x, a, b, c, d, tol):
        if (a - x) / x < MIN_PATTERN_HEIGHT_PCT / 100:
            return None
        if current_price < d * 0.997 or current_price > a:
            return None
        target = d + 0.618 * (a - d)
        stop = x * 0.995
        return ChartPattern(
            name="harmonic_bat",
            side="LONG",
            entry_price=current_price,
            target_price=target,
            stop_price=stop,
            neckline=d,
            pattern_height=a - d,
        )
    if kind == "bearish" and _check_bat_bearish(x, a, b, c, d, tol):
        if (x - a) / x < MIN_PATTERN_HEIGHT_PCT / 100:
            return None
        if current_price > d * 1.003 or current_price < a:
            return None
        target = d - 0.618 * (d - a)
        stop = x * 1.005
        return ChartPattern(
            name="harmonic_bat",
            side="SHORT",
            entry_price=current_price,
            target_price=target,
            stop_price=stop,
            neckline=d,
            pattern_height=d - a,
        )
    return None


def detect_harmonic_crab(
    highs: List[float], lows: List[float], closes: List[float], current_price: float
) -> Optional[ChartPattern]:
    """하모닉 Crab: AB=0.382~0.618 XA, CD=2.618~3.618 BC. PRZ 정확도 높음."""
    t = _get_xabcd_swings(highs, lows)
    if t is None:
        return None
    x, a, b, c, d = t[0], t[1], t[2], t[3], t[4]
    kind = t[5]
    tol = HARMONIC_RATIO_TOL
    if kind == "bullish" and _check_crab_bullish(x, a, b, c, d, tol):
        if (a - x) / x < MIN_PATTERN_HEIGHT_PCT / 100:
            return None
        if current_price < d * 0.997 or current_price > a:
            return None
        target = d + 0.618 * (a - d)
        stop = x * 0.995
        return ChartPattern(
            name="harmonic_crab",
            side="LONG",
            entry_price=current_price,
            target_price=target,
            stop_price=stop,
            neckline=d,
            pattern_height=a - d,
        )
    if kind == "bearish" and _check_crab_bearish(x, a, b, c, d, tol):
        if (x - a) / x < MIN_PATTERN_HEIGHT_PCT / 100:
            return None
        if current_price > d * 1.003 or current_price < a:
            return None
        target = d - 0.618 * (d - a)
        stop = x * 1.005
        return ChartPattern(
            name="harmonic_crab",
            side="SHORT",
            entry_price=current_price,
            target_price=target,
            stop_price=stop,
            neckline=d,
            pattern_height=d - a,
        )
    return None


def detect_chart_pattern(
    highs: List[float], lows: List[float], closes: List[float], current_price: float
) -> Optional[ChartPattern]:
    # 우선순위: 더블 → 삼각형 → 쐐기 → 플래그 → 하모닉 → H&S
    detectors = [
        detect_double_top,
        detect_double_bottom,
        detect_ascending_triangle,
        detect_descending_triangle,
        detect_symmetrical_triangle,
        detect_falling_wedge,
        detect_rising_wedge,
        detect_bull_flag,
        detect_bear_flag,
        detect_head_and_shoulders_top,
        detect_inverse_head_and_shoulders,
        detect_harmonic_gartley,
        detect_harmonic_crab,
        detect_harmonic_bat,
        detect_harmonic_butterfly,
    ]
    for det in detectors:
        pattern = det(highs, lows, closes, current_price)
        if pattern is not None:
            return pattern
    return None
