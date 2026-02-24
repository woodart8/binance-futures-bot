"""페이퍼/백테스트 전용 전략 로직. 수정 시 라이브에 영향 없음.

횡보장: 박스 = 간격 2시간 이상인 고가 2개→상단, 저가 2개→하단. 기울기 0.5% 이내·상하단 비슷. 터치 조건 없음.
진입: 박스 하단 3% 이내 롱, 상단 3% 이내 숏. MA 조건 없음.
장세 이탈: 박스는 직전 봉까지로 계산해, 가격이 상·하단 0.5% 돌파 시 neutral(중립) 전환.

추세장: MA20 기울기 ±2.5% 초과 시 상승/하락 판단.
- 상승장 롱: 가격 ≤ MA20 + RSI ≤ 40(설정), 옵션으로 RSI 상승 전환 시에만 진입(조정 초반 롱 방지)
- 상승장 숏: 가격 ≥ MA20 + RSI ≥ 80
- 하락장 롱: 가격 ≤ MA20 + RSI ≤ 20
- 하락장 숏: 가격 ≥ MA20 + RSI ≥ 62(설정), 옵션으로 RSI 꺾임 시에만 진입(반등 초반 숏 방지)
익절: 5.5%, 손절: 2.5%
"""

from typing import Literal, Optional, List, Union, Tuple

# price_history: List[float] = 종가만, List[Tuple[float,float,float]] = (고가, 저가, 종가)
PriceHistory = Union[List[float], List[Tuple[float, float, float]]]

from config_paper import (
    MA_SHORT_PERIOD,
    MA_LONG_PERIOD,
    REGIME_LOOKBACK_15M,
    SIDEWAYS_BOX_PERIOD,
    SIDEWAYS_BOX_TOP_MARGIN,
    SIDEWAYS_BOX_BOTTOM_MARGIN,
    SIDEWAYS_ENABLED,
    SIDEWAYS_MIN_TOUCHES,
    SIDEWAYS_BOX_RANGE_PCT_MIN,
    SIDEWAYS_BOX_RANGE_MIN,
    SIDEWAYS_BOX_TOUCH_GAP_MIN_HOURS,
    SIDEWAYS_BOX_SLOPE_DIFF_MAX,
    SIDEWAYS_BOX_SLOPE_MAX,
    SIDEWAYS_BOX_EXIT_MARGIN_PCT,
    TREND_SLOPE_BARS,
    TREND_SLOPE_MIN_PCT,
    TREND_UPTREND_LONG_RSI_MAX,
    TREND_UPTREND_LONG_ENABLED,
    TREND_UPTREND_LONG_REQUIRE_RSI_TURNUP,
    TREND_UPTREND_SHORT_RSI_MIN,
    TREND_DOWNTREND_LONG_RSI_MAX,
    TREND_DOWNTREND_SHORT_RSI_MIN,
    TREND_DOWNTREND_SHORT_ENABLED,
    TREND_DOWNTREND_SHORT_REQUIRE_RSI_TURNDOWN,
)

Signal = Literal["long", "short", "flat", "hold"]
MarketRegime = Literal["trend", "sideways", "neutral"]

REGIME_KR = {"sideways": "횡보장", "trend": "추세장", "neutral": "중립"}
BOX_TOUCH_THRESHOLD = 0.012


def _recent_hlc(recent: list) -> Tuple[List[float], List[float], List[float]]:
    """recent가 (고가,저가,종가) 튜플 리스트면 분리, 아니면 종가만 있으면 고/저/종 모두 종가로."""
    if not recent:
        return ([], [], [])
    if isinstance(recent[0], (list, tuple)) and len(recent[0]) >= 3:
        highs = [x[0] for x in recent]
        lows = [x[1] for x in recent]
        closes = [x[2] for x in recent]
        return (highs, lows, closes)
    closes = list(recent)
    return (closes, closes, closes)


def _min_gap_candles(period: int) -> int:
    mins_per_candle = 15 if period == REGIME_LOOKBACK_15M else 5
    return max(1, int(SIDEWAYS_BOX_TOUCH_GAP_MIN_HOURS * 60 / mins_per_candle))


def _box_high_low_from_two_points(
    highs: List[float], lows: List[float], period: int
) -> Optional[Tuple[float, float]]:
    """
    현재보다 이전 봉들 중:
    - 간격 2시간 이상이면서 가장 높은 가격 2개 → 박스 상단(그 중 큰 값)
    - 간격 2시간 이상이면서 가장 낮은 가격 2개 → 박스 하단(그 중 작은 값)
    반환: (box_high, box_low) 또는 조건 미충족 시 None.
    """
    if not highs or not lows or len(highs) != len(lows):
        return None
    n = len(highs)
    gap = _min_gap_candles(period)
    if n < gap + 1:
        return None

    # 상단: 고가 최대인 인덱스 i1, i1과 간격>=gap 인 봉 중 고가 최대인 i2 → box_high = max(highs[i1], highs[i2])
    i_top1 = max(range(n), key=lambda i: highs[i])
    candidates_top2 = [j for j in range(n) if abs(j - i_top1) >= gap]
    if not candidates_top2:
        return None
    i_top2 = max(candidates_top2, key=lambda j: highs[j])
    box_high = max(highs[i_top1], highs[i_top2])

    # 하단: 저가 최소인 인덱스 i1, 간격>=gap 인 봉 중 저가 최소인 i2 → box_low = min(lows[i1], lows[i2])
    i_bot1 = min(range(n), key=lambda i: lows[i])
    candidates_bot2 = [j for j in range(n) if abs(j - i_bot1) >= gap]
    if not candidates_bot2:
        return None
    i_bot2 = min(candidates_bot2, key=lambda j: lows[j])
    box_low = min(lows[i_bot1], lows[i_bot2])

    if box_high <= box_low:
        return None

    # 상단 두 봉 기울기와 하단 두 봉 기울기가 비슷하고, 너무 가파르지 않아야 함
    idx_top = abs(i_top2 - i_top1)
    idx_bot = abs(i_bot2 - i_bot1)
    if idx_top > 0 and idx_bot > 0 and box_low > 0:
        slope_top = (highs[i_top2] - highs[i_top1]) / idx_top
        slope_bot = (lows[i_bot2] - lows[i_bot1]) / idx_bot
        slope_top_norm = slope_top / box_low
        slope_bot_norm = slope_bot / box_low
        if abs(slope_top_norm) > SIDEWAYS_BOX_SLOPE_MAX or abs(slope_bot_norm) > SIDEWAYS_BOX_SLOPE_MAX:
            return None  # 기울기가 너무 가파르면 패스
        if abs(slope_top_norm - slope_bot_norm) > SIDEWAYS_BOX_SLOPE_DIFF_MAX:
            return None

    return (box_high, box_low)


def _box_touch_gap_ok(
    recent_highs: List[float], recent_lows: List[float],
    box_high: float, box_low: float, period: int
) -> tuple:
    """(미사용) 고점/저점 터치 간격 검사. 횡보 판정에서 터치 조건 제거됨."""
    min_gap_candles = _min_gap_candles(period)
    top_indices = [i for i, h in enumerate(recent_highs) if box_high > 0 and abs(h - box_high) / box_high < BOX_TOUCH_THRESHOLD]
    bottom_indices = [i for i, l in enumerate(recent_lows) if box_low > 0 and abs(l - box_low) / box_low < BOX_TOUCH_THRESHOLD]
    top_ok = len(top_indices) >= 2 and (max(top_indices) - min(top_indices)) >= min_gap_candles
    bottom_ok = len(bottom_indices) >= 2 and (max(bottom_indices) - min(bottom_indices)) >= min_gap_candles
    return (top_ok, bottom_ok)


def detect_market_regime(
    short_ma: float,
    long_ma: float,
    price: float,
    ma_50: float = 0.0,
    ma_100: float = 0.0,
    price_history: Optional[PriceHistory] = None,
    box_period: Optional[int] = None,
    ma_long_history: Optional[List[float]] = None,
) -> MarketRegime:
    """
    장세 판별. 순서: 1) 추세(24h 15분봉 MA20 기울기 ±2.5% 초과) 2) 횡보(박스) 3) 중립.
    """
    # 1) 추세장 판별: 24시간(96봉) 15분봉 MA20 기울기 ±2.5% 초과
    if ma_long_history and len(ma_long_history) >= TREND_SLOPE_BARS:
        # 마지막 TREND_SLOPE_BARS개만 사용 (최근 96개)
        recent_ma20 = ma_long_history[-TREND_SLOPE_BARS:]
        ma20_start = recent_ma20[0]
        ma20_end = recent_ma20[-1]
        if ma20_start and ma20_start > 0:
            slope_pct = (ma20_end - ma20_start) / ma20_start * 100.0
            if abs(slope_pct) >= TREND_SLOPE_MIN_PCT:
                return "trend"

    # 2) 횡보장 판별: 박스 조건
    if long_ma <= 0 or ma_50 <= 0 or ma_100 <= 0:
        return "neutral"

    period = box_period or SIDEWAYS_BOX_PERIOD
    if price_history and len(price_history) >= period:
        # 이탈 판정용 박스는 '직전 봉까지'로 계산 (현재 봉 포함하면 뚫고 나갔을 때 상/하단이 넓어져 추세로 안 바뀜)
        if len(price_history) >= period + 1:
            recent_for_box = price_history[-(period + 1) : -1]
        else:
            recent_for_box = price_history[-period:]
        box_len = len(recent_for_box)
        highs, lows, closes = _recent_hlc(recent_for_box)
        pair = _box_high_low_from_two_points(highs, lows, box_len)
        if pair is None:
            return "neutral"
        box_high, box_low = pair
        box_range = box_high - box_low
        if box_range > 0:
            m = SIDEWAYS_BOX_EXIT_MARGIN_PCT / 100
            if price > box_high * (1 + m) or price < box_low * (1 - m):
                return "neutral"  # 박스권 이탈 구간
            pct = box_range / box_low * 100
            if pct >= SIDEWAYS_BOX_RANGE_PCT_MIN and box_range >= SIDEWAYS_BOX_RANGE_MIN and box_low <= price <= box_high:
                return "sideways"

    # 3) 중립
    return "neutral"


def _validate_sideways_box(
    price_history: Optional[PriceHistory], price: float, period: Optional[int] = None
) -> Optional[tuple]:
    p = period or SIDEWAYS_BOX_PERIOD
    if not price_history or len(price_history) < p:
        return None
    recent = price_history[-p:]
    highs, lows, _ = _recent_hlc(recent)
    pair = _box_high_low_from_two_points(highs, lows, p)
    if pair is None:
        return None
    box_high, box_low = pair
    box_range = box_high - box_low
    if box_range < SIDEWAYS_BOX_RANGE_MIN:
        return None
    pct = box_range / box_low * 100
    if pct < SIDEWAYS_BOX_RANGE_PCT_MIN or not (box_low <= price <= box_high):
        return None
    pos = (price - box_low) / box_range
    return (box_high, box_low, box_range, pos)


def calculate_box_range(
    price_history: PriceHistory, period: int = SIDEWAYS_BOX_PERIOD
) -> Optional[tuple]:
    if not price_history or len(price_history) < period:
        return None
    recent = price_history[-period:]
    highs, lows, _ = _recent_hlc(recent)
    pair = _box_high_low_from_two_points(highs, lows, period)
    if pair is None:
        return None
    h, l = pair
    if h <= l:
        return None
    return (h, l, h - l)


def get_sideways_box_bounds(
    price_history: Optional[PriceHistory], period: int
) -> Optional[Tuple[float, float]]:
    """박스 상·하단(간격 2시간 이상인 고가 2개/저가 2개 기준). (box_high, box_low) 또는 None."""
    if not price_history or len(price_history) < period:
        return None
    recent = price_history[-period:]
    highs, lows, _ = _recent_hlc(recent)
    return _box_high_low_from_two_points(highs, lows, period)


def get_trend_direction(price_history: Optional[PriceHistory]) -> Optional[str]:
    """
    추세 방향을 반환. "up" (상승), "down" (하락), 또는 None (데이터 부족).
    24시간(96봉) 15분봉 종가 기울기 기준.
    """
    if not price_history or len(price_history) < TREND_SLOPE_BARS:
        return None
    closes = [p[2] if isinstance(p, (list, tuple)) and len(p) >= 3 else p for p in price_history[-TREND_SLOPE_BARS:]]
    if len(closes) < TREND_SLOPE_BARS or not closes[0] or closes[0] <= 0:
        return None
    slope_pct = (closes[-1] - closes[0]) / closes[0] * 100.0
    if abs(slope_pct) < TREND_SLOPE_MIN_PCT:
        return None  # 추세장이 아님
    return "up" if slope_pct > 0 else "down"


def swing_strategy_signal(
    rsi_value: float,
    price: float,
    short_ma: float,
    long_ma: float,
    has_position: bool,
    is_long: bool,
    regime: MarketRegime,
    price_history: Optional[PriceHistory] = None,
    rsi_prev: Optional[float] = None,
    open_prev: Optional[float] = None,
    close_prev: Optional[float] = None,
    open_curr: Optional[float] = None,
    *,
    regime_short_ma: Optional[float] = None,
    regime_long_ma: Optional[float] = None,
    regime_ma_50: Optional[float] = None,
    regime_ma_100: Optional[float] = None,
    regime_price_history: Optional[PriceHistory] = None,
    regime_ma_long_history: Optional[List[float]] = None,
) -> Signal:
    # 추세장: 기울기로 상승/하락 판단 후 각각 다른 진입 조건
    if regime == "trend":
        ma_short = regime_short_ma if regime_short_ma is not None else short_ma
        ma_long = regime_long_ma if regime_long_ma is not None else long_ma
        ma_50 = regime_ma_50 if regime_ma_50 is not None else None
        if ma_short is None or ma_short <= 0 or ma_long is None or ma_long <= 0:
            return "hold"
        
        # MA20 기울기로 상승/하락 판단
        uptrend = False
        downtrend = False
        if regime_ma_long_history and len(regime_ma_long_history) >= TREND_SLOPE_BARS:
            # 마지막 TREND_SLOPE_BARS개만 사용 (최근 96개)
            recent_ma20 = regime_ma_long_history[-TREND_SLOPE_BARS:]
            ma20_start = recent_ma20[0]
            ma20_end = recent_ma20[-1]
            if ma20_start and ma20_start > 0:
                slope_pct = (ma20_end - ma20_start) / ma20_start * 100.0
                uptrend = slope_pct > TREND_SLOPE_MIN_PCT
                downtrend = slope_pct < -TREND_SLOPE_MIN_PCT
        
        if not has_position:
            if uptrend:
                # 상승장 롱: RSI ≤ MAX + 가격이 MA20 이하 (+ 옵션: RSI 상승 전환 시에만, 조정 초반 롱 방지)
                if TREND_UPTREND_LONG_ENABLED and price <= ma_long and rsi_value <= TREND_UPTREND_LONG_RSI_MAX:
                    if TREND_UPTREND_LONG_REQUIRE_RSI_TURNUP:
                        if rsi_prev is not None and rsi_value > rsi_prev:
                            return "long"
                    else:
                        return "long"
                # 상승장 숏: RSI ≥ 80 + 가격이 MA20 이상
                if price >= ma_long and rsi_value >= TREND_UPTREND_SHORT_RSI_MIN:
                    return "short"
            elif downtrend:
                # 하락장 롱: RSI ≤ 20 + 가격이 MA20 이하
                if price <= ma_long and rsi_value <= TREND_DOWNTREND_LONG_RSI_MAX:
                    return "long"
                # 하락장 숏: RSI ≥ MIN + 가격이 MA20 이상 (+ 옵션: RSI 꺾임 시에만, 반등 초반 숏 방지)
                if TREND_DOWNTREND_SHORT_ENABLED and price >= ma_long and rsi_value >= TREND_DOWNTREND_SHORT_RSI_MIN:
                    if TREND_DOWNTREND_SHORT_REQUIRE_RSI_TURNDOWN:
                        if rsi_prev is not None and rsi_value < rsi_prev:
                            return "short"
                    else:
                        return "short"
        return "hold"

    # 중립: 진입 안 함
    if regime == "neutral":
        return "hold"

    # 횡보: 박스는 항상 15분봉(regime_price_history) 기준. 호출부에서 sideways일 때만 넘김.
    if not regime_price_history or len(regime_price_history) < REGIME_LOOKBACK_15M:
        return "hold"
    ph = regime_price_history
    period = REGIME_LOOKBACK_15M

    ma_short = regime_short_ma if regime_short_ma is not None else short_ma
    ma_long = regime_long_ma if regime_long_ma is not None else long_ma

    if not has_position:
        if SIDEWAYS_ENABLED:
            box = _validate_sideways_box(ph, price, period=period)
            if box:
                _, _, _, pos = box
                top = 1.0 - SIDEWAYS_BOX_TOP_MARGIN
                bottom = SIDEWAYS_BOX_BOTTOM_MARGIN
                # 반등 반영 여부 무관: 저점/고점 근처면 진입 (MA 조건 제거)
                if pos <= bottom:
                    return "long"
                if pos >= top:
                    return "short"
        return "hold"

    box = calculate_box_range(ph, period=period)
    if box:
        h, l, r = box
        if r > 0:
            pos = (price - l) / r
            if is_long:
                if pos >= 1.0 - SIDEWAYS_BOX_TOP_MARGIN:
                    return "flat"
            else:
                if pos <= SIDEWAYS_BOX_BOTTOM_MARGIN:
                    return "flat"

    return "hold"


def get_hold_reason(
    regime: MarketRegime,
    rsi_value: float,
    price: float,
    short_ma: float,
    long_ma: float,
    *,
    regime_short_ma: Optional[float] = None,
    regime_long_ma: Optional[float] = None,
    regime_ma_50: Optional[float] = None,
    regime_ma_100: Optional[float] = None,
    regime_price_history: Optional[PriceHistory] = None,
    regime_ma_long_history: Optional[List[float]] = None,
    price_history: Optional[PriceHistory] = None,
) -> str:
    if regime == "trend":
        ma_short = regime_short_ma if regime_short_ma is not None else short_ma
        ma_long = regime_long_ma if regime_long_ma is not None else long_ma
        ma_50 = regime_ma_50 if regime_ma_50 is not None else None
        if ma_short is None or ma_short <= 0 or ma_long is None or ma_long <= 0:
            return "15분봉 MA7/MA20 데이터 부족"
        
        # MA20 기울기로 상승/하락 판단
        uptrend = False
        downtrend = False
        if regime_ma_long_history and len(regime_ma_long_history) >= TREND_SLOPE_BARS:
            recent_ma20 = regime_ma_long_history[-TREND_SLOPE_BARS:]
            ma20_start = recent_ma20[0]
            ma20_end = recent_ma20[-1]
            if ma20_start and ma20_start > 0:
                slope_pct = (ma20_end - ma20_start) / ma20_start * 100.0
                uptrend = slope_pct > TREND_SLOPE_MIN_PCT
                downtrend = slope_pct < -TREND_SLOPE_MIN_PCT
        
        if uptrend:
            # 상승장 롱 조건 체크: 가격이 MA20 이하
            if price > ma_long:
                return f"상승장 롱: 가격이 MA20 초과 (가격 {price:.2f} > MA20 {ma_long:.2f})"
            if rsi_value > TREND_UPTREND_LONG_RSI_MAX:
                return f"상승장 롱: RSI 과매수 (RSI {rsi_value:.0f} > {TREND_UPTREND_LONG_RSI_MAX})"
            # 상승장 숏 조건 체크: 가격이 MA20 이상
            if price < ma_long:
                return f"상승장 숏: 가격이 MA20 미만 (가격 {price:.2f} < MA20 {ma_long:.2f})"
            if rsi_value < TREND_UPTREND_SHORT_RSI_MIN:
                return f"상승장 숏: RSI 과매도 (RSI {rsi_value:.0f} < {TREND_UPTREND_SHORT_RSI_MIN})"
            return "상승장: 가격·RSI 조건 미충족"
        elif downtrend:
            # 하락장 롱 조건 체크: 가격이 MA20 이하
            if price > ma_long:
                return f"하락장 롱: 가격이 MA20 초과 (가격 {price:.2f} > MA20 {ma_long:.2f})"
            if rsi_value > TREND_DOWNTREND_LONG_RSI_MAX:
                return f"하락장 롱: RSI 과매수 (RSI {rsi_value:.0f} > {TREND_DOWNTREND_LONG_RSI_MAX})"
            # 하락장 숏 조건 체크: 가격이 MA20 이상
            if price < ma_long:
                return f"하락장 숏: 가격이 MA20 미만 (가격 {price:.2f} < MA20 {ma_long:.2f})"
            if rsi_value < TREND_DOWNTREND_SHORT_RSI_MIN:
                return f"하락장 숏: RSI 과매도 (RSI {rsi_value:.0f} < {TREND_DOWNTREND_SHORT_RSI_MIN})"
            return "하락장: 가격·RSI 조건 미충족"
        return "MA20 기울기 데이터 부족 또는 추세 방향 불명확"

    if regime == "neutral":
        # 중립 이유 상세 분석 (detect_market_regime 순서와 동일하게)
        reasons = []
        
        # 1) detect_market_regime의 조건: long_ma <= 0 or ma_50 <= 0 or ma_100 <= 0 체크 (가장 먼저)
        regime_long_ma_val = regime_long_ma if regime_long_ma is not None else long_ma
        regime_ma_50_val = regime_ma_50 if regime_ma_50 is not None else 0.0
        regime_ma_100_val = regime_ma_100 if regime_ma_100 is not None else 0.0
        
        if regime_long_ma_val <= 0 or regime_ma_50_val <= 0 or regime_ma_100_val <= 0:
            ma_issues = []
            if regime_long_ma_val <= 0:
                ma_issues.append(f"MA20={regime_long_ma_val:.2f}")
            if regime_ma_50_val <= 0:
                if regime_ma_50 is None:
                    ma_issues.append("MA50=None")
                else:
                    ma_issues.append(f"MA50={regime_ma_50_val:.2f}")
            if regime_ma_100_val <= 0:
                if regime_ma_100 is None:
                    ma_issues.append("MA100=None")
                else:
                    ma_issues.append(f"MA100={regime_ma_100_val:.2f}")
            if ma_issues:
                reasons.append(f"MA 데이터 부족 ({', '.join(ma_issues)})")
            # long_ma/ma_50/ma_100이 0 이하일 때 이것만 반환 (가장 중요한 이유)
            if reasons:
                return f"중립: {', '.join(reasons)}"
        # 2) 추세장 조건 체크 (long_ma/ma_50/ma_100이 모두 양수일 때만)
        if not regime_ma_long_history or len(regime_ma_long_history) < TREND_SLOPE_BARS:
            reasons.append(f"MA20 데이터 부족 ({len(regime_ma_long_history) if regime_ma_long_history else 0}/{TREND_SLOPE_BARS}개)")
        elif regime_ma_long_history and len(regime_ma_long_history) >= TREND_SLOPE_BARS:
            recent_ma20 = regime_ma_long_history[-TREND_SLOPE_BARS:]
            ma20_start = recent_ma20[0]
            ma20_end = recent_ma20[-1]
            if ma20_start and ma20_start > 0:
                slope_pct = (ma20_end - ma20_start) / ma20_start * 100.0
                if abs(slope_pct) < TREND_SLOPE_MIN_PCT:
                    reasons.append(f"MA20 기울기 {slope_pct:+.2f}% (기준 ±{TREND_SLOPE_MIN_PCT}% 미만)")
        # 3) 횡보장 조건 체크 (중립 사유 표시용. 실제 장세 판정은 detect_market_regime에서 직전 봉 기준 박스 사용)
        if regime_price_history and len(regime_price_history) >= REGIME_LOOKBACK_15M:
            bounds = get_sideways_box_bounds(regime_price_history, REGIME_LOOKBACK_15M)
            if not bounds:
                reasons.append("박스 조건 미충족")
            else:
                box_high, box_low = bounds
                box_range = box_high - box_low
                box_range_pct = box_range / box_low * 100 if box_low > 0 else 0
                m = SIDEWAYS_BOX_EXIT_MARGIN_PCT / 100
                exit_high = box_high * (1 + m)
                exit_low = box_low * (1 - m)
                if price > exit_high:
                    reasons.append(f"박스 상단 이탈 (가격={price:.2f} > 이탈선={exit_high:.2f})")
                elif price < exit_low:
                    reasons.append(f"박스 하단 이탈 (가격={price:.2f} < 이탈선={exit_low:.2f})")
                elif box_range_pct < SIDEWAYS_BOX_RANGE_PCT_MIN:
                    reasons.append(f"박스 폭 부족 ({box_range_pct:.2f}% < {SIDEWAYS_BOX_RANGE_PCT_MIN}%)")
                elif box_range < SIDEWAYS_BOX_RANGE_MIN:
                    reasons.append(f"박스 범위 부족 ({box_range:.2f} < {SIDEWAYS_BOX_RANGE_MIN})")
                elif not (box_low <= price <= box_high):
                    reasons.append(f"가격 박스 외부 (가격={price:.2f})")
        elif not regime_price_history or len(regime_price_history) < REGIME_LOOKBACK_15M:
            reasons.append(f"가격 데이터 부족 ({len(regime_price_history) if regime_price_history else 0}/{REGIME_LOOKBACK_15M}개)")
        
        if reasons:
            return f"중립: {', '.join(reasons)}"
        return "중립 (추세·횡보 아님, 진입 없음)"

    if not regime_price_history or len(regime_price_history) < REGIME_LOOKBACK_15M:
        return "박스 계산용 가격 데이터 부족"
    box = _validate_sideways_box(regime_price_history, price, period=REGIME_LOOKBACK_15M)
    if not box:
        return "박스권 조건 미충족"
    box_high, box_low, box_range, pos = box
    if box_range <= 0:
        return "박스권 범위 없음"
    box_str = f" | 박스 하단={box_low:.2f} 상단={box_high:.2f}"
    pos_pct = pos * 100
    return f"횡보장: 가격이 박스 하단/상단 근처 아님( pos={pos_pct:.1f}% ){box_str}"


def get_entry_reason(
    regime: MarketRegime,
    side: str,
    rsi_value: float,
    price: float,
    short_ma: float,
    long_ma: float,
    *,
    regime_short_ma: Optional[float] = None,
    regime_long_ma: Optional[float] = None,
    regime_ma_50: Optional[float] = None,
    regime_ma_100: Optional[float] = None,
    regime_price_history: Optional[PriceHistory] = None,
    regime_ma_long_history: Optional[List[float]] = None,
    price_history: Optional[PriceHistory] = None,
) -> str:
    if regime == "trend":
        ma_short = regime_short_ma if regime_short_ma is not None else short_ma
        ma_long = regime_long_ma if regime_long_ma is not None else long_ma
        ma_50 = regime_ma_50 if regime_ma_50 is not None else None
        
        # MA20 기울기로 상승/하락 판단
        uptrend = False
        downtrend = False
        if regime_ma_long_history and len(regime_ma_long_history) >= TREND_SLOPE_BARS:
            recent_ma20 = regime_ma_long_history[-TREND_SLOPE_BARS:]
            ma20_start = recent_ma20[0]
            ma20_end = recent_ma20[-1]
            if ma20_start and ma20_start > 0:
                slope_pct = (ma20_end - ma20_start) / ma20_start * 100.0
                uptrend = slope_pct > TREND_SLOPE_MIN_PCT
                downtrend = slope_pct < -TREND_SLOPE_MIN_PCT
        
        if uptrend:
            if side == "LONG":
                return f"상승장 롱: 가격 {price:.2f} ≤ MA20 {ma_long:.2f} + RSI {rsi_value:.0f} ≤ {TREND_UPTREND_LONG_RSI_MAX}"
            else:
                return f"상승장 숏: 가격 {price:.2f} ≥ MA20 {ma_long:.2f} + RSI {rsi_value:.0f} ≥ {TREND_UPTREND_SHORT_RSI_MIN}"
        elif downtrend:
            if side == "LONG":
                return f"하락장 롱: 가격 {price:.2f} ≤ MA20 {ma_long:.2f} + RSI {rsi_value:.0f} ≤ {TREND_DOWNTREND_LONG_RSI_MAX}"
            else:
                return f"하락장 숏: 가격 {price:.2f} ≥ MA20 {ma_long:.2f} + RSI {rsi_value:.0f} ≥ {TREND_DOWNTREND_SHORT_RSI_MIN}"
        return f"추세장 {side}: 가격 {price:.2f} + RSI {rsi_value:.0f}"

    if regime == "neutral":
        return "중립 (진입 없음)"

    if not regime_price_history or len(regime_price_history) < REGIME_LOOKBACK_15M:
        return f"횡보장 {side} (박스)"
    box = _validate_sideways_box(regime_price_history, price, period=REGIME_LOOKBACK_15M)
    if not box:
        return f"횡보장 {side} (박스)"
    _, box_low, box_range, pos = box
    if box_range <= 0:
        return f"횡보장 {side} (박스)"
    pos_pct = pos * 100
    ma_short = regime_short_ma if regime_short_ma is not None else short_ma
    ma_long = regime_long_ma if regime_long_ma is not None else long_ma
    if side == "LONG":
        return "박스 하단 근처(저점) 진입"
    return "박스 상단 근처(고점) 진입"
