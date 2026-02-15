"""실거래 전용 전략 로직. 수정 시 페이퍼/백테스트에 영향 없음.

횡보장: 박스 = 간격 2시간 이상인 고가 2개→상단, 저가 2개→하단. 기울기 0.5% 이내·상하단 비슷. 터치 조건 없음.
진입: 박스 하단 4% 이내 롱, 상단 4% 이내 숏. MA 조건 없음.
"""

from typing import Literal, Optional, List, Union, Tuple

PriceHistory = Union[List[float], List[Tuple[float, float, float]]]

from config_live import (
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
    TREND_PULLBACK_MA_PCT,
    TREND_RSI_LONG_MAX,
    TREND_RSI_SHORT_MIN,
    TREND_MA50_MA100_FILTER,
)

Signal = Literal["long", "short", "flat", "hold"]
MarketRegime = Literal["sideways", "neutral"]

REGIME_KR = {"sideways": "횡보장", "neutral": "추세장"}
BOX_TOUCH_THRESHOLD = 0.012


def _recent_hlc(recent: list) -> Tuple[List[float], List[float], List[float]]:
    if not recent:
        return ([], [], [])
    if isinstance(recent[0], (list, tuple)) and len(recent[0]) >= 3:
        return ([x[0] for x in recent], [x[1] for x in recent], [x[2] for x in recent])
    closes = list(recent)
    return (closes, closes, closes)


def _min_gap_candles(period: int) -> int:
    mins_per_candle = 15 if period == REGIME_LOOKBACK_15M else 5
    return max(1, int(SIDEWAYS_BOX_TOUCH_GAP_MIN_HOURS * 60 / mins_per_candle))


def _box_high_low_from_two_points(
    highs: List[float], lows: List[float], period: int
) -> Optional[Tuple[float, float]]:
    """간격 2시간 이상인 가장 높은 가격 2개 → 상단, 가장 낮은 가격 2개 → 하단. (box_high, box_low) 또는 None."""
    if not highs or not lows or len(highs) != len(lows):
        return None
    n = len(highs)
    gap = _min_gap_candles(period)
    if n < gap + 1:
        return None
    i_top1 = max(range(n), key=lambda i: highs[i])
    candidates_top2 = [j for j in range(n) if abs(j - i_top1) >= gap]
    if not candidates_top2:
        return None
    i_top2 = max(candidates_top2, key=lambda j: highs[j])
    box_high = max(highs[i_top1], highs[i_top2])
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
) -> MarketRegime:
    if long_ma <= 0 or ma_50 <= 0 or ma_100 <= 0:
        return "neutral"

    period = box_period or SIDEWAYS_BOX_PERIOD
    if price_history and len(price_history) >= period:
        recent = price_history[-period:]
        highs, lows, _ = _recent_hlc(recent)
        pair = _box_high_low_from_two_points(highs, lows, period)
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
    macd_line: Optional[float] = None,
    macd_signal: Optional[float] = None,
) -> Signal:
    if regime == "neutral":
        ma_short = regime_short_ma if regime_short_ma is not None else short_ma
        ma_long = regime_long_ma if regime_long_ma is not None else long_ma
        ma_50 = regime_ma_50 if regime_ma_50 is not None else None
        ma_100 = regime_ma_100 if regime_ma_100 is not None else None
        if ma_short is None or ma_long is None or ma_long <= 0:
            return "hold"
        uptrend = ma_short > ma_long
        if TREND_MA50_MA100_FILTER and ma_50 is not None and ma_100 is not None:
            uptrend = uptrend and ma_50 > ma_100
            downtrend = ma_short < ma_long and ma_50 < ma_100
        else:
            downtrend = ma_short < ma_long
        pullback_pct = TREND_PULLBACK_MA_PCT / 100
        if not has_position:
            if uptrend:
                if price <= ma_short * (1 + pullback_pct) and rsi_value <= TREND_RSI_LONG_MAX:
                    return "long"
            if downtrend:
                if price >= ma_short * (1 - pullback_pct) and rsi_value >= TREND_RSI_SHORT_MIN:
                    return "short"
        else:
            if is_long and downtrend:
                return "flat"
            if not is_long and uptrend:
                return "flat"
        return "hold"

    ma_short = regime_short_ma if regime_short_ma is not None else short_ma
    ma_long = regime_long_ma if regime_long_ma is not None else long_ma
    ph = regime_price_history if regime_price_history else price_history

    if not has_position:
        if SIDEWAYS_ENABLED:
            box = _validate_sideways_box(ph, price, period=REGIME_LOOKBACK_15M if regime_price_history else SIDEWAYS_BOX_PERIOD)
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

    box = calculate_box_range(ph or [], period=REGIME_LOOKBACK_15M if regime_price_history else SIDEWAYS_BOX_PERIOD)
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
    price_history: Optional[PriceHistory] = None,
) -> str:
    if regime == "neutral":
        ma_short = regime_short_ma if regime_short_ma is not None else short_ma
        ma_long = regime_long_ma if regime_long_ma is not None else long_ma
        ma_50 = regime_ma_50
        ma_100 = regime_ma_100
        if ma_short is None or ma_long is None or ma_long <= 0:
            return "15분봉 MA 데이터 부족"
        uptrend = ma_short > ma_long
        if TREND_MA50_MA100_FILTER and ma_50 is not None and ma_100 is not None:
            uptrend = uptrend and ma_50 > ma_100
            downtrend = ma_short < ma_long and ma_50 < ma_100
        else:
            downtrend = ma_short < ma_long
        pullback_pct = TREND_PULLBACK_MA_PCT / 100
        if uptrend:
            if price > ma_short * (1 + pullback_pct):
                return f"상승추세 but 가격이 MA7 풀백 구간 아님 (가격 {price:.2f} > MA7×1.01)"
            if rsi_value > TREND_RSI_LONG_MAX:
                return f"상승추세 but RSI 과매수 (RSI {rsi_value:.0f} > {TREND_RSI_LONG_MAX})"
            return "상승추세 but 풀백·RSI 조건 미충족"
        if downtrend:
            if price < ma_short * (1 - pullback_pct):
                return f"하락추세 but 가격이 MA7 풀백 구간 아님 (가격 {price:.2f} < MA7×0.99)"
            if rsi_value < TREND_RSI_SHORT_MIN:
                return f"하락추세 but RSI 과매도 (RSI {rsi_value:.0f} < {TREND_RSI_SHORT_MIN})"
            return "하락추세 but 풀백·RSI 조건 미충족"
        return "상승·하락 추세 아님 (MA7/MA20 또는 MA50/MA100 조건 미충족)"

    ph = regime_price_history if regime_price_history else price_history
    if not ph or len(ph) < (REGIME_LOOKBACK_15M if regime_price_history else SIDEWAYS_BOX_PERIOD):
        return "박스 계산용 가격 데이터 부족"
    box = _validate_sideways_box(ph, price, period=REGIME_LOOKBACK_15M if regime_price_history else SIDEWAYS_BOX_PERIOD)
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
    price_history: Optional[PriceHistory] = None,
) -> str:
    if regime == "neutral":
        ma_short = regime_short_ma if regime_short_ma is not None else short_ma
        if side == "LONG":
            return f"상승추세 + MA7 풀백(가격 {price:.2f} ≤ MA7×1.01) + RSI {rsi_value:.0f}(≤{TREND_RSI_LONG_MAX})"
        return f"하락추세 + MA7 풀백(가격 {price:.2f} ≥ MA7×0.99) + RSI {rsi_value:.0f}(≥{TREND_RSI_SHORT_MIN})"

    ph = regime_price_history if regime_price_history else price_history
    if not ph:
        return f"횡보장 {side} (박스)"
    box = _validate_sideways_box(ph, price, period=REGIME_LOOKBACK_15M if regime_price_history else SIDEWAYS_BOX_PERIOD)
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
