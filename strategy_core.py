"""횡보장 박스: MA7>MA20 하단 롱 / MA7<MA20 상단 숏."""

from typing import Literal, Optional, List

from config import (
    MA_SHORT_PERIOD,
    MA_LONG_PERIOD,
    REGIME_LOOKBACK_15M,
    SIDEWAYS_BOX_PERIOD,
    SIDEWAYS_BOX_TOP_MARGIN,
    SIDEWAYS_BOX_BOTTOM_MARGIN,
    SIDEWAYS_ENABLED,
    SIDEWAYS_MIN_TOUCHES,
    SIDEWAYS_BOX_RANGE_PCT_MIN,
    TREND_PULLBACK_MA_PCT,
    TREND_RSI_LONG_MAX,
    TREND_RSI_SHORT_MIN,
    TREND_MA50_MA100_FILTER,
)

Signal = Literal["long", "short", "flat", "hold"]
MarketRegime = Literal["sideways", "neutral"]

REGIME_KR = {"sideways": "횡보장", "neutral": "추세장"}
BOX_TOUCH_THRESHOLD = 0.012  # 1.2% 이내 = 터치


def detect_market_regime(
    short_ma: float,
    long_ma: float,
    price: float,
    ma_50: float = 0.0,
    ma_100: float = 0.0,
    price_history: Optional[List[float]] = None,
    box_period: Optional[int] = None,
) -> MarketRegime:
    """박스 조건 충족→sideways, 미충족→neutral(거래 쉼)."""
    if long_ma <= 0 or ma_50 <= 0 or ma_100 <= 0:
        return "neutral"

    period = box_period or SIDEWAYS_BOX_PERIOD
    if price_history and len(price_history) >= period:
        box_high = max(price_history[-period:])
        box_low = min(price_history[-period:])
        box_range = box_high - box_low
        if box_range > 0:
            pct = box_range / box_low * 100
            if pct >= SIDEWAYS_BOX_RANGE_PCT_MIN and box_low <= price <= box_high:
                recent = price_history[-period:]
                top_touches = sum(1 for p in recent if abs(p - box_high) / box_high < BOX_TOUCH_THRESHOLD)
                bottom_touches = sum(1 for p in recent if abs(p - box_low) / box_low < BOX_TOUCH_THRESHOLD)
                if top_touches >= SIDEWAYS_MIN_TOUCHES and bottom_touches >= SIDEWAYS_MIN_TOUCHES:
                    return "sideways"

    return "neutral"


def _validate_sideways_box(
    price_history: Optional[List[float]], price: float, period: Optional[int] = None
) -> Optional[tuple]:
    """박스 검증. 반환: (box_high, box_low, box_range, price_position)."""
    p = period or SIDEWAYS_BOX_PERIOD
    if not price_history or len(price_history) < p:
        return None
    recent = price_history[-p:]
    box_high = max(recent)
    box_low = min(recent)
    box_range = box_high - box_low
    if box_range <= 0:
        return None
    pct = box_range / box_low * 100
    if pct < SIDEWAYS_BOX_RANGE_PCT_MIN or not (box_low <= price <= box_high):
        return None
    top_touches = sum(1 for p in recent if abs(p - box_high) / box_high < BOX_TOUCH_THRESHOLD)
    bottom_touches = sum(1 for p in recent if abs(p - box_low) / box_low < BOX_TOUCH_THRESHOLD)
    if top_touches < SIDEWAYS_MIN_TOUCHES or bottom_touches < SIDEWAYS_MIN_TOUCHES:
        return None
    pos = (price - box_low) / box_range
    return (box_high, box_low, box_range, pos)


def calculate_box_range(
    price_history: List[float], period: int = SIDEWAYS_BOX_PERIOD
) -> Optional[tuple]:
    if not price_history or len(price_history) < period:
        return None
    recent = price_history[-period:]
    h, l = max(recent), min(recent)
    if h <= l:
        return None
    return (h, l, h - l)


def swing_strategy_signal(
    rsi_value: float,
    price: float,
    short_ma: float,
    long_ma: float,
    has_position: bool,
    is_long: bool,
    regime: MarketRegime,
    price_history: Optional[List[float]] = None,
    rsi_prev: Optional[float] = None,
    open_prev: Optional[float] = None,
    close_prev: Optional[float] = None,
    open_curr: Optional[float] = None,
    *,
    regime_short_ma: Optional[float] = None,
    regime_long_ma: Optional[float] = None,
    regime_ma_50: Optional[float] = None,
    regime_ma_100: Optional[float] = None,
    regime_price_history: Optional[List[float]] = None,
    macd_line: Optional[float] = None,
    macd_signal: Optional[float] = None,
) -> Signal:
    """neutral=추세 추종 단타(상승추세 롱 풀백, 하락추세 숏 풀백). sideways=박스 하단 롱 / 상단 숏."""
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
                if ma_short > ma_long and pos <= bottom:
                    return "long"
                if ma_short < ma_long and pos >= top:
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
    regime_price_history: Optional[List[float]] = None,
    price_history: Optional[List[float]] = None,
) -> str:
    """진입하지 않은 이유(hold)를 구체적으로 반환. 로그용."""
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
        return "박스권 아님 또는 상·하단 터치 부족(2회 미만)"
    _, box_low, box_range, pos = box
    if box_range <= 0:
        return "박스권 범위 없음"
    pos_pct = pos * 100
    top = (1.0 - SIDEWAYS_BOX_TOP_MARGIN) * 100
    bottom = SIDEWAYS_BOX_BOTTOM_MARGIN * 100
    ma_short = regime_short_ma if regime_short_ma is not None else short_ma
    ma_long = regime_long_ma if regime_long_ma is not None else long_ma
    if ma_short > ma_long:
        if pos_pct > bottom:
            return "횡보장 롱: 가격이 박스 하단 근처 아님"
        return "횡보장 롱: MA 정배열 but 하단 조건 미충족"
    if ma_short < ma_long:
        if pos_pct < top:
            return "횡보장 숏: 가격이 박스 상단 근처 아님"
        return "횡보장 숏: MA 역배열 but 상단 조건 미충족"
    return "횡보장: MA7·MA20 크로스 구간(명확한 정/역배열 아님)"


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
    regime_price_history: Optional[List[float]] = None,
    price_history: Optional[List[float]] = None,
) -> str:
    """진입 이유를 구체적으로 반환. 로그용. side는 'LONG' 또는 'SHORT'."""
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
        return "박스 하단 근처 + MA 정배열(MA7 > MA20)"
    return "박스 상단 근처 + MA 역배열(MA7 < MA20)"
