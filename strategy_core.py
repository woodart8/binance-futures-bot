"""
전략 로직 모듈.

5분봉 기준, 레버리지 3배, 재산의 1/3로만 매매
"""

from dataclasses import dataclass
from typing import Literal, Optional, List

from config import (
    MA_SHORT_PERIOD,
    MA_LONG_PERIOD,
    SIDEWAYS_BOX_PERIOD,
    SIDEWAYS_BOX_TOP_MARGIN,
    SIDEWAYS_BOX_BOTTOM_MARGIN,
)


Signal = Literal["long", "short", "flat", "hold"]
MarketRegime = Literal["bullish", "bearish", "sideways"]


@dataclass
class MovingAverageParams:
    short_period: int = MA_SHORT_PERIOD  # 단기 이동평균선 7
    long_period: int = MA_LONG_PERIOD  # 장기 이동평균선 20
    trend_threshold: float = 0.01  # 추세 판단 임계값 (1%)


def detect_market_regime(
    short_ma: float,
    long_ma: float,
    price: float,
    ma_50: float = 0.0,
    ma_100: float = 0.0,
    params: MovingAverageParams = MovingAverageParams(),
    price_history: Optional[List[float]] = None,
    box_period: Optional[int] = None,
) -> MarketRegime:
    """
    시장 상태 판단 (강세장/약세장/횡보장).
    
    MA 7, 20, 50, 100을 기준으로 판단.
    - 정배열 (MA 7 > MA 20 > MA 50 > MA 100): 상승장
    - 역배열 (MA 7 < MA 20 < MA 50 < MA 100): 하락장
    - 횡보장: 이동평균선이 뒤섞여있음 + 상단 저항과 하단 지지가 강해 위 아래로 반복적으로 왔다갔다 하는 경우
    
    :param short_ma: 단기 이동평균선 값 (MA 7)
    :param long_ma: 장기 이동평균선 값 (MA 20)
    :param price: 현재 가격
    :param ma_50: 중기 이동평균선 값 (MA 50)
    :param ma_100: 장기 이동평균선 값 (MA 100)
    :param params: 이동평균선 파라미터 (사용하지 않지만 호환성을 위해 유지)
    :param price_history: 가격 히스토리 (횡보장 판단용)
    :param box_period: 횡보장 박스권 기간 (None이면 SIDEWAYS_BOX_PERIOD)
    :return: 시장 상태 ("bullish", "bearish", "sideways")
    """
    # MA 값이 유효하지 않으면 횡보장으로 판단
    if long_ma == 0:
        return "sideways"
    
    # MA 50, 100이 없으면 기존 방식 사용 (MA 7, 20만)
    if ma_50 == 0 or ma_100 == 0:
        if short_ma > long_ma:
            return "bullish"
        else:
            return "bearish"
    
    # 정배열 확인: MA 7 > MA 20 > MA 50 > MA 100
    is_bullish_alignment = (short_ma > long_ma and 
                           long_ma > ma_50 and 
                           ma_50 > ma_100)
    
    # 역배열 확인: MA 7 < MA 20 < MA 50 < MA 100
    is_bearish_alignment = (short_ma < long_ma and 
                           long_ma < ma_50 and 
                           ma_50 < ma_100)
    
    # 가격 추세도 고려 (MA가 지연 지표이므로)
    # 가격이 MA 아래에 있으면 하락 추세로 판단
    price_below_ma = (price < short_ma and price < long_ma)
    
    # 최근 가격 추세 확인 (더 긴 기간)
    price_trend_bearish = False
    if price_history and len(price_history) >= 100:
        recent_prices = price_history[-100:]  # 최근 100개 캔들 (약 8시간)
        first_price = recent_prices[0]
        last_price = recent_prices[-1]
        if first_price > 0:
            price_change_pct = ((last_price - first_price) / first_price) * 100
            # 가격이 2.5% 이상 하락하면 하락 추세로 판단 (3→2.5% 완화)
            if price_change_pct < -2.5:
                price_trend_bearish = True
    
    # MA 정배열이어도 가격이 MA 아래에 있거나 가격 하락 추세면 하락장으로 판단
    if is_bullish_alignment and not price_below_ma and not price_trend_bearish:
        return "bullish"  # 상승장 (정배열 + 가격이 MA 위 + 가격 하락 추세 없음)
    elif is_bearish_alignment or price_below_ma or price_trend_bearish:
        return "bearish"  # 하락장 (역배열 또는 가격이 MA 아래 또는 가격 하락 추세)
    else:
        # 이동평균선이 뒤섞여있음 - 횡보장 후보
        # 추가 확인: 상단 저항과 하단 지지가 강해 위 아래로 반복적으로 왔다갔다 하는지 확인
        period = box_period if box_period is not None else SIDEWAYS_BOX_PERIOD
        if price_history and len(price_history) >= period:
            box_info = calculate_box_range(price_history, period)
            if box_info:
                box_high, box_low, box_range = box_info
                
                # 박스권 범위가 충분히 큰지 확인 (너무 작으면 횡보장이 아님)
                if box_range > 0:
                    box_range_pct = box_range / box_low * 100
                    # 박스권 범위가 1% 이상이어야 횡보장으로 인정
                    if box_range_pct >= 1.0:
                        # 가격이 박스권 내에 있는지 확인
                        if box_low <= price <= box_high:
                            # 최근 가격이 상단과 하단을 여러 번 터치했는지 확인
                            recent_prices = price_history[-period:]
                            top_touches = sum(1 for p in recent_prices if abs(p - box_high) / box_high < 0.01)  # 상단 1% 범위
                            bottom_touches = sum(1 for p in recent_prices if abs(p - box_low) / box_low < 0.01)  # 하단 1% 범위
                            
                            # 상단과 하단을 각각 최소 2번 이상 터치했으면 횡보장
                            if top_touches >= 2 and bottom_touches >= 2:
                                return "sideways"  # 횡보장
        
        # 명확하게 상승장/하락장/횡보장이 아닌 경우 관망
        # 이동평균선이 뒤섞여있지만 횡보장 조건을 만족하지 않으면 거래하지 않음
        return "sideways"  # 관망 (횡보장으로 분류하되 실제로는 거래하지 않음)


def detect_sideways_bias(price_history: Optional[List[float]] = None) -> str:
    """
    횡보장 내 추세 판단: 저점 하락 → 숏 위주, 고점 상승 → 롱 위주.
    검증 기간: 1일 (SIDEWAYS_BOX_PERIOD)
    
    :param price_history: 가격 히스토리
    :return: "short_bias" (저점 하락), "long_bias" (고점 상승), "neutral"
    """
    if not price_history or len(price_history) < SIDEWAYS_BOX_PERIOD:
        return "neutral"
    n = SIDEWAYS_BOX_PERIOD
    recent = price_history[-n:]
    first_half = recent[: n // 2]
    second_half = recent[n // 2 :]
    min_first = min(first_half)
    min_second = min(second_half)
    max_first = max(first_half)
    max_second = max(second_half)
    # 저점이 0.5% 이상 하락 → 숏 위주
    lower_lows = min_second < min_first * 0.995 if min_first > 0 else False
    # 고점이 0.5% 이상 상승 → 롱 위주
    higher_highs = max_second > max_first * 1.005 if max_first > 0 else False
    if lower_lows and not higher_highs:
        return "short_bias"
    if higher_highs and not lower_lows:
        return "long_bias"
    return "neutral"


def calculate_box_range(price_history: List[float], period: int = SIDEWAYS_BOX_PERIOD) -> Optional[tuple]:
    """
    박스권 상단/하단 계산.
    
    :param price_history: 가격 히스토리 리스트
    :param period: 분석 기간 (기본값: 288개 5분봉 = 1일)
    :return: (box_high, box_low, box_range) 또는 None
    """
    if not price_history or len(price_history) < period:
        return None
    
    recent_prices = price_history[-period:]
    box_high = max(recent_prices)
    box_low = min(recent_prices)
    box_range = box_high - box_low
    
    if box_range <= 0:
        return None
    
    return (box_high, box_low, box_range)


def swing_strategy_signal(
    rsi_value: float,
    price: float,
    short_ma: float,
    long_ma: float,
    has_position: bool,
    is_long: bool,
    regime: MarketRegime,
    price_history: Optional[List[float]] = None,
) -> Signal:
    """
    RSI + MA 기반 스윙 전략 시그널 생성.
    
    강세장: 롱 포지션만 잡고, 스윙 방식으로 매매
    약세장: 숏 포지션만 잡고, 스윙 방식으로 매매
    횡보장: 롱과 숏 포지션 둘다 잡는데, 박스권 상단/하단 기반
    
    :param rsi_value: RSI 값
    :param price: 현재 가격
    :param short_ma: 단기 이동평균선 (MA 7)
    :param long_ma: 장기 이동평균선 (MA 20)
    :param has_position: 포지션 보유 여부
    :param is_long: 롱 포지션 여부
    :param regime: 시장 상태
    :param price_history: 가격 히스토리 (박스권 계산용)
    :return: 시그널 ("long", "short", "flat", "hold")
    """
    if not has_position:
        # 진입 조건 (RSI + MA만 사용)
        if regime == "bullish":
            # 강세장: 롱만 진입
            # 명확한 상승장이므로 RSI 기준 완화: RSI <= 50 (이전 40)
            # 상승장에서는 RSI가 낮을 때 (조정 후 반등) 진입
            # RSI <= 50 + 가격이 MA 7 위 + 가격이 MA 20 위
            if rsi_value <= 50 and price > short_ma and price > long_ma:
                return "long"
        
        elif regime == "bearish":
            # 약세장: 숏만 진입
            # 진입 조건: RSI >= 51 (강세 RSI 50과 대칭)
            # RSI >= 51 + 가격이 MA 7 아래 + 가격이 MA 20 아래
            if rsi_value >= 51 and price < short_ma and price < long_ma:
                return "short"
        
        elif regime == "sideways":
            # 횡보장: 박스권 + 저점/고점 추세에 따른 편향
            # 저점 하락 → 숏 위주, 고점 상승 → 롱 위주
            bias = detect_sideways_bias(price_history)
            box_info = calculate_box_range(price_history, SIDEWAYS_BOX_PERIOD)
            if box_info:
                box_high, box_low, box_range = box_info
                
                # 박스권 범위가 충분히 크고
                if box_range > 0:
                    box_range_pct = box_range / box_low * 100
                    if box_range_pct >= 1.0 and box_low <= price <= box_high:
                        recent_prices = price_history[-SIDEWAYS_BOX_PERIOD:] if price_history else []
                        top_touches = sum(1 for p in recent_prices if abs(p - box_high) / box_high < 0.01)
                        bottom_touches = sum(1 for p in recent_prices if abs(p - box_low) / box_low < 0.01)
                        price_position = (price - box_low) / box_range
                        
                        # 숏 위주 (저점 하락): 상단 50% 구간에서 숏, 하단에서만 롱
                        if bias == "short_bias":
                            if top_touches >= 1 and bottom_touches >= 1:
                                if price_position >= 0.5 and rsi_value >= 52:
                                    return "short"
                                if price_position <= (0.0 + SIDEWAYS_BOX_BOTTOM_MARGIN) and rsi_value <= 35:
                                    return "long"
                        
                        # 롱 위주 (고점 상승): 하단 50% 구간에서 롱, 상단에서만 숏
                        elif bias == "long_bias":
                            if top_touches >= 1 and bottom_touches >= 1:
                                if price_position <= 0.5 and rsi_value <= 48:
                                    return "long"
                                if price_position >= (1.0 - SIDEWAYS_BOX_TOP_MARGIN) and rsi_value >= 65:
                                    return "short"
                        
                        # 중립: 박스권 상단/하단 진입 (터치 1회 이상)
                        elif top_touches >= 1 and bottom_touches >= 1:
                            if price_position <= (0.0 + SIDEWAYS_BOX_BOTTOM_MARGIN) and rsi_value <= 35:
                                return "long"
                            if price_position >= (1.0 - SIDEWAYS_BOX_TOP_MARGIN) and rsi_value >= 65:
                                return "short"
            
            return "hold"
    
    else:
        # 청산 조건 (RSI + MA만 사용)
        # 실제 청산은 backtest.py의 손절/익절/스탑로스 로직에서 처리됨
        # 여기서는 추가적인 시그널 기반 청산만 제공 (선택적)
        
        if regime == "bullish":
            # 강세장 롱 청산 (단타): RSI 과매수 또는 가격이 MA 7 아래로 떨어짐
            if is_long:
                if rsi_value >= 65 or price < short_ma:
                    return "flat"
        
        elif regime == "bearish":
            # 약세장 숏 청산 (단타): RSI 과매도 또는 가격이 MA 7 위로 올라감
            if not is_long:
                if rsi_value <= 35 or price > short_ma:
                    return "flat"
        
        elif regime == "sideways":
            # 횡보장 청산 조건 (박스권 기반)
            box_info = calculate_box_range(price_history, SIDEWAYS_BOX_PERIOD)
            if box_info:
                box_high, box_low, box_range = box_info
                price_position = (price - box_low) / box_range
                
                if is_long:
                    # 롱 청산: 상단 근처에서 익절 또는 하단 이탈 시 손절
                    if price_position >= (1.0 - SIDEWAYS_BOX_TOP_MARGIN):
                        return "flat"  # 상단 근처 익절
                    if price_position <= (0.0 - SIDEWAYS_BOX_BOTTOM_MARGIN * 0.5):
                        return "flat"  # 하단 이탈 손절
                else:
                    # 숏 청산: 하단 근처에서 익절 또는 상단 이탈 시 손절
                    if price_position <= (0.0 + SIDEWAYS_BOX_BOTTOM_MARGIN):
                        return "flat"  # 하단 근처 익절
                    if price_position >= (1.0 + SIDEWAYS_BOX_TOP_MARGIN * 0.5):
                        return "flat"  # 상단 이탈 손절
    
    return "hold"


