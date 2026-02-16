"""페이퍼/백테스트 전용 청산 로직. 수정 시 라이브에 영향 없음.

박스권 이탈: 가격이 진입 시점 box_high/box_low에서 SIDEWAYS_BOX_EXIT_MARGIN_PCT(0.5%) 벗어나면
상단 이탈(숏 청산) 또는 하단 이탈(롱 청산). 가격 기준이며 ROE 아님.
"""

from typing import Optional

from config_paper import (
    SIDEWAYS_STOP_LOSS,
    SIDEWAYS_PROFIT_TARGET,
    SIDEWAYS_STOP_LOSS_PRICE,
    SIDEWAYS_BOX_EXIT_MARGIN_PCT,
    TREND_STOP_LOSS,
    TREND_PROFIT_TARGET,
    TREND_STOP_LOSS_PRICE,
    LEVERAGE,
)

EXIT_REASON_DISPLAY = {
    "횡보_익절": "횡보 익절",
    "추세_익절": "추세 익절",
    "박스권_하단이탈": "박스권 하단 이탈(0.5% 돌파)",
    "박스권_상단이탈": "박스권 상단 이탈(0.5% 돌파)",
    "손절_횡보": "손절",
    "손절_추세": "손절",
    "스탑로스_횡보": "스탑로스",
    "스탑로스_추세": "스탑로스",
}


def _check_exit(
    is_long: bool,
    pnl_pct: float,
    price: float,
    entry_price: float,
    box_high: float,
    box_low: float,
    regime: str,
) -> Optional[str]:
    is_trend = regime == "trend"
    sl = TREND_STOP_LOSS if is_trend else SIDEWAYS_STOP_LOSS
    tp = TREND_PROFIT_TARGET if is_trend else SIDEWAYS_PROFIT_TARGET
    sl_price = TREND_STOP_LOSS_PRICE if is_trend else SIDEWAYS_STOP_LOSS_PRICE
    reason_suffix = "_추세" if is_trend else "_횡보"

    if pnl_pct <= -sl:
        return f"손절{reason_suffix}"
    stop_pct = sl_price / 100 / LEVERAGE
    if is_long:
        stop = entry_price * (1 - stop_pct)
        if price <= stop:
            return f"스탑로스{reason_suffix}"
        if not is_trend and box_high > box_low:
            m = SIDEWAYS_BOX_EXIT_MARGIN_PCT / 100  # 0.5%: 하단 대비 가격이 m 아래면 이탈
            if price < box_low * (1 - m):
                return "박스권_하단이탈"
    else:
        stop = entry_price * (1 + stop_pct)
        if price >= stop:
            return f"스탑로스{reason_suffix}"
        if not is_trend and box_high > box_low:
            m = SIDEWAYS_BOX_EXIT_MARGIN_PCT / 100  # 0.5%: 상단 대비 가격이 m 위면 이탈
            if price > box_high * (1 + m):
                return "박스권_상단이탈"
    if pnl_pct >= tp:
        return "추세_익절" if is_trend else "횡보_익절"
    return None


def check_long_exit(
    regime: str,
    pnl_pct: float,
    price: float,
    entry_price: float,
    best_pnl_pct: float,
    box_high: float = 0.0,
    box_low: float = 0.0,
) -> Optional[str]:
    return _check_exit(True, pnl_pct, price, entry_price, box_high, box_low, regime)


def check_short_exit(
    regime: str,
    pnl_pct: float,
    price: float,
    entry_price: float,
    best_pnl_pct: float,
    box_high: float = 0.0,
    box_low: float = 0.0,
) -> Optional[str]:
    return _check_exit(False, pnl_pct, price, entry_price, box_high, box_low, regime)


def reason_to_display_message(reason: str, is_long: bool) -> str:
    return EXIT_REASON_DISPLAY.get(reason, reason)
