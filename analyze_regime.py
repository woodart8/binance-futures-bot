"""
최근 24시간 데이터로 현재 장세(횡보/추세/중립) 판정 이유를 단계별로 분석.
실거래/모의투자와 동일하게 1분봉 수집 후 5분봉 리샘플 사용.
실행: python analyze_regime.py
"""

import sys
import config_paper
sys.modules["config"] = config_paper
import strategy_core_paper
sys.modules["strategy_core"] = strategy_core_paper

from exchange_client import get_public_exchange
from data import fetch_ohlcv_1m_min_bars, resample_1m_to_5m, compute_regime_15m
from config_common import (
    REGIME_LOOKBACK_15M,
    TREND_SLOPE_BARS,
    TREND_SLOPE_MIN_PCT,
    SIDEWAYS_BOX_RANGE_PCT_MIN,
    SIDEWAYS_BOX_RANGE_MIN,
    SIDEWAYS_BOX_EXIT_MARGIN_PCT,
)
from strategy_core_paper import (
    _recent_hlc,
    _box_high_low_from_two_points,
)


def main():
    exchange = get_public_exchange()
    # 15분봉 96개+MA100 확보: 15m 196개 이상 → 5m 588개 → 1m 2940개 이상 필요
    df_1m = fetch_ohlcv_1m_min_bars(exchange, min_bars=3000)
    if df_1m.empty or len(df_1m) < 2:
        print("1분봉 데이터 부족")
        return

    df_5m = resample_1m_to_5m(df_1m)
    if df_5m.empty or len(df_5m) < REGIME_LOOKBACK_15M:
        print("5분봉 리샘플 후 데이터 부족:", len(df_5m), "개 (15분봉 96개 확보 필요)")
        return

    current_price = float(df_5m["close"].iloc[-1])
    regime, short_ma, long_ma, ma_50, ma_100, price_history_15m, rsi_15m, _, ma_long_history = compute_regime_15m(
        df_5m, current_price
    )

    if not price_history_15m or len(price_history_15m) < REGIME_LOOKBACK_15M:
        print("15분봉 price_history 부족:", len(price_history_15m or []), "개 (필요:", REGIME_LOOKBACK_15M, ")")
        print("-> 1분봉 3000개 이상 수집 필요 (현재 5분봉", len(df_5m), "개 -> 15분봉 유효 봉 부족)")
        return

    period = REGIME_LOOKBACK_15M

    # ----- 1) 추세장 조건 -----
    trend_ok = False
    slope_pct = None
    if ma_long_history and len(ma_long_history) >= TREND_SLOPE_BARS:
        recent_ma20 = ma_long_history[-TREND_SLOPE_BARS:]
        ma20_start = recent_ma20[0]
        ma20_end = recent_ma20[-1]
        if ma20_start and ma20_start > 0:
            slope_pct = (ma20_end - ma20_start) / ma20_start * 100.0
            trend_ok = abs(slope_pct) >= TREND_SLOPE_MIN_PCT

    # ----- 2) 횡보장 박스 (detect_market_regime과 동일: 직전 봉까지로 박스 계산) -----
    if len(price_history_15m) >= period + 1:
        recent_for_box = price_history_15m[-(period + 1) : -1]
    else:
        recent_for_box = price_history_15m[-period:]
    box_len = len(recent_for_box)
    highs, lows, _ = _recent_hlc(recent_for_box)
    pair = _box_high_low_from_two_points(highs, lows, box_len) if highs and lows else None

    box_high = box_low = box_range_pct = None
    price_in_box = False
    exit_margin_breach = False
    if pair:
        box_high, box_low = pair
        box_range = box_high - box_low
        if box_low > 0:
            box_range_pct = box_range / box_low * 100
        price_in_box = box_low <= current_price <= box_high
        m = SIDEWAYS_BOX_EXIT_MARGIN_PCT / 100
        exit_margin_breach = (box_high is not None and current_price > box_high * (1 + m)) or (
            box_low is not None and current_price < box_low * (1 - m)
        )

    sideways_ok = (
        pair is not None
        and long_ma > 0
        and ma_50 > 0
        and ma_100 > 0
        and box_range_pct is not None
        and box_range_pct >= SIDEWAYS_BOX_RANGE_PCT_MIN
        and (box_high - box_low) >= SIDEWAYS_BOX_RANGE_MIN
        and price_in_box
        and not exit_margin_breach
    )

    # ----- 출력 -----
    print("=" * 60)
    print("중립(neutral) 나오는 이유 테스트 - 최근 24시간(15분봉 96개) 기준")
    print("=" * 60)
    print(f"현재가: {current_price:,.2f}")
    print(f"5분봉 개수: {len(df_5m)}, 15분봉 price_history: {len(price_history_15m)}")
    print()
    print("[1] 추세장 조건 (24h 15분봉 MA20 기울기 ±2.5% 초과)")
    if slope_pct is not None:
        print(f"    MA20 기울기: {slope_pct:+.2f}%  (기준: ±{TREND_SLOPE_MIN_PCT}%)")
        print(f"    충족: {trend_ok}  ->  추세장이면 'trend' 반환")
    else:
        print("    MA20 히스토리 부족 또는 0 -> 추세 판정 생략")
    print()
    print("[2] 횡보장 조건 (박스: 간격 2h 이상 고가2/저가2, 범위≥1.8%, 가격 in 박스, 이탈 아님)")
    print(f"    long_ma>0, ma_50>0, ma_100>0: {long_ma > 0}, {ma_50 > 0}, {ma_100 > 0}")
    if pair:
        print(f"    박스 상단/하단: {box_high:,.2f} / {box_low:,.2f}")
        print(f"    박스 범위: {box_range_pct:.2f}%  (최소 {SIDEWAYS_BOX_RANGE_PCT_MIN}%)")
        print(f"    가격 in 박스 [box_low, box_high]: {price_in_box}")
        print(f"    박스 이탈(상/하단 {SIDEWAYS_BOX_EXIT_MARGIN_PCT}% 밖): {exit_margin_breach}")
        print(f"    횡보 충족: {sideways_ok}  ->  충족하면 'sideways' 반환")
    else:
        print("    박스 미충족 (간격·기울기 조건 등으로 _box_high_low_from_two_points -> None)")
    print()
    print("[3] 최종 판정")
    print(f"    regime = {regime!r}  (sideways=횡보장, trend=추세장, neutral=중립)")
    if regime == "neutral":
        reasons = []
        if not trend_ok and slope_pct is not None:
            reasons.append(f"추세 미충족(기울기 {slope_pct:+.2f}% < ±{TREND_SLOPE_MIN_PCT}%)")
        if not sideways_ok:
            if pair is None:
                reasons.append("박스 미산출(2점/기울기 조건 등)")
            elif exit_margin_breach:
                reasons.append("박스권 이탈")
            elif box_range_pct is not None and box_range_pct < SIDEWAYS_BOX_RANGE_PCT_MIN:
                reasons.append(f"박스 범위 부족({box_range_pct:.2f}% < {SIDEWAYS_BOX_RANGE_PCT_MIN}%)")
            elif not price_in_box:
                reasons.append("가격이 박스 밖")
            else:
                reasons.append("박스 조건 일부 미충족")
        print("    -> 중립 사유:", ", ".join(reasons) if reasons else "(추세·횡보 모두 미충족)")
    print("=" * 60)


if __name__ == "__main__":
    main()
