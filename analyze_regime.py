"""
최근 24시간 데이터로 현재 장세(횡보/추세) 판정 이유 분석.
실행: python analyze_regime.py
"""

import sys
import config_paper
sys.modules["config"] = config_paper
import strategy_core_paper
sys.modules["strategy_core"] = strategy_core_paper


from exchange_client import get_public_exchange
from data import fetch_ohlcv, compute_regime_15m
from config_common import REGIME_LOOKBACK_15M, SIDEWAYS_BOX_RANGE_PCT_MIN, SIDEWAYS_MIN_TOUCHES
from strategy_core_paper import BOX_TOUCH_THRESHOLD

def main():
    exchange = get_public_exchange()
    # 15분봉 96개 = 24h, 5분봉으로는 96*3 + 여유 (dropna 후 부족할 수 있어 더 요청)
    # 15분봉 96개 + MA100으로 인한 dropna(앞 100봉) 여유 -> 15분봉 200개 이상 필요 -> 5분봉 600+ 필요
    limit = 700
    df = fetch_ohlcv(exchange, limit=limit)
    if df.empty or len(df) < REGIME_LOOKBACK_15M * 2:
        print("데이터 부족")
        return

    # compute_regime_15m 호출로 실제 regime 및 15m MA 확보 (data 내부에서 리샘플·MA 계산)
    current_price = float(df["close"].iloc[-1])
    regime, short_ma, long_ma, ma_50, ma_100, price_history_15m, rsi_15m, _, _ = compute_regime_15m(df, current_price)
    if not price_history_15m or len(price_history_15m) < REGIME_LOOKBACK_15M:
        print("15분봉 데이터 부족: 5분봉", len(df), "개 -> 15분봉", len(price_history_15m) if price_history_15m else 0, "개 (필요:", REGIME_LOOKBACK_15M, ")")
        return

    period = REGIME_LOOKBACK_15M

    # 박스 계산
    recent = price_history_15m[-period:]
    box_high = max(recent)
    box_low = min(recent)
    box_range = box_high - box_low
    box_range_pct = (box_range / box_low * 100) if box_low > 0 else 0
    top_touches = sum(1 for p in recent if abs(p - box_high) / box_high < BOX_TOUCH_THRESHOLD)
    bottom_touches = sum(1 for p in recent if abs(p - box_low) / box_low < BOX_TOUCH_THRESHOLD)
    price_in_box = box_low <= current_price <= box_high

    print("=" * 60)
    print("최근 24시간(15분봉 96개) 기준 장세 분석")
    print("=" * 60)
    print(f"현재가: {current_price:,.2f}")
    print(f"기간: 15분봉 {REGIME_LOOKBACK_15M}개 = 24시간")
    print()
    print("[박스(횡보) 조건]")
    print(f"  박스 고가(box_high): {box_high:,.2f}")
    print(f"  박스 저가(box_low):  {box_low:,.2f}")
    print(f"  박스 범위: {box_range:,.2f} ({box_range_pct:.2f}%)")
    print(f"  조건: 범위 >= {SIDEWAYS_BOX_RANGE_PCT_MIN}%  ->  {box_range_pct >= SIDEWAYS_BOX_RANGE_PCT_MIN}")
    print(f"  조건: 가격 in [box_low, box_high]  ->  {price_in_box}  (box_low~box_high 안에 현재가)")
    print(f"  상단 터치(고가의 {BOX_TOUCH_THRESHOLD*100:.1f}% 이내 봉 수): {top_touches}  (필요: >={SIDEWAYS_MIN_TOUCHES})")
    print(f"  하단 터치(저가의 {BOX_TOUCH_THRESHOLD*100:.1f}% 이내 봉 수): {bottom_touches}  (필요: >={SIDEWAYS_MIN_TOUCHES})")
    print()
    print("[15분봉 MA (추세 방향)]")
    print(f"  MA7:  {short_ma:,.2f}")
    print(f"  MA20: {long_ma:,.2f}")
    print(f"  MA50: {ma_50:,.2f}")
    print(f"  MA100: {ma_100:,.2f}")
    print(f"  RSI(15m): {rsi_15m:.1f}" if rsi_15m is not None else "  RSI: -")
    print()
    print("[판정 결과]")
    print(f"  regime = {regime!r}  (sideways=횡보장, neutral=추세장)")
    if regime == "sideways":
        print("  -> 위 박스 조건 4개 모두 충족하여 횡보장으로 판정됨.")
    else:
        reasons = []
        if box_range_pct < SIDEWAYS_BOX_RANGE_PCT_MIN:
            reasons.append(f"박스 범위 부족 ({box_range_pct:.2f}% < {SIDEWAYS_BOX_RANGE_PCT_MIN}%)")
        if not price_in_box:
            reasons.append("현재가가 박스 밖")
        if top_touches < SIDEWAYS_MIN_TOUCHES:
            reasons.append(f"상단 터치 부족 ({top_touches} < {SIDEWAYS_MIN_TOUCHES})")
        if bottom_touches < SIDEWAYS_MIN_TOUCHES:
            reasons.append(f"하단 터치 부족 ({bottom_touches} < {SIDEWAYS_MIN_TOUCHES})")
        print("  -> 추세장(neutral)으로 판정. 미충족:", ", ".join(reasons) if reasons else "-")
    print("=" * 60)

if __name__ == "__main__":
    main()
