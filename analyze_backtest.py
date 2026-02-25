"""백테스트 장별 분석."""

import sys
import config_paper
sys.modules["config"] = config_paper
import exit_logic_paper
sys.modules["exit_logic"] = exit_logic_paper
import strategy_core_paper
sys.modules["strategy_core"] = strategy_core_paper

from typing import Optional, List
import pandas as pd

from backtest import run_backtest, BacktestResult, TradeDetail
from exchange_client import get_public_exchange
from data import fetch_ohlcv_history
from strategy_core import REGIME_KR

REGIME_STRATEGY = {
    "sideways": "박스 하단 3% 롱 / 상단 3% 숏 (박스=2시간 간격 고가2/저가2, 기울기 0.5% 이내, MA 조건 없음)",
    "trend": "추세장 (24h MA20 기울기 ±2.5% 초과. 상승장 롱/숏·하락장 롱/숏 가격·RSI 조건, 익절 5.5% 손절 2.5%)",
    "neutral": "중립 (추세·횡보 아님, 진입 없음)",
}


def analyze_by_regime(trade_details: list) -> dict:
    """장별로 거래 통계 분석."""
    by_regime = {}
    for td in trade_details:
        r = td.regime
        if r not in by_regime:
            by_regime[r] = {
                "trades": [],
                "count": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "long_count": 0,
                "long_wins": 0,
                "short_count": 0,
                "short_wins": 0,
            }
        d = by_regime[r]
        d["trades"].append(td.pnl)
        d["count"] += 1
        d["total_pnl"] += td.pnl
        if td.pnl > 0:
            d["wins"] += 1
        else:
            d["losses"] += 1
        if td.side == "LONG":
            d["long_count"] += 1
            if td.pnl > 0:
                d["long_wins"] += 1
        else:
            d["short_count"] += 1
            if td.pnl > 0:
                d["short_wins"] += 1

    result = {}
    for r, d in by_regime.items():
        trades = d["trades"]
        wins_list = [t for t in trades if t > 0]
        losses_list = [t for t in trades if t <= 0]  # 0 포함 (손익분기)
        result[r] = {
            "count": d["count"],
            "wins": d["wins"],
            "losses": d["losses"],
            "win_rate": (d["wins"] / d["count"] * 100) if d["count"] > 0 else 0.0,
            "total_pnl": d["total_pnl"],
            "avg_pnl": sum(trades) / len(trades) if trades else 0.0,
            "avg_win": sum(wins_list) / len(wins_list) if wins_list else 0.0,
            "avg_loss": sum(losses_list) / len(losses_list) if losses_list else 0.0,
            "long_count": d["long_count"],
            "long_wins": d["long_wins"],
            "long_win_rate": (d["long_wins"] / d["long_count"] * 100) if d["long_count"] > 0 else 0.0,
            "short_count": d["short_count"],
            "short_wins": d["short_wins"],
            "short_win_rate": (d["short_wins"] / d["short_count"] * 100) if d["short_count"] > 0 else 0.0,
        }
    return result


def analyze_by_reason(trade_details: list) -> dict:
    """청산 사유별 통계."""
    by_reason = {}
    for td in trade_details:
        r = (td.reason or "unknown").split("(")[0].strip()
        if r not in by_reason:
            by_reason[r] = {"count": 0, "wins": 0, "total_pnl": 0.0}
        d = by_reason[r]
        d["count"] += 1
        d["total_pnl"] += td.pnl
        if td.pnl > 0:
            d["wins"] += 1
    return by_reason


def _get_entry_ts(td: TradeDetail, df: pd.DataFrame):
    """거래 진입 시각 (UTC)."""
    if td.entry_time < 0 or td.entry_time >= len(df):
        return None
    ts = df.iloc[td.entry_time].get("timestamp")
    return pd.to_datetime(ts) if ts is not None else None


def _get_exit_ts(td: TradeDetail, df: pd.DataFrame):
    """거래 청산 시각 (UTC)."""
    if td.exit_time < 0 or td.exit_time >= len(df):
        return None
    ts = df.iloc[td.exit_time].get("timestamp")
    return pd.to_datetime(ts) if ts is not None else None


def _trend_trade_label(td: TradeDetail) -> str:
    """추세장 거래일 때 '추세매매' 또는 '역추세매매'. 아니면 ''."""
    if (td.regime or "") != "trend":
        return ""
    trend_dir = getattr(td, "trend_direction", "") or ""
    if trend_dir == "up":
        return "추세매매" if td.side == "LONG" else "역추세매매"
    if trend_dir == "down":
        return "추세매매" if td.side == "SHORT" else "역추세매매"
    return ""


def _trade_situation_str(td: TradeDetail) -> str:
    """거래를 서술형 상황으로: ex) 횡보장에서 가격이 계속 상승해 숏 손절"""
    regime_kr = REGIME_KR.get(td.regime or "", td.regime or "중립")
    side_kr = "롱" if td.side == "LONG" else "숏"
    price_up = td.exit_price > td.entry_price
    price_move = "가격이 계속 상승" if price_up else "가격이 계속 하락"
    raw = (td.reason or "").split("(")[0].strip()
    if "손절" in raw:
        outcome = "손절"
    elif "익절" in raw:
        outcome = "익절"
    elif "박스권_상단이탈" in raw or "박스권 상단" in raw:
        outcome = "박스 상단 이탈"
    elif "박스권_하단이탈" in raw or "박스권 하단" in raw:
        outcome = "박스 하단 이탈"
    elif "스탑로스" in raw:
        outcome = "스탑로스"
    else:
        outcome = raw or "청산"
    base = f"{regime_kr}에서 {price_move}해 {side_kr} {outcome}"
    trend_label = _trend_trade_label(td)
    if trend_label:
        base += f" ({trend_label})"
    return base


def print_detailed_trade_analysis(result: BacktestResult, df: pd.DataFrame, top_n: int = 10) -> None:
    """거래 내역 상세 분석: 상황별 서술, 많이 번/잃은 거래, RSI·시간대·청산사유별."""
    details = result.trade_details
    if not details:
        print("\n[상세 거래 분석] 거래 내역 없음.")
        return

    n = min(top_n, len(details))
    sorted_by_pnl = sorted(details, key=lambda x: x.pnl, reverse=True)

    print("\n" + "=" * 60)
    print("[상세 거래 분석] 어떤 상황에서 많이 벌고/잃었는지")
    print("=" * 60)

    # 상황별 서술형 요약 (ex. 횡보장에서 가격 상승해 숏 손절)
    situation_pnl: dict = {}
    for td in details:
        sit = _trade_situation_str(td)
        if sit not in situation_pnl:
            situation_pnl[sit] = {"count": 0, "pnl": 0.0}
        situation_pnl[sit]["count"] += 1
        situation_pnl[sit]["pnl"] += td.pnl

    loss_situations = [(s, d) for s, d in situation_pnl.items() if d["pnl"] < 0]
    profit_situations = [(s, d) for s, d in situation_pnl.items() if d["pnl"] > 0]
    loss_situations.sort(key=lambda x: x[1]["pnl"])
    profit_situations.sort(key=lambda x: -x[1]["pnl"])

    print("\n--- 손실이 났던 상황 (많이 잃은 순) ---")
    for sit, d in loss_situations:
        print(f"  · {sit}: {d['count']}회, 총손실 {d['pnl']:+.2f} USDT")

    print("\n--- 수익이 났던 상황 (많이 번 순) ---")
    for sit, d in profit_situations:
        print(f"  · {sit}: {d['count']}회, 총수익 {d['pnl']:+.2f} USDT")

    # 가장 많이 번 거래 Top N
    print("\n--- 가장 많이 번 거래 (Top {}) ---".format(n))
    for i, td in enumerate(sorted_by_pnl[:n], 1):
        if td.pnl <= 0:
            break
        entry_ts = _get_entry_ts(td, df)
        duration_bars = (td.exit_time - td.entry_time) if td.exit_time >= td.entry_time else 0
        ts_str = entry_ts.strftime("%Y-%m-%d %H:%M") if entry_ts is not None else "-"
        regime_kr = REGIME_KR.get(td.regime, td.regime)
        reason_short = (td.reason or "-").split("(")[0].strip()
        trend_label = _trend_trade_label(td)
        trend_str = f" | {trend_label}" if trend_label else ""
        print(f"  {i}. {td.side} | {regime_kr}{trend_str} | {reason_short} | 진입RSI={td.entry_rsi:.0f} | "
              f"손익 {td.pnl:+.2f} USDT ({td.pnl_pct:+.2f}%) | 진입 {ts_str} | 보유 {duration_bars}봉")

    # 가장 많이 잃은 거래 Top N
    print("\n--- 가장 많이 잃은 거래 (Top {}) ---".format(n))
    for i, td in enumerate(sorted_by_pnl[-n:][::-1], 1):
        if td.pnl >= 0:
            break
        entry_ts = _get_entry_ts(td, df)
        reason_short = (td.reason or "-").split("(")[0].strip()
        regime_kr = REGIME_KR.get(td.regime, td.regime)
        trend_label = _trend_trade_label(td)
        trend_str = f" | {trend_label}" if trend_label else ""
        ts_str = entry_ts.strftime("%Y-%m-%d %H:%M") if entry_ts is not None else "-"
        print(f"  {i}. {td.side} | {regime_kr}{trend_str} | {reason_short} | 진입RSI={td.entry_rsi:.0f} | "
              f"손익 {td.pnl:+.2f} USDT ({td.pnl_pct:+.2f}%) | 진입 {ts_str}")

    # 추세장 추세매매/역추세매매별 손익
    trend_details = [td for td in details if (td.regime or "") == "trend"]
    if trend_details:
        print("\n--- 추세장 추세매매 vs 역추세매매 ---")
        for label in ["추세매매", "역추세매매"]:
            subset = [td for td in trend_details if _trend_trade_label(td) == label]
            if not subset:
                continue
            total = sum(t.pnl for t in subset)
            wins = sum(1 for t in subset if t.pnl > 0)
            print(f"  {label}: {len(subset)}회 | 승률 {wins/len(subset)*100:.1f}% | 총손익 {total:+.2f} USDT")

    # 장세별 수익/손실 요약
    print("\n--- 장세별 손익 요약 ---")
    regime_pnl = {}
    for td in details:
        r = td.regime or "unknown"
        if r not in regime_pnl:
            regime_pnl[r] = {"pnl": 0.0, "wins": 0, "losses": 0, "win_sum": 0.0, "loss_sum": 0.0}
        regime_pnl[r]["pnl"] += td.pnl
        if td.pnl > 0:
            regime_pnl[r]["wins"] += 1
            regime_pnl[r]["win_sum"] += td.pnl
        else:
            regime_pnl[r]["losses"] += 1
            regime_pnl[r]["loss_sum"] += td.pnl
    for r in ["sideways", "trend", "neutral"]:
        if r not in regime_pnl:
            continue
        d = regime_pnl[r]
        kr = REGIME_KR.get(r, r)
        print(f"  {kr}: 총손익 {d['pnl']:+.2f} USDT (수익 {d['wins']}회 합계 {d['win_sum']:+.2f}, 손실 {d['losses']}회 합계 {d['loss_sum']:+.2f})")

    # 청산 사유별 수익/손실 기여
    print("\n--- 청산 사유별 수익/손실 기여 ---")
    reason_breakdown = {}
    for td in details:
        r = (td.reason or "unknown").split("(")[0].strip()
        if r not in reason_breakdown:
            reason_breakdown[r] = {"count": 0, "profit_sum": 0.0, "loss_sum": 0.0}
        reason_breakdown[r]["count"] += 1
        if td.pnl > 0:
            reason_breakdown[r]["profit_sum"] += td.pnl
        else:
            reason_breakdown[r]["loss_sum"] += td.pnl
    for r, d in sorted(reason_breakdown.items(), key=lambda x: -(x[1]["profit_sum"] + x[1]["loss_sum"])):
        print(f"  {r}: {d['count']}회 | 수익 기여 {d['profit_sum']:+.2f} | 손실 기여 {d['loss_sum']:+.2f} | 순손익 {d['profit_sum']+d['loss_sum']:+.2f}")

    # 진입 RSI 구간별
    print("\n--- 진입 RSI 구간별 ---")
    rsi_bands = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]
    band_names = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    for (lo, hi), name in zip(rsi_bands, band_names):
        subset = [td for td in details if lo <= td.entry_rsi < hi]
        if not subset:
            continue
        total = sum(t.pnl for t in subset)
        wins_count = sum(1 for t in subset if t.pnl > 0)
        print(f"  RSI {name}: {len(subset)}회 | 승률 {wins_count/len(subset)*100:.1f}% | 총손익 {total:+.2f} USDT")

    # 진입 시간대(UTC)별
    print("\n--- 진입 시간대(UTC)별 손익 ---")
    hour_pnl = {}
    for td in details:
        entry_ts = _get_entry_ts(td, df)
        h = entry_ts.hour if entry_ts is not None else -1
        if h < 0:
            continue
        if h not in hour_pnl:
            hour_pnl[h] = {"pnl": 0.0, "count": 0}
        hour_pnl[h]["pnl"] += td.pnl
        hour_pnl[h]["count"] += 1
    for h in sorted(hour_pnl.keys()):
        d = hour_pnl[h]
        print(f"  {h:02d}:00 UTC: {d['count']}회 | 총손익 {d['pnl']:+.2f} USDT")
    if hour_pnl:
        by_pnl = sorted(hour_pnl.items(), key=lambda x: x[1]["pnl"], reverse=True)
        print("  (수익 최대 시간대: {})".format(", ".join(f"{h:02d}:00" for h, _ in by_pnl[:3])))
        print("  (손실 최대 시간대: {})".format(", ".join(f"{h:02d}:00" for h, _ in by_pnl[-3:][::-1])))

    # 요일별
    print("\n--- 요일별 손익 ---")
    weekday_names = ["월", "화", "수", "목", "금", "토", "일"]
    dow_pnl = {}
    for td in details:
        entry_ts = _get_entry_ts(td, df)
        w = entry_ts.weekday() if entry_ts is not None else -1
        if w < 0:
            continue
        if w not in dow_pnl:
            dow_pnl[w] = {"pnl": 0.0, "count": 0}
        dow_pnl[w]["pnl"] += td.pnl
        dow_pnl[w]["count"] += 1
    for w in sorted(dow_pnl.keys()):
        d = dow_pnl[w]
        print(f"  {weekday_names[w]}요일: {d['count']}회 | 총손익 {d['pnl']:+.2f} USDT")

    # 장세 x 청산사유 조합
    print("\n--- 장세 x 청산사유 조합별 순손익 (상위/하위) ---")
    cross = {}
    for td in details:
        regime = td.regime or "unknown"
        reason = (td.reason or "unknown").split("(")[0].strip()
        key = (regime, reason)
        if key not in cross:
            cross[key] = 0.0
        cross[key] += td.pnl
    sorted_cross = sorted(cross.items(), key=lambda x: x[1], reverse=True)
    for key, pnl in sorted_cross[:8]:
        kr = REGIME_KR.get(key[0], key[0])
        print(f"  {kr} + {key[1]}: {pnl:+.2f} USDT")
    print("  --- 손실 큰 조합 ---")
    for key, pnl in sorted_cross[-5:]:
        if pnl >= 0:
            continue
        kr = REGIME_KR.get(key[0], key[0])
        print(f"  {kr} + {key[1]}: {pnl:+.2f} USDT")

def run_and_analyze(days: int = 600, use_1m: bool = False) -> None:
    """기본은 5분봉 기준 백테스트. use_1m=True면 1분봉 기준(매 1분 진입/청산 판단)."""
    if use_1m:
        print(f"1분봉 {days}일치 데이터 수집 중...")
        exchange = get_public_exchange()
        df = fetch_ohlcv_history(exchange, days=days, timeframe="1m")
        candle_tf = "1m"
    else:
        print(f"5분봉 {days}일치 데이터 수집 중...")
        exchange = get_public_exchange()
        df = fetch_ohlcv_history(exchange, days=days)
        candle_tf = "5m"
    if df.empty:
        print("데이터 없음.")
        return
    days_actual = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days

    print(f"백테스트 실행 중... (기간: {days_actual}일, {candle_tf} 기준)")
    result = run_backtest(df, exchange=exchange, candle_tf=candle_tf)

    # 전체 요약
    print("\n" + "=" * 60)
    print("[전체 요약]")
    print("=" * 60)
    print(f"총 거래: {result.num_trades}회 | 승: {result.num_wins} / 패: {result.num_losses}")
    print(f"승률: {result.win_rate:.1f}% | 총손익: {result.total_pnl:+.2f} USDT ({result.total_pnl_pct:+.2f}%)")
    print(f"평균 수익(승): {result.avg_win:+.2f} | 평균 손실(패): {result.avg_loss:+.2f}")
    print(f"Profit Factor: {result.profit_factor:.2f} | MDD: {result.max_drawdown:.1f}%")
    months = days_actual / 30.44 if days_actual > 0 else 12
    monthly_avg_pct = result.total_pnl_pct / months if months > 0 else 0
    print(f"월평균 수익률: {monthly_avg_pct:+.2f}% (연 {result.total_pnl_pct:+.1f}% / {months:.1f}개월)")

    # 일치 검증: trade_details 수 = 전체 거래 수, 장별 합 = 전체 손익
    n_details = len(result.trade_details)
    if n_details != result.num_trades:
        print(f"\n[검증] 거래 수 불일치: trade_details={n_details} vs num_trades={result.num_trades}")
    regime_stats = analyze_by_regime(result.trade_details)
    sum_regime_count = sum(regime_stats[r]["count"] for r in regime_stats)
    sum_regime_pnl = sum(regime_stats[r]["total_pnl"] for r in regime_stats)
    if sum_regime_count != result.num_trades:
        print(f"[검증] 장별 거래 합계 불일치: {sum_regime_count} vs {result.num_trades}")
    if abs(sum_regime_pnl - result.total_pnl) > 0.01:
        print(f"[검증] 장별 손익 합계 불일치: {sum_regime_pnl:.2f} vs {result.total_pnl:.2f}")

    # 장별(전략별) 분석: 횡보장 + 추세장
    print("\n" + "=" * 60)
    print("[전략별 분석]")
    print("=" * 60)
    for regime in ["sideways", "trend", "neutral"]:
        s = regime_stats.get(regime, {
            "count": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "total_pnl": 0.0, "avg_pnl": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "long_count": 0, "long_wins": 0, "long_win_rate": 0.0,
            "short_count": 0, "short_wins": 0, "short_win_rate": 0.0,
        })
        kr = REGIME_KR.get(regime, regime)
        strat = REGIME_STRATEGY.get(regime, "")
        print(f"\n【{kr}】 {strat}")
        print(f"  거래 {s['count']}회 | 승률: {s['win_rate']:.1f}% (승 {s['wins']} / 패 {s['losses']})")
        print(f"  총손익: {s['total_pnl']:+.2f} USDT | 평균손익: {s['avg_pnl']:+.2f} | 평균 수익: {s['avg_win']:+.2f} | 평균 손실: {s['avg_loss']:+.2f}")
        if s["long_count"] > 0:
            print(f"  롱: {s['long_count']}회, 승률 {s['long_win_rate']:.1f}%")
        if s["short_count"] > 0:
            print(f"  숏: {s['short_count']}회, 승률 {s['short_win_rate']:.1f}%")
        
        # 추세장의 경우 익절/손절 + 추세매매/역추세매매 구분 표시
        if regime == "trend" and s['count'] > 0:
            trend_trades = [td for td in result.trade_details if td.regime == "trend"]
            trend_profit_count = sum(1 for td in trend_trades if (td.reason or "") == "추세_익절")
            trend_stop_count = sum(1 for td in trend_trades if (td.reason or "").startswith("손절_추세") or (td.reason or "").startswith("스탑로스_추세"))
            trend_profit_pnl = sum(td.pnl for td in trend_trades if (td.reason or "") == "추세_익절")
            trend_stop_pnl = sum(td.pnl for td in trend_trades if (td.reason or "").startswith("손절_추세") or (td.reason or "").startswith("스탑로스_추세"))
            if trend_profit_count > 0 or trend_stop_count > 0:
                print(f"  추세 익절: {trend_profit_count}회, 손익 {trend_profit_pnl:+.2f} USDT")
                print(f"  추세 손절: {trend_stop_count}회, 손익 {trend_stop_pnl:+.2f} USDT")
            # 추세매매 vs 역추세매매
            trend_follow = [td for td in trend_trades if _trend_trade_label(td) == "추세매매"]
            counter_trend = [td for td in trend_trades if _trend_trade_label(td) == "역추세매매"]
            if trend_follow:
                tf_pnl = sum(t.pnl for t in trend_follow)
                tf_wins = sum(1 for t in trend_follow if t.pnl > 0)
                print(f"  추세매매: {len(trend_follow)}회, 승률 {tf_wins/len(trend_follow)*100:.1f}%, 손익 {tf_pnl:+.2f} USDT")
            if counter_trend:
                ct_pnl = sum(t.pnl for t in counter_trend)
                ct_wins = sum(1 for t in counter_trend if t.pnl > 0)
                print(f"  역추세매매: {len(counter_trend)}회, 승률 {ct_wins/len(counter_trend)*100:.1f}%, 손익 {ct_pnl:+.2f} USDT")


    # 청산 사유별
    reason_stats = analyze_by_reason(result.trade_details)
    sum_reason_count = sum(d["count"] for d in reason_stats.values())
    if sum_reason_count != result.num_trades:
        print(f"\n[검증] 청산 사유별 거래 합계 불일치: {sum_reason_count} vs {result.num_trades}")
    print("\n" + "=" * 60)
    print("[청산 사유별]")
    print("=" * 60)
    for reason, d in sorted(reason_stats.items(), key=lambda x: -x[1]["count"]):
        wr = (d["wins"] / d["count"] * 100) if d["count"] > 0 else 0
        print(f"  {reason}: {d['count']}회, 승률 {wr:.1f}%, 손익 {d['total_pnl']:+.2f}")

    # 상세 거래 분석
    print_detailed_trade_analysis(result, df, top_n=10)


def _print_daily_balance(result: BacktestResult, df: pd.DataFrame) -> None:
    """날짜별 종료 잔고와 일일 손익 출력."""
    if not result.equity_curve or len(df) == 0:
        return
    # 봉별 equity_curve[i] = i번째 봉 종료 시점 equity. 날짜별로 마지막 봉의 equity를 취함.
    daily_balance: dict = {}
    n = min(len(result.equity_curve), len(df))
    for i in range(n):
        ts = df.iloc[i].get("timestamp")
        if ts is None:
            continue
        d = pd.to_datetime(ts).date()
        daily_balance[d] = result.equity_curve[i]

    if not daily_balance:
        return
    print("\n" + "=" * 60)
    print("[날짜별 잔고 변화]")
    print("=" * 60)
    print(f"  {'날짜':<12} {'잔고(USDT)':>14} {'일일손익':>12}")
    print("  " + "-" * 40)
    prev_date = None
    initial = result.initial_balance
    for d in sorted(daily_balance.keys()):
        bal = daily_balance[d]
        prev_bal = initial if prev_date is None else daily_balance[prev_date]
        day_pnl = bal - prev_bal
        print(f"  {d!s:<12} {bal:>14,.2f} {day_pnl:>+12,.2f}")
        prev_date = d


if __name__ == "__main__":
    import sys
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    run_and_analyze(days=days)
