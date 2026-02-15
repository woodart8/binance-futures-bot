"""백테스트 장별 분석."""

import sys
import config_paper
sys.modules["config"] = config_paper
import exit_logic_paper
sys.modules["exit_logic"] = exit_logic_paper
import strategy_core_paper
sys.modules["strategy_core"] = strategy_core_paper

from typing import Optional
import pandas as pd

from backtest import run_backtest
from exchange_client import get_public_exchange
from data import fetch_ohlcv_history
from strategy_core import REGIME_KR

REGIME_STRATEGY = {
    "sideways": "박스 하단 4% 롱 / 상단 4% 숏 (박스=2시간 간격 고가2/저가2, 기울기 0.5% 이내, MA 조건 없음)",
    "neutral": "추세 추종 단타 (상승추세 MA 풀백 롱 / 하락추세 MA 풀백 숏, 15분봉 MA·RSI)",
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


def _parse_pattern_name(reason: str) -> Optional[str]:
    """'패턴익절(double_bottom)' -> 'double_bottom', '패턴손절(harmonic_gartley)' -> 'harmonic_gartley'."""
    if not reason or (not reason.startswith("패턴익절") and not reason.startswith("패턴손절")):
        return None
    if "(" in reason and ")" in reason:
        return reason.split("(")[1].rstrip(")").strip()
    return ""


def analyze_by_pattern(trade_details: list) -> dict:
    """패턴 진입 거래만 골라 패턴명별 통계 (추가된 차트/하모닉 패턴 포함)."""
    by_pattern = {}
    for td in trade_details:
        name = _parse_pattern_name(td.reason or "")
        if name is None:
            continue
        key = name or "(패턴명없음)"
        if key not in by_pattern:
            by_pattern[key] = {"count": 0, "wins": 0, "total_pnl": 0.0}
        d = by_pattern[key]
        d["count"] += 1
        d["total_pnl"] += td.pnl
        if td.pnl > 0:
            d["wins"] += 1
    return by_pattern


def run_and_analyze(days: int = 600) -> None:
    print(f"5분봉 {days}일치 데이터 수집 중...")
    exchange = get_public_exchange()
    df = fetch_ohlcv_history(exchange, days=days)
    if df.empty:
        print("데이터 없음.")
        return
    days_actual = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days

    print(f"백테스트 실행 중... (기간: {days_actual}일)")
    result = run_backtest(df)

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
    for regime in ["sideways", "neutral"]:
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

    # 패턴 거래 (패턴 익절/손절로 청산된 거래만)
    pattern_trades = [td for td in result.trade_details if (td.reason or "").startswith("패턴익절") or (td.reason or "").startswith("패턴손절")]
    pattern_wins = sum(1 for td in pattern_trades if td.pnl > 0)
    pattern_losses = len(pattern_trades) - pattern_wins
    pattern_win_rate = (pattern_wins / len(pattern_trades) * 100) if pattern_trades else 0.0
    pattern_pnl = sum(td.pnl for td in pattern_trades)
    print("\n" + "=" * 60)
    print("[패턴 거래]")
    print("=" * 60)
    print(f"  거래: {len(pattern_trades)}회 | 승: {pattern_wins} / 패: {pattern_losses} | 승률: {pattern_win_rate:.1f}% | 총손익: {pattern_pnl:+.2f} USDT")

    # 패턴별 통계 (차트 패턴 + 하모닉 패턴)
    pattern_stats = analyze_by_pattern(result.trade_details)
    if pattern_stats:
        print("\n" + "=" * 60)
        print("[패턴별 통계]")
        print("=" * 60)
        for pname, d in sorted(pattern_stats.items(), key=lambda x: -x[1]["count"]):
            wr = (d["wins"] / d["count"] * 100) if d["count"] > 0 else 0
            print(f"  {pname}: {d['count']}회, 승률 {wr:.1f}%, 손익 {d['total_pnl']:+.2f} USDT")

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


if __name__ == "__main__":
    import sys
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    run_and_analyze(days=days)
