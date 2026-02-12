"""
백테스트 에이전트.

- 현재 설정(config.py)과 후보 설정들을 과거 데이터에 백테스트해 비교한다.
- 지금은 RSI_EXIT 값을 여러 후보로 스윕해서 결과를 보여주는 형태로 구현한다.
"""

from __future__ import annotations

import math
from typing import Dict, List

import pandas as pd

from backtest_core import BacktestResult, run_long_only_backtest
from config import RSI_ENTRY, RSI_EXIT, SYMBOL, TIMEFRAME
from exchange_client import get_public_exchange
from indicators import calculate_rsi
from strategy_core import RsiSwingParams, rsi_swing_signal


def load_ohlcv(limit: int = 1500) -> pd.DataFrame:
    ex = get_public_exchange()
    ohlcv = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["rsi"] = calculate_rsi(df["close"])
    df = df.dropna().reset_index(drop=True)
    return df


def make_signal_fn(params: RsiSwingParams):
    def _signal(row, has_position: bool) -> str:
        return rsi_swing_signal(float(row["rsi"]), has_position, params)

    return _signal


def score(result: BacktestResult) -> float:
    """수익률과 MDD를 함께 고려한 간단한 스코어."""
    return result.pnl_pct - 0.5 * result.max_drawdown


def main() -> None:
    print("5분봉 OHLCV 로드 중...")
    df = load_ohlcv(limit=1500)

    # 현재 설정 포함해서 여러 RSI_EXIT 후보 테스트
    exit_candidates: List[int] = sorted({RSI_EXIT, 55, 60, 65, 70})

    results: Dict[int, BacktestResult] = {}
    for exit_value in exit_candidates:
        print(f"RSI_EXIT={exit_value} 백테스트 중...")
        params = RsiSwingParams(rsi_entry=RSI_ENTRY, rsi_exit=exit_value)
        res = run_long_only_backtest(df, make_signal_fn(params))
        results[exit_value] = res

    print("\n=== RSI_EXIT 후보별 백테스트 결과 ===")
    print("RSI_EXIT | 수익률(%) | MDD(%) | 승률(%) | 트레이드 수")
    print("-" * 60)

    best_exit = None
    best_score_val = -math.inf

    for exit_value in exit_candidates:
        r = results[exit_value]
        s = score(r)
        print(
            f"{exit_value:7d} | {r.pnl_pct:8.2f} | {r.max_drawdown:7.2f} | "
            f"{r.win_rate:7.2f} | {r.num_trades:5d}  (score={s:.2f})"
        )
        if s > best_score_val:
            best_score_val = s
            best_exit = exit_value

    if best_exit is not None:
        best = results[best_exit]
        print("\n=> 추천 RSI_EXIT 값:", best_exit)
        print(
            f"   - 수익률: {best.pnl_pct:.2f}%, MDD: {best.max_drawdown:.2f}%, "
            f"승률: {best.win_rate:.2f}%, 트레이드 수: {best.num_trades}"
        )
        print(
            "\nconfig.py 의 RSI_EXIT 값을 위 추천값으로 변경하면 "
            "실거래/백테스트/모의투자에 동시에 반영됩니다."
        )


if __name__ == "__main__":
    main()

