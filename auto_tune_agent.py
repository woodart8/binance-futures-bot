"""
자동 튜닝 에이전트.

- 퍼블릭 데이터로 여러 RSI_EXIT 후보를 백테스트해서
  현재 설정보다 좋은 값이 있으면 config.py 를 수정한다.
- 그 후 git add/commit/push 까지 수행하여 GitHub 에 자동 반영한다.

주의:
- EC2에서 SSH 키(github_ec2_nopass 등)가 GitHub 에 등록되어 있고
  origin 이 git@github.com:... URL 로 설정되어 있어야 한다.
"""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd

from backtest_core import BacktestResult, run_long_only_backtest
from config import RSI_ENTRY, RSI_EXIT, SYMBOL, TIMEFRAME
from exchange_client import get_public_exchange
from indicators import calculate_rsi
from strategy_core import RsiSwingParams, rsi_swing_signal


CONFIG_PATH = Path(__file__).parent / "config.py"


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


def auto_tune_rsi_exit() -> int | None:
    """
    여러 RSI_EXIT 후보를 백테스트해서,
    현재 값보다 좋은 후보가 있으면 그 값을 반환한다.
    """
    print("5분봉 OHLCV 로드 중...")
    df = load_ohlcv(limit=1500)

    # 현재 설정 포함해서 여러 RSI_EXIT 후보 테스트
    exit_candidates: List[int] = sorted({RSI_EXIT, 55, 60, 65, 70})

    results: Dict[int, BacktestResult] = {}
    for exit_value in exit_candidates:
        print(f"[auto-tune] RSI_EXIT={exit_value} 백테스트 중...")
        params = RsiSwingParams(rsi_entry=RSI_ENTRY, rsi_exit=exit_value)
        res = run_long_only_backtest(df, make_signal_fn(params))
        results[exit_value] = res

    current_result = results[RSI_EXIT]
    current_score = score(current_result)

    print("\n=== 자동 튜닝 후보 결과 ===")
    print("RSI_EXIT | 수익률(%) | MDD(%) | 승률(%) | 트레이드 수 | score")
    print("-" * 70)

    best_exit = RSI_EXIT
    best_score_val = current_score

    for exit_value in exit_candidates:
        r = results[exit_value]
        s = score(r)
        mark = "*" if exit_value == RSI_EXIT else " "
        print(
            f"{exit_value:7d}{mark} | {r.pnl_pct:8.2f} | {r.max_drawdown:7.2f} | "
            f"{r.win_rate:7.2f} | {r.num_trades:5d} | {s:6.2f}"
        )
        if s > best_score_val + 0.1:  # 약간의 여유를 두고 개선일 때만 채택
            best_score_val = s
            best_exit = exit_value

    if best_exit != RSI_EXIT:
        print(
            f"\n=> RSI_EXIT {RSI_EXIT} -> {best_exit} 로 변경 제안 "
            f"(score {current_score:.2f} -> {best_score_val:.2f})"
        )
        return best_exit

    print("\n=> 현재 RSI_EXIT 값이 이미 최적 또는 충분히 우수합니다. 변경 없음.")
    return None


def update_config_rsi_exit(new_exit: int) -> None:
    """
    config.py 의 RSI_EXIT 값을 새 값으로 교체한다.
    """
    text = CONFIG_PATH.read_text(encoding="utf-8")
    pattern = r"(RSI_EXIT\s*=\s*)(\d+)"

    def _repl(match: re.Match) -> str:
        return f"{match.group(1)}{new_exit}"

    new_text, count = re.subn(pattern, _repl, text)
    if count == 0:
        raise RuntimeError("config.py 에서 RSI_EXIT 정의를 찾지 못했습니다.")

    CONFIG_PATH.write_text(new_text, encoding="utf-8")
    print(f"config.py 의 RSI_EXIT 값을 {new_exit} 로 갱신했습니다.")


def git_commit_and_push(new_exit: int) -> None:
    """
    config.py 변경분을 git commit & push 한다.
    """
    msg = f"chore: auto-tune RSI_EXIT to {new_exit}"

    def run(cmd: list[str]) -> None:
        print(f"[git] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    run(["git", "add", "config.py"])
    # 변경이 없으면 commit 이 실패할 수 있으므로 예외는 무시
    try:
        run(["git", "commit", "-m", msg])
    except subprocess.CalledProcessError:
        print("[git] commit 실패 (변경 없음일 수 있음), push 는 건너뜁니다.")
        return

    run(["git", "push", "origin", "main"])
    print("[git] push 완료")


def main() -> None:
    new_exit = auto_tune_rsi_exit()
    if new_exit is None:
        return
    update_config_rsi_exit(new_exit)
    git_commit_and_push(new_exit)


if __name__ == "__main__":
    main()

