"""전략 검증."""

import sys
import config_paper
sys.modules["config"] = config_paper
import exit_logic_paper
sys.modules["exit_logic"] = exit_logic_paper
import strategy_core_paper
sys.modules["strategy_core"] = strategy_core_paper

from backtest import run_backtest, print_backtest_result
from exchange_client import get_public_exchange
from data import fetch_ohlcv_history


def validate_strategy() -> bool:
    exchange = get_public_exchange()
    df = fetch_ohlcv_history(exchange, days=600)
    if df.empty:
        return False
    days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days
    result = run_backtest(df)
    print_backtest_result(result)
    
    monthly_return = (result.total_pnl_pct / days) * 365 / 12 if days > 0 else 0.0

    m_ret_ok = monthly_return >= 2.0
    mdd_ok = result.max_drawdown <= 33.0
    wr_ok = result.win_rate >= 40.0
    pf_ok = result.profit_factor >= 1.0
    passed = m_ret_ok and mdd_ok and wr_ok and pf_ok
    
    print(f"\n[검증 기준별 결과]")
    print(f"  월간수익률 >=2%:   {monthly_return:.2f}% {'[OK]' if m_ret_ok else '[FAIL]'}")
    print(f"  최대낙폭 <=33%:    {result.max_drawdown:.2f}% {'[OK]' if mdd_ok else '[FAIL]'}")
    print(f"  승률 >=40%:       {result.win_rate:.2f}% {'[OK]' if wr_ok else '[FAIL]'}")
    print(f"  Profit Factor >=1.0: {result.profit_factor:.2f} {'[OK]' if pf_ok else '[FAIL]'}")
    print("=" * 60 + "\n")
    
    return passed


if __name__ == "__main__":
    passed = validate_strategy()
    if passed:
        print("\n[PASS] 전략이 검증 기준을 통과했습니다.")
    else:
        print("\n[FAIL] 전략이 검증 기준을 통과하지 못했습니다.")
