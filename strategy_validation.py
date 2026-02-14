"""
전략 검증 시스템.

5분봉 기준으로 1년(365일)치 데이터를 수집하여 백테스트를 실행합니다.
"""

import pandas as pd
from datetime import datetime
import time
import ccxt

from backtest import BacktestResult, run_backtest
from exchange_client import get_public_exchange


def validate_strategy() -> bool:
    """
    전략을 검증합니다.
    
    :return: 검증 통과 여부
    """
    exchange = get_public_exchange()
    
    target_days = 365  # 1년
    target_candles = target_days * 24 * 12  # 5분봉: 하루 288개
    
    all_ohlcv = []
    batch_size = 1500  # 바이낸스 최대 제한
    num_batches = (target_candles + batch_size - 1) // batch_size
    
    end_time = int(datetime.now().timestamp() * 1000)  # 현재 시간 (밀리초)
    current_end = end_time
    
    for i in range(num_batches):
        try:
            # 5분봉 데이터 가져오기
            batch_start = current_end - (batch_size * 5 * 60 * 1000)  # 5분봉 기준
            ohlcv = exchange.fetch_ohlcv("BTC/USDT", "5m", limit=batch_size, since=batch_start)
            
            if not ohlcv:
                break
            
            # 중복 제거
            if all_ohlcv:
                existing_timestamps = {c[0] for c in all_ohlcv}
                ohlcv = [c for c in ohlcv if c[0] not in existing_timestamps]
            
            if ohlcv:
                all_ohlcv.extend(ohlcv)
                # 타임스탬프로 정렬
                all_ohlcv.sort(key=lambda x: x[0])
                
                # 다음 배치를 위한 시간 설정 (5분봉 기준)
                current_end = all_ohlcv[0][0] - (5 * 60 * 1000)
                
            
            # API rate limit 방지
            time.sleep(0.1)
            
            if len(all_ohlcv) >= target_candles:
                break
                
        except Exception as e:
            break
    
    if len(all_ohlcv) == 0:
        return False
    
    # 데이터프레임 생성
    df = pd.DataFrame(
        all_ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    # 타임스탬프로 정렬 (오래된 것부터)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 중복 제거
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    
    days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days
    
    result = run_backtest(df)
    
    from backtest import print_backtest_result
    print_backtest_result(result)
    
    monthly_return = (result.total_pnl_pct / days) * 365 / 12 if days > 0 else 0.0
    annual_return = monthly_return * 12
    
    # 통과 여부 판단 (항목별)
    m_ret_ok = monthly_return >= 2.0
    mdd_ok = result.max_drawdown <= 15.0
    wr_ok = result.win_rate >= 40.0
    pf_ok = result.profit_factor >= 1.0
    passed = m_ret_ok and mdd_ok and wr_ok and pf_ok
    
    print(f"\n[검증 기준별 결과]")
    print(f"  월간수익률 >=2%:   {monthly_return:.2f}% {'[OK]' if m_ret_ok else '[FAIL]'}")
    print(f"  최대낙폭 <=15%:    {result.max_drawdown:.2f}% {'[OK]' if mdd_ok else '[FAIL]'}")
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
