"""백테스트."""

import sys
import config_paper
sys.modules["config"] = config_paper
import exit_logic_paper
sys.modules["exit_logic"] = exit_logic_paper
import strategy_core_paper
sys.modules["strategy_core"] = strategy_core_paper

import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

from config import (
    FEE_RATE,
    INITIAL_BALANCE,
    LEVERAGE,
    POSITION_SIZE_PERCENT,
    RSI_PERIOD,
    MA_SHORT_PERIOD,
    MA_LONG_PERIOD,
    MA_MID_PERIOD,
    MA_LONGEST_PERIOD,
    CONSECUTIVE_LOSS_LIMIT,
    DAILY_LOSS_LIMIT_PCT,
    SIDEWAYS_BOX_PERIOD,
    REGIME_LOOKBACK_15M,
    TREND_SLOPE_BARS,
    SYMBOL,
)
from indicators import calculate_rsi, calculate_ma
from exit_logic import check_long_exit, check_short_exit, reason_to_display_message
from strategy_core import swing_strategy_signal, detect_market_regime, get_sideways_box_bounds
from funding import (
    FUNDING_HOURS_UTC,
    fetch_funding_history,
    compute_funding_pnl,
)


@dataclass
class TradeDetail:
    side: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    entry_time: int  # 인덱스
    exit_time: int  # 인덱스
    pnl: float
    pnl_pct: float
    entry_rsi: float
    exit_rsi: float
    regime: str
    reason: str  # 청산 이유


@dataclass
class BacktestResult:
    initial_balance: float
    final_balance: float
    total_pnl: float
    total_pnl_pct: float
    total_funding_pnl: float  # 펀딩비 누적 손익
    num_trades: int
    num_wins: int
    num_losses: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    equity_curve: List[float]
    trade_details: List[TradeDetail]


def _close_position(
    side: str,
    entry_info: dict,
    entry_price: float,
    exit_price: float,
    entry_time: int,
    exit_time: int,
    pnl_pct: float,
    balance: float,
    current_position_size: float,
    rsi: float,
    regime: str,
    reason: str,
    trades: list,
    wins: list,
    losses: list,
    trade_details: list,
    accumulated_partial_pnl: float = 0.0,
) -> tuple[float, bool]:
    remaining_pnl = pnl_pct / 100 * balance * current_position_size
    fee = balance * current_position_size * FEE_RATE
    net_pnl = remaining_pnl - fee
    total_pnl = accumulated_partial_pnl + net_pnl

    trades.append(total_pnl)
    is_win = total_pnl > 0
    if is_win:
        wins.append(total_pnl)
    else:
        losses.append(total_pnl)

    trade_details.append(TradeDetail(
        side=side,
        entry_price=entry_info.get("entry_price", entry_price),
        exit_price=exit_price,
        entry_time=entry_time,
        exit_time=exit_time,
        pnl=total_pnl,
        pnl_pct=pnl_pct,
        entry_rsi=entry_info.get("rsi", rsi),
        exit_rsi=rsi,
        regime=regime,
        reason=reason
    ))
    
    return net_pnl, is_win


def run_backtest(df: pd.DataFrame, exchange=None) -> BacktestResult:
    """
    백테스트 실행.

    :param df: OHLCV 데이터프레임 (timestamp, open, high, low, close, volume). timestamp는 UTC 가정.
    :param exchange: ccxt 거래소 인스턴스. 주어지면 해당 구간 펀딩 히스토리를 조회해 펀딩비 반영.
    :return: 백테스트 결과 (total_pnl에 매매손익+펀딩손익 포함)
    """
    # 지표 계산 (5분봉)
    df = df.copy()
    df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
    df["ma_short"] = calculate_ma(df["close"], MA_SHORT_PERIOD)
    df["ma_long"] = calculate_ma(df["close"], MA_LONG_PERIOD)
    df["ma_50"] = calculate_ma(df["close"], MA_MID_PERIOD)
    df["ma_100"] = calculate_ma(df["close"], MA_LONGEST_PERIOD)
    
    # 볼륨 평균 계산 (20기간 이동평균)
    df["volume_ma"] = df["volume"].rolling(window=20).mean()
    
    df = df.dropna().reset_index(drop=True)

    # 펀딩 레이트 히스토리 (exchange 있으면 조회, 없으면 펀딩 0으로 적용)
    funding_rates_by_key: dict = {}
    if exchange is not None and len(df) > 0:
        ts_min = pd.to_datetime(df["timestamp"].min())
        ts_max = pd.to_datetime(df["timestamp"].max())
        if getattr(ts_min, "tzinfo", None) is None:
            ts_min = ts_min.tz_localize("UTC")
        if getattr(ts_max, "tzinfo", None) is None:
            ts_max = ts_max.tz_localize("UTC")
        start_ms = int(ts_min.timestamp() * 1000)
        end_ms = int(ts_max.timestamp() * 1000)
        try:
            history = fetch_funding_history(exchange, SYMBOL, start_ms, end_ms)
            for ts_ms, rate in history:
                t = pd.Timestamp(ts_ms, unit="ms", tz="UTC")
                if t.hour in FUNDING_HOURS_UTC:
                    key = f"{t.year:04d}-{t.month:02d}-{t.day:02d}-{t.hour:02d}"
                    funding_rates_by_key[key] = rate
        except Exception:
            pass

    # 15분봉 리샘플 (장세 판별용 24시간)
    df_tmp = df.copy()
    df_tmp["timestamp"] = pd.to_datetime(df_tmp["timestamp"])
    df_15m = df_tmp.set_index("timestamp").resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    df_15m["ma_short"] = calculate_ma(df_15m["close"], MA_SHORT_PERIOD)
    df_15m["ma_long"] = calculate_ma(df_15m["close"], MA_LONG_PERIOD)
    df_15m["ma_50"] = calculate_ma(df_15m["close"], MA_MID_PERIOD)
    df_15m["ma_100"] = calculate_ma(df_15m["close"], MA_LONGEST_PERIOD)
    df_15m["rsi"] = calculate_rsi(df_15m["close"], RSI_PERIOD)
    df_15m = df_15m.dropna().reset_index()
    closes_15m = df_15m["close"].values

    # 초기 상태
    balance = INITIAL_BALANCE
    equity = INITIAL_BALANCE
    equity_curve = [equity]
    
    has_position = False
    is_long = False
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = float("inf")
    partial_profit_taken = False
    trailing_stop_active = False
    best_pnl_pct = 0.0  # 최고 수익률 추적 (트레일링 스톱용)
    
    peak_equity = equity
    max_drawdown = 0.0
    
    trades = []
    wins = []
    losses = []
    trade_details = []
    
    # 일일 손실 한도 / 연속 손실 추적
    daily_start_balance = INITIAL_BALANCE
    daily_start_date = ""
    consecutive_loss_count = 0
    
    # 진입 시점 정보 저장용
    entry_info = {
        "index": 0,
        "rsi": 0.0,
        "regime": "",
        "side": "",
        "entry_price": 0.0,
        "box_high": 0.0,  # 진입 시점의 박스권 상단
        "box_low": 0.0,   # 진입 시점의 박스권 하단
    }
    
    # 포지션 크기
    current_position_size = POSITION_SIZE_PERCENT

    # 펀딩비: 00/08/16 UTC 적용 구간 추적
    applied_funding_keys: set = set()
    total_funding_pnl = 0.0
    
    total_rows = len(df)
    last_progress = -1
    
    # 가격 히스토리를 미리 배열로 변환 (성능 최적화)
    close_prices = df["close"].values
    high_prices = df["high"].values
    low_prices = df["low"].values
    
    for i, row in df.iterrows():
        price = float(row["close"])
        volume = float(row["volume"])
        avg_volume = float(row.get("volume_ma", volume))
        rsi = float(row["rsi"])
        row_ts = row.get("timestamp")
        if row_ts is not None and has_position:
            row_ts = pd.Timestamp(row_ts)
            if getattr(row_ts, "tzinfo", None) is None and hasattr(row_ts, "tz_localize"):
                row_ts = row_ts.tz_localize("UTC")
            hour = getattr(row_ts, "hour", None)
            if hour is not None and hour in FUNDING_HOURS_UTC:
                key = f"{row_ts.year:04d}-{row_ts.month:02d}-{row_ts.day:02d}-{row_ts.hour:02d}"
                if key not in applied_funding_keys:
                    rate = funding_rates_by_key.get(key, 0.0)
                    position_value = balance * current_position_size * LEVERAGE
                    funding_pnl = compute_funding_pnl(position_value, rate, is_long)
                    balance += funding_pnl
                    total_funding_pnl += funding_pnl
                    applied_funding_keys.add(key)
        short_ma = float(row["ma_short"])
        long_ma = float(row["ma_long"])
        ma_50 = float(row.get("ma_50", 0.0))
        ma_100 = float(row.get("ma_100", 0.0))
        # 장세: 15분봉 24시간 기준 (박스권=하단/상단 4% 진입, 추세장=MA20 기울기 ±2.5% 초과 시 진입)
        n_15m = len(closes_15m)
        short_ma_15m, long_ma_15m, price_history_15m = 0.0, 0.0, []
        ma_50_15m, ma_100_15m = 0.0, 0.0
        ma_long_history = None
        if n_15m >= REGIME_LOOKBACK_15M:
            idx_15m = min(i // 3, n_15m - 1)
            short_ma_15m = float(df_15m.iloc[idx_15m]["ma_short"])
            long_ma_15m = float(df_15m.iloc[idx_15m]["ma_long"])
            ma_50_15m = float(df_15m.iloc[idx_15m]["ma_50"])
            ma_100_15m = float(df_15m.iloc[idx_15m]["ma_100"])
            start_15m = max(0, idx_15m - REGIME_LOOKBACK_15M + 1)
            seg_15m = df_15m.iloc[start_15m : idx_15m + 1]
            price_history_15m = list(zip(seg_15m["high"].tolist(), seg_15m["low"].tolist(), seg_15m["close"].tolist()))
            # MA20 히스토리 (추세장 판단용) - 현재 시점까지의 전체 히스토리 (detect_market_regime에서 최근 96개 사용)
            ma_long_history = df_15m.iloc[:idx_15m + 1]["ma_long"].dropna().tolist()
            if len(price_history_15m) >= REGIME_LOOKBACK_15M:
                current_regime = detect_market_regime(
                    short_ma_15m, long_ma_15m, price, ma_50_15m, ma_100_15m,
                    price_history_15m, box_period=REGIME_LOOKBACK_15M,
                    ma_long_history=ma_long_history,
                )
            else:
                current_regime = "neutral"
        else:
            current_regime = "neutral"
        
        # 청산 1순위: exit_logic (TP/SL/트레일링/RSI조기청산)
        if has_position:
            regime_at_entry = entry_info.get("regime", "sideways")
            pnl_pct = (price - entry_price) / entry_price * LEVERAGE * 100 if is_long else (entry_price - price) / entry_price * LEVERAGE * 100
            
            if is_long:
                if price > highest_price:
                    highest_price = price
                if pnl_pct > best_pnl_pct:
                    best_pnl_pct = pnl_pct

                regime = regime_at_entry or ""
                box_high = entry_info.get("box_high", 0.0) or 0
                box_low = entry_info.get("box_low", 0.0) or 0
                reason = check_long_exit(
                    regime=regime, pnl_pct=pnl_pct, price=price,
                    entry_price=entry_price, best_pnl_pct=best_pnl_pct,
                    box_high=box_high, box_low=box_low,
                )
                if reason:
                    msg = reason_to_display_message(reason, is_long=True)
                    acc = entry_info.get("accumulated_partial_pnl", 0)
                    net_pnl, _ = _close_position("LONG", entry_info, entry_price, price, entry_info.get("index", i), i, pnl_pct, balance, current_position_size, rsi, regime_at_entry, msg, trades, wins, losses, trade_details, accumulated_partial_pnl=acc)
                    balance += net_pnl
                    consecutive_loss_count = 0 if net_pnl > 0 else consecutive_loss_count + 1
                    has_position = False
                    partial_profit_taken = False
                    trailing_stop_active = False
                    best_pnl_pct = 0.0
                    current_position_size = POSITION_SIZE_PERCENT
                    equity = balance
                    continue
            else:  # SHORT
                if price < lowest_price:
                    lowest_price = price
                if pnl_pct > best_pnl_pct:
                    best_pnl_pct = pnl_pct

                regime = regime_at_entry or ""
                box_high = entry_info.get("box_high", 0.0) or 0
                box_low = entry_info.get("box_low", 0.0) or 0
                reason = check_short_exit(
                    regime=regime, pnl_pct=pnl_pct, price=price,
                    entry_price=entry_price, best_pnl_pct=best_pnl_pct,
                    box_high=box_high, box_low=box_low,
                )
                if reason:
                    msg = reason_to_display_message(reason, is_long=False)
                    acc = entry_info.get("accumulated_partial_pnl", 0)
                    net_pnl, _ = _close_position("SHORT", entry_info, entry_price, price, entry_info.get("index", i), i, pnl_pct, balance, current_position_size, rsi, regime_at_entry, msg, trades, wins, losses, trade_details, accumulated_partial_pnl=acc)
                    balance += net_pnl
                    consecutive_loss_count = 0 if net_pnl > 0 else consecutive_loss_count + 1
                    has_position = False
                    partial_profit_taken = False
                    trailing_stop_active = False
                    best_pnl_pct = 0.0
                    current_position_size = POSITION_SIZE_PERCENT
                    equity = balance
                    continue

        # 전략 신호 생성 (regime은 5분봉 기반)
        regime = current_regime
        
        # 박스권 판단: 고가/저가/종가 (h,l,c) 리스트
        if i >= SIDEWAYS_BOX_PERIOD:
            seg = df.iloc[i - SIDEWAYS_BOX_PERIOD:i + 1]
            price_history = list(zip(seg["high"].tolist(), seg["low"].tolist(), seg["close"].tolist()))
        else:
            seg = df.iloc[:i + 1]
            price_history = list(zip(seg["high"].tolist(), seg["low"].tolist(), seg["close"].tolist()))
        
        # 일일 손실 한도: 날짜 변경 시 daily_start_balance 갱신
        ts = row.get("timestamp")
        current_date = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10] if ts is not None else ""
        if current_date and daily_start_date != current_date:
            daily_start_balance = balance
            daily_start_date = current_date
            consecutive_loss_count = 0
        
        # 일일 손실 한도 / 연속 손실 한도: 초과 시 진입 불가
        daily_loss_pct = 0.0
        if daily_start_balance > 0:
            daily_loss_pct = (daily_start_balance - balance) / daily_start_balance * 100
        daily_limit_hit = daily_loss_pct >= DAILY_LOSS_LIMIT_PCT
        consecutive_limit_hit = consecutive_loss_count >= CONSECUTIVE_LOSS_LIMIT
        
        # 전략 시그널 생성 (횡보: 박스 하단/상단 4% 진입 MA무관, 추세: MA20 기울기 ±2.5% 초과 시 가격·RSI 조건 진입)
        use_15m_sideways = n_15m >= REGIME_LOOKBACK_15M and len(price_history_15m) >= REGIME_LOOKBACK_15M
        use_15m_trend = regime == "trend" and n_15m >= REGIME_LOOKBACK_15M
        rsi_prev = float(df.iloc[i - 1]["rsi"]) if i > 0 else None
        open_prev = float(df.iloc[i - 1]["open"]) if i > 0 else None
        close_prev = float(df.iloc[i - 1]["close"]) if i > 0 else None
        open_curr = float(row["open"])
        rsi_use = rsi
        regime_short = short_ma_15m if (use_15m_sideways or use_15m_trend) else None
        regime_long = long_ma_15m if (use_15m_sideways or use_15m_trend) else None
        regime_ma50 = ma_50_15m if (use_15m_sideways or use_15m_trend) else None
        regime_ma100 = ma_100_15m if (use_15m_sideways or use_15m_trend) else None
        signal = swing_strategy_signal(
            rsi_value=rsi_use,
            price=price,
            rsi_prev=rsi_prev,
            open_prev=open_prev,
            close_prev=close_prev,
            open_curr=open_curr,
            short_ma=short_ma,
            long_ma=long_ma,
            has_position=has_position,
            is_long=is_long,
            regime=regime,
            price_history=price_history,
            regime_short_ma=regime_short,
            regime_long_ma=regime_long,
            regime_ma_50=regime_ma50,
            regime_ma_100=regime_ma100,
            regime_price_history=price_history_15m if use_15m_sideways else None,
            regime_ma_long_history=ma_long_history if use_15m_trend else None,
        )
        
        # 진입/청산 처리
        if signal == "long" and not has_position and not daily_limit_hit and not consecutive_limit_hit:
            box_high_entry = 0.0
            box_low_entry = 0.0
            if regime == "sideways" and use_15m_sideways:
                bounds = get_sideways_box_bounds(price_history_15m, REGIME_LOOKBACK_15M)
                if bounds:
                    box_high_entry, box_low_entry = bounds
            
            has_position = True
            is_long = True
            entry_price = price
            highest_price = price
            lowest_price = float("inf")
            partial_profit_taken = False
            trailing_stop_active = False
            best_pnl_pct = 0.0
            entry_info = {
                "index": i,
                "entry_price": entry_price,
                "rsi": rsi,
                "regime": regime,
                "side": "LONG",
                "box_high": box_high_entry,
                "box_low": box_low_entry,
                "accumulated_partial_pnl": 0.0,
            }
        elif signal == "short" and not has_position and not daily_limit_hit and not consecutive_limit_hit:
            box_high_entry = 0.0
            box_low_entry = 0.0
            if regime == "sideways" and use_15m_sideways:
                bounds = get_sideways_box_bounds(price_history_15m, REGIME_LOOKBACK_15M)
                if bounds:
                    box_high_entry, box_low_entry = bounds
            
            has_position = True
            is_long = False
            entry_price = price
            highest_price = 0.0
            lowest_price = price
            partial_profit_taken = False
            trailing_stop_active = False
            best_pnl_pct = 0.0
            entry_info = {
                "index": i,
                "entry_price": entry_price,
                "rsi": rsi,
                "regime": regime,
                "side": "SHORT",
                "box_high": box_high_entry,
                "box_low": box_low_entry,
                "accumulated_partial_pnl": 0.0,
            }
        elif signal == "flat" and has_position:
            # 전략 신호로 청산
                if is_long:
                    pnl_pct = (price - entry_price) / entry_price * LEVERAGE * 100
                else:
                    pnl_pct = (entry_price - price) / entry_price * LEVERAGE * 100

                remaining_pnl = pnl_pct / 100 * balance * current_position_size
                fee = balance * current_position_size * FEE_RATE
                net_pnl = remaining_pnl - fee
                acc = entry_info.get("accumulated_partial_pnl", 0)
                total_pnl = acc + net_pnl
                balance += net_pnl
                trades.append(total_pnl)
                if total_pnl > 0:
                    wins.append(total_pnl)
                else:
                    losses.append(total_pnl)
                trade_details.append(TradeDetail(
                    side="LONG" if is_long else "SHORT",
                    entry_price=entry_info.get("entry_price", entry_price),
                    exit_price=price,
                    entry_time=entry_info.get("index", i),
                    exit_time=i,
                    pnl=total_pnl,
                    pnl_pct=pnl_pct,
                    entry_rsi=entry_info.get("rsi", rsi),
                    exit_rsi=rsi,
                    regime=entry_info.get("regime", regime),
                    reason="전략 신호"
                ))
                has_position = False
                partial_profit_taken = False
                trailing_stop_active = False
                best_pnl_pct = 0.0
                current_position_size = POSITION_SIZE_PERCENT
                equity = balance
        
        # 평가손익 계산
        if has_position:
            if is_long:
                unrealized = (price - entry_price) / entry_price * LEVERAGE * balance * current_position_size
            else:
                unrealized = (entry_price - price) / entry_price * LEVERAGE * balance * current_position_size
            equity = balance + unrealized
        else:
            equity = balance
        
        equity_curve.append(equity)
        if equity > peak_equity:
            peak_equity = equity
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)
    
    # 통계 계산
    num_trades = len(trades)
    num_wins = len(wins)
    num_losses = len(losses)
    win_rate = num_wins / num_trades * 100 if num_trades > 0 else 0.0
    avg_win = sum(wins) / num_wins if num_wins > 0 else 0.0
    avg_loss = sum(losses) / num_losses if num_losses > 0 else 0.0
    profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float("inf")
    
    # Sharpe Ratio 계산 (간단 버전)
    if len(equity_curve) > 1:
        returns = pd.Series(equity_curve).pct_change().dropna()
        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5)  # 연율화
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0
    
    total_pnl = balance - INITIAL_BALANCE
    total_pnl_pct = total_pnl / INITIAL_BALANCE * 100
    
    return BacktestResult(
        initial_balance=INITIAL_BALANCE,
        final_balance=balance,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
        total_funding_pnl=total_funding_pnl,
        num_trades=num_trades,
        num_wins=num_wins,
        num_losses=num_losses,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown * 100,
        sharpe_ratio=sharpe_ratio,
        equity_curve=equity_curve,
        trade_details=trade_details,
    )


def print_backtest_result(result: BacktestResult) -> None:
    """백테스트 결과 출력."""
    funding_str = f", 펀딩={result.total_funding_pnl:+.2f}" if result.total_funding_pnl != 0 else ""
    print(f"{result.total_pnl:+.2f} USDT ({result.total_pnl_pct:+.2f}%){funding_str}, 거래: {result.num_trades}회, 승률: {result.win_rate:.2f}%")
