"""
백테스트 시스템.

과거 데이터를 사용하여 전략의 성과를 검증합니다.
"""

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
    BULLISH_PROFIT_TARGET,
    BULLISH_PROFIT_TARGET_PARTIAL,
    BULLISH_PARTIAL_EXIT_PCT,
    BULLISH_STOP_LOSS,
    BULLISH_STOP_LOSS_PRICE,
    BULLISH_EARLY_EXIT_RSI,
    BULLISH_TRAILING_STOP_ACTIVATION,
    BULLISH_TRAILING_STOP_PCT,
    BEARISH_PROFIT_TARGET,
    BEARISH_PROFIT_TARGET_PARTIAL,
    BEARISH_PARTIAL_EXIT_PCT,
    BEARISH_STOP_LOSS,
    BEARISH_STOP_LOSS_PRICE,
    BEARISH_EARLY_EXIT_RSI,
    BEARISH_TRAILING_STOP_ACTIVATION,
    BEARISH_TRAILING_STOP_PCT,
    SIDEWAYS_BOX_PERIOD,
    SIDEWAYS_BOX_TOP_MARGIN,
    SIDEWAYS_BOX_BOTTOM_MARGIN,
)
from indicators import calculate_rsi, calculate_ma
from strategy_core import (
    MovingAverageParams,
    swing_strategy_signal,
    detect_market_regime,
)


@dataclass
class TradeDetail:
    """개별 거래 상세 정보"""
    side: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    entry_time: int  # 인덱스
    exit_time: int  # 인덱스
    pnl: float
    pnl_pct: float
    entry_rsi: float
    exit_rsi: float
    regime: str  # "bullish", "bearish", "sideways"
    reason: str  # 청산 이유


@dataclass
class BacktestResult:
    initial_balance: float
    final_balance: float
    total_pnl: float
    total_pnl_pct: float
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
    trade_details: List[TradeDetail]  # 개별 거래 상세 정보


# 새로운 전략 파라미터는 config.py에서 가져옴


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
) -> tuple[float, bool]:
    """
    포지션 청산 처리 (중복 코드 제거용 헬퍼 함수)
    
    Returns:
        (net_pnl, is_win): 순손익과 승리 여부
    """
    remaining_pnl = pnl_pct / 100 * balance * current_position_size
    fee = balance * current_position_size * FEE_RATE
    net_pnl = remaining_pnl - fee
    
    trades.append(net_pnl)
    is_win = net_pnl > 0
    if is_win:
        wins.append(net_pnl)
    else:
        losses.append(net_pnl)
    
    trade_details.append(TradeDetail(
        side=side,
        entry_price=entry_info.get("entry_price", entry_price),
        exit_price=exit_price,
        entry_time=entry_time,
        exit_time=exit_time,
        pnl=net_pnl,
        pnl_pct=pnl_pct,
        entry_rsi=entry_info.get("rsi", rsi),
        exit_rsi=rsi,
        regime=regime,
        reason=reason
    ))
    
    return net_pnl, is_win


def run_backtest(df: pd.DataFrame) -> BacktestResult:
    """
    백테스트 실행.
    
    :param df: OHLCV 데이터프레임 (timestamp, open, high, low, close, volume)
    :return: 백테스트 결과
    """
    # 지표 계산
    df = df.copy()
    df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
    df["ma_short"] = calculate_ma(df["close"], MA_SHORT_PERIOD)
    df["ma_long"] = calculate_ma(df["close"], MA_LONG_PERIOD)
    df["ma_50"] = calculate_ma(df["close"], MA_MID_PERIOD)
    df["ma_100"] = calculate_ma(df["close"], MA_LONGEST_PERIOD)
    
    # 볼륨 평균 계산 (20기간 이동평균)
    df["volume_ma"] = df["volume"].rolling(window=20).mean()
    
    df = df.dropna().reset_index(drop=True)
    
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
    
    ma_params = MovingAverageParams(
        short_period=MA_SHORT_PERIOD,
        long_period=MA_LONG_PERIOD,
        trend_threshold=0.005,  # 0.5%로 더 엄격하게 (횡보장 판단 강화)
    )
    
    total_rows = len(df)
    last_progress = -1
    
    # 가격 히스토리를 미리 배열로 변환 (성능 최적화)
    close_prices = df["close"].values
    
    for i, row in df.iterrows():
        price = float(row["close"])
        volume = float(row["volume"])
        avg_volume = float(row.get("volume_ma", volume))
        rsi = float(row["rsi"])
        short_ma = float(row["ma_short"])
        long_ma = float(row["ma_long"])
        
        # 손절/수익 실현 체크 (새로운 전략에 맞게)
        if has_position:
            regime_at_entry = entry_info.get("regime", "sideways")
            pnl_pct = (price - entry_price) / entry_price * LEVERAGE * 100 if is_long else (entry_price - price) / entry_price * LEVERAGE * 100
            
            if is_long:
                if price > highest_price:
                    highest_price = price
                
                # 강세장/약세장/횡보장별 익절/손절/스탑로스 적용
                if regime_at_entry == "bullish":
                    # RSI 기반 조기 청산: 하락 신호 감지 시 즉시 청산
                    if rsi <= BULLISH_EARLY_EXIT_RSI and pnl_pct < 0:
                        net_pnl, _ = _close_position(
                            "LONG", entry_info, entry_price, price,
                            entry_info.get("index", i), i, pnl_pct,
                            balance, current_position_size, rsi,
                            regime_at_entry, "RSI 하락 조기 청산",
                            trades, wins, losses, trade_details
                        )
                        balance += net_pnl
                        has_position = False
                        partial_profit_taken = False
                        trailing_stop_active = False
                        best_pnl_pct = 0.0
                        current_position_size = POSITION_SIZE_PERCENT
                        equity = balance
                        continue
                    
                    # 최고 수익률 업데이트
                    if pnl_pct > best_pnl_pct:
                        best_pnl_pct = pnl_pct
                    
                    # 트레일링 스톱 활성화 체크
                    if pnl_pct >= BULLISH_TRAILING_STOP_ACTIVATION:
                        trailing_stop_active = True
                        # 트레일링 스톱 체크: 최고 수익 대비 4% 하락 시 청산
                        if trailing_stop_active and best_pnl_pct - pnl_pct >= BULLISH_TRAILING_STOP_PCT:
                            remaining_pnl = pnl_pct / 100 * balance * current_position_size
                            fee = balance * current_position_size * FEE_RATE
                            net_pnl = remaining_pnl - fee
                            balance += net_pnl
                            trades.append(net_pnl)
                            wins.append(net_pnl)
                            trade_details.append(TradeDetail(
                                side="LONG",
                                entry_price=entry_info.get("entry_price", entry_price),
                                exit_price=price,
                                entry_time=entry_info.get("index", i),
                                exit_time=i,
                                pnl=net_pnl,
                                pnl_pct=pnl_pct,
                                entry_rsi=entry_info.get("rsi", rsi),
                                exit_rsi=rsi,
                                regime=regime_at_entry,
                                reason="트레일링 스톱"
                            ))
                            has_position = False
                            partial_profit_taken = False
                            trailing_stop_active = False
                            best_pnl_pct = 0.0
                            current_position_size = POSITION_SIZE_PERCENT
                            equity = balance
                            continue
                    
                    # 부분 익절: 8%에서 30% 청산
                    if not partial_profit_taken and pnl_pct >= BULLISH_PROFIT_TARGET_PARTIAL:
                        partial_pnl = BULLISH_PROFIT_TARGET_PARTIAL / 100 * balance * current_position_size * BULLISH_PARTIAL_EXIT_PCT
                        fee = balance * current_position_size * BULLISH_PARTIAL_EXIT_PCT * FEE_RATE
                        net_partial_pnl = partial_pnl - fee
                        balance += net_partial_pnl
                        current_position_size *= (1 - BULLISH_PARTIAL_EXIT_PCT)  # 포지션 크기 70%로 감소 (30% 청산)
                        partial_profit_taken = True
                        trailing_stop_active = True  # 트레일링 스톱 활성화
                        best_pnl_pct = pnl_pct  # 현재 수익률을 최고 수익률로 설정
                    
                    # 강세장 롱: 익절 12%
                    if pnl_pct >= BULLISH_PROFIT_TARGET:
                        remaining_pnl = pnl_pct / 100 * balance * current_position_size
                        fee = balance * current_position_size * FEE_RATE
                        net_pnl = remaining_pnl - fee
                        balance += net_pnl
                        trades.append(net_pnl)
                        wins.append(net_pnl)
                        trade_details.append(TradeDetail(
                            side="LONG",
                            entry_price=entry_info.get("entry_price", entry_price),
                            exit_price=price,
                            entry_time=entry_info.get("index", i),
                            exit_time=i,
                            pnl=net_pnl,
                            pnl_pct=pnl_pct,
                            entry_rsi=entry_info.get("rsi", rsi),
                            exit_rsi=rsi,
                            regime=regime_at_entry,
                            reason="익절"
                        ))
                        has_position = False
                        partial_profit_taken = False
                        trailing_stop_active = False
                        best_pnl_pct = 0.0
                        current_position_size = POSITION_SIZE_PERCENT
                        equity = balance
                        continue
                    
                    # 손절 4%
                    if pnl_pct <= -BULLISH_STOP_LOSS:
                        remaining_pnl = pnl_pct / 100 * balance * current_position_size
                        fee = balance * current_position_size * FEE_RATE
                        net_pnl = remaining_pnl - fee
                        balance += net_pnl
                        trades.append(net_pnl)
                        losses.append(net_pnl)
                        trade_details.append(TradeDetail(
                            side="LONG",
                            entry_price=entry_info.get("entry_price", entry_price),
                            exit_price=price,
                            entry_time=entry_info.get("index", i),
                            exit_time=i,
                            pnl=net_pnl,
                            pnl_pct=pnl_pct,
                            entry_rsi=entry_info.get("rsi", rsi),
                            exit_rsi=rsi,
                            regime=regime_at_entry,
                            reason="손절"
                        ))
                        has_position = False
                        partial_profit_taken = False
                        trailing_stop_active = False
                        best_pnl_pct = 0.0
                        current_position_size = POSITION_SIZE_PERCENT
                        equity = balance
                        continue
                    
                    # 스탑로스 5% (가격 기준)
                    stop_loss_price = entry_price * (1 - BULLISH_STOP_LOSS_PRICE / 100 / LEVERAGE)
                    if price <= stop_loss_price:
                        remaining_pnl = pnl_pct / 100 * balance * current_position_size
                        fee = balance * current_position_size * FEE_RATE
                        net_pnl = remaining_pnl - fee
                        balance += net_pnl
                        trades.append(net_pnl)
                        losses.append(net_pnl)
                        trade_details.append(TradeDetail(
                            side="LONG",
                            entry_price=entry_info.get("entry_price", entry_price),
                            exit_price=price,
                            entry_time=entry_info.get("index", i),
                            exit_time=i,
                            pnl=net_pnl,
                            pnl_pct=pnl_pct,
                            entry_rsi=entry_info.get("rsi", rsi),
                            exit_rsi=rsi,
                            regime=regime_at_entry,
                            reason="스탑로스"
                        ))
                        has_position = False
                        partial_profit_taken = False
                        trailing_stop_active = False
                        best_pnl_pct = 0.0
                        current_position_size = POSITION_SIZE_PERCENT
                        equity = balance
                        continue
                
                elif regime_at_entry == "sideways":
                    # 횡보장 롱: 진입 시점의 박스권 하단 이탈 시 즉시 손절
                    box_high_entry = entry_info.get("box_high", 0.0)
                    box_low_entry = entry_info.get("box_low", 0.0)
                    
                    if box_high_entry > 0 and box_low_entry > 0:
                        # 진입 시점의 박스권 하단 이탈 시 즉시 손절 (하단보다 1% 이상 아래로 떨어져야 손절)
                        box_low_threshold = box_low_entry * (1 - 0.01)  # 하단보다 1% 아래
                        if price < box_low_threshold:
                            remaining_pnl = pnl_pct / 100 * balance * current_position_size
                            fee = balance * current_position_size * FEE_RATE
                            net_pnl = remaining_pnl - fee
                            balance += net_pnl
                            trades.append(net_pnl)
                            losses.append(net_pnl)
                            trade_details.append(TradeDetail(
                                side="LONG",
                                entry_price=entry_info.get("entry_price", entry_price),
                                exit_price=price,
                                entry_time=entry_info.get("index", i),
                                exit_time=i,
                                pnl=net_pnl,
                                pnl_pct=pnl_pct,
                                entry_rsi=entry_info.get("rsi", rsi),
                                exit_rsi=rsi,
                                regime=regime_at_entry,
                                reason="박스권 하단 이탈"
                            ))
                            has_position = False
                            partial_profit_taken = False
                            current_position_size = POSITION_SIZE_PERCENT
                            equity = balance
                            continue
                    
                    # 기본 손절 4% (박스권 이탈이 아닌 경우)
                    if pnl_pct <= -4.0:
                        remaining_pnl = pnl_pct / 100 * balance * current_position_size
                        fee = balance * current_position_size * FEE_RATE
                        net_pnl = remaining_pnl - fee
                        balance += net_pnl
                        trades.append(net_pnl)
                        losses.append(net_pnl)
                        trade_details.append(TradeDetail(
                            side="LONG",
                            entry_price=entry_info.get("entry_price", entry_price),
                            exit_price=price,
                            entry_time=entry_info.get("index", i),
                            exit_time=i,
                            pnl=net_pnl,
                            pnl_pct=pnl_pct,
                            entry_rsi=entry_info.get("rsi", rsi),
                            exit_rsi=rsi,
                            regime=regime_at_entry,
                            reason="손절"
                        ))
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
                
                # 약세장/횡보장별 익절/손절/스탑로스 적용
                if regime_at_entry == "bearish":
                    # 최고 수익률 업데이트
                    if pnl_pct > best_pnl_pct:
                        best_pnl_pct = pnl_pct
                    
                    # 트레일링 스톱 활성화 체크
                    if pnl_pct >= BEARISH_TRAILING_STOP_ACTIVATION:
                        trailing_stop_active = True
                        # 트레일링 스톱 체크: 최고 수익 대비 3% 하락 시 청산
                        if trailing_stop_active and best_pnl_pct - pnl_pct >= BEARISH_TRAILING_STOP_PCT:
                            remaining_pnl = pnl_pct / 100 * balance * current_position_size
                            fee = balance * current_position_size * FEE_RATE
                            net_pnl = remaining_pnl - fee
                            balance += net_pnl
                            trades.append(net_pnl)
                            wins.append(net_pnl)
                            trade_details.append(TradeDetail(
                                side="SHORT",
                                entry_price=entry_info.get("entry_price", entry_price),
                                exit_price=price,
                                entry_time=entry_info.get("index", i),
                                exit_time=i,
                                pnl=net_pnl,
                                pnl_pct=pnl_pct,
                                entry_rsi=entry_info.get("rsi", rsi),
                                exit_rsi=rsi,
                                regime=regime_at_entry,
                                reason="트레일링 스톱"
                            ))
                            has_position = False
                            partial_profit_taken = False
                            trailing_stop_active = False
                            best_pnl_pct = 0.0
                            current_position_size = POSITION_SIZE_PERCENT
                            equity = balance
                            continue
                    
                    # 부분 익절: 8%에서 30% 청산
                    if not partial_profit_taken and pnl_pct >= BEARISH_PROFIT_TARGET_PARTIAL:
                        partial_pnl = BEARISH_PROFIT_TARGET_PARTIAL / 100 * balance * current_position_size * BEARISH_PARTIAL_EXIT_PCT
                        fee = balance * current_position_size * BEARISH_PARTIAL_EXIT_PCT * FEE_RATE
                        net_partial_pnl = partial_pnl - fee
                        balance += net_partial_pnl
                        current_position_size *= (1 - BEARISH_PARTIAL_EXIT_PCT)  # 포지션 크기 70%로 감소 (30% 청산)
                        partial_profit_taken = True
                        trailing_stop_active = True  # 트레일링 스톱 활성화
                        best_pnl_pct = pnl_pct  # 현재 수익률을 최고 수익률로 설정
                    
                    # 약세장 숏: 익절 12%
                    if pnl_pct >= BEARISH_PROFIT_TARGET:
                        remaining_pnl = pnl_pct / 100 * balance * current_position_size
                        fee = balance * current_position_size * FEE_RATE
                        net_pnl = remaining_pnl - fee
                        balance += net_pnl
                        trades.append(net_pnl)
                        wins.append(net_pnl)
                        trade_details.append(TradeDetail(
                            side="SHORT",
                            entry_price=entry_info.get("entry_price", entry_price),
                            exit_price=price,
                            entry_time=entry_info.get("index", i),
                            exit_time=i,
                            pnl=net_pnl,
                            pnl_pct=pnl_pct,
                            entry_rsi=entry_info.get("rsi", rsi),
                            exit_rsi=rsi,
                            regime=regime_at_entry,
                            reason="익절"
                        ))
                        has_position = False
                        partial_profit_taken = False
                        trailing_stop_active = False
                        best_pnl_pct = 0.0
                        current_position_size = POSITION_SIZE_PERCENT
                        equity = balance
                        continue
                    
                    # 손절 4%
                    if pnl_pct <= -BEARISH_STOP_LOSS:
                        remaining_pnl = pnl_pct / 100 * balance * current_position_size
                        fee = balance * current_position_size * FEE_RATE
                        net_pnl = remaining_pnl - fee
                        balance += net_pnl
                        trades.append(net_pnl)
                        losses.append(net_pnl)
                        trade_details.append(TradeDetail(
                            side="SHORT",
                            entry_price=entry_info.get("entry_price", entry_price),
                            exit_price=price,
                            entry_time=entry_info.get("index", i),
                            exit_time=i,
                            pnl=net_pnl,
                            pnl_pct=pnl_pct,
                            entry_rsi=entry_info.get("rsi", rsi),
                            exit_rsi=rsi,
                            regime=regime_at_entry,
                            reason="손절"
                        ))
                        has_position = False
                        partial_profit_taken = False
                        trailing_stop_active = False
                        best_pnl_pct = 0.0
                        current_position_size = POSITION_SIZE_PERCENT
                        equity = balance
                        continue
                    
                    # 스탑로스 5% (가격 기준)
                    stop_loss_price = entry_price * (1 + BEARISH_STOP_LOSS_PRICE / 100 / LEVERAGE)
                    if price >= stop_loss_price:
                        remaining_pnl = pnl_pct / 100 * balance * current_position_size
                        fee = balance * current_position_size * FEE_RATE
                        net_pnl = remaining_pnl - fee
                        balance += net_pnl
                        trades.append(net_pnl)
                        losses.append(net_pnl)
                        trade_details.append(TradeDetail(
                            side="SHORT",
                            entry_price=entry_info.get("entry_price", entry_price),
                            exit_price=price,
                            entry_time=entry_info.get("index", i),
                            exit_time=i,
                            pnl=net_pnl,
                            pnl_pct=pnl_pct,
                            entry_rsi=entry_info.get("rsi", rsi),
                            exit_rsi=rsi,
                            regime=regime_at_entry,
                            reason="스탑로스"
                        ))
                        has_position = False
                        partial_profit_taken = False
                        trailing_stop_active = False
                        best_pnl_pct = 0.0
                        current_position_size = POSITION_SIZE_PERCENT
                        equity = balance
                        continue
                
                elif regime_at_entry == "sideways":
                    # 횡보장 숏: 진입 시점의 박스권 상단 이탈 시 즉시 손절
                    box_high_entry = entry_info.get("box_high", 0.0)
                    box_low_entry = entry_info.get("box_low", 0.0)
                    
                    if box_high_entry > 0 and box_low_entry > 0:
                        # 진입 시점의 박스권 상단 이탈 시 즉시 손절 (상단보다 1% 이상 위로 올라가야 손절)
                        box_high_threshold = box_high_entry * (1 + 0.01)  # 상단보다 1% 위
                        if price > box_high_threshold:
                                remaining_pnl = pnl_pct / 100 * balance * current_position_size
                                fee = balance * current_position_size * FEE_RATE
                                net_pnl = remaining_pnl - fee
                                balance += net_pnl
                                trades.append(net_pnl)
                                losses.append(net_pnl)
                                trade_details.append(TradeDetail(
                                    side="SHORT",
                                    entry_price=entry_info.get("entry_price", entry_price),
                                    exit_price=price,
                                    entry_time=entry_info.get("index", i),
                                    exit_time=i,
                                    pnl=net_pnl,
                                    pnl_pct=pnl_pct,
                                    entry_rsi=entry_info.get("rsi", rsi),
                                    exit_rsi=rsi,
                                    regime=regime_at_entry,
                                    reason="박스권 상단 이탈"
                                ))
                                has_position = False
                                partial_profit_taken = False
                                trailing_stop_active = False
                                best_pnl_pct = 0.0
                                current_position_size = POSITION_SIZE_PERCENT
                                equity = balance
                                continue
                    
                    # 기본 손절 4% (박스권 이탈이 아닌 경우)
                    if pnl_pct <= -4.0:
                        remaining_pnl = pnl_pct / 100 * balance * current_position_size
                        fee = balance * current_position_size * FEE_RATE
                        net_pnl = remaining_pnl - fee
                        balance += net_pnl
                        trades.append(net_pnl)
                        losses.append(net_pnl)
                        trade_details.append(TradeDetail(
                            side="SHORT",
                            entry_price=entry_info.get("entry_price", entry_price),
                            exit_price=price,
                            entry_time=entry_info.get("index", i),
                            exit_time=i,
                            pnl=net_pnl,
                            pnl_pct=pnl_pct,
                            entry_rsi=entry_info.get("rsi", rsi),
                            exit_rsi=rsi,
                            regime=regime_at_entry,
                            reason="손절"
                        ))
                        has_position = False
                        partial_profit_taken = False
                        trailing_stop_active = False
                        best_pnl_pct = 0.0
                        current_position_size = POSITION_SIZE_PERCENT
                        equity = balance
                        continue
        
        # 전략 신호 생성
        ma_50 = float(row.get("ma_50", 0.0))
        ma_100 = float(row.get("ma_100", 0.0))
        # 박스권 판단을 위한 가격 히스토리
        if i >= SIDEWAYS_BOX_PERIOD:
            price_history_for_regime = close_prices[i - SIDEWAYS_BOX_PERIOD:i + 1].tolist()
        else:
            price_history_for_regime = close_prices[:i + 1].tolist()
        regime = detect_market_regime(short_ma, long_ma, price, ma_50, ma_100, ma_params, price_history_for_regime)
        
        # 박스권 판단을 위한 가격 히스토리 (성능 최적화: 슬라이싱 사용)
        if i >= SIDEWAYS_BOX_PERIOD:
            price_history = close_prices[i - SIDEWAYS_BOX_PERIOD:i + 1].tolist()
        else:
            price_history = close_prices[:i + 1].tolist()
        
        # 새로운 전략 시그널 생성
        signal = swing_strategy_signal(
            rsi_value=rsi,
            price=price,
            short_ma=short_ma,
            long_ma=long_ma,
            has_position=has_position,
            is_long=is_long,
            regime=regime,
            price_history=price_history,
        )
        
        # 진입/청산 처리
        if signal == "long" and not has_position:
            # 진입 시점의 박스권 저장
            box_high_entry = 0.0
            box_low_entry = 0.0
            if regime == "sideways" and len(price_history) >= SIDEWAYS_BOX_PERIOD:
                box_high_entry = max(price_history[-SIDEWAYS_BOX_PERIOD:])
                box_low_entry = min(price_history[-SIDEWAYS_BOX_PERIOD:])
            
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
            }
        elif signal == "short" and not has_position:
            # 진입 시점의 박스권 저장
            box_high_entry = 0.0
            box_low_entry = 0.0
            if regime == "sideways" and len(price_history) >= SIDEWAYS_BOX_PERIOD:
                box_high_entry = max(price_history[-SIDEWAYS_BOX_PERIOD:])
                box_low_entry = min(price_history[-SIDEWAYS_BOX_PERIOD:])
            
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
            balance += net_pnl
            trades.append(net_pnl)
            if net_pnl > 0:
                wins.append(net_pnl)
            else:
                losses.append(net_pnl)
            trade_details.append(TradeDetail(
                side="LONG" if is_long else "SHORT",
                entry_price=entry_info.get("entry_price", entry_price),
                exit_price=price,
                entry_time=entry_info.get("index", i),
                exit_time=i,
                pnl=net_pnl,
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
    
    # 최종 포지션 청산
    if has_position:
        final_price = float(df.iloc[-1]["close"])
        if is_long:
            pnl_pct = (final_price - entry_price) / entry_price * LEVERAGE * 100
        else:
            pnl_pct = (entry_price - final_price) / entry_price * LEVERAGE * 100
        
        gross_pnl = pnl_pct / 100 * balance * current_position_size
        fee = balance * current_position_size * FEE_RATE
        net_pnl = gross_pnl - fee
        balance += net_pnl
        trades.append(net_pnl)
        if net_pnl > 0:
            wins.append(net_pnl)
        else:
            losses.append(net_pnl)
        equity = balance
    
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
    print(f"{result.total_pnl:+.2f} USDT ({result.total_pnl_pct:+.2f}%), 거래: {result.num_trades}회, 승률: {result.win_rate:.2f}%")
