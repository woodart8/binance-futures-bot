"""
바이낸스 선물(USDS-M) 안정적 트레이딩 모의선물(페이퍼 트레이딩) 봇

- 실제 주문은 전혀 보내지 않고, 가상의 USDT 잔고로만 손익을 계산한다.
- 안정적인 전략: 레버리지 3배, 보수적 포지션 크기(5%), 강화된 리스크 관리
- 시장 상태(강세장/약세장/횡보장)를 판단하여 적절한 전략을 자동 선택한다.
  - 강세장/약세장: 이동평균선(8/15) 기반 추세 추종 전략
  - 횡보장: RSI(35/65) + MACD(10/21/7) 결합 전략
- 슬리피지 및 실제 거래 비용 고려
- 일일/주간 손실 한도로 자금 보호

실행 예시:
    python paper_trading.py
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import ccxt
import pandas as pd

from config import (
    FEE_RATE,
    INITIAL_BALANCE,
    LEVERAGE,
    POSITION_SIZE_PERCENT,
    RSI_PERIOD,
    SLIPPAGE_PCT,
    SYMBOL,
    TIMEFRAME,
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
    MA_MID_PERIOD,
    MA_LONGEST_PERIOD,
)
from indicators import calculate_rsi, calculate_ma
from strategy_core import (
    MovingAverageParams,
    swing_strategy_signal,
    detect_market_regime,
    MarketRegime,
)
from trade_logger import log_trade


RISK_PER_TRADE = POSITION_SIZE_PERCENT

# 전략 파라미터 (config.py에서 가져옴)
from config import MA_SHORT_PERIOD, MA_LONG_PERIOD

# 체크 주기 (초)
CHECK_INTERVAL = 30  # 30초마다 체크 (안정적인 모니터링)


def log(message: str, level: str = "INFO") -> None:
    """간단 로깅 함수."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {message}")


def get_public_exchange() -> ccxt.binanceusdm:
    """
    퍼블릭 데이터 전용 바이낸스 선물 인스턴스.
    (API 키 없이 OHLCV, 티커 조회만 사용)
    """
    return ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})


def fetch_ohlcv(exchange: ccxt.binanceusdm, limit: int = 300) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


@dataclass
class PaperState:
    balance: float
    equity: float
    has_long_position: bool
    has_short_position: bool
    entry_price: float
    position_size: float
    peak_equity: float
    max_drawdown: float
    highest_price: float  # 최고가 추적 (롱 포지션)
    lowest_price: float   # 최저가 추적 (숏 포지션)
    partial_profit_taken: bool  # 부분 익절 여부
    trailing_stop_active: bool  # 트레일링 스톱 활성화 여부
    best_pnl_pct: float  # 최고 수익률 추적 (트레일링 스톱용)
    partial_loss_taken: bool  # 부분 손절 여부
    partial_entry_taken: bool  # 부분 진입 여부 (사용하지 않음, 호환성 유지)
    first_entry_price: float  # 첫 진입 가격 (사용하지 않음, 호환성 유지)
    entry_regime: str  # 진입 시점의 시장 상태
    position_entry_time: float  # 포지션 진입 시간
    box_high: float  # 진입 시점의 박스권 상단 (횡보장용)
    box_low: float   # 진입 시점의 박스권 하단 (횡보장용)


def init_state() -> PaperState:
    balance = INITIAL_BALANCE
    import time
    current_time = time.time()
    return PaperState(
        balance=balance,
        equity=balance,
        has_long_position=False,
        has_short_position=False,
        entry_price=0.0,
        position_size=0.0,
        peak_equity=balance,
        max_drawdown=0.0,
        highest_price=0.0,
        lowest_price=float("inf"),
        partial_profit_taken=False,
        trailing_stop_active=False,
        best_pnl_pct=0.0,
        partial_loss_taken=False,
        partial_entry_taken=False,
        first_entry_price=0.0,
        entry_regime="",
        position_entry_time=0.0,
        box_high=0.0,
        box_low=0.0,
    )


def close_position(
    state: PaperState,
    candle: pd.Series,
    side: str,
    reason: str,
) -> None:
    """포지션 청산 처리."""
    import time
    
    price = float(candle["close"])
    entry_price = state.entry_price  # 청산 전에 저장
    
    if side == "LONG":
        pnl_pct = (price - entry_price) / entry_price * LEVERAGE
    else:  # SHORT
        pnl_pct = (entry_price - price) / entry_price * LEVERAGE
    
    # 슬리피지 고려 (실제 거래 비용)
    slippage_cost = state.balance * RISK_PER_TRADE * SLIPPAGE_PCT / 100
    
    gross_pnl = pnl_pct * state.balance * RISK_PER_TRADE
    fee = state.balance * RISK_PER_TRADE * FEE_RATE
    net_pnl = gross_pnl - fee - slippage_cost
    state.balance += net_pnl
    
    
    # 포지션 초기화
    state.has_long_position = False
    state.has_short_position = False
    state.position_size = 0.0
    state.entry_price = 0.0
    state.highest_price = 0.0
    state.lowest_price = float("inf")
    state.partial_profit_taken = False
    state.partial_loss_taken = False
    state.partial_entry_taken = False
    state.trailing_stop_active = False
    state.best_pnl_pct = 0.0
    state.first_entry_price = 0.0
    state.entry_regime = ""
    state.position_entry_time = 0.0
    state.box_high = 0.0
    state.box_low = 0.0
    
    # 트레이드 기록
    log_trade(
        side=f"PAPER_{side}",
        entry_price=entry_price,
        exit_price=price,
        pnl=net_pnl,
        balance_after=state.balance,
        meta={
            "timeframe": TIMEFRAME,
            "symbol": SYMBOL,
            "mode": "paper",
            "reason": reason,
        },
    )
    
    # PNL 및 ROE 계산
    total_pnl = state.equity - INITIAL_BALANCE
    roe = (total_pnl / INITIAL_BALANCE * 100) if INITIAL_BALANCE > 0 else 0.0
    
    log(f"{side} 청산 ({reason}): 진입가={entry_price:.2f}, 청산가={price:.2f}, 수익률={pnl_pct:.2f}%, 실현손익={net_pnl:+.2f} USDT, 잔고={state.balance:.2f} USDT, 총PNL={total_pnl:+.2f} USDT, ROE={roe:+.2f}%")




def open_position(
    state: PaperState,
    price: float,
    side: str,
    reason: str,
    regime: str = "",
    price_history: Optional[list] = None,
) -> None:
    """포지션 진입 처리."""
    import time
    
    current_time = time.time()
    
    # 포지션 크기 계산
    trade_capital = state.balance * RISK_PER_TRADE
    if trade_capital <= 0:
        return
    
    fee = trade_capital * FEE_RATE
    slippage_cost = trade_capital * SLIPPAGE_PCT / 100  # 슬리피지 고려
    trade_capital_after_fee = trade_capital - fee - slippage_cost
    position_size = trade_capital_after_fee / price
    
    # 진입 시점의 박스권 저장 (횡보장일 경우)
    box_high_entry = 0.0
    box_low_entry = 0.0
    if regime == "sideways" and price_history is not None and len(price_history) >= SIDEWAYS_BOX_PERIOD:
        box_high_entry = max(price_history[-SIDEWAYS_BOX_PERIOD:])
        box_low_entry = min(price_history[-SIDEWAYS_BOX_PERIOD:])
    
    if side == "LONG":
        state.has_long_position = True
    else:  # SHORT
        state.has_short_position = True
    
    state.entry_price = price
    state.first_entry_price = price
    state.position_size = position_size
    state.position_entry_time = current_time
    state.partial_entry_taken = False
    state.partial_profit_taken = False
    state.partial_loss_taken = False
    state.trailing_stop_active = False
    state.best_pnl_pct = 0.0
    state.entry_regime = regime  # 진입 시점의 시장 상태 저장
    state.box_high = box_high_entry
    state.box_low = box_low_entry
    
    # 가격 추적 초기화
    if side == "LONG":
        state.highest_price = price
    else:  # SHORT
        state.lowest_price = price
    
    # PNL 및 ROE 계산
    total_pnl = state.equity - INITIAL_BALANCE
    roe = (total_pnl / INITIAL_BALANCE * 100) if INITIAL_BALANCE > 0 else 0.0
    
    log(f"{side} 진입 ({reason}): 가격={price:.2f}, 잔고={state.balance:.2f} USDT, PNL={total_pnl:+.2f} USDT, ROE={roe:+.2f}%")


def check_scalp_stop_loss_and_profit(state: PaperState, current_price: float, candle: pd.Series) -> bool:
    """
    스캘핑용 손절, 수익 실현, 트레일링 스톱 체크.
    
    :param state: 현재 상태
    :param current_price: 현재 가격
    :param candle: 현재 캔들 데이터
    :return: 포지션이 청산되었으면 True, 아니면 False
    """
    if state.has_long_position:
        # 최고가 업데이트
        if current_price > state.highest_price:
            state.highest_price = current_price
        
        # 수익률 계산
        pnl_pct = (current_price - state.entry_price) / state.entry_price * LEVERAGE * 100
        
        # 강세장/약세장/횡보장별 익절/손절/스탑로스 적용
        if state.entry_regime == "bullish":
            rsi = float(candle.get("rsi", 50.0))
            
            # RSI 기반 조기 청산: 하락 신호 감지 시 즉시 청산
            if rsi <= BULLISH_EARLY_EXIT_RSI and pnl_pct < 0:
                close_position(state, candle, "LONG", f"RSI 하락 조기 청산 ({pnl_pct:.2f}%)")
                return True
            
            # 최고 수익률 업데이트
            if pnl_pct > state.best_pnl_pct:
                state.best_pnl_pct = pnl_pct
            
            # 트레일링 스톱 활성화 체크
            if pnl_pct >= BULLISH_TRAILING_STOP_ACTIVATION:
                state.trailing_stop_active = True
                # 트레일링 스톱 체크: 최고 수익 대비 4% 하락 시 청산
                if state.trailing_stop_active and state.best_pnl_pct - pnl_pct >= BULLISH_TRAILING_STOP_PCT:
                    close_position(state, candle, "LONG", f"트레일링 스톱 ({pnl_pct:.2f}%)")
                    return True
            
            # 부분 익절: 8%에서 30% 청산
            if not state.partial_profit_taken and pnl_pct >= BULLISH_PROFIT_TARGET_PARTIAL:
                partial_pnl = BULLISH_PROFIT_TARGET_PARTIAL / 100 * state.balance * POSITION_SIZE_PERCENT * BULLISH_PARTIAL_EXIT_PCT
                fee = state.balance * POSITION_SIZE_PERCENT * BULLISH_PARTIAL_EXIT_PCT * FEE_RATE
                net_partial_pnl = partial_pnl - fee
                state.balance += net_partial_pnl
                state.position_size *= (1 - BULLISH_PARTIAL_EXIT_PCT)  # 포지션 크기 70%로 감소
                state.partial_profit_taken = True
                state.trailing_stop_active = True  # 트레일링 스톱 활성화
                state.best_pnl_pct = pnl_pct  # 현재 수익률을 최고 수익률로 설정
            
            # 강세장 롱: 익절 12%
            if pnl_pct >= BULLISH_PROFIT_TARGET:
                close_position(state, candle, "LONG", f"익절 ({pnl_pct:.2f}%)")
                return True
            
            # 손절 4%
            if pnl_pct <= -BULLISH_STOP_LOSS:
                close_position(state, candle, "LONG", f"손절 ({pnl_pct:.2f}%)")
                return True
            
            # 스탑로스 5% (가격 기준)
            stop_loss_price = state.entry_price * (1 - BULLISH_STOP_LOSS_PRICE / 100 / LEVERAGE)
            if current_price <= stop_loss_price:
                close_position(state, candle, "LONG", f"스탑로스 ({pnl_pct:.2f}%)")
                return True
        
        elif state.entry_regime == "sideways":
            # 횡보장 롱: 박스권 하단 이탈 시 즉시 손절 (하단보다 1% 이상 아래로 떨어져야 손절)
            if state.box_high > 0 and state.box_low > 0:
                box_low_threshold = state.box_low * (1 - 0.01)  # 하단보다 1% 아래
                if current_price < box_low_threshold:
                    close_position(state, candle, "LONG", f"박스권 하단 이탈 ({pnl_pct:.2f}%)")
                    return True
            
            # 기본 손절 4%
            if pnl_pct <= -4.0:
                close_position(state, candle, "LONG", f"손절 ({pnl_pct:.2f}%)")
                return True
        
        else:
            # 기타 경우 기본 손절/익절 (backtest와 동일하게)
            if pnl_pct >= BULLISH_PROFIT_TARGET:
                close_position(state, candle, "LONG", f"익절 ({pnl_pct:.2f}%)")
                return True
            if pnl_pct <= -BULLISH_STOP_LOSS:
                close_position(state, candle, "LONG", f"손절 ({pnl_pct:.2f}%)")
                return True
        
        # 최대 보유 시간 체크
        import time
        if state.position_entry_time > 0:
            hold_time_min = (time.time() - state.position_entry_time) / 60
            if False:  # 최대 보유 시간 체크 비활성화
                close_position(state, candle, "LONG", f"최대 보유 시간 초과 ({hold_time_min:.0f}분)")
                return True
    
    elif state.has_short_position:
        # 최저가 업데이트
        if current_price < state.lowest_price:
            state.lowest_price = current_price
        
        # 수익률 계산
        pnl_pct = (state.entry_price - current_price) / state.entry_price * LEVERAGE * 100
        
        # 약세장/횡보장별 익절/손절/스탑로스 적용
        if state.entry_regime == "bearish":
            rsi = float(candle.get("rsi", 50.0))
            
            # RSI 기반 조기 청산: 반등 신호 감지 시 즉시 청산
            if rsi >= BEARISH_EARLY_EXIT_RSI and pnl_pct < 0:
                close_position(state, candle, "SHORT", f"RSI 반등 조기 청산 ({pnl_pct:.2f}%)")
                return True
            
            # 최고 수익률 업데이트
            if pnl_pct > state.best_pnl_pct:
                state.best_pnl_pct = pnl_pct
            
            # 트레일링 스톱 활성화 체크
            if pnl_pct >= BEARISH_TRAILING_STOP_ACTIVATION:
                state.trailing_stop_active = True
                # 트레일링 스톱 체크: 최고 수익 대비 4% 하락 시 청산
                if state.trailing_stop_active and state.best_pnl_pct - pnl_pct >= BEARISH_TRAILING_STOP_PCT:
                    close_position(state, candle, "SHORT", f"트레일링 스톱 ({pnl_pct:.2f}%)")
                    return True
            
            # 부분 익절: 8%에서 30% 청산
            if not state.partial_profit_taken and pnl_pct >= BEARISH_PROFIT_TARGET_PARTIAL:
                partial_pnl = BEARISH_PROFIT_TARGET_PARTIAL / 100 * state.balance * POSITION_SIZE_PERCENT * BEARISH_PARTIAL_EXIT_PCT
                fee = state.balance * POSITION_SIZE_PERCENT * BEARISH_PARTIAL_EXIT_PCT * FEE_RATE
                net_partial_pnl = partial_pnl - fee
                state.balance += net_partial_pnl
                state.position_size *= (1 - BEARISH_PARTIAL_EXIT_PCT)  # 포지션 크기 70%로 감소
                state.partial_profit_taken = True
                state.trailing_stop_active = True  # 트레일링 스톱 활성화
                state.best_pnl_pct = pnl_pct  # 현재 수익률을 최고 수익률로 설정
            
            # 약세장 숏: 익절 12%
            if pnl_pct >= BEARISH_PROFIT_TARGET:
                close_position(state, candle, "SHORT", f"익절 ({pnl_pct:.2f}%)")
                return True
            
            # 손절 4%
            if pnl_pct <= -BEARISH_STOP_LOSS:
                close_position(state, candle, "SHORT", f"손절 ({pnl_pct:.2f}%)")
                return True
            
            # 스탑로스 5% (가격 기준)
            stop_loss_price = state.entry_price * (1 + BEARISH_STOP_LOSS_PRICE / 100 / LEVERAGE)
            if current_price >= stop_loss_price:
                close_position(state, candle, "SHORT", f"스탑로스 ({pnl_pct:.2f}%)")
                return True
        
        elif state.entry_regime == "sideways":
            # 횡보장 숏: 박스권 상단 이탈 시 즉시 손절 (상단보다 1% 이상 위로 올라가야 손절)
            if state.box_high > 0 and state.box_low > 0:
                box_high_threshold = state.box_high * (1 + 0.01)  # 상단보다 1% 위
                if current_price > box_high_threshold:
                    close_position(state, candle, "SHORT", f"박스권 상단 이탈 ({pnl_pct:.2f}%)")
                    return True
            
            # 기본 손절 4%
            if pnl_pct <= -4.0:
                close_position(state, candle, "SHORT", f"손절 ({pnl_pct:.2f}%)")
                return True
        
        else:
            # 기타 경우 기본 손절/익절 (backtest와 동일하게)
            if pnl_pct >= BEARISH_PROFIT_TARGET:
                close_position(state, candle, "SHORT", f"익절 ({pnl_pct:.2f}%)")
                return True
            if pnl_pct <= -BEARISH_STOP_LOSS:
                close_position(state, candle, "SHORT", f"손절 ({pnl_pct:.2f}%)")
                return True
        
        # 최대 보유 시간 체크
        import time
        if state.position_entry_time > 0:
            hold_time_min = (time.time() - state.position_entry_time) / 60
            if False:  # 최대 보유 시간 체크 비활성화
                close_position(state, candle, "SHORT", f"최대 보유 시간 초과 ({hold_time_min:.0f}분)")
                return True
    
    return False


def apply_strategy_on_candle(state: PaperState, candle: pd.Series, df: Optional[pd.DataFrame] = None) -> None:
    """시장 상태에 따라 적절한 전략을 선택하여 적용."""
    price = float(candle["close"])
    rsi = float(candle["rsi"])
    short_ma = float(candle["ma_short"])
    long_ma = float(candle["ma_long"])
    
    # 시장 상태 판단
    ma_50 = float(candle.get("ma_50", 0.0))
    ma_100 = float(candle.get("ma_100", 0.0))
    ma_params = MovingAverageParams(
        short_period=MA_SHORT_PERIOD,
        long_period=MA_LONG_PERIOD,
        trend_threshold=0.01,
    )
    # 박스권 판단을 위한 가격 히스토리 (SIDEWAYS_BOX_PERIOD 사용)
    price_history_for_regime = None
    if df is not None and len(df) >= SIDEWAYS_BOX_PERIOD:
        price_history_for_regime = df["close"].tail(SIDEWAYS_BOX_PERIOD + 1).tolist()  # 현재 포함
    regime = detect_market_regime(short_ma, long_ma, price, ma_50, ma_100, ma_params, price_history_for_regime)
    
    has_position = state.has_long_position or state.has_short_position
    is_long = state.has_long_position
    
    # 박스권 판단을 위한 가격 히스토리 (SIDEWAYS_BOX_PERIOD 사용)
    price_history = None
    if df is not None and len(df) >= SIDEWAYS_BOX_PERIOD:
        # 최근 SIDEWAYS_BOX_PERIOD개 캔들의 가격 히스토리
        price_history = df["close"].tail(SIDEWAYS_BOX_PERIOD + 1).tolist()  # 현재 포함
    
    # 스윙 전략 시그널 생성 (RSI + MA만 사용)
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
    
    # 전략 시그널에 따른 진입/청산 처리
    if signal == "long" and not has_position:
        regime_kr = {"bullish": "강세장", "bearish": "약세장", "sideways": "횡보장"}[regime]
        open_position(state, price, "LONG", f"스윙 전략 ({regime_kr})", regime, price_history)
    elif signal == "short" and not has_position:
        regime_kr = {"bullish": "강세장", "bearish": "약세장", "sideways": "횡보장"}[regime]
        open_position(state, price, "SHORT", f"스윙 전략 ({regime_kr})", regime, price_history)
    elif signal == "flat" and has_position:
        side = "LONG" if is_long else "SHORT"
        regime_kr = {"bullish": "강세장", "bearish": "약세장", "sideways": "횡보장"}[regime]
        close_position(state, candle, side, f"스윙 전략 청산 ({regime_kr})")
    
    # 평가손익 계산 (손절/수익 실현은 별도 함수에서 처리)
    if state.has_long_position:
        unrealized = (
            (price - state.entry_price)
            / state.entry_price
            * LEVERAGE
            * state.balance
            * RISK_PER_TRADE
        )
        equity = state.balance + unrealized
    elif state.has_short_position:
        unrealized = (
            (state.entry_price - price)
            / state.entry_price
            * LEVERAGE
            * state.balance
            * RISK_PER_TRADE
        )
        equity = state.balance + unrealized
    else:
        equity = state.balance
    
    state.equity = equity
    if equity > state.peak_equity:
        state.peak_equity = equity
    
    drawdown = (
        (state.peak_equity - equity) / state.peak_equity if state.peak_equity > 0 else 0.0
    )
    state.max_drawdown = max(state.max_drawdown, drawdown)


def main() -> None:
    exchange = get_public_exchange()
    state = init_state()

    log(f"페이퍼 트레이딩 시작: 초기 잔고={INITIAL_BALANCE:.2f} USDT")
    
    last_candle_time = None

    try:
        while True:
            # 충분한 데이터 확보
            limit = max(RSI_PERIOD, MA_LONGEST_PERIOD) + 100
            df = fetch_ohlcv(exchange, limit=limit)
            
            # 지표 계산
            df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
            df["ma_short"] = calculate_ma(df["close"], MA_SHORT_PERIOD)
            df["ma_long"] = calculate_ma(df["close"], MA_LONG_PERIOD)
            df["ma_50"] = calculate_ma(df["close"], MA_MID_PERIOD)
            df["ma_100"] = calculate_ma(df["close"], MA_LONGEST_PERIOD)
            
            # 볼륨 평균 계산 (20기간 이동평균)
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            
            df = df.dropna().reset_index(drop=True)

            latest = df.iloc[-1]
            latest_time = latest["timestamp"]
            current_price = float(latest["close"])

            # 포지션이 있으면 매번 손절/수익 실현 체크
            if state.has_long_position or state.has_short_position:
                if check_scalp_stop_loss_and_profit(state, current_price, latest):
                    # 손절 또는 수익 실현으로 포지션 청산됨
                    pos_status = "NONE"
                    total_pnl = state.equity - INITIAL_BALANCE
                    roe = (total_pnl / INITIAL_BALANCE * 100) if INITIAL_BALANCE > 0 else 0.0
                    
                    time.sleep(CHECK_INTERVAL)
                    continue

            # 아직 첫 루프라면, 마지막 캔들을 기준으로만 시작
            if last_candle_time is None:
                last_candle_time = latest_time
                pos_status = (
                    "LONG" if state.has_long_position
                    else ("SHORT" if state.has_short_position else "NONE")
                )
                total_pnl = state.equity - INITIAL_BALANCE
                roe = (total_pnl / INITIAL_BALANCE * 100) if INITIAL_BALANCE > 0 else 0.0
                log(f"초기화 완료: 포지션={pos_status}, 가격={current_price:.2f}, 잔고={state.balance:.2f} USDT, Equity={state.equity:.2f} USDT, 총PNL={total_pnl:+.2f} USDT, ROE={roe:+.2f}%")
            # 새로운 캔들이 완성되었을 때만 전략 적용
            elif latest_time > last_candle_time:
                price = float(latest["close"])
                rsi = float(latest["rsi"])
                short_ma = float(latest["ma_short"])
                long_ma = float(latest["ma_long"])
                
                # 시장 상태 판단
                ma_50 = float(latest.get("ma_50", 0.0))
                ma_100 = float(latest.get("ma_100", 0.0))
                ma_params = MovingAverageParams(
                    short_period=MA_SHORT_PERIOD,
                    long_period=MA_LONG_PERIOD,
                )
                # 박스권 판단을 위한 가격 히스토리
                price_history_for_regime = None
                if len(df) >= SIDEWAYS_BOX_PERIOD:
                    price_history_for_regime = df["close"].tail(SIDEWAYS_BOX_PERIOD + 1).tolist()
                regime = detect_market_regime(short_ma, long_ma, price, ma_50, ma_100, ma_params, price_history_for_regime)
                
                apply_strategy_on_candle(state, latest, df)
                last_candle_time = latest_time
                
                pos_status = (
                    "LONG" if state.has_long_position
                    else ("SHORT" if state.has_short_position else "NONE")
                )
                
                # PNL 및 ROE 계산
                total_pnl = state.equity - INITIAL_BALANCE
                roe = (total_pnl / INITIAL_BALANCE * 100) if INITIAL_BALANCE > 0 else 0.0
                
                # 포지션이 있을 때 미실현 손익 계산
                unrealized_pnl_pct = 0.0
                if state.has_long_position:
                    unrealized_pnl_pct = (price - state.entry_price) / state.entry_price * LEVERAGE * 100
                elif state.has_short_position:
                    unrealized_pnl_pct = (state.entry_price - price) / state.entry_price * LEVERAGE * 100
                
                # 현재 상태 출력
                log(f"상태: 포지션={pos_status}, 가격={price:.2f}, 잔고={state.balance:.2f} USDT, Equity={state.equity:.2f} USDT, 총PNL={total_pnl:+.2f} USDT, ROE={roe:+.2f}%" + (f", 미실현손익={unrealized_pnl_pct:+.2f}%" if unrealized_pnl_pct != 0 else ""))

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        total_pnl = state.equity - INITIAL_BALANCE
        roe = (total_pnl / INITIAL_BALANCE * 100) if INITIAL_BALANCE > 0 else 0.0
        log(f"프로그램 종료: 최종 잔고={state.balance:.2f} USDT, 총PNL={total_pnl:+.2f} USDT, ROE={roe:+.2f}%")
    except Exception as e:
        log(f"오류 발생: {e}", "ERROR")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
