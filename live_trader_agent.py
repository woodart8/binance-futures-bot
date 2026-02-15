"""실거래 에이전트. .env에 API_KEY, SECRET_KEY 필요."""

import sys
import time
from typing import Optional

# OHLCV 조회 재시도(3회)까지 모두 실패가 이 횟수만큼 연속되면 프로세스 종료 (무한 루프 방지)
OHLCV_CONSECUTIVE_FAILURE_LIMIT = 30

import pandas as pd

from config import (
    CONSECUTIVE_LOSS_LIMIT,
    DAILY_LOSS_LIMIT_PCT,
    LEVERAGE,
    POSITION_SIZE_PERCENT,
    RSI_PERIOD,
    MA_SHORT_PERIOD,
    MA_LONG_PERIOD,
    MA_MID_PERIOD,
    MA_LONGEST_PERIOD,
    SIDEWAYS_BOX_PERIOD,
    SIDEWAYS_PROFIT_TARGET,
    SIDEWAYS_STOP_LOSS_PRICE,
    SYMBOL,
    TIMEFRAME,
    REGIME_LOOKBACK_15M,
    TREND_PROFIT_TARGET,
    TREND_STOP_LOSS_PRICE,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
)
from exchange_client import get_private_exchange
from data import fetch_ohlcv, compute_regime_15m
from indicators import calculate_rsi, calculate_ma, calculate_macd
from chart_patterns import detect_chart_pattern, PATTERN_LOOKBACK
from strategy_core import REGIME_KR, swing_strategy_signal
from exit_logic import check_long_exit, check_short_exit
from trade_logger import log_trade
from logger import log


def get_balance_usdt(exchange) -> float:
    balance = exchange.fetch_balance()
    usdt_info = balance.get("USDT", {})
    free = float(usdt_info.get("free", 0) or 0)
    return free


def set_leverage(exchange) -> None:
    market = exchange.market(SYMBOL)
    exchange.set_leverage(LEVERAGE, market["id"])


def main() -> None:
    exchange = get_private_exchange()
    set_leverage(exchange)

    has_position = False
    is_long = False
    entry_price = 0.0
    entry_balance = 0.0
    entry_regime = ""
    box_high_entry = 0.0
    box_low_entry = 0.0
    highest_price = 0.0
    lowest_price = float("inf")
    trailing_stop_active = False
    best_pnl_pct = 0.0
    pattern_type = ""
    pattern_target = 0.0
    pattern_stop = 0.0

    # 일일 손실 한도 / 연속 손실 추적
    daily_start_balance = 0.0
    daily_start_date = ""
    consecutive_loss_count = 0

    last_candle_time = None
    ohlcv_failure_count = 0

    try:
        while True:
            try:
                # 충분한 데이터 확보 (15분봉 24시간 = 96*3 = 288개 5분봉 필요)
                limit = max(RSI_PERIOD, MA_LONGEST_PERIOD, REGIME_LOOKBACK_15M * 3, PATTERN_LOOKBACK * 3) + 100
                df = fetch_ohlcv(exchange, limit=limit)
                ohlcv_failure_count = 0
            except Exception as e:
                ohlcv_failure_count += 1
                log(f"OHLCV 조회 실패 ({ohlcv_failure_count}/{OHLCV_CONSECUTIVE_FAILURE_LIMIT}): {e}", "ERROR")
                if ohlcv_failure_count >= OHLCV_CONSECUTIVE_FAILURE_LIMIT:
                    log(f"OHLCV 조회가 {OHLCV_CONSECUTIVE_FAILURE_LIMIT}회 연속 실패하여 프로세스를 종료합니다. 네트워크/API를 확인 후 재실행하세요.", "ERROR")
                    sys.exit(1)
                time.sleep(60)
                continue
            
            # 지표 계산
            df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
            macd_line, signal_line, _ = calculate_macd(df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            df["macd_line"] = macd_line
            df["macd_signal"] = signal_line
            df["ma_short"] = calculate_ma(df["close"], MA_SHORT_PERIOD)
            df["ma_long"] = calculate_ma(df["close"], MA_LONG_PERIOD)
            df["ma_50"] = calculate_ma(df["close"], MA_MID_PERIOD)
            df["ma_100"] = calculate_ma(df["close"], MA_LONGEST_PERIOD)
            
            df = df.dropna().reset_index(drop=True)

            latest = df.iloc[-1]
            latest_time = latest["timestamp"]
            price = float(latest["close"])
            rsi = float(latest["rsi"])
            short_ma = float(latest["ma_short"])
            long_ma = float(latest["ma_long"])
            ma_50 = float(latest["ma_50"])
            ma_100 = float(latest["ma_100"])

            if last_candle_time is None:
                last_candle_time = latest_time
                try:
                    bal = get_balance_usdt(exchange)
                except Exception:
                    bal = 0.0
                log(f"[시작] 가격={price:.2f} 잔고={bal:.2f}")
            elif latest_time > last_candle_time:

                # regime: 15분봉 24시간 기준 (MA 정배열/역배열, 박스권)
                rsi_15m, macd_line_15m, macd_signal_15m = None, None, None
                if len(df) >= REGIME_LOOKBACK_15M * 3:
                    regime, short_ma_15m, long_ma_15m, ma_50_15m, ma_100_15m, price_history_15m, rsi_15m, macd_line_15m, macd_signal_15m = compute_regime_15m(df, price)
                else:
                    regime, short_ma_15m, long_ma_15m, ma_50_15m, ma_100_15m, price_history_15m, rsi_15m, macd_line_15m, macd_signal_15m = "neutral", 0.0, 0.0, 0.0, 0.0, [], None, None, None
                # 진입/청산용 price_history: 5분봉 (횡보 시 regime_*로 15분봉 전달)
                price_history = df["close"].tail(SIDEWAYS_BOX_PERIOD + 1).tolist() if len(df) >= SIDEWAYS_BOX_PERIOD else df["close"].tolist()

                # 손절/수익 실현 체크 (backtest.py와 동일한 로직)
                signal = None
                if has_position:
                    pnl_pct = (price - entry_price) / entry_price * LEVERAGE * 100 if is_long else (entry_price - price) / entry_price * LEVERAGE * 100
                    
                    # 차트 패턴 진입 시 패턴 정석 익절/손절만 적용 (config 퍼센트 무시)
                    if pattern_target > 0 and pattern_stop > 0:
                        if is_long:
                            if price >= pattern_target:
                                signal = "flat"
                            elif price <= pattern_stop:
                                signal = "flat"
                        else:  # SHORT
                            if price <= pattern_target:
                                signal = "flat"
                            elif price >= pattern_stop:
                                signal = "flat"
                        # 패턴 대기 중이면 signal 그대로, regime 기반 로직 스킵
                    
                    if signal is None and not (pattern_target > 0 and pattern_stop > 0) and is_long:
                        if price > highest_price:
                            highest_price = price
                        if pnl_pct > best_pnl_pct:
                            best_pnl_pct = pnl_pct

                        regime = entry_regime or ""
                        reason = check_long_exit(
                            regime=regime, pnl_pct=pnl_pct, rsi=rsi, price=price,
                            entry_price=entry_price, best_pnl_pct=best_pnl_pct,
                            box_high=box_high_entry or 0, box_low=box_low_entry or 0,
                        )
                        if reason:
                            signal = "flat"
                    elif signal is None and not (pattern_target > 0 and pattern_stop > 0) and not is_long:  # SHORT
                        if price < lowest_price:
                            lowest_price = price
                        if pnl_pct > best_pnl_pct:
                            best_pnl_pct = pnl_pct

                        regime = entry_regime or ""
                        reason = check_short_exit(
                            regime=regime, pnl_pct=pnl_pct, rsi=rsi, price=price,
                            entry_price=entry_price, best_pnl_pct=best_pnl_pct,
                            box_high=box_high_entry or 0, box_low=box_low_entry or 0,
                        )
                        if reason:
                            signal = "flat"
                else:
                    # 진입 신호 생성 (backtest.py와 동일, 추세추종 단타)
                    use_15m = len(price_history_15m) >= REGIME_LOOKBACK_15M
                    rsi_prev = float(df["rsi"].iloc[-2]) if len(df) >= 2 else None
                    open_prev = float(df["open"].iloc[-2]) if len(df) >= 2 else None
                    close_prev = float(df["close"].iloc[-2]) if len(df) >= 2 else None
                    open_curr = float(latest["open"])
                    rsi_use = rsi
                    regime_price_hist = price_history_15m if (use_15m and regime == "sideways") else None
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
                        regime_short_ma=short_ma_15m if use_15m else None,
                        regime_long_ma=long_ma_15m if use_15m else None,
                        regime_ma_50=ma_50_15m if use_15m else None,
                        regime_ma_100=ma_100_15m if use_15m else None,
                        regime_price_history=regime_price_hist,
                        macd_line=None,
                        macd_signal=None,
                    )


                # 잔고 조회
                try:
                    balance = get_balance_usdt(exchange)
                except Exception as e:
                    log(f"잔고 조회 실패: {e}", "ERROR")
                    time.sleep(60)
                    continue

                # 일일 손실 한도: 날짜 변경 시 daily_start_balance 갱신
                current_date = latest_time.strftime("%Y-%m-%d") if hasattr(latest_time, "strftime") else str(latest_time)[:10]
                if daily_start_date != current_date:
                    daily_start_balance = balance
                    daily_start_date = current_date
                    consecutive_loss_count = 0

                # 일일 손실 한도 / 연속 손실 한도: 초과 시 진입 불가
                daily_loss_pct = 0.0
                if daily_start_balance > 0:
                    daily_loss_pct = (daily_start_balance - balance) / daily_start_balance * 100
                daily_limit_hit = daily_loss_pct >= DAILY_LOSS_LIMIT_PCT
                consecutive_limit_hit = consecutive_loss_count >= CONSECUTIVE_LOSS_LIMIT

                if has_position and signal == "flat":
                    # 패턴 진입(패턴 익절/손절 대기 중)이면 추세 반전으로 청산하지 않음
                    if pattern_target > 0 and pattern_stop > 0:
                        pass  # 패턴 TP/SL만 적용
                    else:
                        # 포지션 종료
                        try:
                            positions = exchange.fetch_positions([SYMBOL])
                        except Exception as e:
                            log(f"포지션 조회 실패: {e}", "ERROR")
                            time.sleep(60)
                            continue
                        contracts = 0.0
                        side_to_close = "long" if is_long else "short"
                        for pos in positions:
                            if pos.get("symbol") == SYMBOL and pos.get("side") == side_to_close:
                                contracts = float(pos.get("contracts", 0) or 0)
                                break

                        if contracts > 0:
                            try:
                                if is_long:
                                    order = exchange.create_market_sell_order(
                                        SYMBOL, contracts, {"reduceOnly": True}
                                    )
                                else:
                                    order = exchange.create_market_buy_order(
                                        SYMBOL, contracts, {"reduceOnly": True}
                                    )
                                side_str = "LONG" if is_long else "SHORT"
                                regime_kr = REGIME_KR.get(entry_regime, entry_regime)
                                log(f"{side_str} 청산 | {regime_kr} | 진입={entry_price:.2f} 청산={price:.2f} 수익률={pnl_pct:+.2f}%")
                            except Exception as e:
                                log(f"청산 주문 실패: {e}", "ERROR")
                                continue  # 상태 유지, 다음 루프에서 재시도

                        # 손익 계산
                        try:
                            new_balance = get_balance_usdt(exchange)
                        except Exception as e:
                            log(f"청산 후 잔고 조회 실패: {e}", "ERROR")
                            new_balance = balance
                        pnl = new_balance - entry_balance
                        if pnl < 0:
                            consecutive_loss_count += 1
                        else:
                            consecutive_loss_count = 0
                        log_trade(
                            side="LONG" if is_long else "SHORT",
                            entry_price=entry_price,
                            exit_price=price,
                            pnl=pnl,
                            balance_after=new_balance,
                            meta={
                                "timeframe": TIMEFRAME,
                                "symbol": SYMBOL,
                                "regime": entry_regime,
                                "pnl_pct": round(pnl_pct, 2),
                                "consecutive_loss": consecutive_loss_count,
                            },
                        )
                        has_position = False
                        is_long = False
                        entry_price = 0.0
                        entry_regime = ""
                        box_high_entry = 0.0
                        box_low_entry = 0.0
                        highest_price = 0.0
                        lowest_price = float("inf")
                        trailing_stop_active = False
                        best_pnl_pct = 0.0
                        pattern_type = ""
                        pattern_target = 0.0
                        pattern_stop = 0.0
                        entry_balance = new_balance

                elif (not has_position) and (signal == "long" or signal == "short"):
                    if daily_limit_hit or consecutive_limit_hit:
                        if daily_limit_hit:
                            log(f"[진입생략] 일일손실한도 {daily_loss_pct:.1f}% | 잔고={balance:.2f}")
                        else:
                            log(f"[진입생략] 연속손실 {consecutive_loss_count}회 당일 중단 | 시그널={signal}")
                    else:
                        # 진입
                        order_usdt = balance * POSITION_SIZE_PERCENT
                        if order_usdt >= 5.0:
                            try:
                                ticker = exchange.fetch_ticker(SYMBOL)
                                mkt_price = float(ticker["last"])
                                amount = order_usdt / mkt_price
                                
                                # 차트 패턴 감지: 15분봉 72시간(288봉) (같은 방향 시그널일 때만)
                                pattern_info = None
                                if len(df) >= PATTERN_LOOKBACK * 3:
                                    df_tmp = df.copy()
                                    df_tmp["timestamp"] = pd.to_datetime(df_tmp["timestamp"])
                                    df_15m = df_tmp.set_index("timestamp").resample("15min").agg(
                                        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                                    ).dropna()
                                    if len(df_15m) >= PATTERN_LOOKBACK:
                                        highs = df_15m["high"].iloc[-PATTERN_LOOKBACK:].tolist()
                                        lows = df_15m["low"].iloc[-PATTERN_LOOKBACK:].tolist()
                                        closes = df_15m["close"].iloc[-PATTERN_LOOKBACK:].tolist()
                                        pattern_info = detect_chart_pattern(highs, lows, closes, mkt_price)
                                
                                if signal == "long":
                                    order = exchange.create_market_buy_order(SYMBOL, amount)
                                    pt, ptg, pst = ("", 0.0, 0.0) if not (pattern_info and pattern_info.side == "LONG") else (pattern_info.name, pattern_info.target_price, pattern_info.stop_price)
                                    regime_kr = REGIME_KR.get(regime, regime)
                                    if pt:
                                        entry_tp_sl = f" 패턴={pt} 목표가={ptg:.2f} 손절가={pst:.2f}"
                                    else:
                                        is_trend = regime == "neutral"
                                        tp_pct = TREND_PROFIT_TARGET if is_trend else SIDEWAYS_PROFIT_TARGET
                                        sl_pct = TREND_STOP_LOSS_PRICE if is_trend else SIDEWAYS_STOP_LOSS_PRICE
                                        pct_per_leverage = 1.0 / LEVERAGE
                                        tg = mkt_price * (1 + tp_pct / 100 * pct_per_leverage)
                                        st = mkt_price * (1 - sl_pct / 100 * pct_per_leverage)
                                        entry_tp_sl = f" 목표가={tg:.2f} 손절가={st:.2f}"
                                    log(f"LONG 진입 | {regime_kr} | 가격={mkt_price:.2f} 잔고={balance:.2f} 투입={order_usdt:.2f} USDT{entry_tp_sl}")
                                    has_position, is_long = True, True
                                    entry_price, highest_price = mkt_price, mkt_price
                                    pattern_type, pattern_target, pattern_stop = pt, ptg, pst
                                else:
                                    order = exchange.create_market_sell_order(SYMBOL, amount)
                                    pt, ptg, pst = ("", 0.0, 0.0) if not (pattern_info and pattern_info.side == "SHORT") else (pattern_info.name, pattern_info.target_price, pattern_info.stop_price)
                                    regime_kr = REGIME_KR.get(regime, regime)
                                    if pt:
                                        entry_tp_sl = f" 패턴={pt} 목표가={ptg:.2f} 손절가={pst:.2f}"
                                    else:
                                        is_trend = regime == "neutral"
                                        tp_pct = TREND_PROFIT_TARGET if is_trend else SIDEWAYS_PROFIT_TARGET
                                        sl_pct = TREND_STOP_LOSS_PRICE if is_trend else SIDEWAYS_STOP_LOSS_PRICE
                                        pct_per_leverage = 1.0 / LEVERAGE
                                        tg = mkt_price * (1 - tp_pct / 100 * pct_per_leverage)
                                        st = mkt_price * (1 + sl_pct / 100 * pct_per_leverage)
                                        entry_tp_sl = f" 목표가={tg:.2f} 손절가={st:.2f}"
                                    log(f"SHORT 진입 | {regime_kr} | 가격={mkt_price:.2f} 잔고={balance:.2f} 투입={order_usdt:.2f} USDT{entry_tp_sl}")
                                    has_position, is_long = True, False
                                    entry_price, lowest_price = mkt_price, mkt_price
                                    pattern_type, pattern_target, pattern_stop = pt, ptg, pst
                                
                                entry_regime = regime
                                entry_balance = balance
                                trailing_stop_active = False
                                best_pnl_pct = 0.0
                                # 횡보장 진입 시 박스권 저장 (15분봉 박스권)
                                box_high_entry = 0.0
                                box_low_entry = 0.0
                                if regime == "sideways" and use_15m and len(price_history_15m) >= REGIME_LOOKBACK_15M:
                                    box_high_entry = max(price_history_15m[-REGIME_LOOKBACK_15M:])
                                    box_low_entry = min(price_history_15m[-REGIME_LOOKBACK_15M:])
                            except Exception as e:
                                log(f"진입 주문 실패: {e}", "ERROR")

                # 5분 간격 상태 로그
                pos_status = "LONG" if has_position and is_long else ("SHORT" if has_position else "NONE")
                try:
                    bal = get_balance_usdt(exchange) if has_position else balance
                except Exception:
                    bal = balance
                unrealized = ""
                if has_position:
                    pnl_pct = (price - entry_price) / entry_price * LEVERAGE * 100 if is_long else (entry_price - price) / entry_price * LEVERAGE * 100
                    unrealized = f" 미실현={pnl_pct:+.2f}%"
                regime_str = f" | {REGIME_KR.get(entry_regime, entry_regime)}" if has_position and entry_regime else ""
                log(f"[5분] {pos_status}{regime_str} | 가격={price:.2f} RSI={rsi:.0f} 잔고={bal:.2f}{unrealized}")

                last_candle_time = latest_time

            time.sleep(300)  # 5분봉이므로 5분(300초) 대기

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
