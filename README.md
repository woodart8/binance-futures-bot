# Binance Futures Bot

바이낸스 USDS-M 선물 5분봉 기반 **횡보장 박스 단타** + **추세장 추세추종 단타** 전략입니다.  
차트 패턴·하모닉 패턴 진입 시에는 패턴 정석 목표가/손절가로 청산합니다.

---

## 1. 전체 구조

- **장세 구분**: 15분봉 기준으로 시장을 **횡보장** / **추세장** 두 가지로 나눕니다.
- **횡보장**: 박스권 조건 만족 → 박스 하단 근처 롱, 상단 근처 숏.
- **추세장**: 박스 아님 → 상승추세 MA 풀백 롱, 하락추세 MA 풀백 숏.
- **차트 패턴**: 15분봉 72시간(288봉) 구간에서 패턴 감지 시, 전략 방향과 일치하면 **패턴 목표가/손절가**로만 청산(전략 청산 없음).
- **공통**: 5분봉이 닫힐 때마다 진입·청산 판단, 목표익절·손절·트레일링 스탑 등으로 청산.

---

## 2. 사용 데이터

| 용도           | 타임프레임 | 비고                    |
|----------------|------------|-------------------------|
| 장세 판별      | 15분봉     | 박스 여부, 추세 방향   |
| 추세 MA        | 15분봉     | MA7, MA20, MA50, MA100 |
| 진입/청산 체크 | 5분봉      | 종가, RSI(14)           |
| 차트 패턴      | 15분봉     | 72시간(288봉) 스윙     |

---

## 3. 횡보장 전략 (Sideways)

### 3.1 횡보장 판별 (15분봉)

- 최근 **REGIME_LOOKBACK_15M(96)** 개 15분봉으로 박스권 판별.
- **박스 하·상단**: 해당 구간 저가 최솟값 = 하단, 고가 최댓값 = 상단.
- **조건**: 박스 폭 1.5% 이상, 현재가가 박스 내, 상·하단 각 2회 이상 터치.

### 3.2 진입·청산

- **롱**: MA7 > MA20 이고 가격이 박스 **하단 근처**(4% 이내).
- **숏**: MA7 < MA20 이고 가격이 박스 **상단 근처**(4% 이내).
- 청산: 목표 익절(SIDEWAYS_PROFIT_TARGET), 손절(SIDEWAYS_STOP_LOSS), 박스 이탈, 전략 신호(추세 전환).

---

## 4. 추세장 전략 (Neutral)

- **롱**: 상승 추세 + MA7 풀백 + RSI ≤ 58.
- **숏**: 하락 추세 + MA7 풀백 + RSI ≥ 42.
- 청산: 목표 익절(TREND_PROFIT_TARGET), 손절(TREND_STOP_LOSS), 전략 신호(추세 반전).

---

## 5. 차트 패턴 (15분봉 72시간)

패턴 감지 시 **진입 방향과 일치**할 때만 패턴 TP/SL 적용. 진입 후에는 **패턴 익절/손절만** 사용하고 전략 청산은 하지 않습니다.

### 5.1 우선순위 (앞쪽이 우선)

1. **더블** — double_top, double_bottom  
2. **삼각형** — ascending_triangle, descending_triangle, symmetrical_triangle  
3. **쐐기** — falling_wedge, rising_wedge  
4. **헤드앤숄더** — head_and_shoulders_top, inverse_head_and_shoulders  
5. **하모닉** — harmonic_gartley, harmonic_crab, harmonic_bat, harmonic_butterfly  

### 5.2 패턴별 요약

| 패턴 | 방향 | 진입 조건 | 목표/손절 |
|------|------|-----------|-----------|
| double_top / double_bottom | SHORT / LONG | 목선 이탈·돌파 | 목선 ± 패턴 높이 |
| ascending / descending / symmetrical_triangle | LONG / SHORT / 양방향 | 저항·지지 돌파·이탈 | 돌파선 ± 높이 |
| falling_wedge / rising_wedge | LONG / SHORT | 상단 돌파 / 하단 이탈 | 쐐기 높이 반영 |
| head_and_shoulders_top / inverse H&S | SHORT / LONG | 목선 이탈·돌파 | 헤드 높이 반영 |
| harmonic_* (Gartley, Crab, Bat, Butterfly) | LONG/SHORT | XABCD 피보나치 비율, PRZ(D) 근처 | D ± 61.8% AD, 손절 X 밖 |

---

## 6. 설정 요약 (config.py)

- **SYMBOL**: BTC/USDT · **TIMEFRAME**: 5m · **LEVERAGE**: 6 · **POSITION_SIZE_PERCENT**: 0.25  
- **DAILY_LOSS_LIMIT_PCT**: 5.0 · **CONSECUTIVE_LOSS_LIMIT**: 4  
- 추세: TREND_PROFIT_TARGET 5%, TREND_STOP_LOSS 2.5%  
- 횡보: SIDEWAYS_PROFIT_TARGET 2.5%, SIDEWAYS_STOP_LOSS 2%

---

## 7. 실행

| 목적 | 명령 |
|------|------|
| **백테스트·분석** | `python analyze_backtest.py` (기본 365일) |
| **기간 지정** | `python analyze_backtest.py 90` (90일) |
| **페이퍼 트레이딩** | `python paper_trading.py` (퍼블릭 데이터만 사용 가능) |
| **실거래** | `python live_trader_agent.py` (.env에 API_KEY, SECRET_KEY 필요) |
| **일일 매매 리포트 이메일** | `python daily_report.py` (cron 등으로 매일 09:00 KST 권장) |

### 7.1 일일 리포트 이메일

- **설정**: `.env`에 `EMAIL_TO`, `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` 설정.
- **내용**: 전날(한국시간) `trades_log.csv` 기준 거래 건별 상세(시간, 방향, 진입가, 청산가, 수익률, 손익, 잔고, 장세, 청산사유) 및 일별 요약.

---

## 8. API·로그

- **타임아웃**: 60초. OHLCV 조회는 타임아웃/네트워크 오류 시 최대 3회 재시도(2초 간격).
- **실거래**: OHLCV가 30회 연속 실패 시 프로세스 종료(재실행 필요).
- **박스권 로그**: 횡보장 시 5분 로그·진입 로그에 박스 하단/상단 가격 표시. 진입안함 사유에도 박스 하단/상단 포함.

---

## 9. 주요 파일

- **strategy_core.py** — 장세 판별, 진입 신호, 진입/보유 사유 문자열  
- **exit_logic.py** — 목표익절·손절·스탑로스·박스 이탈 청산  
- **chart_patterns.py** — 차트 패턴·하모닉 패턴 감지 및 목표가/손절가  
- **config.py** — 상수 설정  
- **backtest.py** / **analyze_backtest.py** — 백테스트 및 장별·패턴별 분석  
- **daily_report.py** — 전날 매매 결과 HTML 리포트 및 이메일 전송  
