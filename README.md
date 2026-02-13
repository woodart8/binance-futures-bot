## Binance Futures Bot (Scalping Strategy)

이 프로젝트는 바이낸스 USDS-M 선물에서 1분봉 스캘핑 전략으로
실거래/모의투자를 수행하는 구조를 가진다.

### 주요 모듈

- `config.py`  
  공통 설정 (심볼, 타임프레임, 레버리지, RSI 진입/청산 값 등)

- `exchange_client.py`  
  - `get_private_exchange()` : .env 의 API_KEY, SECRET_KEY 를 사용한 실거래용 클라이언트  
  - `get_public_exchange()` : 퍼블릭 데이터(OHLCV/티커) 전용 클라이언트

- `indicators.py`  
  - `calculate_rsi()` : RSI 계산

- `strategy_core.py`  
  - `rsi_swing_signal()` : RSI 전략 시그널 생성
  - `rsi_macd_signal()` : RSI+MACD 결합 전략 시그널 생성
  - `ma_trend_signal()` : 이동평균선 기반 추세 추종 전략 시그널 생성

- `trade_logger.py`  
  - `log_trade()` : 실거래 결과를 `trades_log.csv` 에 기록

### 에이전트들

- `live_trader_agent.py`  
  - 1분마다 새로운 캔들을 확인하고, RSI 전략에 따라 **실제 선물 주문**을 전송  
  - 진입/청산 시 `trades_log.csv` 에 트레이드 기록 저장

- `strategy_research_agent.py`  
  - `trades_log.csv` 를 읽어 총 손익, 승률 등을 출력 (리포트용)

- `paper_trading.py`  
  - 실제 주문 없이, 퍼블릭 데이터와 가상 잔고로 1분봉 스캘핑 전략을 실시간 모의투자
  - 시장 상태에 따라 이동평균선 전략 또는 RSI+MACD 전략 자동 선택

### 설치

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`.env` 파일 예시:

```text
API_KEY=YOUR_BINANCE_FUTURES_API_KEY
SECRET_KEY=YOUR_BINANCE_FUTURES_SECRET_KEY
```

### 우분투에서 24시간 돌리기 (예시)

`live_trader_agent.py` 를 systemd 서비스로 등록:

```ini
[Unit]
Description=Binance Futures Live Trader
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/binance-futures-bot
ExecStart=/home/ubuntu/binance-futures-bot/.venv/bin/python live_trader_agent.py
Restart=always
RestartSec=10
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
```

이후:

```bash
sudo systemctl daemon-reload
sudo systemctl enable binance_live_trader
sudo systemctl start binance_live_trader
```

