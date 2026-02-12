## Binance Futures Bot (5m RSI Swing, Multi-Agent)

이 프로젝트는 바이낸스 USDS-M 선물에서 5분봉 RSI_SWING 전략으로
실거래/백테스트/모의투자를 수행하는 구조를 가진다.

### 주요 모듈

- `config.py`  
  공통 설정 (심볼, 타임프레임, 레버리지, RSI 진입/청산 값 등)

- `exchange_client.py`  
  - `get_private_exchange()` : .env 의 API_KEY, SECRET_KEY 를 사용한 실거래용 클라이언트  
  - `get_public_exchange()` : 퍼블릭 데이터(OHLCV/티커) 전용 클라이언트

- `indicators.py`  
  - `calculate_rsi()` : RSI 계산

- `strategy_core.py`  
  - `rsi_swing_signal()` : RSI_SWING 전략 시그널 생성

- `backtest_core.py`  
  - `run_long_only_backtest()` : 롱 전략 공통 백테스트 엔진

- `trade_logger.py`  
  - `log_trade()` : 실거래 결과를 `trades_log.csv` 에 기록

### 에이전트들

- `live_trader_agent.py`  
  - 5분마다 새로운 캔들을 확인하고, RSI_SWING 전략에 따라 **실제 선물 주문**을 전송  
  - 진입/청산 시 `trades_log.csv` 에 트레이드 기록 저장

- `strategy_research_agent.py`  
  - `trades_log.csv` 를 읽어 총 손익, 승률, MDD 등을 출력 (리포트용)

- `backtest_agent.py`  
  - 현재 설정과 여러 RSI_EXIT 후보 값에 대해 백테스트를 수행하고  
  - 수익률과 MDD를 고려한 점수로 추천 파라미터를 제안

- `paper_trading_5m.py`  
  - 실제 주문 없이, 퍼블릭 데이터와 가상 잔고로 5분봉 RSI_SWING 전략을 실시간 모의투자

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
Description=Binance Futures Live Trader (RSI_SWING)
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

