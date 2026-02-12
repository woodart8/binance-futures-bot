"""
공통 설정 모듈.

실거래, 백테스트, 모의투자에서 모두 같은 파라미터를 사용하도록 한다.
"""

SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"

# 포지션/리스크 관리
LEVERAGE = 5  # 선물 레버리지
POSITION_SIZE_PERCENT = 0.10  # 잔고의 10% 사용
FEE_RATE = 0.0004  # 왕복 수수료 0.04% 가정

# RSI_SWING 전략 파라미터
RSI_PERIOD = 14
RSI_ENTRY = 30  # RSI <= 30 진입
RSI_EXIT = 60  # RSI >= 60 청산

# 모의/백테스트 초기 잔고
INITIAL_BALANCE = 1000.0

