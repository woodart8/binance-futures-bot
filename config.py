"""
공통 설정 모듈.

실거래, 백테스트, 모의투자에서 모두 같은 파라미터를 사용하도록 한다.
"""

SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"  # 5분봉 기준

# 포지션/리스크 관리
LEVERAGE = 3  # 레버리지 3배
POSITION_SIZE_PERCENT = 0.333  # 재산의 1/3로만 매매
FEE_RATE = 0.0004  # 왕복 수수료 0.04% 가정
SLIPPAGE_PCT = 0.05  # 슬리피지 0.05% (실제 거래 비용 고려)

# 보조 지표 파라미터
RSI_PERIOD = 14  # 표준 RSI 기간
MA_SHORT_PERIOD = 7  # 단기 이동평균선 7
MA_LONG_PERIOD = 20  # 장기 이동평균선 20
MA_MID_PERIOD = 50  # 중기 이동평균선 50
MA_LONGEST_PERIOD = 100  # 장기 이동평균선 100

# 강세장/약세장 전략 (단타 방식, 5분봉 기준)
PARTIAL_PROFIT_RATIO = 0.75  # 부분 익절 기준 = 익절 × 75%
BULLISH_PROFIT_TARGET = 3.7  # 강세장 롱 익절 3.7% (수익률*레버리지 기준)
BULLISH_PROFIT_TARGET_PARTIAL = BULLISH_PROFIT_TARGET * PARTIAL_PROFIT_RATIO
BULLISH_PARTIAL_EXIT_PCT = 0.3  # 부분 익절 비율 30%
BULLISH_STOP_LOSS = 2.0  # 강세장 롱 손절 2% (수익률*레버리지 기준)
BULLISH_STOP_LOSS_PRICE = 2.0  # 강세장 롱 스탑로스 2% (단타)
BULLISH_EARLY_EXIT_RSI = 35.0  # RSI가 35 이하면 조기 청산 (하락 신호)
BULLISH_TRAILING_STOP_ACTIVATION = 2.0  # 트레일링 스톱 활성화 수익률 2%
BULLISH_TRAILING_STOP_PCT = 1.0  # 트레일링 스톱 1% (최고 수익 대비)

BEARISH_PROFIT_TARGET = 3.7  # 약세장 숏 익절 3.7% (수익률*레버리지 기준)
BEARISH_PROFIT_TARGET_PARTIAL = BEARISH_PROFIT_TARGET * PARTIAL_PROFIT_RATIO
BEARISH_PARTIAL_EXIT_PCT = 0.3  # 부분 익절 비율 30%
BEARISH_STOP_LOSS = 2.0  # 약세장 숏 손절 2% (수익률*레버리지 기준)
BEARISH_STOP_LOSS_PRICE = 2.0  # 약세장 숏 스탑로스 2% (단타)
BEARISH_EARLY_EXIT_RSI = 65.0  # RSI가 65 이상이면 조기 청산 (반등 신호)
BEARISH_TRAILING_STOP_ACTIVATION = 2.0  # 트레일링 스톱 활성화 수익률 2%
BEARISH_TRAILING_STOP_PCT = 1.0  # 트레일링 스톱 1% (최고 수익 대비)

# 횡보장 전략 (박스권 기반)
SIDEWAYS_BOX_PERIOD = 288  # 최근 1일 박스권 (288개 5분봉 = 24시간)
SIDEWAYS_STOP_LOSS = 2.0  # 횡보장 손절 2% (수익률*레버리지 기준)
SIDEWAYS_BOX_TOP_MARGIN = 0.016  # 상단 근처 판단 (1.6%)
SIDEWAYS_BOX_BOTTOM_MARGIN = 0.016  # 하단 근처 판단 (1.6%)

# 일일 손실 한도
DAILY_LOSS_LIMIT_PCT = 3.5  # 일일 손실이 3.5% 초과 시 당일 추가 진입 중단


# 모의/백테스트 초기 잔고
INITIAL_BALANCE = 1000.0

