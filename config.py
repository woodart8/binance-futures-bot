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

# 강세장/약세장 전략 (스윙 방식, 5분봉 기준)
BULLISH_PROFIT_TARGET = 12.0  # 강세장 롱 익절 12%
BULLISH_PROFIT_TARGET_PARTIAL = 8.0  # 강세장 롱 부분 익절 8% (30% 청산)
BULLISH_PARTIAL_EXIT_PCT = 0.3  # 부분 익절 비율 30% (50% → 30%)
BULLISH_STOP_LOSS = 4.0  # 강세장 롱 손절 4% (엄격화: 5% → 4%)
BULLISH_STOP_LOSS_PRICE = 5.0  # 강세장 롱 스탑로스 5% (엄격화: 7% → 5%)
BULLISH_EARLY_EXIT_RSI = 30.0  # RSI가 30 이하면 조기 청산 (하락 신호)
BULLISH_TRAILING_STOP_ACTIVATION = 8.0  # 트레일링 스톱 활성화 수익률 8% (6% → 8%)
BULLISH_TRAILING_STOP_PCT = 4.0  # 트레일링 스톱 4% (최고 수익 대비, 3% → 4%)

BEARISH_PROFIT_TARGET = 12.0  # 약세장 숏 익절 12%
BEARISH_PROFIT_TARGET_PARTIAL = 8.0  # 약세장 숏 부분 익절 8% (30% 청산)
BEARISH_PARTIAL_EXIT_PCT = 0.3  # 부분 익절 비율 30% (50% → 30%)
BEARISH_STOP_LOSS = 4.0  # 약세장 숏 손절 4% (엄격화: 5% → 4%)
BEARISH_STOP_LOSS_PRICE = 5.0  # 약세장 숏 스탑로스 5% (엄격화: 7% → 5%)
BEARISH_EARLY_EXIT_RSI = 70.0  # RSI가 70 이상이면 조기 청산 (반등 신호)
BEARISH_TRAILING_STOP_ACTIVATION = 8.0  # 트레일링 스톱 활성화 수익률 8% (6% → 8%)
BEARISH_TRAILING_STOP_PCT = 4.0  # 트레일링 스톱 4% (최고 수익 대비, 3% → 4%)

# 횡보장 전략 (박스권 기반)
SIDEWAYS_BOX_PERIOD = 288  # 최근 1일 박스권 (288개 5분봉 = 24시간)
SIDEWAYS_BOX_TOP_MARGIN = 0.02  # 상단 근처 판단 (2%)
SIDEWAYS_BOX_BOTTOM_MARGIN = 0.02  # 하단 근처 판단 (2%)


# 모의/백테스트 초기 잔고
INITIAL_BALANCE = 1000.0

