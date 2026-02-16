"""공통 전략/지표 설정. config_paper / config_live 에서 import 후 필요 시 override."""

# 거래
SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"

# 포지션 (paper/live에서 override 가능)
LEVERAGE = 6
POSITION_SIZE_PERCENT = 0.25
FEE_RATE = 0.001
SLIPPAGE_PCT = 0.05

# 지표
RSI_PERIOD = 14
MA_SHORT_PERIOD = 7
MA_LONG_PERIOD = 20
MA_MID_PERIOD = 50
MA_LONGEST_PERIOD = 100

# 장세 (15분봉 24h)
REGIME_LOOKBACK_15M = 96

# 추세장
TREND_SLOPE_BARS = 96  # 24시간 = 96봉 (15분봉 기준)
TREND_SLOPE_MIN_PCT = 2.5  # 24h 기울기 ±2.5% 초과 시 추세장
TREND_PROFIT_TARGET = 5.5
TREND_STOP_LOSS = 2.5
TREND_STOP_LOSS_PRICE = 2.5
# 상승장 진입 조건 (이상/이하 조건)
TREND_UPTREND_LONG_RSI_MAX = 48  # 상승장 롱: RSI ≤ 48
TREND_UPTREND_SHORT_RSI_MIN = 80  # 상승장 숏: RSI ≥ 80
# 하락장 진입 조건 (이상/이하 조건)
TREND_DOWNTREND_LONG_RSI_MAX = 20  # 하락장 롱: RSI ≤ 20
TREND_DOWNTREND_SHORT_RSI_MIN = 52  # 하락장 숏: RSI ≥ 52

# 횡보장 박스
SIDEWAYS_ENABLED = True
SIDEWAYS_BOX_PERIOD = 48
SIDEWAYS_BOX_TOP_MARGIN = 0.04
SIDEWAYS_BOX_BOTTOM_MARGIN = 0.04
SIDEWAYS_MIN_TOUCHES = 2
SIDEWAYS_BOX_TOUCH_GAP_MIN_HOURS = 2  # 박스 상단 두 봉·하단 두 봉 간 최소 간격(시간)
SIDEWAYS_BOX_SLOPE_DIFF_MAX = 0.005  # 상단 두 봉 기울기와 하단 두 봉 기울기 차이 한도(box_low 대비 봉당)
SIDEWAYS_BOX_SLOPE_MAX = 0.005  # 기울기 절대값 한도(box_low 대비 봉당): 0.5%까지 허용, 이보다 가파르면 패스
SIDEWAYS_BOX_RANGE_PCT_MIN = 1.5
SIDEWAYS_BOX_RANGE_MIN = 1  # 횡보장 상단-하단 봉간격 최소 (절대값)
SIDEWAYS_PROFIT_TARGET = 2.5
SIDEWAYS_STOP_LOSS = 2.0
SIDEWAYS_STOP_LOSS_PRICE = 2.0
SIDEWAYS_BOX_EXIT_MARGIN_PCT = 0.5  # 박스 상·하단 돌파(이탈) 기준: 가격이 상단/하단에서 0.5% 벗어나면 이탈

# 손실 한도 (paper/live에서 override 가능)
DAILY_LOSS_LIMIT_PCT = 5.0
CONSECUTIVE_LOSS_LIMIT = 4

# 백테스트/페이퍼 전용
INITIAL_BALANCE = 1000.0
