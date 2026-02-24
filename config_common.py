"""공통 전략/지표 설정. config_paper / config_live 에서 import 후 필요 시 override.

추세장: 상승장 롱 RSI≤40+옵션(RSI 상승 전환), 하락장 숏 RSI≥62+옵션(RSI 꺾임).
당일 진입 중단: 일일 손실 한도(%), 연속 손실 횟수. 당일 시작 잔고는 날짜 변경 후 첫 체크 시점 잔고.
"""

# 거래
SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"

# 포지션 (paper/live에서 override 가능). 1회 주문 = 마진(잔고×POSITION_SIZE_PERCENT)×LEVERAGE = 노션널, 수량(BTC)=노션널/가격
LEVERAGE = 6
POSITION_SIZE_PERCENT = 0.25
FEE_RATE = 0.001
SLIPPAGE_PCT = 0.05

# 지표
RSI_PERIOD = 12
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
TREND_UPTREND_LONG_RSI_MAX = 40  # 상승장 롱: RSI ≤ 40 (조정 초반 롱 방지, 48→40)
TREND_UPTREND_LONG_ENABLED = True  # False면 상승장에서 롱 진입 안 함
TREND_UPTREND_LONG_REQUIRE_RSI_TURNUP = True  # True면 RSI가 전봉 대비 상승할 때만 롱 (조정 exhaustion)
TREND_UPTREND_SHORT_RSI_MIN = 80  # 상승장 숏: RSI ≥ 80
# 하락장 진입 조건 (이상/이하 조건)
TREND_DOWNTREND_LONG_RSI_MAX = 20  # 하락장 롱: RSI ≤ 20
TREND_DOWNTREND_SHORT_RSI_MIN = 62  # 하락장 숏: RSI ≥ 62 (반등 초반 숏 방지, 52→62 상향)
TREND_DOWNTREND_SHORT_ENABLED = True  # False면 하락장에서 숏 진입 안 함
TREND_DOWNTREND_SHORT_REQUIRE_RSI_TURNDOWN = True  # True면 RSI가 전봉 대비 꺾일 때만 숏 (반등 exhaustion)

# 횡보장 박스 (진입/청산 시 박스는 15분봉 96개만 사용. SIDEWAYS_BOX_PERIOD는 detect_market_regime 등 내부용)
SIDEWAYS_ENABLED = True
SIDEWAYS_BOX_PERIOD = 48
SIDEWAYS_BOX_TOP_MARGIN = 0.03   # 박스 상단에서 3% 이내 진입(숏)
SIDEWAYS_BOX_BOTTOM_MARGIN = 0.03  # 박스 하단에서 3% 이내 진입(롱)
SIDEWAYS_MIN_TOUCHES = 2
SIDEWAYS_BOX_TOUCH_GAP_MIN_HOURS = 2  # 박스 상단 두 봉·하단 두 봉 간 최소 간격(시간)
SIDEWAYS_BOX_SLOPE_DIFF_MAX = 0.005  # 상단 두 봉 기울기와 하단 두 봉 기울기 차이 한도(box_low 대비 봉당)
SIDEWAYS_BOX_SLOPE_MAX = 0.005  # 기울기 절대값 한도(box_low 대비 봉당): 0.5%까지 허용, 이보다 가파르면 패스
SIDEWAYS_BOX_RANGE_PCT_MIN = 1.8  # 박스 폭 최소 1.8% (box_low 대비)
SIDEWAYS_BOX_RANGE_MIN = 1  # 횡보장 상단-하단 봉간격 최소 (절대값)
SIDEWAYS_PROFIT_TARGET = 3.5  # 횡보장 익절 3.5% (수수료 0.6% 포함 시 손익비 약 1.12)
SIDEWAYS_STOP_LOSS = 2.0
SIDEWAYS_STOP_LOSS_PRICE = 2.0
SIDEWAYS_BOX_EXIT_MARGIN_PCT = 0.5  # 박스 상·하단 돌파(이탈) 기준: 가격이 상단/하단에서 0.5% 벗어나면 이탈

# 손실 한도 (paper/live에서 override 가능)
DAILY_LOSS_LIMIT_PCT = 5.0
CONSECUTIVE_LOSS_LIMIT = 4

# 백테스트/페이퍼 전용
INITIAL_BALANCE = 1000.0
