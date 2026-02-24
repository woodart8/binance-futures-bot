"""실거래 전용 설정. 수치 변경 시 페이퍼/백테스트에는 영향 없음."""

from config_common import *  # noqa: F401, F403

# OHLCV 조회 주기(초). 1분봉 수집 후 새 1분봉이 뜰 때마다 진입·청산 판단.
LIVE_CHECK_INTERVAL = 10

# 필요 시 실거래 전용으로 override (예: 더 보수적)
# LEVERAGE = 3
# POSITION_SIZE_PERCENT = 0.1
# DAILY_LOSS_LIMIT_PCT = 3.0
# CONSECUTIVE_LOSS_LIMIT = 3
# 추세장 진입 (config_common과 동일 적용됨. 라이브만 다르게 쓰려면 아래 주석 해제)
# TREND_UPTREND_LONG_RSI_MAX = 40
# TREND_UPTREND_LONG_ENABLED = True
# TREND_UPTREND_LONG_REQUIRE_RSI_TURNUP = True   # 상승장 롱: RSI 상승 전환 시에만
# TREND_DOWNTREND_SHORT_RSI_MIN = 62
# TREND_DOWNTREND_SHORT_ENABLED = True
# TREND_DOWNTREND_SHORT_REQUIRE_RSI_TURNDOWN = True  # 하락장 숏: RSI 꺾임 시에만
