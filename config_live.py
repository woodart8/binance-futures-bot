"""실거래 전용 설정. 수치 변경 시 페이퍼/백테스트에는 영향 없음."""

from config_common import *  # noqa: F401, F403

# OHLCV 조회 주기(초). 진입·청산 판단은 새 5분봉이 데이터에 뜰 때만 수행.
LIVE_CHECK_INTERVAL = 10

# 필요 시 실거래 전용으로 override (예: 더 보수적)
# LEVERAGE = 3
# POSITION_SIZE_PERCENT = 0.1
# DAILY_LOSS_LIMIT_PCT = 3.0
# CONSECUTIVE_LOSS_LIMIT = 3
