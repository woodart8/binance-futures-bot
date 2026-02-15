"""페이퍼/백테스트 전용 설정. 수치 변경 시 라이브에는 영향 없음."""

from config_common import *  # noqa: F401, F403

# 모의투자 잔고 저장 파일 (수익/손실 시 자동 갱신)
PAPER_BALANCE_FILE = "paper_balance.json"

# 필요 시 페이퍼 전용으로 override
# LEVERAGE = 6
# POSITION_SIZE_PERCENT = 0.25
# DAILY_LOSS_LIMIT_PCT = 5.0
# CONSECUTIVE_LOSS_LIMIT = 4
# INITIAL_BALANCE = 1000.0
