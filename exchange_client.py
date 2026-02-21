"""
바이낸스 USDT-M 선물(USDⓈ-M Futures)용 ccxt 클라이언트 생성 유틸리티.

실거래용(API 키 필요)과 퍼블릭 데이터 조회용(OHLCV/티커만) 두 가지를 제공한다.
"""

import os
import sys
from typing import Optional

import ccxt
from dotenv import load_dotenv


load_dotenv()


def get_private_exchange() -> ccxt.binanceusdm:
    """
    실거래용 바이낸스 USDT-M 선물 인스턴스 생성.

    API_KEY, SECRET_KEY 는 .env 에서 읽는다.
    """
    api_key: Optional[str] = os.getenv("API_KEY")
    secret_key: Optional[str] = os.getenv("SECRET_KEY")

    if not api_key or not secret_key:
        print("[ERROR] API_KEY와 SECRET_KEY를 .env 파일에 설정해주세요.", file=sys.stderr)
        sys.exit(1)

    exchange = ccxt.binanceusdm(
        {
            "apiKey": api_key,
            "secret": secret_key,
            "enableRateLimit": True,
            "timeout": 60000,  # 60초 타임아웃 (네트워크 지연 대비)
            "options": {"defaultType": "future"},
        }
    )
    return exchange


def get_public_exchange() -> ccxt.binanceusdm:
    """
    퍼블릭 데이터 전용 바이낸스 USDT-M 선물 인스턴스.

    API 키 없이도 시세/캔들 조회만 사용할 때 쓴다.
    """
    return ccxt.binanceusdm(
        {
            "enableRateLimit": True,
            "timeout": 60000,  # 60초 타임아웃 (네트워크 지연 대비)
            "options": {"defaultType": "future"},
        }
    )

