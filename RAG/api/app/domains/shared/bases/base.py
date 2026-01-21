"""SQLAlchemy Base 모델.

모든 데이터베이스 모델의 기본 클래스를 제공합니다.
"""

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import DeclarativeBase

# SQLAlchemy 2.0+ 스타일 (권장)
try:
    class Base(DeclarativeBase):
        """SQLAlchemy Base 클래스."""

        pass

except ImportError:
    # SQLAlchemy 1.x 스타일 (하위 호환)
    Base = declarative_base()
