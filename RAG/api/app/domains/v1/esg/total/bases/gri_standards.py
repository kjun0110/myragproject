"""GRI Standards SQLAlchemy 모델.

GRI(Global Reporting Initiative) 표준 분류 테이블을 위한 데이터베이스 모델입니다.
"""

from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.domains.v1.shared.bases.base import Base


class GRIStandard(Base):
    """GRI 표준 분류 테이블 모델.

    Attributes:
        id: 기본 키 (자동 증가)
        standard_code: GRI 표준 코드 (예: 'GRI 305')
        standard_name: 표준 이름 (예: 'Emissions')
        category: 카테고리 (예: 'Environmental', 'Social', 'Governance', 'Economic')
        published_year: 발행 연도
    """

    __tablename__ = "gri_standards"

    # SQLAlchemy 2.0+ 스타일 (권장)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    standard_code: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    standard_name: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=True)
    published_year: Mapped[int] = mapped_column(Integer, nullable=True, default=2021)

    def __repr__(self) -> str:
        """객체의 문자열 표현을 반환합니다."""
        return (
            f"<GRIStandard(id={self.id}, "
            f"standard_code='{self.standard_code}', "
            f"standard_name='{self.standard_name}', "
            f"category='{self.category}', "
            f"published_year={self.published_year})>"
        )

    def __str__(self) -> str:
        """사용자 친화적인 문자열 표현을 반환합니다."""
        return f"{self.standard_code}: {self.standard_name} ({self.category})"
