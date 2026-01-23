"""GRI Environmental Contents SQLAlchemy 모델.

GRI 환경(Environmental) 도메인 세트 테이블을 위한 데이터베이스 모델입니다.
"""

from typing import Optional

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.domains.v1.shared.bases.base import Base


class GRIEnvContent(Base):
    """GRI 환경(Environmental) 도메인 세트 테이블 모델.

    Attributes:
        id: 기본 키 (자동 증가)
        standard_id: GRI 표준 ID (외래키, gri_standards 테이블 참조)
        disclosure_num: 공개 번호 (예: '305-1')
        content: 지침 전문
        metadata: 측정 단위 등 추가 정보 (JSONB)
    """

    __tablename__ = "gri_env_contents"

    # SQLAlchemy 2.0+ 스타일 (권장)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    standard_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("gri_standards.id", ondelete="CASCADE"),
        nullable=False,
    )
    disclosure_num: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # 관계 설정 (선택사항)
    # standard: Mapped["GRIStandard"] = relationship("GRIStandard", back_populates="env_contents")

    def __repr__(self) -> str:
        """객체의 문자열 표현을 반환합니다."""
        content_preview = (
            f"{self.content[:50]}..." if len(self.content) > 50 else self.content
        )
        return (
            f"<GRIEnvContent(id={self.id}, "
            f"standard_id={self.standard_id}, "
            f"disclosure_num='{self.disclosure_num}', "
            f"content='{content_preview}')>"
        )

    def __str__(self) -> str:
        """사용자 친화적인 문자열 표현을 반환합니다."""
        return f"GRI Env Content {self.disclosure_num or 'N/A'}: {self.content[:100]}..."
