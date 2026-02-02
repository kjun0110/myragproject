"""
ScheduleEmbedding 모델 정의

ERD 기반 SQLAlchemy 모델 정의.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Column, ForeignKey, Text, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from app.domains.v10.shared.models.bases.base import Base

class ScheduleEmbedding(Base):
    """
    경기 일정 임베딩 모델 클래스.
    
    ERD 기반 SQLAlchemy 모델 정의.
    """

    __tablename__ = "schedules_embeddings"

    # 기본 키
    id = Column(BigInteger, primary_key=True, autoincrement=True, nullable=False)

    # 외래 키
    schedule_id = Column(BigInteger, ForeignKey("schedules.id", ondelete="CASCADE"), nullable=False)

    # 임베딩 내용
    content = Column(Text, nullable=False)

    # 벡터 임베딩
    embedding = Column(Vector(768), nullable=False)

    # 생성 시간
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    # 관계 설정
    schedule = relationship("Schedule", back_populates="embeddings")
