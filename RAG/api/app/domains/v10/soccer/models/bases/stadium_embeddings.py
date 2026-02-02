"""StadiumEmbedding 모델 정의.

ERD 기반 SQLAlchemy 모델 정의.

주의:
- 이 파일은 **ORM 테이블 정의만** 포함합니다(엔진/세션/create_all 금지).
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Column, ForeignKey, Text, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from app.domains.v10.shared.models.bases.base import Base


class StadiumEmbedding(Base):
    """경기장 임베딩 모델."""

    __tablename__ = "stadiums_embeddings"

    id = Column(BigInteger, primary_key=True, autoincrement=True, nullable=False)
    stadium_id = Column(
        BigInteger, ForeignKey("stadiums.id", ondelete="CASCADE"), nullable=False
    )

    content = Column(Text, nullable=False)
    embedding = Column(Vector(768), nullable=False)
    created_at = Column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    stadium = relationship("Stadium", back_populates="embeddings")
