"""TeamEmbedding 모델 정의.

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


class TeamEmbedding(Base):
    """팀 임베딩 모델."""

    __tablename__ = "teams_embeddings"

    id = Column(BigInteger, primary_key=True, autoincrement=True, nullable=False)
    team_id = Column(BigInteger, ForeignKey("teams.id", ondelete="CASCADE"), nullable=False)

    content = Column(Text, nullable=False)
    embedding = Column(Vector(768), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    team = relationship("Team", back_populates="embeddings")
