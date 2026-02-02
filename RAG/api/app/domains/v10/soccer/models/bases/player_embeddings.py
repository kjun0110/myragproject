"""PlayerEmbedding 모델 정의.

ERD 기반 SQLAlchemy 모델 정의.

주의:
- 이 파일은 **ORM 테이블 정의만** 포함합니다(엔진/세션/create_all 금지).
- Alembic autogenerate 대상이 되려면 공용 Base(`v10/shared/models/bases/base.py`)를 사용해야 합니다.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Column, ForeignKey, Text, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from app.domains.v10.shared.models.bases.base import Base


class PlayerEmbedding(Base):
    """선수 임베딩 모델 클래스."""

    __tablename__ = "players_embeddings"

    # 기본 키
    id = Column(BigInteger, primary_key=True, autoincrement=True, nullable=False)

    # 외래 키
    player_id = Column(
        BigInteger, ForeignKey("players.id", ondelete="CASCADE"), nullable=False
    )

    # 임베딩 내용
    content = Column(Text, nullable=False)

    # 벡터 임베딩 (pgvector)
    embedding = Column(Vector(768), nullable=False)

    # 생성 시간
    created_at = Column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    # 관계 설정
    player = relationship("Player", back_populates="embeddings")
