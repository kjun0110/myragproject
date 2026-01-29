"""PlayerEmbedding 모델 (자동 생성)

players 테이블을 기반으로 한 임베딩 테이블 모델입니다.
"""

from sqlalchemy import BigInteger, Column, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.types import TIMESTAMP

from pgvector.sqlalchemy import Vector

from app.domains.v10.shared.models.bases.base import Base


class PlayerEmbedding(Base):
    __tablename__ = "players_embeddings"

    id = Column(BigInteger, primary_key=True, autoincrement=True, nullable=False)
    player_id = Column(BigInteger, ForeignKey("players.id", ondelete="CASCADE"), nullable=False)

    content = Column(Text, nullable=False)
    embedding = Column(Vector(768), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    player = relationship("Player", back_populates="embeddings")
