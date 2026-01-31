"""
PlayerEmbedding 모델 정의

ERD 기반 SQLAlchemy 모델 정의.
"""

from sqlalchemy import BigInteger, Column, ForeignKey, Text, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pgvector import create_vector_type
from pgvector.sqlalchemy import Vector

# SQLAlchemy 엔진 생성
engine = create_engine('postgresql://username:password@localhost:5432/database_name')
Session = sessionmaker(bind=engine)
session = Session()

# Vector 타입 생성
VectorType = create_vector_type(engine, Vector(768))

# Base 클래스 정의
Base = declarative_base()


class PlayerEmbedding(Base):
    """
    선수 임베딩 모델 클래스.

    ERD 기반 SQLAlchemy 모델 정의.
    """

    __tablename__ = "players_embeddings"

    # 기본 키
    id = Column(BigInteger, primary_key=True, autoincrement=True, nullable=False)

    # 외래 키
    player_id = Column(BigInteger, ForeignKey("players.id", ondelete="CASCADE"), nullable=False)

    # 임베딩 내용
    content = Column(Text, nullable=False)

    # 벡터 임베딩
    embedding = Column(VectorType, nullable=False)

    # 생성 시간
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    # 관계 설정
    player = relationship("Player", back_populates="embeddings")


# 엔진 종료
session.close()
