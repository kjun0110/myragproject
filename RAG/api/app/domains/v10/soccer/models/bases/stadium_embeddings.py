from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from pgvector import create_vector_type
from pgvector.sqlalchemy import Column as VectorColumn
from app.domains.v10.shared.models.bases.base import Base

# SQLAlchemy 엔진 생성
engine = create_engine('postgresql://username:password@localhost:5432/database_name')
Session = sessionmaker(bind=engine)
session = Session()

# pgvector 벡터 타입 생성
VectorType = create_vector_type(768)  # 임베딩 차원은 768로 고정

# StadiumEmbedding 모델 정의
class StadiumEmbedding(Base):
    __tablename__ = "stadiums_embeddings"

    id = Column(BigInteger, primary_key=True, autoincrement=True, nullable=False)
    stadium_id = Column(BigInteger, ForeignKey("stadiums.id", ondelete="CASCADE"), nullable=False)

    content = Column(Text, nullable=False)
    embedding = Column(VectorColumn(VectorType))
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    def __repr__(self):
        return f"<StadiumEmbedding(id={self.id}, stadium_id={self.stadium_id}, embedding={self.embedding})>"

# 모델 등록
Base.metadata.create_all(bind=engine)
