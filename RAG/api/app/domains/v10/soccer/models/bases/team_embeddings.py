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

# Base 클래스 상속
Base = declarative_base()

# 팀 임베딩 모델 정의
class TeamEmbedding(Base):
    __tablename__ = "teams_embeddings"

    # 기본 키
    id = Column(BigInteger, primary_key=True, autoincrement=True, nullable=False)
    team_id = Column(BigInteger, ForeignKey("teams.id", ondelete="CASCADE"), nullable=False)

    # 임베딩 내용
    content = Column(Text, nullable=False)
    embedding = Column(VectorColumn(create_vector_type(768)))  # pgvector 기반 임베딩 컬럼
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    # 팀과의 관계 설정
    team = relationship("Team", back_populates="embeddings")

# 팀 임베딩 모델 등록
Base.metadata.create_all(bind=engine)
