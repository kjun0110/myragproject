"""
경기 일정(Schedule) 모델.

ERD 기반 SQLAlchemy 모델 정의.
"""

from sqlalchemy import Column, String, Integer, BigInteger, ForeignKey
from sqlalchemy.orm import relationship

from app.domains.v10.shared.models.bases.base import Base


class Schedule(Base):
    """
    경기 일정(Schedule) 모델 클래스.
    
    축구 경기 일정 정보, 경기장 정보, 팀 정보, 경기 결과를 저장하는 SQLAlchemy 모델입니다.
    
    Attributes:
        id: 경기 일정 고유 ID (BigInteger, Primary Key)
        stadium_id: 경기장 ID (Foreign Key -> stadium.id)
        hometeam_id: 홈팀 ID (Foreign Key -> team.id)
        awayteam_id: 원정팀 ID (Foreign Key -> team.id)
        stadium_code: 경기장 코드
        sche_date: 경기 일정 날짜 (YYYYMMDD 형식)
        gubun: 경기 구분 (Y = 진행됨, N = 예정)
        hometeam_code: 홈팀 코드
        awayteam_code: 원정팀 코드
        home_score: 홈팀 득점 (경기 전이면 null)
        away_score: 원정팀 득점 (경기 전이면 null)
    """


    __tablename__ = "schedules"

    # 기본 키
    id = Column(BigInteger, primary_key=True, nullable=False)  # 경기 일정 고유 ID (BigInt 타입)

    # 외래 키
    stadium_id = Column(BigInteger, ForeignKey("stadiums.id"), nullable=True)  # 경기장 ID (FK -> stadiums.id)
    hometeam_id = Column(BigInteger, ForeignKey("teams.id"), nullable=True)  # 홈팀 ID (FK -> teams.id)
    awayteam_id = Column(BigInteger, ForeignKey("teams.id"), nullable=True)  # 원정팀 ID (FK -> teams.id)

    # 경기장 정보
    stadium_code = Column(String, nullable=False)  # 경기장 코드 (예: C02, B04)

    # 경기 일정 정보
    sche_date = Column(String, nullable=False)  # 경기 일정 날짜 (YYYYMMDD 형식)
    gubun = Column(String, nullable=False)  # 경기 구분 (예: "Y" = 진행됨, "N" = 예정)

    # 팀 정보
    hometeam_code = Column(String, nullable=False)  # 홈팀 코드
    awayteam_code = Column(String, nullable=False)  # 원정팀 코드

    # 경기 결과
    home_score = Column(Integer, nullable=True)  # 홈팀 득점 (경기 전이면 null)
    away_score = Column(Integer, nullable=True)  # 원정팀 득점 (경기 전이면 null)
    
    # Relationships
    embeddings = relationship("ScheduleEmbedding", back_populates="schedule", cascade="all, delete-orphan")


