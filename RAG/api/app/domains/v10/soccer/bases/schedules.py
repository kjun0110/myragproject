"""
경기 일정(Schedule) 모델.

ERD 기반 SQLAlchemy 모델 정의.
"""

from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship

from app.domains.v10.shared.bases.base import Base


class Schedule(Base):
    """경기 일정 테이블 모델.

    Attributes:
        sche_date: 경기 일정 날짜 (Primary Key)
        stadium_id: 경기장 ID (Foreign Key -> Stadium)
        gubun: 구분
        hometeam_id: 홈팀 ID
        awayteam_id: 원정팀 ID
        home_score: 홈팀 점수
        away_score: 원정팀 점수
    """

    __tablename__ = "schedule"

    # Primary Key
    sche_date = Column(String(10), primary_key=True, comment="경기 일정 날짜")

    # Foreign Key
    stadium_id = Column(
        String(10),
        ForeignKey("stadium.stadium_id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
        comment="경기장 ID"
    )

    # Attributes
    gubun = Column(String(10), nullable=True, comment="구분")
    hometeam_id = Column(String(10), nullable=True, comment="홈팀 ID")
    awayteam_id = Column(String(10), nullable=True, comment="원정팀 ID")
    home_score = Column(Integer, nullable=True, comment="홈팀 점수")
    away_score = Column(Integer, nullable=True, comment="원정팀 점수")

    # Relationships
    stadium = relationship("Stadium", back_populates="schedules")
