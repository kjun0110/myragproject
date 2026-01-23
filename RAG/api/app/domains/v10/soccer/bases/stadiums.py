"""
경기장(Stadium) 모델.

ERD 기반 SQLAlchemy 모델 정의.
"""

from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from app.domains.v10.shared.bases.base import Base


class Stadium(Base):
    """경기장 테이블 모델.
    
    Attributes:
        stadium_id: 경기장 ID (Primary Key)
        statdium_name: 경기장 이름
        hometeam_id: 홈팀 ID
        seat_count: 좌석 수
        address: 주소
        ddd: 지역번호
        tel: 전화번호
    """
    
    __tablename__ = "stadium"
    
    # Primary Key
    stadium_id = Column(String(10), primary_key=True, comment="경기장 ID")
    
    # Attributes
    statdium_name = Column(String(40), nullable=False, comment="경기장 이름")
    hometeam_id = Column(String(10), nullable=True, comment="홈팀 ID")
    seat_count = Column(Integer, nullable=True, comment="좌석 수")
    address = Column(String(60), nullable=True, comment="주소")
    ddd = Column(String(10), nullable=True, comment="지역번호")
    tel = Column(String(10), nullable=True, comment="전화번호")
    
    # Relationships
    teams = relationship("Team", back_populates="stadium", cascade="all, delete-orphan")
    schedules = relationship("Schedule", back_populates="stadium", cascade="all, delete-orphan")
