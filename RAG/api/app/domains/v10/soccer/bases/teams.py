"""
팀(Team) 모델.

ERD 기반 SQLAlchemy 모델 정의.
"""

from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.orm import relationship

from app.domains.v10.shared.bases.base import Base


class Team(Base):
    """팀 테이블 모델.

    Attributes:
        team_id: 팀 ID (Primary Key)
        stadium_id: 경기장 ID (Foreign Key -> Stadium)
        region_name: 지역명
        team_name: 팀 이름
        e_team_name: 영문 팀 이름
        orig_yyyy: 창단년도
        zip_code1: 우편번호1
        zip_code2: 우편번호2
        address: 주소
        ddd: 지역번호
        tel: 전화번호
        fax: 팩스번호
        homepage: 홈페이지
        owner: 구단주
    """

    __tablename__ = "team"

    # Primary Key
    team_id = Column(String(10), primary_key=True, comment="팀 ID")

    # Foreign Key
    stadium_id = Column(
        String(10),
        ForeignKey("stadium.stadium_id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
        comment="경기장 ID"
    )

    # Attributes
    region_name = Column(String(10), nullable=True, comment="지역명")
    team_name = Column(String(40), nullable=False, comment="팀 이름")
    e_team_name = Column(String(50), nullable=True, comment="영문 팀 이름")
    orig_yyyy = Column(String(10), nullable=True, comment="창단년도")
    zip_code1 = Column(String(10), nullable=True, comment="우편번호1")
    zip_code2 = Column(String(10), nullable=True, comment="우편번호2")
    address = Column(String(80), nullable=True, comment="주소")
    ddd = Column(String(10), nullable=True, comment="지역번호")
    tel = Column(String(10), nullable=True, comment="전화번호")
    fax = Column(String(10), nullable=True, comment="팩스번호")
    homepage = Column(String(50), nullable=True, comment="홈페이지")
    owner = Column(String(10), nullable=True, comment="구단주")

    # Relationships
    stadium = relationship("Stadium", back_populates="teams")
    players = relationship("Player", back_populates="team", cascade="all, delete-orphan")
