"""
선수(Player) 모델.

ERD 기반 SQLAlchemy 모델 정의.
"""

from sqlalchemy import Column, String, Integer, Date, ForeignKey
from sqlalchemy.orm import relationship

from app.domains.v10.shared.bases.base import Base


class Player(Base):
    """선수 테이블 모델.

    Attributes:
        player_id: 선수 ID (Primary Key)
        team_id: 팀 ID (Foreign Key -> Team)
        player_name: 선수 이름
        e_player_name: 영문 선수 이름
        nickname: 별명
        join_yyyy: 입단년도
        position: 포지션
        back_no: 등번호
        nation: 국적
        birth_date: 생년월일
        solar: 양력/음력 구분
        height: 키 (cm)
        weight: 몸무게 (kg)
    """

    __tablename__ = "player"

    # Primary Key
    player_id = Column(String(10), primary_key=True, comment="선수 ID")

    # Foreign Key
    team_id = Column(
        String(10),
        ForeignKey("team.team_id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
        comment="팀 ID"
    )

    # Attributes
    player_name = Column(String(20), nullable=False, comment="선수 이름")
    e_player_name = Column(String(40), nullable=True, comment="영문 선수 이름")
    nickname = Column(String(30), nullable=True, comment="별명")
    join_yyyy = Column(String(10), nullable=True, comment="입단년도")
    position = Column(String(10), nullable=True, comment="포지션")
    back_no = Column(Integer, nullable=True, comment="등번호")
    nation = Column(String(20), nullable=True, comment="국적")
    birth_date = Column(Date, nullable=True, comment="생년월일")
    solar = Column(String(10), nullable=True, comment="양력/음력 구분")
    height = Column(Integer, nullable=True, comment="키 (cm)")
    weight = Column(Integer, nullable=True, comment="몸무게 (kg)")

    # Relationships
    team = relationship("Team", back_populates="players")
