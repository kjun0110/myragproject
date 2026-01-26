"""
선수(Player) 모델.

ERD 기반 SQLAlchemy 모델 정의.
"""

from sqlalchemy import Column, String, Integer, Date, BigInteger, ForeignKey
from sqlalchemy.orm import relationship

from app.domains.v10.shared.bases.base import Base


class Player(Base):
    """
    선수(Player) 모델 클래스.
    
    축구 선수의 기본 정보, 팀 정보, 개인 정보, 신체 정보를 저장하는 SQLAlchemy 모델입니다.
    
    Attributes:
        id: 선수 고유 ID (BigInteger, Primary Key)
        team_id: 팀 ID (Foreign Key -> team.id)
        team_code: 소속 팀 코드
        player_name: 선수 이름 (한글)
        e_player_name: 선수 영문 이름
        nickname: 선수 별명
        join_yyyy: 입단 연도
        position: 포지션 (DF, MF, FW, GK)
        back_no: 등번호
        nation: 국적
        birth_date: 생년월일
        solar: 양력/음력 구분
        height: 키 (cm)
        weight: 몸무게 (kg)
    """


    __tablename__ = "player"

    # 기본 키
    id = Column(BigInteger, primary_key=True, nullable=False)  # 선수 고유 ID (BigInt 타입)

    # 외래 키
    team_id = Column(BigInteger, nullable=True)  # 팀 ID (FK -> team.id)

    # 팀 정보
    team_code = Column(String, nullable=False)  # 소속 팀 코드 (예: K06, K01)

    # 선수 기본 정보
    player_name = Column(String, nullable=False)  # 선수 이름 (한글)
    e_player_name = Column(String, nullable=True)  # 선수 영문 이름
    nickname = Column(String, nullable=True)  # 선수 별명

    # 입단 정보
    join_yyyy = Column(String, nullable=False)  # 입단 연도 (YYYY 형식)

    # 포지션 및 등번호
    position = Column(String, nullable=False)  # 포지션 (예: DF, MF, FW, GK)
    back_no = Column(Integer, nullable=True)  # 등번호

    # 개인 정보
    nation = Column(String, nullable=True)  # 국적
    birth_date = Column(Date, nullable=True)  # 생년월일
    solar = Column(String, nullable=True)  # 양력/음력 구분 (예: "1" = 양력)

    # 신체 정보
    height = Column(Integer, nullable=True)  # 키 (cm)
    weight = Column(Integer, nullable=True)  # 몸무게 (kg)

