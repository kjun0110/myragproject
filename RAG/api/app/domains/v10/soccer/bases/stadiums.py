"""
경기장(Stadium) 모델.

ERD 기반 SQLAlchemy 모델 정의.
"""

from sqlalchemy import Column, Integer, String, BigInteger
from sqlalchemy.orm import relationship

from app.domains.v10.shared.bases.base import Base


class Stadium(Base):
    """
    경기장(Stadium) 모델 클래스.
    
    축구 경기장의 기본 정보, 홈팀 정보, 규모, 위치 및 연락처를 저장하는 SQLAlchemy 모델입니다.
    
    Attributes:
        id: 경기장 고유 ID (BigInteger, Primary Key)
        stadium_code: 경기장 코드
        statdium_name: 경기장 이름
        hometeam_code: 홈팀 코드 (해당 경기장을 홈으로 사용하는 팀)
        seat_count: 수용 인원 수
        address: 경기장 주소
        ddd: 지역번호
        tel: 전화번호
    """


    __tablename__ = "stadium"

    # 기본 키
    id = Column(BigInteger, primary_key=True, nullable=False)  # 경기장 고유 ID (BigInt 타입)

    # 경기장 기본 정보
    stadium_code = Column(String, nullable=False)  # 경기장 코드 (예: D03, B02, C06)
    statdium_name = Column(String, nullable=False)  # 경기장 이름 (예: 전주월드컵경기장, 성남종합운동장)

    # 홈팀 정보
    hometeam_code = Column(String, nullable=True)  # 홈팀 코드 (해당 경기장을 홈으로 사용하는 팀)

    # 경기장 규모
    seat_count = Column(Integer, nullable=True)  # 수용 인원 수

    # 경기장 위치 및 연락처
    address = Column(String, nullable=True)  # 경기장 주소
    ddd = Column(String, nullable=True)  # 지역번호 (예: 063, 031, 054)
    tel = Column(String, nullable=True)  # 전화번호


