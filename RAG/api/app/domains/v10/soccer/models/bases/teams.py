"""
팀(Team) 모델.

ERD 기반 SQLAlchemy 모델 정의.
"""

from sqlalchemy import Column, String, BigInteger, ForeignKey
from sqlalchemy.orm import relationship

from app.domains.v10.shared.models.bases.base import Base


class Team(Base):
    """
    팀(Team) 모델 클래스.
    
    축구 팀의 기본 정보, 역사, 홈 경기장, 주소, 연락처, 소유자 정보를 저장하는 SQLAlchemy 모델입니다.
    
    Attributes:
        id: 팀 고유 ID (BigInteger, Primary Key)
        stadium_id: 경기장 ID (Foreign Key -> stadium.id)
        team_code: 팀 코드
        region_name: 지역명
        team_name: 팀 이름 (한글)
        e_team_name: 팀 영문 이름
        orig_yyyy: 창단 연도 (YYYY 형식)
        stadium_code: 홈 경기장 코드
        zip_code1: 우편번호 앞자리
        zip_code2: 우편번호 뒷자리
        address: 팀 주소
        ddd: 지역번호
        tel: 전화번호
        fax: 팩스번호
        homepage: 홈페이지 URL
        owner: 구단주
    """


    __tablename__ = "teams"

    # 기본 키
    id = Column(BigInteger, primary_key=True, nullable=False)  # 팀 고유 ID (BigInt 타입)

    # 외래 키
    stadium_id = Column(BigInteger, ForeignKey("stadiums.id"), nullable=True)  # 경기장 ID (FK -> stadiums.id)

    # 팀 기본 정보
    team_code = Column(String, nullable=False)  # 팀 코드 (예: K05, K08, K03)
    region_name = Column(String, nullable=False)  # 지역명 (예: 전북, 성남, 포항)
    team_name = Column(String, nullable=False)  # 팀 이름 (한글, 예: 현대모터스, 일화천마)
    e_team_name = Column(String, nullable=True)  # 팀 영문 이름

    # 팀 역사
    orig_yyyy = Column(String, nullable=True)  # 창단 연도 (YYYY 형식)

    # 홈 경기장 정보
    stadium_code = Column(String, nullable=True)  # 홈 경기장 코드

    # 주소 정보
    zip_code1 = Column(String, nullable=True)  # 우편번호 앞자리
    zip_code2 = Column(String, nullable=True)  # 우편번호 뒷자리
    address = Column(String, nullable=True)  # 팀 주소

    # 연락처 정보
    ddd = Column(String, nullable=True)  # 지역번호 (예: 063, 031, 054)
    tel = Column(String, nullable=True)  # 전화번호
    fax = Column(String, nullable=True)  # 팩스번호
    homepage = Column(String, nullable=True)  # 홈페이지 URL

    # 소유자 정보
    owner = Column(String, nullable=True)  # 구단주
    
    # Relationships
    embeddings = relationship("TeamEmbedding", back_populates="team", cascade="all, delete-orphan")

