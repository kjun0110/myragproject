"""Team Transfer(Pydantic) 모델.

`app.domains.v10.soccer.models.bases.teams.Team`(SQLAlchemy ORM)와 동일한 필드 구성을
API 요청/응답 스키마(BaseModel)로 제공합니다.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TeamBase(BaseModel):
    """팀(Team) 기본 스키마."""

    stadium_id: Optional[int] = Field(None, description="경기장 ID (FK -> stadiums.id)")

    team_code: str = Field(..., description="팀 코드 (예: K05, K08, K03)")
    region_name: str = Field(..., description="지역명 (예: 전북, 성남, 포항)")
    team_name: str = Field(..., description="팀 이름(한글)")
    e_team_name: Optional[str] = Field(None, description="팀 영문 이름")

    orig_yyyy: Optional[str] = Field(None, description="창단 연도 (YYYY)")
    stadium_code: Optional[str] = Field(None, description="홈 경기장 코드")

    zip_code1: Optional[str] = Field(None, description="우편번호 앞자리")
    zip_code2: Optional[str] = Field(None, description="우편번호 뒷자리")
    address: Optional[str] = Field(None, description="팀 주소")

    ddd: Optional[str] = Field(None, description="지역번호 (예: 063, 031, 054)")
    tel: Optional[str] = Field(None, description="전화번호")
    fax: Optional[str] = Field(None, description="팩스번호")
    homepage: Optional[str] = Field(None, description="홈페이지 URL")

    owner: Optional[str] = Field(None, description="구단주")


class TeamCreate(TeamBase):
    """팀 생성 요청 스키마."""

    pass


class TeamUpdate(BaseModel):
    """팀 수정 요청 스키마(부분 업데이트)."""

    stadium_id: Optional[int] = Field(None, description="경기장 ID (FK -> stadiums.id)")

    team_code: Optional[str] = Field(None, description="팀 코드")
    region_name: Optional[str] = Field(None, description="지역명")
    team_name: Optional[str] = Field(None, description="팀 이름(한글)")
    e_team_name: Optional[str] = Field(None, description="팀 영문 이름")

    orig_yyyy: Optional[str] = Field(None, description="창단 연도 (YYYY)")
    stadium_code: Optional[str] = Field(None, description="홈 경기장 코드")

    zip_code1: Optional[str] = Field(None, description="우편번호 앞자리")
    zip_code2: Optional[str] = Field(None, description="우편번호 뒷자리")
    address: Optional[str] = Field(None, description="팀 주소")

    ddd: Optional[str] = Field(None, description="지역번호")
    tel: Optional[str] = Field(None, description="전화번호")
    fax: Optional[str] = Field(None, description="팩스번호")
    homepage: Optional[str] = Field(None, description="홈페이지 URL")

    owner: Optional[str] = Field(None, description="구단주")


class TeamResponse(TeamBase):
    """팀 응답 스키마."""

    id: int = Field(..., description="팀 고유 ID (PK)")

    class Config:
        from_attributes = True


class TeamListResponse(BaseModel):
    """팀 목록 응답 스키마."""

    items: list[TeamResponse] = Field(..., description="팀 목록")
    total: int = Field(..., description="전체 개수")

