"""Stadium Transfer(Pydantic) 모델.

`app.domains.v10.soccer.models.bases.stadiums.Stadium`(SQLAlchemy ORM)와 동일한 필드 구성을
API 요청/응답 스키마(BaseModel)로 제공합니다.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class StadiumBase(BaseModel):
    """경기장(Stadium) 기본 스키마."""

    stadium_code: str = Field(..., description="경기장 코드 (예: D03, B02, C06)")
    stadium_name: str = Field(..., description="경기장 이름")

    hometeam_code: Optional[str] = Field(None, description="홈팀 코드")
    seat_count: Optional[int] = Field(None, description="수용 인원 수")

    address: Optional[str] = Field(None, description="경기장 주소")
    ddd: Optional[str] = Field(None, description="지역번호 (예: 063, 031, 054)")
    tel: Optional[str] = Field(None, description="전화번호")


class StadiumCreate(StadiumBase):
    """경기장 생성 요청 스키마."""

    pass


class StadiumUpdate(BaseModel):
    """경기장 수정 요청 스키마(부분 업데이트)."""

    stadium_code: Optional[str] = Field(None, description="경기장 코드 (예: D03, B02, C06)")
    stadium_name: Optional[str] = Field(None, description="경기장 이름")

    hometeam_code: Optional[str] = Field(None, description="홈팀 코드")
    seat_count: Optional[int] = Field(None, description="수용 인원 수")

    address: Optional[str] = Field(None, description="경기장 주소")
    ddd: Optional[str] = Field(None, description="지역번호 (예: 063, 031, 054)")
    tel: Optional[str] = Field(None, description="전화번호")


class StadiumResponse(StadiumBase):
    """경기장 응답 스키마."""

    id: int = Field(..., description="경기장 고유 ID (PK)")

    class Config:
        from_attributes = True


class StadiumListResponse(BaseModel):
    """경기장 목록 응답 스키마."""

    items: list[StadiumResponse] = Field(..., description="경기장 목록")
    total: int = Field(..., description="전체 개수")

