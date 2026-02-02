"""Schedule Transfer(Pydantic) 모델.

`app.domains.v10.soccer.models.bases.schedules.Schedule`(SQLAlchemy ORM)와 동일한 필드 구성을
API 요청/응답 스키마(BaseModel)로 제공합니다.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ScheduleBase(BaseModel):
    """경기 일정(Schedule) 기본 스키마."""

    stadium_id: Optional[int] = Field(None, description="경기장 ID (FK -> stadiums.id)")
    hometeam_id: Optional[int] = Field(None, description="홈팀 ID (FK -> teams.id)")
    awayteam_id: Optional[int] = Field(None, description="원정팀 ID (FK -> teams.id)")

    stadium_code: str = Field(..., description="경기장 코드 (예: C02, B04)")
    sche_date: str = Field(..., description="경기 일정 날짜 (YYYYMMDD)")
    gubun: str = Field(..., description="경기 구분 (예: 'Y'=진행됨, 'N'=예정)")

    hometeam_code: str = Field(..., description="홈팀 코드")
    awayteam_code: str = Field(..., description="원정팀 코드")

    home_score: Optional[int] = Field(None, description="홈팀 득점 (경기 전이면 null)")
    away_score: Optional[int] = Field(None, description="원정팀 득점 (경기 전이면 null)")


class ScheduleCreate(ScheduleBase):
    """경기 일정 생성 요청 스키마."""

    pass


class ScheduleUpdate(BaseModel):
    """경기 일정 수정 요청 스키마(부분 업데이트)."""

    stadium_id: Optional[int] = Field(None, description="경기장 ID (FK -> stadiums.id)")
    hometeam_id: Optional[int] = Field(None, description="홈팀 ID (FK -> teams.id)")
    awayteam_id: Optional[int] = Field(None, description="원정팀 ID (FK -> teams.id)")

    stadium_code: Optional[str] = Field(None, description="경기장 코드 (예: C02, B04)")
    sche_date: Optional[str] = Field(None, description="경기 일정 날짜 (YYYYMMDD)")
    gubun: Optional[str] = Field(None, description="경기 구분 (예: 'Y'=진행됨, 'N'=예정)")

    hometeam_code: Optional[str] = Field(None, description="홈팀 코드")
    awayteam_code: Optional[str] = Field(None, description="원정팀 코드")

    home_score: Optional[int] = Field(None, description="홈팀 득점 (경기 전이면 null)")
    away_score: Optional[int] = Field(None, description="원정팀 득점 (경기 전이면 null)")


class ScheduleResponse(ScheduleBase):
    """경기 일정 응답 스키마."""

    id: int = Field(..., description="경기 일정 고유 ID (PK)")

    class Config:
        from_attributes = True


class ScheduleListResponse(BaseModel):
    """경기 일정 목록 응답 스키마."""

    items: list[ScheduleResponse] = Field(..., description="경기 일정 목록")
    total: int = Field(..., description="전체 개수")

