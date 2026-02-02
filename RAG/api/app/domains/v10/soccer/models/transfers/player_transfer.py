"""Player Transfer(Pydantic) 모델.

`app.domains.v10.soccer.models.bases.players.Player`(SQLAlchemy ORM)와 동일한 필드 구성을
API 요청/응답 스키마(BaseModel)로 제공합니다.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class PlayerBase(BaseModel):
    """선수(Player) 기본 스키마.

    SQLAlchemy 모델(`models/bases/players.py`)의 컬럼을 기준으로 정의합니다.
    """

    team_id: Optional[int] = Field(None, description="팀 ID (FK -> teams.id)")
    team_code: str = Field(..., description="소속 팀 코드 (예: K06, K01)")

    player_name: str = Field(..., description="선수 이름(한글)")
    e_player_name: Optional[str] = Field(None, description="선수 영문 이름")
    nickname: Optional[str] = Field(None, description="선수 별명")

    join_yyyy: Optional[str] = Field(None, description="입단 연도 (YYYY)")
    position: Optional[str] = Field(None, description="포지션 (DF, MF, FW, GK)")
    back_no: Optional[int] = Field(None, description="등번호")

    nation: Optional[str] = Field(None, description="국적")
    birth_date: Optional[date] = Field(None, description="생년월일")
    solar: Optional[str] = Field(None, description="양력/음력 구분 (예: '1'=양력)")

    height: Optional[int] = Field(None, description="키(cm)")
    weight: Optional[int] = Field(None, description="몸무게(kg)")


class PlayerCreate(PlayerBase):
    """선수 생성 요청 스키마."""

    pass


class PlayerUpdate(BaseModel):
    """선수 수정 요청 스키마(부분 업데이트)."""

    team_id: Optional[int] = Field(None, description="팀 ID (FK -> teams.id)")
    team_code: Optional[str] = Field(None, description="소속 팀 코드 (예: K06, K01)")

    player_name: Optional[str] = Field(None, description="선수 이름(한글)")
    e_player_name: Optional[str] = Field(None, description="선수 영문 이름")
    nickname: Optional[str] = Field(None, description="선수 별명")

    join_yyyy: Optional[str] = Field(None, description="입단 연도 (YYYY)")
    position: Optional[str] = Field(None, description="포지션 (DF, MF, FW, GK)")
    back_no: Optional[int] = Field(None, description="등번호")

    nation: Optional[str] = Field(None, description="국적")
    birth_date: Optional[date] = Field(None, description="생년월일")
    solar: Optional[str] = Field(None, description="양력/음력 구분 (예: '1'=양력)")

    height: Optional[int] = Field(None, description="키(cm)")
    weight: Optional[int] = Field(None, description="몸무게(kg)")


class PlayerResponse(PlayerBase):
    """선수 응답 스키마."""

    id: int = Field(..., description="선수 고유 ID (PK)")

    class Config:
        # SQLAlchemy ORM 객체에서 자동 변환 지원
        from_attributes = True


class PlayerListResponse(BaseModel):
    """선수 목록 응답 스키마."""

    items: list[PlayerResponse] = Field(..., description="선수 목록")
    total: int = Field(..., description="전체 개수")

