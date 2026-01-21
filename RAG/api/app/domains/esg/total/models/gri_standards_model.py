"""GRI Standards Pydantic 모델.

API 요청/응답을 위한 Pydantic 스키마입니다.
"""

from typing import Optional

from pydantic import BaseModel, Field


class GRIStandardBase(BaseModel):
    """GRI 표준 기본 모델."""

    standard_code: str = Field(..., description="GRI 표준 코드 (예: 'GRI 305')", max_length=20)
    standard_name: str = Field(..., description="표준 이름 (예: 'Emissions')")
    category: Optional[str] = Field(
        None, description="카테고리 (예: 'Environmental', 'Social', 'Governance', 'Economic')", max_length=50
    )
    published_year: Optional[int] = Field(2021, description="발행 연도")


class GRIStandardCreate(GRIStandardBase):
    """GRI 표준 생성 요청 모델."""

    pass


class GRIStandardUpdate(BaseModel):
    """GRI 표준 수정 요청 모델."""

    standard_code: Optional[str] = Field(None, description="GRI 표준 코드", max_length=20)
    standard_name: Optional[str] = Field(None, description="표준 이름")
    category: Optional[str] = Field(None, description="카테고리", max_length=50)
    published_year: Optional[int] = Field(None, description="발행 연도")


class GRIStandardResponse(GRIStandardBase):
    """GRI 표준 응답 모델."""

    id: int = Field(..., description="기본 키")

    class Config:
        """Pydantic 설정."""

        from_attributes = True  # SQLAlchemy 모델에서 자동 변환


class GRIStandardListResponse(BaseModel):
    """GRI 표준 목록 응답 모델."""

    items: list[GRIStandardResponse] = Field(..., description="GRI 표준 목록")
    total: int = Field(..., description="전체 개수")
