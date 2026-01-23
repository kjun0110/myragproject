"""GRI Governance Contents Pydantic 모델.

API 요청/응답을 위한 Pydantic 스키마입니다.
"""

from typing import Optional

from pydantic import BaseModel, Field


class GRIGovContentBase(BaseModel):
    """GRI 지배구조 도메인 콘텐츠 기본 모델."""

    standard_id: int = Field(..., description="GRI 표준 ID (외래키)")
    disclosure_num: Optional[str] = Field(None, description="공개 번호 (예: '2-9')", max_length=20)
    content: str = Field(..., description="지침 전문")
    metadata: Optional[dict] = Field(None, description="측정 단위 등 추가 정보 (JSONB)")


class GRIGovContentCreate(GRIGovContentBase):
    """GRI 지배구조 도메인 콘텐츠 생성 요청 모델."""

    pass


class GRIGovContentUpdate(BaseModel):
    """GRI 지배구조 도메인 콘텐츠 수정 요청 모델."""

    standard_id: Optional[int] = Field(None, description="GRI 표준 ID")
    disclosure_num: Optional[str] = Field(None, description="공개 번호", max_length=20)
    content: Optional[str] = Field(None, description="지침 전문")
    metadata: Optional[dict] = Field(None, description="추가 정보 (JSONB)")


class GRIGovContentResponse(GRIGovContentBase):
    """GRI 지배구조 도메인 콘텐츠 응답 모델."""

    id: int = Field(..., description="기본 키")

    class Config:
        """Pydantic 설정."""

        from_attributes = True  # SQLAlchemy 모델에서 자동 변환


class GRIGovContentListResponse(BaseModel):
    """GRI 지배구조 도메인 콘텐츠 목록 응답 모델."""

    items: list[GRIGovContentResponse] = Field(..., description="GRI 지배구조 도메인 콘텐츠 목록")
    total: int = Field(..., description="전체 개수")
