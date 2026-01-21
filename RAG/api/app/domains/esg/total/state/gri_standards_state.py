"""
GRI Standards 에이전트 상태 모델 정의.

LangGraph에서 사용하는 상태 구조를 정의합니다.
"""

from typing import Optional

from pydantic import BaseModel, Field


class GRIStandardsAgentState(BaseModel):
    """GRI Standards 에이전트 상태."""

    # 입력
    query: str = Field(..., description="사용자 쿼리 (GRI 표준 검색/조회 요청)")
    standard_code: Optional[str] = Field(None, description="특정 GRI 표준 코드 (예: 'GRI 305')")
    category: Optional[str] = Field(
        None, description="카테고리 필터 (예: 'Environmental', 'Social', 'Governance')"
    )

    # 중간 처리 결과
    search_result: Optional[dict] = Field(None, description="검색 결과")
    standard_details: Optional[dict] = Field(None, description="표준 상세 정보")

    # 최종 결과
    response: Optional[str] = Field(None, description="최종 응답 텍스트")
    standards_list: Optional[list] = Field(None, description="GRI 표준 목록")

    # 메타데이터
    request_id: Optional[str] = Field(None, description="요청 ID (상태 관리용)")
    error: Optional[str] = Field(None, description="오류 메시지")

    class Config:
        """Pydantic 설정."""

        from_attributes = True
        arbitrary_types_allowed = True
