"""
GRI Social Contents 에이전트 상태 모델 정의.

LangGraph에서 사용하는 상태 구조를 정의합니다.
"""

from typing import Optional

from pydantic import BaseModel, Field


class GRISocContentsAgentState(BaseModel):
    """GRI 사회 도메인 콘텐츠 에이전트 상태."""

    # 입력
    query: str = Field(..., description="사용자 쿼리 (사회 도메인 콘텐츠 검색/조회 요청)")
    standard_id: Optional[int] = Field(None, description="GRI 표준 ID")
    disclosure_num: Optional[str] = Field(None, description="공개 번호 (예: '401-1')")

    # 중간 처리 결과
    search_result: Optional[dict] = Field(None, description="검색 결과")
    content_details: Optional[dict] = Field(None, description="콘텐츠 상세 정보")
    related_standard: Optional[dict] = Field(None, description="관련 GRI 표준 정보")

    # 최종 결과
    response: Optional[str] = Field(None, description="최종 응답 텍스트")
    contents_list: Optional[list] = Field(None, description="사회 도메인 콘텐츠 목록")

    # 메타데이터
    request_id: Optional[str] = Field(None, description="요청 ID (상태 관리용)")
    error: Optional[str] = Field(None, description="오류 메시지")

    class Config:
        """Pydantic 설정."""

        from_attributes = True
        arbitrary_types_allowed = True
