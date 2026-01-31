"""규칙 기반 데이터 스키마 정의.

API 요청/응답을 위한 Pydantic 모델들을 정의합니다.
"""

from typing import List, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """챗봇 요청 모델."""

    message: str
    history: Optional[List[dict]] = []
    model_type: Optional[str] = "openai"  # "openai" 또는 "local"


class ChatResponse(BaseModel):
    """챗봇 응답 모델."""

    response: str


class GraphRequest(BaseModel):
    """LangGraph 요청 모델."""

    message: str
    history: Optional[List[dict]] = []


class GraphResponse(BaseModel):
    """LangGraph 응답 모델."""

    response: str

