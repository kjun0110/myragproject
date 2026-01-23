"""
규칙 기반 데이터 스키마 정의.

API 요청/응답을 위한 Pydantic 모델들을 정의합니다.
"""

from typing import Optional

from pydantic import BaseModel


class GateRequest(BaseModel):
    """게이트웨이 요청 모델."""

    email_text: str  # 이메일 텍스트 (제목, 본문 등)
    max_length: Optional[int] = 512  # 최대 토큰 길이
    request_id: Optional[str] = None  # 요청 ID (상태 관리용, 선택사항)


class GateResponse(BaseModel):
    """게이트웨이 응답 모델."""

    spam_prob: float  # 스팸 확률 (0.0 ~ 1.0)
    ham_prob: float  # 정상 확률 (0.0 ~ 1.0)
    label: str  # "spam" 또는 "ham"
    confidence: str  # "low", "medium", "high"
    latency_ms: float  # 처리 시간 (밀리초)
    should_call_exaone: bool  # EXAONE Reader 호출 필요 여부
    request_id: Optional[str] = None  # 요청 ID (있는 경우)


class SpamAnalyzeRequest(BaseModel):
    """스팸 분석 요청 모델."""

    email_text: str  # 이메일 텍스트


class SpamAnalyzeResponse(BaseModel):
    """스팸 분석 응답 모델."""

    gate_result: dict  # KoELECTRA 게이트웨이 결과
    exaone_result: Optional[dict]  # EXAONE Reader 결과 (dict 형태, None일 수 있음)
    final_decision: str  # 최종 결정
