"""
판독기 에이전트 상태 모델 정의.

LangGraph에서 사용하는 상태 구조를 정의합니다.
"""

from typing import Optional, TypedDict


class VerdictAgentState(TypedDict):
    """판독기 에이전트 상태."""

    email_text: str  # 입력 이메일 텍스트
    gate_result: dict  # KoELECTRA 게이트웨이 결과
    exaone_result: Optional[str]  # EXAONE Reader 결과
    should_call_exaone: Optional[bool]  # EXAONE 호출 여부
