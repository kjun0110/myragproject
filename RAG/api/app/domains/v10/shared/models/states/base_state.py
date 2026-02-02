"""
공통 상태 베이스 클래스

LangGraph 상태 관리를 위한 공통 베이스 클래스입니다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class BaseProcessingState(TypedDict):
    """공통 처리 상태 베이스 클래스.

    모든 엔티티 처리 상태의 공통 필드를 정의합니다.
    """

    # 입력 데이터
    records: List[Dict[str, Any]]  # 처리할 레코드 리스트

    # 처리 단계 추적
    current_step: str  # 현재 처리 단계

    # 전략 판단 결과
    strategy_type: Optional[str]  # "policy" 또는 "rule"
    strategy_confidence: Optional[float]  # 전략 판단 신뢰도

    # 처리 결과
    result: Optional[Dict[str, Any]]  # 최종 처리 결과

    # 에러 정보
    errors: List[Dict[str, Any]]  # 에러 리스트

