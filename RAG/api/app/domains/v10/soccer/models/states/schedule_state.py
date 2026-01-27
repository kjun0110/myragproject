"""
Schedule 처리 상태 스키마

LangGraph를 사용한 Schedule 데이터 처리 상태를 정의합니다.
"""

from typing import Any, Dict, List, Optional, TypedDict

from app.domains.v10.soccer.models.states.base_state import BaseProcessingState


class ScheduleProcessingState(BaseProcessingState):
    """Schedule 데이터 처리 상태."""
    records: List[Dict[str, Any]]
    current_step: str
    strategy_type: Optional[str]
    strategy_confidence: Optional[float]
    heuristic_result: Optional[str]
    koelectra_result: Optional[str]
    validated_records: Optional[List[Dict[str, Any]]]
    validation_errors: Optional[List[Dict[str, Any]]]
    result: Optional[Dict[str, Any]]
    processed_count: Optional[int]
    errors: List[Dict[str, Any]]
