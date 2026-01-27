"""
Player 처리 상태 스키마

LangGraph를 사용한 Player 데이터 처리 상태를 정의합니다.
"""

from typing import Any, Dict, List, Optional, TypedDict

from app.domains.v10.soccer.models.states.base_state import BaseProcessingState


class PlayerProcessingState(BaseProcessingState):
    """Player 데이터 처리 상태.
    
    Player Orchestrator의 LangGraph 상태를 정의합니다.
    """
    # 입력 데이터 (BaseProcessingState에서 상속)
    records: List[Dict[str, Any]]  # 처리할 Player 레코드 리스트
    
    # 처리 단계 추적
    current_step: str  # 현재 처리 단계 (validate, determine_strategy, process, finalize)
    
    # 전략 판단 결과
    strategy_type: Optional[str]  # "policy" 또는 "rule"
    strategy_confidence: Optional[float]  # 전략 판단 신뢰도 (0.0 ~ 1.0)
    heuristic_result: Optional[str]  # 휴리스틱 판단 결과
    koelectra_result: Optional[str]  # KoELECTRA 판단 결과
    
    # 검증 결과
    validated_records: Optional[List[Dict[str, Any]]]  # 검증된 레코드 리스트
    validation_errors: Optional[List[Dict[str, Any]]]  # 검증 에러 리스트
    
    # 처리 결과
    result: Optional[Dict[str, Any]]  # 최종 처리 결과
    processed_count: Optional[int]  # 처리된 레코드 수
    
    # 에러 정보
    errors: List[Dict[str, Any]]  # 에러 리스트
