"""
Player 처리 상태 스키마

LangGraph를 사용한 Player 데이터 처리 상태를 정의합니다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from app.domains.v10.shared.models.states.base_state import BaseProcessingState


class PlayerEmbeddingRecord(TypedDict):
    """임베딩 1건에 대응하는 상태용 레코드 (레거시 호환).

    이전 players_embeddings 테이블 형식. 현재는 players.embedding 컬럼을 사용합니다.
    """

    id: int
    player_id: int
    content: str
    # pgvector(Vector) 값은 구현에 따라 list[float] / numpy array 등으로 표현될 수 있어 Any로 둡니다.
    embedding: Any
    created_at: datetime


class PlayerProcessingState(BaseProcessingState):
    """Player 데이터 처리 상태.

    Player Orchestrator의 LangGraph 상태를 정의합니다.
    """
    # BaseProcessingState의 공통 필드 외, Player 도메인 특화 필드들
    heuristic_result: Optional[str]
    koelectra_result: Optional[str]

    validated_records: Optional[List[Dict[str, Any]]]
    validation_errors: Optional[List[Dict[str, Any]]]

    processed_count: Optional[int]

    # (선택) 임베딩 테이블 관련 결과를 상태로 보관할 때 사용
    embedding_records: Optional[List[PlayerEmbeddingRecord]]


# 과거/호환용 별칭 (의미상 PlayerProcessingState가 정식 명칭)
PlayerState = PlayerProcessingState

