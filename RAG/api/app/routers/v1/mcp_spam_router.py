"""
KoELECTRA 게이트웨이 라우터.

POST /api/mcp/gate
이메일 텍스트를 받아 KoELECTRA 게이트웨이로 스팸 확률을 계산합니다.

POST /api/mcp/spam-analyze
KoELECTRA로 판별 후 필요시 EXAONE으로 보내는 전체 스팸 분석을 수행합니다.
"""

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
router_dir = current_file.parent  # api/app/routers/v1/
routers_dir = router_dir.parent  # api/app/routers/
app_dir = routers_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

sys.path.insert(0, str(api_dir))

# 스키마 import
# 오케스트레이터 import (명시적 import로 순환참조 위험 감소)
from app.common.orchestrator.factory import OrchestratorFactory
from app.domains.v1.spam_classifier.models.base_model import (
    GateRequest,
    GateResponse,
    SpamAnalyzeRequest,
    SpamAnalyzeResponse,
)

# 스팸 분류 오케스트레이터 가져오기
_spam_orchestrator = None


def get_spam_orchestrator():
    """스팸 분류 오케스트레이터를 가져옵니다."""
    global _spam_orchestrator
    if _spam_orchestrator is None:
        _spam_orchestrator = OrchestratorFactory.get("spam_classifier")
    return _spam_orchestrator


router = APIRouter(prefix="/api/mcp", tags=["mcp"])


@router.post("/gate", response_model=GateResponse)
async def spam_gate(request: GateRequest):
    """KoELECTRA 게이트웨이 API 엔드포인트.

    중요: KoELECTRA는 스팸 판단을 하지 않습니다!
    - KoELECTRA의 역할: 정책 결정만 (정책 기반 vs 규칙 기반)
    - EXAONE의 역할: 실제 스팸 판단 (스팸 확률, 레이블, 분석)
    
    KoELECTRA는 정책 결정만 수행하며, 스팸 분류 도메인으로 분류되면 EXAONE이 호출됩니다.
    """
    try:
        # 도메인별 오케스트레이터 사용
        orchestrator = get_spam_orchestrator()

        # 도메인 분류만 수행
        gate_result = orchestrator.classify_domain(
            request.email_text, request.max_length, request.request_id
        )

        # 스팸 분류 도메인으로 분류되면 EXAONE 호출
        should_call = gate_result["domain"] == "spam"

        # GateResponse는 레거시 호환성을 위해 spam_prob, ham_prob 등을 포함
        return GateResponse(
            spam_prob=gate_result.get("spam_prob", 0.5),
            ham_prob=gate_result.get("ham_prob", 0.5),
            label=gate_result["domain"],  # "spam" 또는 "other"
            confidence=gate_result["confidence"],
            latency_ms=gate_result["latency_ms"],
            should_call_exaone=should_call,
            request_id=gate_result.get("request_id"),
        )

    except RuntimeError as e:
        error_msg = str(e)
        print(f"[ERROR] 게이트웨이 모델 로드 실패: {error_msg}")
        raise HTTPException(
            status_code=503,
            detail=f"게이트웨이 모델 로드 실패: {error_msg[:300]}",
        )
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] 게이트웨이 처리 실패: {error_msg}")
        import traceback

        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"게이트웨이 처리 중 오류가 발생했습니다: {error_msg[:300]}",
        )


# ==================== 상태 관리 엔드포인트 ====================
@router.get("/gate/state/{request_id}")
async def get_gate_state_endpoint(request_id: str):
    """저장된 게이트웨이 결과를 조회합니다.

    Args:
        request_id: 요청 ID

    Returns:
        게이트웨이 결과
    """
    try:
        orchestrator = get_spam_orchestrator()
        return orchestrator.get_state(request_id)
    except KeyError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )


@router.delete("/gate/state/{request_id}")
async def delete_gate_state_endpoint(request_id: str):
    """저장된 게이트웨이 결과를 삭제합니다.

    Args:
        request_id: 요청 ID

    Returns:
        삭제 성공 메시지
    """
    try:
        orchestrator = get_spam_orchestrator()
        return orchestrator.delete_state(request_id)
    except KeyError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )


@router.get("/gate/state")
async def list_gate_states_endpoint():
    """저장된 모든 게이트웨이 결과의 ID 목록을 반환합니다.

    Returns:
        요청 ID 목록
    """
    orchestrator = get_spam_orchestrator()
    return orchestrator.list_states()


# ==================== 스팸 분석 엔드포인트 ====================
@router.post("/spam-analyze", response_model=SpamAnalyzeResponse)
async def spam_analyze_with_tool(request: SpamAnalyzeRequest):
    """스팸 분석 API 엔드포인트.

    오케스트레이터를 사용하여 KoELECTRA 정책 결정 → EXAONE 스팸 판단 순서로 분석을 수행합니다.
    
    중요:
    - KoELECTRA: 정책 결정만 (정책 기반 vs 규칙 기반)
    - EXAONE: 실제 스팸 판단 (스팸 확률, 레이블, 분석)
    
    플로우:
    1. KoELECTRA로 정책 결정 (spam_prob > 0.3 → ANALYZE_SPAM)
    2. ANALYZE_SPAM 정책이면 EXAONE으로 스팸 판단 수행
    3. BYPASS 정책이면 규칙 기반 처리 (EXAONE 호출 안 함)
    """
    try:
        # 오케스트레이터를 사용하여 전체 플로우 실행
        orchestrator = get_spam_orchestrator()
        
        print("[INFO] /spam 페이지 요청 → 오케스트레이터 실행")
        print("[INFO] 1단계: KoELECTRA 정책 결정 (스팸 판단 아님)")
        print("[INFO] 2단계: 정책에 따라 EXAONE 스팸 판단 또는 규칙 기반 처리")
        result = orchestrator.analyze(email_text=request.email_text)
        
        # 오케스트레이터 결과 파싱
        gate_result = result.get("gate_result", {})
        agent_result = result.get("agent_result")
        service_result = result.get("service_result")
        final_decision = result.get("final_decision", "")
        
        # exaone_result는 agent_result에서 추출
        exaone_result = agent_result if agent_result else None
        
        return SpamAnalyzeResponse(
            gate_result=gate_result,
            exaone_result=exaone_result,
            final_decision=final_decision,
        )

    except RuntimeError as e:
        error_msg = str(e)
        print(f"[ERROR] 모델 로드 실패: {error_msg}")
        raise HTTPException(
            status_code=503,
            detail=f"모델 로드 실패: {error_msg[:300]}",
        )
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] 스팸 분석 실패: {error_msg}")
        import traceback

        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"스팸 분석 중 오류가 발생했습니다: {error_msg[:300]}",
        )
