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
# 오케스트레이터 import
from app.common.orchestrator import OrchestratorFactory
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

    KoELECTRA로 도메인 분류만 수행합니다.
    스팸 분류 도메인으로 분류되면 EXAONE이 자동으로 호출됩니다.
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

    /spam 페이지에서 직접 호출되는 엔드포인트로,
    KoELECTRA 게이트웨이를 건너뛰고 EXAONE이 직접 스팸 확률 계산 및 판단을 수행합니다.
    신뢰도 관계없이 모든 판독은 EXAONE이 담당합니다.
    """
    try:
        # /spam 페이지에서는 KoELECTRA를 건너뛰고 바로 EXAONE 호출
        # 신뢰도 관계없이 모든 판독은 EXAONE이 담당
        import json

        from app.domains.v1.spam_classifier.agents import get_exaone_tool

        print("[INFO] /spam 페이지 요청 → EXAONE 직접 호출 (KoELECTRA 건너뛰기)...")
        exaone_tool = get_exaone_tool()
        exaone_result_str = exaone_tool.invoke({"email_text": request.email_text})

        # JSON 문자열 파싱
        try:
            exaone_result = json.loads(exaone_result_str)
        except (json.JSONDecodeError, TypeError):
            exaone_result = {"analysis": exaone_result_str}

        # gate_result는 참고용으로 생성 (실제로는 EXAONE이 판단)
        gate_result = {
            "domain": "spam",  # /spam 페이지에서는 항상 spam 도메인으로 간주
            "policy": "ANALYZE_SPAM",  # KoELECTRA를 건너뛰므로 정책은 명시적으로 ANALYZE_SPAM
            "confidence": "high",
            "latency_ms": 0.0,  # 게이트웨이를 건너뛰므로 0
            "spam_prob": exaone_result.get("spam_prob", 0.5)
            if isinstance(exaone_result, dict)
            else 0.5,
            "ham_prob": exaone_result.get("ham_prob", 0.5)
            if isinstance(exaone_result, dict)
            else 0.5,
        }

        # 최종 결정 생성 (EXAONE이 전부 수행)
        if isinstance(exaone_result, dict):
            spam_prob = exaone_result.get("spam_prob", 0.5)
            label = exaone_result.get("label", "unknown")
            confidence = exaone_result.get("confidence", "medium")
            analysis = exaone_result.get("analysis", "")

            final_decision = f"""EXAONE Reader 스팸 분석:
- 스팸 확률: {spam_prob:.4f}
- 레이블: {label}
- 신뢰도: {confidence}
- 상세 분석:
{analysis}"""
        else:
            final_decision = f"EXAONE Reader 스팸 분석:\n{exaone_result}"

        return SpamAnalyzeResponse(
            gate_result=gate_result,
            exaone_result=exaone_result,  # 이미 dict 형태로 파싱됨
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
