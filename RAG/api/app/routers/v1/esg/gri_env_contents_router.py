"""GRI Environmental Contents 라우터.

GRI 환경 도메인 콘텐츠 관련 API 엔드포인트를 제공합니다.
"""

import logging
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
router_dir = current_file.parent.parent.parent  # api/app/routers/
app_dir = router_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

sys.path.insert(0, str(api_dir))

from app.domains.esg.total.models.gri_env_contents_model import (
    GRIEnvContentCreate,
    GRIEnvContentListResponse,
    GRIEnvContentResponse,
    GRIEnvContentUpdate,
)
from app.domains.esg.total.orchestrator.gri_env_contents_flow import (
    get_gri_env_contents_orchestrator,
)

router = APIRouter()


# 채팅 요청/응답 모델
class ChatRequest(BaseModel):
    """채팅 요청 모델."""
    message: str
    history: list[dict] | None = None


class ChatResponse(BaseModel):
    """채팅 응답 모델."""
    response: str


@router.post("/chat", response_model=ChatResponse)
async def chat_gri_env_content(request: ChatRequest, http_request: Request):
    """GRI 환경 도메인 콘텐츠 채팅 엔드포인트."""
    # 연결 확인 로그
    client_host = http_request.client.host if http_request.client else None
    logger.info("=" * 70)
    logger.info(f"[GRI_ENV_MCP] 채팅 요청 수신")
    logger.info(f"[GRI_ENV_MCP] 클라이언트 호스트: {client_host}")
    logger.info(f"[GRI_ENV_MCP] 메시지: {request.message}")
    logger.info(f"[GRI_ENV_MCP] 히스토리 길이: {len(request.history) if request.history else 0}")
    logger.info("=" * 70)

    try:
        # orchestrator를 사용하여 메시지 처리
        orchestrator = get_gri_env_contents_orchestrator()
        result = orchestrator.analyze(text=request.message)

        # 응답 생성 (실제로는 orchestrator의 결과를 사용)
        response_text = f"GRI 환경 도메인 콘텐츠에 대한 응답: {request.message}"

        logger.info(f"[GRI_ENV_MCP] 응답 생성 완료: {response_text[:100]}...")

        return ChatResponse(response=response_text)
    except Exception as e:
        logger.error(f"[GRI_ENV_MCP] 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"GRI 환경 도메인 콘텐츠 채팅 실패: {str(e)}"
        )


@router.post("/", response_model=GRIEnvContentResponse)
async def create_gri_env_content(request: GRIEnvContentCreate):
    """GRI 환경 도메인 콘텐츠 생성 엔드포인트."""
    try:
        orchestrator = get_gri_env_contents_orchestrator()
        result = orchestrator.analyze(
            text=f"GRI 환경 도메인 콘텐츠 생성: 표준 ID {request.standard_id} - {request.disclosure_num}",
        )

        # TODO: 실제 데이터베이스 저장 로직 구현
        return GRIEnvContentResponse(
            id=1,
            standard_id=request.standard_id,
            disclosure_num=request.disclosure_num,
            content=request.content,
            metadata=request.metadata,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"GRI 환경 도메인 콘텐츠 생성 실패: {str(e)}"
        )


@router.get("/", response_model=GRIEnvContentListResponse)
async def list_gri_env_contents(
    standard_id: int | None = None,
    disclosure_num: str | None = None,
):
    """GRI 환경 도메인 콘텐츠 목록 조회 엔드포인트."""
    try:
        query = "GRI 환경 도메인 콘텐츠 조회"
        if standard_id:
            query += f" 표준 ID: {standard_id}"
        if disclosure_num:
            query += f" 공개 번호: {disclosure_num}"

        orchestrator = get_gri_env_contents_orchestrator()
        result = orchestrator.analyze(text=query)

        # TODO: 실제 데이터베이스 조회 로직 구현
        return GRIEnvContentListResponse(items=[], total=0)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"GRI 환경 도메인 콘텐츠 조회 실패: {str(e)}"
        )


@router.get("/{content_id}", response_model=GRIEnvContentResponse)
async def get_gri_env_content(content_id: int):
    """GRI 환경 도메인 콘텐츠 상세 조회 엔드포인트."""
    try:
        query = f"GRI 환경 도메인 콘텐츠 ID {content_id} 상세 조회"
        orchestrator = get_gri_env_contents_orchestrator()
        result = orchestrator.analyze(text=query)

        # TODO: 실제 데이터베이스 조회 로직 구현
        raise HTTPException(
            status_code=404, detail="GRI 환경 도메인 콘텐츠를 찾을 수 없습니다."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"GRI 환경 도메인 콘텐츠 조회 실패: {str(e)}"
        )


@router.put("/{content_id}", response_model=GRIEnvContentResponse)
async def update_gri_env_content(content_id: int, request: GRIEnvContentUpdate):
    """GRI 환경 도메인 콘텐츠 수정 엔드포인트."""
    try:
        query = f"GRI 환경 도메인 콘텐츠 ID {content_id} 수정"
        orchestrator = get_gri_env_contents_orchestrator()
        result = orchestrator.analyze(text=query)

        # TODO: 실제 데이터베이스 수정 로직 구현
        raise HTTPException(
            status_code=404, detail="GRI 환경 도메인 콘텐츠를 찾을 수 없습니다."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"GRI 환경 도메인 콘텐츠 수정 실패: {str(e)}"
        )


@router.delete("/{content_id}")
async def delete_gri_env_content(content_id: int):
    """GRI 환경 도메인 콘텐츠 삭제 엔드포인트."""
    try:
        query = f"GRI 환경 도메인 콘텐츠 ID {content_id} 삭제"
        orchestrator = get_gri_env_contents_orchestrator()
        result = orchestrator.analyze(text=query)

        # TODO: 실제 데이터베이스 삭제 로직 구현
        return {"message": f"GRI 환경 도메인 콘텐츠 ID {content_id}가 삭제되었습니다."}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"GRI 환경 도메인 콘텐츠 삭제 실패: {str(e)}"
        )
