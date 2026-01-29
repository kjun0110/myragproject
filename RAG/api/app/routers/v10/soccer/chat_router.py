"""
Chat 라우터 - 채팅 질문을 Chat Orchestrator로 전달
"""

import logging
from pydantic import BaseModel

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v10/soccer/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    """채팅 요청 모델."""
    question: str


@router.post("")
async def chat_with_orchestrator(
    request: ChatRequest,
):
    """
    채팅 질문을 Chat Orchestrator로 전달합니다.
    
    Args:
        request: 채팅 요청 (question 포함)
    
    Returns:
        처리 결과
    """
    logger.info("[ROUTER] Chat 라우터 도달")
    logger.info(f"[ROUTER] 질문: {request.question}")
    
    try:
        # Chat Orchestrator를 통한 처리
        logger.info("[ROUTER] Chat Orchestrator를 통한 처리 시작")
        from app.domains.v10.soccer.hub.orchestrators.chat_orchestrator import ChatOrchestrator
        
        orchestrator = ChatOrchestrator()
        result = await orchestrator.process_question(request.question)
        
        logger.info(f"[ROUTER] Chat 처리 완료")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "question": request.question,
                "result": result,
            },
        )
    
    except Exception as e:
        logger.error(f"채팅 처리 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"채팅 처리 실패: {str(e)}",
        )
