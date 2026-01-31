"""
FastAPI 기준의 API 엔드포인트 계층입니다.

chat_router.py
POST /api/chain
세션 ID, 메시지 리스트 등을 받아 대화형 응답 반환.
"""

# 스키마 import
from app.domains.v1.chat.models.transfers.base_model import ChatRequest, ChatResponse
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api", tags=["chat"])


def get_chat_service():
    """ChatService 인스턴스를 반환하는 함수.

    이 함수는 main.py 또는 mainbackup.py의 전역 변수에 접근하기 위해
    main 모듈에서 import하여 사용합니다.
    순환 import 방지를 위해 함수 내부에서 import합니다.
    """
    # 순환 import 방지를 위해 함수 내부에서 import
    import sys
    import importlib

    # main 또는 mainbackup 모듈 찾기
    main = None
    
    # app.mainbackup이 로드되어 있으면 사용 (우선순위)
    if "app.mainbackup" in sys.modules:
        main = importlib.import_module("app.mainbackup")
    # app.main이 로드되어 있으면 사용
    elif "app.main" in sys.modules:
        from ... import main
    else:
        # 둘 다 없으면 mainbackup 우선 시도
        try:
            main = importlib.import_module("app.mainbackup")
        except ImportError:
            # mainbackup이 없으면 main 시도
            main = importlib.import_module("app.main")

    if main is None or not hasattr(main, "chat_service"):
        raise RuntimeError(
            "ChatService가 초기화되지 않았습니다. "
            "main.py 또는 mainbackup.py에서 chat_service를 초기화해주세요."
        )

    return main.chat_service


@router.post("/chain", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request):
    """챗봇 API 엔드포인트 - ChatService를 사용한 RAG 체인."""
    # ChatService 인스턴스 가져오기
    chat_service = get_chat_service()
    if chat_service is None:
        raise HTTPException(
            status_code=503,
            detail="ChatService가 초기화되지 않았습니다. 서버를 재시작해주세요.",
        )

    try:
        # 라우팅/정책/환경 분기는 오케스트레이터에서 담당
        from app.domains.v1.chat.hub.orchestrators.chat_orchestrator import ChatOrchestrator

        orch = ChatOrchestrator()
        result = orch.route_chat(
            message=request.message,
            history=request.history or [],
            client_host=http_request.client.host if http_request.client else None,
            chat_service=chat_service,
        )

        if result.mode == "stream":
            return StreamingResponse(
                result.stream,  # type: ignore[arg-type]
                media_type="text/plain; charset=utf-8",
            )

        return ChatResponse(response=result.text or "")

    except ValueError as e:
        error_msg = str(e)
        print(f"[ERROR] 잘못된 요청: {error_msg}")
        raise HTTPException(
            status_code=400,
            detail=error_msg,
        )

    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] 챗봇 응답 생성 실패: {error_msg}")

        # OpenAI API 할당량 초과 에러 확인 (1번만 체크)
        if (
            "할당량" in error_msg
            or "quota" in error_msg.lower()
            or "429" in error_msg
            or "insufficient_quota" in error_msg
            or "exceeded" in error_msg.lower()
        ):
            error_detail = (
                "⚠️ OpenAI API 할당량이 초과되었습니다.\n\n"
                "해결 방법:\n"
                "1. OpenAI 계정의 사용량 및 할당량을 확인하세요\n"
                "2. OpenAI 계정에 결제 정보를 추가하거나 할당량을 늘리세요\n"
                "3. 또는 '🖥️ 로컬 모델' 버튼을 선택하여 로컬 EXAONE 모델을 사용하세요"
            )
            raise HTTPException(
                status_code=429,
                detail=error_detail,
            )
        else:
            # RuntimeError는 503, 기타는 500
            status_code = 503 if isinstance(e, RuntimeError) else 500
            raise HTTPException(
                status_code=status_code,
                detail=f"응답 생성 중 오류가 발생했습니다: {error_msg[:200]}",
            )
