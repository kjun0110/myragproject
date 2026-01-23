"""
FastAPI 기준의 API 엔드포인트 계층입니다.

chat_router.py
POST /api/chain
세션 ID, 메시지 리스트 등을 받아 대화형 응답 반환.
"""

import os

# 스키마 import
from app.domains.v1.chat.models.base_model import ChatRequest, ChatResponse
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
import asyncio

router = APIRouter(prefix="/api", tags=["chat"])


def get_chat_service():
    """ChatService 인스턴스를 반환하는 함수.

    이 함수는 main.py의 전역 변수에 접근하기 위해
    main 모듈에서 import하여 사용합니다.
    순환 import 방지를 위해 함수 내부에서 import합니다.
    """
    # 순환 import 방지를 위해 함수 내부에서 import
    import sys

    # main 모듈이 이미 로드되어 있는지 확인
    if "app.main" in sys.modules:
        from ... import main
    else:
        # 모듈이 아직 로드되지 않은 경우 직접 import
        import importlib

        main = importlib.import_module("app.main")

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

    # 요청자의 IP 주소 확인 (localhost 여부 판단)
    client_host = http_request.client.host if http_request.client else None
    is_localhost = (
        client_host == "127.0.0.1"
        or client_host == "localhost"
        or client_host == "::1"
        or (client_host and client_host.startswith("127."))
    )

    # 모델 타입에 따라 적절한 RAG 체인 선택
    # 프론트엔드에서 전달된 model_type이 없으면 .env의 LLM_PROVIDER 사용
    model_type = request.model_type or os.getenv("LLM_PROVIDER", "openai")
    if model_type:
        model_type = model_type.lower()

    # 환경과 모델 타입 불일치 검증
    if is_localhost and model_type == "openai":
        raise HTTPException(
            status_code=400,
            detail="현재 로컬환경입니다. 로컬 모델을 사용해주세요.",
        )

    if not is_localhost and model_type in ["local", "graph"]:
        raise HTTPException(
            status_code=400,
            detail="현재 로컬 환경이 아닙니다. OpenAI 모델을 사용해주세요.",
        )

    # 디버깅: 받은 model_type 로그 출력
    print(
        f"[DEBUG] 받은 model_type: {request.model_type}, 처리된 model_type: {model_type}, client_host: {client_host}, is_localhost: {is_localhost}"
    )

    try:
        # graph 모드일 때는 LangGraph 사용 (Exaone 모델)
        if model_type == "graph":
            print("[DEBUG] graph 모드 감지 - LangGraph (Exaone) 사용")
            from app.domains.v1.chat.agents.graph import run_once

            response_text = run_once(request.message)
            return ChatResponse(response=response_text)

        # 그 외 모드는 ChatService를 통해 RAG 체인 실행 (스트리밍)
        # 스트리밍 제너레이터 생성
        async def stream_response():
            try:
                # 적절한 RAG 체인 선택
                if model_type == "openai":
                    if not chat_service.openai_rag_chain:
                        if chat_service.openai_quota_exceeded:
                            yield "OpenAI API 할당량이 초과되었습니다."
                            return
                        else:
                            yield "OpenAI RAG 체인이 초기화되지 않았습니다."
                            return
                    current_rag_chain = chat_service.openai_rag_chain
                elif model_type == "local":
                    if not chat_service.local_rag_chain:
                        yield "로컬 RAG 체인이 초기화되지 않았습니다."
                        return
                    current_rag_chain = chat_service.local_rag_chain
                else:
                    yield f"지원하지 않는 모델 타입입니다: {model_type}"
                    return

                # 대화 기록을 LangChain 메시지 형식으로 변환
                chat_history = []
                if request.history:
                    for msg in request.history:
                        if msg.get("role") == "user":
                            chat_history.append(HumanMessage(content=msg.get("content", "")))
                        elif msg.get("role") == "assistant":
                            chat_history.append(AIMessage(content=msg.get("content", "")))

                # RAG 체인 스트리밍 실행
                accumulated_text = ""
                full_response = ""
                async for chunk in current_rag_chain.astream(
                    {
                        "input": request.message,
                        "chat_history": chat_history,
                    }
                ):
                    # chunk에서 answer 추출
                    if isinstance(chunk, dict):
                        answer = chunk.get("answer", "")
                        if answer:
                            full_response = answer
                            # 새로 추가된 부분만 추출
                            if len(answer) > len(accumulated_text):
                                delta = answer[len(accumulated_text):]
                                accumulated_text = answer
                                # 한 글자씩 스트리밍
                                for char in delta:
                                    yield char
                                    await asyncio.sleep(0.01)
                    elif isinstance(chunk, str):
                        full_response = chunk
                        if len(chunk) > len(accumulated_text):
                            delta = chunk[len(accumulated_text):]
                            accumulated_text = chunk
                            # 한 글자씩 스트리밍
                            for char in delta:
                                yield char
                                await asyncio.sleep(0.01)

                # 스트리밍 완료 후 최종 응답 정리 (태그 제거 등)
                # 이미 chat_with_rag에서 정리하지만, 스트리밍에서는 직접 처리하므로
                # 여기서도 정리 로직을 적용할 수 있음 (선택사항)

            except Exception as e:
                error_msg = str(e)
                print(f"[ERROR] 스트리밍 중 오류: {error_msg}")
                yield f"\n\n[오류 발생: {error_msg}]"

        # 스트리밍 응답 반환
        return StreamingResponse(
            stream_response(),
            media_type="text/plain; charset=utf-8",
        )

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
                "3. 또는 '🖥️ 로컬 모델' 버튼을 선택하여 로컬 Midm 모델을 사용하세요"
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
