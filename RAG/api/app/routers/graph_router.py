"""
FastAPI 기준의 API 엔드포인트 계층입니다.

graph_router.py
POST /api/graph
LangGraph를 사용한 대화형 응답 반환 (스트리밍).
"""

# 스키마 import
import re
from app.domains.chat.models.base_model import GraphRequest, GraphResponse
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

router = APIRouter(prefix="/api", tags=["graph"])


@router.post("/graph")
async def graph_chat(request: GraphRequest, http_request: Request):
    """LangGraph API 엔드포인트 - graph.stream을 사용한 스트리밍 대화."""
    # 요청자의 IP 주소 확인 (localhost 여부 판단)
    client_host = http_request.client.host if http_request.client else None
    is_localhost = (
        client_host == "127.0.0.1"
        or client_host == "localhost"
        or client_host == "::1"
        or (client_host and client_host.startswith("127."))
    )

    # 로컬 환경이 아니면 에러 (graph는 로컬 midm 모델 사용)
    if not is_localhost:
        raise HTTPException(
            status_code=400,
            detail="현재 로컬 환경이 아닙니다. OpenAI 모델을 사용해주세요.",
        )

    # 디버깅 로그
    print(
        f"[DEBUG] LangGraph 요청 - client_host: {client_host}, is_localhost: {is_localhost}"
    )

    try:
        # graph import (lazy loading이므로 모델은 실제 사용 시 로드됨)
        from app.domains.chat.agents.graph import graph

        # 대화 기록을 LangChain 메시지 형식으로 변환
        messages = [
            SystemMessage(content="You are a helpful assistant. Use tools when needed.")
        ]

        # 히스토리 추가
        if request.history:
            for msg in request.history:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        # 현재 메시지 추가
        messages.append(HumanMessage(content=request.message))

        # Graph 실행 (스트리밍)
        state = {"messages": messages}

        # 증분 업데이트를 위한 이전 텍스트 길이 추적
        previous_length = 0

        async def stream_generator():
            """스트리밍 제너레이터 - 토큰 단위 증분 업데이트."""
            try:
                print("[DEBUG] graph.astream_events 시작")
                event_count = 0

                # astream_events를 사용하여 토큰 단위 스트리밍
                async for event in graph.astream_events(state, version="v1"):
                    event_count += 1
                    event_type = event.get("event")

                    # on_chat_model_stream 이벤트만 처리 (토큰 단위)
                    if event_type == "on_chat_model_stream":
                        # chunk에서 content 추출
                        chunk_data = event.get("data", {})
                        chunk = chunk_data.get("chunk")

                        if chunk and hasattr(chunk, "content"):
                            content = chunk.content
                            if content:
                                # 태그 제거 (실시간으로)
                                if not any(tag in content for tag in ["[[system]]", "[[endofturn]]", "[[user]]", "[[assistant]]"]):
                                    print(f"[DEBUG] 토큰 전송: {content}")
                                    yield content
                                    import asyncio
                                    await asyncio.sleep(0.01)  # 10ms 지연

                    # 디버깅: 다른 중요한 이벤트도 로그
                    elif event_type in ["on_chat_model_start", "on_chat_model_end"]:
                        print(f"[DEBUG] 이벤트: {event_type}")

                print(f"[DEBUG] graph.astream_events 완료, 총 {event_count}개 이벤트 처리")

            except Exception as e:
                print(f"[ERROR] 스트리밍 중 오류: {e}")
                import traceback
                traceback.print_exc()
                yield f"\n\n[오류 발생: {str(e)}]"

        # StreamingResponse 반환
        return StreamingResponse(
            stream_generator(),
            media_type="text/plain; charset=utf-8",
        )

    except FileNotFoundError as e:
        error_msg = str(e)
        print(f"[ERROR] Exaone3.5 모델을 찾을 수 없습니다: {error_msg}")
        import traceback

        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=503,
            detail=f"Exaone3.5 모델을 찾을 수 없습니다. 모델 경로를 확인해주세요: {error_msg[:300]}",
        )
    except RuntimeError as e:
        error_msg = str(e)
        print(f"[ERROR] Exaone3.5 모델 로드 실패: {error_msg}")
        import traceback

        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=503,
            detail=f"Exaone3.5 모델 로드 실패: {error_msg[:300]}",
        )
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] LangGraph 응답 생성 실패: {error_msg}")
        import traceback

        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")

        raise HTTPException(
            status_code=500,
            detail=f"LangGraph 응답 생성 중 오류가 발생했습니다: {error_msg[:300]}",
        )
