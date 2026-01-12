"""
FastAPI 기준의 API 엔드포인트 계층입니다.

graph_router.py
POST /api/graph
LangGraph를 사용한 대화형 응답 반환.
"""

import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["graph"])


class GraphRequest(BaseModel):
    """LangGraph 요청 모델."""

    message: str
    history: Optional[List[dict]] = []


class GraphResponse(BaseModel):
    """LangGraph 응답 모델."""

    response: str


@router.post("/graph", response_model=GraphResponse)
async def graph_chat(request: GraphRequest, http_request: Request):
    """LangGraph API 엔드포인트 - graph.invoke를 사용한 대화."""
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
        from app.graph import graph

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

        # Graph 실행
        state = {"messages": messages}
        out = graph.invoke(state)

        # 마지막 메시지에서 응답 추출
        answer = out["messages"][-1].content

        return GraphResponse(response=answer)

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
