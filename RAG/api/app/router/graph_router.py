"""
FastAPI 기준의 API 엔드포인트 계층입니다.

graph_router.py
POST /api/graph
LangGraph를 사용한 대화형 응답 반환.
"""

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

        # 마지막 AIMessage만 찾아서 응답 추출
        import re

        answer = ""

        # messages에서 마지막 AIMessage 찾기
        for msg in reversed(out["messages"]):
            if isinstance(msg, AIMessage):
                answer = msg.content if hasattr(msg, "content") else str(msg)
                break

        # 응답이 없으면 마지막 메시지 사용
        if not answer:
            answer = (
                out["messages"][-1].content
                if hasattr(out["messages"][-1], "content")
                else str(out["messages"][-1])
            )

        # 응답에서 이전 대화 내용 제거 (태그 및 중복 방지)
        if answer:
            # 1. [system], [assistant], [user], [endofturn] 태그 제거
            answer = re.sub(r"\[system\].*?\[endofturn\]", "", answer, flags=re.DOTALL)
            answer = re.sub(
                r"\[assistant\](.*?)\[endofturn\]", r"\1", answer, flags=re.DOTALL
            )
            answer = re.sub(r"\[user\].*?\[endofturn\]", "", answer, flags=re.DOTALL)
            answer = re.sub(r"\[endofturn\]", "", answer)

            # 2. 이전 대화 패턴 제거 (Human:, Assistant:, AI:, 사용자:, 어시스턴트:)
            if any(
                marker in answer
                for marker in ["Human:", "Assistant:", "AI:", "사용자:", "어시스턴트:"]
            ):
                # 마지막 Assistant/AI/어시스턴트 응답만 추출
                patterns = [
                    r"Assistant:\s*(.+?)(?:\nHuman:|$)",
                    r"AI:\s*(.+?)(?:\nHuman:|$)",
                    r"어시스턴트:\s*(.+?)(?:\n사용자:|$)",
                    r"AI:\s*(.+?)(?:\n사용자:|$)",
                ]
                for pattern in patterns:
                    match = re.search(pattern, answer, re.DOTALL)
                    if match:
                        answer = match.group(1).strip()
                        break

            # 3. 중복된 대화 내용 제거 (같은 문장이 여러 번 반복되는 경우)
            # 마지막 실제 응답만 추출 (이전 대화와 겹치지 않는 부분)
            lines = answer.split("\n")
            cleaned_lines = []
            seen_content = set()

            # 뒤에서부터 읽어서 중복되지 않은 첫 번째 의미있는 응답 찾기
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                # 태그나 시스템 메시지가 아닌 실제 응답인지 확인
                if not any(
                    tag in line
                    for tag in [
                        "[system]",
                        "[assistant]",
                        "[user]",
                        "[endofturn]",
                        "Human:",
                        "Assistant:",
                    ]
                ):
                    # 중복 체크 (간단한 해시 사용)
                    line_hash = hash(line.lower())
                    if line_hash not in seen_content:
                        seen_content.add(line_hash)
                        cleaned_lines.insert(0, line)

            # cleaned_lines가 있으면 사용, 없으면 원본 정리된 버전 사용
            if cleaned_lines:
                answer = " ".join(cleaned_lines).strip()
            else:
                # 태그만 제거한 버전 사용
                answer = re.sub(r"\[.*?\]", "", answer).strip()

            # 4. 앞뒤 공백 및 불필요한 줄바꿈 제거
            answer = re.sub(r"\n+", " ", answer).strip()
            answer = re.sub(r"\s+", " ", answer)

        # 빈 응답 방지
        if not answer or not answer.strip():
            answer = "답변을 생성할 수 없습니다."

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
