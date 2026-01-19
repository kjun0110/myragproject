"""
FastAPI 기준의 API 엔드포인트 계층입니다.

graph_router.py
POST /api/graph
LangGraph를 사용한 대화형 응답 반환.
"""

# 스키마 import
from app.domains.chat.models.base_model import GraphRequest, GraphResponse
from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

router = APIRouter(prefix="/api", tags=["graph"])


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

        # Graph 실행
        state = {"messages": messages}
        out = graph.invoke(state)

        # 마지막 AIMessage만 찾아서 응답 추출
        import re

        answer = ""

        # messages에서 마지막 AIMessage 찾기 (현재 요청 이후의 AIMessage만)
        # 현재 요청 메시지는 messages의 마지막 HumanMessage
        current_request_idx = len(messages) - 1

        # 현재 요청 이후의 AIMessage만 찾기
        for i, msg in enumerate(out["messages"]):
            if i > current_request_idx and isinstance(msg, AIMessage):
                answer = msg.content if hasattr(msg, "content") else str(msg)
                break

        # 위에서 찾지 못했으면 뒤에서부터 찾기
        if not answer:
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

        # 응답에서 이전 대화 내용 제거
        if answer:
            # 히스토리에 있는 모든 메시지 텍스트 수집 (현재 요청 제외)
            history_texts = []
            for msg in messages[:-1]:  # 현재 요청 전의 메시지만
                if hasattr(msg, "content"):
                    content = msg.content
                    if content and content.strip():
                        history_texts.append(content.strip())

            # 응답을 줄 단위로 나누고 히스토리에 없는 줄만 남기기
            answer_lines = answer.split("\n")
            filtered_lines = []

            for line in answer_lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue

                # 이 줄이 히스토리 메시지와 유사한지 확인
                is_history_line = False
                for hist_text in history_texts:
                    # 히스토리 텍스트가 이 줄에 포함되어 있으면 제외
                    if hist_text in line_stripped or line_stripped in hist_text:
                        is_history_line = True
                        break
                    # 부분 일치도 확인 (긴 문장의 경우)
                    if len(hist_text) > 20 and len(line_stripped) > 20:
                        # 공통 단어가 많으면 히스토리로 간주
                        hist_words = set(hist_text.split())
                        line_words = set(line_stripped.split())
                        common_ratio = len(hist_words & line_words) / max(
                            len(hist_words), len(line_words), 1
                        )
                        if common_ratio > 0.7:  # 70% 이상 일치하면 히스토리로 간주
                            is_history_line = True
                            break

                if not is_history_line:
                    filtered_lines.append(line)

            # 필터링된 줄이 있으면 사용, 없으면 원본 사용
            if filtered_lines:
                answer = "\n".join(filtered_lines).strip()
            else:
                # 필터링으로 모든 줄이 제거되면 원본 사용 (히스토리 제거만 시도)
                for hist_text in history_texts:
                    if hist_text in answer:
                        # 히스토리 텍스트를 정확히 찾아서 제거
                        hist_text_escaped = re.escape(hist_text)
                        answer = re.sub(hist_text_escaped, "", answer, flags=re.DOTALL)

            # 응답에서 이전 대화 내용 제거 (태그 및 중복 방지)
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
