"""
채팅 도메인 오케스트레이터 (v1/chat hub).

라우터 → 정책 판단 → agents/services 라우팅
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional, Sequence

from app.common.orchestrator.base_orchestrator import BaseOrchestrator


@dataclass(frozen=True)
class ChatRouteResult:
    """라우터 계층이 그대로 반환할 수 있는 결과 타입."""

    mode: str  # "stream" | "text"
    # mode == "text"
    text: Optional[str] = None
    # mode == "stream"
    stream: Optional[AsyncIterator[str]] = None


class ChatOrchestrator(BaseOrchestrator):
    """채팅 도메인 오케스트레이터.

    역할:
    1. 라우터로부터 요청 수신
    2. 정책 판단 (LLM 기반 vs 규칙 기반)
    3. 결정에 따라 agents/ 또는 services/ 폴더 기능 호출

    플로우:
    - 정책 관련 (복잡한 대화, RAG 필요) → agents 폴더 (LLM 기반)
    - 규칙 기반 (간단한 응답, 템플릿) → services 폴더 (비즈니스 로직)
    """

    def _get_default_thresholds(self) -> Dict[str, float]:
        """기본 임계치를 반환합니다."""
        return {
            "complexity_threshold": 0.5,  # 복잡도가 이 값 이상이면 agents 사용
        }

    def classify_domain(
        self, message: str, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """메시지를 분석하여 정책을 결정합니다."""

        complexity = self._calculate_complexity(message)
        threshold = self.thresholds.get("complexity_threshold", 0.5)

        # 정책 결정
        if complexity > threshold:
            policy = "USE_LLM"
            use_agents = True
        else:
            policy = "USE_RULES"
            use_agents = False

        # 신뢰도
        if abs(complexity - 0.5) < 0.2:
            confidence = "medium"
        else:
            confidence = "high"

        gate_result: Dict[str, Any] = {
            "domain": "chat",
            "policy": policy,
            "confidence": confidence,
            "complexity": complexity,
            "use_agents": use_agents,
        }

        if request_id:
            gate_result["request_id"] = request_id
            self._cache_result(request_id, gate_result)

        print(f"[CHAT_ORCHESTRATOR] 정책: {policy}, 복잡도: {complexity:.2f}")
        print(
            f"[CHAT_ORCHESTRATOR] 사용 폴더: {'agents' if use_agents else 'services'}"
        )

        return gate_result

    def _calculate_complexity(self, message: str) -> float:
        """메시지 복잡도를 계산합니다."""
        complexity = 0.3  # 기본값

        # 길이에 따른 복잡도
        if len(message) > 100:
            complexity += 0.2
        if len(message) > 200:
            complexity += 0.2

        # 질문 형태
        if "?" in message or "뭐" in message or "어떻게" in message or "왜" in message:
            complexity += 0.3

        # 특정 키워드 (검색, 분석 등)
        keywords = ["검색", "찾아", "분석", "설명", "알려줘", "도와줘"]
        for keyword in keywords:
            if keyword in message:
                complexity += 0.1
                break

        return min(complexity, 1.0)

    def route_chat(
        self,
        *,
        message: str,
        history: Optional[Sequence[dict]] = None,
        client_host: Optional[str] = None,
        chat_service: Any = None,
    ) -> ChatRouteResult:
        """라우터에서 쓰는 '실제 라우팅' 엔트리포인트.

        정책(휴리스틱) 기반으로 **무조건** 라우팅합니다.
        - 복잡도 낮음: 규칙 기반(간단 템플릿)
        - 복잡도 높음: LLM 기반
          - localhost: 로컬 EXAONE(LangGraph)
          - 그 외: OpenAI RAG 스트리밍
        """
        history = list(history or [])
        is_localhost = bool(
            client_host
            and (
                client_host == "127.0.0.1"
                or client_host == "localhost"
                or client_host == "::1"
                or client_host.startswith("127.")
            )
        )

        gate_result = self.classify_domain(message)
        use_agents = bool(gate_result.get("use_agents"))

        # 규칙 기반(간단 응답)
        if not use_agents:
            return ChatRouteResult(
                mode="text",
                text="규칙 기반 응답 - 간단한 인사말이나 템플릿 응답",
            )

        # LLM 기반
        if is_localhost:
            from app.domains.v1.chat.spokes.agents.graph import run_once_with_history

            text = run_once_with_history(user_message=message, history=history)
            return ChatRouteResult(mode="text", text=text)

        # 원격(비-localhost): OpenAI RAG 스트리밍
        if chat_service is None:
            raise RuntimeError(
                "ChatService가 주입되지 않았습니다. 라우터에서 chat_service를 전달해야 합니다."
            )

        async def _stream_openai() -> AsyncIterator[str]:
            # 적절한 RAG 체인 선택
            if not getattr(chat_service, "openai_rag_chain", None):
                if getattr(chat_service, "openai_quota_exceeded", False):
                    yield "OpenAI API 할당량이 초과되었습니다."
                    return
                yield "OpenAI RAG 체인이 초기화되지 않았습니다."
                return

            current_rag_chain = chat_service.openai_rag_chain

            # 대화 기록을 LangChain 메시지 형식으로 변환
            from langchain_core.messages import AIMessage, HumanMessage

            chat_history = []
            for msg in history:
                if msg.get("role") == "user":
                    chat_history.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    chat_history.append(AIMessage(content=msg.get("content", "")))

            accumulated_text = ""
            async for chunk in current_rag_chain.astream(
                {
                    "input": message,
                    "chat_history": chat_history,
                }
            ):
                # chunk에서 answer 추출
                if isinstance(chunk, dict):
                    answer = chunk.get("answer", "")
                    if not answer:
                        continue
                    if len(answer) > len(accumulated_text):
                        delta = answer[len(accumulated_text) :]
                        accumulated_text = answer
                        for char in delta:
                            yield char
                elif isinstance(chunk, str):
                    if len(chunk) > len(accumulated_text):
                        delta = chunk[len(accumulated_text) :]
                        accumulated_text = chunk
                        for char in delta:
                            yield char

        return ChatRouteResult(mode="stream", stream=_stream_openai())

    def analyze(self, message: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """메시지를 분석하고 적절한 응답을 생성합니다."""
        # 1. 정책 판단
        gate_result = self.classify_domain(message, request_id=request_id)
        policy = gate_result.get("policy", "USE_RULES")
        use_agents = gate_result.get("use_agents", False)

        agent_result = None
        service_result = None
        final_response = ""

        # 2. 정책에 따라 라우팅
        if use_agents and policy == "USE_LLM":
            print("[CHAT_ORCHESTRATOR] → agents 폴더로 라우팅 (LLM 기반)")
            try:
                # NOTE: 실제 라우팅/실행은 route_chat()에서 수행합니다.
                agent_result = {"policy": policy}
                final_response = "LLM 기반 응답"

            except Exception as e:
                print(f"[ERROR] agents 폴더 처리 실패: {e}")
                final_response = f"[ERROR] LLM 처리 실패: {str(e)}"

        else:
            print("[CHAT_ORCHESTRATOR] → services 폴더로 라우팅 (규칙 기반)")
            # TODO: services 폴더 기능 구현
            service_result = {
                "response": "규칙 기반 응답 - 간단한 인사말이나 템플릿 응답",
                "policy": policy,
            }
            final_response = service_result["response"]

        return {
            "gate_result": gate_result,
            "agent_result": agent_result,
            "service_result": service_result,
            "final_response": final_response,
        }


__all__ = ["ChatOrchestrator"]

