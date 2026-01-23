"""
채팅 도메인 오케스트레이터.

라우터 → 정책 판단 → agents/services 라우팅
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
orchestrator_dir = current_file.parent  # api/app/domains/chat/orchestrator/
chat_dir = orchestrator_dir.parent  # api/app/domains/chat/
domains_dir = chat_dir.parent  # api/app/domains/
app_dir = domains_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

sys.path.insert(0, str(api_dir))

from app.common.orchestrator.base_orchestrator import BaseOrchestrator


class ChatOrchestrator(BaseOrchestrator):
    """채팅 도메인 오케스트레이터.

    역할:
    1. 라우터로부터 요청 수신
    2. 정책 판단 (LLM 기반 vs 규칙 기반)
    3. 결정에 따라 agents/ 또는 services/ 폴더 기능 호출

    플로우:
    - 정책 관련 (복잡한 대화, RAG 필요) → agents/ 폴더 (LLM 기반)
    - 규칙 기반 (간단한 응답, 템플릿) → services/ 폴더 (비즈니스 로직)
    """

    def _get_default_thresholds(self) -> Dict[str, float]:
        """기본 임계치를 반환합니다."""
        return {
            "complexity_threshold": 0.5,  # 복잡도가 이 값 이상이면 agents 사용
        }

    def classify_domain(
        self, message: str, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """메시지를 분석하여 정책을 결정합니다.

        Args:
            message: 분석할 메시지
            request_id: 요청 ID (선택사항)

        Returns:
            {
                "domain": str,  # "chat"
                "policy": str,  # "USE_LLM" 또는 "USE_RULES"
                "confidence": str,  # "low", "medium", "high"
                "complexity": float,  # 메시지 복잡도
                "use_agents": bool,  # True: agents 사용, False: services 사용
            }
        """
        # 간단한 규칙 기반 복잡도 판단
        # TODO: 향후 더 정교한 분류 모델 추가 가능

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

        gate_result = {
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
        """메시지 복잡도를 계산합니다.

        간단한 휴리스틱:
        - 짧은 메시지 → 낮은 복잡도
        - 질문이 있으면 → 높은 복잡도
        - 특정 키워드 포함 → 높은 복잡도

        Args:
            message: 메시지 텍스트

        Returns:
            복잡도 (0.0 ~ 1.0)
        """
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

    def analyze(self, message: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """메시지를 분석하고 적절한 응답을 생성합니다.

        플로우:
        1. 정책 판단
        2. 정책에 따라 agents 또는 services 호출

        Args:
            message: 분석할 메시지
            request_id: 요청 ID (선택사항)

        Returns:
            {
                "gate_result": dict,  # 정책 결정 결과
                "agent_result": Optional[dict],  # agents 폴더 결과
                "service_result": Optional[dict],  # services 폴더 결과
                "final_response": str  # 최종 응답
            }
        """
        # 1. 정책 판단
        gate_result = self.classify_domain(message, request_id=request_id)
        policy = gate_result.get("policy", "USE_RULES")
        use_agents = gate_result.get("use_agents", False)

        agent_result = None
        service_result = None
        final_response = ""

        # 2. 정책에 따라 라우팅
        if use_agents and policy == "USE_LLM":
            # agents 폴더 사용 (LLM 기반)
            print("[CHAT_ORCHESTRATOR] → agents 폴더로 라우팅 (LLM 기반)")
            try:
                # TODO: agents 폴더의 LLM 기능 호출
                # 예: ChatService 또는 graph 사용
                agent_result = {
                    "response": "LLM 기반 응답 (구현 예정)",
                    "policy": policy,
                }
                final_response = agent_result["response"]

            except Exception as e:
                print(f"[ERROR] agents 폴더 처리 실패: {e}")
                final_response = f"[ERROR] LLM 처리 실패: {str(e)}"

        else:
            # services 폴더 사용 (규칙 기반)
            print("[CHAT_ORCHESTRATOR] → services 폴더로 라우팅 (규칙 기반)")

            # TODO: services 폴더 기능 구현
            # 현재는 간단한 규칙 기반 응답 반환
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
