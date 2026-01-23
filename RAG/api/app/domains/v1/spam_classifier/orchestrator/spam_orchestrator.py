"""
스팸 분류 도메인 오케스트레이터.

라우터 → KoELECTRA 판단 → agents/services 라우팅
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
orchestrator_dir = current_file.parent  # api/app/domains/spam_classifier/orchestrator/
spam_classifier_dir = orchestrator_dir.parent  # api/app/domains/spam_classifier/
domains_dir = spam_classifier_dir.parent  # api/app/domains/
app_dir = domains_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

sys.path.insert(0, str(api_dir))

from app.common.orchestrator.base_orchestrator import BaseOrchestrator

from .koelectra_loader import load_koelectra_gate


class SpamClassifierOrchestrator(BaseOrchestrator):
    """스팸 분류 도메인 오케스트레이터.

    역할:
    1. 라우터로부터 요청 수신
    2. KoELECTRA로 정책 결정 (정책 관련 vs 규칙 기반)
    3. 결정에 따라 agents/ 또는 services/ 폴더 기능 호출

    플로우:
    - 정책 관련 (ANALYZE_SPAM) → agents/ 폴더 (EXAONE 등 AI 기반)
    - 규칙 기반 (BYPASS 등) → services/ 폴더 (비즈니스 로직)
    """

    def _get_default_thresholds(self) -> Dict[str, float]:
        """기본 임계치를 반환합니다."""
        return {
            "policy_threshold": 0.3,  # 스팸 확률이 이 값 이상이면 ANALYZE_SPAM
        }

    def classify_domain(
        self, email_text: str, max_length: int = 512, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """KoELECTRA로 정책 결정을 수행합니다.

        Args:
            email_text: 분석할 이메일 텍스트
            max_length: 최대 토큰 길이
            request_id: 요청 ID (선택사항)

        Returns:
            {
                "domain": str,  # "spam" 또는 "other"
                "policy": str,  # "ANALYZE_SPAM" 또는 "BYPASS"
                "confidence": str,  # "low", "medium", "high"
                "spam_prob": float,
                "ham_prob": float,
                "latency_ms": float,
                "use_agents": bool,  # True: agents 사용, False: services 사용
            }
        """
        start_time = time.time()

        try:
            model, tokenizer = load_koelectra_gate()
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # 토크나이징
            inputs = tokenizer(
                email_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 예측
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)

            spam_prob = probabilities[0][1].item()
            ham_prob = probabilities[0][0].item()

            # 정책 결정
            threshold = self.thresholds.get("policy_threshold", 0.3)
            policy = "ANALYZE_SPAM" if spam_prob > threshold else "BYPASS"
            domain = "spam" if policy == "ANALYZE_SPAM" else "other"

            # 신뢰도 계산
            if abs(spam_prob - 0.5) < 0.1:
                confidence = "low"
            elif abs(spam_prob - 0.5) < 0.3:
                confidence = "medium"
            else:
                confidence = "high"

            # agents vs services 결정
            use_agents = self.should_use_agents(policy, spam_prob)

            latency_ms = (time.time() - start_time) * 1000

            gate_result = {
                "domain": domain,
                "policy": policy,
                "confidence": confidence,
                "spam_prob": float(spam_prob),
                "ham_prob": float(ham_prob),
                "latency_ms": round(latency_ms, 2),
                "use_agents": use_agents,  # agents vs services 플래그
            }

            if request_id:
                gate_result["request_id"] = request_id
                self._cache_result(request_id, gate_result)

            print(f"[SPAM_ORCHESTRATOR] 정책: {policy}, 도메인: {domain}")
            print(
                f"[SPAM_ORCHESTRATOR] 사용 폴더: {'agents' if use_agents else 'services'}"
            )

            return gate_result

        except Exception as e:
            print(f"[ERROR] 스팸 오케스트레이터 처리 실패: {e}")
            raise

    def analyze(
        self, email_text: str, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """이메일 텍스트를 분석합니다.

        플로우:
        1. KoELECTRA로 정책 결정
        2. 정책에 따라 agents 또는 services 호출

        Args:
            email_text: 분석할 이메일 텍스트
            request_id: 요청 ID (선택사항)

        Returns:
            {
                "gate_result": dict,  # KoELECTRA 결과
                "agent_result": Optional[dict],  # agents 폴더 결과
                "service_result": Optional[dict],  # services 폴더 결과
                "final_decision": str  # 최종 결정
            }
        """
        # 1. KoELECTRA 정책 결정
        gate_result = self.classify_domain(email_text, request_id=request_id)
        policy = gate_result.get("policy", "BYPASS")
        use_agents = gate_result.get("use_agents", False)

        agent_result = None
        service_result = None
        final_decision = ""

        # 2. 정책에 따라 라우팅
        if use_agents and policy == "ANALYZE_SPAM":
            # agents 폴더 사용 (정책 관련 - AI 기반)
            print("[SPAM_ORCHESTRATOR] → agents 폴더로 라우팅 (정책 관련)")
            try:
                from ..agents import get_exaone_tool

                exaone_tool = get_exaone_tool()
                agent_result_str = exaone_tool.invoke({"email_text": email_text})

                # JSON 파싱 시도
                import json

                try:
                    agent_result = json.loads(agent_result_str)
                except (json.JSONDecodeError, TypeError):
                    agent_result = {"analysis": agent_result_str}

                # 최종 결정 생성
                if isinstance(agent_result, dict):
                    spam_prob = agent_result.get("spam_prob", 0.5)
                    label = agent_result.get("label", "unknown")
                    confidence = agent_result.get("confidence", "medium")
                    analysis = agent_result.get("analysis", "")

                    final_decision = f"""[AGENTS 폴더] EXAONE 스팸 분석:
- 스팸 확률: {spam_prob:.4f}
- 레이블: {label}
- 신뢰도: {confidence}
- 상세 분석:
{analysis}"""
                else:
                    final_decision = f"[AGENTS 폴더] EXAONE 스팸 분석:\n{agent_result}"

            except Exception as e:
                print(f"[ERROR] agents 폴더 처리 실패: {e}")
                final_decision = f"[ERROR] agents 처리 실패: {str(e)}"

        else:
            # services 폴더 사용 (규칙 기반)
            print("[SPAM_ORCHESTRATOR] → services 폴더로 라우팅 (규칙 기반)")

            # TODO: services 폴더 기능 구현
            # 현재는 간단한 규칙 기반 응답 반환
            service_result = {
                "message": "규칙 기반 처리 - 현재 구현되지 않은 서비스입니다.",
                "policy": policy,
                "domain": gate_result["domain"],
            }

            final_decision = f"""[SERVICES 폴더] 규칙 기반 처리:
- 정책: {policy}
- 도메인: {gate_result["domain"]}
- 메시지: {service_result["message"]}"""

        return {
            "gate_result": gate_result,
            "agent_result": agent_result,
            "service_result": service_result,
            "final_decision": final_decision,
        }
