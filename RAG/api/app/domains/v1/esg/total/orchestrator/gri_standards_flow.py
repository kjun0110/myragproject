"""GRI Standards 오케스트레이터.

라우터 → KoELECTRA 판단 → agents/services 라우팅
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
orchestrator_dir = current_file.parent  # api/app/domains/esg/total/orchestrator/
total_dir = orchestrator_dir.parent  # api/app/domains/esg/total/
esg_dir = total_dir.parent  # api/app/domains/esg/
domains_dir = esg_dir.parent  # api/app/domains/
app_dir = domains_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

sys.path.insert(0, str(api_dir))

from app.common.orchestrator.base_orchestrator import BaseOrchestrator

# 전역 캐싱
_gri_orchestrator = None

# KoELECTRA 모델 전역 캐싱
_koelectra_model = None
_koelectra_tokenizer = None
_koelectra_loading = False
_koelectra_error = None


def load_koelectra_gate() -> tuple:
    """KoELECTRA 게이트웨이 모델을 로드합니다.

    Returns:
        (model, tokenizer) 튜플

    Raises:
        RuntimeError: 모델 로드 실패 시
    """
    global _koelectra_model, _koelectra_tokenizer, _koelectra_loading, _koelectra_error

    if _koelectra_model is not None and _koelectra_tokenizer is not None:
        return _koelectra_model, _koelectra_tokenizer

    if _koelectra_error is not None:
        raise RuntimeError(f"이전 KoELECTRA 모델 로드 실패: {_koelectra_error}")

    if _koelectra_loading:
        raise RuntimeError("KoELECTRA 모델이 현재 로딩 중입니다.")

    _koelectra_loading = True
    try:
        print("[INFO] KoELECTRA 게이트웨이 모델 로딩 시작...")

        # 공통 모델 로더 사용 (HuggingFace 캐시 + 로컬 아답터)
        from app.common.loaders import load_koelectra_with_spam_adapter

        _koelectra_model, _koelectra_tokenizer = load_koelectra_with_spam_adapter()

        print("[OK] KoELECTRA 게이트웨이 모델 로드 완료")
        _koelectra_loading = False
        _koelectra_error = None
        return _koelectra_model, _koelectra_tokenizer

    except Exception as e:
        _koelectra_loading = False
        error_msg = f"KoELECTRA 모델 로드 실패: {str(e)}"
        _koelectra_error = error_msg
        print(f"[ERROR] {error_msg}")
        import traceback

        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
        raise RuntimeError(error_msg) from e


class GRIStandardsOrchestrator(BaseOrchestrator):
    """GRI Standards 도메인 오케스트레이터.

    역할:
    1. 라우터로부터 요청 수신
    2. KoELECTRA로 정책 결정 (정책 관련 vs 규칙 기반)
    3. 결정에 따라 agents/ 또는 services/ 폴더 기능 호출

    플로우:
    - 정책 관련 (ANALYZE_GRI) → agents/ 폴더 (LLM 기반)
    - 규칙 기반 (BYPASS 등) → services/ 폴더 (비즈니스 로직)
    """

    def _get_default_thresholds(self) -> Dict[str, float]:
        """기본 임계치를 반환합니다."""
        return {
            "policy_threshold": 0.3,  # 정책 관련 확률이 이 값 이상이면 ANALYZE_GRI
        }

    def classify_domain(
        self, text: str, max_length: int = 512, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """KoELECTRA로 정책 결정을 수행합니다.

        Args:
            text: 분석할 텍스트
            max_length: 최대 토큰 길이
            request_id: 요청 ID (선택사항)

        Returns:
            {
                "domain": str,  # "gri" 또는 "other"
                "policy": str,  # "ANALYZE_GRI" 또는 "BYPASS"
                "confidence": str,  # "low", "medium", "high"
                "gri_prob": float,
                "other_prob": float,
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
                text,
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

            gri_prob = probabilities[0][1].item()  # GRI 관련 확률
            other_prob = probabilities[0][0].item()  # 기타 확률

            # 정책 결정
            threshold = self.thresholds.get("policy_threshold", 0.3)
            policy = "ANALYZE_GRI" if gri_prob > threshold else "BYPASS"
            domain = "gri" if policy == "ANALYZE_GRI" else "other"

            # 신뢰도 계산
            if abs(gri_prob - 0.5) < 0.1:
                confidence = "low"
            elif abs(gri_prob - 0.5) < 0.3:
                confidence = "medium"
            else:
                confidence = "high"

            # agents vs services 결정
            use_agents = self.should_use_agents(policy, gri_prob)

            latency_ms = (time.time() - start_time) * 1000

            gate_result = {
                "domain": domain,
                "policy": policy,
                "confidence": confidence,
                "gri_prob": float(gri_prob),
                "other_prob": float(other_prob),
                "latency_ms": round(latency_ms, 2),
                "use_agents": use_agents,  # agents vs services 플래그
            }

            if request_id:
                gate_result["request_id"] = request_id
                self._cache_result(request_id, gate_result)

            print(f"[GRI_ORCHESTRATOR] 정책: {policy}, 도메인: {domain}")
            print(
                f"[GRI_ORCHESTRATOR] 사용 폴더: {'agents' if use_agents else 'services'}"
            )

            return gate_result

        except Exception as e:
            print(f"[ERROR] GRI 오케스트레이터 처리 실패: {e}")
            raise

    def analyze(
        self, text: str, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """텍스트를 분석합니다.

        플로우:
        1. KoELECTRA로 정책 결정
        2. 정책에 따라 agents 또는 services 호출

        Args:
            text: 분석할 텍스트
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
        gate_result = self.classify_domain(text, request_id=request_id)
        policy = gate_result.get("policy", "BYPASS")
        use_agents = gate_result.get("use_agents", False)

        agent_result = None
        service_result = None
        final_decision = ""

        # 2. 정책에 따라 라우팅
        if use_agents and policy == "ANALYZE_GRI":
            # agents 폴더 사용 (정책 관련 - LLM 기반)
            print("[GRI_ORCHESTRATOR] → agents 폴더로 라우팅 (정책 관련)")
            try:
                from ..agents.gri_standards_agent import GRIStandardsAgent

                agent = GRIStandardsAgent()
                agent_result = agent.analyze(text)

                final_decision = f"""[AGENTS 폴더] LLM 기반 GRI 분석:
{agent_result.get('response', '분석 완료')}"""

            except Exception as e:
                print(f"[ERROR] agents 폴더 처리 실패: {e}")
                final_decision = f"[ERROR] agents 처리 실패: {str(e)}"

        else:
            # services 폴더 사용 (규칙 기반)
            print("[GRI_ORCHESTRATOR] → services 폴더로 라우팅 (규칙 기반)")

            try:
                from ..services.gri_standards_service import GRIStandardsService

                service = GRIStandardsService()
                service_result = service.process(text)

                final_decision = f"""[SERVICES 폴더] 규칙 기반 처리:
{service_result.get('message', '처리 완료')}"""

            except Exception as e:
                print(f"[ERROR] services 폴더 처리 실패: {e}")
                final_decision = f"[ERROR] services 처리 실패: {str(e)}"

        return {
            "gate_result": gate_result,
            "agent_result": agent_result,
            "service_result": service_result,
            "final_decision": final_decision,
        }


def get_gri_standards_orchestrator() -> GRIStandardsOrchestrator:
    """GRI Standards 오케스트레이터를 가져옵니다."""
    global _gri_orchestrator
    if _gri_orchestrator is None:
        _gri_orchestrator = GRIStandardsOrchestrator()
    return _gri_orchestrator
