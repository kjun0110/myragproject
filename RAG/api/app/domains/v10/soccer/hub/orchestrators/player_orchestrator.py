"""
Player Orchestrator - LangGraph 기반 정책/규칙 라우팅

KoELECTRA를 사용하여 데이터가 정책 기반인지 규칙 기반인지 판단하고,
LangGraph StateGraph를 통해 적절한 경로로 처리합니다.
"""

import json
import logging
from typing import Any, Dict, List, Literal

import torch
from langgraph.graph import END, START, StateGraph
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.common.loaders import ModelLoader
from app.domains.v10.soccer.models.states.player_state import PlayerProcessingState

logger = logging.getLogger(__name__)


class PlayerOrchestrator:
    """Player 데이터 처리 오케스트레이터 (LangGraph 기반).
    
    KoELECTRA를 사용하여 데이터가 정책 기반인지 규칙 기반인지 판단하고,
    LangGraph StateGraph를 통해 적절한 경로로 처리합니다.
    """
    
    def __init__(self):
        # KoELECTRA 모델 로드 (판단용으로 유지)
        self.model = None
        self.tokenizer = None
        self.device = None
        self._load_model()
        
        # LangGraph 빌드
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    def _load_model(self):
        """KoELECTRA 모델을 로드합니다."""
        try:
            logger.info("[ORCHESTRATOR] KoELECTRA 모델 로드 시작...")
            
            # 디바이스 설정
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"[ORCHESTRATOR] 디바이스: {self.device}")
            
            # 모델 로드 (분류를 위한 모델 - 정책/규칙 판단용)
            # num_labels=2: 정책 기반(0) vs 규칙 기반(1)
            self.model, self.tokenizer = ModelLoader.load_koelectra_model(
                adapter_name=None,  # 베이스 모델 사용
                device=self.device,
                num_labels=2,  # 정책/규칙 이진 분류
            )
            
            self.model.eval()  # 평가 모드
            logger.info("[ORCHESTRATOR] KoELECTRA 모델 로드 완료")
            
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] KoELECTRA 모델 로드 실패: {e}", exc_info=True)
            raise
    
    def _build_graph(self) -> StateGraph:
        """LangGraph StateGraph를 빌드합니다."""
        graph = StateGraph(PlayerProcessingState)
        
        # 노드 추가
        graph.add_node("validate", self._validate_node)
        graph.add_node("determine_strategy", self._determine_strategy_node)
        graph.add_node("policy_process", self._policy_process_node)
        graph.add_node("rule_process", self._rule_process_node)
        graph.add_node("finalize", self._finalize_node)
        
        # 엣지 추가
        graph.add_edge(START, "validate")
        graph.add_edge("validate", "determine_strategy")
        graph.add_conditional_edges(
            "determine_strategy",
            self._route_strategy,
            {
                "policy": "policy_process",
                "rule": "rule_process",
            }
        )
        graph.add_edge("policy_process", "finalize")
        graph.add_edge("rule_process", "finalize")
        graph.add_edge("finalize", END)
        
        return graph
    
    async def _validate_node(self, state: PlayerProcessingState) -> Dict[str, Any]:
        """데이터 검증 노드."""
        logger.info(f"[ORCHESTRATOR] 검증 노드 시작: {len(state['records'])}개 레코드")
        
        # 기본 검증 (빈 리스트 체크)
        if not state['records']:
            return {
                "current_step": "validate",
                "validation_errors": [{"error": "레코드가 없습니다."}],
                "errors": [{"step": "validate", "error": "레코드가 없습니다."}],
            }
        
        # 첫 5개 레코드 로그 출력
        first_five_records = state['records'][:5]
        logger.info(f"[ORCHESTRATOR] 오케스트레이터에 도달한 데이터 상위 5개 레코드:")
        for idx, record in enumerate(first_five_records, 1):
            logger.info(f"[ORCHESTRATOR] 레코드 {idx}: {json.dumps(record, ensure_ascii=False, indent=2)}")
        
        return {
            "current_step": "validate",
            "validated_records": state['records'],  # 실제 검증은 Service에서 수행
            "validation_errors": [],
        }
    
    async def _determine_strategy_node(self, state: PlayerProcessingState) -> Dict[str, Any]:
        """전략 판단 노드."""
        logger.info("[ORCHESTRATOR] 전략 판단 노드 시작")
        
        records = state.get('validated_records', state['records'])
        
        # 휴리스틱 판단
        heuristic_result = self._determine_strategy_heuristic(records)
        
        # KoELECTRA 판단 (참고용)
        koelectra_result = None
        confidence = None
        try:
            koelectra_result, confidence = self._determine_strategy_koelectra_with_confidence(records)
            logger.info(
                f"[ORCHESTRATOR] 판단 비교 - 휴리스틱: {heuristic_result}, "
                f"KoELECTRA: {koelectra_result}"
            )
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] KoELECTRA 판단 스킵: {e}")
        
        # 최종 결정: 휴리스틱 결과 사용
        final_strategy = heuristic_result
        logger.info(
            f"[ORCHESTRATOR] 최종 전략 결정: {final_strategy} "
            f"(휴리스틱 기반, JSONL → players 테이블 저장)"
        )
        
        return {
            "current_step": "determine_strategy",
            "strategy_type": final_strategy,
            "strategy_confidence": confidence,
            "heuristic_result": heuristic_result,
            "koelectra_result": koelectra_result,
        }
    
    def _route_strategy(self, state: PlayerProcessingState) -> Literal["policy", "rule"]:
        """전략 라우팅 함수."""
        strategy = state.get("strategy_type", "rule")
        logger.info(f"[ORCHESTRATOR] 전략 라우팅: {strategy}")
        return strategy  # type: ignore
    
    async def _policy_process_node(self, state: PlayerProcessingState) -> Dict[str, Any]:
        """정책 기반 처리 노드 (Agent 호출)."""
        logger.info("[ORCHESTRATOR] 정책 기반 처리 노드 시작")
        
        try:
            from app.domains.v10.soccer.spokes.agents.player_agent import PlayerAgent
            
            agent = PlayerAgent()
            records = state.get('validated_records', state['records'])
            result = await agent.process(records)
            
            logger.info("[ORCHESTRATOR] 정책 기반 처리 완료")
            
            return {
                "current_step": "policy_process",
                "result": result,
                "processed_count": result.get("processed_records", len(records)),
            }
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] 정책 기반 처리 실패: {e}", exc_info=True)
            return {
                "current_step": "policy_process",
                "errors": state.get("errors", []) + [{"step": "policy_process", "error": str(e)}],
            }
    
    async def _rule_process_node(self, state: PlayerProcessingState) -> Dict[str, Any]:
        """규칙 기반 처리 노드 (Service 호출)."""
        logger.info("[ORCHESTRATOR] 규칙 기반 처리 노드 시작")
        
        try:
            from app.domains.v10.soccer.spokes.services.player_service import PlayerService
            
            service = PlayerService()
            records = state.get('validated_records', state['records'])
            result = await service.process(records)
            
            logger.info("[ORCHESTRATOR] 규칙 기반 처리 완료")
            
            return {
                "current_step": "rule_process",
                "result": result,
                "processed_count": result.get("saved_records", len(records)),
            }
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] 규칙 기반 처리 실패: {e}", exc_info=True)
            return {
                "current_step": "rule_process",
                "errors": state.get("errors", []) + [{"step": "rule_process", "error": str(e)}],
            }
    
    async def _finalize_node(self, state: PlayerProcessingState) -> Dict[str, Any]:
        """최종 정리 노드."""
        logger.info("[ORCHESTRATOR] 최종 정리 노드 시작")
        
        result = state.get("result", {})
        if not result:
            result = {
                "success": False,
                "message": "처리 실패",
                "errors": state.get("errors", []),
            }
        
        logger.info("[ORCHESTRATOR] Player 처리 완료")
        
        return {
            "current_step": "finalize",
            "result": result,
        }
    
    def _determine_strategy_heuristic(self, records: List[Dict[str, Any]]) -> str:
        """휴리스틱을 사용하여 정책 기반인지 규칙 기반인지 판단합니다."""
        try:
            if not records:
                logger.warning("[ORCHESTRATOR] 레코드가 없어 규칙 기반으로 기본 설정")
                return "rule"
            
            # 샘플 레코드 확인 (처음 5개)
            sample_records = records[:5]
            
            # 휴리스틱 1: 데이터 복잡도 체크
            has_complex_fields = False
            complex_indicators = ["metadata", "nested_data", "calculated_fields", "ai_analysis"]
            
            for record in sample_records:
                record_str = json.dumps(record, ensure_ascii=False).lower()
                if any(indicator in record_str for indicator in complex_indicators):
                    has_complex_fields = True
                    break
            
            # 휴리스틱 2: 예외/에러 케이스 체크
            has_exceptions = any(
                record.get("exception_flag") or 
                record.get("validation_errors") or
                record.get("requires_ai_judgment")
                for record in sample_records
            )
            
            # 휴리스틱 3: 비정형 데이터 존재 여부
            has_unstructured_data = any(
                isinstance(record.get("notes"), str) and len(record.get("notes", "")) > 500 or
                isinstance(record.get("description"), str) and len(record.get("description", "")) > 500
                for record in sample_records
            )
            
            # 판단 로직: 대부분의 경우 규칙 기반으로 처리
            if has_complex_fields or has_exceptions or has_unstructured_data:
                strategy = "policy"
                reason = []
                if has_complex_fields:
                    reason.append("복잡한 필드 존재")
                if has_exceptions:
                    reason.append("예외 케이스 존재")
                if has_unstructured_data:
                    reason.append("비정형 데이터 존재")
                
                logger.info(
                    f"[ORCHESTRATOR] 휴리스틱 판단 결과: {strategy} "
                    f"(이유: {', '.join(reason)})"
                )
            else:
                strategy = "rule"
                logger.info(
                    f"[ORCHESTRATOR] 휴리스틱 판단 결과: {strategy} "
                    f"(구조화된 데이터, 단순 검증 필요)"
                )
            
            return strategy
            
        except Exception as e:
            logger.warning(
                f"[ORCHESTRATOR] 휴리스틱 판단 실패, 기본값(규칙 기반) 사용: {e}"
            )
            return "rule"
    
    def _determine_strategy_koelectra_with_confidence(
        self, records: List[Dict[str, Any]]
    ) -> tuple[str, float]:
        """KoELECTRA를 사용하여 전략을 판단하고 신뢰도도 반환합니다."""
        try:
            # 레코드들을 텍스트로 변환 (샘플 데이터 사용)
            sample_records = records[:5]  # 처음 5개만 사용
            text_data = json.dumps(sample_records, ensure_ascii=False)
            
            # 프롬프트 구성
            prompt = f"""
다음 Player 데이터를 분석하여 정책 기반 처리인지 규칙 기반 처리인지 판단하세요.

정책 기반: 복잡한 비즈니스 로직, 동적 판단, AI/ML 기반 의사결정이 필요한 경우
규칙 기반: 단순한 규칙, 정형화된 로직, 명확한 조건문으로 처리 가능한 경우

데이터:
{text_data[:1000]}  # 최대 1000자만 사용

답변 형식: "policy" 또는 "rule"
"""
            
            # 토크나이징
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 추론
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probs, dim=-1).item()
            
            # 0: 정책 기반, 1: 규칙 기반
            strategy = "policy" if predicted_class == 0 else "rule"
            confidence = probs[0][predicted_class].item()
            
            logger.info(
                f"[ORCHESTRATOR] KoELECTRA 판단 결과: {strategy} "
                f"(신뢰도: {confidence:.2%})"
            )
            
            return strategy, confidence
            
        except Exception as e:
            logger.warning(
                f"[ORCHESTRATOR] KoELECTRA 판단 실패: {e}"
            )
            raise
    
    async def process(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Player 레코드들을 처리합니다.
        
        LangGraph StateGraph를 통해 정책 기반 또는 규칙 기반으로 처리합니다.
        
        Args:
            records: 처리할 Player 레코드 리스트
            
        Returns:
            처리 결과 딕셔너리
        """
        logger.info(f"[ORCHESTRATOR] Player 처리 시작: {len(records)}개 레코드")
        
        # 초기 상태 생성
        initial_state: PlayerProcessingState = {
            "records": records,
            "current_step": "start",
            "strategy_type": None,
            "strategy_confidence": None,
            "heuristic_result": None,
            "koelectra_result": None,
            "validated_records": None,
            "validation_errors": None,
            "result": None,
            "processed_count": None,
            "errors": [],
        }
        
        # LangGraph 실행
        final_state = await self.app.ainvoke(initial_state)
        
        # 결과 반환
        result = final_state.get("result", {})
        if not result:
            result = {
                "success": False,
                "message": "처리 실패",
                "errors": final_state.get("errors", []),
            }
        
        return result
