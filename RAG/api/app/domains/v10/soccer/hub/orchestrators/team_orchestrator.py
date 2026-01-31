"""
Team Orchestrator - LangGraph 기반 정책/규칙 라우팅

KoELECTRA를 사용하여 데이터가 정책 기반인지 규칙 기반인지 판단하고,
LangGraph StateGraph를 통해 적절한 경로로 처리합니다.
"""

import json
import logging
from typing import Any, Dict, List, Literal

import torch
from langgraph.graph import END, START, StateGraph
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.loaders import ModelLoader
from app.domains.v10.soccer.models.states.team_state import TeamProcessingState

logger = logging.getLogger(__name__)


class TeamOrchestrator:
    """Team 데이터 처리 오케스트레이터 (LangGraph 기반)."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self._load_model()
        
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    def _load_model(self):
        """KoELECTRA 모델을 로드합니다."""
        try:
            logger.info("[ORCHESTRATOR] KoELECTRA 모델 로드 시작...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"[ORCHESTRATOR] 디바이스: {self.device}")
            
            self.model, self.tokenizer = ModelLoader.load_koelectra_model(
                adapter_name=None,
                device=self.device,
                num_labels=2,
            )
            
            self.model.eval()
            logger.info("[ORCHESTRATOR] KoELECTRA 모델 로드 완료")
            
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] KoELECTRA 모델 로드 실패: {e}", exc_info=True)
            raise
    
    def _build_graph(self) -> StateGraph:
        """LangGraph StateGraph를 빌드합니다."""
        graph = StateGraph(TeamProcessingState)
        
        graph.add_node("validate", self._validate_node)
        graph.add_node("determine_strategy", self._determine_strategy_node)
        graph.add_node("policy_process", self._policy_process_node)
        graph.add_node("rule_process", self._rule_process_node)
        graph.add_node("finalize", self._finalize_node)
        
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
    
    async def _validate_node(self, state: TeamProcessingState) -> Dict[str, Any]:
        """데이터 검증 노드."""
        logger.info(f"[ORCHESTRATOR] 검증 노드 시작: {len(state['records'])}개 레코드")
        
        if not state['records']:
            return {
                "current_step": "validate",
                "validation_errors": [{"error": "레코드가 없습니다."}],
                "errors": [{"step": "validate", "error": "레코드가 없습니다."}],
            }
        
        first_five_records = state['records'][:5]
        logger.info(f"[ORCHESTRATOR] 오케스트레이터에 도달한 데이터 상위 5개 레코드:")
        for idx, record in enumerate(first_five_records, 1):
            logger.info(f"[ORCHESTRATOR] 레코드 {idx}: {json.dumps(record, ensure_ascii=False, indent=2)}")
        
        return {
            "current_step": "validate",
            "validated_records": state['records'],
            "validation_errors": [],
        }
    
    async def _determine_strategy_node(self, state: TeamProcessingState) -> Dict[str, Any]:
        """전략 판단 노드."""
        logger.info("[ORCHESTRATOR] 전략 판단 노드 시작")
        
        records = state.get('validated_records', state['records'])
        heuristic_result = self._determine_strategy_heuristic(records)
        
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
        
        final_strategy = heuristic_result
        logger.info(
            f"[ORCHESTRATOR] 최종 전략 결정: {final_strategy} "
            f"(휴리스틱 기반, JSONL → team 테이블 저장)"
        )
        
        return {
            "current_step": "determine_strategy",
            "strategy_type": final_strategy,
            "strategy_confidence": confidence,
            "heuristic_result": heuristic_result,
            "koelectra_result": koelectra_result,
        }
    
    def _route_strategy(self, state: TeamProcessingState) -> Literal["policy", "rule"]:
        """전략 라우팅 함수."""
        strategy = state.get("strategy_type", "rule")
        logger.info(f"[ORCHESTRATOR] 전략 라우팅: {strategy}")
        return strategy  # type: ignore
    
    async def _policy_process_node(self, state: TeamProcessingState) -> Dict[str, Any]:
        """정책 기반 처리 노드 (Agent 호출)."""
        logger.info("[ORCHESTRATOR] 정책 기반 처리 노드 시작")
        
        try:
            from app.domains.v10.soccer.spokes.agents.team_agent import TeamAgent
            
            agent = TeamAgent()
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
    
    async def _rule_process_node(self, state: TeamProcessingState) -> Dict[str, Any]:
        """규칙 기반 처리 노드 (Service 호출)."""
        logger.info("[ORCHESTRATOR] 규칙 기반 처리 노드 시작")
        
        try:
            from app.domains.v10.soccer.spokes.services.team_service import TeamService
            
            service = TeamService()
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
    
    async def _finalize_node(self, state: TeamProcessingState) -> Dict[str, Any]:
        """최종 정리 노드."""
        logger.info("[ORCHESTRATOR] 최종 정리 노드 시작")
        
        result = state.get("result", {})
        if not result:
            result = {
                "success": False,
                "message": "처리 실패",
                "errors": state.get("errors", []),
            }
        
        logger.info("[ORCHESTRATOR] Team 처리 완료")
        
        return {
            "current_step": "finalize",
            "result": result,
        }
    
    def _determine_strategy_heuristic(self, records: List[Dict[str, Any]]) -> str:
        """휴리스틱을 사용하여 정책 기반인지 규칙 기반인지 판단합니다."""
        try:
            if not records:
                return "rule"
            
            sample_records = records[:5]
            has_complex_fields = False
            complex_indicators = ["metadata", "nested_data", "calculated_fields", "ai_analysis"]
            
            for record in sample_records:
                record_str = json.dumps(record, ensure_ascii=False).lower()
                if any(indicator in record_str for indicator in complex_indicators):
                    has_complex_fields = True
                    break
            
            has_exceptions = any(
                record.get("exception_flag") or 
                record.get("validation_errors") or
                record.get("requires_ai_judgment")
                for record in sample_records
            )
            
            has_unstructured_data = any(
                isinstance(record.get("notes"), str) and len(record.get("notes", "")) > 500 or
                isinstance(record.get("description"), str) and len(record.get("description", "")) > 500
                for record in sample_records
            )
            
            if has_complex_fields or has_exceptions or has_unstructured_data:
                strategy = "policy"
                logger.info(f"[ORCHESTRATOR] 휴리스틱 판단 결과: {strategy}")
            else:
                strategy = "rule"
                logger.info(f"[ORCHESTRATOR] 휴리스틱 판단 결과: {strategy} (구조화된 데이터, 단순 검증 필요)")
            
            return strategy
            
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] 휴리스틱 판단 실패, 기본값(규칙 기반) 사용: {e}")
            return "rule"
    
    def _determine_strategy_koelectra_with_confidence(
        self, records: List[Dict[str, Any]]
    ) -> tuple[str, float]:
        """KoELECTRA를 사용하여 전략을 판단하고 신뢰도도 반환합니다."""
        try:
            sample_records = records[:5]
            text_data = json.dumps(sample_records, ensure_ascii=False)
            
            prompt = f"""
다음 Team 데이터를 분석하여 정책 기반 처리인지 규칙 기반 처리인지 판단하세요.

정책 기반: 복잡한 비즈니스 로직, 동적 판단, AI/ML 기반 의사결정이 필요한 경우
규칙 기반: 단순한 규칙, 정형화된 로직, 명확한 조건문으로 처리 가능한 경우

데이터:
{text_data[:1000]}

답변 형식: "policy" 또는 "rule"
"""
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probs, dim=-1).item()
            
            strategy = "policy" if predicted_class == 0 else "rule"
            confidence = probs[0][predicted_class].item()
            
            logger.info(f"[ORCHESTRATOR] KoELECTRA 판단 결과: {strategy} (신뢰도: {confidence:.2%})")
            
            return strategy, confidence
            
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] KoELECTRA 판단 실패: {e}")
            raise
    
    async def process(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Team 레코드들을 처리합니다."""
        logger.info(f"[ORCHESTRATOR] Team 처리 시작: {len(records)}개 레코드")
        
        initial_state: TeamProcessingState = {
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
        
        final_state = await self.app.ainvoke(initial_state)
        
        result = final_state.get("result", {})
        if not result:
            result = {
                "success": False,
                "message": "처리 실패",
                "errors": final_state.get("errors", []),
            }
        
        return result
    
    async def process_question(self, question: str) -> Dict[str, Any]:
        """
        채팅 질문을 처리합니다.
        
        KoELECTRA 분류기 어댑터를 사용하여 질문이 정책 기반인지 규칙 기반인지 판단합니다.
        
        Args:
            question: 사용자가 입력한 질문
            
        Returns:
            처리 결과 딕셔너리
        """
        logger.info("=" * 60)
        logger.info("[TEAM ORCHESTRATOR] 채팅 질문 처리 시작")
        logger.info(f"[TEAM ORCHESTRATOR] 받은 질문: {question}")
        logger.info("=" * 60)
        
        # 질문을 프린트
        print(f"\n{'='*60}")
        print(f"[TEAM ORCHESTRATOR] 채팅 질문 수신")
        print(f"[TEAM ORCHESTRATOR] 질문 내용: {question}")
        print(f"{'='*60}\n")
        
        # KoELECTRA 분류기 어댑터 로드 및 판단
        try:
            logger.info("[TEAM ORCHESTRATOR] KoELECTRA 분류기 어댑터 로드 시작...")
            
            # 어댑터 경로 직접 지정 (koelectra_classifier 디렉토리)
            from pathlib import Path
            from peft import PeftModel
            
            current_file = Path(__file__).resolve()
            # team_orchestrator.py 위치: api/app/domains/v10/soccer/hub/orchestrators/team_orchestrator.py
            # api_dir까지: parent 7번 (orchestrators -> hub -> soccer -> v10 -> domains -> app -> api)
            api_dir = current_file.parent.parent.parent.parent.parent.parent.parent  # api/ 디렉토리
            classifier_adapter_base = api_dir / "artifacts" / "koelectra" / "koelectra_classifier" / "koelectra-small-v3-discriminator-classifier-lora"
            
            # 최신 타임스탬프 디렉토리 찾기
            adapter_path = None
            if classifier_adapter_base.exists():
                subdirs = [d for d in classifier_adapter_base.iterdir() if d.is_dir()]
                if subdirs:
                    latest_adapter_path = max(subdirs, key=lambda x: x.stat().st_mtime)
                    adapter_path = str(latest_adapter_path)
                    logger.info(f"[TEAM ORCHESTRATOR] 어댑터 경로: {adapter_path}")
                else:
                    logger.warning(f"[TEAM ORCHESTRATOR] 어댑터 서브디렉토리를 찾을 수 없습니다: {classifier_adapter_base}")
            else:
                logger.warning(f"[TEAM ORCHESTRATOR] 어댑터 디렉토리를 찾을 수 없습니다: {classifier_adapter_base}")
            
            # 베이스 모델 로드
            model_path = ModelLoader.KOELECTRA_MODEL_ID
            logger.info(f"[TEAM ORCHESTRATOR] 베이스 모델 로드: {model_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=3,  # 3-class 분류
            )
            base_model.to(self.device)
            
            # 어댑터 로드
            if adapter_path:
                logger.info(f"[TEAM ORCHESTRATOR] 어댑터 로드 중: {adapter_path}")
                classifier_model = PeftModel.from_pretrained(base_model, adapter_path)
                classifier_tokenizer = tokenizer
                logger.info("[TEAM ORCHESTRATOR] 어댑터 로드 완료")
            else:
                logger.warning("[TEAM ORCHESTRATOR] 어댑터를 찾을 수 없어 베이스 모델만 사용합니다.")
                classifier_model = base_model
                classifier_tokenizer = tokenizer
            
            classifier_model.eval()
            logger.info("[TEAM ORCHESTRATOR] KoELECTRA 분류기 어댑터 로드 완료")
            
            # 질문 분류
            logger.info("[TEAM ORCHESTRATOR] 질문 분류 시작...")
            inputs = classifier_tokenizer(
                question,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = classifier_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][predicted_class].item()
            
            # 라벨 매핑: 0=도메인 외, 1=정책 기반, 2=규칙 기반
            label_map = {
                0: "도메인 외 (OUT_OF_DOMAIN)",
                1: "정책 기반 (POLICY_BASED)",
                2: "규칙 기반 (RULE_BASED)",
            }
            
            predicted_label = label_map.get(predicted_class, f"Unknown ({predicted_class})")
            
            # 각 클래스별 확률
            prob_out_of_domain = probs[0][0].item()
            prob_policy = probs[0][1].item()
            prob_rule = probs[0][2].item()
            
            # 결과 프린트
            print(f"\n{'='*60}")
            print(f"[TEAM ORCHESTRATOR] 질문 분류 결과")
            print(f"{'='*60}")
            print(f"질문: {question}")
            print(f"판단 결과: {predicted_label}")
            print(f"신뢰도: {confidence:.2%}")
            print(f"\n각 클래스별 확률:")
            print(f"  - 도메인 외 (0): {prob_out_of_domain:.2%}")
            print(f"  - 정책 기반 (1): {prob_policy:.2%}")
            print(f"  - 규칙 기반 (2): {prob_rule:.2%}")
            print(f"{'='*60}\n")
            
            logger.info(f"[TEAM ORCHESTRATOR] 질문 분류 완료: {predicted_label} (신뢰도: {confidence:.2%})")
            
            # 간단한 응답 반환
            from datetime import datetime
            result = {
                "success": True,
                "question": question,
                "classification": {
                    "label": predicted_class,
                    "label_name": predicted_label,
                    "confidence": float(confidence),
                    "probabilities": {
                        "out_of_domain": float(prob_out_of_domain),
                        "policy_based": float(prob_policy),
                        "rule_based": float(prob_rule),
                    },
                },
                "message": f"질문 '{question}'을 받았습니다. 분류 결과: {predicted_label}",
                "processed_at": datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"[TEAM ORCHESTRATOR] 분류기 로드/분류 실패: {e}", exc_info=True)
            print(f"\n{'='*60}")
            print(f"[TEAM ORCHESTRATOR] 분류기 오류")
            print(f"오류: {str(e)}")
            print(f"{'='*60}\n")
            
            # 오류 발생 시 기본 응답
            from datetime import datetime
            result = {
                "success": False,
                "question": question,
                "error": str(e),
                "message": f"질문 '{question}'을 받았지만 분류에 실패했습니다.",
                "processed_at": datetime.now().isoformat(),
            }
        
        logger.info(f"[TEAM ORCHESTRATOR] 채팅 질문 처리 완료")
        
        return result
