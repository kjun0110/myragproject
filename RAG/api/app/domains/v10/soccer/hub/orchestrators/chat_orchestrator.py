"""
Chat Orchestrator - 질문을 받아서 적절한 오케스트레이터로 라우팅

KoELECTRA로 도메인 외 여부만 판단합니다.
- 도메인 외: 기본 오케스트레이터(player)로 전달.
- 도메인 내: KoELECTRA 라우팅 모델로 player, schedule, stadium, team 중 해당하는 곳으로 전달.
  (라우팅 어댑터가 없으면 키워드 기반으로 폴백)
정책/규칙 판단은 각 오케스트레이터에서 수행합니다.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.common.loaders import ModelLoader

logger = logging.getLogger(__name__)

# 라우팅 KoELECTRA 클래스 인덱스 → 오케스트레이터 이름 (학습 시 label 순서와 동일해야 함)
ORCHESTRATOR_LABELS: List[str] = ["player", "schedule", "stadium", "team"]


class ChatOrchestrator:
    """채팅 질문을 적절한 오케스트레이터로 라우팅하는 오케스트레이터."""
    
    def __init__(self):
        """ChatOrchestrator 초기화. KoELECTRA 도메인 분류기 및 라우팅 분류기 어댑터를 로드합니다."""
        self.model = None
        self.tokenizer = None
        self.device = None
        self.routing_model = None
        self.routing_tokenizer = None
        self._load_classifier()
        self._load_routing_classifier()
        logger.info("[CHAT ORCHESTRATOR] 초기화 완료")
    
    def _load_classifier(self) -> None:
        """KoELECTRA 분류기 어댑터를 로드합니다."""
        try:
            logger.info("[CHAT ORCHESTRATOR] KoELECTRA 분류기 어댑터 로드 시작...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"[CHAT ORCHESTRATOR] 디바이스: {self.device}")
            
            current_file = Path(__file__).resolve()
            api_dir = current_file.parent.parent.parent.parent.parent.parent.parent
            classifier_adapter_base = api_dir / "artifacts" / "koelectra" / "koelectra_classifier" / "koelectra-small-v3-discriminator-classifier-lora"
            
            adapter_path = None
            if classifier_adapter_base.exists():
                subdirs = [d for d in classifier_adapter_base.iterdir() if d.is_dir()]
                if subdirs:
                    latest = max(subdirs, key=lambda x: x.stat().st_mtime)
                    adapter_path = str(latest)
                    logger.info(f"[CHAT ORCHESTRATOR] 어댑터 경로: {adapter_path}")
                else:
                    logger.warning(f"[CHAT ORCHESTRATOR] 어댑터 서브디렉토리 없음: {classifier_adapter_base}")
            else:
                logger.warning(f"[CHAT ORCHESTRATOR] 어댑터 디렉토리 없음: {classifier_adapter_base}")
            
            model_path = ModelLoader.KOELECTRA_MODEL_ID
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=3,
            )
            base_model.to(self.device)
            
            if adapter_path:
                self.model = PeftModel.from_pretrained(base_model, adapter_path)
                self.tokenizer = tokenizer
                logger.info("[CHAT ORCHESTRATOR] 어댑터 로드 완료")
            else:
                self.model = base_model
                self.tokenizer = tokenizer
                logger.warning("[CHAT ORCHESTRATOR] 어댑터 없음, 베이스 모델만 사용")
            
            self.model.eval()
            logger.info("[CHAT ORCHESTRATOR] KoELECTRA 분류기 로드 완료")
        except Exception as e:
            logger.error(f"[CHAT ORCHESTRATOR] 분류기 로드 실패: {e}", exc_info=True)
            self.model = None
            self.tokenizer = None

    def _load_routing_classifier(self) -> None:
        """4클래스 라우팅용 KoELECTRA 어댑터를 로드합니다 (player, schedule, stadium, team)."""
        try:
            logger.info("[CHAT ORCHESTRATOR] KoELECTRA 라우팅 분류기 어댑터 로드 시작...")
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            current_file = Path(__file__).resolve()
            api_dir = current_file.parent.parent.parent.parent.parent.parent
            routing_base = api_dir / "artifacts" / "koelectra" / "koelectra_routing" / "koelectra-small-v3-discriminator-routing-lora"
            adapter_path = None
            if routing_base.exists():
                subdirs = [d for d in routing_base.iterdir() if d.is_dir()]
                if subdirs:
                    latest = max(subdirs, key=lambda x: x.stat().st_mtime)
                    adapter_path = str(latest)
                    logger.info(f"[CHAT ORCHESTRATOR] 라우팅 어댑터 경로: {adapter_path}")
                else:
                    logger.warning(f"[CHAT ORCHESTRATOR] 라우팅 어댑터 서브디렉토리 없음: {routing_base}")
            else:
                logger.warning(f"[CHAT ORCHESTRATOR] 라우팅 어댑터 디렉토리 없음: {routing_base}")
            if not adapter_path:
                self.routing_model = None
                self.routing_tokenizer = None
                logger.info("[CHAT ORCHESTRATOR] 라우팅 어댑터 없음 → 도메인 내 시 키워드로 폴백")
                return
            model_path = ModelLoader.KOELECTRA_MODEL_ID
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=len(ORCHESTRATOR_LABELS),
            )
            base_model.to(self.device)
            self.routing_model = PeftModel.from_pretrained(base_model, adapter_path)
            self.routing_tokenizer = tokenizer
            self.routing_model.eval()
            logger.info("[CHAT ORCHESTRATOR] KoELECTRA 라우팅 분류기 로드 완료")
        except Exception as e:
            logger.error(f"[CHAT ORCHESTRATOR] 라우팅 분류기 로드 실패: {e}", exc_info=True)
            self.routing_model = None
            self.routing_tokenizer = None

    def _determine_orchestrator_with_koelectra(self, question: str) -> Optional[str]:
        """
        KoELECTRA 라우팅 모델로 질문을 분류하여 오케스트레이터를 결정합니다.
        Returns:
            'player' | 'schedule' | 'stadium' | 'team' 또는 실패 시 None
        """
        if self.routing_model is None or self.routing_tokenizer is None:
            return None
        try:
            inputs = self.routing_tokenizer(
                question,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.routing_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probs, dim=-1).item()
            if predicted_class < 0 or predicted_class >= len(ORCHESTRATOR_LABELS):
                return None
            selected = ORCHESTRATOR_LABELS[predicted_class]
            logger.info(
                f"[CHAT ORCHESTRATOR] KoELECTRA 라우팅: {selected} (클래스={predicted_class}, "
                f"확률={probs[0][predicted_class].item():.2%})"
            )
            return selected
        except Exception as e:
            logger.error(f"[CHAT ORCHESTRATOR] KoELECTRA 라우팅 분류 실패: {e}", exc_info=True)
            return None

    def _classify_domain_with_koelectra(self, question: str) -> Optional[Tuple[bool, float]]:
        """
        KoELECTRA로 질문이 도메인 외인지 도메인 내인지만 판단합니다.
        정책/규칙 구분은 하지 않습니다.
        
        Returns:
            (is_out_of_domain, prob_out_of_domain) 또는 실패 시 None
        """
        if self.model is None or self.tokenizer is None:
            return None
        try:
            inputs = self.tokenizer(
                question,
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
            
            # 0=도메인 외, 1/2=도메인 내 (정책/규칙 구분 없음)
            prob_out_of_domain = probs[0][0].item()
            is_out_of_domain = predicted_class == 0
            return (is_out_of_domain, prob_out_of_domain)
        except Exception as e:
            logger.error(f"[CHAT ORCHESTRATOR] KoELECTRA 분류 실패: {e}", exc_info=True)
            return None
    
    def _determine_orchestrator(self, question: str) -> str:
        """
        키워드 기반으로 질문을 분석하여 적절한 오케스트레이터를 결정합니다.
        
        Args:
            question: 사용자 질문
            
        Returns:
            오케스트레이터 이름: 'player', 'schedule', 'stadium', 'team'
        """
        question_lower = question.lower()
        
        # 키워드 기반 점수
        player_keywords = ['선수', 'player', '플레이어', '골', '득점', '어시스트', '이름']
        schedule_keywords = ['일정', 'schedule', '경기', '매치', 'vs', '대전', '날짜', '시간']
        stadium_keywords = ['경기장', 'stadium', '구장', '아레나', '장소']
        team_keywords = ['팀', 'team', '클럽', '구단', '코드']
        
        player_score = sum(1 for keyword in player_keywords if keyword in question_lower)
        schedule_score = sum(1 for keyword in schedule_keywords if keyword in question_lower)
        stadium_score = sum(1 for keyword in stadium_keywords if keyword in question_lower)
        team_score = sum(1 for keyword in team_keywords if keyword in question_lower)
        
        # 가장 높은 점수의 오케스트레이터 선택
        scores = {
            'player': player_score,
            'schedule': schedule_score,
            'stadium': stadium_score,
            'team': team_score,
        }
        
        max_score = max(scores.values())
        if max_score == 0:
            # 기본값: player
            selected = 'player'
        else:
            selected = max(scores.keys(), key=lambda k: scores[k])
        
        logger.info(f"[CHAT ORCHESTRATOR] 오케스트레이터 선택: {selected} (점수: player={player_score}, schedule={schedule_score}, stadium={stadium_score}, team={team_score})")
        return selected
    
    async def process_question(self, question: str) -> Dict[str, Any]:
        """
        채팅 질문을 처리하고 적절한 오케스트레이터로 라우팅합니다.
        
        Args:
            question: 사용자가 입력한 질문
            
        Returns:
            처리 결과 딕셔너리
        """
        logger.info("=" * 60)
        logger.info("[CHAT ORCHESTRATOR] 채팅 질문 처리 시작")
        logger.info(f"[CHAT ORCHESTRATOR] 받은 질문: {question}")
        logger.info("=" * 60)
        
        # Chat Orchestrator 프린트
        print(f"\n{'='*60}")
        print(f"[CHAT ORCHESTRATOR] 채팅 질문 수신")
        print(f"[CHAT ORCHESTRATOR] 질문 내용: {question}")
        print(f"{'='*60}\n")
        
        # KoELECTRA로 도메인 외 여부만 판단 (정책/규칙 구분 없음)
        domain_result = self._classify_domain_with_koelectra(question)
        if domain_result is not None:
            is_out_of_domain, prob_out_of_domain = domain_result
            domain_label = "도메인 외 (OUT_OF_DOMAIN)" if is_out_of_domain else "도메인 내 (IN_DOMAIN)"
            print(f"\n{'='*60}")
            print(f"[CHAT ORCHESTRATOR] KoELECTRA 도메인 판단")
            print(f"{'='*60}")
            print(f"질문: {question}")
            print(f"판단 결과: {domain_label}")
            print(f"도메인 외 확률: {prob_out_of_domain:.2%}")
            print(f"{'='*60}\n")
            logger.info(f"[CHAT ORCHESTRATOR] KoELECTRA 도메인 판단: {domain_label}")
            
            if is_out_of_domain:
                # 도메인 외 → 기본 오케스트레이터(player)로 전달
                orchestrator_name = "player"
                logger.info("[CHAT ORCHESTRATOR] 도메인 외 → 기본 오케스트레이터(player)로 라우팅")
            else:
                # 도메인 내 → KoELECTRA 라우팅으로 4개 오케스트레이터 중 선택, 실패 시 키워드 폴백
                orchestrator_name = self._determine_orchestrator_with_koelectra(question)
                if orchestrator_name is None:
                    orchestrator_name = self._determine_orchestrator(question)
                    logger.info(f"[CHAT ORCHESTRATOR] 도메인 내 → 키워드 폴백 오케스트레이터: {orchestrator_name}")
                    print(f"[CHAT ORCHESTRATOR] 도메인 내 → 키워드 폴백으로 선택: {orchestrator_name.upper()}\n")
                else:
                    logger.info(f"[CHAT ORCHESTRATOR] 도메인 내 → KoELECTRA 라우팅 오케스트레이터: {orchestrator_name}")
                    print(f"[CHAT ORCHESTRATOR] 도메인 내 → KoELECTRA 라우팅으로 선택: {orchestrator_name.upper()}\n")
        else:
            print(f"\n[CHAT ORCHESTRATOR] KoELECTRA 분류기 없음, 키워드만으로 오케스트레이터 선택\n")
            logger.warning("[CHAT ORCHESTRATOR] KoELECTRA 분류기 없음")
            orchestrator_name = self._determine_orchestrator(question)
        
        print(f"\n{'='*60}")
        print(f"[CHAT ORCHESTRATOR] 오케스트레이터 라우팅 결정")
        print(f"{'='*60}")
        print(f"질문: {question}")
        print(f"선택된 오케스트레이터: {orchestrator_name.upper()}")
        print(f"{'='*60}\n")
        
        # 선택된 오케스트레이터로 질문 전달
        try:
            if orchestrator_name == 'player':
                from app.domains.v10.soccer.hub.orchestrators.player_orchestrator import PlayerOrchestrator
                orchestrator = PlayerOrchestrator()
            elif orchestrator_name == 'schedule':
                from app.domains.v10.soccer.hub.orchestrators.schedule_orchestrator import ScheduleOrchestrator
                orchestrator = ScheduleOrchestrator()
            elif orchestrator_name == 'stadium':
                from app.domains.v10.soccer.hub.orchestrators.stadium_orchestrator import StadiumOrchestrator
                orchestrator = StadiumOrchestrator()
            elif orchestrator_name == 'team':
                from app.domains.v10.soccer.hub.orchestrators.team_orchestrator import TeamOrchestrator
                orchestrator = TeamOrchestrator()
            else:
                raise ValueError(f"알 수 없는 오케스트레이터: {orchestrator_name}")
            
            # 오케스트레이터로 질문 전달
            result = await orchestrator.process_question(question)
            
            # 결과에 오케스트레이터 정보 추가
            result['orchestrator'] = orchestrator_name
            result['routed_by'] = 'ChatOrchestrator'
            
            logger.info(f"[CHAT ORCHESTRATOR] {orchestrator_name} 오케스트레이터로 라우팅 완료")
            
            return result
            
        except Exception as e:
            logger.error(f"[CHAT ORCHESTRATOR] 오케스트레이터 라우팅 실패: {e}", exc_info=True)
            print(f"\n{'='*60}")
            print(f"[CHAT ORCHESTRATOR] 오케스트레이터 라우팅 오류")
            print(f"오류: {str(e)}")
            print(f"{'='*60}\n")
            
            from datetime import datetime
            return {
                "success": False,
                "question": question,
                "error": str(e),
                "message": f"질문 '{question}'을 라우팅하는데 실패했습니다.",
                "processed_at": datetime.now().isoformat(),
            }
