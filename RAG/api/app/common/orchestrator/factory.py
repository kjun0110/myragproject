"""
오케스트레이터 팩토리.

도메인에 맞는 오케스트레이터를 생성/반환합니다.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Type

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
orchestrator_dir = current_file.parent  # api/app/common/orchestrator/
common_dir = orchestrator_dir.parent  # api/app/common/
app_dir = common_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

sys.path.insert(0, str(api_dir))

from .base_orchestrator import BaseOrchestrator


class OrchestratorFactory:
    """오케스트레이터 팩토리 클래스."""

    _orchestrators: Dict[str, Type[BaseOrchestrator]] = {}
    _instances: Dict[str, BaseOrchestrator] = {}
    _default_orchestrator: Optional[str] = None

    @classmethod
    def register(
        cls,
        name: str,
        orchestrator_class: Type[BaseOrchestrator],
        is_default: bool = False,
    ) -> None:
        """오케스트레이터 클래스를 등록합니다.

        Args:
            name: 오케스트레이터 이름 (예: "spam_classifier", "chat")
            orchestrator_class: 오케스트레이터 클래스
            is_default: 기본 오케스트레이터로 설정할지 여부
        """
        cls._orchestrators[name] = orchestrator_class
        if is_default or cls._default_orchestrator is None:
            cls._default_orchestrator = name

    @classmethod
    def get(
        cls, name: Optional[str] = None, use_singleton: bool = True
    ) -> BaseOrchestrator:
        """오케스트레이터 인스턴스를 반환합니다.

        Args:
            name: 오케스트레이터 이름 (None이면 기본 오케스트레이터 사용)
            use_singleton: 싱글톤 패턴 사용 여부 (기본값: True)

        Returns:
            오케스트레이터 인스턴스

        Raises:
            ValueError: 오케스트레이터가 등록되지 않았거나 찾을 수 없는 경우
        """
        orchestrator_name = name or cls._default_orchestrator

        if orchestrator_name is None:
            raise ValueError("오케스트레이터가 등록되지 않았습니다.")

        if orchestrator_name not in cls._orchestrators:
            raise ValueError(f"오케스트레이터 '{orchestrator_name}'을 찾을 수 없습니다.")

        # 싱글톤 패턴 사용
        if use_singleton:
            if orchestrator_name not in cls._instances:
                cls._instances[orchestrator_name] = cls._orchestrators[
                    orchestrator_name
                ]()
            return cls._instances[orchestrator_name]
        else:
            # 매번 새 인스턴스 생성
            return cls._orchestrators[orchestrator_name]()

    @classmethod
    def list_orchestrators(cls) -> list:
        """등록된 오케스트레이터 목록을 반환합니다.

        Returns:
            오케스트레이터 이름 목록
        """
        return list(cls._orchestrators.keys())

    @classmethod
    def get_default(cls) -> Optional[str]:
        """기본 오케스트레이터 이름을 반환합니다.

        Returns:
            기본 오케스트레이터 이름
        """
        return cls._default_orchestrator


# 도메인별 오케스트레이터 자동 등록
def _register_domain_orchestrators():
    """도메인별 오케스트레이터를 자동으로 등록합니다."""
    try:
        from app.domains.v1.spam_classifier.orchestrator import SpamClassifierOrchestrator
        OrchestratorFactory.register("spam_classifier", SpamClassifierOrchestrator, is_default=True)
        print("[INFO] spam_classifier 오케스트레이터 등록 완료")
    except ImportError as e:
        print(f"[WARNING] spam_classifier 오케스트레이터 로드 실패: {e}")

    try:
        from app.domains.v1.chat.orchestrator import ChatOrchestrator
        OrchestratorFactory.register("chat", ChatOrchestrator)
        print("[INFO] chat 오케스트레이터 등록 완료")
    except ImportError as e:
        print(f"[WARNING] chat 오케스트레이터 로드 실패: {e}")


# 모듈 로드 시 자동 등록
_register_domain_orchestrators()
