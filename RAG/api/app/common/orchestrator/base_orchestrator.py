"""
Base 오케스트레이터 추상 클래스.

모든 오케스트레이터의 공통 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseOrchestrator(ABC):
    """모든 오케스트레이터의 기본 추상 클래스."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """오케스트레이터 초기화.

        Args:
            thresholds: 임계치 설정 (도메인별로 다를 수 있음)
        """
        self.thresholds = thresholds or self._get_default_thresholds()
        self._cache: Dict[str, Dict[str, Any]] = {}

    @abstractmethod
    def _get_default_thresholds(self) -> Dict[str, float]:
        """도메인별 기본 임계치를 반환합니다.

        Returns:
            기본 임계치 딕셔너리
        """
        pass

    @abstractmethod
    def classify_domain(self, text: str) -> Dict[str, Any]:
        """KoELECTRA로 텍스트를 분류하고 정책/규칙 기반을 결정합니다.

        Args:
            text: 분석할 텍스트

        Returns:
            분류 결과 딕셔너리 (policy, confidence, domain 등)
        """
        pass

    @abstractmethod
    def analyze(self, text: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """텍스트를 분석하고 결과를 반환합니다.

        Args:
            text: 분석할 텍스트
            request_id: 요청 ID (상태 관리용, 선택사항)

        Returns:
            분석 결과 딕셔너리
        """
        pass

    def should_use_agents(self, policy: str, confidence: float) -> bool:
        """정책 관련 기능(agents)을 사용해야 하는지 결정합니다.

        Args:
            policy: 정책 결정 결과
            confidence: 신뢰도

        Returns:
            True: agents 폴더 기능 사용 (정책 관련)
            False: services 폴더 기능 사용 (규칙 기반)
        """
        # 정책이 ANALYZE_* 계열이면 agents 사용
        if policy.startswith("ANALYZE_"):
            return True

        # BYPASS나 기타 정책은 services 사용 (규칙 기반)
        return False

    def get_state(self, request_id: str) -> Dict[str, Any]:
        """저장된 결과를 조회합니다.

        Args:
            request_id: 요청 ID

        Returns:
            저장된 결과

        Raises:
            KeyError: 요청 ID가 없을 경우
        """
        if request_id not in self._cache:
            raise KeyError(f"요청 ID '{request_id}'에 대한 결과를 찾을 수 없습니다.")
        return self._cache[request_id]

    def delete_state(self, request_id: str) -> Dict[str, str]:
        """저장된 결과를 삭제합니다.

        Args:
            request_id: 요청 ID

        Returns:
            삭제 성공 메시지

        Raises:
            KeyError: 요청 ID가 없을 경우
        """
        if request_id not in self._cache:
            raise KeyError(f"요청 ID '{request_id}'에 대한 결과를 찾을 수 없습니다.")
        del self._cache[request_id]
        return {"message": f"요청 ID '{request_id}'의 결과가 삭제되었습니다."}

    def list_states(self) -> Dict[str, list]:
        """저장된 모든 결과의 ID 목록을 반환합니다.

        Returns:
            요청 ID 목록
        """
        return {"request_ids": list(self._cache.keys())}

    def _cache_result(self, request_id: str, result: Dict[str, Any]) -> None:
        """결과를 캐시에 저장합니다.

        Args:
            request_id: 요청 ID
            result: 저장할 결과
        """
        self._cache[request_id] = result
