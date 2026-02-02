"""데이터 처리 전략 타입 Enum 정의."""

from enum import Enum


class StrategyType(str, Enum):
    """데이터 처리 전략 타입.

    v10 soccer 오케스트레이터의 `strategy_type`과 동일한 값 집합을 공유합니다.
    """

    POLICY = "policy"  # 정책 기반 처리
    RULE = "rule"  # 규칙 기반 처리

    def __str__(self) -> str:
        return str(self.value)
