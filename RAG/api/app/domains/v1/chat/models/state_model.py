"""
채팅 에이전트 상태 모델 정의.

LangGraph에서 사용하는 상태 구조를 정의합니다.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """채팅 에이전트 상태."""

    # messages: 대화 로그(누적)
    messages: Annotated[list, add_messages]
