"""채팅 에이전트 상태 모델 정의 (BaseModel 기반).

LangGraph에서 사용하는 상태 구조를 정의합니다.

의도:
- `models/transfers`: API 요청/응답(전달용 DTO)
- `models/states`: 정책(에이전트/그래프) 실행 시 내부 상태
"""

from typing import Annotated, Any, List

from langgraph.graph.message import add_messages

try:
    # pydantic v2
    from pydantic import BaseModel, Field
    from pydantic import ConfigDict  # type: ignore

    _USE_V2 = True
except Exception:  # pragma: no cover
    # pydantic v1
    from pydantic import BaseModel, Field  # type: ignore

    _USE_V2 = False


class AgentState(BaseModel):
    """채팅 에이전트 상태.

    - messages: 누적 대화 메시지(툴 호출 포함)
    """

    # LangGraph의 message aggregator를 유지하기 위해 Annotated를 사용
    messages: Annotated[List[Any], add_messages] = Field(default_factory=list)

    if _USE_V2:
        model_config = ConfigDict(arbitrary_types_allowed=True)
    else:

        class Config:  # type: ignore
            arbitrary_types_allowed = True

