# ✅ 로컬 Exaone3.5 모델 사용 (model_service를 통해 로드)
import re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

# State 정의 import
from app.domains.chat.models.state_model import AgentState


# -------------------------
# 2) Tool 정의(예시)
# -------------------------
@tool
def get_server_time() -> str:
    """Return server time as ISO string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


TOOLS = [get_server_time]

# -------------------------
# 3) Model 노드 - 로컬 Exaone3.5 모델 로드
# -------------------------
# 모델은 서버 시작 시 미리 로드 (eager loading)
_llm = None
_llm_loading = False
_llm_error = None


def _load_exaone_model():
    """Exaone3.5 모델을 로드하는 함수."""
    global _llm, _llm_loading, _llm_error

    if _llm is not None:
        return _llm

    # 이전 에러가 있으면 재시도하지 않고 바로 에러 반환
    if _llm_error is not None:
        raise RuntimeError(f"이전 모델 로드 실패: {_llm_error}")

    if _llm_loading:
        raise RuntimeError("모델이 현재 로딩 중입니다. 잠시 후 다시 시도해주세요.")

    _llm_loading = True
    try:
        # chat/agents에서 import
        from .model_loader import load_exaone_model_for_service

        # Exaone 모델 로드
        print("[INFO] Exaone3.5 모델 로딩 시작... (이 작업은 몇 분이 걸릴 수 있습니다)")
        _llm = load_exaone_model_for_service()

        # Tool binding
        _llm = _llm.bind_tools(TOOLS)
        print("[OK] 로컬 Exaone3.5 모델이 graph.py에서 로드되었습니다.")
        print("[INFO] Tool calling이 활성화되었습니다.")
        _llm_loading = False
        _llm_error = None  # 성공 시 에러 상태 초기화
        return _llm

    except Exception as e:
        _llm_loading = False
        error_msg = f"Exaone3.5 모델 로드 실패: {str(e)}"
        _llm_error = error_msg
        print(f"[ERROR] {error_msg}")
        import traceback

        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
        raise RuntimeError(error_msg) from e


def preload_exaone_model():
    """서버 시작 시 Exaone 모델을 미리 로드하는 함수."""
    global _llm
    if _llm is None:
        print("[INFO] Exaone3.5 모델을 미리 로드합니다...")
        try:
            _load_exaone_model()
            print("[OK] Exaone3.5 모델 사전 로드 완료")
        except Exception as e:
            print(f"[WARNING] Exaone3.5 모델 사전 로드 실패: {str(e)}")
            print("[INFO] 첫 요청 시 로드됩니다.")


def model_node(state: AgentState):
    # state["messages"]는 누적 메시지
    # Lazy loading: 모델이 아직 로드되지 않았으면 지금 로드
    llm_with_tools = _load_exaone_model()
    resp = llm_with_tools.invoke(state["messages"])
    return {"messages": [resp]}


# -------------------------
# 4) Tool 노드
# -------------------------
def tool_node(state: AgentState):
    # 가장 최근 AIMessage가 tool_calls를 갖는지 확인
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []

    results = []
    # tool_calls는 {"name": ..., "args": ...} 형태를 포함
    tool_map = {t.name: t for t in TOOLS}

    for call in tool_calls:
        name = call["name"]
        args = call.get("args") or {}
        if name not in tool_map:
            results.append(f"Tool {name} not found")
            continue
        out = tool_map[name].invoke(args)
        # Tool 결과는 ToolMessage로 넣는 게 정석이지만,
        # LangChain이 반환 타입을 ToolMessage로 감싸주는 경로가 있어
        # 간단히 문자열로도 동작합니다(테스트 목적).
        results.append(out)

    # Tool 결과를 다음 모델 입력 메시지로 연결
    from langchain_core.messages import ToolMessage

    tool_messages = []
    for i, out in enumerate(results):
        tool_messages.append(
            ToolMessage(content=str(out), tool_call_id=tool_calls[i]["id"])
        )
    return {"messages": tool_messages}


# -------------------------
# 5) 조건 분기(도구 호출 여부)
# -------------------------
def should_use_tools(state: AgentState):
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if tool_calls:
        return "tools"
    return "end"


# -------------------------
# 6) Graph 빌드
# -------------------------
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("model", model_node)
    g.add_node("tools", tool_node)

    g.set_entry_point("model")
    g.add_conditional_edges(
        "model",
        should_use_tools,
        {
            "tools": "tools",
            "end": END,
        },
    )
    g.add_edge("tools", "model")
    return g.compile()


graph = build_graph()


# -------------------------
# 7) 간단 실행 헬퍼
# -------------------------
def run_once(user_text: str):
    init_state: AgentState = {
        "messages": [
            SystemMessage(
                content="You are a helpful assistant. Use tools when needed."
            ),
            HumanMessage(content=user_text),
        ]
    }
    out = graph.invoke(init_state)
    response_text = out["messages"][-1].content

    # 응답 정리: 시스템 프롬프트, 태그, 과거 대화 내용 제거
    if response_text:
        # 1. [[system]], [[endofturn]], [[assistant]] 같은 태그 제거
        response_text = re.sub(r'\[\[system\]\].*?\[\[endofturn\]\]\s*', '', response_text, flags=re.DOTALL)
        response_text = re.sub(r'\[\[assistant\]\]\s*', '', response_text, flags=re.IGNORECASE)
        response_text = re.sub(r'\[\[endofturn\]\]\s*', '', response_text, flags=re.IGNORECASE)
        response_text = re.sub(r'\[\[user\]\]\s*', '', response_text, flags=re.IGNORECASE)

        # 2. Human:, Assistant: 같은 이전 대화 형식 제거
        if "Human:" in response_text or "Assistant:" in response_text:
            # 마지막 Assistant: 이후만 추출
            assistant_match = re.search(
                r"Assistant:\s*(.+?)(?:\nHuman:|$)", response_text, re.DOTALL
            )
            if assistant_match:
                response_text = assistant_match.group(1).strip()

        # 3. 간단한 인사에 대한 응답 정리
        if any(greeting in user_text.lower() for greeting in ["안녕", "안녕하세요", "hi", "hello"]):
            # 인사에 대한 간단한 응답만 남기기
            lines = response_text.split('\n')
            clean_lines = []
            for line in lines:
                line = line.strip()
                # 빈 줄, 태그, 시스템 메시지 제외
                if not line or line.startswith('[') or 'system' in line.lower() or 'endofturn' in line.lower():
                    continue
                # 사용자 질문 반복 제거
                if any(greeting in line.lower() for greeting in ["너 이름이", "what's your name", "who are you"]):
                    continue
                clean_lines.append(line)

            if clean_lines:
                response_text = '\n'.join(clean_lines)
            else:
                # 기본 인사 응답
                response_text = "안녕하세요! 어떻게 도와드릴 수 있을까요? 궁금한 점이 있거나 도움이 필요한 사항이 있으면 말씀해 주세요. 😊"

    return response_text
