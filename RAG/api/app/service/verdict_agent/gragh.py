"""
EXAONE Reader 판독기 기능을 LangGraph로 구현.

KoELECTRA 게이트웨이 결과를 받아 EXAONE으로 정밀 검사를 수행합니다.
"""

import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
verdict_agent_dir = current_file.parent  # api/app/service/verdict_agent/
service_dir = verdict_agent_dir.parent  # api/app/service/
app_dir = service_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

sys.path.insert(0, str(api_dir))

from app.service.verdict_agent.base_model import load_exaone_model
from app.service.verdict_agent.state_model import VerdictAgentState


# ==================== 공통 분석 로직 ====================
def _execute_exaone_analysis(email_text: str, gate_result: dict) -> str:
    """EXAONE 분석 공통 로직.

    Args:
        email_text: 분석할 이메일 텍스트
        gate_result: KoELECTRA 게이트웨이 결과

    Returns:
        EXAONE Reader의 정밀 검사 결과
    """
    try:
        llm = load_exaone_model()

        # EXAONE에게 스팸 판별 및 근거 추출 요청
        prompt = f"""다음 이메일이 스팸인지 판별하고, 근거를 설명해주세요.

이메일 내용:
{email_text}

KoELECTRA 게이트웨이 결과:
- 스팸 확률: {gate_result.get("spam_prob", 0):.4f}
- 레이블: {gate_result.get("label", "unknown")}
- 신뢰도: {gate_result.get("confidence", "unknown")}

다음 형식으로 답변해주세요:
1. 최종 판단: 스팸 또는 정상
2. 판단 근거: 구체적인 이유를 설명
3. 주요 특징: 브랜드 사칭, 도메인 불일치, 긴급 언어, 금전 요청 등"""

        messages = [
            SystemMessage(
                content="당신은 스팸 메일 분석 전문가입니다. 이메일을 분석하여 스팸 여부를 판별하고 구체적인 근거를 제시합니다."
            ),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)
        result = response.content if hasattr(response, "content") else str(response)

        print("[EXAONE] 정밀 검사 완료")
        return result

    except Exception as e:
        error_msg = f"EXAONE Reader 호출 실패: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return error_msg


# ==================== LangGraph 노드 ====================
def exaone_reader_node(state: VerdictAgentState) -> VerdictAgentState:
    """EXAONE Reader 노드: 정밀 검사 및 근거 추출."""
    email_text = state["email_text"]
    gate_result = state.get("gate_result", {})

    # 공통 분석 로직 호출
    exaone_result = _execute_exaone_analysis(email_text, gate_result)

    return {
        **state,
        "exaone_result": exaone_result,
        "should_call_exaone": True,
    }


# ==================== LangGraph 구성 ====================
def build_verdict_agent_graph():
    """판독기 에이전트 LangGraph를 빌드합니다."""
    graph = StateGraph(VerdictAgentState)

    # 노드 추가
    graph.add_node("exaone_reader", exaone_reader_node)

    # 엔트리 포인트
    graph.set_entry_point("exaone_reader")

    # EXAONE Reader → 종료
    graph.add_edge("exaone_reader", END)

    return graph.compile()


# 그래프 인스턴스 생성
_verdict_agent_graph = None


def get_verdict_agent_graph():
    """판독기 에이전트 그래프를 가져옵니다 (lazy initialization)."""
    global _verdict_agent_graph
    if _verdict_agent_graph is None:
        _verdict_agent_graph = build_verdict_agent_graph()
    return _verdict_agent_graph


# ==================== LangChain Tool 래핑 ====================
@tool
def exaone_spam_analyzer(
    email_text: str, spam_prob: float, label: str, confidence: str
) -> str:
    """EXAONE Reader로 스팸 메일을 정밀 분석합니다.

    Args:
        email_text: 분석할 이메일 텍스트
        spam_prob: KoELECTRA 게이트웨이가 계산한 스팸 확률 (0.0 ~ 1.0)
        label: KoELECTRA 게이트웨이 레이블 ("spam" 또는 "ham")
        confidence: KoELECTRA 게이트웨이 신뢰도 ("low", "medium", "high")

    Returns:
        EXAONE Reader의 정밀 검사 결과 (스팸 판단 근거 및 주요 특징)
    """
    # gate_result 구성
    gate_result = {
        "spam_prob": spam_prob,
        "label": label,
        "confidence": confidence,
    }

    # 공통 분석 로직 호출
    return _execute_exaone_analysis(email_text, gate_result)


# 툴 목록
VERDICT_TOOLS = [exaone_spam_analyzer]


# ==================== 공개 함수 ====================
def analyze_with_exaone(email_text: str, gate_result: dict) -> dict:
    """EXAONE Reader로 이메일을 분석합니다.

    Args:
        email_text: 이메일 텍스트
        gate_result: KoELECTRA 게이트웨이 결과

    Returns:
        {
            "exaone_result": str,
            "should_call_exaone": bool
        }
    """
    graph = get_verdict_agent_graph()

    initial_state: VerdictAgentState = {
        "email_text": email_text,
        "gate_result": gate_result,
        "exaone_result": None,
        "should_call_exaone": None,
    }

    result = graph.invoke(initial_state)

    return {
        "exaone_result": result.get("exaone_result"),
        "should_call_exaone": result.get("should_call_exaone", False),
    }


def get_exaone_tool():
    """EXAONE 스팸 분석기 툴을 반환합니다.

    Returns:
        exaone_spam_analyzer 툴
    """
    return exaone_spam_analyzer
