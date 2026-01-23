"""
EXAONE Reader 판독기 기능을 LangGraph로 구현.

EXAONE이 스팸 확률 계산 및 판단을 전부 수행합니다.
KoELECTRA는 도메인 분류만 수행하고, 스팸 분류 도메인으로 라우팅되면 EXAONE이 호출됩니다.
"""

import sys
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
agents_dir = current_file.parent  # api/app/domains/spam_classifier/agents/
spam_classifier_dir = agents_dir.parent  # api/app/domains/spam_classifier/
domains_dir = spam_classifier_dir.parent  # api/app/domains/
app_dir = domains_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

sys.path.insert(0, str(api_dir))

from ..models.state_model import VerdictAgentState
from .model_loader import load_exaone_model


# ==================== 공통 분석 로직 ====================
def _execute_exaone_analysis(email_text: str) -> Dict[str, Any]:
    """EXAONE 분석 공통 로직 - 스팸 확률 계산 및 판단을 전부 수행.

    Args:
        email_text: 분석할 이메일 텍스트

    Returns:
        {
            "spam_prob": float,  # 스팸 확률 (0.0 ~ 1.0)
            "ham_prob": float,   # 정상 메일 확률 (0.0 ~ 1.0)
            "label": str,        # "spam" 또는 "ham"
            "confidence": str,   # "low", "medium", "high"
            "analysis": str,     # 상세 분석 결과 (판단 근거 및 주요 특징)
        }
    """
    try:
        llm = load_exaone_model()

        # EXAONE에게 스팸 확률 계산 및 판별 요청
        prompt = f"""다음 이메일을 분석하여 스팸 확률을 계산하고, 스팸인지 판별해주세요.

이메일 내용:
{email_text}

다음 형식으로 답변해주세요:
1. 스팸 확률: 0.0 ~ 1.0 사이의 숫자 (예: 0.85)
2. 정상 메일 확률: 0.0 ~ 1.0 사이의 숫자 (예: 0.15)
3. 최종 판단: 스팸 또는 정상
4. 판단 근거: 구체적인 이유를 설명
5. 주요 특징: 브랜드 사칭, 도메인 불일치, 긴급 언어, 금전 요청 등

JSON 형식으로 답변해주세요:
{{
    "spam_prob": 0.85,
    "ham_prob": 0.15,
    "label": "spam",
    "confidence": "high",
    "analysis": "상세 분석 내용..."
}}"""

        messages = [
            SystemMessage(
                content="당신은 스팸 메일 분석 전문가입니다. 이메일을 분석하여 스팸 확률을 계산하고, 스팸 여부를 판별하며 구체적인 근거를 제시합니다. 반드시 JSON 형식으로 답변하세요."
            ),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)
        result_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        # JSON 파싱 시도
        import json
        import re

        # JSON 부분만 추출
        json_match = re.search(r"\{[^{}]*\}", result_text, re.DOTALL)
        if json_match:
            try:
                result_dict = json.loads(json_match.group())
                # 필수 필드 확인 및 기본값 설정
                spam_prob = float(result_dict.get("spam_prob", 0.5))
                ham_prob = float(result_dict.get("ham_prob", 1.0 - spam_prob))
                label = result_dict.get("label", "spam" if spam_prob > 0.5 else "ham")
                confidence = result_dict.get("confidence", "medium")
                analysis = result_dict.get("analysis", result_text)

                # 신뢰도 자동 계산 (확률 기반)
                if abs(spam_prob - 0.5) < 0.1:
                    confidence = "low"
                elif abs(spam_prob - 0.5) < 0.3:
                    confidence = "medium"
                else:
                    confidence = "high"

                result = {
                    "spam_prob": spam_prob,
                    "ham_prob": ham_prob,
                    "label": label,
                    "confidence": confidence,
                    "analysis": analysis,
                }

                print(
                    f"[EXAONE] 스팸 확률: {spam_prob:.4f}, 레이블: {label}, 신뢰도: {confidence}"
                )
                return result
            except json.JSONDecodeError:
                pass

        # JSON 파싱 실패 시 텍스트에서 확률 추출 시도
        spam_prob_match = re.search(r"스팸 확률[:\s]*([0-9.]+)", result_text)
        spam_prob = float(spam_prob_match.group(1)) if spam_prob_match else 0.5
        ham_prob = 1.0 - spam_prob
        label = "spam" if spam_prob > 0.5 else "ham"

        if abs(spam_prob - 0.5) < 0.1:
            confidence = "low"
        elif abs(spam_prob - 0.5) < 0.3:
            confidence = "medium"
        else:
            confidence = "high"

        result = {
            "spam_prob": spam_prob,
            "ham_prob": ham_prob,
            "label": label,
            "confidence": confidence,
            "analysis": result_text,
        }

        print(
            f"[EXAONE] 스팸 확률: {spam_prob:.4f}, 레이블: {label}, 신뢰도: {confidence}"
        )
        return result

    except Exception as e:
        error_msg = f"EXAONE Reader 호출 실패: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback

        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
        return {
            "spam_prob": 0.5,
            "ham_prob": 0.5,
            "label": "unknown",
            "confidence": "low",
            "analysis": error_msg,
        }


# ==================== LangGraph 노드 ====================
def exaone_reader_node(state: VerdictAgentState) -> VerdictAgentState:
    """EXAONE Reader 노드: 스팸 확률 계산 및 판단."""
    email_text = state["email_text"]

    # 공통 분석 로직 호출 (스팸 확률 계산 및 판단 전부 수행)
    exaone_result = _execute_exaone_analysis(email_text)

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
def exaone_spam_analyzer(email_text: str) -> str:
    """EXAONE Reader로 스팸 메일을 분석합니다 (스팸 확률 계산 및 판단 전부 수행).

    Args:
        email_text: 분석할 이메일 텍스트

    Returns:
        EXAONE Reader의 분석 결과 (JSON 형식 문자열)
        {
            "spam_prob": float,
            "ham_prob": float,
            "label": str,
            "confidence": str,
            "analysis": str
        }
    """
    import json

    # 공통 분석 로직 호출 (스팸 확률 계산 및 판단 전부 수행)
    result = _execute_exaone_analysis(email_text)

    # JSON 문자열로 반환
    return json.dumps(result, ensure_ascii=False, indent=2)


# 툴 목록
VERDICT_TOOLS = [exaone_spam_analyzer]


# ==================== 공개 함수 ====================
def analyze_with_exaone(email_text: str) -> dict:
    """EXAONE Reader로 이메일을 분석합니다 (스팸 확률 계산 및 판단 전부 수행).

    Args:
        email_text: 이메일 텍스트

    Returns:
        {
            "spam_prob": float,
            "ham_prob": float,
            "label": str,
            "confidence": str,
            "analysis": str
        }
    """
    graph = get_verdict_agent_graph()

    initial_state: VerdictAgentState = {
        "email_text": email_text,
        "gate_result": {},  # 더 이상 사용하지 않음
        "exaone_result": None,
        "should_call_exaone": None,
    }

    result = graph.invoke(initial_state)
    exaone_result = result.get("exaone_result")

    # exaone_result가 문자열인 경우 JSON 파싱
    if isinstance(exaone_result, str):
        import json

        try:
            return json.loads(exaone_result)
        except json.JSONDecodeError:
            return {
                "spam_prob": 0.5,
                "ham_prob": 0.5,
                "label": "unknown",
                "confidence": "low",
                "analysis": exaone_result,
            }

    return exaone_result or {
        "spam_prob": 0.5,
        "ham_prob": 0.5,
        "label": "unknown",
        "confidence": "low",
        "analysis": "분석 실패",
    }


def get_exaone_tool():
    """EXAONE 스팸 분석기 툴을 반환합니다.

    Returns:
        exaone_spam_analyzer 툴
    """
    return exaone_spam_analyzer
