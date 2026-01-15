"""
EXAONE Reader 베이스 모델 로딩.

이 모듈은 EXAONE 모델의 로딩과 전역 캐싱을 담당합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
verdict_agent_dir = current_file.parent  # api/app/service/verdict_agent/
service_dir = verdict_agent_dir.parent  # api/app/service/
app_dir = service_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

sys.path.insert(0, str(api_dir))

# 전역 변수 (모델 캐싱용)
_exaone_llm = None
_exaone_loading = False
_exaone_error = None


def load_exaone_model():
    """EXAONE Reader 모델을 로드합니다.

    전역 캐싱을 사용하여 한 번만 로드합니다.

    Returns:
        LangChain LLM 객체

    Raises:
        RuntimeError: 모델 로드 실패 시
    """
    global _exaone_llm, _exaone_loading, _exaone_error

    # 이미 로드된 경우
    if _exaone_llm is not None:
        return _exaone_llm

    # 이전 로드 실패한 경우
    if _exaone_error is not None:
        raise RuntimeError(f"이전 EXAONE 모델 로드 실패: {_exaone_error}")

    # 현재 로딩 중인 경우
    if _exaone_loading:
        raise RuntimeError("EXAONE 모델이 현재 로딩 중입니다.")

    _exaone_loading = True
    try:
        from app.service.model_service import load_exaone_model_for_service

        print("[INFO] EXAONE Reader 모델 로딩 시작...")
        _exaone_llm = load_exaone_model_for_service()
        print("[OK] EXAONE Reader 모델 로드 완료")

        _exaone_loading = False
        _exaone_error = None
        return _exaone_llm

    except Exception as e:
        _exaone_loading = False
        error_msg = f"EXAONE 모델 로드 실패: {str(e)}"
        _exaone_error = error_msg
        print(f"[ERROR] {error_msg}")

        import traceback

        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
        raise RuntimeError(error_msg) from e


def get_exaone_model():
    """이미 로드된 EXAONE 모델을 반환합니다.

    Returns:
        LangChain LLM 객체 또는 None
    """
    return _exaone_llm


def is_exaone_model_loaded() -> bool:
    """EXAONE 모델이 로드되었는지 확인합니다.

    Returns:
        로드 완료 여부
    """
    return _exaone_llm is not None
