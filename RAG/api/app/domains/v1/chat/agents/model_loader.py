"""
MIDM 및 EXAONE 모델 로더.

서비스 레이어에서 사용할 모델 로딩 기능을 제공합니다.
"""

import sys
from pathlib import Path
from typing import Optional

import torch
from langchain_core.language_models import BaseChatModel

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
agents_dir = current_file.parent  # api/app/domains/chat/agents/
chat_dir = agents_dir.parent  # api/app/domains/chat/
domains_dir = chat_dir.parent  # api/app/domains/
app_dir = domains_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

sys.path.insert(0, str(api_dir))

from app.common.agents.utils import resolve_model_path

from .exaone_model import ExaoneLLM
from .midm_model import MidmLLM

# 전역 변수 (모델 캐싱용)
_midm_llm = None
_exaone_llm = None
_midm_loading = False
_exaone_loading = False
_midm_error = None
_exaone_error = None


def load_midm_model_for_service(
    model_path: Optional[str] = None,
    clear_cache: bool = True,
) -> BaseChatModel:
    """서비스에서 사용할 Midm 모델을 로드합니다.

    Args:
        model_path: 모델 경로 (None이면 환경 변수 또는 기본 경로 사용)
        clear_cache: GPU 캐시를 정리할지 여부

    Returns:
        LangChain 호환 모델

    Raises:
        RuntimeError: 모델 로드 실패 시
    """
    global _midm_llm, _midm_loading, _midm_error

    # 이미 로드된 경우 반환
    if _midm_llm is not None:
        return _midm_llm

    # 이전 에러가 있으면 재시도하지 않고 바로 에러 반환
    if _midm_error is not None:
        raise RuntimeError(f"이전 Midm 모델 로드 실패: {_midm_error}")

    if _midm_loading:
        raise RuntimeError("Midm 모델이 현재 로딩 중입니다. 잠시 후 다시 시도해주세요.")

    # GPU 메모리 정리
    if clear_cache:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    _midm_loading = True
    try:
        # 모델 경로 해석
        if model_path is None:
            default_path = Path("artifacts/midm/midm")  # api/ 기준 상대 경로
            model_path = resolve_model_path(
                "LOCAL_MODEL_DIR", default_path=default_path
            )

        # 모델 로드
        print(f"[INFO] Midm 모델 로딩 시작: {model_path or 'HuggingFace Hub'}")
        midm_model = MidmLLM(
            model_path=model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        _midm_llm = midm_model.get_langchain_model()
        print("[OK] Midm 모델 서비스 초기화 완료")
        _midm_loading = False
        _midm_error = None
        return _midm_llm

    except Exception as e:
        _midm_loading = False
        error_msg = str(e)
        _midm_error = error_msg
        print(f"[WARNING] Midm 모델 로드 실패: {error_msg[:200]}...")
        raise RuntimeError(f"Midm 모델 로드 실패: {error_msg}") from e


def load_exaone_model_for_service(
    model_path: Optional[str] = None,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = True,
    clear_cache: bool = True,
) -> BaseChatModel:
    """서비스에서 사용할 Exaone 모델을 로드합니다.

    Args:
        model_path: 모델 경로 (None이면 환경 변수 또는 기본 경로 사용)
        device_map: 디바이스 매핑 ("auto", "cpu", "cuda" 등)
        dtype: 토치 데이터 타입 ("auto", "float16", "float32" 등)
        trust_remote_code: 원격 코드 신뢰 여부
        clear_cache: GPU 캐시를 정리할지 여부

    Returns:
        LangChain 호환 모델

    Raises:
        RuntimeError: 모델 로드 실패 시
    """
    global _exaone_llm, _exaone_loading, _exaone_error

    # 이미 로드된 경우 반환
    if _exaone_llm is not None:
        return _exaone_llm

    # 이전 에러가 있으면 재시도하지 않고 바로 에러 반환
    if _exaone_error is not None:
        raise RuntimeError(f"이전 Exaone 모델 로드 실패: {_exaone_error}")

    if _exaone_loading:
        raise RuntimeError(
            "Exaone 모델이 현재 로딩 중입니다. 잠시 후 다시 시도해주세요."
        )

    # GPU 메모리 정리
    if clear_cache:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    _exaone_loading = True
    try:
        # 모델 경로 해석
        if model_path is None:
            # 실제 디렉토리 이름: exaone3.5-2.4b (하이픈 포함)
            # 실제 경로: artifacts/exaone/exaone3.5-2.4b
            default_path = Path(
                "artifacts/exaone/exaone3.5-2.4b"
            )  # api/ 기준 상대 경로
            model_path = resolve_model_path(
                "EXAONE_MODEL_DIR", default_path=default_path
            )

            if model_path is None:
                raise FileNotFoundError(
                    f"Exaone 모델을 찾을 수 없습니다. "
                    f"예상 경로: {default_path} 또는 EXAONE_MODEL_DIR 환경 변수를 설정하세요."
                )

        # 모델 로드
        print(f"[INFO] Exaone 모델 로딩 시작: {model_path}")
        exaone_model = ExaoneLLM(
            model_path=model_path,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        _exaone_llm = exaone_model.get_langchain_model()
        print("[OK] Exaone 모델 서비스 초기화 완료")
        _exaone_loading = False
        _exaone_error = None
        return _exaone_llm

    except Exception as e:
        _exaone_loading = False
        error_msg = str(e)
        _exaone_error = error_msg
        print(f"[WARNING] Exaone 모델 로드 실패: {error_msg[:200]}...")
        raise RuntimeError(f"Exaone 모델 로드 실패: {error_msg}") from e
