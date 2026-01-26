"""
EXAONE 모델 로더.

서비스 레이어에서 사용할 모델 로딩 기능을 제공합니다.
HuggingFace 캐시를 활용하여 EXAONE 모델을 로드합니다.
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

from .exaone_model import ExaoneLLM

# 전역 변수 (모델 캐싱용)
_midm_llm = None  # 호환성을 위해 유지 (실제로는 EXAONE 사용)
_exaone_llm = None
_midm_loading = False
_exaone_loading = False
_midm_error = None
_exaone_error = None


def load_midm_model_for_service(
    model_path: Optional[str] = None,
    clear_cache: bool = True,
) -> BaseChatModel:
    """서비스에서 사용할 로컬 모델을 로드합니다.
    
    주의: 이 함수는 이제 EXAONE 모델을 사용합니다 (midm 대신).
    호환성을 위해 함수명은 유지되었습니다.

    Args:
        model_path: 모델 경로 (None이면 HuggingFace 캐시 사용)
        clear_cache: GPU 캐시를 정리할지 여부

    Returns:
        LangChain 호환 모델

    Raises:
        RuntimeError: 모델 로드 실패 시
    """
    # EXAONE 모델 로더로 위임
    return load_exaone_model_for_service(
        model_path=None,  # HuggingFace 캐시 사용
        device_map="auto",
        dtype="auto",
        trust_remote_code=True,
        clear_cache=clear_cache,
    )


def load_exaone_model_for_service(
    model_path: Optional[str] = None,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = True,
    clear_cache: bool = True,
) -> BaseChatModel:
    """서비스에서 사용할 Exaone 모델을 로드합니다.
    
    HuggingFace 캐시를 활용하여 EXAONE 모델을 로드합니다.

    Args:
        model_path: 모델 경로 (None이면 HuggingFace 캐시 사용)
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
        # HuggingFace 캐시 사용 (model_path는 무시하고 항상 캐시에서 로드)
        print(f"[INFO] Exaone 모델 로딩 시작 (HuggingFace 캐시 사용)")
        exaone_model = ExaoneLLM(
            model_path=None,  # None이면 HuggingFace 캐시 사용
            model_id="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
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
