"""Midm 모델 로더 서비스."""

import os
from pathlib import Path
from typing import Optional

from langchain_core.language_models import BaseChatModel

from .loader import resolve_model_path


def load_midm_model_for_service(
    model_path: Optional[str] = None,
    register: bool = False,
    clear_cache: bool = True,
) -> BaseChatModel:
    """서비스에서 사용할 Midm 모델을 로드합니다.

    Args:
        model_path: 모델 경로 (None이면 환경 변수 또는 기본 경로 사용)
        register: LLMFactory에 등록할지 여부
        clear_cache: GPU 캐시를 정리할지 여부

    Returns:
        LangChain 호환 모델

    Raises:
        RuntimeError: 모델 로드 실패 시
    """
    from .midm_model_loader import load_midm_model

    # GPU 메모리 정리
    if clear_cache:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # 모델 경로 해석
    if model_path is None:
        default_path = Path("app/model/midm")  # api/ 기준 상대 경로
        model_path = resolve_model_path("LOCAL_MODEL_DIR", default_path=default_path)

    # 모델 로드
    try:
        midm_model = load_midm_model(
            model_path=model_path,
            register=register,
            is_default=False,
        )
        langchain_model = midm_model.get_langchain_model()
        print("[OK] Midm 모델 서비스 초기화 완료")
        return langchain_model
    except Exception as e:
        error_msg = str(e)
        print(f"[WARNING] Midm 모델 로드 실패: {error_msg[:200]}...")
        raise RuntimeError(f"Midm 모델 로드 실패: {error_msg}") from e
