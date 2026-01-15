"""Midm 모델 로더 - 모델 초기화 및 등록."""

from pathlib import Path
from typing import Optional

from .factory import LLMFactory
from .midm_model import MidmLLM


def load_midm_model(
    model_path: Optional[str] = None,
    model_id: str = "K-intelligence/Midm-2.0-Mini-Instruct",
    register: bool = True,
    is_default: bool = False,
) -> MidmLLM:
    """Midm 모델 로드 및 등록.

    Args:
        model_path: 로컬 모델 경로 (None이면 현재 디렉토리 또는 model_id 사용)
        model_id: HuggingFace 모델 ID
        register: LLMFactory에 등록할지 여부
        is_default: 기본 모델로 설정할지 여부

    Returns:
        MidmLLM: 로드된 Midm 모델 인스턴스
    """
    # 모델 경로 결정
    if model_path is None:
        # api/app/model/midm 확인
        current_file = Path(__file__)
        app_dir = current_file.parent.parent.parent  # api/app
        midm_dir = app_dir / "model" / "midm"
        if midm_dir.exists() and (midm_dir / "config.json").exists():
            model_path = str(midm_dir)
        else:
            model_path = None  # model_id 사용

    # 모델 생성
    model = MidmLLM(
        model_path=model_path,
        model_id=model_id,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    # 등록
    if register:
        LLMFactory.register("midm", model, is_default=is_default)
        print(
            f"[OK] Midm 모델이 LLMFactory에 등록되었습니다. (기본 모델: {is_default})"
        )

    return model
