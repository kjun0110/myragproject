"""
EXAONE 모델 및 어댑터 로더.

EXAONE 모델과 LoRA 어댑터를 로드하고 전역 캐싱을 관리합니다.
"""

import sys
from pathlib import Path
from typing import Optional

from langchain_core.language_models import BaseChatModel

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
agents_dir = current_file.parent  # api/app/domains/spam_classifier/agents/
spam_classifier_dir = agents_dir.parent  # api/app/domains/spam_classifier/
domains_dir = spam_classifier_dir.parent  # api/app/domains/
app_dir = domains_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

sys.path.insert(0, str(api_dir))

from app.common.agents.utils import resolve_model_path

# 전역 변수 (모델 캐싱용)
_exaone_llm = None
_exaone_loading = False
_exaone_error = None


def load_exaone_model(
    model_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = True,
) -> BaseChatModel:
    """EXAONE 모델을 로드합니다.

    전역 캐싱을 사용하여 한 번만 로드합니다.
    LoRA 어댑터가 제공되면 함께 로드합니다.

    Args:
        model_path: 모델 경로 (None이면 환경 변수 또는 기본 경로 사용)
        adapter_path: LoRA 어댑터 경로 (None이면 어댑터 없이 로드)
        device_map: 디바이스 매핑 ("auto", "cpu", "cuda" 등)
        dtype: 토치 데이터 타입 ("auto", "float16", "float32" 등)
        trust_remote_code: 원격 코드 신뢰 여부

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
        print("[INFO] EXAONE Reader 모델 로딩 시작...")

        # HuggingFace 캐시 사용 (로컬 경로 검색 제거)
        # model_path가 명시적으로 제공되지 않으면 None으로 설정하여
        # ExaoneLLM이 HuggingFace 모델 ID를 사용하도록 함
        if model_path is None:
            print("[INFO] HuggingFace 캐시에서 EXAONE 모델 로드 (로컬 경로 검색 건너뛰기)")
            model_path = None  # None으로 설정하면 ExaoneLLM이 model_id 사용

        # 어댑터 경로 해석 (없으면 자동 탐색)
        if adapter_path is None:
            # 어댑터 자동 탐색: exaone3.5-2.4b-spam-lora 폴더에서 최신 타임스탬프 폴더 찾기
            # 실제 경로: artifacts/exaone/spam_adapter/exaone3.5-2.4b-spam-lora
            adapter_dir = (
                api_dir
                / "artifacts"
                / "exaone"
                / "spam_adapter"
                / "exaone3.5-2.4b-spam-lora"
            )
            if adapter_dir.exists():
                subdirs = [d for d in adapter_dir.iterdir() if d.is_dir()]
                if subdirs:
                    adapter_path = str(max(subdirs, key=lambda x: x.stat().st_mtime))
                    print(f"[INFO] 자동 탐색된 어댑터: {adapter_path}")
                else:
                    adapter_path = str(adapter_dir)
                    print(f"[INFO] 어댑터 디렉토리 사용: {adapter_path}")
            else:
                print("[INFO] LoRA 어댑터를 찾을 수 없습니다. 기본 모델만 로드합니다.")

        # EXAONE 모델 구현체 로드
        from .exaone_model import ExaoneLLM

        # model_path=None이면 ExaoneLLM이 HuggingFace 모델 ID 사용
        print(f"[INFO] Exaone 모델 로딩 시작 (HuggingFace 캐시 사용)")
        exaone_model = ExaoneLLM(
            model_path=model_path,  # None이면 HuggingFace 모델 ID 사용
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            adapter_path=adapter_path,  # 어댑터 경로 전달
        )

        langchain_model = exaone_model.get_langchain_model()
        _exaone_llm = langchain_model

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


def get_exaone_model() -> Optional[BaseChatModel]:
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
