"""Exaone 모델 로더 서비스."""

from pathlib import Path
from typing import Optional

from langchain_core.language_models import BaseChatModel

from .loader import resolve_model_path


def load_exaone_model_for_service(
    model_path: Optional[str] = None,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = True,
) -> BaseChatModel:
    """서비스에서 사용할 Exaone 모델을 로드합니다.

    Args:
        model_path: 모델 경로 (None이면 환경 변수 또는 기본 경로 사용)
        device_map: 디바이스 매핑 ("auto", "cpu", "cuda" 등)
        dtype: 토치 데이터 타입 ("auto", "float16", "float32" 등)
        trust_remote_code: 원격 코드 신뢰 여부

    Returns:
        LangChain 호환 모델

    Raises:
        RuntimeError: 모델 로드 실패 시
    """
    from .exaone_model import ExaoneLLM

    # 모델 경로 해석
    if model_path is None:
        # 실제 디렉토리 이름: exaone3.5-2.4b (하이픈 포함)
        default_path = Path("app/model/exaone3.5-2.4b")  # api/ 기준 상대 경로
        model_path = resolve_model_path("EXAONE_MODEL_DIR", default_path=default_path)

        if model_path is None:
            raise FileNotFoundError(
                f"Exaone 모델을 찾을 수 없습니다. "
                f"예상 경로: {default_path} 또는 EXAONE_MODEL_DIR 환경 변수를 설정하세요."
            )

    # 모델 로드
    try:
        print(f"[INFO] Exaone 모델 로딩 시작: {model_path}")
        exaone_model = ExaoneLLM(
            model_path=model_path,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        langchain_model = exaone_model.get_langchain_model()
        print("[OK] Exaone 모델 서비스 초기화 완료")
        return langchain_model
    except Exception as e:
        error_msg = str(e)
        print(f"[WARNING] Exaone 모델 로드 실패: {error_msg[:200]}...")
        raise RuntimeError(f"Exaone 모델 로드 실패: {error_msg}") from e
