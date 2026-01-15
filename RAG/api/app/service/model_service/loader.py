"""모델 로딩 공통 유틸리티."""

import os
from pathlib import Path
from typing import Optional


def resolve_model_path(
    env_var_name: str,
    default_path: Optional[Path] = None,
) -> Optional[str]:
    """환경 변수 또는 기본 경로에서 모델 경로를 해석합니다.

    Args:
        env_var_name: 환경 변수 이름 (예: "LOCAL_MODEL_DIR", "EXAONE_MODEL_DIR")
        default_path: 기본 경로 (환경 변수가 없을 때 사용, api/ 기준 상대 경로)

    Returns:
        해석된 모델 경로 (문자열), 없으면 None

    Example:
        >>> path = resolve_model_path("LOCAL_MODEL_DIR", Path("app/model/midm"))
        >>> path = resolve_model_path("EXAONE_MODEL_DIR", Path("app/model/exaone3.5-2.4b"))
    """
    # 프로젝트 루트 (api/) 계산
    # api/app/service/model_service/loader.py 기준
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent  # api/

    # 환경 변수 확인
    env_path = os.getenv(env_var_name)
    if env_path:
        # 상대 경로를 절대 경로로 변환
        if not Path(env_path).is_absolute():
            env_path = str(project_root / env_path)
        return env_path

    # 기본 경로 확인
    if default_path:
        # 상대 경로인 경우 절대 경로로 변환 시도
        resolved = project_root / default_path
        if resolved.exists():
            return str(resolved)

    return None
