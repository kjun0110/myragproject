"""애플리케이션 설정 및 환경 변수 초기화.

이 모듈은 Pydantic BaseSettings를 사용하여 환경 변수와 .env 파일에서
설정을 자동으로 로드합니다.
"""

import os
from typing import Optional

# ============================================================================
# Pydantic BaseSettings Import
# ============================================================================

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore

    _USE_V2 = True
except ImportError:
    try:
        from pydantic import BaseSettings  # type: ignore

        _USE_V2 = False
    except ImportError:
        raise ImportError(
            "pydantic 또는 pydantic-settings가 필요합니다. "
            "pip install pydantic 또는 pip install pydantic-settings를 실행하세요."
        )


# ============================================================================
# HuggingFace Hub 환경 변수 설정
# ============================================================================

# Windows에서 심볼릭 링크 생성 시 관리자 권한이 필요하므로 비활성화
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# HuggingFace Hub 원격 코드 신뢰 설정
os.environ.setdefault("HF_HUB_TRUST_REMOTE_CODE", "true")


# ============================================================================
# Settings 클래스
# ============================================================================


class Settings(BaseSettings):
    """애플리케이션 설정 클래스.

    환경 변수 또는 .env 파일에서 설정을 자동으로 로드합니다.
    """

    # Pydantic 설정 (v1/v2 호환)
    if _USE_V2:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )
    else:

        class Config:  # type: ignore
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False
            extra = "ignore"

    # ------------------------------------------------------------------------
    # 데이터베이스 설정
    # ------------------------------------------------------------------------

    # Neon DB 또는 기타 PostgreSQL 연결 문자열 (우선 사용)
    database_url: Optional[str] = None

    # 개별 데이터베이스 설정 (database_url이 없을 때 사용)
    postgres_host: str = "localhost"
    postgres_port: str = "5432"
    postgres_user: str = "langchain"
    postgres_password: str = "langchain123"
    postgres_db: str = "langchain_db"

    # ------------------------------------------------------------------------
    # 모델 설정
    # ------------------------------------------------------------------------

    default_llm_model: Optional[str] = None
    default_embedding_model: Optional[str] = None

    # ------------------------------------------------------------------------
    # OpenAI 설정
    # ------------------------------------------------------------------------

    openai_api_key: Optional[str] = None

    # ------------------------------------------------------------------------
    # Computed Properties
    # ------------------------------------------------------------------------

    @property
    def connection_string(self) -> str:
        """PostgreSQL 연결 문자열을 반환합니다.

        DATABASE_URL이 설정되어 있으면 우선 사용하고,
        없으면 개별 설정으로부터 연결 문자열을 생성합니다.

        Returns:
            PostgreSQL 연결 문자열 (postgresql://user:password@host:port/db)
        """
        # DATABASE_URL이 있으면 우선 사용
        if self.database_url:
            conn_str = self.database_url.strip()
            # "psql" 접두사가 있으면 제거 (일부 환경에서 붙을 수 있음)
            if conn_str.startswith("psql "):
                conn_str = conn_str[5:].strip()
            return conn_str

        # 개별 설정으로부터 연결 문자열 생성
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def collection_name(self) -> str:
        """벡터 컬렉션 이름을 반환합니다.

        Returns:
            벡터 컬렉션 이름
        """
        return "langchain_collection"


# ============================================================================
# 싱글톤 패턴
# ============================================================================

_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """설정 싱글톤 인스턴스를 반환합니다.

    Returns:
        Settings 인스턴스 (싱글톤)
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

