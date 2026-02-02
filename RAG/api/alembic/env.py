"""
Alembic 환경 설정
데이터베이스 마이그레이션을 위한 설정 파일
"""
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from sqlalchemy import text

from alembic import context

# app.core.database 설정 import
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
api_dir = current_file.parent.parent  # api/
project_root = api_dir.parent  # 프로젝트 루트
sys.path.insert(0, str(api_dir))

# .env 파일 로드 (프로젝트 루트에서)
try:
    from dotenv import load_dotenv
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()
except ImportError:
    pass  # python-dotenv가 없으면 환경 변수만 사용

# Base와 모든 모델 import (metadata에 등록하기 위해)
from app.domains.v10.shared.models.bases.base import Base
from app.domains.v10.soccer.models.bases import Player, Schedule, Stadium, Team

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_url():
    """데이터베이스 URL을 동적으로 가져옵니다."""
    from app.core.config.config import get_settings
    
    settings = get_settings()
    database_url = settings.connection_string
    
    if not database_url:
        raise ValueError("DATABASE_URL이 설정되지 않았습니다.")
    
    # Alembic은 동기 엔진을 사용하므로 asyncpg를 psycopg2로 변환
    if database_url.startswith("postgresql+asyncpg://"):
        database_url = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    elif database_url.startswith("postgresql://"):
        pass  # 이미 동기 형식
    
    return database_url


def include_object(object, name, type_, reflected, compare_to):
    """기존 테이블 삭제를 방지하는 함수.
    
    Alembic autogenerate가 모델에 없는 기존 테이블을 삭제하지 않도록 합니다.
    """
    # 기존 테이블 목록 (삭제하지 않을 테이블들)
    protected_tables = {
        'users',
        'refresh_tokens',
        'gri_standards',
        'langchain_pg_collection',
        'langchain_pg_embedding',
        'alembic_version',  # Alembic 자체 테이블
    }
    
    # 테이블 삭제를 방지
    if type_ == "table" and name in protected_tables:
        return False
    
    return True


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,  # 기존 테이블 보호
        # Neon/role 환경에서 search_path가 달라져도 버전 테이블은 public로 고정
        version_table="alembic_version",
        version_table_schema="public",
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        # search_path를 public로 고정 (테이블이 다른 스키마에 생기는 문제 방지)
        # - Neon/role 설정에 따라 기본 search_path가 "$user", public 처럼 바뀌어
        #   create_table이 public이 아닌 스키마로 들어갈 수 있습니다.
        connection.execute(text("SET search_path TO public"))

        # pgvector extension 보장 (임베딩 테이블에서 vector 타입 사용)
        # - 실패 시 조용히 넘어가면 이후 마이그레이션이 "적용된 것처럼 보이지만 실제로는 롤백"
        #   되는 케이스를 추적하기 어렵습니다. 실패하면 그대로 에러를 노출합니다.
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # 위 2개 실행으로 SQLAlchemy가 트랜잭션을 autobegin 할 수 있습니다.
        # 이 상태로 Alembic이 별도 트랜잭션을 시작하지 못하면(이미 트랜잭션이 열려있다고 판단)
        # 마이그레이션 전체가 커밋되지 않고 커넥션 종료 시 롤백되는 현상이 발생할 수 있어,
        # 여기서 명시적으로 커밋하여 트랜잭션을 닫습니다.
        try:
            connection.commit()
        except Exception:
            pass

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,  # 기존 테이블 보호
            # Neon/role 환경에서 search_path가 달라져도 버전 테이블은 public로 고정
            version_table="alembic_version",
            version_table_schema="public",
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
