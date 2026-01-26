# Alembic 데이터베이스 마이그레이션 전략

## 개요

이 문서는 `backend/app/main.py` 실행 시 Alembic을 사용하여 Soccer 테이블들을 Neon PostgreSQL 데이터베이스에 자동으로 생성하는 전체 프로세스를 설명합니다.

---

## 1. 전체 프로세스 흐름

```
main.py 실행
    ↓
lifespan 함수 시작
    ↓
.env 파일 로드 (DATABASE_URL)
    ↓
Alembic 설정 로드 (alembic.ini)
    ↓
Alembic env.py 실행 (모델 import)
    ↓
마이그레이션 스크립트 생성 (필요시)
    ↓
마이그레이션 적용 (alembic upgrade head)
    ↓
Soccer 테이블 생성 (player, schedule, stadium, team)
    ↓
서버 시작
```

---

## 2. 핵심 파일 및 역할

### 2.1 `backend/app/main.py`
**역할**: FastAPI 서버 진입점 및 Alembic 자동 실행

**주요 코드 섹션**:

```python
@asynccontextmanager
async def lifespan(app):
    """애플리케이션 생명주기 관리."""
    
    # 1. 설정 로드
    settings = get_settings()  # DATABASE_URL 포함
    
    # 2. Alembic 설정 로드
    alembic_ini_path = backend_dir / "alembic.ini"
    alembic_cfg = Config(str(alembic_ini_path))
    
    # 3. script_location 절대 경로 설정
    alembic_dir = backend_dir / "alembic"
    alembic_cfg.set_main_option("script_location", str(alembic_dir))
    
    # 4. 데이터베이스 URL 설정 (asyncpg -> psycopg2 변환)
    database_url = settings.connection_string
    if database_url.startswith("postgresql+asyncpg://"):
        database_url = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    
    # 5. Soccer 모델 import (metadata에 등록)
    from app.domains.v10.soccer.bases import Player, Schedule, Stadium, Team
    
    # 6. 마이그레이션 파일 존재 여부 확인
    versions_dir = alembic_dir / "versions"
    existing_migrations = [파일 목록]
    
    # 7. 마이그레이션 파일이 없으면 자동 생성 (무한 루프 방지)
    if not existing_migrations:
        command.revision(alembic_cfg, autogenerate=True, message="...")
    
    # 8. 마이그레이션 적용
    command.upgrade(alembic_cfg, "head")
```

**중요 포인트**:
- `backend_dir` 변수: `backend/app/main.py` 파일 기준으로 `backend/` 디렉토리 경로
- `asyncpg` → `psycopg2` 변환: Alembic은 동기 엔진 사용
- 무한 루프 방지: 마이그레이션 파일이 이미 있으면 생성하지 않음

---

### 2.2 `backend/alembic.ini`
**역할**: Alembic 설정 파일 (CLI 및 프로그래밍 방식 모두 사용)

**주요 설정**:

```ini
[alembic]
# 마이그레이션 스크립트 위치
script_location = %(here)s/alembic

# Python 경로 추가
prepend_sys_path = .

# 데이터베이스 URL (env.py에서 동적으로 설정됨)
# sqlalchemy.url은 주석 처리 (env.py에서 동적 설정)
```

**중요 포인트**:
- `%(here)s/alembic`: `alembic.ini` 파일이 있는 디렉토리 기준 상대 경로
- `sqlalchemy.url`은 주석 처리: `env.py`에서 `.env` 파일의 `DATABASE_URL` 사용
- 인코딩 주의: UTF-8로 저장, 한글 주석 사용 시 cp949 오류 발생 가능

---

### 2.3 `backend/alembic/env.py`
**역할**: Alembic 환경 설정 및 마이그레이션 실행 로직

**주요 코드 섹션**:

```python
# 1. 프로젝트 경로 설정
current_file = Path(__file__).resolve()
backend_dir = current_file.parent.parent  # backend/
project_root = backend_dir.parent  # 프로젝트 루트
sys.path.insert(0, str(backend_dir))

# 2. .env 파일 로드
from dotenv import load_dotenv
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)

# 3. Base와 모든 모델 import (metadata에 등록)
from app.domains.v10.shared.bases.base import Base
from app.domains.v10.soccer.bases import Player, Schedule, Stadium, Team

# 4. metadata 설정
target_metadata = Base.metadata

# 5. 데이터베이스 URL 동적 가져오기
def get_url():
    from app.common.config.config import get_settings
    settings = get_settings()
    database_url = settings.connection_string
    
    # asyncpg -> psycopg2 변환
    if database_url.startswith("postgresql+asyncpg://"):
        database_url = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    
    return database_url

# 6. 기존 테이블 보호 (삭제 방지)
def include_object(object, name, type_, reflected, compare_to):
    protected_tables = {
        'users', 'refresh_tokens', 'gri_standards',
        'langchain_pg_collection', 'langchain_pg_embedding',
        'alembic_version',
    }
    
    # 보호된 테이블은 삭제하지 않음
    if type_ == "table" and name in protected_tables:
        return False
    
    return True

# 7. 마이그레이션 실행
def run_migrations_online():
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    
    connectable = engine_from_config(configuration, ...)
    
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,  # 기존 테이블 보호
        )
        
        with context.begin_transaction():
            context.run_migrations()
```

**중요 포인트**:
- `.env` 파일 로드: `DATABASE_URL` 환경 변수 사용
- 모델 import: `Base.metadata`에 등록되어야 Alembic이 인식
- `include_object`: 기존 테이블 삭제 방지 (중요!)
- `get_url()`: 동적으로 데이터베이스 URL 가져오기

---

### 2.4 `backend/alembic/script.py.mako`
**역할**: 마이그레이션 스크립트 템플릿

**구조**:
```python
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision}
Create Date: ${create_date}
"""

def upgrade() -> None:
    ${upgrades if upgrades else "pass"}

def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

**중요 포인트**:
- `upgrade()`: 마이그레이션 적용 (테이블 생성/수정)
- `downgrade()`: 마이그레이션 롤백 (테이블 삭제/복구)
- Alembic이 자동으로 채워줌

---

### 2.5 `backend/alembic/versions/*.py`
**역할**: 실제 마이그레이션 스크립트 (자동 생성됨)

**예시**: `6a639c5fd25e_auto_generate_soccer_tables.py`

```python
def upgrade() -> None:
    # Soccer 테이블 생성
    op.create_table('player',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('team_id', sa.BigInteger(), nullable=True),
        # ... 모든 컬럼
        sa.PrimaryKeyConstraint('id')
    )
    # schedule, stadium, team도 동일하게 생성
    
    # 기존 테이블 삭제 (문제 발생 원인!)
    op.drop_table('langchain_pg_embedding')
    op.drop_table('gri_standards')
    # ...

def downgrade() -> None:
    # 롤백 시 기존 테이블 복구
    op.create_table('refresh_tokens', ...)
    op.create_table('users', ...)
    # ...
    
    # Soccer 테이블 삭제
    op.drop_table('team')
    op.drop_table('stadium')
    op.drop_table('schedule')
    op.drop_table('player')
```

**문제점**:
- 첫 번째 마이그레이션에서 기존 테이블 삭제
- `include_object` 함수가 없었기 때문

---

### 2.6 Soccer 모델 파일들

#### `backend/app/domains/v10/soccer/bases/players.py`
```python
from sqlalchemy import Column, String, Integer, Date, BigInteger, ForeignKey
from app.domains.v10.shared.bases.base import Base

class Player(Base):
    __tablename__ = "player"
    
    id = Column(BigInteger, primary_key=True, nullable=False)
    team_id = Column(BigInteger, nullable=True)  # FK -> team.id
    team_code = Column(String, nullable=False)
    player_name = Column(String, nullable=False)
    # ... 기타 컬럼들
```

#### `backend/app/domains/v10/soccer/bases/schedules.py`
```python
class Schedule(Base):
    __tablename__ = "schedule"
    
    id = Column(BigInteger, primary_key=True, nullable=False)
    stadium_id = Column(BigInteger, nullable=True)  # FK -> stadium.id
    hometeam_id = Column(BigInteger, nullable=True)  # FK -> team.id
    awayteam_id = Column(BigInteger, nullable=True)  # FK -> team.id
    # ... 기타 컬럼들
```

#### `backend/app/domains/v10/soccer/bases/stadiums.py`
```python
class Stadium(Base):
    __tablename__ = "stadium"
    
    id = Column(BigInteger, primary_key=True, nullable=False)
    stadium_code = Column(String, nullable=False)
    statdium_name = Column(String, nullable=False)
    # ... 기타 컬럼들
```

#### `backend/app/domains/v10/soccer/bases/teams.py`
```python
class Team(Base):
    __tablename__ = "team"
    
    id = Column(BigInteger, primary_key=True, nullable=False)
    stadium_id = Column(BigInteger, nullable=True)  # FK -> stadium.id
    team_code = Column(String, nullable=False)
    # ... 기타 컬럼들
```

**중요 포인트**:
- 모든 모델이 `Base`를 상속받음
- `__tablename__`: 실제 데이터베이스 테이블 이름
- Foreign Key: `team_id`, `stadium_id` 등 (관계형 구조)
- Alembic이 이 모델들을 읽어 마이그레이션 스크립트 생성

---

### 2.7 `backend/app/domains/v10/shared/bases/base.py`
**역할**: 모든 모델의 기본 클래스 제공

```python
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    """SQLAlchemy Base 클래스."""
    pass
```

**중요 포인트**:
- SQLAlchemy 2.0+ 스타일
- `Base.metadata`: 모든 모델의 메타데이터 저장
- Alembic이 이 metadata를 읽어 테이블 구조 파악

---

### 2.8 `backend/app/core/database/session.py`
**역할**: 데이터베이스 세션 관리 (FastAPI 의존성 주입용)

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

def create_database_engine():
    database_url = settings.connection_string
    
    # PostgreSQL -> asyncpg 변환 (비동기 엔진용)
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    return create_async_engine(database_url, ...)

# 전역 엔진 및 세션 팩토리
engine = create_database_engine()
AsyncSessionLocal = async_sessionmaker(engine, ...)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 의존성 주입용 세션"""
    async with AsyncSessionLocal() as session:
        yield session
```

**중요 포인트**:
- 비동기 엔진 (`asyncpg`)
- Alembic과는 별개 (Alembic은 동기 엔진 사용)
- FastAPI 라우터에서 사용

---

### 2.9 `backend/app/common/config/config.py`
**역할**: 환경 변수 및 설정 관리

```python
class Settings(BaseSettings):
    database_url: Optional[str] = None  # .env의 DATABASE_URL
    
    @property
    def connection_string(self) -> str:
        """데이터베이스 연결 문자열 반환"""
        if self.database_url:
            return self.database_url
        # fallback 로직...

def get_settings() -> Settings:
    """싱글톤 패턴으로 설정 반환"""
    return Settings()
```

**중요 포인트**:
- `.env` 파일에서 `DATABASE_URL` 자동 로드
- `connection_string` 속성으로 URL 제공
- `main.py`, `env.py` 모두 이 설정 사용

---

## 3. 데이터베이스 연결 URL 처리

### 3.1 URL 형식 차이

| 용도 | 드라이버 | URL 형식 |
|------|---------|---------|
| Alembic (동기) | psycopg2 | `postgresql://user:pass@host/db` |
| SQLAlchemy (비동기) | asyncpg | `postgresql+asyncpg://user:pass@host/db` |

### 3.2 변환 로직

**main.py에서**:
```python
database_url = settings.connection_string
if database_url.startswith("postgresql+asyncpg://"):
    database_url = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
```

**session.py에서**:
```python
database_url = settings.connection_string
if database_url.startswith("postgresql://"):
    database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
```

**중요 포인트**:
- Alembic: `postgresql://` (동기)
- FastAPI/SQLAlchemy: `postgresql+asyncpg://` (비동기)
- 같은 데이터베이스, 다른 드라이버

---

## 4. Alembic 마이그레이션 생성 및 적용 과정

### 4.1 마이그레이션 생성 (`command.revision`)

```python
command.revision(alembic_cfg, autogenerate=True, message="Auto-generate soccer tables")
```

**동작 과정**:
1. `env.py` 실행 → Soccer 모델 import
2. `Base.metadata` 읽기 → 모델 정의 파악
3. 데이터베이스 현재 스키마 읽기
4. 차이점 비교 (모델 vs DB)
5. 마이그레이션 스크립트 생성 (`versions/*.py`)

**생성되는 내용**:
- `upgrade()`: 모델에 있지만 DB에 없는 테이블 생성
- `downgrade()`: 롤백 시 복구 로직
- **문제**: 모델에 없지만 DB에 있는 테이블 삭제 (기본 동작)

### 4.2 마이그레이션 적용 (`command.upgrade`)

```python
command.upgrade(alembic_cfg, "head")
```

**동작 과정**:
1. `alembic_version` 테이블 확인 (현재 리비전)
2. 적용되지 않은 마이그레이션 찾기
3. `upgrade()` 함수 순차 실행
4. `alembic_version` 테이블 업데이트

**결과**:
- Soccer 테이블 생성 (`player`, `schedule`, `stadium`, `team`)
- Primary Key, Foreign Key 설정
- 인덱스 생성

---

## 5. 기존 테이블 보호 전략

### 5.1 문제 상황
첫 번째 마이그레이션에서 기존 테이블들이 삭제됨:
- `users`, `refresh_tokens`, `gri_standards`
- `langchain_pg_collection`, `langchain_pg_embedding`

### 5.2 원인
Alembic의 `autogenerate`는:
- 모델에 정의된 테이블 → 생성/수정
- **모델에 없지만 DB에 있는 테이블 → 삭제** (기본 동작)

Soccer 모델만 import했기 때문에 다른 테이블이 삭제 대상으로 인식됨.

### 5.3 해결 방법: `include_object` 함수

```python
def include_object(object, name, type_, reflected, compare_to):
    """기존 테이블 삭제를 방지하는 함수."""
    protected_tables = {
        'users', 'refresh_tokens', 'gri_standards',
        'langchain_pg_collection', 'langchain_pg_embedding',
        'alembic_version',
    }
    
    # 보호된 테이블은 마이그레이션에서 제외
    if type_ == "table" and name in protected_tables:
        return False
    
    return True
```

**적용 위치**:
```python
context.configure(
    connection=connection,
    target_metadata=target_metadata,
    include_object=include_object,  # 추가!
)
```

---

## 6. 무한 루프 방지 전략

### 6.1 문제 상황
1. `command.revision()` 실행 → 마이그레이션 파일 생성
2. watchfiles가 파일 변경 감지 → 서버 리로드
3. `lifespan` 재실행 → 또 `command.revision()` 실행
4. 무한 반복...

### 6.2 해결 방법: 조건부 생성

```python
# versions 디렉토리에 마이그레이션 파일이 있는지 확인
versions_dir = alembic_dir / "versions"
existing_migrations = [
    f for f in versions_dir.glob("*.py") 
    if f.name != "__init__.py" and f.name != ".gitkeep"
] if versions_dir.exists() else []

# 마이그레이션 파일이 없을 때만 자동 생성
if not existing_migrations:
    command.revision(alembic_cfg, autogenerate=True, message="...")
else:
    logger.info(f"기존 마이그레이션 {len(existing_migrations)}개 발견 - 자동 생성 스킵")
```

**중요 포인트**:
- 첫 실행: 마이그레이션 생성 → 적용
- 이후 실행: 생성 스킵 → 기존 마이그레이션만 적용
- 모델 변경 시: 수동으로 `alembic revision --autogenerate` 실행

---

## 7. 데이터베이스 테이블 구조

### 7.1 Soccer 테이블들 (관계형 DB)

```
┌─────────────┐
│   stadium   │
│─────────────│
│ id (PK)     │
│ stadium_code│
│ ...         │
└─────────────┘
       ↑
       │ stadium_id (FK)
       │
┌─────────────┐
│    team     │
│─────────────│
│ id (PK)     │
│ stadium_id  │ ← FK
│ team_code   │
│ ...         │
└─────────────┘
       ↑
       │ team_id (FK)
       │
┌─────────────┐
│   player    │
│─────────────│
│ id (PK)     │
│ team_id     │ ← FK
│ player_name │
│ ...         │
└─────────────┘

┌─────────────┐
│  schedule   │
│─────────────│
│ id (PK)     │
│ stadium_id  │ ← FK
│ hometeam_id │ ← FK
│ awayteam_id │ ← FK
│ sche_date   │
│ ...         │
└─────────────┘
```

**특징**:
- 일반적인 관계형 데이터베이스 구조
- Primary Key (id)
- Foreign Key 관계
- 벡터 컬럼 없음

### 7.2 벡터 스토어 테이블들 (별도)

```
┌──────────────────────────┐
│ langchain_pg_collection  │
│──────────────────────────│
│ uuid (PK)                │
│ name                     │
│ cmetadata                │
└──────────────────────────┘
       ↑
       │ collection_id (FK)
       │
┌──────────────────────────┐
│ langchain_pg_embedding   │
│──────────────────────────│
│ uuid (PK)                │
│ collection_id            │ ← FK
│ embedding (vector)       │ ← 벡터 데이터!
│ document                 │
│ cmetadata                │
└──────────────────────────┘
```

**특징**:
- LangChain PGVector 전용
- `embedding` 컬럼: pgvector 타입
- Soccer 테이블과 독립적

---

## 8. 실행 흐름 상세 분석

### 8.1 서버 시작 시 (`main.py` 실행)

```
1. Python 경로 설정
   - sys.path.insert(0, str(backend_dir))
   
2. .env 파일 로드
   - load_dotenv(project_root / ".env")
   - DATABASE_URL 환경 변수 로드
   
3. 로깅 설정
   - logging.basicConfig(...)
   
4. FastAPI 앱 생성
   - app = FastAPI(lifespan=lifespan)
   
5. lifespan 함수 실행 (서버 시작 시)
   ↓
```

### 8.2 lifespan 함수 내부

```
1. 설정 로드
   - settings = get_settings()
   - settings.connection_string 확인
   
2. 데이터베이스 연결 테스트
   - psycopg2.connect(settings.connection_string)
   - 연결 성공 확인
   
3. Alembic 설정 로드
   - alembic_cfg = Config("alembic.ini")
   - script_location 설정
   - sqlalchemy.url 설정
   
4. Soccer 모델 import
   - from app.domains.v10.soccer.bases import Player, Schedule, Stadium, Team
   - Base.metadata에 등록됨
   
5. 마이그레이션 파일 확인
   - versions/*.py 파일 개수 확인
   
6. 마이그레이션 생성 (조건부)
   - if not existing_migrations:
       command.revision(alembic_cfg, autogenerate=True)
   
7. 마이그레이션 적용
   - command.upgrade(alembic_cfg, "head")
   - env.py 실행 → upgrade() 함수 실행
   
8. 서버 초기화 완료
```

### 8.3 Alembic env.py 실행 시

```
1. Python 경로 설정
   - sys.path.insert(0, str(backend_dir))
   
2. .env 파일 로드
   - load_dotenv(project_root / ".env")
   
3. 모델 import
   - from app.domains.v10.shared.bases.base import Base
   - from app.domains.v10.soccer.bases import Player, Schedule, Stadium, Team
   
4. metadata 설정
   - target_metadata = Base.metadata
   
5. 데이터베이스 URL 가져오기
   - get_url() 함수 실행
   - settings.connection_string 읽기
   - asyncpg -> psycopg2 변환
   
6. 마이그레이션 실행
   - run_migrations_online() 함수
   - 데이터베이스 연결
   - context.configure(include_object=include_object)
   - context.run_migrations()
   
7. upgrade() 함수 실행
   - op.create_table('player', ...)
   - op.create_table('schedule', ...)
   - op.create_table('stadium', ...)
   - op.create_table('team', ...)
```

---

## 9. 주요 오류 및 해결 방법

### 9.1 "Path doesn't exist: alembic" 오류

**원인**:
- Alembic이 `script_location`을 찾지 못함
- 작업 디렉토리가 `backend/` 가 아닌 다른 곳

**해결**:
```python
# main.py에서 절대 경로 설정
alembic_cfg.set_main_option("script_location", str(alembic_dir))
```

```ini
# alembic.ini에서
script_location = %(here)s/alembic
```

### 9.2 "connection to localhost:5432 refused" 오류

**원인**:
- `alembic.ini`에 하드코딩된 `sqlalchemy.url`
- `env.py`의 동적 URL 설정이 무시됨

**해결**:
```ini
# alembic.ini에서 주석 처리
# sqlalchemy.url is set dynamically in env.py
# sqlalchemy.url = driver://user:pass@localhost/dbname
```

```python
# env.py에서 동적 설정
def get_url():
    settings = get_settings()
    return settings.connection_string
```

### 9.3 "UnicodeDecodeError: cp949" 오류

**원인**:
- `alembic.ini`에 한글 주석
- Windows에서 cp949 인코딩으로 읽으려 함

**해결**:
- 한글 주석을 영어로 변경
- 파일을 UTF-8로 저장

### 9.4 기존 테이블 삭제 문제

**원인**:
- Alembic autogenerate가 모델에 없는 테이블을 삭제 대상으로 인식

**해결**:
```python
# env.py에 include_object 함수 추가
def include_object(object, name, type_, reflected, compare_to):
    protected_tables = {'users', 'refresh_tokens', ...}
    if type_ == "table" and name in protected_tables:
        return False
    return True

# context.configure에 추가
context.configure(..., include_object=include_object)
```

### 9.5 무한 루프 (마이그레이션 반복 생성)

**원인**:
- 마이그레이션 파일 생성 → watchfiles 감지 → 리로드 → 다시 생성

**해결**:
```python
# 마이그레이션 파일이 이미 있으면 생성하지 않음
if not existing_migrations:
    command.revision(...)
```

---

## 10. 모델 변경 시 마이그레이션 생성 방법

### 10.1 수동 생성 (권장)

```bash
cd backend
alembic revision --autogenerate -m "Add new column to player"
alembic upgrade head
```

### 10.2 자동 생성 (개발 환경)

현재 `main.py`는 마이그레이션 파일이 없을 때만 자동 생성합니다.
모델 변경 후 자동 생성하려면:

1. 기존 마이그레이션 파일 삭제 (주의!)
2. 서버 재시작
3. 또는 수동 생성 사용 (`cd backend && alembic revision --autogenerate`)

---

## 11. 디렉토리 구조

```
backend/
├── alembic.ini                          # Alembic 설정 파일
├── alembic/                             # Alembic 디렉토리
│   ├── env.py                          # 환경 설정 (모델 import)
│   ├── script.py.mako                  # 마이그레이션 템플릿
│   └── versions/                       # 마이그레이션 스크립트
│       ├── .gitkeep
│       ├── 6a639c5fd25e_*.py          # 첫 번째 마이그레이션
│       ├── cb807d95ba42_*.py          # 두 번째 마이그레이션
│       └── ...
├── app/
│   ├── main.py                         # FastAPI 진입점 (Alembic 자동 실행)
│   ├── common/
│   │   ├── config/
│   │   │   └── config.py              # 설정 관리 (DATABASE_URL)
│   │   └── database/
│   │       └── vector_store.py        # 벡터 스토어 (별도)
│   ├── core/
│   │   └── database/                   # SQLAlchemy 설정
│   │       ├── __init__.py            # 패키지 초기화
│   │       ├── base.py                # Base 클래스
│   │       ├── mixin.py               # 믹스인
│   │       └── session.py             # 세션 관리 (비동기)
│   └── domains/
│       └── v10/
│           ├── shared/
│           │   └── bases/
│           │       └── base.py        # Base 클래스 (실제 사용)
│           └── soccer/
│               └── bases/              # Soccer 모델들
│                   ├── __init__.py    # 모델 export
│                   ├── players.py     # Player 모델
│                   ├── schedules.py   # Schedule 모델
│                   ├── stadiums.py    # Stadium 모델
│                   └── teams.py       # Team 모델
└── .env                                # 환경 변수 (DATABASE_URL)
```

---

## 12. 환경 변수 설정

### 12.1 `.env` 파일

```env
# Neon PostgreSQL 연결 문자열
DATABASE_URL=postgresql://user:password@host.neon.tech/dbname?sslmode=require

# 기타 설정
DEBUG=False
OPENAI_API_KEY=sk-...
```

### 12.2 사용 위치

1. `config.py`: `Settings` 클래스에서 자동 로드
2. `env.py`: `get_url()` 함수에서 사용
3. `main.py`: `settings.connection_string`으로 접근

---

## 13. Alembic CLI 명령어

### 13.1 마이그레이션 생성
```bash
cd backend
alembic revision --autogenerate -m "설명"
```

### 13.2 마이그레이션 적용
```bash
alembic upgrade head      # 최신 버전으로
alembic upgrade +1        # 1단계 업그레이드
```

### 13.3 마이그레이션 롤백
```bash
alembic downgrade -1      # 1단계 롤백
alembic downgrade base    # 모든 마이그레이션 롤백
```

### 13.4 현재 상태 확인
```bash
alembic current           # 현재 리비전
alembic history           # 마이그레이션 히스토리
alembic show <revision>   # 특정 리비전 상세
```

---

## 14. 트러블슈팅 체크리스트

### 14.1 Alembic 실행 전 확인사항

- [ ] `.env` 파일에 `DATABASE_URL` 설정됨
- [ ] `alembic.ini` 파일 존재
- [ ] `alembic/env.py` 파일 존재
- [ ] `alembic/versions/` 디렉토리 존재
- [ ] Soccer 모델 파일들 존재
- [ ] `Base` 클래스 정의됨

### 14.2 오류 발생 시 확인사항

1. 경로 오류:
   - `script_location`이 올바른지 확인
   - 절대 경로로 설정했는지 확인

2. 연결 오류:
   - `DATABASE_URL`이 올바른지 확인
   - `alembic.ini`의 하드코딩된 URL 주석 처리 확인
   - 네트워크 연결 확인

3. 인코딩 오류:
   - `alembic.ini`에 한글 주석 없는지 확인
   - UTF-8로 저장되었는지 확인

4. 모델 인식 오류:
   - `env.py`에서 모델 import 확인
   - `Base.metadata`에 등록되었는지 확인

5. 기존 테이블 삭제:
   - `include_object` 함수 추가 확인
   - `protected_tables` 목록 확인

---

## 15. 베스트 프랙티스

### 15.1 개발 환경
- 마이그레이션 자동 생성: 첫 실행 시만
- 모델 변경 시: 수동으로 `alembic revision` 실행
- 테스트 후 적용: `alembic upgrade head`

### 15.2 프로덕션 환경
- 자동 생성 비활성화
- 검증된 마이그레이션만 적용
- 백업 후 마이그레이션 실행
- 롤백 계획 준비

### 15.3 팀 협업
- 마이그레이션 파일 버전 관리 (Git)
- 충돌 방지: 마이그레이션 순서 관리
- 리뷰: 마이그레이션 스크립트 코드 리뷰

---

## 16. 요약

### Soccer 테이블은 관계형 데이터베이스입니다

- **저장 방식**: PostgreSQL 일반 테이블
- **관리 도구**: SQLAlchemy ORM + Alembic 마이그레이션
- **테이블 구조**: Primary Key, Foreign Key, 일반 컬럼
- **벡터 스토어 아님**: 벡터 검색 기능 없음

### 데이터 흐름

```
JSONL 파일 (데이터)
    ↓
SQLAlchemy 모델 (Python 클래스)
    ↓
Alembic 마이그레이션 (DDL 생성)
    ↓
PostgreSQL 테이블 (Neon DB)
    ↓
FastAPI 라우터 (CRUD 작업)
```

### 벡터 스토어와의 관계

- Soccer 테이블: 관계형 DB (일반 CRUD)
- 벡터 스토어: 별도 테이블 (벡터 검색)
- 같은 PostgreSQL 사용, 다른 목적

---

## 17. 다음 단계

### 17.1 데이터 입력
JSONL 파일의 데이터를 테이블에 입력:
```python
# 예시
from app.domains.v10.soccer.bases import Player
from app.core.database import get_db

async def load_players():
    async with get_db() as db:
        # JSONL 파일 읽기
        # Player 객체 생성
        # db.add(player)
        # await db.commit()
```

### 17.2 API 엔드포인트 구현
```python
@router.get("/players")
async def get_players(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Player))
    players = result.scalars().all()
    return players
```

### 17.3 Repository 패턴 구현
```python
# backend/app/domains/v10/soccer/repositories/player_repository.py
class PlayerRepository:
    async def get_all(self, db: AsyncSession):
        ...
    async def get_by_id(self, db: AsyncSession, player_id: int):
        ...
```

---

## 18. 참고 자료

- SQLAlchemy 공식 문서: https://docs.sqlalchemy.org/
- Alembic 공식 문서: https://alembic.sqlalchemy.org/
- FastAPI 데이터베이스 가이드: https://fastapi.tiangolo.com/tutorial/sql-databases/
- Neon PostgreSQL 문서: https://neon.tech/docs/

---

## 마지막 체크포인트

현재 상태:
- ✅ Soccer 테이블 4개 생성됨 (player, schedule, stadium, team)
- ✅ 기존 테이블 복구됨 (users, refresh_tokens, gri_standards, langchain_pg_*)
- ✅ Alembic 설정 완료
- ✅ 무한 루프 방지 로직 적용
- ✅ 기존 테이블 보호 로직 적용

다음 작업:
- JSONL 데이터를 테이블에 입력
- API 엔드포인트 구현
- Repository/Service 레이어 구현
