# Alembic 오류 수정 가이드

## 개요

이 문서는 Alembic 마이그레이션 실행 중 발생한 주요 오류들과 그 해결 방법을 정리한 것입니다.

---

## 발생한 오류 목록

### 1. 기존 테이블 삭제 문제 ⚠️

**문제 상황:**
- Alembic autogenerate가 Soccer 테이블을 생성하면서 기존 테이블들을 삭제하려고 시도
- 삭제 대상: `users`, `refresh_tokens`, `gri_standards`, `langchain_pg_collection`, `langchain_pg_embedding`

**원인:**
- Alembic의 `autogenerate`는 모델에 정의되지 않은 기존 테이블을 삭제 대상으로 인식
- Soccer 모델만 import했기 때문에 다른 테이블들이 모델에 없어서 삭제 대상으로 인식됨

**해결 방법:**

`api/alembic/env.py`에 `include_object` 함수 추가:

```python
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
```

`context.configure`에 추가:

```python
context.configure(
    connection=connection,
    target_metadata=target_metadata,
    include_object=include_object,  # 기존 테이블 보호
)
```

**적용 위치:**
- `api/alembic/env.py`의 `run_migrations_offline()` 함수
- `api/alembic/env.py`의 `run_migrations_online()` 함수

---

### 2. 경로 문제 (Path doesn't exist: alembic) 🗂️

**문제 상황:**
- Alembic이 `script_location`을 찾지 못하는 오류 발생
- 작업 디렉토리가 `api/`가 아닌 다른 곳에서 실행될 때 발생

**원인:**
- `alembic.ini`의 상대 경로 설정이 작업 디렉토리에 의존
- 절대 경로로 설정하지 않아 발생

**해결 방법:**

`api/app/main.py`에서 절대 경로로 설정:

```python
# Alembic 설정 파일 경로
alembic_ini_path = api_dir / "alembic.ini"

if alembic_ini_path.exists():
    # Alembic 설정 로드
    alembic_cfg = Config(str(alembic_ini_path))
    
    # script_location을 절대 경로로 설정 (작업 디렉토리 문제 해결)
    alembic_dir = api_dir / "alembic"
    alembic_cfg.set_main_option("script_location", str(alembic_dir))
```

**적용 위치:**
- `api/app/main.py`의 `lifespan` 함수 내부

---

### 3. 데이터베이스 URL 변환 문제 🔄

**문제 상황:**
- Alembic은 동기 드라이버(psycopg2)를 사용하지만, FastAPI는 비동기 드라이버(asyncpg)를 사용
- URL 형식이 맞지 않아 연결 실패

**원인:**
- Alembic은 동기 엔진만 지원 (`postgresql://`)
- FastAPI는 비동기 엔진 사용 (`postgresql+asyncpg://`)

**해결 방법:**

`api/app/main.py`에서 URL 변환:

```python
# 데이터베이스 URL을 동기식으로 변환 (psycopg2 사용)
database_url = settings.connection_string
# asyncpg -> psycopg2로 변환 (Alembic은 동기 드라이버 필요)
if database_url.startswith("postgresql+asyncpg://"):
    database_url = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
elif database_url.startswith("postgresql://"):
    pass  # 이미 동기 형식
else:
    # 다른 형식도 처리
    database_url = database_url.replace("+asyncpg", "")

# Alembic에 동기 URL 설정
alembic_cfg.set_main_option("sqlalchemy.url", database_url)
```

`api/alembic/env.py`에서도 동일한 변환:

```python
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
```

**적용 위치:**
- `api/app/main.py`의 `lifespan` 함수
- `api/alembic/env.py`의 `get_url()` 함수

---

### 4. 무한 루프 문제 (마이그레이션 반복 생성) 🔁

**문제 상황:**
1. `command.revision()` 실행 → 마이그레이션 파일 생성
2. watchfiles가 파일 변경 감지 → 서버 리로드
3. `lifespan` 재실행 → 또 `command.revision()` 실행
4. 무한 반복...

**원인:**
- 마이그레이션 파일 생성 시 watchfiles가 변경 감지
- 서버가 자동 리로드되면서 다시 마이그레이션 생성 시도

**해결 방법:**

기존 마이그레이션 파일이 있으면 생성하지 않도록 조건 추가:

```python
# versions 디렉토리에 마이그레이션 파일이 있는지 확인
versions_dir = alembic_dir / "versions"
existing_migrations = [
    f for f in versions_dir.glob("*.py") 
    if f.name != "__init__.py" and f.name != ".gitkeep"
] if versions_dir.exists() else []

logger.info(f"[INFO] 기존 마이그레이션 {len(existing_migrations)}개 발견 - 자동 생성 스킵")

# 마이그레이션 파일이 없을 때만 자동 생성
if not existing_migrations:
    command.revision(alembic_cfg, autogenerate=True, message="Auto-generate soccer tables")
else:
    logger.info(f"기존 마이그레이션 {len(existing_migrations)}개 발견 - 자동 생성 스킵")
```

**적용 위치:**
- `api/app/main.py`의 `lifespan` 함수 내부

---

### 5. 서버 시작 블로킹 문제 ⏱️

**문제 상황:**
- Alembic 마이그레이션이 서버 시작을 블로킹
- 마이그레이션이 오래 걸리면 서버 시작이 지연됨

**원인:**
- `lifespan` 함수에서 마이그레이션을 동기적으로 실행
- `yield` 전에 실행되어 서버 시작이 지연됨

**해결 방법:**

백그라운드에서 비동기로 실행:

```python
# 서버 시작 후 백그라운드에서 Alembic 마이그레이션 실행
if 'alembic_config_data' in locals():
    async def run_alembic_in_background():
        """서버 시작 후 백그라운드에서 Alembic 마이그레이션 실행"""
        import os
        from alembic import command
        await asyncio.sleep(2)  # 서버가 완전히 시작될 때까지 대기
        original_cwd = os.getcwd()
        try:
            os.chdir(str(alembic_config_data["api_dir"]))
            logger.info("[INFO] 백그라운드에서 Alembic 마이그레이션 시작...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, command.upgrade, alembic_config_data["alembic_cfg"], "head")
            logger.info("[✓] Alembic 마이그레이션 적용 완료 (Soccer 테이블 생성됨)")
        except Exception as e:
            logger.error(f"[ERROR] Alembic 마이그레이션 실행 중 오류: {e}")
            import traceback
            traceback.print_exc()
        finally:
            os.chdir(original_cwd)

    asyncio.create_task(run_alembic_in_background())
```

**적용 위치:**
- `api/app/main.py`의 `lifespan` 함수 내부 (`yield` 이후)

---

## 수정된 파일 목록

### 1. `api/alembic/env.py`
- ✅ `include_object` 함수 추가 (기존 테이블 보호)
- ✅ `get_url()` 함수에서 URL 변환 로직 추가
- ✅ `run_migrations_offline()`에 `include_object` 적용
- ✅ `run_migrations_online()`에 `include_object` 적용

### 2. `api/app/main.py`
- ✅ 절대 경로로 `script_location` 설정
- ✅ 데이터베이스 URL 변환 로직 추가
- ✅ 무한 루프 방지 로직 추가 (기존 마이그레이션 파일 확인)
- ✅ 백그라운드에서 마이그레이션 실행 (서버 시작 블로킹 방지)

---

## 수정 전후 비교

### 수정 전

**문제점:**
- ❌ 기존 테이블 삭제 시도
- ❌ 경로 오류 발생
- ❌ URL 형식 불일치
- ❌ 무한 루프 발생
- ❌ 서버 시작 지연

### 수정 후

**개선사항:**
- ✅ 기존 테이블 보호 (`include_object` 함수)
- ✅ 절대 경로 사용으로 경로 문제 해결
- ✅ URL 자동 변환 (asyncpg ↔ psycopg2)
- ✅ 무한 루프 방지 (조건부 마이그레이션 생성)
- ✅ 백그라운드 실행으로 서버 시작 지연 없음

---

## 테스트 방법

### 1. 기존 테이블 보호 확인

```bash
# 서버 시작
python -m api.app.main

# 데이터베이스에서 테이블 확인
# 기존 테이블들이 삭제되지 않았는지 확인
```

### 2. 마이그레이션 적용 확인

```bash
# Alembic 현재 상태 확인
cd api
alembic current

# 마이그레이션 히스토리 확인
alembic history

# Soccer 테이블 생성 확인
# player, schedule, stadium, team 테이블이 생성되었는지 확인
```

### 3. 무한 루프 방지 확인

```bash
# 서버 시작 후 로그 확인
# "기존 마이그레이션 X개 발견 - 자동 생성 스킵" 메시지 확인
# 마이그레이션 파일이 반복 생성되지 않는지 확인
```

---

## 주의사항

### 1. 보호된 테이블 목록 관리

새로운 테이블을 추가할 때는 `protected_tables` 목록에 추가하지 않아도 됩니다.
- Soccer 모델로 정의된 테이블은 자동으로 관리됨
- 다른 모델로 정의된 테이블도 자동으로 관리됨
- **모델에 정의되지 않은 기존 테이블만 보호 목록에 추가**

### 2. 모델 변경 시 마이그레이션 생성

자동 생성은 첫 실행 시에만 동작합니다. 모델을 변경한 후에는:

```bash
cd api
alembic revision --autogenerate -m "모델 변경 설명"
alembic upgrade head
```

### 3. 프로덕션 환경

프로덕션 환경에서는:
- 자동 마이그레이션 생성 비활성화 권장
- 검증된 마이그레이션만 적용
- 백업 후 마이그레이션 실행

---

## 참고 자료

- [Alembic 공식 문서](https://alembic.sqlalchemy.org/)
- [SQLAlchemy 문서](https://docs.sqlalchemy.org/)
- [프로젝트 마이그레이션 전략 문서](./ALEMBIC_MIGRATION_STRATEGY.md)

---

## 요약

| 오류 | 원인 | 해결 방법 | 적용 파일 |
|------|------|----------|----------|
| 기존 테이블 삭제 | autogenerate가 모델에 없는 테이블 삭제 | `include_object` 함수 추가 | `env.py` |
| 경로 오류 | 상대 경로 의존성 | 절대 경로 설정 | `main.py` |
| URL 변환 | asyncpg/psycopg2 불일치 | URL 변환 로직 추가 | `main.py`, `env.py` |
| 무한 루프 | 파일 변경 감지로 리로드 | 조건부 생성 로직 | `main.py` |
| 서버 블로킹 | 동기 실행 | 백그라운드 비동기 실행 | `main.py` |

---

**작성일:** 2026-01-26  
**작성자:** AI Assistant  
**버전:** 1.0
