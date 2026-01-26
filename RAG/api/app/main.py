"""FastAPI 백엔드 서버 - ESG/GRI 응답 에이전트.

ESG/GRI 관련 서비스를 제공하는 API 서버입니다.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# __pycache__ 파일 생성 방지 (가장 먼저 설정)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
app_dir = current_file.parent  # api/app/
api_dir = app_dir.parent  # api/
project_root = api_dir.parent  # 프로젝트 루트

sys.path.insert(0, str(api_dir))

# .env 파일 로드 - 환경 변수 로드
try:
    from dotenv import load_dotenv  # type: ignore

    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"[INFO] .env 파일 로드 완료: {env_file}")
    else:
        load_dotenv()
        print("[INFO] 현재 디렉토리에서 .env 파일 시도")
except ImportError:
    print("[WARNING] python-dotenv가 설치되지 않았습니다. 환경 변수만 사용합니다.")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)

# Config import
from app.common.config.config import get_settings

# 비동기 컨텍스트 매니저, 데이터베이스 초기화
@asynccontextmanager
async def lifespan(app):
    """애플리케이션 생명주기 관리."""
    # 시작 시 초기화
    logger.info("=" * 70)
    logger.info("🚀 ESG/GRI 응답 에이전트 서버 초기화 시작")
    logger.info("=" * 70)

    try:
        settings = get_settings()
        logger.info(f"[INFO] Config 로드 완료")
        logger.info(f"[INFO] DATABASE_URL: {'설정됨' if settings.database_url else '설정 안 됨'}")

        # 데이터베이스 연결 확인 (선택사항)
        try:
            import psycopg2  # type: ignore

            conn = psycopg2.connect(settings.connection_string)
            conn.close()
            logger.info("[✓] 데이터베이스 연결 확인 완료")
        except ImportError:
            logger.warning("[WARNING] psycopg2가 설치되지 않았습니다.")
        except Exception as e:
            logger.warning(f"[WARNING] 데이터베이스 연결 확인 실패: {e}")

        # Alembic 마이그레이션 자동 실행 (Soccer 테이블 생성)
        # 별도의 동기 엔진을 사용하여 Alembic 실행 (비동기 루프와 분리)
        try:
            from alembic.config import Config
            from alembic import command
            from pathlib import Path
            import asyncio

            # Alembic 설정 파일 경로
            alembic_ini_path = api_dir / "alembic.ini"

            if alembic_ini_path.exists():
                logger.info("[INFO] Alembic 마이그레이션 시작...")

                # Alembic 설정 로드
                alembic_cfg = Config(str(alembic_ini_path))

                # script_location을 절대 경로로 설정 (작업 디렉토리 문제 해결)
                alembic_dir = api_dir / "alembic"
                alembic_cfg.set_main_option("script_location", str(alembic_dir))

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

                # Alembic에 동기 URL 설정 (Alembic이 자체적으로 동기 엔진 생성)
                alembic_cfg.set_main_option("sqlalchemy.url", database_url)

                # 모든 모델 import (metadata에 등록)
                from app.domains.v10.soccer.bases import Player, Schedule, Stadium, Team
                logger.info("[INFO] Soccer 모델 로드 완료: Player, Schedule, Stadium, Team")

                # versions 디렉토리에 마이그레이션 파일이 있는지 확인
                versions_dir = alembic_dir / "versions"
                existing_migrations = [f for f in versions_dir.glob("*.py") if f.name != "__init__.py" and f.name != ".gitkeep"] if versions_dir.exists() else []

                logger.info(f"[INFO] 기존 마이그레이션 {len(existing_migrations)}개 발견 - 자동 생성 스킵")

                # Alembic 설정을 저장 (yield 이후 백그라운드에서 실행)
                # 서버 시작을 블로킹하지 않기 위해 yield 이후에 실행
                alembic_config_data = {
                    "alembic_cfg": alembic_cfg,
                    "api_dir": api_dir,
                }
                logger.info("[INFO] 마이그레이션은 서버 시작 후 백그라운드에서 실행됩니다.")
            else:
                logger.warning(f"[WARNING] Alembic 설정 파일을 찾을 수 없습니다: {alembic_ini_path}")
        except ImportError:
            logger.warning("[WARNING] Alembic이 설치되지 않았습니다. 'pip install alembic'을 실행하세요.")
        except Exception as e:
            logger.error(f"[ERROR] Alembic 마이그레이션 실행 실패: {e}")
            import traceback
            traceback.print_exc()

        logger.info("[OK] 서버 초기화 완료!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"[ERROR] 초기화 실패: {e}")
        import traceback
        traceback.print_exc()

    try:
        yield

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
    finally:
        # 종료 시 정리 작업
        logger.info("서버 종료 중...")


# fastapi 인스턴스 생성
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(
    title="ESG/GRI 응답 에이전트 API",
    description="ESG/GRI 관련 응답을 생성하는 에이전트 API",
    version="1.0.0",
    lifespan=lifespan,
)

# 미들웨어 설정 (CORS, 로깅, 에러 처리)
# CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 환경에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """요청/응답 로깅 미들웨어."""
    import time

    start_time = time.time()
    logger.info(f"[REQUEST] {request.method} {request.url.path}")

    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(
            f"[RESPONSE] {request.method} {request.url.path} - "
            f"Status: {response.status_code} - Time: {process_time:.3f}s"
        )
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"[ERROR] {request.method} {request.url.path} - "
            f"Error: {str(e)} - Time: {process_time:.3f}s"
        )
        raise


# 에러 처리 핸들러
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """요청 검증 오류 핸들러."""
    logger.error(f"[VALIDATION ERROR] {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """일반 예외 핸들러."""
    logger.error(f"[EXCEPTION] {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "message": str(exc)},
    )


# 라우터 등록 (API 엔드포인트 정의)
# ESG/GRI 라우터 (라우터가 구현되면 자동으로 등록됨)
api_prefix = "/api/v1/esg"

try:
    from app.routers.v1.esg.gri_standards_router import router as gri_standards_router  # type: ignore

    app.include_router(
        gri_standards_router,
        prefix=api_prefix + "/gri-standards",
        tags=["GRI Standards"],
    )
except (ImportError, AttributeError):
    pass

try:
    from app.routers.v1.esg.gri_env_contents_router import router as gri_env_contents_router  # type: ignore

    app.include_router(
        gri_env_contents_router,
        prefix=api_prefix + "/gri-env-contents",
        tags=["GRI Environmental Contents"],
    )
except (ImportError, AttributeError):
    pass

try:
    from app.routers.v1.esg.gri_soc_contents_router import router as gri_soc_contents_router  # type: ignore

    app.include_router(
        gri_soc_contents_router,
        prefix=api_prefix + "/gri-soc-contents",
        tags=["GRI Social Contents"],
    )
except (ImportError, AttributeError):
    pass

try:
    from app.routers.v1.esg.gri_gov_contents_router import router as gri_gov_contents_router  # type: ignore

    app.include_router(
        gri_gov_contents_router,
        prefix=api_prefix + "/gri-gov-contents",
        tags=["GRI Governance Contents"],
    )
except (ImportError, AttributeError):
    pass

# Soccer Player 라우터
try:
    from app.routers.v10.soccer.player_router import router as player_router  # type: ignore

    app.include_router(player_router)
    logger.info("[✓] Soccer Player 라우터 등록 완료: /api/v10/soccer/player")
except ImportError as e:
    logger.warning(f"[WARNING] Soccer Player 라우터 import 실패: {e}")
except AttributeError as e:
    logger.warning(f"[WARNING] Soccer Player 라우터 속성 오류: {e}")
except Exception as e:
    logger.error(f"[ERROR] Soccer Player 라우터 등록 실패: {e}")

# Soccer Team 라우터
try:
    from app.routers.v10.soccer.team_router import router as team_router  # type: ignore

    app.include_router(team_router)
    logger.info("[✓] Soccer Team 라우터 등록 완료: /api/v10/soccer/team")
except ImportError as e:
    logger.warning(f"[WARNING] Soccer Team 라우터 import 실패: {e}")
except AttributeError as e:
    logger.warning(f"[WARNING] Soccer Team 라우터 속성 오류: {e}")
except Exception as e:
    logger.error(f"[ERROR] Soccer Team 라우터 등록 실패: {e}")

# Soccer Stadium 라우터
try:
    from app.routers.v10.soccer.stadium_router import router as stadium_router  # type: ignore

    app.include_router(stadium_router)
    logger.info("[✓] Soccer Stadium 라우터 등록 완료: /api/v10/soccer/stadium")
except ImportError as e:
    logger.warning(f"[WARNING] Soccer Stadium 라우터 import 실패: {e}")
except AttributeError as e:
    logger.warning(f"[WARNING] Soccer Stadium 라우터 속성 오류: {e}")
except Exception as e:
    logger.error(f"[ERROR] Soccer Stadium 라우터 등록 실패: {e}")

# Soccer Schedule 라우터
try:
    from app.routers.v10.soccer.schedule_router import router as schedule_router  # type: ignore

    app.include_router(schedule_router)
    logger.info("[✓] Soccer Schedule 라우터 등록 완료: /api/v10/soccer/schedule")
except ImportError as e:
    logger.warning(f"[WARNING] Soccer Schedule 라우터 import 실패: {e}")
except AttributeError as e:
    logger.warning(f"[WARNING] Soccer Schedule 라우터 속성 오류: {e}")
except Exception as e:
    logger.error(f"[ERROR] Soccer Schedule 라우터 등록 실패: {e}")

# 루트 엔드포인트
@app.get("/")
async def root():
    """루트 엔드포인트."""
    return {
        "message": "ESG/GRI 응답 에이전트 API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "gri_standards": "/api/v1/esg/gri-standards",
            "gri_env_contents": "/api/v1/esg/gri-env-contents",
            "gri_soc_contents": "/api/v1/esg/gri-soc-contents",
            "gri_gov_contents": "/api/v1/esg/gri-gov-contents",
        },
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트."""
    try:
        settings = get_settings()
        return {
            "status": "healthy",
            "service": "esg-gri-agent",
            "database": "configured" if settings.database_url else "not configured",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "esg-gri-agent",
            "error": str(e),
        }


if __name__ == "__main__":
    import uvicorn  # type: ignore

    # 작업 디렉토리 기준으로 api 디렉토리만 감시
    # python -m api.app.main 실행 시 프로젝트 루트(RAG/)에서 실행됨
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        reload_dirs=["api/app"],  # api/app 디렉토리만 감시 (프론트엔드 제외)
        reload_excludes=["**/__pycache__/**", "**/*.pyc", "**/*.pyo", "**/*.log"],
    )
