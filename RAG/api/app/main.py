"""FastAPI 백엔드 서버 - ESG/GRI 응답 에이전트.

ESG/GRI 관련 서비스를 제공하는 API 서버입니다.
"""

import asyncio
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
    else:
        load_dotenv()
except ImportError:
    pass  # python-dotenv가 없으면 환경 변수만 사용

# 로깅 설정 - 중복 핸들러 방지
# Uvicorn reloader는 메인 프로세스와 워커 프로세스를 각각 실행하므로
# 로깅 핸들러가 중복 설정되지 않도록 체크
root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,  # 기존 설정이 있어도 덮어쓰기 (중복 방지)
    )

# MCP client의 정상적인 연결 종료 로그 숨기기 (오해 방지)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

# Config import
from app.core.config.config import get_settings

# 임베딩 큐 워커: 온디맨드 — job 추가 시에만 실행, 큐가 비면 종료


async def _embedding_worker_run_once() -> None:
    """대기 중인 임베딩 job을 모두 처리하고 큐가 비면 종료합니다."""
    from app.routers.shared.embedding_queue import (
        STATUS_COMPLETED,
        STATUS_FAILED,
        STATUS_PROCESSING,
        pop_pending_job_async,
        update_status_async,
    )
    from app.domains.v10.soccer.hub.orchestrators.player_orchestrator import (
        PlayerOrchestrator,
    )

    logger.info("[EMBED_WORKER] 임베딩 워커 실행 (온디맨드)")
    orch = PlayerOrchestrator()
    processed = 0
    try:
        while True:
            job_id = await pop_pending_job_async("player")
            if not job_id:
                break
            await update_status_async(job_id, STATUS_PROCESSING)
            try:
                result = await orch.run_embedding_batch()
                await update_status_async(job_id, STATUS_COMPLETED, result=result)
                processed += 1
            except Exception as exc:
                logger.exception("[EMBED_WORKER] job_id=%s 처리 실패", job_id)
                await update_status_async(job_id, STATUS_FAILED, error=str(exc))
            await asyncio.sleep(0)
        logger.info("[EMBED_WORKER] 큐 비음 — 워커 종료 (처리 job 수: %s)", processed)
    except asyncio.CancelledError:
        logger.info("[EMBED_WORKER] 워커 취소됨")
    finally:
        # 완료 시 app.state에서 태스크 참조 제거 (다음 트리거에서 새 태스크 생성 가능)
        pass


def _trigger_embedding_worker(app) -> None:
    """이미 워커가 돌고 있지 않으면, 온디맨드 워커 태스크를 한 번 시작합니다."""
    task = getattr(app.state, "_embedding_worker_task", None)
    if task is not None and not task.done():
        return
    new_task = asyncio.create_task(_embedding_worker_run_once())
    app.state._embedding_worker_task = new_task

    def _clear_when_done(t):
        if hasattr(app, "state") and getattr(app.state, "_embedding_worker_task", None) is t:
            app.state._embedding_worker_task = None

    new_task.add_done_callback(_clear_when_done)


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
                # Alembic 설정 로드 (로그 최소화)
                alembic_cfg = Config(str(alembic_ini_path))

                # script_location을 절대 경로로 설정 (작업 디렉토리 문제 해결)
                alembic_dir = api_dir / "alembic"
                alembic_cfg.set_main_option("script_location", str(alembic_dir))
                # path_separator=os 로 DeprecationWarning 제거
                alembic_cfg.set_main_option("path_separator", "os")

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
                from app.domains.v10.soccer.models.bases import Player, Schedule, Stadium, Team

                # Alembic 설정을 저장 (yield 이후 서버 완전 시작 후 실행)
                alembic_config_data = {
                    "alembic_cfg": alembic_cfg,
                    "api_dir": api_dir,
                }
            else:
                logger.warning(f"[WARNING] Alembic 설정 파일을 찾을 수 없습니다: {alembic_ini_path}")
        except ImportError as e:
            logger.warning(f"[WARNING] Alembic 관련 모듈 import 실패: {e}")
            logger.warning("[WARNING] Alembic이 설치되지 않았거나, 모델 import에 문제가 있을 수 있습니다.")
        except Exception as e:
            logger.error(f"[ERROR] Alembic 마이그레이션 실행 실패: {e}")
            import traceback
            traceback.print_exc()

        # 라우터 등록 (서버 시작 시 한 번만 실행)
        _register_routers(app)

        # MCP 서버(LLM 9100, player 9001)를 같은 프로세스에서 서브프로세스로 기동 (한 번만 실행)
        # stderr=None → 9001/9100 로그·에러가 이 터미널에 출력되어 디버깅 가능
        _mcp_processes = []
        if os.getenv("SOCCER_START_MCP_WITH_MAIN", "1").strip().lower() in ("1", "true", "yes"):
            import subprocess

            cwd = str(api_dir)
            env = os.environ.copy()
            try:
                p_llm = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "app.domains.v10.soccer.spokes.mcp.llm_server:app",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        "9100",
                    ],
                    cwd=cwd,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=None,
                )
                _mcp_processes.append(("LLM(9100)", p_llm))
                logger.info("[OK] MCP LLM 서버(9100) 서브프로세스 기동 (stderr는 이 터미널에 출력)")
            except Exception as e:
                logger.warning("[WARNING] MCP LLM 서버 기동 실패: %s", e)
            try:
                p_player = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "app.domains.v10.soccer.spokes.mcp.player_server:app",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        "9001",
                    ],
                    cwd=cwd,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=None,
                )
                _mcp_processes.append(("Player(9001)", p_player))
                logger.info("[OK] MCP Player 서버(9001) 서브프로세스 기동 (stderr는 이 터미널에 출력)")
            except Exception as e:
                logger.warning("[WARNING] MCP Player 서버 기동 실패: %s", e)
            app.state._mcp_processes = _mcp_processes
            logger.info("[MCP] 엑사원(9100)은 첫 요청 시 모델 로딩으로 1~2분 걸릴 수 있음. 9001/9100 로그는 이 터미널에 함께 출력됨")
        else:
            app.state._mcp_processes = []

        # 임베딩 워커: 상시 폴링 없음. POST /embed 시에만 온디맨드로 워커 실행 → 큐 비면 종료
        app.state._embedding_worker_task = None
        app.state._embedding_worker_trigger = _trigger_embedding_worker
        logger.info("[OK] 임베딩 온디맨드 워커 준비 (job 추가 시 자동 실행)")

        logger.info("[OK] 서버 초기화 완료!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"[ERROR] 초기화 실패: {e}")
        import traceback
        traceback.print_exc()

    try:
        yield

        # 서버 완전 시작 후 Alembic 마이그레이션 실행
        if 'alembic_config_data' in locals():
            async def run_alembic_after_startup():
                """서버 완전 시작 후 Alembic 마이그레이션 실행"""
                import os
                import warnings
                from alembic import command
                await asyncio.sleep(3)  # 서버가 완전히 시작될 때까지 대기
                original_cwd = os.getcwd()
                try:
                    os.chdir(str(alembic_config_data["api_dir"]))
                    logger.info("[INFO] 서버 시작 완료 - Alembic 마이그레이션 실행 중...")
                    loop = asyncio.get_event_loop()

                    def _run_upgrade():
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                category=DeprecationWarning,
                                module="alembic.config",
                            )
                            command.upgrade(alembic_config_data["alembic_cfg"], "head")

                    await loop.run_in_executor(None, _run_upgrade)
                    logger.info("[✓] Alembic 마이그레이션 적용 완료")
                except Exception as e:
                    logger.error(f"[ERROR] Alembic 마이그레이션 실행 중 오류: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    os.chdir(original_cwd)

            asyncio.create_task(run_alembic_after_startup())
    finally:
        # 임베딩 워커 태스크 취소 (Ctrl+C 시 오래 기다리지 않고 종료)
        _embed_task = getattr(app.state, "_embedding_worker_task", None)
        if _embed_task is not None and not _embed_task.done():
            _embed_task.cancel()
            try:
                await asyncio.wait_for(_embed_task, timeout=3.0)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning("[OK] 임베딩 워커 취소 대기 타임아웃(3초), 종료 진행")
            else:
                logger.info("[OK] 임베딩 워커 태스크 종료")
        # MCP 서브프로세스 종료
        _procs = getattr(app.state, "_mcp_processes", [])
        for name, proc in _procs:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=5)
                logger.info("[OK] MCP 서버 종료: %s", name)
            except Exception as e:
                logger.warning("[WARNING] MCP 서버 종료 실패 %s: %s", name, e)
        logger.info("서버 종료 중...")


# fastapi 인스턴스 생성
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

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


# 로깅 미들웨어 (상태 조회 폴링은 로그 폭주 방지로 DEBUG만, Starlette 1.0 권장: 클래스 미들웨어)
def _is_embed_status_poll(path: str) -> bool:
    return "embed/status" in path


class LogRequestsMiddleware(BaseHTTPMiddleware):
    """요청/응답 로깅 미들웨어."""

    async def dispatch(self, request: Request, call_next):
        import time

        start_time = time.time()
        skip_info = _is_embed_status_poll(request.url.path)
        if skip_info:
            logger.debug("[REQUEST] %s %s", request.method, request.url.path)
        else:
            logger.info("[REQUEST] %s %s", request.method, request.url.path)

        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            if skip_info:
                logger.debug(
                    "[RESPONSE] %s %s - Status: %s - Time: %.3fs",
                    request.method, request.url.path, response.status_code, process_time,
                )
            else:
                logger.info(
                    "[RESPONSE] %s %s - Status: %s - Time: %.3fs",
                    request.method, request.url.path, response.status_code, process_time,
                )
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                "[ERROR] %s %s - Error: %s - Time: %.3fs",
                request.method, request.url.path, str(e), process_time,
            )
            raise


app.add_middleware(LogRequestsMiddleware)


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


# 라우터 등록 함수 (lifespan에서 한 번만 실행)
def _register_routers(app_instance):
    """라우터를 등록합니다. lifespan에서 한 번만 실행됩니다."""
    # ESG/GRI 라우터 (라우터가 구현되면 자동으로 등록됨)
    api_prefix = "/api/v1/esg"

    try:
        from app.routers.v1.esg.gri_standards_router import router as gri_standards_router  # type: ignore

        app_instance.include_router(
            gri_standards_router,
            prefix=api_prefix + "/gri-standards",
            tags=["GRI Standards"],
        )
    except (ImportError, AttributeError):
        pass

    try:
        from app.routers.v1.esg.gri_env_contents_router import router as gri_env_contents_router  # type: ignore

        app_instance.include_router(
            gri_env_contents_router,
            prefix=api_prefix + "/gri-env-contents",
            tags=["GRI Environmental Contents"],
        )
    except (ImportError, AttributeError):
        pass

    try:
        from app.routers.v1.esg.gri_soc_contents_router import router as gri_soc_contents_router  # type: ignore

        app_instance.include_router(
            gri_soc_contents_router,
            prefix=api_prefix + "/gri-soc-contents",
            tags=["GRI Social Contents"],
        )
    except (ImportError, AttributeError):
        pass

    try:
        from app.routers.v1.esg.gri_gov_contents_router import router as gri_gov_contents_router  # type: ignore

        app_instance.include_router(
            gri_gov_contents_router,
            prefix=api_prefix + "/gri-gov-contents",
            tags=["GRI Governance Contents"],
        )
    except (ImportError, AttributeError):
        pass

    # Soccer Player 라우터
    try:
        from app.routers.v10.soccer.player_router import router as player_router  # type: ignore

        app_instance.include_router(player_router)
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

        app_instance.include_router(team_router)
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

        app_instance.include_router(stadium_router)
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

        app_instance.include_router(schedule_router)
        logger.info("[✓] Soccer Schedule 라우터 등록 완료: /api/v10/soccer/schedule")
    except ImportError as e:
        logger.warning(f"[WARNING] Soccer Schedule 라우터 import 실패: {e}")
    except AttributeError as e:
        logger.warning(f"[WARNING] Soccer Schedule 라우터 속성 오류: {e}")
    except Exception as e:
        logger.error(f"[ERROR] Soccer Schedule 라우터 등록 실패: {e}")

    # Soccer Chat 라우터
    try:
        from app.routers.v10.soccer.chat_router import router as chat_router  # type: ignore

        app_instance.include_router(chat_router)
        logger.info("[✓] Soccer Chat 라우터 등록 완료: /api/v10/soccer/chat")
    except ImportError as e:
        logger.warning(f"[WARNING] Soccer Chat 라우터 import 실패: {e}")
    except AttributeError as e:
        logger.warning(f"[WARNING] Soccer Chat 라우터 속성 오류: {e}")
    except Exception as e:
        logger.error(f"[ERROR] Soccer Chat 라우터 등록 실패: {e}")

    # Auth 라우터 (JWT 발급 + Redis/BullMQ 연동용)
    try:
        from app.routers.shared.auth_router import router as auth_router  # type: ignore

        app_instance.include_router(auth_router)
        logger.info("[✓] Auth 라우터 등록 완료: /api/v1/auth")
    except ImportError as e:
        logger.warning(f"[WARNING] Auth 라우터 import 실패: {e}")
    except AttributeError as e:
        logger.warning(f"[WARNING] Auth 라우터 속성 오류: {e}")
    except Exception as e:
        logger.error(f"[ERROR] Auth 라우터 등록 실패: {e}")

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
        # reloader 중복 로그 방지: 워커 프로세스에서만 로그 출력
        use_colors=True,
    )
