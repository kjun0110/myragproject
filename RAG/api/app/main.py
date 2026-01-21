"""FastAPI 백엔드 서버 - ESG/GRI 응답 에이전트.

ESG/GRI 관련 서비스를 제공하는 API 서버입니다.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

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

        logger.info("[OK] 서버 초기화 완료!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"[ERROR] 초기화 실패: {e}")
        import traceback
        traceback.print_exc()

    yield

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

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
