"""
스팸 필터 에이전트 서버.

KoELECTRA 게이트웨이 + EXAONE Reader 기반 스팸 메일 분석 서버입니다.

실행 방법:
    python -m uvicorn app.agent:app --reload --port 8001
    또는
    python app/agent.py
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
app_dir = current_file.parent  # api/app/
api_dir = app_dir.parent  # api/
project_root = api_dir.parent  # 프로젝트 루트

sys.path.insert(0, str(api_dir))

# .env 파일 로드
try:
    from dotenv import load_dotenv

    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()
except ImportError:
    pass  # python-dotenv가 없으면 환경 변수만 사용

# FastAPI 및 라우터 import
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# FastAPI 앱 생성
app = FastAPI(
    title="Spam Filter Agent API",
    description="KoELECTRA 게이트웨이 + EXAONE Reader 기반 스팸 메일 분석 API",
    version="1.0.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 환경에서는 특정 도메인만 허용하도록 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
from app.routers.v1.mcp_spam_router import router as mcp_router

app.include_router(mcp_router)


@app.get("/")
async def root():
    """루트 엔드포인트."""
    return {
        "message": "Spam Filter Agent API",
        "version": "1.0.0",
        "endpoints": {
            "gate": "/api/mcp/gate",
            "spam_analyze": "/api/mcp/spam-analyze",
            "gate_state": "/api/mcp/gate/state/{request_id}",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트."""
    return {"status": "healthy", "service": "spam-filter-agent"}


if __name__ == "__main__":
    import uvicorn

    # 포트 설정 (환경 변수 또는 기본값)
    port = int(os.getenv("PORT", 8000))  # 기본값 8000
    host = os.getenv("HOST", "0.0.0.0")

    print("=" * 60)
    print("🚀 Spam Filter Agent 서버 시작")
    print("=" * 60)
    print(f"📍 서버 주소: http://{host}:{port}")
    print(f"📚 API 문서: http://{host}:{port}/docs")
    print(f"🔍 헬스 체크: http://{host}:{port}/health")
    print("=" * 60)
    print("\n주요 엔드포인트:")
    print("  - POST /api/mcp/gate          : KoELECTRA 게이트웨이 (도메인 분류)")
    print("  - POST /api/mcp/spam-analyze  : 전체 스팸 분석 (KoELECTRA + EXAONE)")
    print("  - GET  /api/mcp/gate/state    : 상태 조회")
    print("=" * 60)
    print()

    uvicorn.run(
        "app.agent:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
