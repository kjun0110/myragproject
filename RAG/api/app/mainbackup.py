"""
FastAPI 백엔드 서버 - 채팅 서비스.

채팅 관련 서비스를 제공하는 API 서버입니다.
"""

import os
import sys
from pathlib import Path

# .env 파일 로드
try:
    from dotenv import load_dotenv

    # 프로젝트 루트 찾기
    current_file = Path(__file__).resolve()
    app_dir = current_file.parent  # api/app/
    api_dir = app_dir.parent  # api/
    project_root = api_dir.parent  # 프로젝트 루트

    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()
except ImportError:
    pass

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(api_dir))

# 공통 모듈 import
from app.common.database.vector_store import (
    COLLECTION_NAME,
    CONNECTION_STRING,
    wait_for_postgres,
)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# FastAPI 앱 생성
app = FastAPI(
    title="Chat Service API",
    description="채팅 서비스를 제공하는 API",
    version="1.0.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 환경에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수: ChatService 인스턴스
chat_service = None


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화 작업."""
    global chat_service

    print("\n" + "=" * 60)
    print("🚀 채팅 서비스 서버 초기화 시작")
    print("=" * 60)

    # 환경 변수 확인
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    local_model_dir = os.getenv("LOCAL_MODEL_DIR", "기본값 사용")
    print(f"\n[INFO] LLM_PROVIDER: {llm_provider}")
    print(f"[INFO] LOCAL_MODEL_DIR: {local_model_dir}")

    # 1. Neon PostgreSQL 연결 대기
    print("\n1. Neon PostgreSQL 연결 확인 중...")
    wait_for_postgres()

    # 2. ChatService 초기화
    print("\n2. ChatService 초기화 중...")
    from app.domains.chat.agents.chat_service import ChatService

    chat_service = ChatService(
        connection_string=CONNECTION_STRING,
        collection_name=COLLECTION_NAME,
        model_name_or_path=local_model_dir
        if local_model_dir != "기본값 사용"
        else None,
    )

    # 3. Embedding 모델 초기화
    print("\n3. Embedding 모델 초기화 중...")
    chat_service.initialize_embeddings()

    # 4. LLM 모델 초기화
    print("\n4. LLM 모델 초기화 중...")
    chat_service.initialize_llm()

    # 5. PGVector 스토어 초기화
    print("\n5. PGVector 스토어 초기화 중...")
    from app.common.database.vector_store import initialize_vector_store

    initialize_vector_store()

    # 6. RAG 체인 초기화
    print("\n6. RAG 체인 초기화 중...")
    chat_service.initialize_rag_chain()

    # 7. Exaone 모델 사전 로드 (LangGraph용)
    print("\n7. Exaone3.5 모델 사전 로드 중...")
    try:
        from app.domains.chat.agents.graph import preload_exaone_model

        preload_exaone_model()
    except Exception as e:
        print(f"[WARNING] Exaone 모델 사전 로드 실패: {str(e)}")
        print("[INFO] 첫 요청 시 로드됩니다 (시간이 오래 걸릴 수 있습니다).")

    print("\n" + "=" * 60)
    print("[OK] 채팅 서비스 서버 초기화 완료!")
    print("=" * 60)


# 라우터 등록
from app.routers.chat_router import router as chat_router
from app.routers.graph_router import router as graph_router

app.include_router(chat_router)
app.include_router(graph_router)


@app.get("/")
async def root():
    """루트 엔드포인트."""
    return {
        "message": "Chat Service API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트."""
    global chat_service
    if chat_service is None:
        return {
            "status": "initializing",
            "chat_service": "not initialized",
        }

    return {
        "status": "healthy",
        "chat_service": "initialized",
        "openai_embeddings": "initialized"
        if chat_service.openai_embeddings
        else "not initialized",
        "local_embeddings": "initialized"
        if chat_service.local_embeddings
        else "not initialized",
        "openai_llm": "initialized" if chat_service.openai_llm else "not initialized",
        "local_llm": "initialized" if chat_service.local_llm else "not initialized",
        "openai_rag_chain": "initialized"
        if chat_service.openai_rag_chain
        else "not initialized",
        "local_rag_chain": "initialized"
        if chat_service.local_rag_chain
        else "not initialized",
        "openai_quota_exceeded": chat_service.openai_quota_exceeded,
    }


if __name__ == "__main__":
    import uvicorn

    # 포트 설정
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    print("=" * 60)
    print("🚀 채팅 서비스 서버 시작")
    print("=" * 60)
    print(f"📍 서버 주소: http://{host}:{port}")
    print(f"📚 API 문서: http://{host}:{port}/docs")
    print(f"🔍 헬스 체크: http://{host}:{port}/health")
    print("=" * 60)
    print("\n주요 엔드포인트:")
    print("  - POST /api/chain    : 채팅 API (RAG 체인)")
    print("  - POST /api/graph    : 그래프 API (LangGraph)")
    print("=" * 60)
    print()

    uvicorn.run(
        "app.mainbackup:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
