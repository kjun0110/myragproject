# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.config import get_settings
from app.utils.scheduler import EmailScheduler
from app.routers import webhooks
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 전역 스케줄러
email_scheduler = EmailScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시
    email_scheduler.start()
    yield
    # 종료 시
    email_scheduler.stop()

app = FastAPI(
    title="Gmail-Slack Integration",
    description="Forward Gmail messages to Slack using FastAPI",
    version="1.0.0",
    lifespan=lifespan
)

# 라우터 등록
app.include_router(webhooks.router, prefix="/api/v1", tags=["webhooks"])

@app.get("/")
async def root():
    return {
        "message": "Gmail-Slack Integration Service",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}