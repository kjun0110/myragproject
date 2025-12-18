FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY api/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY api/ ./api/

# .env 파일 복사 (런타임에서 사용)
COPY .env .env

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV MODE=api
ENV PYTHONUNBUFFERED=1

# 포트 노출
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 실행 명령
CMD ["python", "-m", "api.app.main"]

