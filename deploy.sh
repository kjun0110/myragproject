#!/bin/bash
set -e

echo "🚀 Starting deployment..."

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Docker 설치 확인
if ! command -v docker &> /dev/null; then
    log_warn "Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    log_info "Docker installed successfully"
fi

# Docker Compose 설치 확인
if ! command -v docker-compose &> /dev/null; then
    log_warn "Docker Compose not found. Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    log_info "Docker Compose installed successfully"
fi

# .env 파일 확인
if [ ! -f .env ]; then
    log_error ".env file not found!"
    exit 1
fi

log_info ".env file found"

# 기존 컨테이너 중지 및 제거
log_info "Stopping existing containers..."
docker-compose down 2>/dev/null || true

# 오래된 이미지 정리
log_info "Cleaning up old images..."
docker image prune -f

# Docker 이미지 빌드 및 컨테이너 시작
log_info "Building and starting containers..."
docker-compose up -d --build

# 컨테이너 시작 대기
log_info "Waiting for containers to start..."
sleep 10

# 컨테이너 상태 확인
if docker-compose ps | grep -q "Up"; then
    log_info "Containers are running"
else
    log_error "Containers failed to start"
    docker-compose logs
    exit 1
fi

# 헬스 체크
log_info "Running health check..."
MAX_RETRIES=5
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_info "Health check passed!"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        log_warn "Health check failed. Retry $RETRY_COUNT/$MAX_RETRIES..."
        sleep 5
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    log_error "Health check failed after $MAX_RETRIES attempts"
    log_error "Container logs:"
    docker-compose logs --tail=50
    exit 1
fi

# 배포 정보 출력
log_info "==================================="
log_info "✅ Deployment successful!"
log_info "==================================="
log_info "Application: FastAPI RAG Service"
log_info "Status: Running"
log_info "Port: 8000"
log_info "Health: http://localhost:8000/health"
log_info "API Docs: http://localhost:8000/docs"
log_info "==================================="

# 컨테이너 상태 출력
docker-compose ps

