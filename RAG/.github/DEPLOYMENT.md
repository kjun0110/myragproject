# CI/CD 배포 가이드

## 개요

이 프로젝트는 GitHub Actions를 통해 AWS EC2에 자동으로 배포됩니다.

## 배포 아키텍처

```
GitHub Repository (main branch)
    ↓ [push/merge]
GitHub Actions CI/CD
    ↓ [SSH + rsync]
AWS EC2 (ubuntu@172.31.41.169)
    ↓ [Docker Compose]
FastAPI Container (Port 8000)
    ↓ [Connection]
Neon PostgreSQL (Cloud)
```

## 사전 준비사항

### 1. GitHub Secrets 설정

Repository → Settings → Secrets and variables → Actions에서 다음 Secrets를 추가하세요:

#### 필수 Secrets

- **EC2_HOST**: `13.125.99.225` 또는 `ec2-13-125-99-225.ap-northeast-2.compute.amazonaws.com`
- **EC2_USERNAME**: `ubuntu`
- **EC2_SSH_KEY**: `Kjun.pem` 파일의 전체 내용 (BEGIN부터 END까지)
- **DATABASE_URL**: Neon PostgreSQL 연결 문자열
  ```
  postgresql://user:password@host/database?sslmode=require
  ```

#### 선택적 Secrets

- **OPENAI_API_KEY**: OpenAI API 키 (OpenAI 모델 사용 시)
- **LOCAL_MODEL_DIR**: 로컬 모델 경로 (예: `api/app/model/midm`)

### 2. EC2 인스턴스 초기 설정

SSH로 EC2에 접속하여 다음을 실행:

```bash
# Docker 및 Docker Compose 설치 (deploy.sh가 자동으로 설치하지만, 미리 설치 가능)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker

# 애플리케이션 디렉토리 생성
mkdir -p ~/app
```

### 3. Security Group 설정

AWS EC2 Security Group에서 다음 포트 허용:

- **Inbound**:
  - SSH (22): GitHub Actions IP 또는 사용자 IP
  - HTTP (8000): 필요한 경우 (또는 Load Balancer를 통해)

- **Outbound**: 모두 허용

## 배포 프로세스

### 자동 배포 (권장)

1. 코드 변경 후 `main` 브랜치에 push
   ```bash
   git add .
   git commit -m "Your commit message"
   git push origin main
   ```

2. GitHub Actions 자동 실행
   - Repository → Actions 탭에서 진행 상황 확인

3. 배포 완료 확인
   - 약 3-5분 소요
   - Health check 통과 시 배포 성공

### 수동 배포

GitHub Repository → Actions → Deploy to EC2 → Run workflow

## 배포 후 확인

### 1. 헬스 체크

```bash
curl http://13.125.99.225:8000/health
```

예상 응답:
```json
{
  "status": "healthy",
  "vector_store": "initialized",
  "openai_embeddings": "initialized",
  "local_embeddings": "initialized",
  "openai_llm": "initialized",
  "local_llm": "initialized",
  "openai_rag_chain": "initialized",
  "local_rag_chain": "initialized",
  "openai_quota_exceeded": false
}
```

### 2. API 문서 확인

브라우저에서: `http://13.125.99.225:8000/docs`

### 3. 로그 확인 (EC2에서)

```bash
# 컨테이너 로그 실시간 확인
docker-compose logs -f

# 최근 로그만 확인
docker-compose logs --tail=100

# 컨테이너 상태 확인
docker-compose ps
```

## 트러블슈팅

### 배포 실패 시

1. **GitHub Actions 로그 확인**
   - Repository → Actions → 실패한 워크플로우 클릭
   - 각 단계별 로그 확인

2. **EC2 컨테이너 로그 확인**
   ```bash
   ssh -i "Kjun.pem" ubuntu@ec2-13-125-99-225.ap-northeast-2.compute.amazonaws.com
   cd ~/app
   docker-compose logs
   ```

3. **컨테이너 재시작**
   ```bash
   docker-compose restart
   ```

4. **완전히 재배포**
   ```bash
   docker-compose down
   ./deploy.sh
   ```

### 일반적인 문제

#### 1. SSH 연결 실패
- EC2_SSH_KEY가 올바르게 설정되었는지 확인
- EC2 인스턴스가 실행 중인지 확인
- Security Group에서 SSH 포트(22)가 열려있는지 확인

#### 2. Docker 빌드 실패
- `api/requirements.txt`의 패키지 버전 확인
- EC2 디스크 공간 확인: `df -h`
- Docker 이미지 정리: `docker system prune -a`

#### 3. 헬스 체크 실패
- .env 파일의 DATABASE_URL 확인
- Neon PostgreSQL 연결 확인
- 컨테이너 로그에서 오류 메시지 확인

#### 4. 메모리 부족
- EC2 인스턴스 타입 확인 (최소 t3.medium 권장)
- 로컬 모델 사용 시 GPU 인스턴스 필요 (g4dn 계열)

## 롤백

### 이전 버전으로 롤백

```bash
# EC2에 SSH 접속
ssh -i "Kjun.pem" ubuntu@ec2-13-125-99-225.ap-northeast-2.compute.amazonaws.com

# 이전 커밋으로 롤백
cd ~/app
git fetch origin
git checkout <이전-커밋-해시>
./deploy.sh
```

또는 GitHub에서 이전 커밋을 main 브랜치에 push하여 자동 배포

## 모니터링

### 실시간 모니터링

```bash
# CPU/메모리 사용량
docker stats

# 컨테이너 상태
watch -n 5 'docker-compose ps'

# 애플리케이션 로그
docker-compose logs -f --tail=100
```

### 로그 위치

- 컨테이너 로그: Docker가 자동 관리 (최대 10MB × 3개 파일)
- 로그 확인: `docker-compose logs`

## 성능 최적화

### 권장 EC2 인스턴스 타입

- **개발/테스트**: t3.small (2 vCPU, 2GB RAM)
- **프로덕션 (OpenAI만)**: t3.medium (2 vCPU, 4GB RAM)
- **프로덕션 (로컬 모델)**: g4dn.xlarge (4 vCPU, 16GB RAM, GPU)

## 보안 고려사항

1. **환경 변수**
   - 모든 민감 정보는 GitHub Secrets에 저장
   - .env 파일은 절대 Git에 커밋하지 않음

2. **SSH 키**
   - Kjun.pem 파일은 안전하게 보관
   - 정기적으로 키 교체 권장

3. **방화벽**
   - Security Group에서 필요한 포트만 열기
   - SSH는 특정 IP만 허용 권장

4. **업데이트**
   - 정기적인 보안 업데이트: `sudo apt update && sudo apt upgrade`

## 연락처 및 지원

문제 발생 시 다음을 확인하세요:
1. GitHub Actions 로그
2. EC2 컨테이너 로그
3. Neon PostgreSQL 상태
4. API 문서 (`/docs`)

## 추가 리소스

- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [Docker 문서](https://docs.docker.com/)
- [GitHub Actions 문서](https://docs.github.com/en/actions)
- [AWS EC2 문서](https://docs.aws.amazon.com/ec2/)

