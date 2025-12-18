# GitHub Secrets 설정 가이드

CI/CD 배포를 위해 GitHub Repository에 다음 Secrets를 설정해야 합니다.

## 설정 방법

1. GitHub Repository 페이지로 이동
2. `Settings` 탭 클릭
3. 왼쪽 메뉴에서 `Secrets and variables` → `Actions` 클릭
4. `New repository secret` 버튼 클릭
5. 아래 각 Secret을 추가

## 필수 Secrets

### 1. EC2_HOST
- **Name**: `EC2_HOST`
- **Value**: `13.125.99.225` 또는 `ec2-13-125-99-225.ap-northeast-2.compute.amazonaws.com`
- **설명**: EC2 인스턴스의 퍼블릭 IP 또는 호스트네임

### 2. EC2_USERNAME
- **Name**: `EC2_USERNAME`
- **Value**: `ubuntu`
- **설명**: EC2 인스턴스의 SSH 사용자명

### 3. EC2_SSH_KEY
- **Name**: `EC2_SSH_KEY`
- **Value**: `Kjun.pem` 파일의 전체 내용
- **설정 방법**:
  1. 로컬에서 `Kjun.pem` 파일 열기
  2. 파일 내용 전체 복사 (-----BEGIN RSA PRIVATE KEY----- 부터 -----END RSA PRIVATE KEY----- 까지)
  3. GitHub Secrets에 붙여넣기

예시:
```
-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...
(여러 줄)
...
-----END RSA PRIVATE KEY-----
```

### 4. DATABASE_URL
- **Name**: `DATABASE_URL`
- **Value**: Neon PostgreSQL 연결 문자열
- **형식**: `postgresql://[user]:[password]@[host]/[database]?sslmode=require`

예시:
```
postgresql://neondb_owner:npg_bNXv7Ll1mrBJ@ep-empty-tree-a15rzl4v-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require
```

**가져오는 방법**:
- Neon Dashboard → 프로젝트 선택 → Connection String 복사

## 선택적 Secrets

### 5. OPENAI_API_KEY (OpenAI 사용 시)
- **Name**: `OPENAI_API_KEY`
- **Value**: OpenAI API 키
- **설명**: OpenAI GPT 모델을 사용하려면 필요

예시:
```
sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**가져오는 방법**:
- OpenAI Platform → API Keys → Create new secret key

### 6. LOCAL_MODEL_DIR (로컬 모델 사용 시)
- **Name**: `LOCAL_MODEL_DIR`
- **Value**: 로컬 모델 경로
- **설명**: 로컬 Midm 모델을 사용하는 경우

예시:
```
api/app/model/midm
```

또는 HuggingFace 모델:
```
K-intelligence/Midm-2.0-Mini-Instruct
```

## 설정 확인

모든 Secrets 설정 후:

1. Repository → Settings → Secrets and variables → Actions
2. 다음 목록이 보여야 합니다:
   - ✅ EC2_HOST
   - ✅ EC2_USERNAME
   - ✅ EC2_SSH_KEY
   - ✅ DATABASE_URL
   - ✅ OPENAI_API_KEY (선택)
   - ✅ LOCAL_MODEL_DIR (선택)

## 테스트

설정 완료 후 테스트:

```bash
# 코드 변경 후 push
git add .
git commit -m "test: GitHub Actions CI/CD"
git push origin main

# GitHub Repository → Actions 탭에서 진행 상황 확인
```

## 보안 주의사항

⚠️ **중요**: GitHub Secrets는 암호화되어 저장되며, 한 번 저장하면 내용을 다시 볼 수 없습니다.

- Secrets는 절대 코드에 포함하지 마세요
- .env 파일도 Git에 커밋하지 마세요 (.gitignore에 포함됨)
- SSH 키는 정기적으로 교체하세요
- 사용하지 않는 Secrets는 삭제하세요

## 문제 해결

### Secret을 잘못 입력한 경우
1. 해당 Secret 클릭
2. `Update` 버튼 클릭
3. 새 값 입력 후 저장

### Secret이 작동하지 않는 경우
1. Secret 이름이 정확한지 확인 (대소문자 구분)
2. 값에 공백이나 줄바꿈이 포함되지 않았는지 확인
3. GitHub Actions 로그에서 오류 메시지 확인

## 다음 단계

Secrets 설정이 완료되면:

1. ✅ 코드를 main 브랜치에 push
2. ✅ GitHub Actions가 자동으로 실행됨
3. ✅ 약 3-5분 후 EC2에 배포 완료
4. ✅ `http://13.125.99.225:8000/health` 접속하여 확인

자세한 내용은 `DEPLOYMENT.md`를 참조하세요.

