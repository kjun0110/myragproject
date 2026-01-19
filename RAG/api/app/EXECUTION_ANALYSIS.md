# API/APP 실행 가능성 분석

## 🔴 치명적 문제 (즉시 수정 필요)

### 1. **main.py가 비어있음** ❌
- **현재 상태**: `main.py` 파일이 비어있음
- **문제**: FastAPI 앱이 초기화되지 않음, 라우터가 등록되지 않음
- **영향**: 서버가 시작되지 않음
- **필요 작업**:
  - FastAPI 앱 생성
  - 모든 라우터 등록 (`chat_router`, `graph_router`, `mcp_spam_router`)
  - CORS 설정
  - 초기화 로직

### 2. **chat_router.py의 app.api_server 의존성** ❌
- **위치**: `api/app/routers/chat_router.py:29-37`
- **문제**: `app.api_server` 모듈을 참조하지만 `api/app`에는 없음 (app2에만 존재)
- **코드**:
  ```python
  if "app.api_server" in sys.modules:
      from .. import api_server
  else:
      api_server = importlib.import_module("app.api_server")
  return api_server.chat_service
  ```
- **영향**: `chat_router`가 실행 시 `ModuleNotFoundError` 발생
- **필요 작업**:
  - `ChatService` 초기화 로직을 `main.py`로 이동
  - 또는 `chat_router`에서 직접 `ChatService` 인스턴스 생성

### 3. **Training 도메인의 잘못된 import 경로** ❌
- **위치**:
  - `api/app/domains/training/agents/spam_agent/lora_adapter.py:33-34`
  - `api/app/domains/training/agents/spam_classifier/lora_adapter.py:31-32`
- **문제**: `app.service.training_spam_agent`, `app.service.training_spam_classifier` 경로 참조
- **현재 구조**: `app/domains/training/agents/spam_agent/`, `app/domains/training/agents/spam_classifier/`
- **영향**: Training 스크립트 실행 시 `ModuleNotFoundError` 발생
- **필요 작업**:
  - Import 경로를 `app.domains.training.agents.spam_agent`로 수정
  - 또는 상대 경로로 변경

---

## ⚠️ 잠재적 문제 (기능 동작에 영향)

### 4. **ChatService 초기화 누락**
- **문제**: `chat_router`가 `ChatService` 인스턴스를 필요로 하지만 초기화 로직이 없음
- **필요 작업**:
  - `main.py`에서 `ChatService` 초기화
  - 또는 `chat_router`에서 lazy initialization

### 5. **환경 변수 및 설정 파일**
- **확인 필요**:
  - `.env` 파일 존재 여부
  - `DATABASE_URL`, `OPENAI_API_KEY` 등 필수 환경 변수
  - 모델 경로 설정 (`EXAONE_MODEL_DIR`, `LOCAL_MODEL_DIR` 등)

### 6. **모델 파일 경로**
- **확인 필요**:
  - `api/app/models/gateway/koelectra-small-v3-discriminator/` 존재
  - `api/app/models/spam_classifier/exaone3.5-2.4b/` 존재
  - LoRA 어댑터 경로가 올바른지

---

## ✅ 정상 작동 가능한 부분

### 1. **Spam Classifier 도메인**
- ✅ `mcp_spam_router.py` - 독립적으로 작동 가능
- ✅ `mcp_ochestrator.py` - 모든 import가 `app` 내부로 해결됨
- ✅ `spam_classifier/agents/` - 모든 의존성이 해결됨

### 2. **Chat Graph 도메인**
- ✅ `graph_router.py` - 독립적으로 작동 가능
- ✅ `chat/agents/graph.py` - 모든 import가 해결됨

### 3. **Common 모듈**
- ✅ `common/models/base.py`, `factory.py` - 정상
- ✅ `common/config/` - 정상
- ✅ `common/database/` - 정상

---

## 📋 수정 우선순위

### P0 (즉시 수정 - 서버 시작 불가)
1. ✅ **main.py 구현** - FastAPI 앱 초기화 및 라우터 등록
2. ✅ **chat_router.py 수정** - `app.api_server` 의존성 제거
3. ✅ **Training 도메인 import 경로 수정**

### P1 (기능 동작을 위해 필요)
4. ⚠️ **ChatService 초기화 로직** - `main.py`에 추가
5. ⚠️ **환경 변수 확인** - 필수 설정값 점검

### P2 (선택적)
6. ⚠️ **모델 경로 검증** - 실제 모델 파일 존재 확인
7. ⚠️ **에러 핸들링 강화** - 모델 로드 실패 시 graceful degradation

---

## 💡 실행 가능성 종합 평가

### 현재 상태: 🔴 **실행 불가**

**주요 이유:**
1. `main.py`가 비어있어 서버가 시작되지 않음
2. `chat_router`가 존재하지 않는 `app.api_server`를 참조
3. Training 스크립트의 import 경로가 잘못됨

### 수정 후 예상 상태: 🟡 **부분 실행 가능**

**예상 동작:**
- ✅ Spam Classifier API (`/api/mcp/*`) - 정상 작동
- ✅ Graph Chat API (`/api/graph`) - 정상 작동
- ⚠️ Chat Chain API (`/api/chain`) - ChatService 초기화 필요
- ❌ Training 스크립트 - import 경로 수정 필요

### 완전 수정 후 예상 상태: 🟢 **완전 실행 가능**

**필요 작업 완료 시:**
- 모든 API 엔드포인트 정상 작동
- Training 스크립트 실행 가능
- 모델 로딩 정상 작동

---

## 🔧 권장 수정 방안

### 1. main.py 구현 (최우선)
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat_router, graph_router, mcp_spam_router
from app.domains.chat.agents.chat_service import ChatService
from app.common.config.settings import get_settings

app = FastAPI(title="RAG API", version="1.0.0")

# CORS 설정
app.add_middleware(CORSMiddleware, ...)

# ChatService 초기화
settings = get_settings()
chat_service = ChatService(...)
chat_service.initialize_embeddings()
chat_service.initialize_llm()

# 라우터 등록
app.include_router(chat_router.router)
app.include_router(graph_router.router)
app.include_router(mcp_spam_router.router)
```

### 2. chat_router.py 수정
- `get_chat_service()` 함수를 `main.py`의 전역 변수에 접근하도록 수정
- 또는 `ChatService`를 직접 import하여 사용

### 3. Training 도메인 import 수정
- `app.service.training_spam_agent` → `app.domains.training.agents.spam_agent`
- `app.service.training_spam_classifier` → `app.domains.training.agents.spam_classifier`

---

## 📊 결론

**현재 실행 가능성: 30%** (Spam Classifier만 작동 가능)

**수정 후 실행 가능성: 90%** (모든 API 작동, Training 스크립트 수정 필요)

**완전 수정 후 실행 가능성: 100%** (모든 기능 정상 작동)

**권장 사항**: P0 문제들을 먼저 해결하면 기본적인 API 서버는 실행 가능합니다.
