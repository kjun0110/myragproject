# API/APP2 → API/APP 마이그레이션 확인

## 📋 파일별 마이그레이션 상태

### ✅ 완전히 옮겨진 파일들

#### 1. **Routers**
| app2 경로 | app 경로 | 상태 |
|-----------|----------|------|
| `router/chat_router.py` | `routers/chat_router.py` | ✅ 옮겨짐 |
| `router/graph_router.py` | `routers/graph_router.py` | ✅ 옮겨짐 |
| `router/mcp_router.py` | `routers/mcp_spam_router.py` | ✅ 옮겨짐 (이름 변경) |

#### 2. **Chat Service**
| app2 경로 | app 경로 | 상태 |
|-----------|----------|------|
| `service/chat_service/chat_service.py` | `domains/chat/agents/chat_service.py` | ✅ 옮겨짐 |
| `service/chat_service/graph.py` | `domains/chat/agents/graph.py` | ✅ 옮겨짐 |

#### 3. **Spam Classifier (Verdict Agent)**
| app2 경로 | app 경로 | 상태 |
|-----------|----------|------|
| `service/verdict_agent/graph.py` | `domains/spam_classifier/agents/graph.py` | ✅ 옮겨짐 |
| `service/verdict_agent/base_model.py` | `domains/spam_classifier/agents/model_loader.py` | ✅ 통합됨 |
| `service/verdict_agent/state_model.py` | `domains/spam_classifier/schemas/state_model.py` | ✅ 옮겨짐 |
| `service/verdict_agent/vector_model.py` | `domains/spam_classifier/schemas/vector_model.py` | ✅ 옮겨짐 |

#### 4. **Training - Spam Agent**
| app2 경로 | app 경로 | 상태 |
|-----------|----------|------|
| `service/training_spam_agent/*.py` | `domains/training/agents/spam_agent/*.py` | ✅ 옮겨짐 |
| `service/training_spam_agent/README.md` | `domains/training/agents/spam_agent/README.md` | ✅ 옮겨짐 |
| `service/training_spam_agent/TRAINING_ANALYSIS.md` | `domains/training/agents/spam_agent/TRAINING_ANALYSIS.md` | ✅ 옮겨짐 |

#### 5. **Training - Spam Classifier**
| app2 경로 | app 경로 | 상태 |
|-----------|----------|------|
| `service/training_spam_classifier/*.py` | `domains/training/agents/spam_classifier/*.py` | ✅ 옮겨짐 |

#### 6. **Model Service (일부)**
| app2 경로 | app 경로 | 상태 |
|-----------|----------|------|
| `service/model_service/base.py` | `common/models/base.py` | ✅ 옮겨짐 |
| `service/model_service/factory.py` | `common/models/factory.py` | ✅ 옮겨짐 |
| `service/model_service/loader.py` | `domains/*/agents/model_loader.py` (중복 구현) | ⚠️ 중복 |
| `service/model_service/exaone_model.py` | `domains/chat/agents/exaone_model.py`<br>`domains/spam_classifier/agents/exaone_model.py` | ✅ 옮겨짐 (2곳) |
| `service/model_service/midm_model.py` | `domains/chat/agents/midm_model.py` | ✅ 옮겨짐 |
| `service/model_service/exaone_loader.py` | `domains/chat/agents/model_loader.py`<br>`domains/spam_classifier/agents/model_loader.py` | ✅ 통합됨 |
| `service/model_service/midm_loader.py` | `domains/chat/agents/model_loader.py` | ✅ 통합됨 |
| `service/model_service/midm_model_loader.py` | (사용 안 함) | ⚠️ 미사용 |

#### 7. **Config**
| app2 경로 | app 경로 | 상태 |
|-----------|----------|------|
| `config/config.py` | `common/config/config.py` | ✅ 옮겨짐 |
| `config/settings.py` | `common/config/settings.py` | ✅ 옮겨짐 |

#### 8. **Database/Vector Store**
| app2 경로 | app 경로 | 상태 |
|-----------|----------|------|
| `app.py` (vector store) | `common/database/vector_store.py` | ✅ 옮겨짐 |
| `repository/vector_store.py` | `common/database/vector_store.py` | ✅ 통합됨 |

#### 9. **Orchestrator**
| app2 경로 | app 경로 | 상태 |
|-----------|----------|------|
| `router/mcp_router.py` (일부 로직) | `orchestrator/mcp_orchestrator.py` | ✅ 옮겨짐 |

---

### ⚠️ 옮겨지지 않았거나 미사용 파일들

#### 1. **Entry Point 파일들**
| app2 경로 | app 경로 | 상태 | 비고 |
|-----------|----------|------|------|
| `agent.py` | `main.py` | ❌ 미구현 | `agent.py`는 스팸 필터 전용, `main.py`는 통합 서버 |
| `api_server.py` | `main.py` | ❌ 미구현 | `main.py`가 비어있음 |
| `main.py` | - | ❌ 미사용 | app2의 main.py는 거의 비어있음 |

#### 2. **미사용/템플릿 파일들**
| app2 경로 | 상태 | 비고 |
|-----------|------|------|
| `service/embedding_ingest_service_t.py` | ❌ 미옮김 | 템플릿 파일 (8줄, 거의 비어있음) |
| `service/rag_service_t.py` | ❌ 미옮김 | 템플릿 파일 (14줄, 거의 비어있음) |
| `service/training_service_t.py` | ❌ 미옮김 | 템플릿 파일 (19줄, 거의 비어있음) |
| `router/rag_router_t.py` | ❌ 미옮김 | 템플릿 파일 (12줄, 거의 비어있음) |
| `router/training_router_t.py` | ❌ 미옮김 | 템플릿 파일 (12줄, 거의 비어있음) |

#### 3. **Repository 파일들**
| app2 경로 | app 경로 | 상태 | 비고 |
|-----------|----------|------|------|
| `repository/document_repository_t.py` | - | ❌ 미옮김 | 템플릿 파일 |
| `repository/model_checkpoint_repository_t.py` | - | ❌ 미옮김 | 템플릿 파일 |
| `repository/training_dataset_repository_t.py` | - | ❌ 미옮김 | 템플릿 파일 |
| `repository/vector_repository_t.py` | - | ❌ 미옮김 | 템플릿 파일 |
| `repository/vector_store.py` | `common/database/vector_store.py` | ✅ 통합됨 | `app.py`와 통합 |

#### 4. **문서 파일들**
| app2 경로 | app 경로 | 상태 |
|-----------|----------|------|
| `README.md` | - | ❌ 미옮김 (선택적) |
| `service/CODE_REFACTORING_STRATEGY.md` | - | ❌ 미옮김 (선택적) |
| `service/SPAM_FILTER_ARCHITECTURE_STRATEGY.md` | - | ❌ 미옮김 (선택적) |

---

## 🔍 기능별 마이그레이션 상태

### ✅ 완전히 옮겨진 기능들

1. **Chat Service (RAG Chain)**
   - ✅ `ChatService` 클래스
   - ✅ `initialize_embeddings()`, `initialize_llm()`
   - ✅ RAG 체인 생성 및 실행
   - ✅ 세션 관리

2. **Chat Graph (LangGraph)**
   - ✅ `graph.py` - LangGraph 구성
   - ✅ `preload_exaone_model()`
   - ✅ Tool calling 기능

3. **Spam Classifier**
   - ✅ KoELECTRA 게이트웨이 로직
   - ✅ EXAONE Reader (verdict_agent)
   - ✅ 오케스트레이터 로직
   - ✅ 상태 관리

4. **Training**
   - ✅ Spam Agent 학습 파이프라인
   - ✅ Spam Classifier 학습 파이프라인
   - ✅ 데이터 추출/변환 스크립트
   - ✅ LoRA 어댑터 학습

5. **Model Loading**
   - ✅ EXAONE 모델 로딩
   - ✅ MIDM 모델 로딩
   - ✅ KoELECTRA 모델 로딩
   - ✅ LoRA 어댑터 로딩

6. **Common Infrastructure**
   - ✅ BaseLLM, BaseEmbedding 인터페이스
   - ✅ LLMFactory, EmbeddingFactory
   - ✅ 설정 관리
   - ✅ Vector Store 관리

---

### ❌ 옮겨지지 않은 기능들

1. **api_server.py의 초기화 로직**
   - ❌ FastAPI 앱 생성 및 설정
   - ❌ ChatService 전역 초기화
   - ❌ Vector Store 초기화
   - ❌ 라우터 등록
   - **현재 상태**: `main.py`가 비어있음

2. **agent.py의 스팸 필터 전용 서버**
   - ❌ 독립적인 스팸 필터 FastAPI 서버
   - **현재 상태**: `main.py`에 통합 필요

3. **템플릿 파일들** (의도적으로 미옮김 가능)
   - ❌ `*_t.py` 파일들 (템플릿)
   - **비고**: 향후 구현을 위한 템플릿일 수 있음

---

## 📊 마이그레이션 완성도

### 전체 파일 기준
- **옮겨진 파일**: 약 50개
- **미옮김 파일**: 약 13개 (대부분 템플릿/문서)
- **완성도**: **약 80%**

### 핵심 기능 기준
- **옮겨진 기능**: 약 95%
- **미옮김 기능**:
  - FastAPI 서버 초기화 로직 (`main.py` 구현 필요)
  - 일부 템플릿 파일들
- **완성도**: **약 95%**

---

## ⚠️ 주의사항

### 1. **중복 구현**
- `resolve_model_path()` 함수가 여러 곳에 중복 구현됨
  - `domains/chat/agents/model_loader.py`
  - `domains/spam_classifier/agents/model_loader.py`
  - 원본: `app2/service/model_service/loader.py`

### 2. **Import 경로 문제**
- Training 도메인에서 여전히 `app.service.training_*` 경로 참조
  - `domains/training/agents/spam_agent/lora_adapter.py:33-34`
  - `domains/training/agents/spam_classifier/lora_adapter.py:31-32`
  - 수정 필요: `app.domains.training.agents.spam_agent`로 변경

### 3. **chat_router.py의 의존성**
- `app.api_server` 모듈을 참조하지만 `api/app`에는 없음
  - `chat_router.py:29-37`
  - 수정 필요: `main.py`에서 ChatService 초기화 후 접근 방식 변경

---

---

## 🔍 api_server.py 기능별 마이그레이션 상태

### ✅ 옮겨진 기능들

| api_server.py 함수 | app 위치 | 상태 |
|-------------------|----------|------|
| `wait_for_postgres()` | `common/database/vector_store.py:wait_for_postgres()` | ✅ 옮겨짐 |
| `initialize_vector_store()` | `common/database/vector_store.py:initialize_vector_store()` | ✅ 옮겨짐 |
| `initialize_embeddings()` | `domains/chat/agents/chat_service.py:initialize_embeddings()` | ✅ 옮겨짐 (ChatService 내부) |
| `initialize_llm()` | `domains/chat/agents/chat_service.py:initialize_llm()` | ✅ 옮겨짐 (ChatService 내부) |
| `create_rag_chain()` | `domains/chat/agents/chat_service.py:initialize_rag_chain()` | ✅ 옮겨짐 (ChatService 내부) |
| `initialize_rag_chain()` | `domains/chat/agents/chat_service.py:initialize_rag_chain()` | ✅ 옮겨짐 (ChatService 내부) |
| `startup_event()` | - | ❌ 미구현 (`main.py`에 필요) |

### ⚠️ 전역 변수 관리

| api_server.py 전역 변수 | app 위치 | 상태 |
|----------------------|----------|------|
| `vector_store` | `common/database/vector_store.py` (함수 내부) | ⚠️ 전역 변수 없음 |
| `openai_embeddings` | `chat_service.openai_embeddings` | ✅ ChatService 인스턴스 변수 |
| `local_embeddings` | `chat_service.local_embeddings` | ✅ ChatService 인스턴스 변수 |
| `openai_llm` | `chat_service.openai_llm` | ✅ ChatService 인스턴스 변수 |
| `local_llm` | `chat_service.local_llm` | ✅ ChatService 인스턴스 변수 |
| `chat_service` | - | ❌ 전역 변수 없음 (`main.py`에 필요) |

---

## ✅ 결론

### 잘 옮겨진 부분
- ✅ 모든 도메인 로직 (chat, spam_classifier, training)
- ✅ 모든 모델 로딩 기능
- ✅ 모든 라우터 (기능적으로)
- ✅ 공통 인프라 (common/)
- ✅ api_server.py의 모든 핵심 기능 (ChatService로 캡슐화됨)
- ✅ Vector Store 관리 기능

### 아직 완료되지 않은 부분
- ❌ `main.py` 구현 (FastAPI 서버 초기화 및 startup_event)
- ❌ Training 도메인 import 경로 수정
- ❌ `chat_router.py`의 `app.api_server` 의존성 제거
- ❌ 전역 `chat_service` 인스턴스 관리 (main.py에서 초기화 필요)

### 전체 평가
**마이그레이션 완성도: 약 95%**

핵심 기능은 모두 옮겨졌으며, 남은 작업은 주로:
1. 서버 진입점(`main.py`) 구현 - FastAPI 앱 생성, ChatService 초기화, 라우터 등록
2. 일부 import 경로 수정 - Training 도메인
3. 템플릿 파일들은 의도적으로 미옮김일 가능성

**결론**: 대부분의 파일과 기능은 잘 옮겨졌습니다. 남은 작업은 주로 통합 및 초기화 로직입니다.
