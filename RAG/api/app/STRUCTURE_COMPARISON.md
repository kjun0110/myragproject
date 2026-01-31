# API/APP 폴더 구조 비교 분석

## 📋 목표 구조 (Target Structure)

```
api/
└── app/
    ├── main.py                     # [Entry Point] 앱 초기화 및 도메인별 라우터 통합
    │
    ├── data/                       # 전역 사용 데이터 모음
    │   ├── training/
    │   ├── spam_classifier/
    │   ├── chat/
    │   └── ESG/
    │
    ├── models/                     # [Shared Model Assets] 공통 AI 모델 및 어댑터
    │   ├── gateway/              # KoELECTRA 관련 모델 파일 및 로더
    │   │   ├── koelectra_model/
    │   │   └── koelectra_adapter/  # 도메인 모델에 맞게 데이터를 변환하는 어댑터
    │   └── spam_classifier/                 # EXAONE 모델 자산 (다양한 LLM 지원)
    │       ├── exaone_model/
    │       └── exaone_adapter/
    │
    ├── orchestrator/                # [Application Coordination] 서비스 간 워크플로우 제어
    │   └── mcp_orchestrator/        # 여러 도메인(Spam, ESG 등)을 조합한 복합 시나리오 관리
    │
    ├── routers/                     # [Global API Interface] 외부 노출 엔드포인트
    │   └── mcp_router.py           # 오케스트레이터와 연결된 통합 API 경로
    │
    ├── domains/                     # [Bounded Contexts] 비즈니스 경계별 독립 모듈
    │   ├── training/               # 1. 모델 학습 및 튜닝 전용 도메인
    │   │   ├── data/
    │   │   │   ├── spam_classifier/
    │   │   │   │   ├── dataset/     # 트레이닝에 사용할 raw데이터, jsonl 변환 데이터
    │   │   │   │   ├── traing/      # trainig data
    │   │   │   │   ├── validation/  # validation data
    │   │   │   │   └── test/        # test data
    │   │   │   └── spam_agent/
    │   │   ├── agents/              # 학습 파이프라인 제어 에이전트
    │   │   │   ├── spam_classifier/
    │   │   │   │   ├Extract.py
    │   │   │   │   ├transform.py
    │   │   │   │   ├lora_adapter.py
    │   │   │   │   └load.py
    │   │   │   └── spam_agent/
    │   │   └── services/            # 학습 데이터 전처리 및 배치 로직
    │   │
    │   ├── spam_classifier/        # 2. 스팸 분류 추론 도메인 (핵심 엔진)
    │   │   ├── agents/              # [Reasoning] 추론 판단 로직
    │   │   │   └── verdict_agent/
    │   │   │       ├── graph.py    # LangGraph 기반의 추론 상태 및 흐름도
    │   │   │       └── model_loader.py # 사용 모델 및 아댑터 로드
    │   │   ├── services/            # 규칙 기반 서비스 로직
    │   │   ├── models/       # [Domain Model] 스팸 도메인 전용 데이터 규격
    │   │   │   ├── base_model.py   # 기본 데이터 구조 (Pydantic)
    │   │   │   ├── state_model.py  # 에이전트 상태값 정의
    │   │   │   └── vector_model.py # 벡터 DB 검색을 위한 스키마
    │   │   └── repositories/         # [Infrastructure] 영속성 계층
    │   │       └── db_handler.py   # 도메인 결과 저장 및 이력 관리
    │   │
    │   ├── chat/           # 3. 대화형 인터페이스 도메인
    │   └── ESG/            # 4. ESG 분석 전문 도메인
    │
    └── common/                     # [Cross-cutting Concerns] 공통 보안, 로그, 유틸리티
        └── config/
```

---

## 📊 현재 구조 (Current Structure)

```
api/
├── artifacts/                      ✅ [Shared Model Assets] 공통 AI 모델 및 어댑터
│   ├── exaone/                     ✅ EXAONE 모델 자산
│   │   ├── exaone3.5-2.4b/         ✅ EXAONE 3.5 2.4B 베이스 모델 파일
│   │   │                           # 스팸 분류 및 채팅에 사용되는 LLM 모델
│   │   └── spam_adapter/           ✅ EXAONE 스팸 분류용 LoRA 어댑터
│   │       └── exaone3.5-2.4b-spam-lora/
│   │           └── {timestamp}/   # 타임스탬프별 체크포인트 저장
│   │
│   ├── koelectra/                  ✅ KoELECTRA 게이트웨이 모델 (Policy Router)
│   │   ├── koelectra-small-v3-discriminator/  ✅ 베이스 모델
│   │   │                           # 정책 결정용 시퀀스 분류 모델
│   │   │                           # "이 데이터는 스팸 분석이 필요한가?" 정책 판단
│   │   └── spam_adapter/           ✅ KoELECTRA 정책 결정용 LoRA 어댑터
│   │       └── koelectra-small-v3-discriminator-spam-lora/
│   │           └── {timestamp}/   # 타임스탬프별 체크포인트 저장
│   │
└── app/
    ├── main.py                     ✅ [Entry Point] 채팅 서비스 서버 진입점
    │                               # FastAPI 앱 초기화, 채팅/그래프 라우터 통합
    │
    ├── agent.py                     ✅ [Entry Point] 스팸 분석 서버 진입점
    │                               # FastAPI 앱 초기화, MCP 스팸 라우터 통합
    │
    ├── routers/                    ✅ [Global API Interface] 외부 노출 엔드포인트
    │   ├── chat_router.py          ✅ 채팅 서비스 라우터
    │   │                           # POST /api/chat - 일반 채팅 API
    │   ├── graph_router.py          ✅ LangGraph 기반 채팅 라우터
    │   │                           # POST /api/graph - RAG 기반 대화형 채팅 API
    │   └── mcp_spam_router.py      ✅ MCP 스팸 분석 라우터
    │                               # POST /api/mcp/gate - KoELECTRA 정책 결정 API
    │                               # POST /api/mcp/spam-analyze - 스팸 분석 API
    │                               #   - /spam 페이지: KoELECTRA 건너뛰고 EXAONE 직접 호출
    │                               #   - 기타 경로: KoELECTRA 정책 결정 → 필요시 EXAONE 호출
    │
    ├── domains/                    ✅ [Bounded Contexts] 비즈니스 경계별 독립 모듈
    │   │
    │   ├── training/               ✅ 1. 모델 학습 및 튜닝 전용 도메인
    │   │   ├── agents/             ✅ 학습 파이프라인 제어 에이전트
    │   │   │   ├── spam_classifier/ ✅ KoELECTRA 학습 에이전트
    │   │   │   │   ├── extract_jsonl.py      ✅ JSONL 데이터 추출
    │   │   │   │   ├── extract_dpo.py        ✅ DPO 데이터 추출
    │   │   │   │   ├── transform_cleanup_duplicates.py  ✅ 중복 제거
    │   │   │   │   ├── transform_data_preprocessor.py  ✅ 전처리
    │   │   │   │   ├── transform_data_splitter.py      ✅ 데이터 분할
    │   │   │   │   ├── transform_prepare_datasets.py  ✅ 데이터셋 준비
    │   │   │   │   ├── transform_dataset_utils.py      ✅ 데이터셋 유틸리티
    │   │   │   │   ├── transform_tokenizer_utils.py   ✅ 토크나이저 유틸리티
    │   │   │   │   ├── lora_adapter.py       ✅ LoRA 어댑터 학습 로직
    │   │   │   │   ├── lora_adapter2.py      ✅ LoRA 어댑터 학습 로직 (추가)
    │   │   │   │   ├── load_model.py         ✅ 모델 로드
    │   │   │   │   └── test_model.py         ✅ 모델 테스트
    │   │   │   └── spam_agent/     ✅ EXAONE 학습 에이전트
    │   │   │       ├── extract_jsonl.py      ✅ JSONL 데이터 추출
    │   │   │       ├── extract_dpo.py        ✅ DPO 데이터 추출
    │   │   │       ├── transform_cleanup_duplicates.py  ✅ 중복 제거
    │   │   │       ├── transform_data_preprocessor.py  ✅ 전처리
    │   │   │       ├── transform_data_splitter.py      ✅ 데이터 분할
    │   │   │       ├── transform_prepare_datasets.py  ✅ 데이터셋 준비
    │   │   │       ├── transform_dataset_utils.py      ✅ 데이터셋 유틸리티
    │   │   │       ├── transform_tokenizer_utils.py   ✅ 토크나이저 유틸리티
    │   │   │       ├── lora_adapter.py       ✅ LoRA 어댑터 학습 로직
    │   │   │       ├── load_model.py         ✅ 모델 로드
    │   │   │       ├── README.md             ✅ 문서
    │   │   │       └── TRAINING_ANALYSIS.md  ✅ 학습 분석 문서
    │   │   │
    │   │   └── services/           ✅ 학습 데이터 전처리 및 배치 로직
    │   │                           # 현재 비어있음 (향후 확장용)
    │   │
    │   ├── spam_classifier/        ✅ 2. 스팸 분류 추론 도메인 (핵심 엔진)
    │   │   ├── orchestrator/       ✅ [Application Coordination] 도메인별 오케스트레이터
    │   │   │   ├── __init__.py     ✅ 모듈 진입점
    │   │   │   ├── koelectra_loader.py  ✅ KoELECTRA 모델 로더 (캐싱)
    │   │   │   └── spam_orchestrator.py ✅ 스팸 분류 오케스트레이터
    │   │   │                           # KoELECTRA 정책 결정 → agents/services 라우팅
    │   │   │                           # ANALYZE_SPAM → agents/ (EXAONE 등 AI 기반)
    │   │   │                           # BYPASS → services/ (규칙 기반)
    │   │   │
    │   │   ├── agents/             ✅ [Reasoning] 추론 판단 로직 (정책 관련)
    │   │   │   ├── __init__.py     ✅ 모듈 진입점
    │   │   │   ├── graph.py        ✅ LangGraph 기반 추론 상태 및 흐름도
    │   │   │   │                   # EXAONE 스팸 분석 툴 정의 및 그래프 구성
    │   │   │   ├── model_loader.py ✅ EXAONE 모델 및 어댑터 로더
    │   │   │   │                   # 전역 캐싱을 통한 모델 로드 관리
    │   │   │   │                   # LoRA 어댑터 자동 탐색 및 로드 기능 구현
    │   │   │   └── exaone_model.py ✅ EXAONE 모델 구현체
    │   │   │                       # BaseLLM을 상속한 EXAONE LLM 래퍼
    │   │   │                       # PeftModel을 통한 LoRA 어댑터 로드 지원
    │   │   │
    │   │   ├── services/           ✅ 규칙 기반 서비스 로직
    │   │   │   └── __init__.py     ✅ 모듈 진입점 (현재 비어있음)
    │   │   │
    │   │   ├── models/            ✅ [Domain Model] 스팸 도메인 전용 데이터 규격
    │   │   │   ├── __init__.py     ✅ 모듈 진입점
    │   │   │   ├── base_model.py   ✅ 기본 데이터 구조 (Pydantic)
    │   │   │   │                   # GateRequest, SpamAnalyzeRequest 등 API 스키마
    │   │   │   │                   # SpamAnalyzeResponse.exaone_result: Optional[dict]
    │   │   │   └── state_model.py  ✅ 에이전트 상태값 정의
    │   │   │                       # VerdictAgentState - LangGraph 상태 모델
    │   │   │
    │   │   ├── bases/              ✅ [Domain Base] 스팸 도메인 베이스 클래스
    │   │   │   └── vector_model.py ✅ 벡터 DB 검색을 위한 스키마
    │   │   │                       # 벡터 검색 관련 요청/응답 모델
    │   │   │
    │   │   └── repositories/       ✅ [Infrastructure] 영속성 계층
    │   │       └── __init__.py     ✅ 모듈 진입점 (현재 비어있음)
    │   │                           # db_handler.py 구현 필요
    │   │
    │   ├── chat/                   ✅ 3. 대화형 인터페이스 도메인
    │   │   ├── orchestrator/       ✅ [Application Coordination] 도메인별 오케스트레이터
    │   │   │   ├── __init__.py     ✅ 모듈 진입점
    │   │   │   └── chat_orchestrator.py ✅ 채팅 오케스트레이터
    │   │   │                           # 복잡도 분석 → agents/services 라우팅
    │   │   │                           # 복잡한 대화 → agents/ (LLM 기반)
    │   │   │                           # 간단한 응답 → services/ (템플릿/규칙 기반)
    │   │   │
    │   │   ├── agents/             ✅ 채팅 에이전트 및 모델 관리 (정책 관련)
    │   │   │   ├── __init__.py     ✅ 모듈 진입점
    │   │   │   ├── chat_service.py ✅ 채팅 서비스 구현체
    │   │   │   │                   # LangChain 기반 채팅 서비스
    │   │   │   ├── graph.py        ✅ LangGraph 기반 RAG 채팅 그래프
    │   │   │   │                   # 벡터 검색 + LLM을 통한 대화형 RAG
    │   │   │   ├── model_loader.py ✅ EXAONE 모델 로더
    │   │   │   │                   # 전역 캐싱을 통한 모델 로드 관리
    │   │   │   └── exaone_model.py ✅ EXAONE 모델 구현체 (채팅용)
    │   │   │
    │   │   ├── services/           ✅ 채팅 서비스 로직 (규칙 기반)
    │   │   │   └── __init__.py     ✅ 모듈 진입점 (현재 비어있음)
    │   │   │
    │   │   ├── models/            ✅ 채팅 도메인 데이터 규격
    │   │   │   ├── __init__.py     ✅ 모듈 진입점
    │   │   │   ├── base_model.py   ✅ GraphRequest, GraphResponse 등 API 스키마
    │   │   │   └── state_model.py  ✅ ChatAgentState - LangGraph 상태 모델
    │   │   │
    │   │   ├── bases/              ✅ [Domain Base] 채팅 도메인 베이스 클래스
    │   │   │   └── vector_model.py ✅ 벡터 검색 스키마
    │   │   │
    │   │   └── repositories/       ✅ 채팅 영속성 계층
    │   │       └── __init__.py     ✅ 모듈 진입점 (현재 비어있음)
    │   │
    │   └── ESG/                    ❌ 없음
    │                               # ESG 분석 전문 도메인 (미구현)
    │
    └── common/                     ✅ [Cross-cutting Concerns] 공통 보안, 로그, 유틸리티
        ├── config/                 ✅ 설정 관리
        │   ├── __init__.py         ✅ 모듈 진입점
        │   ├── config.py           ✅ 환경 변수 및 설정 로드
        │   └── settings.py        ✅ Pydantic 기반 설정 모델
        │
        ├── database/               ✅ 데이터베이스 관련 공통 기능
        │   ├── __init__.py         ✅ 모듈 진입점
        │   └── vector_store.py    ✅ 벡터 스토어 관리
        │                           # PGVector 기반 벡터 검색 기능
        │
        ├── agents/                 ✅ 공통 모델 인터페이스 및 유틸리티
        │   ├── __init__.py         ✅ 모듈 진입점
        │   ├── base.py             ✅ BaseLLM 추상 클래스
        │   │                       # 모든 LLM 모델의 공통 인터페이스
        │   ├── factory.py           ✅ 모델 팩토리
        │   │                       # 모델 타입별 인스턴스 생성 및 관리
        │   └── utils.py            ✅ 모델 관련 유틸리티 함수
        │                           # resolve_model_path() 등 경로 해석 함수
        │
        └── orchestrator/           ✅ [Application Coordination] 공통 오케스트레이터
            ├── __init__.py         ✅ 모듈 진입점
            ├── base_orchestrator.py ✅ 추상 베이스 클래스
            │                       # 모든 오케스트레이터의 공통 인터페이스
            └── factory.py          ✅ 오케스트레이터 팩토리
                                    # 도메인별 오케스트레이터 자동 등록 및 관리
```

---

## 🏗️ 오케스트레이터 아키텍처

### 아키텍처 개요

```
라우터 (api/app/routers/)
    ↓
도메인별 오케스트레이터 (api/app/domains/{domain}/orchestrator/)
    ↓
KoELECTRA 판단
    ├─ 정책 관련 기능 → agents/ 폴더 기능 사용
    └─ 규칙 기반 기능 → services/ 폴더 기능 사용
```

### 공통 오케스트레이터 (전역 설정)

**위치**: `api/app/common/orchestrator/`

- `base_orchestrator.py`: 추상 베이스 클래스
  - 모든 오케스트레이터의 공통 인터페이스 정의
  - `classify_domain()`: 정책 결정 메서드
  - `analyze()`: 분석 수행 메서드
  - `should_use_agents()`: agents vs services 결정 로직
  - 상태 관리 메서드 (`get_state`, `delete_state`, `list_states`)

- `factory.py`: 오케스트레이터 팩토리
  - 도메인별 오케스트레이터 자동 등록
  - 싱글톤 패턴 지원
  - `OrchestratorFactory.get("spam_classifier")` 형태로 사용

### 도메인별 오케스트레이터

#### 1. 스팸 분류 도메인 (`api/app/domains/spam_classifier/orchestrator/`)

**역할:**
- KoELECTRA로 정책 결정 (ANALYZE_SPAM vs BYPASS)
- ANALYZE_SPAM → `agents/` 폴더 (EXAONE 등 AI 기반 분석)
- BYPASS → `services/` 폴더 (규칙 기반 처리)

**주요 클래스:**
```python
class SpamClassifierOrchestrator(BaseOrchestrator):
    def classify_domain(self, email_text: str) -> Dict[str, Any]:
        """
        KoELECTRA로 정책 결정:
        - spam_prob > 0.3 → ANALYZE_SPAM (agents 사용)
        - spam_prob <= 0.3 → BYPASS (services 사용)
        """

    def analyze(self, email_text: str) -> Dict[str, Any]:
        """
        1. KoELECTRA 정책 결정
        2. ANALYZE_SPAM → agents/get_exaone_tool() 호출
        3. BYPASS → services 호출 (향후 구현)
        """
```

#### 2. 채팅 도메인 (`api/app/domains/chat/orchestrator/`)

**역할:**
- 메시지 복잡도 분석
- 복잡한 대화 → `agents/` 폴더 (LLM 기반 응답)
- 간단한 응답 → `services/` 폴더 (템플릿/규칙 기반)

**주요 클래스:**
```python
class ChatOrchestrator(BaseOrchestrator):
    def classify_domain(self, message: str) -> Dict[str, Any]:
        """
        복잡도 분석:
        - 복잡도 > 0.5 → USE_LLM (agents 사용)
        - 복잡도 <= 0.5 → USE_RULES (services 사용)
        """

    def analyze(self, message: str) -> Dict[str, Any]:
        """
        1. 복잡도 분석
        2. USE_LLM → agents 호출 (LLM 기반)
        3. USE_RULES → services 호출 (규칙 기반)
        """
```

### 사용 방법

#### 1. 팩토리를 통한 오케스트레이터 사용
```python
from app.common.orchestrator.factory import OrchestratorFactory

# 스팸 분류 오케스트레이터 가져오기
spam_orch = OrchestratorFactory.get("spam_classifier")

# 분석 수행
result = spam_orch.analyze("스팸 메시지 내용")
```

#### 2. 직접 인스턴스 생성
```python
from app.domains.spam_classifier.orchestrator import SpamClassifierOrchestrator

spam_orch = SpamClassifierOrchestrator()
result = spam_orch.analyze("스팸 메시지 내용")
```

### 응답 형식

#### SpamClassifierOrchestrator 응답
```json
{
  "gate_result": {
    "domain": "spam",
    "policy": "ANALYZE_SPAM",
    "confidence": "high",
    "spam_prob": 0.85,
    "ham_prob": 0.15,
    "latency_ms": 42.5,
    "use_agents": true
  },
  "agent_result": {
    "spam_prob": 0.92,
    "label": "spam",
    "confidence": "high",
    "analysis": "..."
  },
  "service_result": null,
  "final_decision": "[AGENTS 폴더] EXAONE 스팸 분석: ..."
}
```

#### ChatOrchestrator 응답
```json
{
  "gate_result": {
    "domain": "chat",
    "policy": "USE_LLM",
    "confidence": "high",
    "complexity": 0.75,
    "use_agents": true
  },
  "agent_result": {
    "response": "LLM 기반 응답..."
  },
  "service_result": null,
  "final_response": "LLM 기반 응답..."
}
```

---

## 🔍 주요 차이점 분석

### ✅ 일치하는 부분

1. **기본 구조**: `domains/`, `routers/`, `common/` 폴더 구조 일치
2. **Training 도메인**: `agents/`, `data/`, `services/` 구조 일치
3. **Spam Classifier 도메인**: `agents/`, `models/`, `repositories/`, `services/`, `orchestrator/` 구조 일치
4. **Chat 도메인**: 기본 구조 일치, `orchestrator/` 추가됨
5. **Common 폴더**: `config/`, `database/`, `agents/`, `orchestrator/` 구조 일치

### ⚠️ 차이점 및 개선 필요 사항

#### 1. **데이터 폴더 위치**
- **목표**: `api/app/data/` (전역)
- **현재**: `api/app/domains/training/data/` (도메인 내부)
- **조치**: 데이터를 전역 `data/` 폴더로 이동 또는 목표 구조 수정

#### 2. **Training 데이터 구조**
- **목표**: `dataset/`, `training/`, `validation/`, `test/` 폴더로 분리
- **현재**: `koelectra/`, `exaone/` 폴더에 모든 데이터 혼재
- **조치**: 데이터를 목표 구조에 맞게 재구성

#### 3. **Spam Classifier Agents 구조**
- **목표**: `agents/verdict_agent/` 하위 폴더
- **현재**: `agents/` 바로 아래에 파일들
- **조치**: `verdict_agent/` 폴더 생성 후 이동 (선택적)

#### 4. **Models 파일 이름**
- **목표**: `spam_base_model.py`, `spam_state.py`, `spam_vector.py`
- **현재**: `base_model.py`, `state_model.py`, `vector_model.py`
- **조치**: 파일 이름 변경 (도메인별 구분) - 선택적

#### 5. **Routers 파일 이름**
- **목표**: `mcp_router.py`
- **현재**: `mcp_spam_router.py`
- **조치**: 파일 이름 변경 또는 목표 구조 수정 (선택적)

#### 6. **Repositories 구현**
- **목표**: `repositories/db_handler.py`
- **현재**: `repositories/` 폴더만 존재 (비어있음)
- **조치**: `db_handler.py` 구현 필요

#### 7. **ESG 도메인**
- **목표**: `domains/ESG/` 존재
- **현재**: 없음
- **조치**: ESG 도메인 생성 필요 (향후)

#### 8. **Models 폴더 위치**
- **목표**: `api/app/models/`
- **현재**: `api/artifacts/` (프로젝트 루트)
- **조치**: 현재 구조 유지 (프로젝트 루트가 더 적합)

#### 9. **Orchestrator 구조**
- **목표**: `api/app/orchestrator/` (전역)
- **현재**: `api/app/common/orchestrator/` (공통) + `api/app/domains/{domain}/orchestrator/` (도메인별)
- **조치**: ✅ 현재 구조가 더 적합 (도메인별 분리)

#### 10. **Schemas → Models 변경**
- **목표**: `schemas/` 폴더
- **현재**: `models/` 폴더로 변경됨
- **조치**: ✅ 변경 완료

#### 11. **Common Models → Agents 변경**
- **목표**: `common/models/`
- **현재**: `common/agents/`로 변경됨
- **조치**: ✅ 변경 완료

---

## 📝 권장 조치 사항

### 우선순위 높음 (P1)
1. ✅ **Orchestrator 구조 재설계**: 도메인별 오케스트레이터 구현 완료
2. ✅ **Schemas → Models 변경**: 도메인별 models 폴더로 변경 완료
3. ✅ **Common Models → Agents 변경**: `common/agents/`로 변경 완료
4. ✅ **Models 폴더 이동**: `api/artifacts/`로 이동 완료
5. ⚠️ **Repositories 구현**: `db_handler.py` 생성 필요

### 우선순위 중간 (P2)
6. ⚠️ **Training 데이터 구조 정리**: `dataset/`, `training/`, `validation/`, `test/` 폴더로 분리
7. ⚠️ **Spam Classifier Agents 구조**: `verdict_agent/` 폴더 생성 (선택적)
8. ⚠️ **데이터 폴더 위치**: 전역 `data/` vs 도메인 내부 `data/` 결정

### 우선순위 낮음 (P3)
9. ⚠️ **Routers 파일 이름**: `mcp_spam_router.py` → `mcp_router.py` (선택적)
10. ⚠️ **ESG 도메인**: 향후 필요 시 생성
11. ⚠️ **Models 파일 이름**: 도메인 prefix 추가 (선택적)

---

## 📌 참고 사항

- 현재 구조는 대부분 목표 구조와 일치하며, 일부 세부 사항만 조정이 필요합니다.
- `common/database/`, `common/agents/`, `common/orchestrator/`는 목표 구조에 명시되지 않았지만 유용한 구조입니다.
- Training 데이터는 현재 구조가 더 명확할 수 있습니다 (모델별로 분리).
- 파일 이름 변경은 import 경로 수정이 필요하므로 신중히 진행해야 합니다.
- `api/app/orchestrator/` 폴더는 **삭제 완료**되었고, 모든 기능이 도메인별 오케스트레이터로 이동되었습니다.

---

## 🔄 최근 변경 사항 (Recent Changes)

### 1. **오케스트레이터 구조 재설계** ✅
- **이전**: `api/app/orchestrator/` (전역 단일 오케스트레이터)
- **현재**:
  - `api/app/common/orchestrator/` (공통 베이스 및 팩토리)
  - `api/app/domains/{domain}/orchestrator/` (도메인별 오케스트레이터)
- **장점**: 도메인별 독립성, 확장성 향상

### 2. **KoELECTRA 역할 재정의** (Policy Router)
- **이전**: 스팸 판단 수행
- **현재**: 정책 결정자(Policy Router) - "이 데이터는 스팸 분석 파이프라인(EXAONE)으로 보낼 대상인가?" 판단
- **결과**: `"ANALYZE_SPAM"` (분석 필요) 또는 `"BYPASS"` (건너뛰기)

### 3. **EXAONE 역할 명확화** (The Specialist)
- 정책이 `"ANALYZE_SPAM"`이면 EXAONE이 스팸 확률 계산 및 판단을 전부 수행
- LoRA 어댑터 자동 로드 기능 구현 완료

### 4. **`/spam` 페이지 동작 변경**
- `/spam` 페이지에서 `/api/spam-analyze` 호출 시: KoELECTRA를 건너뛰고 EXAONE 직접 호출
- 신뢰도 관계없이 모든 판독은 EXAONE이 담당

### 5. **스키마 타입 변경**
- `SpamAnalyzeResponse.exaone_result`: `Optional[str]` → `Optional[dict]`
- EXAONE 결과가 dict 형태로 반환되도록 수정

### 6. **어댑터 로드 기능 구현**
- `ExaoneLLM` 클래스에 `adapter_path` 파라미터 추가
- `PeftModel.from_pretrained()`를 통한 LoRA 어댑터 자동 로드 구현

### 7. **폴더 구조 변경**
- `api/app/common/models/` → `api/app/common/agents/` ✅
- `api/app/domains/{domain}/schemas/` → `api/app/domains/{domain}/models/` ✅
- `api/app/models/` → `api/artifacts/` ✅
- `api/app/orchestrator/` → 삭제, `api/app/common/orchestrator/` + 도메인별 오케스트레이터로 분리 ✅

### 8. **Agents vs Services 분리**
- **agents/**: 정책 관련 기능 (AI 모델 기반 복잡한 로직)
- **services/**: 규칙 기반 기능 (비즈니스 로직, 템플릿 응답)
- 오케스트레이터가 KoELECTRA 판단 결과에 따라 자동 라우팅

---

## 🚀 향후 확장 가이드

### 새로운 도메인 추가
1. `api/app/domains/{new_domain}/orchestrator/` 폴더 생성
2. `{new_domain}_orchestrator.py` 구현 (BaseOrchestrator 상속)
3. `api/app/common/orchestrator/factory.py`의 `_register_domain_orchestrators()`에 등록 추가

### agents/services 폴더 기능 추가
- `agents/`: AI 모델 기반 복잡한 로직
- `services/`: 규칙 기반 비즈니스 로직

### 라우터에서 오케스트레이터 사용
```python
from app.common.orchestrator.factory import OrchestratorFactory

@router.post("/endpoint")
async def endpoint(request: Request):
    orchestrator = OrchestratorFactory.get("domain_name")
    result = orchestrator.analyze(request.text)
    return result
```
