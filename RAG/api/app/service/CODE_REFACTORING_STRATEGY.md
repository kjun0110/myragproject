# 스팸 필터 코드 리팩토링 전략

## 📋 목차
1. [현재 구조 분석](#현재-구조-분석)
2. [코드 사용 흐름](#코드-사용-흐름)
3. [각 파일 목적별 평가](#각-파일-목적별-평가)
4. [주요 문제점](#주요-문제점)
5. [권장 구조](#권장-구조)
6. [구체적 개선 액션 아이템](#구체적-개선-액션-아이템)

---

## 📊 현재 구조 분석

### 시스템 아키텍처
```
┌─────────────────────────────────────────────────┐
│  Frontend (localhost:3000/spam)                │
│  사용자가 이메일 입력                             │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  mcp_router.py (라우터)                         │
│  ├─ POST /api/mcp/spam-analyze                 │
│  │  1) KoELECTRA 게이트웨이 호출                │
│  │  2) should_call_exaone() 판단                │
│  │  3) EXAONE 툴 호출 (필요시)                  │
│  │  4) 최종 결과 반환                           │
│  └─ 문제: KoELECTRA 모델 로드 기능 포함         │
└─────────────────┬───────────────────────────────┘
                  │
         ┌────────┴────────┐
         ▼                 ▼
┌──────────────┐  ┌─────────────────────┐
│ KoELECTRA    │  │ verdict_agent/      │
│ (라우터 내부) │  │ gragh.py           │
└──────────────┘  │ ├─ @tool wrapper    │
                  │ ├─ EXAONE 모델 로드 │
                  │ └─ LangGraph 구성   │
                  └─────────────────────┘
```

### 파일 구조
```
api/app/
├── router/
│   └── mcp_router.py              # FastAPI 라우터
│
└── service/
    └── verdict_agent/             # EXAONE 판독기
        ├── __init__.py
        ├── gragh.py               # 판독 기능 + 모델 로드 (문제)
        ├── state_model.py         # 상태 정의 (완벽)
        └── base_model.py          # 비어있음 (구현 필요)
```

---

## 🎯 코드 사용 흐름

### 1️⃣ 사용자가 이메일 입력 (Frontend)
```http
POST http://localhost:3000/api/mcp/spam-analyze
Content-Type: application/json

{
  "email_text": "이메일 내용..."
}
```

### 2️⃣ `mcp_router.py`에서 처리
```python
# api/app/router/mcp_router.py (354-427줄)
@router.post("/spam-analyze")
async def spam_analyze_with_tool(request: SpamAnalyzeRequest):
    # 1단계: KoELECTRA 게이트웨이로 스팸 확률 계산
    gate_result = predict_spam_probability(request.email_text)
    # 결과: {spam_prob: 0.65, label: "spam", confidence: "medium"}

    # 2단계: EXAONE 호출 여부 결정
    should_call = should_call_exaone(gate_result["spam_prob"])
    # 0.35 ~ 0.8 범위면 True (애매한 구간)

    # 3단계: EXAONE 툴 호출 (필요시)
    if should_call:
        exaone_tool = get_exaone_tool()
        exaone_result = exaone_tool.invoke({
            "email_text": request.email_text,
            "spam_prob": spam_prob,
            "label": gate_result["label"],
            "confidence": gate_result["confidence"],
        })

    # 4단계: 최종 결과 반환
    return SpamAnalyzeResponse(
        gate_result=gate_result,
        exaone_result=exaone_result,
        final_decision=final_decision
    )
```

### 3️⃣ `verdict_agent/gragh.py`에서 EXAONE 실행
```python
# api/app/service/verdict_agent/gragh.py (152-200줄)
@tool
def exaone_spam_analyzer(email_text: str, spam_prob: float,
                        label: str, confidence: str) -> str:
    # 1단계: EXAONE 모델 로드 (전역 캐싱)
    llm = load_exaone_reader()  # 31-63줄

    # 2단계: LangChain 메시지 구성
    messages = [
        SystemMessage(content="스팸 메일 분석 전문가..."),
        HumanMessage(content=prompt)
    ]

    # 3단계: EXAONE 호출
    response = llm.invoke(messages)

    # 4단계: 결과 반환
    return response.content
```

---

## 🔍 각 파일 목적별 평가

### 1. `mcp_router.py` (라우터)
**목적**: 라우트 기능만 담당

#### 현재 상태
- ✅ FastAPI 엔드포인트 정의 잘 됨
- ✅ Pydantic 모델 정의 잘 됨 (228-351줄)
- ❌ **문제**: KoELECTRA 모델 로드 코드 포함 (46-111줄)
- ❌ **문제**: 게이트웨이 예측 로직 포함 (114-192줄)
- ❌ **문제**: 비즈니스 로직 포함 (194-224줄)

#### 현재 책임 (SRP 위반)
```python
mcp_router.py
├─ 라우팅 (✅)
├─ 모델 로드 (❌ 분리 필요)
├─ 예측 로직 (❌ 분리 필요)
├─ 비즈니스 로직 (❌ 분리 필요)
└─ 상태 관리 (❌ 분리 필요)
```

#### 개선 필요사항
1. **KoELECTRA 모델 로드 분리**
   - `load_koelectra_gate()` (46-111줄) → `gate_agent/base_model.py`로 이동

2. **게이트웨이 예측 로직 분리**
   - `predict_spam_probability()` (114-192줄) → `gate_agent/classifier.py`로 이동
   - `should_call_exaone()` (194-224줄) → `gate_agent/decision.py`로 이동

3. **상태 관리 분리**
   - `_gate_results_cache` (36줄) → `gate_agent/state_manager.py`로 이동
   - 상태 관리 엔드포인트 (290-336줄) → 별도 라우터로 분리

4. **라우터는 순수하게 유지 (이상적인 형태)**
   ```python
   # mcp_router.py (개선 후)
   from app.service.gate_agent import GateClassifier
   from app.service.verdict_agent import VerdictAnalyzer

   @router.post("/spam-analyze")
   async def spam_analyze(request: SpamAnalyzeRequest):
       # 1. 게이트웨이 분석
       gate_result = GateClassifier.predict(request.email_text)

       # 2. EXAONE 호출 여부 판단
       if GateClassifier.should_call_exaone(gate_result["spam_prob"]):
           exaone_result = VerdictAnalyzer.analyze(
               request.email_text,
               gate_result
           )
       else:
           exaone_result = None

       # 3. 응답 조합
       return build_response(gate_result, exaone_result)
   ```

---

### 2. `verdict_agent/gragh.py` (판독 기능)
**목적**: EXAONE 판독 기능 담당

#### 현재 상태
- ✅ EXAONE Reader 기능 구현 잘 됨
- ✅ LangChain @tool 래핑 잘 됨 (152-200줄)
- ✅ LangGraph 구성 잘 됨 (123-148줄)
- ❌ **문제**: EXAONE 모델 로드 코드 포함 (31-63줄)
- ❌ **중복**: `exaone_reader_node`와 `exaone_spam_analyzer` 거의 동일 (67-119줄 vs 152-200줄)

#### 코드 중복 문제
```python
# 중복 1: exaone_reader_node (67-119줄)
def exaone_reader_node(state: VerdictAgentState):
    llm = load_exaone_reader()
    prompt = f"""다음 이메일이 스팸인지 판별..."""
    messages = [SystemMessage(...), HumanMessage(...)]
    response = llm.invoke(messages)
    return {...}

# 중복 2: exaone_spam_analyzer (152-200줄)
@tool
def exaone_spam_analyzer(email_text: str, spam_prob: float, ...):
    llm = load_exaone_reader()
    prompt = f"""다음 이메일이 스팸인지 판별..."""
    messages = [SystemMessage(...), HumanMessage(...)]
    response = llm.invoke(messages)
    return result
```

#### 개선 필요사항
1. **모델 로드 분리**
   ```python
   # gragh.py (개선 전)
   def load_exaone_reader():  # 31-63줄
       # EXAONE 모델 로드 로직
       ...

   # → base_model.py (개선 후)
   def load_exaone_model():
       # EXAONE 모델 로드 로직
       ...
   ```

2. **코드 중복 제거**
   ```python
   # gragh.py (개선 후)
   def _execute_exaone_analysis(email_text: str, gate_result: dict) -> str:
       """EXAONE 분석 공통 로직 (내부 함수)"""
       llm = load_exaone_model()
       prompt = _build_prompt(email_text, gate_result)
       messages = [SystemMessage(...), HumanMessage(prompt)]
       response = llm.invoke(messages)
       return response.content

   # LangGraph 노드용
   def exaone_reader_node(state: VerdictAgentState):
       result = _execute_exaone_analysis(
           state["email_text"],
           state["gate_result"]
       )
       return {**state, "exaone_result": result}

   # Tool용
   @tool
   def exaone_spam_analyzer(email_text: str, spam_prob: float, ...):
       gate_result = {"spam_prob": spam_prob, "label": label, ...}
       return _execute_exaone_analysis(email_text, gate_result)
   ```

3. **LangGraph 노드 구조 재검토**
   - 현재: 1개 노드만 있음 (exaone_reader → END)
   - 문제: LangGraph 사용 이유가 약함
   - 해결책 1: 단순 함수로 변경 (LangGraph 제거)
   - 해결책 2: 다단계 노드 구성 (향후 확장 고려)
     ```python
     # 확장 예시
     graph.add_node("preprocess", preprocess_node)
     graph.add_node("analyze", exaone_reader_node)
     graph.add_node("postprocess", postprocess_node)
     graph.add_edge("preprocess", "analyze")
     graph.add_edge("analyze", "postprocess")
     graph.add_edge("postprocess", END)
     ```

---

### 3. `verdict_agent/state_model.py` (상태 관리)
**목적**: LangGraph 상태 정의

#### 현재 상태
```python
# state_model.py (1-17줄)
from typing import Optional, TypedDict

class VerdictAgentState(TypedDict):
    """판독기 에이전트 상태."""
    email_text: str                    # 입력 이메일 텍스트
    gate_result: dict                  # KoELECTRA 게이트웨이 결과
    exaone_result: Optional[str]       # EXAONE Reader 결과
    should_call_exaone: Optional[bool] # EXAONE 호출 여부
```

#### 평가
- ✅ **완벽함**: TypedDict로 상태 정의 잘 됨
- ✅ 간결하고 명확함
- ✅ 타입 힌트 완벽함
- ✅ 주석으로 각 필드 설명

#### 개선 필요사항
- **없음** - 현재 구조가 목적에 완벽히 부합

---

### 4. `verdict_agent/base_model.py` (베이스 모델 로드)
**목적**: EXAONE 베이스 모델 불러오기

#### 현재 상태
```python
# base_model.py (전체)
# 파일이 비어있음 (File is empty)
```

#### 문제점
- ❌ **치명적**: 아무 코드도 없음
- ❌ 모델 로드 코드가 `gragh.py`에 있음 (31-63줄)
- ❌ 파일 목적과 실제 구현 불일치

#### 개선 필요사항 (최우선)
```python
# base_model.py (개선 후)
"""
EXAONE Reader 베이스 모델 로딩.

이 모듈은 EXAONE 모델의 로딩과 전역 캐싱을 담당합니다.
"""

import sys
from pathlib import Path

# 전역 변수 (모델 캐싱용)
_exaone_llm = None
_exaone_loading = False
_exaone_error = None


def load_exaone_model():
    """EXAONE Reader 모델을 로드합니다.

    전역 캐싱을 사용하여 한 번만 로드합니다.

    Returns:
        LangChain LLM 객체

    Raises:
        RuntimeError: 모델 로드 실패 시
    """
    global _exaone_llm, _exaone_loading, _exaone_error

    # 이미 로드된 경우
    if _exaone_llm is not None:
        return _exaone_llm

    # 이전 로드 실패한 경우
    if _exaone_error is not None:
        raise RuntimeError(f"이전 EXAONE 모델 로드 실패: {_exaone_error}")

    # 현재 로딩 중인 경우
    if _exaone_loading:
        raise RuntimeError("EXAONE 모델이 현재 로딩 중입니다.")

    _exaone_loading = True
    try:
        from app.service.model_service import load_exaone_model_for_service

        print("[INFO] EXAONE Reader 모델 로딩 시작...")
        _exaone_llm = load_exaone_model_for_service()
        print("[OK] EXAONE Reader 모델 로드 완료")

        _exaone_loading = False
        _exaone_error = None
        return _exaone_llm

    except Exception as e:
        _exaone_loading = False
        error_msg = f"EXAONE 모델 로드 실패: {str(e)}"
        _exaone_error = error_msg
        print(f"[ERROR] {error_msg}")

        import traceback
        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
        raise RuntimeError(error_msg) from e


def get_exaone_model():
    """이미 로드된 EXAONE 모델을 반환합니다.

    Returns:
        LangChain LLM 객체 또는 None
    """
    return _exaone_llm


def is_exaone_model_loaded() -> bool:
    """EXAONE 모델이 로드되었는지 확인합니다.

    Returns:
        로드 완료 여부
    """
    return _exaone_llm is not None
```

---

## 🚨 주요 문제점

### 1. 관심사 분리 위반 (Separation of Concerns)
```
현재 구조:
mcp_router.py (428줄)
├─ 라우팅 (✅)
├─ 모델 로드 (❌ 65줄)
├─ 예측 로직 (❌ 78줄)
├─ 비즈니스 로직 (❌ 30줄)
└─ 상태 관리 (❌ 46줄)

gragh.py (245줄)
├─ 판독 기능 (✅)
├─ 모델 로드 (❌ 32줄)
└─ 코드 중복 (❌ 중복 로직)
```

### 2. 모델 로드 중복
- **KoELECTRA**: `mcp_router.py` (46-111줄, 65줄)
- **EXAONE**: `gragh.py` (31-63줄, 32줄)
- **문제**: 같은 패턴의 코드가 2군데 존재
- **해결**: 각각 `base_model.py`로 분리

### 3. 코드 중복
- `exaone_reader_node` (67-119줄): LangGraph 노드용
- `exaone_spam_analyzer` (152-200줄): Tool용
- **문제**: 90% 동일한 로직
- **해결**: 공통 함수 추출 `_execute_exaone_analysis()`

### 4. 테스트 어려움
```python
# 현재 (테스트 어려움)
@router.post("/spam-analyze")
async def spam_analyze(request: SpamAnalyzeRequest):
    model, tokenizer = load_koelectra_gate()  # 모델 로드가 라우터 내부에
    # Mock이 어려움

# 개선 후 (테스트 쉬움)
@router.post("/spam-analyze")
async def spam_analyze(
    request: SpamAnalyzeRequest,
    classifier: GateClassifier = Depends(get_classifier)  # DI
):
    result = classifier.predict(request.email_text)
    # Mock 주입 가능
```

### 5. 단일 책임 원칙(SRP) 위반
- `mcp_router.py`: 5가지 책임
- `gragh.py`: 3가지 책임
- **문제**: 한 파일이 너무 많은 일을 함
- **해결**: 책임별로 파일 분리

### 6. `base_model.py` 미구현
- **목적**: 모델 로드
- **현실**: 비어있음
- **영향**: 구조와 구현 불일치

---

## 💡 권장 구조

### 디렉토리 구조
```
api/app/
├── router/
│   ├── __init__.py
│   ├── mcp_router.py              # 순수 라우팅만 (100줄 이하)
│   └── state_router.py            # 상태 관리 엔드포인트 (선택)
│
├── service/
│   ├── gate_agent/                # KoELECTRA 게이트웨이
│   │   ├── __init__.py
│   │   ├── base_model.py          # KoELECTRA 모델 로드
│   │   ├── classifier.py          # 스팸 확률 예측
│   │   ├── decision.py            # EXAONE 호출 여부 판단
│   │   └── state_manager.py       # 상태 관리 (캐시)
│   │
│   └── verdict_agent/             # EXAONE 판독기
│       ├── __init__.py
│       ├── base_model.py          # EXAONE 모델 로드 (✅ 구현 필요)
│       ├── gragh.py               # 판독 기능 (모델 로드 제거)
│       ├── state_model.py         # 상태 정의 (✅ 완벽)
│       └── analyzer.py            # 공통 분석 로직 (선택)
│
└── model/                         # 모델 파일 저장소
    ├── koelectra-small-v3-discriminator/
    ├── koelectra-small-v3-discriminator-spam-lora/
    └── exaone3.5-2.4b-spam-lora/
```

### 파일 크기 가이드
```
mcp_router.py:        428줄 → 100줄 (76% 감소)
gragh.py:             245줄 → 150줄 (39% 감소)

새로 생성:
gate_agent/base_model.py:      ~80줄
gate_agent/classifier.py:      ~100줄
gate_agent/decision.py:        ~50줄
verdict_agent/base_model.py:   ~80줄
```

---

## 📋 구체적 개선 액션 아이템

### 🔧 Phase 1: `verdict_agent/base_model.py` 작성 (최우선)

**파일**: `api/app/service/verdict_agent/base_model.py`

**작업 내용**:
1. `gragh.py`의 31-63줄 코드를 이동
2. `load_exaone_model()` 함수 작성
3. 전역 캐싱 변수 추가
4. Docstring 및 주석 추가

**예상 코드**:
```python
"""EXAONE Reader 베이스 모델 로딩."""

_exaone_llm = None
_exaone_loading = False
_exaone_error = None

def load_exaone_model():
    """EXAONE Reader 모델을 로드합니다."""
    global _exaone_llm, _exaone_loading, _exaone_error

    if _exaone_llm is not None:
        return _exaone_llm

    # ... (gragh.py의 31-63줄 로직 이동)

    return _exaone_llm
```

**영향**:
- `gragh.py`: 32줄 감소
- `base_model.py`: 80줄 증가
- 파일 목적과 구현 일치

---

### 🔧 Phase 2: `verdict_agent/gragh.py` 리팩토링

**파일**: `api/app/service/verdict_agent/gragh.py`

**작업 내용**:
1. `load_exaone_reader()` 함수 삭제 (31-63줄)
2. `from .base_model import load_exaone_model` 추가
3. 중복 로직 통합:
   - `_execute_exaone_analysis()` 공통 함수 생성
   - `exaone_reader_node`에서 호출
   - `exaone_spam_analyzer`에서 호출

**예상 코드**:
```python
from app.service.verdict_agent.base_model import load_exaone_model

def _execute_exaone_analysis(email_text: str, gate_result: dict) -> str:
    """EXAONE 분석 공통 로직."""
    llm = load_exaone_model()
    prompt = _build_prompt(email_text, gate_result)
    messages = [SystemMessage(...), HumanMessage(prompt)]
    response = llm.invoke(messages)
    return response.content

def exaone_reader_node(state: VerdictAgentState):
    """LangGraph 노드용."""
    result = _execute_exaone_analysis(
        state["email_text"],
        state["gate_result"]
    )
    return {**state, "exaone_result": result}

@tool
def exaone_spam_analyzer(email_text: str, spam_prob: float, ...):
    """Tool용."""
    gate_result = {"spam_prob": spam_prob, ...}
    return _execute_exaone_analysis(email_text, gate_result)
```

**영향**:
- 코드 중복 제거: ~50줄 감소
- 유지보수성 향상
- 테스트 용이성 증가

---

### 🔧 Phase 3: `gate_agent/` 폴더 생성

#### 3-1. `gate_agent/base_model.py` 작성

**작업 내용**:
- `mcp_router.py`의 46-111줄 코드 이동
- `load_koelectra_model()` 함수 작성

**예상 코드**:
```python
"""KoELECTRA 게이트웨이 베이스 모델 로딩."""

import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

_koelectra_model = None
_koelectra_tokenizer = None
_koelectra_loading = False
_koelectra_error = None

def load_koelectra_model():
    """KoELECTRA 게이트웨이 모델을 로드합니다.

    Returns:
        tuple: (model, tokenizer)
    """
    global _koelectra_model, _koelectra_tokenizer, _koelectra_loading, _koelectra_error

    if _koelectra_model is not None and _koelectra_tokenizer is not None:
        return _koelectra_model, _koelectra_tokenizer

    # ... (mcp_router.py의 46-111줄 로직)

    return _koelectra_model, _koelectra_tokenizer
```

#### 3-2. `gate_agent/classifier.py` 작성

**작업 내용**:
- `mcp_router.py`의 114-192줄 코드 이동
- `predict_spam_probability()` 함수 작성

**예상 코드**:
```python
"""KoELECTRA 게이트웨이 스팸 분류기."""

import time
import torch
from typing import Optional
from .base_model import load_koelectra_model

def predict_spam_probability(
    email_text: str,
    max_length: int = 512
) -> dict:
    """텍스트에 대한 스팸 확률을 계산합니다.

    Args:
        email_text: 분류할 텍스트
        max_length: 최대 토큰 길이

    Returns:
        {
            "spam_prob": float,
            "ham_prob": float,
            "label": str,
            "confidence": str,
            "latency_ms": float
        }
    """
    start_time = time.time()

    model, tokenizer = load_koelectra_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ... (mcp_router.py의 114-192줄 로직)

    return gate_result
```

#### 3-3. `gate_agent/decision.py` 작성

**작업 내용**:
- `mcp_router.py`의 194-224줄 코드 이동
- `should_call_exaone()` 함수 작성

**예상 코드**:
```python
"""EXAONE Reader 호출 여부 판단."""

from typing import Optional

def should_call_exaone(
    spam_prob: float,
    thresholds: Optional[dict] = None
) -> bool:
    """EXAONE Reader를 호출해야 하는지 결정합니다.

    Args:
        spam_prob: 스팸 확률 (0.0 ~ 1.0)
        thresholds: 임계치 설정

    Returns:
        True: EXAONE Reader 호출 필요
        False: EXAONE Reader 호출 불필요
    """
    if thresholds is None:
        thresholds = {
            "low": 0.2,
            "high": 0.85,
            "ambiguous_low": 0.35,
            "ambiguous_high": 0.8,
        }

    # ... (mcp_router.py의 194-224줄 로직)

    return True or False
```

#### 3-4. `gate_agent/__init__.py` 작성

**예상 코드**:
```python
"""KoELECTRA 게이트웨이 에이전트 모듈."""

from .base_model import load_koelectra_model
from .classifier import predict_spam_probability
from .decision import should_call_exaone

__all__ = [
    "load_koelectra_model",
    "predict_spam_probability",
    "should_call_exaone",
]
```

---

### 🔧 Phase 4: `mcp_router.py` 슬림화

**파일**: `api/app/router/mcp_router.py`

**작업 내용**:
1. 모델 로드 코드 삭제 (46-111줄)
2. 예측 로직 코드 삭제 (114-192줄)
3. 비즈니스 로직 코드 삭제 (194-224줄)
4. Import 추가
5. 엔드포인트 함수만 남김

**예상 코드**:
```python
"""KoELECTRA 게이트웨이 라우터."""

import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import 서비스 모듈
from app.service.gate_agent import (
    predict_spam_probability,
    should_call_exaone
)
from app.service.verdict_agent import get_exaone_tool

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


# ==================== Pydantic 모델 ====================
class GateRequest(BaseModel):
    email_text: str
    max_length: Optional[int] = 512
    request_id: Optional[str] = None


class GateResponse(BaseModel):
    spam_prob: float
    ham_prob: float
    label: str
    confidence: str
    latency_ms: float
    should_call_exaone: bool
    request_id: Optional[str] = None


class SpamAnalyzeRequest(BaseModel):
    email_text: str


class SpamAnalyzeResponse(BaseModel):
    gate_result: dict
    exaone_result: Optional[str]
    final_decision: str


# ==================== 엔드포인트 ====================
@router.post("/gate", response_model=GateResponse)
async def spam_gate(request: GateRequest):
    """KoELECTRA 게이트웨이 API 엔드포인트."""
    try:
        gate_result = predict_spam_probability(
            request.email_text,
            request.max_length
        )

        should_call = should_call_exaone(gate_result["spam_prob"])

        return GateResponse(
            spam_prob=gate_result["spam_prob"],
            ham_prob=gate_result["ham_prob"],
            label=gate_result["label"],
            confidence=gate_result["confidence"],
            latency_ms=gate_result["latency_ms"],
            should_call_exaone=should_call,
            request_id=request.request_id
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"게이트웨이 처리 중 오류: {str(e)[:300]}"
        )


@router.post("/spam-analyze", response_model=SpamAnalyzeResponse)
async def spam_analyze_with_tool(request: SpamAnalyzeRequest):
    """EXAONE 툴을 사용한 스팸 분석 API."""
    try:
        # 1. 게이트웨이 분석
        gate_result = predict_spam_probability(request.email_text)
        spam_prob = gate_result["spam_prob"]

        # 2. EXAONE 호출 여부 판단
        should_call = should_call_exaone(spam_prob)

        # 3. EXAONE 호출 (필요시)
        exaone_result = None
        if should_call:
            exaone_tool = get_exaone_tool()
            exaone_result = exaone_tool.invoke({
                "email_text": request.email_text,
                "spam_prob": spam_prob,
                "label": gate_result["label"],
                "confidence": gate_result["confidence"],
            })

        # 4. 최종 결정
        final_decision = _build_final_decision(
            gate_result,
            exaone_result
        )

        return SpamAnalyzeResponse(
            gate_result=gate_result,
            exaone_result=exaone_result,
            final_decision=final_decision
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"스팸 분석 중 오류: {str(e)[:300]}"
        )


def _build_final_decision(gate_result: dict, exaone_result: Optional[str]) -> str:
    """최종 결정 메시지 생성."""
    spam_prob = gate_result["spam_prob"]

    if exaone_result:
        return f"""KoELECTRA 게이트웨이 결과:
- 스팸 확률: {spam_prob:.4f}
- 레이블: {gate_result['label']}
- 신뢰도: {gate_result['confidence']}

EXAONE Reader 정밀 검사:
{exaone_result}

최종 판단: {'스팸으로 판단됩니다' if spam_prob > 0.5 else '정상 메일로 판단됩니다'}"""
    else:
        return f"""KoELECTRA 게이트웨이 결과:
- 스팸 확률: {spam_prob:.4f}
- 레이블: {gate_result['label']}
- 신뢰도: {gate_result['confidence']}

최종 판단: {'스팸으로 판단됩니다' if spam_prob > 0.5 else '정상 메일로 판단됩니다'} (EXAONE 호출 없음)"""
```

**결과**:
- 428줄 → ~130줄 (70% 감소)
- 순수 라우팅 + 응답 조합만
- 테스트 가능성 향상

---

### 🔧 Phase 5: 상태 관리 분리 (선택사항)

**파일**: `api/app/service/gate_agent/state_manager.py`

**작업 내용**:
- `mcp_router.py`의 상태 관리 코드 이동 (36줄, 290-336줄)
- 클래스 기반으로 리팩토링

**예상 코드**:
```python
"""게이트웨이 결과 상태 관리."""

from typing import Optional

class GateStateManager:
    """게이트웨이 결과 인메모리 캐시 관리.

    실제 운영 환경에서는 Redis나 DB 사용 권장.
    """

    def __init__(self):
        self._cache: dict[str, dict] = {}

    def save(self, request_id: str, gate_result: dict) -> None:
        """결과 저장."""
        self._cache[request_id] = gate_result

    def get(self, request_id: str) -> Optional[dict]:
        """결과 조회."""
        return self._cache.get(request_id)

    def delete(self, request_id: str) -> bool:
        """결과 삭제."""
        if request_id in self._cache:
            del self._cache[request_id]
            return True
        return False

    def list_ids(self) -> list[str]:
        """저장된 ID 목록."""
        return list(self._cache.keys())

    def clear(self) -> None:
        """전체 삭제."""
        self._cache.clear()


# 전역 인스턴스
_state_manager = GateStateManager()

def get_state_manager() -> GateStateManager:
    """상태 관리자 인스턴스 반환."""
    return _state_manager
```

---

## 📊 개선 효과 예측

### 코드 품질 지표

| 지표 | 개선 전 | 개선 후 | 개선율 |
|------|---------|---------|--------|
| `mcp_router.py` 크기 | 428줄 | ~130줄 | 70% 감소 |
| `gragh.py` 크기 | 245줄 | ~150줄 | 39% 감소 |
| 코드 중복 | 높음 | 낮음 | 80% 감소 |
| 단일 책임 원칙 | 위반 | 준수 | ✅ |
| 테스트 가능성 | 낮음 | 높음 | ✅ |
| 유지보수성 | 어려움 | 쉬움 | ✅ |

### 파일 분포

**개선 전**:
```
mcp_router.py: 428줄 (63%)
gragh.py: 245줄 (36%)
state_model.py: 17줄 (1%)
base_model.py: 0줄 (0%)
───────────────────────
합계: 690줄
```

**개선 후**:
```
mcp_router.py: 130줄 (13%)
gate_agent/base_model.py: 80줄 (8%)
gate_agent/classifier.py: 100줄 (10%)
gate_agent/decision.py: 50줄 (5%)
gate_agent/state_manager.py: 60줄 (6%)
verdict_agent/base_model.py: 80줄 (8%)
verdict_agent/gragh.py: 150줄 (15%)
verdict_agent/state_model.py: 17줄 (2%)
verdict_agent/analyzer.py: 80줄 (8%)
───────────────────────────────
합계: 747줄 (8% 증가, 주석/Docstring 포함)
```

### 테스트 커버리지 향상

**개선 전**:
- 라우터 테스트: 어려움 (모델 로드 Mock 필요)
- 비즈니스 로직 테스트: 불가능 (라우터에 결합)

**개선 후**:
- 각 모듈 독립 테스트 가능
- Mock 주입 쉬움
- 단위 테스트 작성 가능

---

## ✨ 현재 코드의 장점 (유지할 부분)

1. ✅ **명확한 API 엔드포인트**
   - FastAPI 구조 잘 설계됨
   - RESTful 원칙 준수

2. ✅ **Pydantic 모델**
   - 타입 안전성 확보
   - 자동 검증 및 문서화

3. ✅ **LangChain Tool 래핑**
   - EXAONE을 Tool로 사용 가능
   - LangGraph 통합 잘 됨

4. ✅ **상태 모델 정의**
   - `state_model.py` 완벽함
   - TypedDict 활용 우수

5. ✅ **에러 핸들링**
   - try-except 구조 잘 됨
   - 사용자 친화적 에러 메시지

6. ✅ **전역 캐싱**
   - 모델 로드 최적화
   - 성능 향상

---

## 🎯 최종 평가 및 우선순위

### 파일별 평가

| 파일 | 목적 부합도 | 개선 필요도 | 우선순위 | 평가 |
|------|------------|------------|---------|------|
| `mcp_router.py` | 50% | 🔴 높음 | P1 | 비즈니스 로직 분리 필요 |
| `gragh.py` | 80% | 🟡 중간 | P2 | 모델 로드 분리 필요 |
| `state_model.py` | 100% | ✅ 없음 | - | 완벽함 |
| `base_model.py` | 0% | 🔴 매우 높음 | P0 | 비어있음, 즉시 구현 필요 |

**종합 점수**: 58/100

### 우선순위별 작업 순서

#### 🚀 P0 (즉시): `verdict_agent/base_model.py` 구현
- **이유**: 파일이 비어있음, 구조와 구현 불일치
- **소요 시간**: 30분
- **영향도**: 높음

#### 🚀 P1 (1주 내): `gragh.py` 리팩토링
- **이유**: 모델 로드 분리, 코드 중복 제거
- **소요 시간**: 1-2시간
- **영향도**: 중간

#### 🚀 P2 (2주 내): `gate_agent/` 폴더 생성
- **이유**: KoELECTRA 로직 분리
- **소요 시간**: 3-4시간
- **영향도**: 높음

#### 🚀 P3 (3주 내): `mcp_router.py` 슬림화
- **이유**: 순수 라우팅으로 변경
- **소요 시간**: 2-3시간
- **영향도**: 높음

#### 🚀 P4 (선택): 상태 관리 분리
- **이유**: 클래스 기반 상태 관리
- **소요 시간**: 1-2시간
- **영향도**: 낮음

---

## 📝 마이그레이션 체크리스트

### Phase 1: `verdict_agent/base_model.py` 작성
- [ ] `gragh.py`의 31-63줄 코드 복사
- [ ] `load_exaone_model()` 함수 작성
- [ ] 전역 변수 추가
- [ ] Docstring 작성
- [ ] Import 테스트
- [ ] `gragh.py`에서 import 변경
- [ ] 기존 `load_exaone_reader()` 삭제
- [ ] 통합 테스트

### Phase 2: `verdict_agent/gragh.py` 리팩토링
- [ ] `_execute_exaone_analysis()` 함수 작성
- [ ] `exaone_reader_node` 리팩토링
- [ ] `exaone_spam_analyzer` 리팩토링
- [ ] 중복 코드 제거
- [ ] 단위 테스트 작성
- [ ] 통합 테스트

### Phase 3: `gate_agent/` 폴더 생성
- [ ] 폴더 생성
- [ ] `__init__.py` 작성
- [ ] `base_model.py` 작성 및 코드 이동
- [ ] `classifier.py` 작성 및 코드 이동
- [ ] `decision.py` 작성 및 코드 이동
- [ ] `state_manager.py` 작성 (선택)
- [ ] Import 테스트
- [ ] 단위 테스트 작성

### Phase 4: `mcp_router.py` 슬림화
- [ ] Import 추가
- [ ] 모델 로드 코드 삭제
- [ ] 예측 로직 코드 삭제
- [ ] 비즈니스 로직 코드 삭제
- [ ] 엔드포인트 함수 리팩토링
- [ ] `_build_final_decision()` 함수 추가
- [ ] 통합 테스트
- [ ] API 테스트 (Postman/curl)

### Phase 5: 최종 검증
- [ ] 모든 엔드포인트 정상 작동 확인
- [ ] 에러 핸들링 테스트
- [ ] 성능 테스트
- [ ] 코드 리뷰
- [ ] 문서 업데이트

---

## 🔗 참고 자료

### 설계 원칙
- **SOLID 원칙**: 단일 책임, 개방-폐쇄, 리스코프 치환, 인터페이스 분리, 의존성 역전
- **관심사 분리 (SoC)**: Separation of Concerns
- **DRY 원칙**: Don't Repeat Yourself

### FastAPI 베스트 프랙티스
- [FastAPI 공식 문서 - Bigger Applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
- [FastAPI 프로젝트 구조](https://fastapi.tiangolo.com/tutorial/bigger-applications/#an-example-file-structure)

### LangChain 패턴
- [LangChain Tool 문서](https://python.langchain.com/docs/modules/tools/)
- [LangGraph 문서](https://langchain-ai.github.io/langgraph/)

---

## 📞 문의사항

이 문서에 대한 질문이나 제안사항이 있으면 개발팀에 문의하세요.

**문서 버전**: 1.0
**작성일**: 2024-01-15
**최종 수정일**: 2024-01-15
