# 스팸 필터 2-Tier 아키텍처 구현 전략

## 📋 현재 상황 분석 및 향후 전략

### ✅ 현재 상태 체크리스트

**완료된 작업:**
1. **KoELECTRA 모델 학습 환경 구축**
   - 데이터셋: `koelectra` 폴더 (train/validation/test 분할 완료)
   - 학습 스크립트: `training_spam_classifier/lora_adapter.py`
   - 테스트 스크립트: `training_spam_classifier/test_model.py`
   - 출력 경로: `koelectra-small-v3-discriminator-spam-lora/{timestamp}`

2. **EXAONE 모델 학습 환경 구축**
   - 데이터셋: `exaone` 폴더 (train/validation/test 분할 완료)
   - 학습 스크립트: `training_spam_agent/lora_adapter.py`
   - 출력 경로: `exaone3.5-2.4b-spam-lora/{timestamp}`
   - 속도 최적화 적용 완료 (배치 크기, DataLoader, 시퀀스 길이)

3. **기존 인프라**
   - FastAPI 서버: `api_server.py`
   - LangGraph: `graph.py` (현재 EXAONE만 사용)
   - Router: `graph_router.py` (현재 단순 대화형)

**❌ 아직 미완료:**
1. KoELECTRA 모델 학습 완료
2. EXAONE 모델 학습 완료
3. 2-tier 게이트 아키텍처 구현
4. MCP 도구 인터페이스 구현
5. Policy 엔진 구현
6. Notify 시스템 구현

---

## 🎯 향후 작업 전략 (우선순위 순)

### Phase 1: 모델 학습 완료 (최우선)

#### Task 1.1: KoELECTRA 학습 실행
- **현재 상태**: 데이터셋 준비 완료 (19,657 train, 2,458 val)
- **실행 명령**: `python api/app/service/training_spam_classifier/lora_adapter.py`
- **예상 시간**: 약 2-3시간 (GPU 사용 시)
- **결과 확인**: 정확도, F1 스코어 확인
- **검증**: `test_model.py`로 test.jsonl 평가

#### Task 1.2: EXAONE 학습 실행
- **현재 상태**: 데이터셋 준비 완료 (19,644 train, 2,455 val)
- **실행 명령**: `python api/app/service/training_spam_agent/lora_adapter.py`
- **예상 시간**: 약 3.8-4.0시간 (최적화 적용 후)
- **결과 확인**: Loss, 생성 품질 확인
- **검증**: 수동으로 샘플 생성 테스트

#### Task 1.3: 모델 성능 평가
- **KoELECTRA**: 정확도, Precision, Recall, F1 스코어 측정
- **EXAONE**: 설명 품질, JSON 형식 준수율 확인
- **임계치 결정**: 어느 확률 구간을 "애매"로 볼지 결정

---

### Phase 2: 1차 필터 (KoELECTRA Gate) 서비스화

#### Task 2.1: KoELECTRA 추론 서비스 구현
```
api/app/service/spam_gate/
  ├── __init__.py
  ├── gate_service.py      # KoELECTRA 추론 서비스
  └── schemas.py           # EmailArtifact, GateDecision 스키마
```

**구현 내용:**
- `gate_service.py`:
  - KoELECTRA 모델 로드 (`koelectra-small-v3-discriminator-spam-lora`)
  - `predict(email_text) → GateDecision` 함수
  - spam_prob, label, confidence 반환
- `schemas.py`:
  - `EmailArtifact` (Pydantic 모델)
  - `GateDecision` (Pydantic 모델)

#### Task 2.2: FastAPI 엔드포인트 추가
```python
# api/app/router/spam_gate_router.py
@router.post("/spam/gate")
async def spam_gate(email: EmailArtifact) -> GateDecision:
    return gate_service.predict(email)
```

#### Task 2.3: 임계치 설정 및 테스트
- LOW, HIGH, AMBIGUOUS 구간 결정
- Test 데이터셋으로 각 구간별 샘플 수 확인
- 운영 정책 결정 (보수적 vs 비용 우선)

---

### Phase 3: 2차 필터 (EXAONE Reader) 서비스화

#### Task 3.1: EXAONE Reader 서비스 구현
```
api/app/service/spam_reader/
  ├── __init__.py
  ├── reader_service.py    # EXAONE 추론 서비스
  ├── schemas.py           # ReaderEvidence 스키마
  └── prompts.py           # EXAONE 프롬프트 템플릿
```

**구현 내용:**
- `reader_service.py`:
  - EXAONE 모델 로드 (`exaone3.5-2.4b-spam-lora`)
  - `extract_evidence(email, gate) → ReaderEvidence` 함수
  - 구조화된 JSON 출력 (features, evidence, user_summary)
- `prompts.py`:
  - EXAONE에게 주는 프롬프트 템플릿
  - 출력 형식 강제 (JSON schema)

#### Task 3.2: FastAPI 엔드포인트 추가
```python
# api/app/router/spam_reader_router.py
@router.post("/spam/reader")
async def spam_reader(email: EmailArtifact, gate: GateDecision) -> ReaderEvidence:
    return reader_service.extract_evidence(email, gate)
```

---

### Phase 4: Policy 엔진 구현

#### Task 4.1: Policy 엔진 구현
```
api/app/service/spam_policy/
  ├── __init__.py
  ├── policy_engine.py     # 정책 엔진
  ├── rules.py             # 룰 기반 정책
  ├── schemas.py           # PolicyDecision 스키마
  └── config.py            # 임계치, 화이트/블랙리스트
```

**구현 내용:**
- `policy_engine.py`:
  - `decide(email, gate, reader) → PolicyDecision` 함수
  - 임계치 기반 결정
  - 화이트/블랙리스트 체크
  - 리스크 스코어 계산
- `rules.py`:
  - 도메인 화이트리스트 (trusted_domain)
  - 블랙리스트 (known_spam_domain)
  - 예외 규칙 (VIP 발신자 등)
- `config.py`:
  - LOW, HIGH, AMBIGUOUS 임계치
  - 정책 설정 (보수적/비용 우선 모드)

#### Task 4.2: FastAPI 엔드포인트 추가
```python
# api/app/router/spam_policy_router.py
@router.post("/spam/policy")
async def spam_policy(
    email: EmailArtifact,
    gate: GateDecision,
    reader: Optional[ReaderEvidence] = None
) -> PolicyDecision:
    return policy_engine.decide(email, gate, reader)
```

---

### Phase 5: LangGraph 통합

#### Task 5.1: LangGraph 상태 및 노드 구현
```
api/app/service/spam_graph/
  ├── __init__.py
  ├── graph.py             # LangGraph 정의
  ├── nodes.py             # 각 노드 구현
  ├── state.py             # State 스키마
  └── routes.py            # 분기 로직
```

**구현 내용:**
- `state.py`: State 정의 (email, gate, reader, policy, notify)
- `nodes.py`: 각 노드 구현
  - `decider_gate_node`
  - `reader_exaone_node`
  - `policy_node`
  - `notify_node`
- `routes.py`: `route_by_gate` 분기 로직
- `graph.py`: StateGraph 조립

#### Task 5.2: FastAPI 엔드포인트 추가
```python
# api/app/router/spam_graph_router.py
@router.post("/spam/analyze")
async def spam_analyze(email: EmailArtifact) -> PolicyDecision:
    graph = create_spam_graph()
    result = await graph.ainvoke({"email": email})
    return result["policy"]
```

---

### Phase 6: Notify 시스템 구현

#### Task 6.1: Notify 서비스 구현
```
api/app/service/spam_notify/
  ├── __init__.py
  ├── notify_service.py    # 통지 서비스
  ├── channels.py          # Slack/Email/Push 채널
  └── schemas.py           # NotifyPayload 스키마
```

**구현 내용:**
- Slack 통지
- 이메일 통지
- 앱 푸시 통지 (선택)

---

### Phase 7: 운영 기능 추가

#### Task 7.1: Audit Log 구현
- 모든 메일 처리 기록 저장
- message_id, gate_prob, used_reader, action, latency
- 데이터베이스 또는 파일로 저장

#### Task 7.2: 화이트/블랙리스트 관리
- 화이트리스트: trusted_domain, VIP 발신자
- 블랙리스트: known_spam_domain
- 관리 UI 또는 API

#### Task 7.3: 피드백 루프 구현
- `ask_user_confirm` 케이스 수집
- 사용자 피드백 수집 (스팸/정상)
- 재학습 데이터셋에 추가

---

### Phase 8: 배포 및 모니터링

#### Task 8.1: Docker Compose 구성
- KoELECTRA 서비스 (CPU/GPU)
- EXAONE 서비스 (GPU 필수)
- Policy 엔진 (CPU)
- FastAPI 게이트웨이

#### Task 8.2: 모니터링 설정
- 처리량 모니터링
- 레이턴시 모니터링
- 게이트 구간별 분포
- EXAONE 호출 비율

---

## 📅 우선순위별 실행 순서

### 즉시 실행 (1-2주)
1. ✅ KoELECTRA 학습 실행 및 검증
2. ✅ EXAONE 학습 실행 및 검증
3. 임계치 결정 (test.jsonl로 ROC 커브 분석)
4. KoELECTRA Gate 서비스 구현 및 엔드포인트 추가

### 단기 (2-4주)
5. EXAONE Reader 서비스 구현 및 엔드포인트 추가
6. Policy 엔진 구현 (임계치 + 화이트/블랙리스트)
7. LangGraph 통합 (노드 + 분기 로직)

### 중기 (1-2개월)
8. Notify 시스템 구현 (Slack 우선)
9. Audit Log 구현
10. 화이트/블랙리스트 관리 API
11. 피드백 루프 구현

### 장기 (2-3개월)
12. Docker Compose 배포
13. 모니터링 및 알림 시스템
14. 성능 튜닝 및 최적화
15. 데이터 드리프트 대응 자동화

---

## 🔧 기술적 고려사항

### 1. 모델 배포 전략
- **KoELECTRA**: CPU로도 충분 (빠름, 저렴)
- **EXAONE**: GPU 필수 (느림, 비쌈) → 호출 최소화

### 2. 확장성
- MCP 도구로 분리 → 독립 서비스 가능
- FastAPI 엔드포인트 → 마이크로서비스 전환 용이

### 3. 비용 최적화
- KoELECTRA로 70-80% 필터링 (LLM 호출 없음)
- EXAONE 호출 비율: 10-20% 목표
- 전체 비용: LLM 호출 대비 80-90% 절감

### 4. 성능 목표
- KoELECTRA: <100ms (CPU)
- EXAONE: <2s (GPU, 애매 구간만)
- 전체 파이프라인: 평균 <500ms

---

## 🚀 다음 단계 (즉시 실행)

### 1. KoELECTRA 학습 완료
```bash
python api/app/service/training_spam_classifier/lora_adapter.py
```

### 2. EXAONE 학습 완료
```bash
python api/app/service/training_spam_agent/lora_adapter.py
```

### 3. 임계치 분석
- test.jsonl로 KoELECTRA 성능 평가
- 확률 구간별 분포 확인
- LOW/HIGH/AMBIGUOUS 구간 결정

### 4. 아키텍처 설계 확정
- MCP vs 직접 호출 결정
- 서비스 분리 수준 결정
- 배포 방식 결정 (Docker Compose vs 모놀리식)

---

## 📝 참고: 아키텍처 설계안

### 전체 아키텍처 요약
- **Decider(게이트)**: KoELECTRA/KoLECTRA 분류기
  - 빠르고 저렴하게 spam_prob 산출
- **Reader(예외처리)**: EXAONE
  - spam_prob가 애매할 때만 호출
  - 근거/특징을 구조화해서 Policy에 공급
- **Policy**:
  - 임계치 + 예외 규칙 + 화이트/블랙리스트 + 리스크 스코어로 최종 결정
- **Notify**:
  - 사용자에게 전달(메일 전달/경고/격리 안내 등)

### 메시지 스키마

#### EmailArtifact
```json
{
  "message_id": "gmail:18c9...",
  "received_at": "2026-01-14T10:12:34+09:00",
  "from": {"name": "PayPal", "email": "notice@paypa1.com"},
  "to": [{"email": "user@domain.com"}],
  "subject": "긴급: 결제 확인 필요",
  "headers": {
    "reply_to": "support@paypa1.com",
    "return_path": "bounce@mailer.bad.com"
  },
  "body": {
    "text": "지금 확인하지 않으면...",
    "html": "<p>...</p>"
  },
  "urls": [
    {"url": "http://paypa1.com/login", "domain": "paypa1.com"}
  ],
  "attachments": [
    {"filename": "invoice.pdf", "mime": "application/pdf", "size": 183002}
  ],
  "auth": {
    "spf": "fail",
    "dkim": "none",
    "dmarc": "fail"
  }
}
```

#### GateDecision
```json
{
  "model": "koelectra-spam-v1",
  "spam_prob": 0.72,
  "label": "spam|ham",
  "confidence": "low|medium|high",
  "latency_ms": 12
}
```

#### ReaderEvidence
```json
{
  "model": "exaone-reader-v1",
  "features": {
    "brand_impersonation": true,
    "sender_domain_mismatch": true,
    "urgent_language": true,
    "money_request": true,
    "url_obfuscation": false,
    "attachment_risky": false,
    "spf_fail": true,
    "dmarc_fail": true,
    "reply_to_mismatch": true
  },
  "evidence": [
    {"code": "URGENT_MONEY", "snippet": "지금 확인하지 않으면 계정이 정지됩니다", "weight": 0.8},
    {"code": "DOMAIN_SPOOF", "snippet": "paypa1.com", "weight": 0.9}
  ],
  "user_summary": "발신 도메인이 정상 PayPal과 다르고, 긴급 결제 확인을 유도하며 의심스러운 로그인 링크가 포함되어 있습니다."
}
```

#### PolicyDecision
```json
{
  "action": "deliver|deliver_with_warning|quarantine|reject|ask_user_confirm",
  "risk_score": 0.0,
  "reason_codes": ["DOMAIN_SPOOF", "DMARC_FAIL", "URGENT_MONEY"],
  "explain_to_user": "발신 도메인 불일치와 인증 실패(DMARC/SPF), 긴급 결제 유도 문구가 확인되어 스팸으로 격리했습니다.",
  "audit": {
    "gate_prob": 0.72,
    "gate_label": "spam",
    "used_reader": true,
    "thresholds": {"low": 0.2, "high": 0.85, "ambiguous_low": 0.35, "ambiguous_high": 0.8}
  }
}
```

### 임계치 정책 (게이트 구간) 기본값
- **LOW <= 0.20**: deliver (LLM 호출 없음)
- **HIGH >= 0.85**: quarantine (LLM 호출 없음, 필요하면 통지 문구만 템플릿)
- **AMBIGUOUS 0.35 ~ 0.80**: EXAONE Reader 호출
- **그 외(0.20~0.35, 0.80~0.85)**: 운영 성향에 따라
  - 보수적이면 Reader 호출
  - 비용 우선이면 "deliver_with_warning" 또는 "quarantine"으로 바로 처리

### LangGraph 상태 설계
```python
State = {
  "email": EmailArtifact,
  "gate": GateDecision | None,
  "reader": ReaderEvidence | None,
  "policy": PolicyDecision | None,
  "notify": {"ok": bool} | None
}
```

### LangGraph 노드 구성
- **DECIDER_GATE**: 코일렉 점수 산출
- **ROUTE_BY_GATE**: 조건 분기(Reader 호출 여부)
- **READER_EXAONE**: 애매 구간만 근거 추출
- **POLICY**: 최종 결정(항상 실행)
- **NOTIFY**: 사용자 통지(필요 시)

---

## 📌 운영에서 꼭 넣어야 하는 3가지

1. **감사 로그 (audit log)**
   - message_id, gate_prob, used_reader, reason_codes, action, latency를 반드시 저장

2. **화이트리스트/블랙리스트 우선순위**
   - Policy에서 trusted_domain이면 게이트 점수가 높아도 "deliver_with_warning"로 낮추는 등 예외 규칙 필요

3. **데이터 드리프트 대응**
   - ask_user_confirm 케이스(사용자 피드백)를 학습 데이터로 재수집하는 루프 설계

---

**작성일**: 2026-01-14
**버전**: 1.0
