# Attention 구현 방식 선택 가이드

## 핵심 개념

**xFormers와 Flash Attention은 둘 중 하나만 선택 가능합니다!**

둘 다 attention 연산을 최적화하는 라이브러리이므로 동시에 사용할 수 없습니다.

---

## 선택 전략

### 전략 1: Unsloth 사용 (권장)

```
Unsloth 사용
    ↓
자동으로 Flash Attention 포함
    ↓
xFormers 불필요 (설정 안 해도 됨)
```

**장점:**
- 가장 빠름 (2-5배 향상)
- Flash Attention 자동 포함
- 별도 설정 불필요

**단점:**
- 코드 수정 필요
- EXAONE 커스텀 모델 호환성 확인 필요

---

### 전략 2: Unsloth 미사용 + Flash Attention

```
Unsloth 사용 안 함
    ↓
수동으로 Flash Attention 설정
    ↓
xFormers 사용 안 함
```

**장점:**
- 빠름 (1.5-2.5배 향상)
- 현재 코드 구조 유지

**단점:**
- Windows에서 설치 어려움
- 수동 설정 필요

---

### 전략 3: Unsloth 미사용 + xFormers

```
Unsloth 사용 안 함
    ↓
수동으로 xFormers 설정
    ↓
Flash Attention 사용 안 함
```

**장점:**
- Windows에서 설치 쉬움
- 안정적
- 현재 코드 구조 유지

**단점:**
- Flash Attention보다 약간 느림 (1.2-2배 향상)

---

## 의사결정 트리

```
시작
  ↓
Unsloth 사용 가능?
  ├─ YES → Unsloth 사용 (Flash Attention 자동 포함)
  │         xFormers 불필요
  │
  └─ NO → Windows 환경?
            ├─ YES → xFormers 사용 (설치 쉬움)
            │
            └─ NO → Flash Attention 사용 (더 빠름)
```

---

## 실제 적용 예시

### 케이스 1: Unsloth 사용 (권장)

```python
from unsloth import FastLanguageModel

# Unsloth 사용 → Flash Attention 자동 포함
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    max_seq_length=256,
    dtype=None,
    load_in_4bit=True,
    trust_remote_code=True,
    # attn_implementation 설정 불필요 (자동으로 Flash Attention 사용)
)
```

**결과:**
- ✅ Flash Attention 자동 활성화
- ❌ xFormers 불필요

---

### 케이스 2: Unsloth 미사용 + Flash Attention

```python
from transformers import AutoModelForCausalLM

# Flash Attention 수동 설정
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",  # Flash Attention 사용
    # xFormers는 사용 안 함
)
```

**결과:**
- ✅ Flash Attention 사용
- ❌ xFormers 사용 안 함

---

### 케이스 3: Unsloth 미사용 + xFormers

```python
from transformers import AutoModelForCausalLM

# xFormers 수동 설정
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="xformers",  # xFormers 사용
    # Flash Attention은 사용 안 함
)
```

**결과:**
- ✅ xFormers 사용
- ❌ Flash Attention 사용 안 함

---

## 호환성 매트릭스

| 조합 | 가능 여부 | 설명 |
|-----|----------|------|
| Unsloth + Flash Attention | ✅ 자동 | Unsloth가 Flash Attention 포함 |
| Unsloth + xFormers | ❌ 불필요 | Unsloth가 Flash Attention 사용하므로 xFormers 불필요 |
| Flash Attention + xFormers | ❌ 불가 | 둘 중 하나만 선택 가능 |
| 기본 (eager) | ✅ 가능 | 둘 다 사용 안 함 |

---

## 권장 사항

### Windows 환경
1. **Unsloth 사용 가능하면**: Unsloth 사용 (Flash Attention 자동)
2. **Unsloth 사용 불가하면**: xFormers 사용

### Linux 환경
1. **Unsloth 사용 가능하면**: Unsloth 사용 (Flash Attention 자동)
2. **Unsloth 사용 불가하면**: Flash Attention 사용

---

## 요약

### 핵심 원칙
1. **Unsloth 사용 시**: Flash Attention 자동 포함, xFormers 불필요
2. **Unsloth 미사용 시**: xFormers 또는 Flash Attention 중 하나만 선택
3. **xFormers와 Flash Attention은 동시 사용 불가**

### 선택 기준
- **가장 빠름**: Unsloth (Flash Attention 자동 포함)
- **Windows 친화적**: xFormers
- **Linux 최적**: Flash Attention

---

## 실제 적용 순서

### Step 1: Unsloth 시도
```python
try:
    from unsloth import FastLanguageModel
    # Unsloth 사용 (Flash Attention 자동 포함)
    model, tokenizer = FastLanguageModel.from_pretrained(...)
    print("✅ Unsloth 사용 (Flash Attention 자동 포함)")
except:
    # Unsloth 사용 불가 → 다음 단계
    pass
```

### Step 2: Flash Attention 시도 (Unsloth 실패 시)
```python
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
    )
    print("✅ Flash Attention 사용")
except:
    # Flash Attention 사용 불가 → xFormers 사용
    pass
```

### Step 3: xFormers 사용 (Flash Attention 실패 시)
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="xformers",  # xFormers 사용
)
print("✅ xFormers 사용")
```

---

## 성능 비교

| 방식 | 속도 향상 | 설치 난이도 | Windows 지원 |
|-----|----------|------------|-------------|
| **Unsloth** | 2-5배 | 중간 | ✅ |
| **Flash Attention** | 1.5-2.5배 | 어려움 | ❌ |
| **xFormers** | 1.2-2배 | 쉬움 | ✅ |
| **기본 (eager)** | 기준 | 기본 | ✅ |

---

## 결론

**사용자의 이해가 정확합니다!**

1. ✅ xFormers와 Flash Attention은 둘 중 하나만 선택
2. ✅ Unsloth 사용 시 Flash Attention 자동 포함
3. ✅ Unsloth 미사용 시 xFormers 또는 Flash Attention 선택

**권장 순서:**
1. Unsloth 시도 (호환되면 Flash Attention 자동 사용)
2. Unsloth 실패 시 → xFormers 사용 (Windows 친화적)
