# Unsloth와 EXAONE 모델 호환성 가이드

## 모델 정보

- **모델 ID**: `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`
- **특징**: 커스텀 `modeling_exaone.py` 사용
- **요구사항**: `trust_remote_code=True` 필요

---

## 호환성 확인 방법

### 테스트 스크립트 실행

```bash
cd api
python scripts/test_unsloth_compatibility.py
```

### 예상 결과

#### ✅ 호환되는 경우
```
✅ Unsloth로 EXAONE 모델 로드 성공!
✅ 호환성 확인 완료: Unsloth와 EXAONE 모델이 호환됩니다!
```

#### ❌ 호환되지 않는 경우
```
❌ Unsloth로 EXAONE 모델 로드 실패
❌ 호환성 확인 실패: Unsloth와 EXAONE 모델이 호환되지 않을 수 있습니다.
```

---

## 알려진 이슈

### EXAONE 4.0과의 호환성 문제

웹 검색 결과, Unsloth는 **EXAONE 4.0**과 호환성 문제가 있습니다:
- GitHub 이슈: [#3015](https://github.com/unslothai/unsloth/issues/3015)
- 오류: `ValueError: 'aimv2' is already used by a Transformers config`

### EXAONE 3.5와의 호환성

**EXAONE 3.5**는 아직 확인되지 않았습니다:
- EXAONE 3.5는 EXAONE 4.0과 다른 구조일 수 있음
- 실제 테스트 필요

---

## 호환성 확인 절차

### 1. Unsloth 설치 확인

```bash
pip install unsloth
# 또는
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 2. 테스트 스크립트 실행

```bash
python api/scripts/test_unsloth_compatibility.py
```

### 3. 결과에 따른 대응

#### ✅ 호환되는 경우
- Unsloth 사용 권장
- Flash Attention 자동 포함
- xFormers 불필요

#### ❌ 호환되지 않는 경우
- xFormers 사용 (대안)
- Flash Attention 수동 설정 (선택)
- 현재 방식 유지

---

## 대안 전략

### 전략 1: xFormers 사용 (권장)

Unsloth가 호환되지 않을 경우:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    attn_implementation="xformers",  # xFormers 사용
    trust_remote_code=True,
    # ... 기타 옵션
)
```

**효과**: 1.2-2배 속도 향상

---

### 전략 2: Flash Attention 수동 설정

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    attn_implementation="flash_attention_2",  # Flash Attention 사용
    trust_remote_code=True,
    # ... 기타 옵션
)
```

**효과**: 1.5-2.5배 속도 향상 (Windows에서는 설치 어려움)

---

### 전략 3: 현재 방식 유지

```python
# PEFT + Transformers Trainer
# 현재 코드 그대로 사용
```

**효과**: 기준 속도 (최적화 없음)

---

## 예상 시나리오

### 시나리오 1: Unsloth 호환 ✅

```
Unsloth 설치
    ↓
EXAONE 모델 로드 성공
    ↓
Flash Attention 자동 포함
    ↓
속도 2-5배 향상
```

### 시나리오 2: Unsloth 비호환 ❌

```
Unsloth 설치
    ↓
EXAONE 모델 로드 실패
    ↓
xFormers 사용
    ↓
속도 1.2-2배 향상
```

---

## 다음 단계

1. **테스트 스크립트 실행**: `python api/scripts/test_unsloth_compatibility.py`
2. **결과 확인**: 호환 여부 확인
3. **대안 선택**: 
   - 호환 시 → Unsloth 사용
   - 비호환 시 → xFormers 사용

---

## 참고 자료

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth 이슈 #3015 (EXAONE 4.0)](https://github.com/unslothai/unsloth/issues/3015)
- [EXAONE 모델 페이지](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct)
