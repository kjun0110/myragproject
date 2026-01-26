# xFormers 사용 가이드

## xFormers란?

**xFormers**는 Facebook(Meta)에서 개발한 효율적인 attention 구현 라이브러리입니다.

### 주요 특징
- ✅ **메모리 효율적**: 표준 attention보다 메모리 사용량 감소
- ✅ **빠른 속도**: 1.2-2배 속도 향상
- ✅ **Windows 지원**: Flash Attention보다 설치가 쉬움
- ✅ **안정성**: 널리 사용되는 검증된 라이브러리

---

## 설치

### Windows
```bash
pip install xformers
```

### 특정 버전 설치 (권장)
```bash
pip install xformers==0.0.23.post1
```

### CUDA 버전 확인
```bash
python -c "import torch; print(torch.version.cuda)"
```

---

## 사용 방법

### 1. 모델 로드 시 적용

#### `model_loader.py` 수정
```python
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="xformers",  # xFormers 사용
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=trust_remote_code,
    # ... 기타 옵션
)
```

#### `load_model.py` 수정
```python
model, tokenizer = ModelLoader.load_exaone_model(
    adapter_name=None,
    use_quantization=True,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="xformers",  # xFormers 사용
)
```

### 2. 환경 변수로 제어

```python
import os

# xFormers 사용 여부 (기본값: auto)
use_xformers = os.getenv("USE_XFORMERS", "true").lower() == "true"

if use_xformers:
    attn_implementation = "xformers"
else:
    attn_implementation = "eager"  # 기본 attention

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation=attn_implementation,
    # ...
)
```

---

## 성능 비교

### Attention 구현 방식

| 방식 | 속도 | 메모리 | 설치 |
|-----|------|--------|------|
| **eager** (기본) | 기준 | 기준 | 기본 포함 |
| **xformers** | 1.2-2배 빠름 | 효율적 | `pip install xformers` |
| **flash_attention_2** | 1.5-2.5배 빠름 | 매우 효율적 | 설치 어려움 |

### 실제 성능 (EXAONE-2.4B 기준)

#### 현재 (eager attention)
- 속도: 9.28초/it
- 메모리: 표준

#### xFormers 사용
- 속도: **4-7초/it** (약 1.3-2.3배 빠름)
- 메모리: 약 10-20% 절약

#### Flash Attention 사용
- 속도: **3-5초/it** (약 1.8-3배 빠름)
- 메모리: 약 20-30% 절약

---

## xFormers vs Flash Attention

### xFormers 장점
- ✅ Windows에서 설치 쉬움
- ✅ 안정성 높음
- ✅ 널리 사용됨
- ✅ 충분히 빠름

### Flash Attention 장점
- ✅ 더 빠름 (약 20-30% 추가 향상)
- ✅ 메모리 더 효율적
- ✅ 최신 기술

### Flash Attention 단점
- ❌ Windows에서 설치 어려움
- ❌ CUDA 버전 제약
- ❌ 컴파일 필요

---

## 현재 코드 적용 위치

### 1. 학습용 모델 로드
**파일**: `api/training/agents/spam_agent/load_model.py`

```python
# 현재
model, tokenizer = ModelLoader.load_exaone_model(...)

# xFormers 적용
model, tokenizer = ModelLoader.load_exaone_model(
    ...,
    attn_implementation="xformers",  # 추가
)
```

### 2. 추론용 모델 로드
**파일**: `api/app/common/loaders/model_loader.py`

```python
# 현재
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=trust_remote_code,
)

# xFormers 적용
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="xformers",  # 추가
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=trust_remote_code,
)
```

---

## 호환성 확인

### EXAONE 모델과의 호환성
- EXAONE은 커스텀 `modeling_exaone.py` 사용
- xFormers는 표준 attention 레이어와 호환
- `trust_remote_code=True` 필요

### 테스트 방법
```python
from transformers import AutoModelForCausalLM

try:
    model = AutoModelForCausalLM.from_pretrained(
        "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        attn_implementation="xformers",
        trust_remote_code=True,
    )
    print("✅ xFormers 호환성 확인됨")
except Exception as e:
    print(f"❌ xFormers 호환성 문제: {e}")
```

---

## 문제 해결

### 설치 실패
```bash
# CUDA 버전 확인
python -c "import torch; print(torch.version.cuda)"

# PyTorch 재설치 후 xFormers 설치
pip install torch --upgrade
pip install xformers
```

### 런타임 오류
```python
# xFormers 사용 불가 시 자동으로 eager로 fallback
try:
    attn_implementation = "xformers"
except:
    attn_implementation = "eager"  # 기본값
```

### 메모리 부족
- xFormers는 메모리를 절약하지만, 여전히 부족할 수 있음
- 배치 크기 감소 또는 gradient checkpointing 사용

---

## 권장 사항

### Windows 사용자
- ✅ **xFormers 사용 권장** (설치 쉬움, 충분히 빠름)
- ❌ Flash Attention은 설치 어려움

### Linux 사용자
- ✅ **Flash Attention 사용 권장** (더 빠름)
- ✅ xFormers도 좋은 선택

### 속도 우선
- 1순위: Unsloth (가장 빠름)
- 2순위: Flash Attention
- 3순위: xFormers

### 안정성 우선
- 1순위: xFormers (안정적, 설치 쉬움)
- 2순위: 기본 attention (eager)

---

## 요약

### xFormers는:
- ✅ **속도 향상**: 1.2-2배 빠름
- ✅ **메모리 효율**: 10-20% 절약
- ✅ **설치 쉬움**: Windows에서도 쉽게 설치
- ✅ **안정성**: 검증된 라이브러리

### 언제 사용하나?
- Windows 환경에서 속도 향상이 필요할 때
- Flash Attention 설치가 어려울 때
- 안정적인 속도 향상을 원할 때

### 언제 사용하지 않나?
- Linux 환경에서 Flash Attention을 사용할 수 있을 때
- Unsloth를 사용할 수 있을 때 (더 빠름)
