# Safetensors 완전 정리

## Safetensors란?

**Safetensors**는 HuggingFace에서 개발한 **안전하고 빠른 텐서 저장 형식**입니다.

### 핵심 포인트
- ✅ **PyTorch 전용**이 아닙니다 (PyTorch, TensorFlow, JAX 모두 지원)
- ✅ **TensorFlow가 아닙니다** (완전히 다른 형식)
- ✅ **안전한** 텐서 저장 형식 (보안 + 성능)

---

## Safetensors vs 기존 형식

### 1. **PyTorch Pickle 형식** (기존 방식)

#### 파일 확장자
- `.bin` (모델 가중치)
- `.pt` 또는 `.pth` (PyTorch 체크포인트)

#### 특징
```python
# PyTorch pickle 형식
torch.save(model.state_dict(), "model.bin")  # pickle 사용
model = torch.load("model.bin")  # pickle 역직렬화
```

**문제점:**
1. ⚠️ **보안 취약점**: pickle은 임의의 Python 코드 실행 가능
2. ⚠️ **느린 로딩**: Python pickle 역직렬화는 느림
3. ⚠️ **플랫폼 의존성**: 다른 Python 버전/OS에서 호환성 문제
4. ⚠️ **파일 크기**: 비효율적인 직렬화

---

### 2. **Safetensors 형식** (새로운 방식)

#### 파일 확장자
- `.safetensors` (모델 가중치)

#### 특징
```python
# Safetensors 형식
from safetensors.torch import save_file, load_file

save_file(model.state_dict(), "model.safetensors")  # 안전한 저장
state_dict = load_file("model.safetensors")  # 빠른 로딩
```

**장점:**
1. ✅ **보안**: Python 코드 실행 불가능 (순수 텐서 데이터만)
2. ✅ **빠른 로딩**: C++ 기반 파서로 2-3배 빠름
3. ✅ **플랫폼 독립**: Python 버전/OS와 무관
4. ✅ **효율적**: 더 작은 파일 크기
5. ✅ **다중 프레임워크 지원**: PyTorch, TensorFlow, JAX 모두 지원

---

## 비교표

| 항목 | PyTorch Pickle (.bin) | Safetensors (.safetensors) |
|-----|----------------------|---------------------------|
| **보안** | ⚠️ 취약 (코드 실행 가능) | ✅ 안전 (텐서만) |
| **로딩 속도** | 느림 | **2-3배 빠름** |
| **파일 크기** | 큼 | 작음 |
| **플랫폼 호환성** | Python 버전 의존 | 독립적 |
| **프레임워크** | PyTorch만 | PyTorch, TensorFlow, JAX |
| **사용 예시** | `torch.load()` | `safetensors.load_file()` |

---

## 실제 사용 예시

### HuggingFace Transformers에서 자동 사용

```python
from transformers import AutoModelForCausalLM

# use_safetensors=True (기본값)
model = AutoModelForCausalLM.from_pretrained(
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    use_safetensors=True,  # ← safetensors 형식 사용
    trust_remote_code=True
)
```

**동작 방식:**
1. HuggingFace Hub에서 모델 다운로드
2. `.safetensors` 파일이 있으면 자동으로 사용
3. 없으면 `.bin` 파일 사용 (하위 호환성)

---

## 현재 코드에서의 사용

### model_loader.py

```python
# 최적화 옵션
use_safetensors = os.getenv("USE_SAFETENSORS", "true").lower() == "true"

load_kwargs = {
    "use_safetensors": use_safetensors,  # ← safetensors 사용
    # ...
}

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    **load_kwargs
)
```

**효과:**
- 체크포인트 로딩 속도 **2-3배 향상**
- 보안 강화 (악성 코드 실행 방지)

---

## Safetensors 파일 구조

### 실제 파일 예시

```
model.safetensors
├── metadata (메타데이터)
│   ├── format: "pt" (PyTorch)
│   └── shape: [768, 768] (텐서 크기)
└── data (실제 텐서 데이터)
    └── 바이너리 형식 (효율적)
```

### vs PyTorch Pickle

```
model.bin (PyTorch Pickle)
└── Python 객체 직렬화
    ├── 클래스 정보
    ├── 메서드 정보
    └── 텐서 데이터
```

---

## 왜 Safetensors가 빠른가?

### 1. **C++ 기반 파서**
- Python pickle은 Python 인터프리터 사용 (느림)
- Safetensors는 C++로 구현 (빠름)

### 2. **직렬화 최적화**
- 텐서 데이터만 저장 (불필요한 메타데이터 제거)
- 메모리 매핑 지원 (큰 파일도 빠르게 로드)

### 3. **병렬 처리**
- 여러 체크포인트를 동시에 로드 가능
- I/O 병목 감소

---

## Safetensors vs TensorFlow

### 완전히 다른 개념

| 항목 | Safetensors | TensorFlow |
|-----|------------|-----------|
| **정의** | 텐서 저장 형식 | 딥러닝 프레임워크 |
| **용도** | 모델 가중치 저장 | 모델 학습/추론 |
| **파일 형식** | `.safetensors` | `.pb`, `.h5`, `.savedmodel` |
| **프레임워크** | 형식 (저장용) | 프레임워크 (실행용) |

**비유:**
- **Safetensors**: ZIP 파일 형식 (압축 형식)
- **TensorFlow**: 압축 프로그램 (실행 프로그램)

**관계:**
- Safetensors는 TensorFlow 모델도 저장할 수 있음
- TensorFlow는 Safetensors 파일을 읽을 수 있음
- 하지만 서로 다른 개념!

---

## 실제 성능 비교

### 체크포인트 로딩 속도

#### PyTorch Pickle (.bin)
```
Loading checkpoint shards: 100%|██████████| 2/2 [01:03<00:00, 31.96s/it]
```

#### Safetensors (.safetensors)
```
Loading checkpoint shards: 100%|██████████| 2/2 [00:20<00:00, 10.00s/it]
```

**개선: 약 3배 빠름!**

---

## Safetensors 지원 여부 확인

### 모델이 Safetensors를 지원하는지 확인

```python
from huggingface_hub import hf_hub_download
from safetensors import safe_open

# 모델 파일 목록 확인
model_id = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

# .safetensors 파일이 있는지 확인
try:
    safetensors_file = hf_hub_download(
        repo_id=model_id,
        filename="model.safetensors.index.json"
    )
    print("✅ Safetensors 형식 지원")
except:
    print("⚠️ Safetensors 형식 없음, .bin 파일 사용")
```

---

## 환경 변수 설정

### .env 파일

```bash
# Safetensors 사용 (기본값: true)
USE_SAFETENSORS=true  # true: safetensors 사용, false: .bin 파일 사용
```

### 코드에서 확인

```python
import os

use_safetensors = os.getenv("USE_SAFETENSORS", "true").lower() == "true"

if use_safetensors:
    print("✅ Safetensors 형식 사용 (빠른 로딩)")
else:
    print("⚠️ PyTorch pickle 형식 사용 (느린 로딩)")
```

---

## 요약

### Safetensors는:
1. ✅ **텐서 저장 형식** (파일 형식)
2. ✅ **PyTorch, TensorFlow, JAX 모두 지원**
3. ✅ **TensorFlow가 아님** (완전히 다른 개념)
4. ✅ **안전하고 빠름** (보안 + 성능)
5. ✅ **현재 코드에서 사용 중** (`use_safetensors=True`)

### Safetensors의 장점:
- 🔒 **보안**: 악성 코드 실행 불가능
- ⚡ **속도**: 2-3배 빠른 로딩
- 📦 **효율**: 작은 파일 크기
- 🔄 **호환성**: 플랫폼 독립적

---

## 참고 자료

- [Safetensors 공식 문서](https://huggingface.co/docs/safetensors/)
- [HuggingFace Transformers - Safetensors](https://huggingface.co/docs/transformers/main/en/serialization#safetensors)
- [GitHub - Safetensors](https://github.com/huggingface/safetensors)
