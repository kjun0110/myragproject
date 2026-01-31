# 모델 로딩 속도 최적화 가이드

## 개요

체크포인트 로딩 속도(`Loading checkpoint shards`)를 최적화하기 위한 방법들을 정리합니다.

**현재 속도**: 약 31.96초/체크포인트  
**목표**: 최대한 빠른 로딩 속도 달성

---

## 적용된 최적화 방법

### 1. **safetensors 형식 사용** ✅

**효과**: PyTorch pickle 형식보다 **2-3배 빠른** 로딩 속도

```python
# 자동으로 safetensors 형식 사용
use_safetensors = True  # 기본값
```

**환경 변수 제어**:
```bash
USE_SAFETENSORS=true  # 기본값: true
```

**참고**: 
- safetensors는 보안상 더 안전하고 빠름
- 모델이 safetensors 형식으로 저장되어 있어야 함 (대부분의 최신 모델 지원)

---

### 2. **낮은 CPU 메모리 사용 모드** ✅

**효과**: 메모리 효율적 로딩으로 **I/O 병목 감소**

```python
low_cpu_mem_usage = True  # 기본값
```

**환경 변수 제어**:
```bash
LOW_CPU_MEM_USAGE=true  # 기본값: true
```

**작동 원리**:
- 모델을 한 번에 메모리에 올리지 않고 청크 단위로 로드
- CPU 메모리 압박 감소 → 디스크 I/O 효율 향상

---

### 3. **명시적 torch_dtype 설정** ✅

**효과**: 데이터 타입 변환 오버헤드 감소

```python
torch_dtype = torch.bfloat16  # GPU에서 사용
```

**참고**: 양자화 사용 시에는 `quantization_config`가 자동으로 처리

---

### 4. **GPU 메모리 제한 설정** ✅

**효과**: 메모리 관리 최적화로 로딩 안정성 향상

```bash
MAX_MEMORY_GB=8  # GPU 메모리 제한 (GB)
```

**사용 예시**:
- GPU 메모리가 부족한 경우 제한 설정
- 여러 모델을 동시에 로드할 때 유용

---

## 추가 최적화 방법 (수동 적용 가능)

### 5. **디스크 I/O 최적화**

#### SSD 사용
- **효과**: HDD 대비 **5-10배 빠른** 읽기 속도
- 체크포인트 파일이 SSD에 있는지 확인

#### 로컬 캐시 사용
- 네트워크 드라이브 대신 로컬 캐시 사용
- HuggingFace 캐시 경로 확인:
  ```bash
  # Windows
  %USERPROFILE%\.cache\huggingface
  
  # Linux/Mac
  ~/.cache/huggingface
  ```

#### 캐시 경로 변경 (SSD로 이동)
```bash
# .env 파일에 추가
HF_HOME=D:/ssd_cache/huggingface
TRANSFORMERS_CACHE=D:/ssd_cache/huggingface
```

---

### 6. **모델 사전 로딩 (Preloading)**

**효과**: 애플리케이션 시작 시 모델을 미리 로드하여 첫 요청 지연 제거

```python
# main.py 또는 mainbackup.py에서
from app.core.loaders import ModelLoader

# 애플리케이션 시작 시 모델 사전 로드
@app.on_event("startup")
async def preload_models():
    print("[INFO] 모델 사전 로딩 시작...")
    ModelLoader.load_exaone_model(adapter_name="exaone3.5-2.4b-spam-lora")
    ModelLoader.load_koelectra_model(adapter_name="koelectra-small-v3-discriminator-spam-lora")
    print("[OK] 모델 사전 로딩 완료")
```

**장점**:
- 첫 요청 시 지연 없음
- 모델을 메모리에 유지하여 재로딩 불필요

**단점**:
- 애플리케이션 시작 시간 증가
- 메모리 사용량 증가

---

### 7. **병렬 I/O (고급)**

**효과**: 여러 체크포인트를 동시에 로드

**참고**: HuggingFace Transformers는 내부적으로 최적화되어 있지만, 
커스텀 구현 시 `threading` 또는 `multiprocessing` 사용 가능

---

### 8. **양자화 최적화**

**효과**: 4-bit 양자화 사용 시 모델 크기 감소 → 로딩 시간 단축

```python
# 이미 적용됨
use_quantization = True
```

**참고**: 
- 모델 크기가 약 4배 감소 (예: 2.4GB → 0.6GB)
- 로딩 시간도 비례하여 감소

---

## 성능 비교

### 최적화 전
- **형식**: PyTorch pickle
- **메모리 모드**: 기본
- **예상 속도**: ~31.96초/체크포인트

### 최적화 후 (예상)
- **형식**: safetensors ✅
- **메모리 모드**: low_cpu_mem_usage ✅
- **예상 속도**: **~10-15초/체크포인트** (약 50-70% 개선)

### 추가 최적화 (SSD + 양자화)
- **예상 속도**: **~5-8초/체크포인트** (약 75-85% 개선)

---

## 환경 변수 설정 예시

```bash
# .env 파일

# HuggingFace 캐시 (SSD 경로로 변경 권장)
HF_HOME=D:/ssd_cache/huggingface
TRANSFORMERS_CACHE=D:/ssd_cache/huggingface

# 모델 로딩 최적화
USE_SAFETENSORS=true
LOW_CPU_MEM_USAGE=true
MAX_MEMORY_GB=8  # GPU 메모리 제한 (선택사항)

# 아답터 경로
EXAONE_ADAPTER_DIR=api/artifacts/exaone/spam_adapter
KOELECTRA_ADAPTER_DIR=api/artifacts/koelectra/spam_adapter
```

---

## 모니터링

### 로딩 속도 확인
```python
import time

start_time = time.time()
model, tokenizer = ModelLoader.load_exaone_model(...)
load_time = time.time() - start_time
print(f"로딩 시간: {load_time:.2f}초")
```

### 디스크 I/O 확인
- Windows: 작업 관리자 → 성능 → 디스크
- Linux: `iostat -x 1`

### 메모리 사용량 확인
```python
import torch

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"GPU 메모리: {allocated:.2f}GB 할당, {reserved:.2f}GB 예약")
```

---

## 문제 해결

### safetensors 형식이 없는 경우
- 모델이 safetensors 형식으로 저장되어 있지 않으면 자동으로 PyTorch pickle 사용
- 모델을 다시 다운로드하거나 변환 필요

### 메모리 부족 오류
- `MAX_MEMORY_GB` 환경 변수로 제한 설정
- `low_cpu_mem_usage=True` 유지
- 양자화 사용 (`use_quantization=True`)

### 여전히 느린 경우
1. **디스크 확인**: SSD 사용 여부 확인
2. **캐시 위치 확인**: 네트워크 드라이브가 아닌지 확인
3. **다른 프로세스 확인**: 디스크 I/O를 사용하는 다른 프로세스가 있는지 확인
4. **바이러스 스캔**: 실시간 보호가 파일 접근을 지연시키는지 확인

---

## 참고 자료

- [HuggingFace Transformers 문서 - 모델 로딩 최적화](https://huggingface.co/docs/transformers/main/en/performance#efficient-inference-on-a-single-gpu)
- [safetensors 문서](https://huggingface.co/docs/safetensors/)
- [BitsAndBytes 양자화 문서](https://huggingface.co/docs/transformers/main/en/quantization#bitsandbytes-integration)
