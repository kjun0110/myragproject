# 모델 관리 가이드

## 개요

이 프로젝트는 **HuggingFace 캐시 + 로컬 아답터** 하이브리드 전략을 사용합니다.

- **베이스 모델**: HuggingFace Hub에서 자동 다운로드 → 전역 캐시에 저장 → 모든 프로젝트에서 공유
- **아답터(LoRA)**: 프로젝트 내부 `api/artifacts/` 폴더에 저장 → 프로젝트별 독립 관리

---

## 모델 구조

```
프로젝트 루트/
├── api/
│   ├── artifacts/                    # 로컬 아티팩트 (Git 제외)
│   │   ├── exaone/
│   │   │   ├── exaone3.5-2.4b/      # 커스텀 코드만 유지 (modeling_exaone.py 등)
│   │   │   └── spam_adapter/        # LoRA 아답터 (로컬 관리)
│   │   │       └── exaone3.5-2.4b-spam-lora/
│   │   │           └── 20260114-161237/
│   │   │               └── adapter_model.safetensors
│   │   └── koelectra/
│   │       └── spam_adapter/        # LoRA 아답터 (로컬 관리)
│   │           └── koelectra-small-v3-discriminator-spam-lora/
│   │               ├── 20260114-144831/
│   │               └── 20260115-162855/
│   └── app/
│       └── common/
│           └── loaders/
│               └── model_loader.py  # 공통 모델 로더
│
└── ~/.cache/huggingface/            # HuggingFace 전역 캐시 (자동 관리)
    └── hub/
        ├── models--LGAI-EXAONE--EXAONE-3.5-2.4B-Instruct/
        └── models--monologg--koelectra-small-v3-discriminator/
```

---

## 사용 모델

| 모델 | HuggingFace ID | 용도 | 아답터 |
|------|---------------|------|--------|
| **EXAONE 3.5 2.4B** | `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct` | 대화형 생성 | `exaone3.5-2.4b-spam-lora` |
| **KoELECTRA Small v3** | `monologg/koelectra-small-v3-discriminator` | 텍스트 분류 | `koelectra-small-v3-discriminator-spam-lora` |

---

## 공통 모델 로더 사용법

### 1. KoELECTRA 로드 (베이스 모델만)

```python
from app.common.loaders import ModelLoader

model, tokenizer = ModelLoader.load_koelectra_model(
    adapter_name=None,  # 아답터 없이
    device="cuda",
    num_labels=2,
)
```

### 2. KoELECTRA 로드 (스팸 아답터 포함)

```python
from app.common.loaders import load_koelectra_with_spam_adapter

model, tokenizer = load_koelectra_with_spam_adapter()
```

### 3. EXAONE 로드 (베이스 모델만)

```python
from app.common.loaders import ModelLoader

model, tokenizer = ModelLoader.load_exaone_model(
    adapter_name=None,  # 아답터 없이
    use_quantization=True,
    device_map="auto",
)
```

### 4. EXAONE 로드 (스팸 아답터 포함)

```python
from app.common.loaders import load_exaone_with_spam_adapter

model, tokenizer = load_exaone_with_spam_adapter(
    use_quantization=True,
    device_map="auto",
)
```

---

## 환경 변수 설정

### `.env` 파일 예시

```bash
# HuggingFace 캐시 경로 (선택사항, 기본값: ~/.cache/huggingface)
HF_HOME=C:/models/huggingface_cache
TRANSFORMERS_CACHE=C:/models/huggingface_cache

# 로컬 아답터 경로 (선택사항, 기본값: api/artifacts/{model}/spam_adapter)
EXAONE_ADAPTER_DIR=C:/path/to/exaone/adapters
KOELECTRA_ADAPTER_DIR=C:/path/to/koelectra/adapters

# HuggingFace 토큰 (private 모델 접근 시 필요)
HF_TOKEN=your_huggingface_token_here
```

---

## 학습 워크플로우

### 1. 베이스 모델 로드 (HF 캐시 활용)

```python
from app.common.loaders import ModelLoader

# 첫 실행 시: HuggingFace Hub에서 다운로드 → 캐시 저장
# 이후 실행: 캐시에서 바로 로드 (네트워크 불필요)
model, tokenizer = ModelLoader.load_koelectra_model(adapter_name=None)
```

### 2. LoRA 파인튜닝

```python
from peft import LoraConfig, get_peft_model

# LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
)

# LoRA 적용
peft_model = get_peft_model(model, lora_config)

# 학습...
trainer.train()
```

### 3. 아답터 저장 (로컬 폴더)

```python
import os
from datetime import datetime

# 타임스탬프 기반 디렉터리 생성
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = f"api/artifacts/koelectra/spam_adapter/koelectra-small-v3-discriminator-spam-lora/{timestamp}"

# 아답터만 저장 (베이스 모델은 저장 안 함)
peft_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
```

### 4. 아답터 로드 (추론 시)

```python
from app.common.loaders import load_koelectra_with_spam_adapter

# 최신 아답터 자동 선택 및 로드
model, tokenizer = load_koelectra_with_spam_adapter()
```

---

## 여러 프로젝트에서 공유하기

### 방법 1: 공용 HF 캐시 사용 (권장)

모든 프로젝트에서 동일한 `HF_HOME` 환경 변수 설정:

```bash
# Windows
set HF_HOME=C:/models/huggingface_cache

# Linux/Mac
export HF_HOME=/shared/models/huggingface_cache
```

**장점:**
- 베이스 모델을 한 번만 다운로드
- 디스크 공간 절약
- 네트워크 대역폭 절약

### 방법 2: 아답터 공유 폴더

여러 프로젝트에서 동일한 아답터를 사용하려면:

```bash
# 프로젝트 A
KOELECTRA_ADAPTER_DIR=C:/shared/adapters/koelectra

# 프로젝트 B (동일한 경로)
KOELECTRA_ADAPTER_DIR=C:/shared/adapters/koelectra
```

---

## 배포 전략

### 개발 환경

- **베이스 모델**: HuggingFace 캐시 사용 (자동 다운로드)
- **아답터**: 로컬 `api/artifacts/` 폴더

### 프로덕션 환경

#### 옵션 1: Docker 이미지에 포함

```dockerfile
# Dockerfile
FROM python:3.10

# 베이스 모델 미리 다운로드 (빌드 시)
RUN python -c "from transformers import AutoModel; \
    AutoModel.from_pretrained('monologg/koelectra-small-v3-discriminator')"

# 아답터 복사
COPY api/artifacts/ /app/api/artifacts/

# 앱 실행
CMD ["python", "main.py"]
```

#### 옵션 2: 외부 스토리지 (S3, GCS 등)

```python
# 배포 시 아답터 다운로드
import boto3

s3 = boto3.client('s3')
s3.download_file(
    'my-bucket',
    'models/koelectra-spam-adapter.tar.gz',
    'api/artifacts/koelectra/spam_adapter.tar.gz'
)
```

---

## 디스크 관리

### 캐시 정리

HuggingFace 캐시는 자동으로 관리되지만, 수동 정리도 가능:

```bash
# 전체 캐시 삭제
rm -rf ~/.cache/huggingface

# 특정 모델만 삭제
rm -rf ~/.cache/huggingface/hub/models--monologg--koelectra-small-v3-discriminator
```

### 아답터 버전 관리

오래된 체크포인트 정리:

```bash
# 최신 아답터만 남기고 삭제
cd api/artifacts/koelectra/spam_adapter/koelectra-small-v3-discriminator-spam-lora/
ls -t | tail -n +2 | xargs rm -rf
```

---

## 트러블슈팅

### 1. 모델 다운로드 실패

```
ConnectionError: Couldn't reach 'https://huggingface.co'
```

**해결:**
- 네트워크 연결 확인
- 프록시 설정 확인
- HuggingFace 토큰 설정 (private 모델인 경우)

### 2. 캐시 권한 오류

```
PermissionError: [Errno 13] Permission denied: '~/.cache/huggingface'
```

**해결:**
```bash
# 캐시 디렉터리 권한 수정
chmod -R 755 ~/.cache/huggingface
```

### 3. 아답터를 찾을 수 없음

```
[WARNING] 아답터를 찾을 수 없음: exaone3.5-2.4b-spam-lora
```

**해결:**
- 아답터 경로 확인: `api/artifacts/exaone/spam_adapter/exaone3.5-2.4b-spam-lora/`
- 환경 변수 `EXAONE_ADAPTER_DIR` 확인

### 4. EXAONE 커스텀 코드 오류

```
ImportError: cannot import name 'ExaoneForCausalLM'
```

**해결:**
- `api/artifacts/exaone/exaone3.5-2.4b/` 폴더에 `modeling_exaone.py`, `configuration_exaone.py` 파일 확인
- `trust_remote_code=True` 옵션 확인

---

## 베스트 프랙티스

1. **개발 시**: HuggingFace 캐시 사용 (편의성)
2. **배포 시**: Docker 이미지에 모델 포함 또는 외부 스토리지 사용 (안정성)
3. **아답터**: 항상 타임스탬프 기반 디렉터리로 버전 관리
4. **Git**: 아답터는 `.gitignore`에 추가 (용량 문제)
5. **백업**: 학습된 아답터는 별도 스토리지에 백업
6. **문서화**: 각 아답터의 학습 날짜, 성능 지표를 `README.md`에 기록

---

## 참고 자료

- [HuggingFace Transformers 문서](https://huggingface.co/docs/transformers)
- [PEFT (LoRA) 문서](https://huggingface.co/docs/peft)
- [EXAONE 3.5 모델 카드](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct)
- [KoELECTRA 모델 카드](https://huggingface.co/monologg/koelectra-small-v3-discriminator)
