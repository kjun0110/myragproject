# Spam Agent 데이터 전처리 모듈

스팸 메일 판단 에이전트를 위한 SFT 학습 데이터 전처리 파이프라인입니다.

## 주요 기능

1. **데이터 전처리** (`transform_data_preprocessor.py`)
   - SFT 학습용 텍스트 포맷 변환
   - 데이터 품질 검증 및 정제
   - 데이터 품질 분석 리포트

2. **토크나이징** (`transform_tokenizer_utils.py`)
   - 토크나이저 준비 및 시퀀스 길이 관리
   - 시퀀스 길이 통계 분석
   - 최적 max_length 자동 계산

3. **데이터 분할** (`transform_data_splitter.py`)
   - Train/Validation/Test 분할 (계층화 지원)
   - 분할된 데이터셋 저장/로드

4. **Dataset 유틸리티** (`transform_dataset_utils.py`)
   - HuggingFace Dataset 객체 생성 및 관리
   - SFTTrainer에서 바로 사용 가능한 형식으로 변환

## 빠른 시작

### 개별 모듈 사용

```python
from pathlib import Path
from app.service.training_spam_agent import (
    DataPreprocessor,
    DataSplitter,
    TokenizerUtils,
    create_datasets_from_examples,
    save_datasets,
    load_datasets,
    prepare_for_sft_trainer,
)

# 1. 데이터 전처리
preprocessor = DataPreprocessor()
preprocessed_data, quality_report = preprocessor.process(
    jsonl_path=Path("data/spam.sft.jsonl"),
    output_path=Path("data/preprocessed.jsonl"),
)

# 2. Dataset 변환
dataset = create_datasets_from_examples(preprocessed_data, text_column="text")

# 3. 데이터 분할
splitter = DataSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
dataset_dict = splitter.split_dataset(dataset)

# 4. Dataset 저장 (SFTTrainer용)
save_datasets(
    train_dataset=dataset_dict["train"],
    val_dataset=dataset_dict["validation"],
    test_dataset=dataset_dict["test"],
    output_dir=Path("data/processed"),
    save_jsonl=True,
    save_arrow=True,
)

# 5. Dataset 로드 (SFTTrainer에서 사용)
train_dataset, val_dataset = load_datasets(
    Path("data/processed"),
    splits=["train", "validation"],
    format="arrow",
)

# 6. SFTTrainer 준비
train_prepared, val_prepared = prepare_for_sft_trainer(
    train_dataset,
    val_dataset,
    text_column="text",
)
```

## SFTTrainer 통합

### Dataset 객체의 장점

HuggingFace Dataset 객체를 사용하면 SFTTrainer가 다음을 자동으로 처리합니다:

- **shuffle**: 데이터 셔플링
- **batch**: 배치 생성
- **collate**: 데이터 배치화
- **multiprocessing**: 병렬 처리
- **cache**: 캐싱

### 사용 예시

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from app.service.training_spam_agent import load_datasets
from pathlib import Path

# Dataset 로드
dataset_dir = Path("api/app/data/spam_agent_processed")
train_dataset, val_dataset = load_datasets(dataset_dir, format="arrow")

# 모델 및 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)

# SFTTrainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,  # Dataset 객체 직접 전달
    eval_dataset=val_dataset,     # Dataset 객체 직접 전달
    tokenizer=tokenizer,
    args=training_args,
    # ... 기타 설정
)

# 학습 실행
trainer.train()
```

## 파일 구조

전처리 후 생성되는 파일 구조:

```
api/app/data/
├── processed/
│   ├── preprocessed.jsonl              # 전처리된 원본 데이터
│   ├── splits/                          # JSONL 형식 분할 데이터
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   ├── spam_agent_processed/            # SFTTrainer용 Dataset
│   │   ├── train_dataset/               # Arrow 형식
│   │   │   ├── data-00000-of-00001.arrow
│   │   │   ├── dataset_info.json
│   │   │   └── state.json
│   │   ├── validation_dataset/
│   │   ├── test_dataset/
│   │   ├── train.jsonl                  # JSONL 형식 (백업)
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   └── tokenized/                       # 토크나이징된 데이터
│       ├── train/
│       ├── validation/
│       └── test/
```

## 모듈 상세 설명

### DataPreprocessor (`transform_data_preprocessor.py`)

데이터 전처리 및 품질 검증을 수행합니다.

```python
preprocessor = DataPreprocessor(
    max_subject_length=200,  # 제목 최대 길이
    min_confidence=0.85,      # 최소 confidence
)

# 전체 파이프라인 실행
preprocessed_data, quality_report = preprocessor.process(
    jsonl_path=Path("data/spam.sft.jsonl"),
    output_path=Path("data/preprocessed.jsonl"),
)
```

### TokenizerUtils (`transform_tokenizer_utils.py`)

토크나이저 준비 및 시퀀스 길이 분석을 수행합니다.

```python
tokenizer_utils = TokenizerUtils(
    model_path="api/app/model/exaone3.5/exaone-2.4b",
    max_length=512,
    trust_remote_code=True,
)

# 시퀀스 길이 분석
tokenizer_utils.print_length_statistics(dataset, text_column="text")

# 최적 max_length 계산
optimal_length = tokenizer_utils.get_optimal_max_length(
    dataset, percentile=0.95
)

# 데이터셋 토크나이징
tokenized_dataset = tokenizer_utils.prepare_dataset(dataset, text_column="text")
```

### DataSplitter (`transform_data_splitter.py`)

데이터를 Train/Validation/Test로 분할합니다.

```python
splitter = DataSplitter(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_state=42,
    stratify=True,  # 클래스 비율 유지
)

# 분할
dataset_dict = splitter.split_dataset(dataset)

# 저장
splitter.save_splits(dataset_dict, Path("data/splits"), format="arrow")
```

### Dataset Utils (`transform_dataset_utils.py`)

SFTTrainer에서 바로 사용할 수 있는 Dataset을 관리합니다.

```python
# 예제에서 Dataset 생성
dataset = create_datasets_from_examples(examples, text_column="text")

# Dataset 저장
save_datasets(
    train_dataset=dataset_dict["train"],
    val_dataset=dataset_dict["validation"],
    output_dir=Path("data/processed"),
    save_jsonl=True,   # JSONL 형식도 저장
    save_arrow=True,   # Arrow 형식도 저장
)

# Dataset 로드
train_dataset, val_dataset = load_datasets(
    Path("data/processed"),
    format="arrow",  # 또는 "jsonl"
)

# SFTTrainer 준비
train_prepared, val_prepared = prepare_for_sft_trainer(
    train_dataset,
    val_dataset,
    text_column="text",
)
```

## 주의사항

1. **메모리 사용량**: 대용량 데이터셋의 경우 Arrow 형식이 메모리 효율적입니다.
2. **토크나이저 경로**: Exaone 모델 경로가 올바르게 설정되어 있는지 확인하세요.
3. **데이터 형식**: SFT 형식은 `{"text": "### 지시문\n...\n### 출력\n..."}` 형식이어야 합니다.

## 의존성

- `datasets>=2.14.0`
- `transformers>=4.30.0`
- `scikit-learn>=1.3.0`
- `numpy>=1.24.0`

## 참고

- [HuggingFace Datasets 문서](https://huggingface.co/docs/datasets/)
- [TRL SFTTrainer 문서](https://huggingface.co/docs/trl/sft_trainer)
