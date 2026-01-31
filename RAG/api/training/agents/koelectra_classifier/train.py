#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KoELECTRA 분류기 학습 스크립트.

KoElectra-small-v3-discriminator 모델에 LoRA 어댑터를 학습하여
시멘틱(정책 기반) 질문인지 아닌지 판단하는 3-class 분류기를 만듭니다.

라벨:
- 0: 도메인 외 질문 (OUT_OF_DOMAIN)
- 1: 정책 기반 질문 (POLICY_BASED) - 시멘틱
- 2: 규칙 기반 질문 (RULE_BASED) - 비시멘틱
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
koelectra_classifier_dir = current_file.parent  # api/training/agents/koelectra_classifier/
agents_dir = koelectra_classifier_dir.parent  # api/training/agents/
training_dir = agents_dir.parent  # api/training/
api_dir = training_dir.parent  # api/

sys.path.insert(0, str(api_dir))

from app.core.loaders import ModelLoader


def load_koelectra_model(num_labels: int = 3):
    """KoELECTRA 모델을 로드합니다.
    
    Args:
        num_labels: 분류 클래스 수 (기본값: 3)
    
    Returns:
        (model, tokenizer) 튜플
    """
    print("=" * 60)
    print("KoElectra-small-v3-discriminator 모델 로드")
    print("=" * 60)
    
    print("\n[INFO] HuggingFace 캐시를 활용하여 베이스 모델 로드")
    print(f"[INFO] 모델 ID: {ModelLoader.KOELECTRA_MODEL_ID}")
    print(f"[INFO] 분류 클래스 수: {num_labels}")
    
    # CUDA 사용 가능 여부 확인
    cuda_available = torch.cuda.is_available()
    torch_version = torch.__version__
    
    print(f"\n[INFO] PyTorch 버전: {torch_version}")
    print(f"[INFO] CUDA 사용 가능 여부: {cuda_available}")
    
    # CPU 버전 PyTorch가 설치된 경우 즉시 에러
    if "+cpu" in torch_version:
        print("\n" + "=" * 60)
        print("[ERROR] PyTorch CPU 버전이 설치되어 있습니다!")
        print("=" * 60)
        print("\n[문제 설명]")
        print("  현재 설치된 PyTorch는 CPU 전용 버전입니다.")
        print("  CUDA는 이미 시스템에 설치되어 있을 수 있지만,")
        print("  PyTorch가 CUDA를 지원하지 않아 GPU를 사용할 수 없습니다.")
        print("\n[해결 방법]")
        print("  CUDA를 재설치할 필요는 없습니다!")
        print("  PyTorch만 CUDA 버전으로 재설치하면 됩니다.\n")
        print("  1. 현재 PyTorch 제거:")
        print("     pip uninstall torch torchvision torchaudio")
        print("\n  2. 시스템 CUDA 버전 확인:")
        print("     nvidia-smi")
        print("     (출력된 CUDA Version 확인)")
        print("\n  3. CUDA 버전에 맞는 PyTorch 설치:")
        print("     # CUDA 11.8인 경우:")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\n     # CUDA 12.1인 경우:")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("\n     # 또는 공식 사이트에서 자동 명령어 생성:")
        print("     https://pytorch.org/get-started/locally/")
        print("\n[참고]")
        print("  - CUDA는 재설치할 필요 없습니다 (이미 설치되어 있음)")
        print("  - PyTorch만 CUDA 지원 버전으로 재설치하면 됩니다")
        print("  - 설치 후 torch.cuda.is_available()이 True가 됩니다")
        raise RuntimeError(
            "PyTorch CPU 버전이 설치되어 있습니다. GPU 학습을 위해 CUDA 버전의 PyTorch를 설치해야 합니다.\n"
            "위의 해결 방법을 참고하여 PyTorch를 재설치하세요."
        )
    
    # CUDA가 사용 불가능한 경우 (PyTorch는 CUDA 버전이지만 CUDA가 감지되지 않음)
    if not cuda_available:
        print("\n" + "=" * 60)
        print("[ERROR] CUDA를 사용할 수 없습니다!")
        print("=" * 60)
        print("\n[가능한 원인]")
        print("  1. GPU 드라이버가 설치되지 않았거나 오래된 버전")
        print("  2. CUDA Toolkit이 설치되지 않음")
        print("  3. PyTorch와 CUDA 버전이 호환되지 않음")
        print("\n[확인 방법]")
        print("  nvidia-smi  # GPU와 드라이버 확인")
        print("  python -c \"import torch; print(torch.cuda.is_available())\"  # PyTorch CUDA 확인")
        print("\n[해결 방법]")
        print("  1. NVIDIA GPU 드라이버 설치/업데이트")
        print("  2. CUDA Toolkit 설치 (PyTorch와 호환되는 버전)")
        print("  3. PyTorch 재설치 (CUDA 버전)")
        raise RuntimeError(
            "CUDA를 사용할 수 없습니다. GPU 학습을 위해 CUDA가 필요합니다.\n"
            "위의 확인 방법과 해결 방법을 참고하세요."
        )
    
    # GPU 사용 가능
    device = "cuda"
    print(f"\n[OK] GPU 사용 가능!")
    print(f"[INFO] GPU 디바이스: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] PyTorch CUDA 버전: {torch.version.cuda}")
    print(f"[INFO] GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 모델 로드 (어댑터 없이 베이스 모델만)
    print("\n[INFO] 모델 로드 중...")
    try:
        model, tokenizer = ModelLoader.load_koelectra_model(
            adapter_name=None,  # 학습용이므로 어댑터 없이
            device=device,
            num_labels=num_labels,
        )
        print("[OK] 모델 로드 완료")
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        import traceback
        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
        raise
    
    return model, tokenizer


def load_classification_dataset(
    dataset_path: Optional[Path] = None,
    text_column: str = "question",
    label_column: str = "label",
):
    """분류 데이터셋을 로드합니다.
    
    Args:
        dataset_path: 데이터셋 파일 경로 (None이면 기본 경로 사용)
        text_column: 텍스트 컬럼명
        label_column: 레이블 컬럼명
    
    Returns:
        (train_dataset, val_dataset) 튜플
    """
    if dataset_path is None:
        dataset_path = (
            training_dir
            / "data"
            / "koelectra_classifier"
            / "koelectra_training_dataset.sft.jsonl"
        )
    
    print(f"\n[INFO] 데이터셋 로드: {dataset_path}")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
    
    # JSONL 파일 읽기
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # 데이터셋 형식에 맞게 변환
                question = item.get("input", {}).get("question", "")
                label = item.get("label", item.get("output", {}).get("label", -1))
                
                if question and label != -1:
                    data.append({text_column: question, label_column: label})
    
    if len(data) == 0:
        raise ValueError("데이터셋이 비어있습니다.")
    
    print(f"[INFO] 로드된 샘플 수: {len(data)}")
    
    # 라벨 분포 확인
    label_counts = {}
    for item in data:
        label = item[label_column]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\n[INFO] 라벨 분포:")
    for label, count in sorted(label_counts.items()):
        label_name = {0: "도메인 외", 1: "정책 기반", 2: "규칙 기반"}.get(label, f"Label {label}")
        print(f"  - Label {label} ({label_name}): {count}개 ({count/len(data)*100:.1f}%)")
    
    # 데이터셋을 Dataset 객체로 변환
    dataset = Dataset.from_list(data)
    
    # Train/Validation 분할 (80:20)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    
    print(f"\n[INFO] 데이터셋 분할:")
    print(f"  - 학습 데이터: {len(train_dataset)}개")
    print(f"  - 검증 데이터: {len(val_dataset)}개")
    
    return train_dataset, val_dataset


def setup_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[list] = None,
) -> LoraConfig:
    """LoRA 설정을 생성합니다.
    
    Args:
        r: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        target_modules: LoRA를 적용할 모듈 목록 (None이면 KoElectra 기본값)
    
    Returns:
        LoRA 설정 객체
    """
    if target_modules is None:
        # KoElectra 모델의 attention 모듈
        target_modules = ["query", "key", "value", "dense"]
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,  # Sequence Classification
    )
    
    print("\n[INFO] LoRA 설정:")
    print(f"  - Rank (r): {r}")
    print(f"  - Alpha: {alpha}")
    print(f"  - Dropout: {dropout}")
    print(f"  - Target modules: {target_modules}")
    
    return lora_config


def setup_training_args(
    output_dir: Path,
    num_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: Optional[int] = None,
    save_total_limit: int = 3,
    has_eval_dataset: bool = False,
    device: str = "cuda",
) -> TrainingArguments:
    """학습 인자를 설정합니다.
    
    Args:
        output_dir: 출력 디렉토리
        num_epochs: 에포크 수
        per_device_train_batch_size: 디바이스당 배치 크기
        gradient_accumulation_steps: 그래디언트 누적 스텝 수
        learning_rate: 학습률
        warmup_steps: 워밍업 스텝 수
        logging_steps: 로깅 스텝 수
        save_steps: 저장 스텝 수
        eval_steps: 평가 스텝 수 (None이면 save_steps와 동일)
        save_total_limit: 저장할 체크포인트 수 제한
        has_eval_dataset: 검증 데이터셋 존재 여부
    
    Returns:
        TrainingArguments 객체
    """
    if eval_steps is None:
        eval_steps = save_steps
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps if has_eval_dataset else None,
        eval_strategy="steps" if has_eval_dataset else "no",  # evaluation_strategy → eval_strategy로 변경
        save_strategy="steps",
        save_total_limit=save_total_limit,
        load_best_model_at_end=True if has_eval_dataset else False,
        metric_for_best_model="f1_weighted" if has_eval_dataset else None,
        greater_is_better=True if has_eval_dataset else None,
        fp16=(device == "cuda" and torch.cuda.is_available()),  # GPU가 있으면 FP16 사용
        bf16=False,
        dataloader_num_workers=0,  # Windows 호환성
        remove_unused_columns=False,
        report_to="none",  # wandb 등 사용 안 함
    )
    
    print("\n[INFO] 학습 인자:")
    print(f"  - 에포크 수: {num_epochs}")
    print(f"  - 배치 크기: {per_device_train_batch_size}")
    print(f"  - 그래디언트 누적: {gradient_accumulation_steps}")
    print(f"  - 학습률: {learning_rate}")
    print(f"  - 워밍업 스텝: {warmup_steps}")
    print(f"  - FP16: {training_args.fp16}")
    
    return training_args


def compute_metrics(eval_pred):
    """평가 메트릭을 계산합니다.
    
    Args:
        eval_pred: 평가 예측 결과
    
    Returns:
        메트릭 딕셔너리
    """
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    # 클래스별 F1 점수
    f1_macro = f1_score(labels, predictions, average="macro")
    
    return {
        "accuracy": accuracy,
        "f1_weighted": f1,
        "f1_macro": f1_macro,
        "f1_class_0": f1_per_class[0] if len(f1_per_class) > 0 else 0.0,
        "f1_class_1": f1_per_class[1] if len(f1_per_class) > 1 else 0.0,
        "f1_class_2": f1_per_class[2] if len(f1_per_class) > 2 else 0.0,
        "precision": precision.mean() if len(precision) > 0 else 0.0,
        "recall": recall.mean() if len(recall) > 0 else 0.0,
    }


def train_lora_adapter(
    model,
    tokenizer,
    train_dataset,
    val_dataset: Optional[Dataset] = None,
    output_dir: Optional[Path] = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    num_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 100,
    max_seq_length: int = 512,
    logging_steps: int = 10,
    save_steps: int = 500,
) -> Path:
    """LoRA 어댑터를 학습합니다.
    
    Args:
        model: 베이스 모델
        tokenizer: 토크나이저
        train_dataset: 학습 데이터셋
        val_dataset: 검증 데이터셋 (선택)
        output_dir: 출력 디렉토리
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        num_epochs: 에포크 수
        per_device_train_batch_size: 디바이스당 배치 크기
        gradient_accumulation_steps: 그래디언트 누적 스텝 수
        learning_rate: 학습률
        warmup_steps: 워밍업 스텝 수
        max_seq_length: 최대 시퀀스 길이
        logging_steps: 로깅 스텝 수
        save_steps: 저장 스텝 수
    
    Returns:
        저장된 어댑터 경로
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = (
            api_dir
            / "artifacts"
            / "koelectra"
            / "koelectra_classifier"
            / f"koelectra-small-v3-discriminator-classifier-lora"
            / timestamp
        )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("LoRA 어댑터 학습")
    print("=" * 60)
    print(f"\n[INFO] 출력 디렉토리: {output_dir}")
    
    # 1. LoRA 설정
    print("\n[1단계] LoRA 설정")
    print("-" * 60)
    lora_config = setup_lora_config(
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )
    
    # 2. LoRA 적용
    print("\n[2단계] LoRA 적용")
    print("-" * 60)
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    print("[OK] LoRA 적용 완료")
    
    # 3. 데이터 토크나이징
    print("\n[3단계] 데이터 토크나이징")
    print("-" * 60)
    
    def tokenize_function(examples):
        """토크나이징 함수."""
        # 텍스트 토크나이징
        tokenized = tokenizer(
            examples["question"],
            truncation=True,
            padding=False,
            max_length=max_seq_length,
        )
        # labels 추가 (분류 작업에 필수)
        tokenized["labels"] = examples["label"]
        return tokenized
    
    print("[INFO] 학습 데이터셋 토크나이징 중...")
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[
            col
            for col in train_dataset.column_names
            if col not in ["input_ids", "attention_mask", "labels"]
        ],
        desc="토크나이징 (train)",
    )
    print(f"[OK] 학습 데이터셋 토크나이징 완료: {len(train_tokenized)} 샘플")
    
    val_tokenized = None
    if val_dataset is not None:
        print("[INFO] 검증 데이터셋 토크나이징 중...")
        val_tokenized = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[
                col
                for col in val_dataset.column_names
                if col not in ["input_ids", "attention_mask", "labels"]
            ],
            desc="토크나이징 (validation)",
        )
        print(f"[OK] 검증 데이터셋 토크나이징 완료: {len(val_tokenized)} 샘플")
    
    # 4. 데이터 콜레이터 설정
    print("\n[4단계] 데이터 콜레이터 설정")
    print("-" * 60)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print("[OK] 데이터 콜레이터 설정 완료")
    
    # 5. 학습 인자 설정
    print("\n[5단계] 학습 인자 설정")
    print("-" * 60)
    # device 정보 가져오기 (모델에서)
    model_device = next(model.parameters()).device.type if hasattr(model, 'parameters') and len(list(model.parameters())) > 0 else "cuda"
    
    training_args = setup_training_args(
        output_dir=output_dir,
        num_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        has_eval_dataset=val_tokenized is not None,
        device=model_device,
    )
    print("[OK] 학습 인자 설정 완료")
    
    # 6. Trainer 생성
    print("\n[6단계] Trainer 생성")
    print("-" * 60)
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized if val_tokenized is not None else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics if val_tokenized is not None else None,
    )
    print("[OK] Trainer 생성 완료")
    
    # 7. 학습 실행
    print("\n[7단계] 학습 실행")
    print("-" * 60)
    print("[INFO] 학습 시작...")
    print(f"  - 학습 샘플 수: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"  - 검증 샘플 수: {len(val_dataset)}")
    
    trainer.train()
    print("[OK] 학습 완료")
    
    # 8. LoRA 어댑터 저장
    print("\n[8단계] LoRA 어댑터 저장")
    print("-" * 60)
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    print(f"[OK] LoRA 어댑터 저장 완료: {output_dir}")
    
    print("\n" + "=" * 60)
    print("[OK] LoRA 어댑터 학습 완료!")
    print("=" * 60)
    print(f"\n저장된 어댑터 경로: {output_dir}")
    
    return output_dir


def main():
    """메인 함수."""
    print("=" * 60)
    print("KoELECTRA 분류기 학습")
    print("=" * 60)
    print("\n목적: 시멘틱(정책 기반) 질문인지 아닌지 판단하는 3-class 분류기 학습")
    print("라벨:")
    print("  - 0: 도메인 외 질문 (OUT_OF_DOMAIN)")
    print("  - 1: 정책 기반 질문 (POLICY_BASED) - 시멘틱")
    print("  - 2: 규칙 기반 질문 (RULE_BASED) - 비시멘틱")
    
    # 1. 모델 로드
    print("\n[1단계] 모델 로드")
    print("-" * 60)
    model, tokenizer = load_koelectra_model(num_labels=3)
    model.train()  # 학습 모드로 전환
    
    # 2. 데이터셋 로드
    print("\n[2단계] 데이터셋 로드")
    print("-" * 60)
    try:
        train_dataset, val_dataset = load_classification_dataset()
        if len(train_dataset) == 0:
            raise ValueError(
                "학습 데이터셋이 비어있습니다. 데이터셋 경로와 형식을 확인하세요."
            )
        print("[OK] 데이터셋 로드 완료")
    except Exception as e:
        print(f"[ERROR] 데이터셋 로드 실패: {e}")
        print("\n데이터셋 형식:")
        print('  {"input": {"question": "...", "intent": "..."}, "output": {...}, "label": 0|1|2}')
        import traceback
        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
        sys.exit(1)
    
    # 3. LoRA 어댑터 학습
    print("\n[3단계] LoRA 어댑터 학습")
    print("-" * 60)
    adapter_path = train_lora_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        num_epochs=5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=100,
        max_seq_length=512,
        logging_steps=10,
        save_steps=500,
    )
    
    print("\n" + "=" * 60)
    print("[SUCCESS] 모든 작업이 완료되었습니다!")
    print("=" * 60)
    print(f"\nLoRA 어댑터 경로: {adapter_path}")
    print("\n사용 방법:")
    print("  from app.core.loaders import ModelLoader")
    print("  ")
    print("  model, tokenizer = ModelLoader.load_koelectra_model(")
    print("      adapter_name='koelectra_classifier',")
    print("      num_labels=3,")
    print("  )")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] 학습 중 오류 발생: {e}")
        import traceback
        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
        sys.exit(1)
