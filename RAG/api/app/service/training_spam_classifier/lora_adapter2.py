#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LoRA 어댑터 학습 스크립트 (mixed_email_dataset.jsonl 사용).

KoElectra-small-v3-discriminator 모델에 LoRA 어댑터를 학습하고 저장합니다.
mixed_email_dataset.jsonl 파일을 학습 데이터로 사용합니다.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
spam_classifier_dir = current_file.parent  # api/app/service/spam_classifier/
service_dir = spam_classifier_dir.parent  # api/app/service/
app_dir = service_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

sys.path.insert(0, str(api_dir))

from app.service.training_spam_classifier.load_model import load_koelectra_model


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

    print("[INFO] LoRA 설정:")
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
) -> TrainingArguments:
    """학습 하이퍼파라미터를 설정합니다.

    Args:
        output_dir: 출력 디렉토리
        num_epochs: 에포크 수
        per_device_train_batch_size: 디바이스당 배치 크기
        gradient_accumulation_steps: 그래디언트 누적 스텝
        learning_rate: 학습률
        warmup_steps: 워밍업 스텝
        logging_steps: 로깅 스텝
        save_steps: 저장 스텝
        eval_steps: 평가 스텝 (None이면 save_steps와 동일)
        save_total_limit: 최대 체크포인트 수

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
        save_total_limit=save_total_limit,
        fp16=torch.cuda.is_available(),
        bf16=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        eval_strategy="steps" if has_eval_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=has_eval_dataset,
        metric_for_best_model="eval_accuracy" if has_eval_dataset else None,
        greater_is_better=True if has_eval_dataset else None,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )

    print("[INFO] 학습 하이퍼파라미터:")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch size (per device): {per_device_train_batch_size}")
    print(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Warmup steps: {warmup_steps}")

    return training_args


def compute_metrics(eval_pred):
    """평가 메트릭 계산."""
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


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
        model: 로드된 모델
        tokenizer: 토크나이저
        train_dataset: 학습 데이터셋 (text, label 컬럼 필요)
        val_dataset: 검증 데이터셋 (선택)
        output_dir: 출력 디렉토리 (None이면 자동 생성)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        num_epochs: 에포크 수
        per_device_train_batch_size: 디바이스당 배치 크기
        gradient_accumulation_steps: 그래디언트 누적 스텝
        learning_rate: 학습률
        warmup_steps: 워밍업 스텝
        max_seq_length: 최대 시퀀스 길이
        logging_steps: 로깅 스텝
        save_steps: 저장 스텝

    Returns:
        저장된 LoRA 어댑터 경로
    """
    print("=" * 60)
    print("LoRA 어댑터 학습 시작")
    print("=" * 60)

    # 출력 디렉토리 설정 (koelectra-small-v3-discriminator와 형제 폴더로 저장)
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = (
            app_dir / "model" / "koelectra-small-v3-discriminator-spam-lora" / timestamp
        )
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
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

    # 3. 학습 하이퍼파라미터 설정
    print("\n[3단계] 학습 하이퍼파라미터 설정")
    print("-" * 60)
    training_args = setup_training_args(
        output_dir=output_dir,
        num_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        has_eval_dataset=val_dataset is not None,
    )

    # 4. 데이터셋 토크나이징
    print("\n[4단계] 데이터셋 토크나이징")
    print("-" * 60)

    def tokenize_function(examples):
        """텍스트를 토크나이징합니다."""
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
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

    # 5. 데이터 콜레이터 설정
    print("\n[5단계] 데이터 콜레이터 설정")
    print("-" * 60)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print("[OK] 데이터 콜레이터 설정 완료")

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


def load_classification_dataset(
    dataset_path: Optional[Path] = None,
    text_column: str = "text",
    label_column: str = "label",
):
    """mixed_email_dataset.jsonl 파일에서 분류 데이터셋을 로드합니다.

    Args:
        dataset_path: 데이터셋 경로 (None이면 기본 경로 사용)
        text_column: 텍스트 컬럼명
        label_column: 레이블 컬럼명

    Returns:
        (train_dataset, val_dataset) 튜플
    """
    if dataset_path is None:
        dataset_path = (
            app_dir / "data" / "spam_agent_processed" / "mixed_email_dataset.jsonl"
        )

    if not dataset_path.exists():
        raise FileNotFoundError(f"데이터셋 경로를 찾을 수 없습니다: {dataset_path}")

    import json

    all_data = []

    # mixed_email_dataset.jsonl 로드
    print(f"[INFO] {dataset_path} 로드 중...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())

                # instruct_text를 텍스트로 사용
                text = item.get("instruct_text", "")

                # category를 레이블로 사용 (0: 정상, 1: 스팸)
                category = item.get("dataset_info", {}).get("category", 0)
                label = (
                    1 if category == 1 else 0
                )  # category 1이면 스팸(1), 아니면 정상(0)

                if text:
                    all_data.append({text_column: text, label_column: label})
            except json.JSONDecodeError as e:
                print(f"[WARNING] {dataset_path} {line_num}번째 줄 파싱 실패: {e}")
                continue

    print(f"[OK] mixed_email_dataset.jsonl 로드 완료: {len(all_data)} 샘플")

    # 80% train, 20% validation으로 분할
    train_data = []
    val_data = []
    for idx, data in enumerate(all_data):
        if idx % 5 != 0:  # 0, 5, 10, ... 인덱스는 validation (20%)
            train_data.append(data)
        else:
            val_data.append(data)

    print(f"  - Train: {len(train_data)} 샘플")
    print(f"  - Validation: {len(val_data)} 샘플")

    if not train_data:
        raise ValueError(
            f"데이터셋을 찾을 수 없습니다. 다음 경로를 확인하세요:\n  - {dataset_path}"
        )

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data) if val_data else None

    print("\n[INFO] 최종 데이터셋:")
    print(f"  - Train: {len(train_dataset)} 샘플")
    if val_dataset:
        print(f"  - Validation: {len(val_dataset)} 샘플")

    return train_dataset, val_dataset


def main():
    """메인 함수."""
    print("=" * 60)
    print("KoElectra LoRA 어댑터 학습 (mixed_email_dataset.jsonl)")
    print("=" * 60)

    # 1. 모델 로드
    print("\n[1단계] 모델 로드")
    print("-" * 60)
    model, tokenizer = load_koelectra_model()
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
        print("  - instruct_text: 텍스트 데이터")
        print("  - dataset_info.category: 레이블 (0: 정상, 1: 스팸)")
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] 오류 발생: {e}")
        import traceback

        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
        sys.exit(1)
