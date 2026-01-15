#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LoRA 어댑터 학습 스크립트.

EXAONE-2.4B 모델에 LoRA 어댑터를 학습하고 저장합니다.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# trust_remote_code를 위한 환경 변수 설정
os.environ["TRUST_REMOTE_CODE"] = "True"

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
spam_agent_dir = current_file.parent  # api/app/service/spam_agent/
service_dir = spam_agent_dir.parent  # api/app/service/
app_dir = service_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

sys.path.insert(0, str(api_dir))

from app.service.training_spam_agent.load_model import load_exaone_model
from app.service.training_spam_agent.transform_dataset_utils import load_datasets


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
        target_modules: LoRA를 적용할 모듈 목록 (None이면 자동 감지)

    Returns:
        LoRA 설정 객체
    """
    if target_modules is None:
        # EXAONE 모델의 attention 모듈 (일반적인 구조)
        target_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
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
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: Optional[int] = None,
    save_total_limit: int = 3,
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
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        fp16=False,  # 4-bit 양자화는 bfloat16 사용
        bf16=True,
        optim="paged_adamw_8bit",  # 8-bit 옵티마이저
        lr_scheduler_type="cosine",
        report_to="none",  # wandb/tensorboard 비활성화
        eval_strategy="steps",  # evaluation strategy를 steps로 설정
        save_strategy="steps",  # save strategy를 명시적으로 설정
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # 메모리 최적화 설정 (RTX 3050 6GB용)
        gradient_checkpointing=True,  # QLoRA 필수 (메모리 절약)
        # 속도 개선: DataLoader 최적화 (TRAINING_ANALYSIS.md 반영)
        dataloader_num_workers=4,  # 병렬 데이터 로딩 (속도 향상: 2 → 4)
        dataloader_pin_memory=True,  # CPU-GPU 전송 최적화 (속도 향상)
        dataloader_prefetch_factor=2,  # 미리 로드할 배치 수 (속도 향상)
        remove_unused_columns=False,  # 이미 제거됨
        ddp_find_unused_parameters=False,  # DDP 최적화
    )

    print("[INFO] 학습 하이퍼파라미터:")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch size (per device): {per_device_train_batch_size}")
    print(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
    print(
        f"  - Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}"
    )
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Warmup steps: {warmup_steps}")
    print(f"  - Logging steps: {logging_steps}")
    print(f"  - Save steps: {save_steps}")
    print("  - Optimizer: paged_adamw_8bit")
    print("  - LR scheduler: cosine")
    print("\n[INFO] 메모리 및 속도 최적화 설정:")
    print("  - Gradient checkpointing: 활성화 (메모리 절약, QLoRA 필수)")
    print("  - DataLoader workers: 4 (병렬 데이터 로딩, 속도 향상)")
    print("  - Pin memory: 활성화 (CPU-GPU 전송 최적화, 속도 향상)")
    print("  - Prefetch factor: 2 (미리 로드할 배치 수, 속도 향상)")

    return training_args


def train_lora_adapter(
    model,
    tokenizer,
    train_dataset,
    val_dataset: Optional = None,
    output_dir: Optional[Path] = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    num_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 100,
    max_seq_length: int = 256,  # 속도 개선: 512 → 384 → 256 (VRAM 절약, 성능 영향 미미)
    logging_steps: int = 10,
    save_steps: int = 500,
) -> Path:
    """LoRA 어댑터를 학습합니다.

    Args:
        model: 로드된 모델 (4-bit 양자화)
        tokenizer: 토크나이저
        train_dataset: 학습 데이터셋
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

    # 출력 디렉토리 설정 (exaone3.5-2.4b와 형제 폴더로 저장)
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = app_dir / "model" / "exaone3.5-2.4b-spam-lora" / timestamp
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] 출력 디렉토리: {output_dir}")

    # 1. PEFT 모델 준비
    print("\n[1단계] PEFT 모델 준비")
    print("-" * 60)
    print("[INFO] 모델을 k-bit 학습용으로 준비 중...")
    model = prepare_model_for_kbit_training(model)

    # Gradient checkpointing 활성화 (메모리 절약, QLoRA 필수)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[INFO] Gradient checkpointing 활성화 (메모리 절약)")

    # 모델 config 수정 (SFTTrainer가 내부적으로 processor 로드 시 trust_remote_code 필요)
    if hasattr(model, "config"):
        # _name_or_path를 HuggingFace 모델 ID로 변경 (로컬 경로 대신)
        # 이렇게 하면 SFTTrainer가 processor를 로드할 때 문제가 발생할 수 있으므로
        # 대신 config에 trust_remote_code 정보 추가
        original_name_or_path = getattr(model.config, "_name_or_path", None)
        if original_name_or_path and Path(original_name_or_path).exists():
            # 로컬 경로인 경우 HuggingFace 모델 ID로 변경 시도
            # 또는 config에 trust_remote_code 플래그 추가
            try:
                # config에 trust_remote_code 속성 추가
                setattr(model.config, "trust_remote_code", True)
            except:
                pass

    print("[OK] 모델 준비 완료")

    # 2. LoRA 설정
    print("\n[2단계] LoRA 설정")
    print("-" * 60)
    lora_config = setup_lora_config(
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )

    # 3. LoRA 적용
    print("\n[3단계] LoRA 적용")
    print("-" * 60)
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    print("[OK] LoRA 적용 완료")

    # 4. 학습 하이퍼파라미터 설정
    print("\n[4단계] 학습 하이퍼파라미터 설정")
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
    )

    # 5. 데이터셋 토크나이징
    print("\n[5단계] 데이터셋 토크나이징")
    print("-" * 60)

    def tokenize_function(examples):
        """텍스트를 토크나이징합니다."""
        texts = examples["text"]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        # labels를 input_ids와 동일하게 설정 (causal LM)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("[INFO] 학습 데이터셋 토크나이징 중...")
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="토크나이징 (train)",
        num_proc=4,  # 병렬 처리로 토크나이징 속도 향상
    )
    print(f"[OK] 학습 데이터셋 토크나이징 완료: {len(train_tokenized)} 샘플")

    val_tokenized = None
    if val_dataset is not None:
        print("[INFO] 검증 데이터셋 토크나이징 중...")
        val_tokenized = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="토크나이징 (validation)",
            num_proc=4,  # 병렬 처리로 토크나이징 속도 향상
        )
        print(f"[OK] 검증 데이터셋 토크나이징 완료: {len(val_tokenized)} 샘플")

    # 6. 데이터 콜레이터 설정
    print("\n[6단계] 데이터 콜레이터 설정")
    print("-" * 60)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    print("[OK] 데이터 콜레이터 설정 완료")

    # 7. Trainer 생성 (SFTTrainer 대신 Trainer 사용)
    print("\n[7단계] Trainer 생성")
    print("-" * 60)
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized if val_tokenized is not None else None,
        data_collator=data_collator,
    )
    print("[OK] Trainer 생성 완료")

    # 8. 학습 실행
    print("\n[8단계] 학습 실행")
    print("-" * 60)
    print("[INFO] 학습 시작...")
    print(f"  - 학습 샘플 수: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"  - 검증 샘플 수: {len(val_dataset)}")
    print(
        f"  - 총 스텝 수: {len(train_dataset) // (per_device_train_batch_size * gradient_accumulation_steps) * num_epochs}"
    )

    trainer.train()
    print("[OK] 학습 완료")

    # 9. LoRA 어댑터 저장
    print("\n[9단계] LoRA 어댑터 저장")
    print("-" * 60)
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    print(f"[OK] LoRA 어댑터 저장 완료: {output_dir}")

    print("\n" + "=" * 60)
    print("[OK] LoRA 어댑터 학습 완료!")
    print("=" * 60)
    print(f"\n저장된 어댑터 경로: {output_dir}")
    print("\n사용 방법:")
    print("  from peft import PeftModel")
    print("  ")
    print("  # Base 모델 로드")
    print("  base_model = AutoModelForCausalLM.from_pretrained(...)")
    print("  ")
    print("  # LoRA 어댑터 로드")
    print(f"  model = PeftModel.from_pretrained(base_model, '{output_dir}')")

    return output_dir


def main():
    """메인 함수."""
    print("=" * 60)
    print("EXAONE-2.4B LoRA 어댑터 학습")
    print("=" * 60)

    # 1. 모델 로드
    print("\n[1단계] 모델 로드")
    print("-" * 60)
    model, tokenizer = load_exaone_model()

    # 2. 데이터셋 로드
    print("\n[2단계] 데이터셋 로드")
    print("-" * 60)
    dataset_dir = app_dir / "data" / "spam_agent_processed" / "exaone"

    if not dataset_dir.exists():
        print(f"[ERROR] 데이터셋 디렉토리를 찾을 수 없습니다: {dataset_dir}")
        print("\n먼저 다음 명령을 실행하여 데이터셋을 준비하세요:")
        print("  python api/app/service/training_spam_agent/transform_prepare_datasets.py")
        sys.exit(1)

    print(f"[INFO] 데이터셋 디렉토리: {dataset_dir}")
    train_dataset, val_dataset = load_datasets(
        dataset_dir,
        splits=["train", "validation"],
        format="arrow",
    )

    print("[OK] 데이터셋 로드 완료")
    print(f"  - Train: {len(train_dataset)} 샘플")
    print(f"  - Validation: {len(val_dataset)} 샘플")

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
        num_epochs=1,
        per_device_train_batch_size=4,  # 속도 개선: 2 → 4 (TRAINING_ANALYSIS.md 반영)
        gradient_accumulation_steps=4,  # 속도 개선: 8 → 4 (배치 크기 증가로 Effective batch size 유지: 4 * 4 = 16)
        learning_rate=2e-4,
        warmup_steps=100,
        max_seq_length=256,  # 속도 개선: 512 → 384 → 256 (VRAM 절약)
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
