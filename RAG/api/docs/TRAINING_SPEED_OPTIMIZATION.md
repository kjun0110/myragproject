# 학습 속도 최적화 가이드

## 현재 상황

- **현재 속도**: 9.28초/it (매우 느림)
- **총 스텝**: 1,228 스텝
- **예상 시간**: 약 3.2시간

---

## 현재 사용 중인 라이브러리

### ✅ 사용 중
1. **PEFT** (LoRA) - 사용 중
2. **Transformers Trainer** - 사용 중
3. **BitsAndBytes** (4-bit 양자화) - 사용 중
4. **paged_adamw_8bit** (8-bit 옵티마이저) - 사용 중

### ❌ 사용 안 함
1. **xFormers** - 사용 안 함
2. **Flash Attention** - 사용 안 함
3. **Unsloth** - 사용 안 함
4. **torch.compile** - 사용 안 함

---

## ⚠️ 중요: Attention 구현 방식 선택

**xFormers와 Flash Attention은 둘 중 하나만 선택 가능합니다!**

- 둘 다 attention 연산을 최적화하는 라이브러리
- 동시에 사용할 수 없음
- 하나를 선택하면 다른 하나는 사용 불가

### 선택 기준
1. **Unsloth 사용 시**: Flash Attention 자동 포함 (xFormers 불필요)
2. **Unsloth 미사용 시**: xFormers 또는 Flash Attention 중 하나 선택
   - Windows: xFormers 권장 (설치 쉬움)
   - Linux: Flash Attention 권장 (더 빠름)

---

## 속도 개선 방법 (우선순위 순)

### 🥇 1순위: Unsloth 사용 (가장 효과적)

**효과**: 2-5배 속도 향상 가능

**장점:**
- PEFT + Transformers 대비 훨씬 빠름
- 메모리 효율적
- LoRA 학습에 최적화
- **Flash Attention 자동 포함** (xFormers 불필요)

**⚠️ 중요:**
- Unsloth를 사용하면 **자동으로 Flash Attention이 활성화**됨
- xFormers를 별도로 설정할 필요 없음
- Unsloth가 호환되지 않으면 → xFormers 사용

**설치:**
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# 또는
pip install unsloth
```

**사용 방법:**
```python
from unsloth import FastLanguageModel

# 모델 로드 (Unsloth 사용)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    max_seq_length=256,
    dtype=None,  # 자동 감지
    load_in_4bit=True,  # 4-bit 양자화
    trust_remote_code=True,
)

# LoRA 설정
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
)

# 학습 (Unsloth Trainer 사용)
from unsloth import FastSFTTrainer

trainer = FastSFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    max_seq_length=256,
    dataset_text_field="text",
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)
```

**예상 속도 개선:**
- 현재: 9.28초/it
- Unsloth 사용: **2-4초/it** (약 2-4배 빠름)

---

### 🥈 2순위: Flash Attention 사용 (Unsloth 미사용 시)

**효과**: 1.5-2배 속도 향상 가능

**⚠️ 중요:**
- **Unsloth를 사용하면 Flash Attention이 자동 포함되므로 별도 설정 불필요**
- Unsloth를 사용하지 않을 때만 수동으로 설정

**설치:**
```bash
pip install flash-attn --no-build-isolation
# 또는
pip install flash-attn
```

**사용 방법:**
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",  # Flash Attention 사용
    # ... 기타 옵션
)
```

**주의사항:**
- CUDA 11.8+ 필요
- Windows에서는 설치가 어려울 수 있음
- **xFormers와 동시 사용 불가** (둘 중 하나만 선택)

---

### 🥉 3순위: torch.compile 사용

**효과**: 1.2-1.5배 속도 향상 가능

**사용 방법:**
```python
import torch

# 모델 컴파일
model = torch.compile(model, mode="reduce-overhead")

# 또는 더 공격적인 최적화
model = torch.compile(model, mode="max-autotune")
```

**주의사항:**
- PyTorch 2.0+ 필요
- 첫 실행 시 컴파일 시간 소요 (느림)
- 이후 실행 시 빠름

---

### 4순위: xFormers 사용 ⚠️ 중요 (Unsloth 미사용 시)

**효과**: 1.2-2배 속도 향상 가능 (Flash Attention과 유사)

**⚠️ 중요:**
- **Unsloth를 사용하면 Flash Attention이 자동 포함되므로 xFormers 불필요**
- Unsloth를 사용하지 않을 때만 xFormers 사용
- **Flash Attention과 동시 사용 불가** (둘 중 하나만 선택)

**xFormers란?**
- Facebook(Meta)에서 개발한 효율적인 attention 구현
- 메모리 효율적이고 빠른 attention 연산
- Flash Attention과 유사한 성능
- Windows에서 설치가 더 쉬움

**설치:**
```bash
# Windows (권장)
pip install xformers

# 또는 특정 버전
pip install xformers==0.0.23.post1
```

**사용 방법:**
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="xformers",  # xFormers 사용
    # ... 기타 옵션
)
```

**현재 코드 적용 위치:**
- `api/app/common/loaders/model_loader.py`의 `load_exaone_model()` 함수
- `api/training/agents/spam_agent/load_model.py`의 `load_exaone_model()` 함수

**주의사항:**
- Windows에서도 설치 가능 (최신 버전)
- Flash Attention보다 설치가 쉬움
- CUDA 11.8+ 권장
- 일부 모델에서는 Flash Attention이 더 빠를 수 있음

**xFormers vs Flash Attention:**
| 항목 | xFormers | Flash Attention |
|-----|---------|----------------|
| **속도** | 빠름 (1.2-2배) | 매우 빠름 (1.5-2.5배) |
| **설치** | 쉬움 (Windows 지원) | 어려움 (Windows 어려움) |
| **메모리** | 효율적 | 매우 효율적 |
| **호환성** | 높음 | 중간 |

---

## 권장 조합

### 최적 조합 1: Unsloth 사용 (가장 빠름)
1. **Unsloth** (필수) - 2-5배 향상
2. **torch.compile** (선택) - 추가 1.2-1.5배 향상

**예상 속도:**
- 현재: 9.28초/it
- 최적화 후: **1.5-3초/it** (약 3-6배 빠름)

### 최적 조합 2: xFormers 사용 (Unsloth 대안)
1. **xFormers** (필수) - 1.2-2배 향상
2. **torch.compile** (선택) - 추가 1.2-1.5배 향상

**예상 속도:**
- 현재: 9.28초/it
- 최적화 후: **3-5초/it** (약 2-3배 빠름)

### 최적 조합 3: Flash Attention 사용 (가장 빠르지만 설치 어려움)
1. **Flash Attention** (필수) - 1.5-2.5배 향상
2. **torch.compile** (선택) - 추가 1.2-1.5배 향상

**예상 속도:**
- 현재: 9.28초/it
- 최적화 후: **2-4초/it** (약 2.5-4.5배 빠름)

---

## Unsloth vs 현재 방식 비교

| 항목 | 현재 (PEFT + Trainer) | Unsloth |
|-----|---------------------|---------|
| **속도** | 9.28초/it | 2-4초/it |
| **메모리** | 표준 | 더 효율적 |
| **설치** | 간단 | 간단 |
| **호환성** | 높음 | 높음 |
| **Flash Attention** | 수동 설정 | 자동 포함 |
| **최적화** | 수동 | 자동 |

---

## Unsloth 사용 시 주의사항

### 1. 모델 호환성
- EXAONE 모델이 Unsloth를 지원하는지 확인 필요
- `trust_remote_code=True` 필요할 수 있음

### 2. 커스텀 모델링 코드
- EXAONE은 커스텀 `modeling_exaone.py` 사용
- Unsloth가 이를 지원하는지 확인 필요

### 3. Windows 호환성
- Unsloth는 Linux/Colab에서 더 잘 작동
- Windows에서도 작동하지만 일부 최적화가 제한될 수 있음

---

## 즉시 적용 가능한 최적화 (코드 수정 없이)

### 1. 배치 크기 증가
- 현재: `per_device_train_batch_size=4`
- 권장: 메모리 허용 시 8-16으로 증가
- 효과: GPU 활용률 증가 → 속도 향상

### 2. 시퀀스 길이 감소
- 현재: `max_seq_length=256`
- 이미 최적화됨 (512 → 256)

### 3. Gradient Accumulation 감소
- 현재: `gradient_accumulation_steps=4`
- 배치 크기 증가 시 2로 감소 가능
- 효과: 업데이트 빈도 증가 → 약간 빠름

---

## 단계별 적용 계획

### Phase 1: 즉시 적용 (코드 수정 최소)
1. 배치 크기 증가 (메모리 확인 후)
2. Gradient Accumulation 감소

**예상 개선**: 10-20% 속도 향상

### Phase 2: Unsloth 도입 (권장)
1. Unsloth 설치
2. 코드 수정 (모델 로드 부분)
3. Trainer를 FastSFTTrainer로 변경

**예상 개선**: 2-4배 속도 향상

### Phase 3: 추가 최적화
1. torch.compile 적용
2. Flash Attention 확인 (Unsloth에 포함되어 있을 수 있음)

**예상 개선**: 추가 20-50% 속도 향상

---

## 예상 최종 속도

### 현재
- 속도: 9.28초/it
- 총 시간: 약 3.2시간

### Phase 1 적용 후
- 속도: 약 8초/it
- 총 시간: 약 2.7시간

### Phase 2 적용 후

#### xFormers 사용 시
- 속도: 약 4-7초/it
- 총 시간: 약 1.4-2.4시간

#### Unsloth 사용 시
- 속도: 약 2-4초/it
- 총 시간: 약 0.7-1.4시간

### Phase 3 적용 후 (전체 최적화)
- 속도: 약 1.5-3초/it
- 총 시간: 약 0.5-1시간

---

## 참고 자료

- [Unsloth 공식 문서](https://github.com/unslothai/unsloth)
- [Flash Attention 문서](https://github.com/Dao-AILab/flash-attention)
- [torch.compile 문서](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
