# LoRA 학습 과정 및 속도 분석

## 실행 흐름 (`lora_adapter.py`)

### 1단계: 모델 로드
- **파일**: `load_model.py`
- **작업**: EXAONE-2.4B 모델을 4-bit 양자화로 로드
- **위치**: `api/app/model/exaone3.5/exaone-2.4b/`
- **속도 영향**: 초기 로드만 (약 3분), 학습 중 영향 없음

### 2단계: 데이터셋 로드
- **파일**: `transform_dataset_utils.py` → `load_datasets()`
- **데이터 위치**: `api/app/data/spam_agent_processed/`
  - `train_dataset/` (Arrow 형식)
  - `validation_dataset/` (Arrow 형식)
- **작업**: HuggingFace Dataset 객체로 로드
- **속도 영향**: 초기 로드만, 학습 중 영향 없음

### 3단계: PEFT 모델 준비
- **작업**: `prepare_model_for_kbit_training()`
- **속도 영향**: 초기화만

### 4단계: LoRA 적용
- **작업**: `get_peft_model()` - LoRA 어댑터 추가
- **속도 영향**: 초기화만

### 5단계: 데이터셋 토크나이징 ⚠️
- **작업**: `train_dataset.map(tokenize_function, num_proc=4)`
- **위치**: 메모리에서 실행 (이미 로드된 데이터셋)
- **속도 영향**:
  - **병목 가능**: `num_proc=4`로 병렬 처리하지만 CPU 기반
  - **한 번만 실행**: 학습 시작 전에만 실행
  - **현재**: 이미 완료된 상태 (캐시됨)

### 6단계: 데이터 콜레이터 설정
- **작업**: `DataCollatorForLanguageModeling` 생성
- **속도 영향**: 초기화만

### 7단계: Trainer 생성
- **작업**: `Trainer` 객체 생성
- **속도 영향**: 초기화만

### 8단계: 학습 실행 ⚠️⚠️⚠️
- **작업**: `trainer.train()`
- **현재 속도**: **19.03초/it** (매우 느림)

## 학습 중 실행되는 과정 (각 스텝마다)

### Forward Pass (순전파)
1. **데이터 로딩** (CPU → GPU)
   - DataLoader가 배치를 로드
   - `num_workers=0`: 메인 프로세스에서 로드 (느림)
   - `pin_memory=False`: CPU-GPU 전송 최적화 없음
   - **속도 영향**: ⚠️ 중간

2. **모델 Forward**
   - 입력 토큰 → 모델 레이어들 통과
   - 30개 레이어 (ExaoneForCausalLM)
   - 4-bit 양자화된 가중치 사용
   - **속도 영향**: ⚠️⚠️ 높음 (GPU 연산)

3. **Gradient Checkpointing**
   - 활성화됨 (`gradient_checkpointing=True`)
   - 중간 활성화를 저장하지 않고 필요시 재계산
   - **속도 영향**: ⚠️⚠️⚠️ 매우 높음 (재계산으로 인한 속도 저하)
   - **메모리**: 절약 (필수)

### Backward Pass (역전파)
1. **Loss 계산**
   - 모델 출력과 labels 비교
   - **속도 영향**: 낮음

2. **Gradient 계산**
   - 각 파라미터에 대한 그래디언트 계산
   - LoRA 파라미터만 업데이트 (전체 모델의 일부)
   - **속도 영향**: ⚠️⚠️ 높음

3. **Gradient Accumulation**
   - `gradient_accumulation_steps=8`: 8번 누적 후 업데이트
   - **속도 영향**: 중간 (업데이트 빈도 감소)

### 파라미터 업데이트
1. **Optimizer Step**
   - `paged_adamw_8bit`: 8-bit 옵티마이저
   - LoRA 파라미터만 업데이트
   - **속도 영향**: 낮음

## 속도에 영향을 미치는 주요 요소

### 1. Gradient Checkpointing ⚠️⚠️⚠️ (가장 큰 영향)
- **현재**: 활성화됨
- **영향**: Forward pass를 2번 실행 (저장 + 재계산)
- **속도 저하**: 약 2배 느려짐
- **필수**: 메모리 부족 방지 (6GB GPU)

### 2. 배치 크기 ⚠️⚠️
- **현재**: `per_device_train_batch_size=2`
- **영향**: 작은 배치 = GPU 활용률 낮음
- **속도**: 배치 크기 증가 시 속도 향상 (메모리 허용 시)

### 3. Gradient Accumulation ⚠️
- **현재**: `gradient_accumulation_steps=8`
- **영향**: 8번 forward 후 1번 업데이트
- **속도**: 업데이트 빈도 감소로 약간 느림

### 4. DataLoader 설정 ⚠️
- **현재**: `num_workers=0`, `pin_memory=False`
- **영향**: 데이터 로딩이 학습과 동기화 (병목 가능)
- **속도**: 병렬 로딩 없음

### 5. 시퀀스 길이 ⚠️
- **현재**: `max_seq_length=512`
- **영향**: 긴 시퀀스 = 더 많은 연산
- **속도**: 시퀀스 길이에 비례

### 6. 모델 크기
- **현재**: EXAONE-2.4B (30 레이어)
- **영향**: 큰 모델 = 더 많은 연산
- **속도**: 모델 크기에 비례

### 7. GPU 성능
- **현재**: RTX 3050 (6GB)
- **영향**: GPU 연산 속도
- **속도**: GPU 성능에 비례

## 현재 설정 요약

```
배치 크기: 2
Gradient Accumulation: 8
Effective Batch Size: 16
Gradient Checkpointing: 활성화 (필수)
DataLoader Workers: 0
Pin Memory: False
시퀀스 길이: 512
```

## 속도 개선 방안

### 즉시 적용 가능 (메모리 허용 시)
1. **배치 크기 증가**: 2 → 4 (메모리 확인 필요)
2. **Gradient Accumulation 감소**: 8 → 4 (배치 크기 증가 시)
3. **DataLoader 최적화**: `num_workers=2`, `pin_memory=True` (Windows 호환성 확인)

### 장기적 개선
1. **시퀀스 길이 감소**: 512 → 384 (성능 영향 확인 필요)
2. **더 강력한 GPU**: RTX 3050 → RTX 3060 이상
3. **Mixed Precision 최적화**: bf16 사용 중 (이미 최적)

## 예상 학습 시간

- **현재 속도**: 19.03초/it
- **총 스텝**: 1,228 스텝
- **예상 시간**: 약 6.5시간

### 최적화 후 예상
- **배치 크기 4, Gradient Accumulation 4**: 약 12-15초/it
- **예상 시간**: 약 4-5시간
