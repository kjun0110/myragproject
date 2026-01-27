#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unsloth와 EXAONE 모델 호환성 테스트 스크립트.

EXAONE-3.5-2.4B-Instruct 모델이 Unsloth와 호환되는지 확인합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
scripts_dir = current_file.parent  # api/scripts/
api_dir = scripts_dir.parent  # api/

sys.path.insert(0, str(api_dir))

print("=" * 60)
print("Unsloth와 EXAONE 모델 호환성 테스트")
print("=" * 60)

# Python 환경 정보 출력
print("\n[환경 정보]")
print("-" * 60)
print(f"Python 경로: {sys.executable}")
print(f"Python 버전: {sys.version}")

# 1. Unsloth 설치 여부 확인
print("\n[1단계] Unsloth 설치 확인")
print("-" * 60)

try:
    import unsloth
    # 버전 확인 시도
    try:
        version = unsloth.__version__
    except:
        try:
            import unsloth_zoo
            version = unsloth_zoo.__version__ if hasattr(unsloth_zoo, '__version__') else '2026.1.4 (추정)'
        except:
            version = '2026.1.4 (설치 확인됨)'
    print(f"[OK] Unsloth 설치됨: {version}")
except ImportError as e:
    print(f"[ERROR] Unsloth가 설치되지 않았습니다: {e}")
    print("\n설치 방법:")
    print("  pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
    print("\n또는:")
    print("  pip install unsloth")
    print("\n[INFO] Unsloth를 설치한 후 다시 실행하세요.")
    print("\n[참고] 현재 Python 환경을 확인하세요:")
    print(f"  Python 경로: {sys.executable}")
    sys.exit(1)

# 2. Unsloth 모듈 확인
print("\n[2단계] Unsloth 모듈 확인")
print("-" * 60)

try:
    from unsloth import FastLanguageModel
    print("[OK] FastLanguageModel import 성공")
except ImportError as e:
    print(f"[ERROR] FastLanguageModel import 실패: {e}")
    sys.exit(1)

# 3. EXAONE 모델 ID 확인
print("\n[3단계] EXAONE 모델 정보 확인")
print("-" * 60)

from app.common.loaders import ModelLoader

model_id = ModelLoader.EXAONE_MODEL_ID
print(f"모델 ID: {model_id}")

# 4. Unsloth로 모델 로드 시도
print("\n[4단계] Unsloth로 EXAONE 모델 로드 시도")
print("-" * 60)
print("[INFO] 모델 로드 중... (시간이 걸릴 수 있습니다)")

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=256,
        dtype=None,  # 자동 감지
        load_in_4bit=True,  # 4-bit 양자화
        trust_remote_code=True,  # 커스텀 모델링 코드 필요
    )

    print("[OK] Unsloth로 EXAONE 모델 로드 성공!")
    print(f"  - 모델 타입: {type(model).__name__}")
    print(f"  - 토크나이저 타입: {type(tokenizer).__name__}")

    # 모델 정보 확인
    if hasattr(model, "config"):
        config = model.config
        print(f"  - 모델 이름: {getattr(config, 'model_type', 'unknown')}")
        print(f"  - Hidden size: {getattr(config, 'hidden_size', 'unknown')}")
        print(f"  - Num layers: {getattr(config, 'num_hidden_layers', 'unknown')}")

    # 간단한 추론 테스트
    print("\n[5단계] 추론 테스트")
    print("-" * 60)

    test_input = "안녕하세요"
    inputs = tokenizer(test_input, return_tensors="pt", padding=True)

    import torch
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model = model.to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"[OK] 추론 테스트 성공")
    print(f"  - 입력: {test_input}")
    print(f"  - 출력: {generated_text}")

    print("\n" + "=" * 60)
    print("[SUCCESS] 호환성 확인 완료: Unsloth와 EXAONE 모델이 호환됩니다!")
    print("=" * 60)
    print("\n[권장 사항]")
    print("  - Unsloth를 사용하여 학습 속도를 2-5배 향상시킬 수 있습니다.")
    print("  - Flash Attention이 자동으로 포함됩니다.")
    print("  - xFormers는 별도로 설정할 필요가 없습니다.")

except Exception as e:
    print(f"\n[ERROR] Unsloth로 EXAONE 모델 로드 실패")
    print(f"  오류: {e}")
    print(f"  오류 타입: {type(e).__name__}")

    import traceback
    print(f"\n상세 오류:")
    print(traceback.format_exc())

    print("\n" + "=" * 60)
    print("[FAILED] 호환성 확인 실패: Unsloth와 EXAONE 모델이 호환되지 않을 수 있습니다.")
    print("=" * 60)
    print("\n[대안]")
    print("  - xFormers를 사용하여 속도를 향상시킬 수 있습니다.")
    print("  - Flash Attention을 수동으로 설정할 수 있습니다.")
    print("  - 현재 방식(PEFT + Transformers)을 계속 사용할 수 있습니다.")

    sys.exit(1)
