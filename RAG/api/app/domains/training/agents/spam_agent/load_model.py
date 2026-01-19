#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXAONE-2.4B 모델 로드 스크립트.

4-bit 양자화를 사용하여 EXAONE-2.4B 모델을 로드하고 검증합니다.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
spam_agent_dir = current_file.parent  # api/app/service/spam_agent/
service_dir = spam_agent_dir.parent  # api/app/service/
app_dir = service_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

sys.path.insert(0, str(api_dir))


def load_exaone_model():
    """EXAONE-2.4B 모델을 4-bit 양자화로 로드합니다.

    Returns:
        (model, tokenizer) 튜플
    """
    print("=" * 60)
    print("EXAONE-2.4B 모델 로드 (4-bit 양자화)")
    print("=" * 60)

    # 모델 경로 결정 (실제 디렉토리명: exaone3.5-2.4b)
    model_dir = app_dir / "model" / "exaone3.5-2.4b"

    # 대체 경로도 확인 (exaone3.5/exaone-2.4b)
    if not (model_dir.exists() and (model_dir / "config.json").exists()):
        model_dir = app_dir / "model" / "exaone3.5" / "exaone-2.4b"

    if model_dir.exists() and (model_dir / "config.json").exists():
        model_path = str(model_dir)
        print(f"\n[INFO] 로컬 모델 경로: {model_path}")
    else:
        # HuggingFace 모델 ID 사용 (하지만 이 모델은 private이므로 로컬 모델이 필요함)
        model_path = "ai-datacenter/exaone-2.4b"
        print("\n[WARNING] 로컬 모델을 찾을 수 없습니다.")
        print("  확인한 경로:")
        print(f"    - {app_dir / 'model' / 'exaone3.5-2.4b'}")
        print(f"    - {app_dir / 'model' / 'exaone3.5' / 'exaone-2.4b'}")
        print(
            "\n[ERROR] EXAONE 모델은 private repository이므로 로컬 모델이 필요합니다."
        )
        print("  모델을 다음 경로 중 하나에 배치하세요:")
        print(f"    - {app_dir / 'model' / 'exaone3.5-2.4b'}")
        print(f"    - {app_dir / 'model' / 'exaone3.5' / 'exaone-2.4b'}")
        raise FileNotFoundError(
            f"EXAONE 모델을 찾을 수 없습니다. "
            f"로컬 모델 경로를 확인하세요: {app_dir / 'model' / 'exaone3.5-2.4b'}"
        )

    # GPU 사용 가능 여부 확인
    if torch.cuda.is_available():
        print(f"[INFO] GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        print(
            f"[INFO] GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    else:
        print("[WARNING] GPU를 사용할 수 없습니다. CPU로 로드됩니다.")

    # 4-bit 양자화 설정
    print("\n[INFO] 4-bit 양자화 설정 중...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print("[OK] 양자화 설정 완료")

    # 토크나이저 로드
    print("\n[INFO] 토크나이저 로드 중...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        # pad_token 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        print("[OK] 토크나이저 로드 완료")
        print(f"  - Vocab size: {tokenizer.vocab_size}")
        print(f"  - Pad token: {tokenizer.pad_token}")
        print(f"  - EOS token: {tokenizer.eos_token}")
    except Exception as e:
        print(f"[ERROR] 토크나이저 로드 실패: {e}")
        raise

    # 모델 로드 (4-bit 양자화 적용)
    print("\n[INFO] 모델 로드 중 (4-bit 양자화)...")
    print("  이 과정은 몇 분이 걸릴 수 있습니다...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",  # GPU 자동 할당
            trust_remote_code=True,
        )
        print("[OK] 모델 로드 완료")
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        import traceback

        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
        raise

    # 디바이스 정보 확인
    print("\n[INFO] 모델 디바이스 정보:")
    if hasattr(model, "hf_device_map"):
        device_map = model.hf_device_map
        print(f"  - Device map: {device_map}")
        for layer_name, device in device_map.items():
            if isinstance(device, str):
                print(f"    {layer_name}: {device}")
    elif hasattr(model, "device"):
        print(f"  - Device: {model.device}")
    else:
        # 첫 번째 파라미터의 디바이스 확인
        first_param = next(model.parameters())
        print(f"  - Device: {first_param.device}")

    # GPU 메모리 사용량 확인
    if torch.cuda.is_available():
        print("\n[INFO] GPU 메모리 사용량:")
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"  - 할당된 메모리: {allocated:.2f} GB")
        print(f"  - 예약된 메모리: {reserved:.2f} GB")

    # 모델 정보 출력
    print("\n[INFO] 모델 정보:")
    print(f"  - 모델 타입: {type(model).__name__}")
    print("  - 양자화: 4-bit NF4")
    print("  - Compute dtype: bfloat16")

    # 간단한 추론 테스트
    print("\n[INFO] 추론 테스트 중...")
    try:
        test_input = "안녕하세요"
        inputs = tokenizer(test_input, return_tensors="pt")

        # 디바이스로 이동
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("[OK] 추론 테스트 완료")
        print(f"  - 입력: {test_input}")
        print(f"  - 출력: {generated_text}")
    except Exception as e:
        print(f"[WARNING] 추론 테스트 실패: {e}")
        print("  모델은 로드되었지만 추론에 문제가 있을 수 있습니다.")

    print("\n" + "=" * 60)
    print("[OK] EXAONE-2.4B 모델 로드 완료!")
    print("=" * 60)

    return model, tokenizer


if __name__ == "__main__":
    try:
        model, tokenizer = load_exaone_model()
        print("\n[SUCCESS] 모델이 성공적으로 로드되었습니다.")
        print("모델과 토크나이저를 사용할 수 있습니다.")
    except Exception as e:
        print(f"\n[FAILED] 모델 로드 실패: {e}")
        sys.exit(1)
