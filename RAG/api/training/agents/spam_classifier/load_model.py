#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KoElectra-small-v3-discriminator 모델 로드 스크립트."""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
spam_classifier_dir = current_file.parent  # api/training/agents/spam_classifier/
agents_dir = spam_classifier_dir.parent  # api/training/agents/
training_dir = agents_dir.parent  # api/training/
api_dir = training_dir.parent  # api/

sys.path.insert(0, str(api_dir))


def load_koelectra_model():
    """KoElectra-small-v3-discriminator 모델을 로드합니다.

    Returns:
        (model, tokenizer) 튜플
    """
    print("=" * 60)
    print("KoElectra-small-v3-discriminator 모델 로드")
    print("=" * 60)

    # 공통 모델 로더 사용 (HuggingFace 캐시 활용)
    from app.core.loaders import ModelLoader

    print("\n[INFO] HuggingFace 캐시를 활용하여 베이스 모델 로드")
    print(f"[INFO] 모델 ID: {ModelLoader.KOELECTRA_MODEL_ID}")

    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] 디바이스: {device}")

    # 모델 로드 (아답터 없이 베이스 모델만)
    print("\n[INFO] 모델 로드 중...")
    try:
        model, tokenizer = ModelLoader.load_koelectra_model(
            adapter_name=None,  # 학습용이므로 아답터 없이
            device=device,
            num_labels=2,
        )
        print("[OK] 모델 로드 완료")
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        import traceback
        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
        raise

    # 간단한 추론 테스트
    print("\n[INFO] 추론 테스트 중...")
    try:
        test_input = "안녕하세요"
        inputs = tokenizer(test_input, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        print("[OK] 추론 테스트 완료")
        print(f"  - 입력: {test_input}")
        print(f"  - 출력 shape: {logits.shape}")
    except Exception as e:
        print(f"[WARNING] 추론 테스트 실패: {e}")

    print("\n" + "=" * 60)
    print("[OK] KoElectra 모델 로드 완료!")
    print("=" * 60)

    return model, tokenizer


if __name__ == "__main__":
    try:
        model, tokenizer = load_koelectra_model()
        print("\n[SUCCESS] 모델이 성공적으로 로드되었습니다.")
    except Exception as e:
        print(f"\n[FAILED] 모델 로드 실패: {e}")
        sys.exit(1)
