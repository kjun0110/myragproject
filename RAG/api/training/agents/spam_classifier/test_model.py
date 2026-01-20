#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""학습된 LoRA 어댑터 모델 테스트 스크립트."""

import json
import sys
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
spam_classifier_dir = current_file.parent  # api/training/agents/spam_classifier/
agents_dir = spam_classifier_dir.parent  # api/training/agents/
training_dir = agents_dir.parent  # api/training/
api_dir = training_dir.parent  # api/

sys.path.insert(0, str(api_dir))


def load_model_with_lora(
    base_model_path: Optional[str] = None,
    lora_adapter_path: Optional[str] = None,
):
    """Base 모델과 LoRA 어댑터를 로드합니다.

    Args:
        base_model_path: Base 모델 경로 (None이면 HuggingFace에서 로드)
        lora_adapter_path: LoRA 어댑터 경로 (None이면 자동으로 찾기)

    Returns:
        (model, tokenizer) 튜플
    """
    print("=" * 60)
    print("LoRA 어댑터 모델 로드")
    print("=" * 60)

    # Base 모델 경로 결정 (artifacts 디렉토리 사용)
    # 실제 경로: api/artifacts/koelectra/koelectra-small-v3-discriminator
    if base_model_path is None:
        model_dir = api_dir / "artifacts" / "koelectra" / "koelectra-small-v3-discriminator"
        if model_dir.exists() and (model_dir / "config.json").exists():
            base_model_path = str(model_dir)
        else:
            base_model_path = "monologg/koelectra-small-v3-discriminator"

    # LoRA 어댑터 경로 결정 (artifacts 디렉토리 사용)
    # 실제 경로: api/artifacts/koelectra/spam_adapter/koelectra-small-v3-discriminator-spam-lora
    if lora_adapter_path is None:
        lora_dir = api_dir / "artifacts" / "koelectra" / "spam_adapter" / "koelectra-small-v3-discriminator-spam-lora"
        if lora_dir.exists():
            # 가장 최근 디렉토리 찾기
            subdirs = [d for d in lora_dir.iterdir() if d.is_dir()]
            if subdirs:
                lora_adapter_path = str(max(subdirs, key=lambda x: x.stat().st_mtime))
            else:
                lora_adapter_path = str(lora_dir)
        else:
            raise FileNotFoundError(
                f"LoRA 어댑터를 찾을 수 없습니다: {lora_dir}\n"
                "lora_adapter_path를 직접 지정하세요."
            )

    print(f"\n[INFO] Base 모델: {base_model_path}")
    print(f"[INFO] LoRA 어댑터: {lora_adapter_path}")

    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] 디바이스: {device}")

    # 토크나이저 로드
    print("\n[INFO] 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    print("[OK] 토크나이저 로드 완료")

    # Base 모델 로드
    print("\n[INFO] Base 모델 로드 중...")
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_path)
    base_model.to(device)
    print("[OK] Base 모델 로드 완료")

    # LoRA 어댑터 로드
    print("\n[INFO] LoRA 어댑터 로드 중...")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    model.eval()
    print("[OK] LoRA 어댑터 로드 완료")

    print("\n" + "=" * 60)
    print("[OK] 모델 로드 완료!")
    print("=" * 60)

    return model, tokenizer


def predict(model, tokenizer, text: str, device: str = "cuda"):
    """텍스트에 대한 스팸 분류 예측을 수행합니다.

    Args:
        model: 로드된 모델
        tokenizer: 토크나이저
        text: 분류할 텍스트
        device: 디바이스

    Returns:
        (predicted_label, confidence) 튜플
        - predicted_label: 0 (정상) 또는 1 (스팸)
        - confidence: 신뢰도 (확률)
    """
    # 토크나이징
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 예측
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)

    # 레이블과 확률 추출
    predicted_label = probabilities.argmax(dim=-1).item()
    confidence = probabilities[0][predicted_label].item()

    return predicted_label, confidence


def load_test_dataset(test_jsonl_path: Path) -> list:
    """test.jsonl 파일을 로드합니다.

    Args:
        test_jsonl_path: test.jsonl 파일 경로

    Returns:
        [(text, label), ...] 형식의 리스트
    """
    test_data = []
    if not test_jsonl_path.exists():
        print(f"[WARNING] {test_jsonl_path} 파일을 찾을 수 없습니다.")
        return test_data

    print(f"[INFO] {test_jsonl_path} 로드 중...")
    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                text = item.get("text", "")
                label = item.get("label", 0)
                if text:
                    test_data.append((text, label))
            except json.JSONDecodeError as e:
                print(f"[WARNING] {test_jsonl_path} {line_num}번째 줄 파싱 실패: {e}")
                continue

    print(f"[OK] test.jsonl 로드 완료: {len(test_data)} 샘플")
    return test_data


def test_model(
    model,
    tokenizer,
    test_texts: Optional[list] = None,
    device: str = "cuda",
):
    """모델을 테스트합니다.

    Args:
        model: 로드된 모델
        tokenizer: 토크나이저
        test_texts: 테스트할 텍스트 리스트 (None이면 test.jsonl 사용)
        device: 디바이스
    """
    print("\n" + "=" * 60)
    print("모델 테스트")
    print("=" * 60)

    if test_texts is None:
        # test.jsonl 파일 로드
        test_jsonl_path = (
            app_dir / "data" / "spam_agent_processed" / "koelectra" / "test.jsonl"
        )
        test_texts = load_test_dataset(test_jsonl_path)

        # test.jsonl이 없거나 비어있으면 기본 예시 사용
        if not test_texts:
            print("[INFO] test.jsonl이 없어 기본 예시를 사용합니다.")
            test_texts = [
                ("제목: (광고) 건강보험료 할인 이벤트", 1),  # 스팸 예상
                ("제목: 회의 일정 안내\n첨부파일: 없음", 0),  # 정상 예상
                ("제목: 무료 체험 이벤트 참여하세요!", 1),  # 스팸 예상
                ("제목: 프로젝트 진행 상황 보고", 0),  # 정상 예상
            ]

    print("\n[INFO] 테스트 시작...\n")
    correct = 0
    total = len(test_texts)

    # 처음 5개 샘플에 대해 상세 로그 출력
    detailed_samples = min(5, total)

    for i, (text, expected_label) in enumerate(test_texts, 1):
        predicted_label, confidence = predict(model, tokenizer, text, device)

        # 결과 표시
        status = "✓" if predicted_label == expected_label else "✗"
        label_name = "스팸" if predicted_label == 1 else "정상"
        expected_name = "스팸" if expected_label == 1 else "정상"

        # 처음 5개 샘플에 대해 상세 로그 출력
        if i <= detailed_samples:
            print(f"\n{'=' * 60}")
            print(f"[예시 {i}/{detailed_samples}]")
            print(f"{'=' * 60}")
            print("입력 텍스트:")
            print(f"  {text[:200]}{'...' if len(text) > 200 else ''}")
            print(f"\n예상 레이블: {expected_name} ({expected_label})")
            print(f"예측 레이블: {label_name} ({predicted_label})")
            print(f"신뢰도: {confidence:.4f} ({confidence * 100:.2f}%)")
            print(
                f"정확도: {status} {'(정답)' if predicted_label == expected_label else '(오답)'}"
            )

            # 각 클래스별 확률도 출력
            inputs = tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                prob_normal = probabilities[0][0].item()
                prob_spam = probabilities[0][1].item()

            print("\n클래스별 확률:")
            print(f"  정상 (0): {prob_normal:.4f} ({prob_normal * 100:.2f}%)")
            print(f"  스팸 (1): {prob_spam:.4f} ({prob_spam * 100:.2f}%)")
            print(f"{'=' * 60}\n")

        # 진행 상황 출력 (1000개마다)
        if i % 1000 == 0 or i == total:
            current_accuracy = correct / (i - 1) * 100 if i > 1 else 0
            print(
                f"[진행] {i}/{total} 처리 완료... (정확도: {correct}/{i - 1} = {current_accuracy:.2f}%)"
            )

        if predicted_label == expected_label:
            correct += 1

    # 정확도 출력
    accuracy = correct / total * 100
    print("\n" + "=" * 60)
    print(f"최종 정확도: {correct}/{total} ({accuracy:.2f}%)")
    print("=" * 60)


def main():
    """메인 함수."""
    import argparse

    parser = argparse.ArgumentParser(description="LoRA 어댑터 모델 테스트")
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="LoRA 어댑터 경로 (None이면 자동으로 찾기)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base 모델 경로 (None이면 HuggingFace에서 로드)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="테스트할 단일 텍스트",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="테스트할 JSONL 파일 경로 (None이면 koelectra/test.jsonl 사용)",
    )

    args = parser.parse_args()

    try:
        # 모델 로드
        model, tokenizer = load_model_with_lora(
            base_model_path=args.base_model,
            lora_adapter_path=args.lora_path,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 단일 텍스트 테스트
        if args.text:
            print("\n[INFO] 단일 텍스트 예측:")
            predicted_label, confidence = predict(model, tokenizer, args.text, device)
            label_name = "스팸" if predicted_label == 1 else "정상"
            print(f"  입력: {args.text}")
            print(f"  예측: {label_name} ({predicted_label})")
            print(f"  신뢰도: {confidence:.4f}")
        else:
            # test.jsonl 파일 사용
            if args.test_file:
                test_jsonl_path = Path(args.test_file)
            else:
                test_jsonl_path = (
                    app_dir
                    / "data"
                    / "spam_agent_processed"
                    / "koelectra"
                    / "test.jsonl"
                )

            test_data = load_test_dataset(test_jsonl_path)
            test_model(
                model,
                tokenizer,
                test_texts=test_data if test_data else None,
                device=device,
            )

        print("\n[SUCCESS] 테스트 완료!")

    except Exception as e:
        print(f"\n[ERROR] 테스트 실패: {e}")
        import traceback

        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
