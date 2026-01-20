#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""스팸 메일 판단 에이전트 데이터 전처리 실행 스크립트.

이 스크립트를 실행하면 SFT 학습을 위한 Dataset 파일들이 생성됩니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
spam_agent_dir = current_file.parent  # api/training/agents/spam_agent/
agents_dir = spam_agent_dir.parent  # api/training/agents/
training_dir = agents_dir.parent  # api/training/
api_dir = training_dir.parent  # api/
project_root = api_dir.parent  # 프로젝트 루트

sys.path.insert(0, str(api_dir))

from training.agents.spam_agent import (
    DataPreprocessor,
    DataSplitter,
    TokenizerUtils,
    create_datasets_from_examples,
    save_datasets,
)


def main():
    """메인 함수."""
    # 경로 설정 (training 폴더 내 data 디렉토리 사용)
    data_dir = training_dir / "data" / "spam_agent"
    sft_jsonl_path = (
        data_dir / "한국우편사업진흥원_스팸메일 수신차단 목록_20241231.sft.jsonl"
    )

    # 출력 디렉토리
    output_dir = data_dir / "spam_agent_processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("스팸 메일 판단 에이전트 데이터 전처리")
    print("=" * 60)

    # 입력 파일 확인
    if not sft_jsonl_path.exists():
        print(f"\n[ERROR] 입력 파일을 찾을 수 없습니다: {sft_jsonl_path}")
        print("\n사용 가능한 파일:")
        for f in data_dir.glob("*.sft.jsonl"):
            print(f"  - {f.name}")
        sys.exit(1)

    # 1. 데이터 전처리
    print("\n[1단계] 데이터 전처리")
    print("-" * 60)
    preprocessor = DataPreprocessor(
        max_subject_length=200,
        min_confidence=0.85,
    )

    preprocessed_data, quality_report = preprocessor.process(
        jsonl_path=sft_jsonl_path,
        output_path=None,  # 중간 파일 저장 안 함
    )

    # 2. 데이터셋 변환
    print("\n[2단계] HuggingFace Dataset 변환")
    print("-" * 60)
    dataset = create_datasets_from_examples(preprocessed_data, text_column="text")

    # 3. 토크나이저 준비 및 시퀀스 길이 분석
    print("\n[3단계] 토크나이저 준비 및 시퀀스 길이 분석")
    print("-" * 60)

    # Exaone 모델 경로 (artifacts 디렉토리 사용)
    # 실제 경로: api/artifacts/exaone/exaone3.5-2.4b
    model_dir = api_dir / "artifacts" / "exaone" / "exaone3.5-2.4b"
    if not model_dir.exists():
        # HuggingFace 모델 ID 사용
        model_path = "ai-datacenter/exaone-2.4b"
        print(f"[INFO] 로컬 모델을 찾을 수 없어 HuggingFace 모델을 사용합니다: {model_path}")
    else:
        model_path = str(model_dir)
        print(f"[INFO] 로컬 모델 사용: {model_path}")

    try:
        tokenizer_utils = TokenizerUtils(
            model_path=model_path,
            max_length=512,
            trust_remote_code=True,
        )

        # 시퀀스 길이 분석
        tokenizer_utils.print_length_statistics(dataset, text_column="text")

        # 최적 max_length 계산
        optimal_max_length = tokenizer_utils.get_optimal_max_length(
            dataset, text_column="text", percentile=0.95
        )

        # max_length 업데이트 (필요한 경우)
        if optimal_max_length != tokenizer_utils.max_length:
            print(f"\n[INFO] max_length를 {optimal_max_length}로 업데이트합니다.")
            tokenizer_utils.max_length = optimal_max_length

    except Exception as e:
        print(f"\n[WARNING] 토크나이저 로드 실패: {e}")
        print("[INFO] 토크나이징 단계를 건너뜁니다. Dataset만 생성합니다.")

    # 4. 데이터 분할
    print("\n[4단계] Train/Validation/Test 분할")
    print("-" * 60)
    splitter = DataSplitter(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=42,
        stratify=True,
    )

    dataset_dict = splitter.split_dataset(dataset)

    # 5. Dataset 저장 (SFTTrainer용)
    print("\n[5단계] Dataset 저장 (SFTTrainer용)")
    print("-" * 60)
    save_datasets(
        train_dataset=dataset_dict["train"],
        val_dataset=dataset_dict["validation"],
        test_dataset=dataset_dict["test"],
        output_dir=output_dir,
        save_jsonl=False,  # JSONL은 불필요 (Arrow 형식만 사용)
        save_arrow=True,   # Arrow 형식만 저장 (SFTTrainer에서 바로 사용)
    )

    print("\n" + "=" * 60)
    print("데이터 전처리 완료!")
    print("=" * 60)
    print(f"\n출력 디렉토리: {output_dir}")
    print(f"\n생성된 파일 (Arrow 형식만 - SFTTrainer에서 바로 사용 가능):")
    print(f"  - train_dataset/")
    print(f"  - validation_dataset/")
    print(f"  - test_dataset/")
    print(f"\n사용 방법:")
        print(f"  from training.agents.spam_agent import load_datasets")
    print(f"  from pathlib import Path")
    print(f"  ")
    print(f"  train_dataset, val_dataset = load_datasets(")
    print(f"      Path('app/data/spam_agent_processed'),")
    print(f"      splits=['train', 'validation'],")
    print(f"      format='arrow'")
    print(f"  )")
    print(f"\n[참고] JSONL 파일은 생성하지 않습니다 (Arrow 형식만 사용).")


if __name__ == "__main__":
    main()
