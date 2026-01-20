#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset 유틸리티 모듈.

SFTTrainer에서 바로 사용할 수 있는 Dataset 객체를 생성하고 관리합니다.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from datasets import Dataset, DatasetDict


def create_datasets_from_examples(
    examples: List[dict], text_column: str = "text"
) -> Dataset:
    """예제 리스트를 HuggingFace Dataset 객체로 변환합니다.

    Args:
        examples: 예제 딕셔너리 리스트 (각 딕셔너리는 text 필드를 포함해야 함)
        text_column: 텍스트 컬럼명 (기본: "text")

    Returns:
        HuggingFace Dataset 객체

    Example:
        >>> examples = [{"text": "### 지시문\n..."}, {"text": "### 지시문\n..."}]
        >>> dataset = create_datasets_from_examples(examples)
        >>> print(len(dataset))
        2
    """
    if not examples:
        raise ValueError("예제 리스트가 비어있습니다.")

    # text_column 검증
    if text_column not in examples[0]:
        raise ValueError(
            f"예제에 '{text_column}' 필드가 없습니다. "
            f"사용 가능한 필드: {list(examples[0].keys())}"
        )

    dataset = Dataset.from_list(examples)
    print(f"[OK] Dataset 생성 완료: {len(dataset)} 샘플")
    return dataset


def save_datasets(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    output_dir: Path = Path("processed"),
    save_jsonl: bool = True,
    save_arrow: bool = True,
) -> None:
    """Dataset 객체를 디스크에 저장합니다.

    Args:
        train_dataset: 학습용 Dataset
        val_dataset: 검증용 Dataset (선택)
        test_dataset: 테스트용 Dataset (선택)
        output_dir: 출력 디렉토리
        save_jsonl: JSONL 형식으로도 저장할지 여부
        save_arrow: Arrow 형식으로도 저장할지 여부

    Example:
        >>> train_dataset = Dataset.from_list([{"text": "..."}])
        >>> val_dataset = Dataset.from_list([{"text": "..."}])
        >>> save_datasets(train_dataset, val_dataset, output_dir=Path("data/processed"))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {"train": train_dataset}
    if val_dataset is not None:
        datasets["validation"] = val_dataset
        # val 별칭은 저장하지 않음 (중복 방지)
    if test_dataset is not None:
        datasets["test"] = test_dataset

    # Arrow 형식 저장 (HuggingFace Dataset 네이티브 형식)
    if save_arrow:
        print("[INFO] Arrow 형식으로 저장 중...")
        for split_name, dataset in datasets.items():
            arrow_path = output_dir / f"{split_name}_dataset"
            print(f"  [{split_name}] {arrow_path}")
            dataset.save_to_disk(str(arrow_path))
            print(f"    [OK] {len(dataset)} 샘플 저장 완료")

    # JSONL 형식 저장 (인간이 읽을 수 있는 형식)
    if save_jsonl:
        print("[INFO] JSONL 형식으로 저장 중...")
        import json

        for split_name, dataset in datasets.items():
            jsonl_path = output_dir / f"{split_name}.jsonl"
            print(f"  [{split_name}] {jsonl_path}")
            with jsonl_path.open("w", encoding="utf-8") as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"    [OK] {len(dataset)} 샘플 저장 완료")

    print(f"\n[OK] 모든 데이터셋 저장 완료: {output_dir}")


def load_datasets(
    dataset_dir: Path,
    splits: Optional[List[str]] = None,
    format: str = "arrow",
) -> Tuple[Dataset, ...]:
    """저장된 Dataset 객체를 로드합니다.

    Args:
        dataset_dir: 데이터셋 디렉토리
        splits: 로드할 split 목록 (None이면 ["train", "validation"] 자동 감지)
        format: 로드 형식 ("arrow" 또는 "jsonl")

    Returns:
        로드된 Dataset 객체들의 튜플 (순서: train, val, test)

    Example:
        >>> train_dataset, val_dataset = load_datasets(Path("data/processed"))
        >>> print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    """
    dataset_dir = Path(dataset_dir)

    if splits is None:
        # 자동 감지: arrow 형식 우선, 없으면 jsonl
        arrow_dirs = list(dataset_dir.glob("*_dataset"))
        jsonl_files = list(dataset_dir.glob("*.jsonl"))

        if format == "arrow" and arrow_dirs:
            splits = [d.stem.replace("_dataset", "") for d in arrow_dirs]
            # val 별칭을 validation으로 변환
            splits = ["validation" if s == "val" else s for s in splits]
        elif jsonl_files:
            splits = [f.stem for f in jsonl_files]
            # val 별칭을 validation으로 변환
            splits = ["validation" if s == "val" else s for s in splits]
        else:
            raise FileNotFoundError(
                f"데이터셋을 찾을 수 없습니다: {dataset_dir}"
            )

    datasets = []

    for split_name in splits:
        # val 별칭을 validation으로 변환
        actual_split_name = "validation" if split_name == "val" else split_name
        
        if format == "arrow":
            # Arrow 형식 로드 (validation 또는 val_dataset 모두 시도)
            arrow_path = dataset_dir / f"{actual_split_name}_dataset"
            if not arrow_path.exists() and split_name == "val":
                # val_dataset도 시도
                arrow_path = dataset_dir / "val_dataset"
            if not arrow_path.exists():
                print(f"[WARNING] {arrow_path}를 찾을 수 없습니다. 건너뜁니다.")
                continue

            print(f"[INFO] {split_name} Dataset 로드 중: {arrow_path}")
            dataset = Dataset.load_from_disk(str(arrow_path))
            print(f"[OK] {split_name} 로드 완료: {len(dataset)} 샘플")
            datasets.append(dataset)

        elif format == "jsonl":
            # JSONL 형식 로드 (validation.jsonl 또는 val.jsonl 모두 시도)
            jsonl_path = dataset_dir / f"{actual_split_name}.jsonl"
            if not jsonl_path.exists() and split_name == "val":
                # val.jsonl도 시도
                jsonl_path = dataset_dir / "val.jsonl"
            if not jsonl_path.exists():
                print(f"[WARNING] {jsonl_path}를 찾을 수 없습니다. 건너뜁니다.")
                continue

            print(f"[INFO] {split_name} Dataset 로드 중: {jsonl_path}")
            import json

            data_list = []
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            dataset = Dataset.from_list(data_list)
            print(f"[OK] {split_name} 로드 완료: {len(dataset)} 샘플")
            datasets.append(dataset)

        else:
            raise ValueError(f"지원하지 않는 형식: {format}")

    if not datasets:
        raise FileNotFoundError(f"로드된 데이터셋이 없습니다: {dataset_dir}")

    return tuple(datasets)


def prepare_for_sft_trainer(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    text_column: str = "text",
) -> Tuple[Dataset, Optional[Dataset]]:
    """SFTTrainer에서 바로 사용할 수 있도록 Dataset을 준비합니다.

    SFTTrainer는 "text" 필드를 가진 Dataset을 요구합니다.
    이 함수는 필요한 검증과 변환을 수행합니다.

    Args:
        train_dataset: 학습용 Dataset
        val_dataset: 검증용 Dataset (선택)
        text_column: 텍스트 컬럼명 (기본: "text")

    Returns:
        (준비된 train_dataset, 준비된 val_dataset) 튜플

    Example:
        >>> train_dataset = Dataset.from_list([{"text": "..."}])
        >>> val_dataset = Dataset.from_list([{"text": "..."}])
        >>> train, val = prepare_for_sft_trainer(train_dataset, val_dataset)
        >>> # SFTTrainer에 직접 전달 가능
        >>> trainer = SFTTrainer(..., train_dataset=train, eval_dataset=val)
    """
    # text_column 검증
    if text_column not in train_dataset.column_names:
        raise ValueError(
            f"train_dataset에 '{text_column}' 컬럼이 없습니다. "
            f"사용 가능한 컬럼: {train_dataset.column_names}"
        )

    if val_dataset is not None:
        if text_column not in val_dataset.column_names:
            raise ValueError(
                f"val_dataset에 '{text_column}' 컬럼이 없습니다. "
                f"사용 가능한 컬럼: {val_dataset.column_names}"
            )

    print(f"[OK] SFTTrainer 준비 완료:")
    print(f"  Train: {len(train_dataset)} 샘플")
    if val_dataset:
        print(f"  Validation: {len(val_dataset)} 샘플")

    return train_dataset, val_dataset


def get_dataset_info(dataset: Dataset) -> dict:
    """Dataset의 정보를 반환합니다.

    Args:
        dataset: 정보를 확인할 Dataset

    Returns:
        Dataset 정보 딕셔너리
    """
    info = {
        "num_samples": len(dataset),
        "column_names": dataset.column_names,
        "features": str(dataset.features),
    }

    # text 컬럼이 있으면 샘플 길이 통계 추가
    if "text" in dataset.column_names:
        text_lengths = [len(item.get("text", "")) for item in dataset]
        if text_lengths:
            import numpy as np

            info["text_length_stats"] = {
                "min": int(np.min(text_lengths)),
                "max": int(np.max(text_lengths)),
                "mean": float(np.mean(text_lengths)),
                "median": int(np.median(text_lengths)),
            }

    return info
