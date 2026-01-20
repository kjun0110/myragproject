#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""데이터 분할 모듈.

- Train/Validation 분할
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


class DataSplitter:
    """데이터 분할 클래스."""

    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42,
        stratify: bool = True,
    ):
        """초기화.

        Args:
            train_ratio: 학습 데이터 비율 (기본 0.8)
            val_ratio: 검증 데이터 비율 (기본 0.1)
            test_ratio: 테스트 데이터 비율 (기본 0.1)
            random_state: 랜덤 시드
            stratify: 계층화 분할 사용 여부 (클래스 비율 유지)
        """
        # 비율 검증
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"비율의 합이 1.0이어야 합니다. 현재: {total}"
            )

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.stratify = stratify

    def _extract_label(self, item: Dict) -> str:
        """아이템에서 레이블을 추출합니다.

        Args:
            item: 데이터 아이템

        Returns:
            레이블 문자열 ("BLOCK" 또는 "ALLOW")
        """
        # SFT 형식 데이터인 경우
        if "text" in item:
            # text에서 action 추출 시도
            text = item["text"]
            if '"action": "BLOCK"' in text or "'action': 'BLOCK'" in text:
                return "BLOCK"
            elif '"action": "ALLOW"' in text or "'action': 'ALLOW'" in text:
                return "ALLOW"

        # 원본 형식 데이터인 경우
        if "output" in item:
            action = item["output"].get("action", "").upper()
            if action in ["BLOCK", "ALLOW"]:
                return action

        # 기본값
        return "UNKNOWN"

    def split_dataset(
        self, dataset: Dataset, stratify_column: Optional[str] = None
    ) -> DatasetDict:
        """데이터셋을 Train/Validation/Test로 분할합니다.

        Args:
            dataset: 분할할 데이터셋
            stratify_column: 계층화에 사용할 컬럼명 (None이면 자동 추출)

        Returns:
            분할된 데이터셋 딕셔너리
        """
        print(f"[INFO] 데이터셋 분할 중 (Train: {self.train_ratio}, Val: {self.val_ratio}, Test: {self.test_ratio})...")

        # 데이터셋을 리스트로 변환
        data_list = [item for item in dataset]

        # 계층화 레이블 추출
        labels = None
        if self.stratify:
            if stratify_column and stratify_column in dataset.column_names:
                # 지정된 컬럼 사용
                labels = [item[stratify_column] for item in data_list]
            else:
                # 자동으로 레이블 추출
                labels = [self._extract_label(item) for item in data_list]

            # 레이블 분포 확인
            from collections import Counter
            label_counts = Counter(labels)
            print(f"[INFO] 레이블 분포: {dict(label_counts)}")

            # UNKNOWN이 너무 많으면 계층화 비활성화
            unknown_ratio = label_counts.get("UNKNOWN", 0) / len(labels)
            if unknown_ratio > 0.5:
                print(
                    f"[WARNING] UNKNOWN 레이블 비율이 높습니다 ({unknown_ratio:.2%}). "
                    "계층화를 비활성화합니다."
                )
                labels = None
                self.stratify = False

        # 1단계: Train + (Val + Test) 분할
        train_size = self.train_ratio
        val_test_size = self.val_ratio + self.test_ratio

        train_data, val_test_data = train_test_split(
            data_list,
            test_size=val_test_size,
            random_state=self.random_state,
            stratify=labels if self.stratify else None,
        )

        # 2단계: Val + Test 분할
        val_size_in_val_test = self.val_ratio / val_test_size

        val_test_labels = None
        if self.stratify and labels:
            # val_test에 해당하는 레이블 추출
            val_test_indices = set(range(len(data_list))) - set(
                range(len(train_data))
            )
            val_test_labels = [labels[i] for i in val_test_indices]

        val_data, test_data = train_test_split(
            val_test_data,
            test_size=1 - val_size_in_val_test,
            random_state=self.random_state,
            stratify=val_test_labels if self.stratify else None,
        )

        # DatasetDict 생성
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        test_dataset = Dataset.from_list(test_data)

        dataset_dict = DatasetDict(
            {
                "train": train_dataset,
                "validation": val_dataset,
                "test": test_dataset,
            }
        )

        print(f"[OK] 데이터셋 분할 완료:")
        print(f"  Train: {len(train_dataset)} 샘플 ({len(train_dataset)/len(dataset)*100:.2f}%)")
        print(f"  Validation: {len(val_dataset)} 샘플 ({len(val_dataset)/len(dataset)*100:.2f}%)")
        print(f"  Test: {len(test_dataset)} 샘플 ({len(test_dataset)/len(dataset)*100:.2f}%)")

        return dataset_dict

    def save_splits(
        self,
        dataset_dict: DatasetDict,
        output_dir: Path,
        format: str = "jsonl",
    ) -> None:
        """분할된 데이터셋을 파일로 저장합니다.

        Args:
            dataset_dict: 분할된 데이터셋 딕셔너리
            output_dir: 출력 디렉토리
            format: 저장 형식 ("jsonl" 또는 "arrow")
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            import json

            for split_name, dataset in dataset_dict.items():
                output_path = output_dir / f"{split_name}.jsonl"
                print(f"[INFO] {split_name} 데이터 저장 중: {output_path}")

                with output_path.open("w", encoding="utf-8") as f:
                    for item in dataset:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

                print(f"[OK] {split_name} 저장 완료: {len(dataset)} 샘플")

        elif format == "arrow":
            for split_name, dataset in dataset_dict.items():
                output_path = output_dir / f"{split_name}"
                print(f"[INFO] {split_name} 데이터 저장 중: {output_path}")
                dataset.save_to_disk(str(output_path))
                print(f"[OK] {split_name} 저장 완료: {len(dataset)} 샘플")

        else:
            raise ValueError(f"지원하지 않는 형식: {format}")

    def load_splits(
        self, input_dir: Path, format: str = "jsonl"
    ) -> DatasetDict:
        """저장된 분할 데이터셋을 로드합니다.

        Args:
            input_dir: 입력 디렉토리
            format: 로드 형식 ("jsonl" 또는 "arrow")

        Returns:
            분할된 데이터셋 딕셔너리
        """
        if format == "jsonl":
            import json

            dataset_dict = {}
            for split_name in ["train", "validation", "test"]:
                input_path = input_dir / f"{split_name}.jsonl"
                if not input_path.exists():
                    print(f"[WARNING] {split_name}.jsonl 파일을 찾을 수 없습니다.")
                    continue

                print(f"[INFO] {split_name} 데이터 로드 중: {input_path}")
                data_list = []
                with input_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data_list.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

                dataset_dict[split_name] = Dataset.from_list(data_list)
                print(f"[OK] {split_name} 로드 완료: {len(dataset_dict[split_name])} 샘플")

            return DatasetDict(dataset_dict)

        elif format == "arrow":
            dataset_dict = {}
            for split_name in ["train", "validation", "test"]:
                input_path = input_dir / split_name
                if not input_path.exists():
                    print(f"[WARNING] {split_name} 디렉토리를 찾을 수 없습니다.")
                    continue

                print(f"[INFO] {split_name} 데이터 로드 중: {input_path}")
                dataset_dict[split_name] = Dataset.load_from_disk(str(input_path))
                print(f"[OK] {split_name} 로드 완료: {len(dataset_dict[split_name])} 샘플")

            return DatasetDict(dataset_dict)

        else:
            raise ValueError(f"지원하지 않는 형식: {format}")
