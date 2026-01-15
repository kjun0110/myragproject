#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""토크나이저 유틸리티 모듈.

- 토크나이징 준비 및 시퀀스 길이 관리
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer


class TokenizerUtils:
    """토크나이저 유틸리티 클래스."""

    def __init__(
        self,
        model_path: str,
        max_length: int = 512,
        trust_remote_code: bool = True,
    ):
        """초기화.

        Args:
            model_path: 모델 경로 (토크나이저 로드용)
            max_length: 최대 시퀀스 길이
            trust_remote_code: 원격 코드 신뢰 여부
        """
        self.model_path = model_path
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code
        self.tokenizer: Optional[AutoTokenizer] = None
        self._load_tokenizer()

    def _load_tokenizer(self) -> None:
        """토크나이저를 로드합니다."""
        print(f"[INFO] 토크나이저 로드 중: {self.model_path}")

        # 모델 경로 결정
        model_path = self._resolve_model_path()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code,
            )

            # pad_token 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            print("[OK] 토크나이저 로드 완료")
            print(f"[INFO] Vocab size: {self.tokenizer.vocab_size}")
            print(f"[INFO] Pad token: {self.tokenizer.pad_token}")
            print(f"[INFO] EOS token: {self.tokenizer.eos_token}")

        except Exception as e:
            print(f"[ERROR] 토크나이저 로드 실패: {e}")
            raise

    def _resolve_model_path(self) -> str:
        """모델 경로를 해석합니다.

        Returns:
            실제 모델 경로
        """
        # 절대 경로인 경우
        if Path(self.model_path).is_absolute():
            if Path(self.model_path).exists():
                return self.model_path

        # 상대 경로인 경우 - exaone3.5 디렉토리 확인
        current_dir = Path(__file__).parent.parent.parent / "model"
        exaone_dir = current_dir / "exaone3.5" / "exaone-2.4b"

        if exaone_dir.exists() and (exaone_dir / "tokenizer_config.json").exists():
            return str(exaone_dir)

        # HuggingFace 모델 ID인 경우
        if "/" in self.model_path:
            return self.model_path

        # 기본 경로
        return self.model_path

    def tokenize_text(self, text: str) -> Dict[str, Any]:
        """텍스트를 토크나이징합니다.

        Args:
            text: 토크나이징할 텍스트

        Returns:
            토크나이징 결과 (input_ids, attention_mask 등)
        """
        if self.tokenizer is None:
            raise RuntimeError("토크나이저가 로드되지 않았습니다.")

        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,  # 리스트 반환
        )

    def analyze_sequence_lengths(
        self, dataset: Dataset, text_column: str = "text"
    ) -> Dict[str, Any]:
        """데이터셋의 시퀀스 길이를 분석합니다.

        Args:
            dataset: 분석할 데이터셋
            text_column: 텍스트 컬럼명

        Returns:
            분석 결과 딕셔너리
        """
        if self.tokenizer is None:
            raise RuntimeError("토크나이저가 로드되지 않았습니다.")

        print("[INFO] 시퀀스 길이 분석 중...")

        lengths = []
        truncated_count = 0

        for item in dataset:
            text = item.get(text_column, "")
            if not text:
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            length = len(tokens)
            lengths.append(length)

            if length > self.max_length:
                truncated_count += 1

        if not lengths:
            return {
                "total_samples": 0,
                "min_length": 0,
                "max_length": 0,
                "mean_length": 0.0,
                "median_length": 0,
                "std_length": 0.0,
                "p50": 0,
                "p75": 0,
                "p90": 0,
                "p95": 0,
                "p99": 0,
                "truncated_count": 0,
                "truncated_ratio": 0.0,
            }

        lengths_array = np.array(lengths)

        return {
            "total_samples": len(lengths),
            "min_length": int(np.min(lengths_array)),
            "max_length": int(np.max(lengths_array)),
            "mean_length": float(np.mean(lengths_array)),
            "median_length": int(np.median(lengths_array)),
            "std_length": float(np.std(lengths_array)),
            "p50": int(np.percentile(lengths_array, 50)),
            "p75": int(np.percentile(lengths_array, 75)),
            "p90": int(np.percentile(lengths_array, 90)),
            "p95": int(np.percentile(lengths_array, 95)),
            "p99": int(np.percentile(lengths_array, 99)),
            "truncated_count": truncated_count,
            "truncated_ratio": truncated_count / len(lengths) if lengths else 0.0,
        }

    def prepare_dataset(
        self, dataset: Dataset, text_column: str = "text"
    ) -> Dataset:
        """데이터셋을 토크나이징하여 학습 준비를 합니다.

        Args:
            dataset: 토크나이징할 데이터셋
            text_column: 텍스트 컬럼명

        Returns:
            토크나이징된 데이터셋
        """
        if self.tokenizer is None:
            raise RuntimeError("토크나이저가 로드되지 않았습니다.")

        print(f"[INFO] 데이터셋 토크나이징 중 (max_length={self.max_length})...")

        def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, Any]:
            texts = examples[text_column]
            return self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors=None,
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="토크나이징",
        )

        print("[OK] 토크나이징 완료")
        return tokenized_dataset

    def get_optimal_max_length(
        self, dataset: Dataset, text_column: str = "text", percentile: float = 0.95
    ) -> int:
        """최적의 max_length를 계산합니다.

        Args:
            dataset: 분석할 데이터셋
            text_column: 텍스트 컬럼명
            percentile: 사용할 백분위수 (기본 95%)

        Returns:
            최적의 max_length
        """
        analysis = self.analyze_sequence_lengths(dataset, text_column)
        optimal_length = int(analysis[f"p{int(percentile * 100)}"])

        # 32의 배수로 반올림 (GPU 효율성)
        optimal_length = ((optimal_length + 31) // 32) * 32

        print(f"[INFO] 최적 max_length (P{int(percentile * 100)}): {optimal_length}")
        return optimal_length

    def print_length_statistics(self, dataset: Dataset, text_column: str = "text") -> None:
        """시퀀스 길이 통계를 출력합니다.

        Args:
            dataset: 분석할 데이터셋
            text_column: 텍스트 컬럼명
        """
        analysis = self.analyze_sequence_lengths(dataset, text_column)

        print("\n[INFO] 시퀀스 길이 통계:")
        print(f"  총 샘플 수: {analysis['total_samples']}")
        print(f"  최소 길이: {analysis['min_length']}")
        print(f"  최대 길이: {analysis['max_length']}")
        print(f"  평균 길이: {analysis['mean_length']:.2f}")
        print(f"  중앙값: {analysis['median_length']}")
        print(f"  표준편차: {analysis['std_length']:.2f}")
        print(f"  P50: {analysis['p50']}")
        print(f"  P75: {analysis['p75']}")
        print(f"  P90: {analysis['p90']}")
        print(f"  P95: {analysis['p95']}")
        print(f"  P99: {analysis['p99']}")
        print(f"  잘린 샘플 수: {analysis['truncated_count']}")
        print(f"  잘림 비율: {analysis['truncated_ratio']:.4f}")

        # 권장 max_length 제안
        if analysis["p95"] < self.max_length:
            print(f"\n[INFO] 현재 max_length({self.max_length})는 충분합니다.")
        else:
            print(
                f"\n[WARNING] 현재 max_length({self.max_length})가 부족할 수 있습니다."
            )
            print(f"  P95 기준 권장: {analysis['p95']}")
            print(f"  P99 기준 권장: {analysis['p99']}")
