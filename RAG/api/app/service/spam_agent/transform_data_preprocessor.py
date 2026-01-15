#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SFT 학습용 데이터 전처리 모듈.

- SFT 학습용 텍스트 포맷 변환
- 데이터 품질 검증 및 정제
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset


class DataPreprocessor:
    """SFT 학습용 데이터 전처리 클래스."""

    def __init__(self, max_subject_length: int = 200, min_confidence: float = 0.85):
        """초기화.

        Args:
            max_subject_length: 제목 최대 길이 (초과 시 자름)
            min_confidence: 최소 confidence 값 (이하 제거)
        """
        self.max_subject_length = max_subject_length
        self.min_confidence = min_confidence

    def load_jsonl(self, jsonl_path: Path) -> List[Dict[str, Any]]:
        """JSONL 파일을 로드합니다.

        Args:
            jsonl_path: JSONL 파일 경로

        Returns:
            데이터 리스트
        """
        data = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    # BOM 문자 제거
                    if line.startswith("\ufeff"):
                        line = line[1:]
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[WARNING] 라인 {lineno} 파싱 실패: {e}")
                    continue
        return data

    def validate_data_structure(self, data: List[Dict[str, Any]]) -> Tuple[int, int]:
        """데이터 구조를 검증합니다.

        Args:
            data: 검증할 데이터 리스트

        Returns:
            (유효한 샘플 수, 무효한 샘플 수) 튜플
        """
        valid_count = 0
        invalid_count = 0

        required_keys = ["instruction", "input", "output"]
        required_input_keys = ["subject", "attachments", "received_at"]
        required_output_keys = ["action", "reason", "confidence"]

        for item in data:
            # 필수 키 확인
            if not all(key in item for key in required_keys):
                invalid_count += 1
                continue

            # input 필수 키 확인
            if not isinstance(item.get("input"), dict):
                invalid_count += 1
                continue

            if not all(key in item["input"] for key in required_input_keys):
                invalid_count += 1
                continue

            # output 필수 키 확인
            if not isinstance(item.get("output"), dict):
                invalid_count += 1
                continue

            if not all(key in item["output"] for key in required_output_keys):
                invalid_count += 1
                continue

            valid_count += 1

        return valid_count, invalid_count

    def clean_text(self, text: str) -> str:
        """텍스트를 정제합니다.

        Args:
            text: 정제할 텍스트

        Returns:
            정제된 텍스트
        """
        if not text:
            return ""

        # BOM 문자 제거
        text = text.lstrip("\ufeff")

        # 제어 문자 제거 (탭, 개행은 유지)
        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", text)

        # 연속된 공백 정리 (단, 개행은 유지)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"[ \t]+\n", "\n", text)

        # 앞뒤 공백 제거
        text = text.strip()

        return text

    def clean_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """데이터를 정제합니다.

        Args:
            data: 정제할 데이터 리스트

        Returns:
            정제된 데이터 리스트
        """
        cleaned = []

        for item in data:
            try:
                # instruction 정제
                instruction = self.clean_text(item.get("instruction", ""))

                # input 정제
                input_data = item.get("input", {})
                subject = self.clean_text(input_data.get("subject", ""))
                received_at = self.clean_text(input_data.get("received_at", ""))

                # 제목 길이 제한
                if len(subject) > self.max_subject_length:
                    subject = subject[: self.max_subject_length] + "..."

                # attachments 정제
                attachments = input_data.get("attachments", [])
                if isinstance(attachments, str):
                    attachments = [a.strip() for a in attachments.split(",") if a.strip()]
                elif isinstance(attachments, list):
                    attachments = [self.clean_text(str(a)) for a in attachments if a]

                # output 정제
                output_data = item.get("output", {})
                action = output_data.get("action", "").strip().upper()
                reason = self.clean_text(output_data.get("reason", ""))
                confidence = output_data.get("confidence", 0.0)

                # confidence 검증
                try:
                    confidence = float(confidence)
                    if confidence < self.min_confidence:
                        continue  # 최소 confidence 미만 제거
                except (ValueError, TypeError):
                    continue  # 유효하지 않은 confidence 제거

                # action 검증
                if action not in ["BLOCK", "ALLOW"]:
                    continue  # 유효하지 않은 action 제거

                # 빈 필드 검증
                if not subject or not instruction:
                    continue  # 필수 필드가 비어있으면 제거

                cleaned.append(
                    {
                        "instruction": instruction,
                        "input": {
                            "subject": subject,
                            "attachments": attachments,
                            "received_at": received_at,
                        },
                        "output": {
                            "action": action,
                            "reason": reason,
                            "confidence": confidence,
                        },
                    }
                )
            except Exception as e:
                print(f"[WARNING] 데이터 정제 중 오류 (건너뜀): {e}")
                continue

        return cleaned

    def format_for_sft(self, item: Dict[str, Any]) -> str:
        """SFT 학습용 텍스트 포맷으로 변환합니다.

        Args:
            item: 데이터 아이템

        Returns:
            SFT 형식의 텍스트
        """
        instruction = item["instruction"]
        input_data = item["input"]
        output_data = item["output"]

        # 입력 텍스트 구성
        subject = input_data["subject"]
        attachments = input_data.get("attachments", [])
        received_at = input_data.get("received_at", "")

        attachments_str = ", ".join(attachments) if attachments else "없음"

        input_text = f"제목: {subject}\n첨부파일: {attachments_str}\n수신시간: {received_at}"

        # 출력 JSON 구성
        output_json = json.dumps(
            {
                "action": output_data["action"],
                "reason": output_data["reason"],
                "confidence": output_data["confidence"],
            },
            ensure_ascii=False,
        )

        # 최종 SFT 형식 텍스트
        sft_text = f"### 지시문\n{instruction}\n\n### 입력\n{input_text}\n\n### 출력\n{output_json}\n"

        return sft_text

    def convert_to_sft_format(
        self, data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """데이터를 SFT 형식으로 변환합니다.

        Args:
            data: 변환할 데이터 리스트

        Returns:
            SFT 형식 데이터 리스트 ({"text": sft_text} 형식)
        """
        sft_data = []

        for item in data:
            try:
                sft_text = self.format_for_sft(item)
                sft_data.append({"text": sft_text})
            except Exception as e:
                print(f"[WARNING] SFT 변환 실패 (건너뜀): {e}")
                continue

        return sft_data

    def analyze_data_quality(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """데이터 품질을 분석합니다.

        Args:
            data: 분석할 데이터 리스트

        Returns:
            분석 결과 딕셔너리
        """
        if not data:
            return {
                "total_samples": 0,
                "block_count": 0,
                "allow_count": 0,
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "avg_subject_length": 0.0,
                "max_subject_length": 0,
                "min_subject_length": 0,
                "samples_with_attachments": 0,
            }

        block_count = 0
        allow_count = 0
        confidences = []
        subject_lengths = []
        samples_with_attachments = 0

        for item in data:
            output = item.get("output", {})
            action = output.get("action", "").upper()

            if action == "BLOCK":
                block_count += 1
            elif action == "ALLOW":
                allow_count += 1

            confidence = output.get("confidence", 0.0)
            if isinstance(confidence, (int, float)):
                confidences.append(float(confidence))

            subject = item.get("input", {}).get("subject", "")
            subject_lengths.append(len(subject))

            attachments = item.get("input", {}).get("attachments", [])
            if attachments and len(attachments) > 0:
                samples_with_attachments += 1

        total = len(data)

        return {
            "total_samples": total,
            "block_count": block_count,
            "allow_count": allow_count,
            "block_ratio": block_count / total if total > 0 else 0.0,
            "allow_ratio": allow_count / total if total > 0 else 0.0,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "min_confidence": min(confidences) if confidences else 0.0,
            "max_confidence": max(confidences) if confidences else 0.0,
            "avg_subject_length": sum(subject_lengths) / len(subject_lengths)
            if subject_lengths
            else 0.0,
            "max_subject_length": max(subject_lengths) if subject_lengths else 0,
            "min_subject_length": min(subject_lengths) if subject_lengths else 0,
            "samples_with_attachments": samples_with_attachments,
            "attachment_ratio": samples_with_attachments / total if total > 0 else 0.0,
        }

    def process(
        self, jsonl_path: Path, output_path: Optional[Path] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """전체 전처리 파이프라인을 실행합니다.

        Args:
            jsonl_path: 입력 JSONL 파일 경로
            output_path: 출력 JSONL 파일 경로 (None이면 저장하지 않음)

        Returns:
            (전처리된 데이터, 품질 분석 결과) 튜플
        """
        print(f"[INFO] JSONL 파일 로드 중: {jsonl_path}")
        data = self.load_jsonl(jsonl_path)
        print(f"[INFO] 로드된 샘플 수: {len(data)}")

        # 데이터 구조 검증
        print("[INFO] 데이터 구조 검증 중...")
        valid_count, invalid_count = self.validate_data_structure(data)
        print(f"[INFO] 유효한 샘플: {valid_count}, 무효한 샘플: {invalid_count}")

        if invalid_count > 0:
            print(f"[WARNING] {invalid_count}개의 무효한 샘플이 발견되었습니다.")

        # 데이터 정제
        print("[INFO] 데이터 정제 중...")
        cleaned_data = self.clean_data(data)
        print(f"[INFO] 정제 후 샘플 수: {len(cleaned_data)}")

        # 품질 분석
        print("[INFO] 데이터 품질 분석 중...")
        quality_report = self.analyze_data_quality(cleaned_data)
        print("\n[INFO] 데이터 품질 분석 결과:")
        for key, value in quality_report.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        # SFT 형식으로 변환
        print("[INFO] SFT 형식으로 변환 중...")
        sft_data = self.convert_to_sft_format(cleaned_data)
        print(f"[INFO] SFT 변환 완료: {len(sft_data)}개 샘플")

        # 저장
        if output_path:
            print(f"[INFO] 전처리된 데이터 저장 중: {output_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                for item in sft_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print("[OK] 저장 완료")

        return sft_data, quality_report

    def to_dataset(self, sft_data: List[Dict[str, Any]]) -> Dataset:
        """SFT 데이터를 HuggingFace Dataset으로 변환합니다.

        Args:
            sft_data: SFT 형식 데이터 리스트

        Returns:
            HuggingFace Dataset 객체
        """
        return Dataset.from_list(sft_data)
