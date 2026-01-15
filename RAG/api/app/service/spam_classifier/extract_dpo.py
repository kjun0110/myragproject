#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JSONL 파일을 SFT(Supervised Fine-Tuning) 형식으로 변환하는 모듈."""

import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """JSONL을 한 줄씩 읽습니다. 깨진 라인은 건너뜁니다.

    Args:
        path: 읽을 JSONL 파일 경로

    Yields:
        각 행의 JSON 객체 딕셔너리
    """
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # 운영에서는 별도 로그 파일로 남기는 것이 좋습니다.
                continue


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """JSONL 파일로 저장합니다.

    Args:
        path: 저장할 파일 경로
        rows: 저장할 딕셔너리들의 반복 가능 객체
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_attachments(raw: str) -> List[str]:
    """첨부파일 문자열에서 파일명 목록을 추출합니다.

    "Offer.docx (16.4 K), Offer - contextual advertising.docx (15.8 K)"
    -> ["Offer.docx", "Offer - contextual advertising.docx"]

    Args:
        raw: 첨부파일 원본 문자열

    Returns:
        파일명 목록
    """
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    names: List[str] = []
    for p in parts:
        # 뒤의 "(16.4 K)" 같은 용량 표기를 제거
        p = re.sub(r"\s*\([^)]*\)\s*$", "", p).strip()
        if p:
            names.append(p)
    return names


def normalize_raw(row: Dict[str, Any]) -> Dict[str, Any]:
    """raw 한 줄을 정규화한 clean 레코드로 변환합니다.

    Args:
        row: 원본 JSONL 행 데이터

    Returns:
        정규화된 딕셔너리
    """
    # BOM 문자 제거를 위해 키를 정리
    cleaned_row = {}
    for key, value in row.items():
        # BOM 문자 제거
        clean_key = key.lstrip("\ufeff").strip()
        cleaned_row[clean_key] = value

    date = (cleaned_row.get("수신일자") or "").strip()
    time = (cleaned_row.get("수신시간") or "").strip()
    subject = (cleaned_row.get("제목") or "").strip()
    mail_type = (cleaned_row.get("메일 종류") or "").strip()
    attach_raw = (cleaned_row.get("첨부") or "").strip()
    if attach_raw is None:
        attach_raw = ""
    attachments = parse_attachments(attach_raw)

    received_at = f"{date} {time}".strip()
    return {
        "received_date": date,
        "received_time": time,
        "received_at": received_at,
        "subject": subject,
        "attachments": attachments,
        "mail_type": mail_type,
        # 원본 보존용으로도 남겨두면 유용합니다.
        "attachments_raw": attach_raw,
    }


def dedup_key(clean: Dict[str, Any], mode: str = "subject+attachments") -> Tuple:
    """중복 제거 기준을 선택합니다.

    Args:
        clean: 정규화된 데이터 딕셔너리
        mode: 중복 제거 모드
            - "subject+attachments": 제목과 첨부가 같으면 중복
            - "datetime+subject+attachments": 시간까지 포함(더 엄격)

    Returns:
        중복 체크용 튜플 키
    """
    if mode == "datetime+subject+attachments":
        return (
            clean.get("received_at", ""),
            clean.get("subject", ""),
            tuple(clean.get("attachments", [])),
        )
    return (clean.get("subject", ""), tuple(clean.get("attachments", [])))


def rule_label(clean: Dict[str, Any]) -> Tuple[str, str, float]:
    """rule-based labeling을 수행합니다.

    Args:
        clean: 정규화된 데이터 딕셔너리

    Returns:
        (action, reason, confidence) 튜플
        - action: BLOCK/ALLOW
        - reason: 짧은 근거
        - confidence: 0~1 사이의 신뢰도
    """
    subject = (clean.get("subject") or "").lower()
    attachments = clean.get("attachments") or []
    mail_type = (clean.get("mail_type") or "").strip()

    # 1) 원천 데이터에 '스팸' 라벨이 있으면 기본적으로 BLOCK
    if mail_type == "스팸":
        base_action = "BLOCK"
    elif mail_type == "정상":
        base_action = "ALLOW"
    else:
        # 라벨이 없으면 rule로 추정(초기엔 보수적으로)
        base_action = "BLOCK"

    # 2) reason / confidence 규칙
    reasons = []
    score = 0.0

    if "(광고)" in (clean.get("subject") or ""):
        reasons.append("제목에 (광고) 표기가 포함됨")
        score += 0.5

    # 흔한 스팸 키워드 예시(필요시 확장)
    spam_keywords = [
        "offer",
        "보험",
        "임플란트",
        "치아보험",
        "이벤트",
        "할인",
        "진단금",
        "간병",
    ]
    if any(k.lower() in subject for k in [kw.lower() for kw in spam_keywords]):
        reasons.append("스팸/광고성 키워드 패턴이 포함됨")
        score += 0.35

    if attachments:
        reasons.append("첨부파일이 포함됨")
        score += 0.2

    if not reasons:
        reasons.append("메타데이터만으로는 뚜렷한 단서가 적음")
        score += 0.1

    # confidence는 0.85~0.99 사이로 클램프(초기 모델 학습 안정 목적)
    confidence = min(0.99, max(0.85, 0.80 + score))

    # action은 base_action을 따르되, 이유가 약하면 confidence만 낮게
    action = base_action
    reason = " / ".join(reasons)
    return action, reason, float(f"{confidence:.2f}")


def to_sft(clean: Dict[str, Any]) -> Dict[str, Any]:
    """정규화된 데이터를 SFT 형식으로 변환합니다.

    Args:
        clean: 정규화된 데이터 딕셔너리

    Returns:
        SFT 형식의 딕셔너리
    """
    action, reason, confidence = rule_label(clean)
    return {
        "instruction": "다음 이메일 메타데이터를 분석하여 스팸 여부를 판정하고 JSON 형식으로만 답하세요.",
        "input": {
            "subject": clean.get("subject", ""),
            "attachments": clean.get("attachments", []),
            "received_at": clean.get("received_at", ""),
        },
        "output": {
            "action": action,
            "reason": reason,
            "confidence": confidence,
        },
    }


def to_koelectra(clean: Dict[str, Any]) -> Dict[str, Any]:
    """정규화된 데이터를 KoElectra 분류 학습 형식으로 변환합니다.

    Args:
        clean: 정규화된 데이터 딕셔너리

    Returns:
        KoElectra 형식의 딕셔너리 (text와 label 필드 포함)
    """
    action, reason, confidence = rule_label(clean)

    # 입력 텍스트 구성
    subject = clean.get("subject", "")
    attachments = clean.get("attachments", [])
    received_at = clean.get("received_at", "")

    # 첨부파일 문자열 생성
    if attachments:
        attachments_str = ", ".join(attachments)
    else:
        attachments_str = "없음"

    # 입력 부분만 추출 (KoElectra 학습용)
    text = f"제목: {subject}\n첨부파일: {attachments_str}\n수신시간: {received_at}"

    # 레이블 생성 (BLOCK이면 1, 아니면 0)
    label = 1 if action == "BLOCK" else 0

    return {
        "text": text,
        "label": label,
    }


def convert_jsonl_to_sft(
    input_jsonl_path: Path,
    output_sft_path: Path,
    output_dedup_path: Optional[Path] = None,
    output_clean_path: Optional[Path] = None,
    dedup_mode: str = "subject+attachments",
) -> Tuple[int, int, int]:
    """JSONL 파일을 SFT 형식으로 변환합니다.

    Args:
        input_jsonl_path: 입력 JSONL 파일 경로
        output_sft_path: 출력 SFT JSONL 파일 경로
        output_dedup_path: (선택) 중복 제거된 JSONL 파일 경로
        output_clean_path: (선택) 정규화된 JSONL 파일 경로
        dedup_mode: 중복 제거 모드

    Returns:
        (sft_count, dedup_count, clean_count) 튜플
    """
    seen = set()
    dedup_rows = []
    clean_rows = []
    sft_rows = []

    for row in iter_jsonl(input_jsonl_path):
        clean = normalize_raw(row)
        key = dedup_key(clean, mode=dedup_mode)
        if key in seen:
            continue
        seen.add(key)

        # dedup 단계 산출물(원본 형태 유지가 필요하면 row를, 정규화 형태면 clean을 저장)
        if output_dedup_path is not None:
            dedup_rows.append(row)

        # clean 단계 산출물
        if output_clean_path is not None:
            clean_rows.append(clean)

        # sft 산출물
        sft_rows.append(to_sft(clean))

    # 저장
    write_jsonl(output_sft_path, sft_rows)
    if output_dedup_path is not None:
        write_jsonl(output_dedup_path, dedup_rows)
    if output_clean_path is not None:
        write_jsonl(output_clean_path, clean_rows)

    return len(sft_rows), len(dedup_rows), len(clean_rows)


def convert_jsonl_to_koelectra(
    input_jsonl_path: Path,
    output_koelectra_path: Path,
    output_dedup_path: Optional[Path] = None,
    output_clean_path: Optional[Path] = None,
    dedup_mode: str = "subject+attachments",
) -> Tuple[int, int, int]:
    """JSONL 파일을 KoElectra 분류 학습 형식으로 변환합니다.

    Args:
        input_jsonl_path: 입력 JSONL 파일 경로
        output_koelectra_path: 출력 KoElectra JSONL 파일 경로
        output_dedup_path: (선택) 중복 제거된 JSONL 파일 경로
        output_clean_path: (선택) 정규화된 JSONL 파일 경로
        dedup_mode: 중복 제거 모드

    Returns:
        (koelectra_count, dedup_count, clean_count) 튜플
    """
    seen = set()
    dedup_rows = []
    clean_rows = []
    koelectra_rows = []

    for row in iter_jsonl(input_jsonl_path):
        clean = normalize_raw(row)
        key = dedup_key(clean, mode=dedup_mode)
        if key in seen:
            continue
        seen.add(key)

        # dedup 단계 산출물(원본 형태 유지가 필요하면 row를, 정규화 형태면 clean을 저장)
        if output_dedup_path is not None:
            dedup_rows.append(row)

        # clean 단계 산출물
        if output_clean_path is not None:
            clean_rows.append(clean)

        # koelectra 산출물
        koelectra_rows.append(to_koelectra(clean))

    # 저장
    write_jsonl(output_koelectra_path, koelectra_rows)
    if output_dedup_path is not None:
        write_jsonl(output_dedup_path, dedup_rows)
    if output_clean_path is not None:
        write_jsonl(output_clean_path, clean_rows)

    return len(koelectra_rows), len(dedup_rows), len(clean_rows)


def split_dataset(
    input_jsonl_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Tuple[int, int, int]:
    """JSONL 파일을 train, validation, test로 분할합니다.

    Args:
        input_jsonl_path: 입력 JSONL 파일 경로
        output_dir: 출력 디렉토리
        train_ratio: 학습 데이터 비율 (기본값: 0.8)
        validation_ratio: 검증 데이터 비율 (기본값: 0.1)
        test_ratio: 테스트 데이터 비율 (기본값: 0.1)
        random_seed: 랜덤 시드

    Returns:
        (train_count, validation_count, test_count) 튜플
    """
    # 비율 합이 1인지 확인
    total_ratio = train_ratio + validation_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"비율의 합이 1이어야 합니다. 현재: {total_ratio}")

    # 데이터 로드
    all_data = list(iter_jsonl(input_jsonl_path))

    if not all_data:
        print(f"[WARNING] {input_jsonl_path}에 데이터가 없습니다.")
        return 0, 0, 0

    # 랜덤 셔플
    random.seed(random_seed)
    random.shuffle(all_data)

    total = len(all_data)
    train_end = int(total * train_ratio)
    validation_end = train_end + int(total * validation_ratio)

    train_data = all_data[:train_end]
    validation_data = all_data[train_end:validation_end]
    test_data = all_data[validation_end:]

    # 저장
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    test_path = output_dir / "test.jsonl"

    write_jsonl(train_path, train_data)
    write_jsonl(validation_path, validation_data)
    write_jsonl(test_path, test_data)

    print("[INFO] 데이터셋 분할 완료:")
    print(f"  - Train: {len(train_data)} 샘플 ({len(train_data) / total * 100:.1f}%)")
    print(
        f"  - Validation: {len(validation_data)} 샘플 ({len(validation_data) / total * 100:.1f}%)"
    )
    print(f"  - Test: {len(test_data)} 샘플 ({len(test_data) / total * 100:.1f}%)")

    return len(train_data), len(validation_data), len(test_data)


if __name__ == "__main__":
    # data 디렉토리 경로
    current_file = Path(__file__)
    spam_classifier_dir = current_file.parent  # api/app/service/spam_classifier/
    service_dir = spam_classifier_dir.parent  # api/app/service/
    app_dir = service_dir.parent  # api/app/
    data_dir = app_dir / "data"  # api/app/data

    if not data_dir.exists():
        print(f"오류: data 디렉토리를 찾을 수 없습니다: {data_dir}")
        sys.exit(1)

    # JSONL 파일 찾기 (data 디렉토리와 하위 디렉토리 모두 재귀적으로 검색)
    jsonl_files = list(data_dir.glob("**/*.jsonl"))

    # 제외할 파일들
    spam_agent_dir = data_dir / "spam_agent_processed"
    koelectra_dir = spam_agent_dir / "koelectra"
    exaone_dir = spam_agent_dir / "exaone"

    # 제외할 파일 목록
    exclude_files = set()

    # koelectra 폴더의 JSONL은 제외 (출력 파일이므로)
    if koelectra_dir.exists():
        exclude_files.update(koelectra_dir.glob("*.jsonl"))

    # exaone 폴더의 train.jsonl, validation.jsonl, test.jsonl, *.sft.jsonl은 제외
    if exaone_dir.exists():
        exclude_files.update(exaone_dir.glob("train.jsonl"))
        exclude_files.update(exaone_dir.glob("validation.jsonl"))
        exclude_files.update(exaone_dir.glob("test.jsonl"))
        exclude_files.update(exaone_dir.glob("*.sft.jsonl"))

    jsonl_files = [f for f in jsonl_files if f not in exclude_files]

    if not jsonl_files:
        print(f"오류: '{data_dir}' 디렉토리에서 JSONL 파일을 찾을 수 없습니다.")
        sys.exit(1)

    # KoElectra 출력 디렉토리 설정
    koelectra_output_dir = app_dir / "data" / "spam_agent_processed" / "koelectra"
    koelectra_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] KoElectra 출력 디렉토리: {koelectra_output_dir}")

    # 모든 JSONL 파일 변환
    for jsonl_file in jsonl_files:
        base_name = jsonl_file.stem  # 확장자 제외한 파일명

        # .sft.jsonl 파일은 .sft를 제거
        if base_name.endswith(".sft"):
            base_name = base_name[:-4]  # .sft 제거

        koelectra_file = koelectra_output_dir / f"{base_name}.jsonl"
        print(f"\n[INFO] KoElectra 변환 중: {jsonl_file.name} -> {koelectra_file.name}")

        try:
            koelectra_count, _, _ = convert_jsonl_to_koelectra(
                input_jsonl_path=jsonl_file,
                output_koelectra_path=koelectra_file,
                dedup_mode="subject+attachments",
            )
            print(
                f"[OK] KoElectra 변환 완료: {koelectra_file.name} (samples={koelectra_count})"
            )

            # 변환된 파일을 train/validation/test로 분할
            print(f"\n[INFO] 데이터셋 분할 중: {koelectra_file.name}")
            try:
                train_count, val_count, test_count = split_dataset(
                    input_jsonl_path=koelectra_file,
                    output_dir=koelectra_output_dir,
                    train_ratio=0.8,
                    validation_ratio=0.1,
                    test_ratio=0.1,
                    random_seed=42,
                )
                print(
                    f"[OK] 데이터셋 분할 완료: Train={train_count}, Val={val_count}, Test={test_count}"
                )
            except Exception as e:
                print(f"[WARNING] 데이터셋 분할 실패: {e}")
                import traceback

                print(f"[WARNING] 상세 오류:\n{traceback.format_exc()}")

        except Exception as e:
            print(f"[ERROR] {jsonl_file.name} KoElectra 변환 실패: {e}")
            import traceback

            print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
            continue

    print("\n모든 변환 작업이 완료되었습니다.")
