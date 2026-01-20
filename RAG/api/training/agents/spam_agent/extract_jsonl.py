"""CSV 파일을 JSONL 형식으로 변환하는 스크립트."""

import csv
import json
from pathlib import Path
from typing import Any, Dict


def convert_csv_to_jsonl(csv_path: Path, jsonl_path: Path) -> None:
    """CSV 파일을 JSONL 형식으로 변환.

    Args:
        csv_path: 입력 CSV 파일 경로
        jsonl_path: 출력 JSONL 파일 경로
    """
    print(f"[INFO] CSV 파일 읽는 중: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as csv_file:
        # CSV 리더 생성
        csv_reader = csv.DictReader(csv_file)

        # JSONL 파일 쓰기
        with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
            row_count = 0
            for row in csv_reader:
                # 각 행을 JSON 객체로 변환
                json_obj: Dict[str, Any] = {}
                for key, value in row.items():
                    # 빈 값은 None으로 처리
                    json_obj[key.strip()] = value.strip() if value else None

                # JSONL 형식으로 쓰기 (한 줄에 하나의 JSON 객체)
                jsonl_file.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
                row_count += 1

                # 진행 상황 출력 (10,000줄마다)
                if row_count % 10000 == 0:
                    print(f"[INFO] 처리 중... {row_count:,}줄 완료")

    print(f"[OK] 변환 완료: {row_count:,}줄 → {jsonl_path}")


def find_csv_files(data_dir: Path) -> list[Path]:
    """data 디렉토리에서 모든 CSV 파일 찾기.

    Args:
        data_dir: data 디렉토리 경로

    Returns:
        CSV 파일 경로 리스트
    """
    csv_files = list(data_dir.glob("*.csv"))
    return csv_files


def main():
    """메인 함수: data 디렉토리의 모든 CSV 파일을 JSONL로 변환."""
    # 현재 파일의 위치 기준으로 data 디렉토리 찾기
    current_file = Path(__file__)
    service_dir = current_file.parent
    app_dir = service_dir.parent
    data_dir = app_dir / "data"

    if not data_dir.exists():
        print(f"[ERROR] data 디렉토리를 찾을 수 없습니다: {data_dir}")
        return

    # CSV 파일 찾기
    csv_files = find_csv_files(data_dir)

    if not csv_files:
        print(f"[WARNING] data 디렉토리에 CSV 파일이 없습니다: {data_dir}")
        return

    print(f"[INFO] {len(csv_files)}개의 CSV 파일을 찾았습니다.")

    # 각 CSV 파일을 JSONL로 변환
    for csv_file in csv_files:
        # 출력 파일명: 확장자만 .jsonl로 변경
        jsonl_file = csv_file.with_suffix(".jsonl")

        print(f"\n[INFO] 변환 시작: {csv_file.name} → {jsonl_file.name}")
        try:
            convert_csv_to_jsonl(csv_file, jsonl_file)
        except Exception as e:
            print(f"[ERROR] 변환 실패 ({csv_file.name}): {str(e)}")
            import traceback

            print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
