#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""중복 파일 정리 스크립트.

기존에 생성된 val.jsonl, val_dataset/ 등 중복 파일을 제거합니다.
"""

from pathlib import Path
import shutil


def cleanup_duplicates(data_dir: Path):
    """중복 파일을 정리합니다.

    Args:
        data_dir: spam_agent_processed 디렉토리 경로
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"[ERROR] 디렉토리를 찾을 수 없습니다: {data_dir}")
        return

    print("=" * 60)
    print("중복 파일 정리")
    print("=" * 60)

    removed = []

    # val.jsonl 제거 (validation.jsonl과 중복)
    val_jsonl = data_dir / "val.jsonl"
    if val_jsonl.exists():
        val_jsonl.unlink()
        removed.append("val.jsonl")
        print(f"[OK] 제거: {val_jsonl}")

    # val_dataset/ 제거 (validation_dataset/과 중복)
    val_dataset = data_dir / "val_dataset"
    if val_dataset.exists():
        shutil.rmtree(val_dataset)
        removed.append("val_dataset/")
        print(f"[OK] 제거: {val_dataset}")

    if removed:
        print(f"\n[OK] {len(removed)}개의 중복 파일/디렉토리를 제거했습니다.")
    else:
        print("\n[INFO] 중복 파일이 없습니다.")

    # 남은 파일 목록
    print("\n[INFO] 남은 파일:")
    for item in sorted(data_dir.iterdir()):
        if item.is_dir():
            print(f"  📁 {item.name}/")
        else:
            print(f"  📄 {item.name}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # 경로 설정
    current_file = Path(__file__).resolve()
    spam_agent_dir = current_file.parent
    service_dir = spam_agent_dir.parent
    app_dir = service_dir.parent
    
    data_dir = app_dir / "data" / "spam_agent_processed"
    
    cleanup_duplicates(data_dir)
