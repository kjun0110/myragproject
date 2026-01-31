#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PGVector 모니터링 스크립트(개발/운영 도구).

사용 예:
  python api/scripts/monitor_pgvector.py
"""

import sys
import time
from pathlib import Path


def _bootstrap_import_path() -> None:
    # 이 파일: api/scripts/monitor_pgvector.py
    current_file = Path(__file__).resolve()
    api_dir = current_file.parent.parent  # api/
    sys.path.insert(0, str(api_dir))


def main() -> None:
    _bootstrap_import_path()

    from app.core.store.pgvector_store import (
        create_embeddings,
        get_vector_store_config,
        get_pgvector_store,
        wait_for_postgres,
    )

    cfg = get_vector_store_config()
    print("[INFO] PGVector monitor 시작")
    print(f"[INFO] collection_name={cfg.collection_name}")

    print("[INFO] PostgreSQL 연결 대기...")
    wait_for_postgres(cfg.connection_string)
    print("[OK] PostgreSQL 연결 확인 완료")

    embeddings = create_embeddings(prefer_openai=True)
    store = get_pgvector_store(embeddings=embeddings, config=cfg)

    interval = 30
    test_queries = ["LangChain", "pgvector", "Hello", "프레임워크"]

    while True:
        time.sleep(interval)
        print("-" * 60)
        print(f"[INFO] 조회 (interval={interval}s)")

        for q in test_queries:
            try:
                results = store.similarity_search(q, k=3)
                print(f"  - q='{q}' -> {len(results)} results")
            except Exception as e:
                print(f"  - q='{q}' -> ERROR: {e}")


if __name__ == "__main__":
    main()

