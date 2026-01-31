#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PGVector 컬렉션 초기화/시딩 스크립트(개발 도구).

사용 예:
  python api/scripts/init_pgvector.py
"""

import sys
from pathlib import Path

from langchain_core.documents import Document


def _bootstrap_import_path() -> None:
    # 이 파일: api/scripts/init_pgvector.py
    current_file = Path(__file__).resolve()
    api_dir = current_file.parent.parent  # api/
    sys.path.insert(0, str(api_dir))


def main() -> None:
    _bootstrap_import_path()

    from app.core.store.pgvector_store import (
        create_embeddings,
        get_vector_store_config,
        upsert_documents,
        wait_for_postgres,
    )

    cfg = get_vector_store_config()
    print("[INFO] PGVector init 시작")
    print(f"[INFO] collection_name={cfg.collection_name}")

    print("[INFO] PostgreSQL 연결 대기...")
    wait_for_postgres(cfg.connection_string)
    print("[OK] PostgreSQL 연결 확인 완료")

    embeddings = create_embeddings(prefer_openai=True)
    documents = [
        Document(
            page_content="LangChain은 LLM 애플리케이션 개발을 위한 프레임워크입니다.",
            metadata={"source": "intro"},
        ),
        Document(
            page_content="pgvector는 PostgreSQL에서 벡터 검색을 가능하게 하는 확장입니다.",
            metadata={"source": "pgvector"},
        ),
        Document(
            page_content="Hello World는 프로그래밍의 첫 번째 예제입니다.",
            metadata={"source": "hello"},
        ),
    ]

    upsert_documents(embeddings=embeddings, documents=documents, config=cfg)
    print("[OK] 문서 시딩 완료")


if __name__ == "__main__":
    main()

