"""(Deprecated) 레거시 호환용 vector_store 모듈.

이 파일은 과거의 import 경로(`app.core.store.vector_store`)를 깨지 않기 위해 유지됩니다.
실제 프로덕션용 API는 `app.core.store.pgvector_store`를 사용하세요.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document

from .pgvector_store import (
    VectorStoreConfig,
    create_embeddings,
    get_vector_store_config,
    similarity_search,
    upsert_documents,
    wait_for_postgres,
)


def _default_documents() -> List[Document]:
    return [
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


def initialize_vector_store(embeddings=None, documents: Optional[List[Document]] = None):
    """(Deprecated) 기본 문서를 넣어 컬렉션을 초기화/생성합니다.

    - 기존 코드 호환을 위해 남겨둔 함수입니다.
    - 신규 코드는 `upsert_documents`를 직접 사용하세요.
    """
    if embeddings is None:
        embeddings = create_embeddings(prefer_openai=True)
    docs = documents or _default_documents()
    return upsert_documents(embeddings=embeddings, documents=docs)


def test_vector_search(vector_store, query: str = "프레임워크", k: int = 2):
    """(Deprecated) 기존 테스트 함수. 신규 코드는 `similarity_search` 사용."""
    return vector_store.similarity_search(query, k=k)


# 과거 전역 상수 호환(가능한 한 사용 지양)
_cfg = get_vector_store_config()
CONNECTION_STRING = _cfg.connection_string
COLLECTION_NAME = _cfg.collection_name

__all__ = [
    "VectorStoreConfig",
    "get_vector_store_config",
    "create_embeddings",
    "similarity_search",
    "upsert_documents",
    "wait_for_postgres",
    "initialize_vector_store",
    "test_vector_search",
    "CONNECTION_STRING",
    "COLLECTION_NAME",
]

