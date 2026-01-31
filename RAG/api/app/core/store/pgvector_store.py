"""PGVector(=pgvector) 스토어 어댑터.

목표:
- 실행/모니터링/데모 로직을 배제한 "순수 라이브러리" 형태
- 설정은 app.core.config.Settings(get_settings)에서 주입
- LangChain VectorStore(PGVector) 생성/조회에 필요한 최소 API 제공
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from app.core.config import get_settings

from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_openai import OpenAIEmbeddings


@dataclass(frozen=True)
class VectorStoreConfig:
    """Vector store 연결 설정."""

    connection_string: str
    collection_name: str


def _ensure_sslmode(connection_string: str, sslmode: str = "require") -> str:
    """psycopg2 등에서 sslmode가 필요할 때 보정."""
    if "sslmode=" in connection_string:
        return connection_string
    parsed = urlparse(connection_string)
    if parsed.scheme not in {"postgresql", "postgres"}:
        return connection_string
    query = parse_qs(parsed.query)
    query["sslmode"] = [sslmode]
    new_query = urlencode(query, doseq=True)
    return urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment)
    )


def get_vector_store_config() -> VectorStoreConfig:
    """설정에서 VectorStoreConfig 생성."""
    settings = get_settings()
    # Neon/운영 환경에서 psycopg2 연결용 sslmode 누락 방지
    conn = _ensure_sslmode(settings.connection_string, sslmode=os.getenv("sslmode", "require"))
    return VectorStoreConfig(connection_string=conn, collection_name=settings.collection_name)


def create_embeddings(prefer_openai: bool = True) -> Embeddings:
    """Embedding 생성.

    - OpenAI 키가 있으면 OpenAIEmbeddings 사용
    - 없으면 FakeEmbeddings로 fallback (개발/테스트 편의)
    """
    settings = get_settings()

    if prefer_openai:
        # Settings에 키가 있는데 env에 없으면 보정 (LangChain이 env를 읽는 케이스 대비)
        if settings.openai_api_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key and openai_api_key != "your-api-key-here":
            return OpenAIEmbeddings()

    # pgvector 컬렉션/테이블 생성 테스트용 최소 임베딩
    return FakeEmbeddings(size=1536)


def get_pgvector_store(
    embeddings: Embeddings,
    config: Optional[VectorStoreConfig] = None,
) -> PGVector:
    """기존 컬렉션에 연결하는 PGVector 스토어를 반환."""
    cfg = config or get_vector_store_config()
    return PGVector(
        embedding_function=embeddings,
        collection_name=cfg.collection_name,
        connection_string=cfg.connection_string,
    )


def upsert_documents(
    embeddings: Embeddings,
    documents: Sequence[Document],
    config: Optional[VectorStoreConfig] = None,
) -> PGVector:
    """문서를 임베딩하여 스토어에 저장(필요 시 컬렉션 생성 포함)."""
    cfg = config or get_vector_store_config()
    return PGVector.from_documents(
        embedding=embeddings,
        documents=list(documents),
        collection_name=cfg.collection_name,
        connection_string=cfg.connection_string,
    )


def similarity_search(
    embeddings: Embeddings,
    query: str,
    k: int = 3,
    config: Optional[VectorStoreConfig] = None,
) -> List[Document]:
    """유사도 검색."""
    store = get_pgvector_store(embeddings=embeddings, config=config)
    return store.similarity_search(query, k=k)


def wait_for_postgres(
    connection_string: Optional[str] = None,
    max_retries: int = 30,
    delay: int = 2,
) -> None:
    """PostgreSQL이 준비될 때까지 대기(헬스체크/로컬 개발용)."""
    import psycopg2

    cfg = get_vector_store_config()
    conn_str = connection_string or cfg.connection_string

    for i in range(max_retries):
        try:
            conn = psycopg2.connect(conn_str)
            conn.close()
            return
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(delay)
            else:
                raise ConnectionError(f"PostgreSQL 연결 실패: {e}")

