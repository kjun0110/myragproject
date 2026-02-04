"""
임베딩 클라이언트 - 로컬 Hugging Face 모델 사용.

- jhgan/ko-sroberta-multitask (HF 캐시 또는 로컬)
- 동기/비동기 encode 제공
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List

logger = logging.getLogger(__name__)

MODEL_ID = "jhgan/ko-sroberta-multitask"
EMBEDDING_DIM = 768

# 전역 모델 (지연 로드)
_model = None
_executor = ThreadPoolExecutor(max_workers=1)


def _get_model():
    """jhgan/ko-sroberta-multitask 모델 로드 (HF 캐시 사용)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("[EMBEDDING] 모델 로드 중: %s", MODEL_ID)
            _model = SentenceTransformer(MODEL_ID)
            logger.info("[EMBEDDING] 모델 로드 완료")
        except ImportError as e:
            raise RuntimeError(
                "sentence_transformers가 필요합니다. pip install sentence-transformers"
            ) from e
    return _model


def encode(texts: List[str]) -> List[List[float]]:
    """동기: 텍스트 리스트를 벡터 리스트로 변환 (768차원)."""
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()


async def encode_async(texts: List[str]) -> List[List[float]]:
    """비동기: encode를 스레드 풀에서 실행."""
    if not texts:
        return []
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, encode, texts)
