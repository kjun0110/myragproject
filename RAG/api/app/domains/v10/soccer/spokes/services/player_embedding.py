"""
Player Embedding Service - 임베딩 배치 처리.

- 엑사원이 생성한 content를 받아서 벡터 생성 및 players_embeddings 테이블에 저장.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.session import AsyncSessionLocal
from app.domains.v10.soccer.hub.repositories.player_embedding_repository import (
    PlayerEmbeddingRepository,
)
from app.domains.v10.soccer.spokes.infrastructure.embedding_client import encode_async

logger = logging.getLogger(__name__)


class PlayerEmbeddingService:
    """선수 임베딩 배치 처리 서비스."""

    async def run_batch_indexing(
        self, items: List[Tuple[int, str]], session: AsyncSession | None = None
    ) -> Dict[str, Any]:
        """
        (player_id, content) 리스트를 받아 임베딩 생성 후 DB에 저장.

        Args:
            items: [(player_id, content), ...]
            session: DB 세션 (없으면 새로 생성)

        Returns:
            {success, message, saved_count, total, errors}
        """
        if not items:
            return {"success": True, "message": "처리할 항목이 없습니다.", "saved_count": 0, "total": 0}

        total = len(items)
        errors: List[Dict] = []

        async def _run(sess: AsyncSession) -> int:
            repo = PlayerEmbeddingRepository(sess)
            contents = [content for _, content in items]
            try:
                embeddings = await encode_async(contents)
            except Exception as e:
                logger.exception("[EMBEDDING] 벡터 생성 실패: %s", e)
                raise

            count = 0
            for (player_id, content), embedding in zip(items, embeddings):
                try:
                    await repo.delete_by_player_id(player_id)
                    await repo.create(player_id, content, embedding)
                    count += 1
                except Exception as e:
                    # 에러 발생 시 즉시 rollback하여 세션 복구
                    await repo.rollback()
                    error_type = type(e).__name__
                    logger.error(
                        "[EMBEDDING] player_id=%s DB 저장 실패 (%s): %s",
                        player_id,
                        error_type,
                        str(e),
                        exc_info=False  # 스택 트레이스 축소
                    )
                    errors.append({"player_id": player_id, "error": str(e)})

            await repo.commit()
            return count

        if session:
            saved_count = await _run(session)
        else:
            async with AsyncSessionLocal() as sess:
                saved_count = await _run(sess)

        return {
            "success": True,
            "message": f"임베딩 배치 완료: {saved_count}/{total} 저장",
            "saved_count": saved_count,
            "total": total,
            "errors": errors if errors else None,
        }
