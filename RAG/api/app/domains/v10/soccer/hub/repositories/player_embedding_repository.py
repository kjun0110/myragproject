"""
PlayerEmbedding Repository - players_embeddings 테이블 접근.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.v10.soccer.models.bases.player_embeddings import PlayerEmbedding

logger = logging.getLogger(__name__)


class PlayerEmbeddingRepository:
    """PlayerEmbedding DB 접근."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def delete_by_player_id(self, player_id: int) -> int:
        """player_id에 해당하는 기존 임베딩 삭제."""
        result = await self.session.execute(
            delete(PlayerEmbedding).where(PlayerEmbedding.player_id == player_id)
        )
        return result.rowcount or 0

    async def create(self, player_id: int, content: str, embedding: List[float]) -> PlayerEmbedding:
        """단일 PlayerEmbedding 생성."""
        pe = PlayerEmbedding(
            player_id=player_id,
            content=content,
            embedding=embedding,
        )
        self.session.add(pe)
        await self.session.flush()
        return pe

    async def create_bulk(self, items: List[Tuple[int, str, List[float]]]) -> int:
        """player_id, content, embedding 리스트로 일괄 생성."""
        count = 0
        for player_id, content, embedding in items:
            await self.delete_by_player_id(player_id)
            await self.create(player_id, content, embedding)
            count += 1
        return count

    async def commit(self) -> None:
        await self.session.commit()

    async def rollback(self) -> None:
        """트랜잭션 롤백 (에러 복구용)"""
        await self.session.rollback()
