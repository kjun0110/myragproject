"""
Player Embedding Service - 임베딩 배치 처리.

- DB 선수 조회, 임베딩 텍스트 생성, 벡터화, players.embedding 컬럼 저장까지 담당.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.session import AsyncSessionLocal
from app.domains.v10.soccer.hub.repositories.player_repository import PlayerRepository
from app.domains.v10.soccer.spokes.infrastructure.embedding_client import encode_async

logger = logging.getLogger(__name__)

EMBEDDING_BATCH_SIZE = 50


class PlayerEmbeddingService:
    """선수 임베딩 배치 처리 서비스."""

    @staticmethod
    def _player_to_dict(player: Any) -> Dict[str, Any]:
        """Player ORM을 임베딩용 dict로 변환."""
        return {
            "id": player.id,
            "team_id": getattr(player, "team_id", None),
            "team_code": getattr(player, "team_code", None),
            "player_name": getattr(player, "player_name", None),
            "e_player_name": getattr(player, "e_player_name", None),
            "nickname": getattr(player, "nickname", None),
            "join_yyyy": getattr(player, "join_yyyy", None),
            "position": getattr(player, "position", None),
            "back_no": getattr(player, "back_no", None),
            "nation": getattr(player, "nation", None),
            "birth_date": str(player.birth_date) if getattr(player, "birth_date", None) else None,
            "solar": getattr(player, "solar", None),
            "height": getattr(player, "height", None),
            "weight": getattr(player, "weight", None),
        }

    @staticmethod
    def _create_embedding_text(player_dict: Dict[str, Any]) -> str:
        """선수 정보를 임베딩용 텍스트로 변환."""
        parts = [
            f"이름: {player_dict.get('player_name', '')}" if player_dict.get('player_name') else "",
            f"영문명: {player_dict.get('e_player_name', '')}" if player_dict.get('e_player_name') else "",
            f"별명: {player_dict.get('nickname', '')}" if player_dict.get('nickname') else "",
            f"포지션: {player_dict.get('position', '')}" if player_dict.get('position') else "",
            f"팀: {player_dict.get('team_code', '')}" if player_dict.get('team_code') else "",
            f"입단연도: {player_dict.get('join_yyyy', '')}" if player_dict.get('join_yyyy') else "",
            f"등번호: {player_dict.get('back_no', '')}" if player_dict.get('back_no') else "",
            f"국적: {player_dict.get('nation', '')}" if player_dict.get('nation') else "",
            f"생년월일: {player_dict.get('birth_date', '')}" if player_dict.get('birth_date') else "",
            f"키: {player_dict.get('height', '')}cm" if player_dict.get('height') else "",
            f"몸무게: {player_dict.get('weight', '')}kg" if player_dict.get('weight') else "",
        ]
        text = " ".join(p for p in parts if p)
        return text if text else str(player_dict.get("id", ""))

    async def run_full_embedding_batch(
        self, session: AsyncSession | None = None
    ) -> Dict[str, Any]:
        """
        DB 전체 선수 조회 → 50명 단위 chunk → 임베딩 텍스트 생성 → 벡터화·저장.

        Returns:
            {success, message, saved_count, total, failed_player_ids}
        """
        async def _run(sess: AsyncSession) -> Dict[str, Any]:
            player_repo = PlayerRepository(sess)
            players = await player_repo.get_all()
            logger.info("[EMBEDDING] DB에서 선수 %d명 조회", len(players))

            if not players:
                return {
                    "success": True,
                    "message": "처리할 선수가 없습니다.",
                    "saved_count": 0,
                    "total": 0,
                }

            player_dicts = [self._player_to_dict(p) for p in players]
            total_count = len(player_dicts)
            total_saved = 0
            all_failed: List[Dict[str, Any]] = []

            for start in range(0, total_count, EMBEDDING_BATCH_SIZE):
                chunk = player_dicts[start : start + EMBEDDING_BATCH_SIZE]
                batch_num = start // EMBEDDING_BATCH_SIZE + 1
                logger.info(
                    "[EMBEDDING] 배치 %d/%d 처리 중 (%d명)",
                    batch_num,
                    (total_count + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE,
                    len(chunk),
                )

                items: List[Tuple[int, str]] = []
                for player_dict in chunk:
                    embedding_text = self._create_embedding_text(player_dict)
                    items.append((player_dict["id"], embedding_text))

                result = await self.run_batch_indexing(items, session=sess)
                total_saved += result.get("saved_count", 0)
                errors = result.get("errors") or []
                for err in errors:
                    logger.warning(
                        "[EMBEDDING] 임베딩 DB 저장 실패 player_id=%s: %s",
                        err.get("player_id"),
                        err.get("error"),
                    )
                if errors:
                    all_failed.extend(errors)

            message = f"임베딩 완료: 총 {total_saved}/{total_count}명 저장 (50명 단위 배치)"
            if all_failed:
                failed_ids_str = ", ".join(str(e.get("player_id")) for e in all_failed)
                message += f" 저장 실패: player_id=[{failed_ids_str}]"
            logger.info("[EMBEDDING] %s", message)

            return {
                "success": True,
                "message": message,
                "saved_count": total_saved,
                "total": total_count,
                "failed_player_ids": [e.get("player_id") for e in all_failed] if all_failed else None,
            }

        if session:
            return await _run(session)
        async with AsyncSessionLocal() as sess:
            return await _run(sess)

    async def run_batch_indexing(
        self, items: List[Tuple[int, str]], session: AsyncSession | None = None
    ) -> Dict[str, Any]:
        """
        (player_id, embedding_text) 리스트를 받아 임베딩 생성 후 players.embedding 컬럼에 저장.

        Args:
            items: [(player_id, embedding_text), ...]
            session: DB 세션 (없으면 새로 생성)

        Returns:
            {success, message, saved_count, total, errors}
        """
        if not items:
            return {"success": True, "message": "처리할 항목이 없습니다.", "saved_count": 0, "total": 0}

        total = len(items)
        errors: List[Dict] = []

        async def _run(sess: AsyncSession) -> int:
            repo = PlayerRepository(sess)
            embedding_texts = [text for _, text in items]
            try:
                embeddings = await encode_async(embedding_texts)
            except Exception as e:
                logger.exception("[EMBEDDING] 벡터 생성 실패: %s", e)
                raise

            count = 0
            for (player_id, embedding_text), embedding in zip(items, embeddings):
                try:
                    await repo.update_embedding(
                        player_id,
                        embedding,
                        embedding_content=embedding_text,
                    )
                    count += 1
                except ValueError as e:
                    # Player를 찾을 수 없는 경우
                    logger.warning(
                        "[EMBEDDING] player_id=%s 찾을 수 없음: %s",
                        player_id,
                        str(e),
                    )
                    errors.append({"player_id": player_id, "error": str(e)})
                except Exception as e:
                    # 기타 에러 발생 시
                    await sess.rollback()
                    error_type = type(e).__name__
                    logger.error(
                        "[EMBEDDING] player_id=%s 임베딩 업데이트 실패 (%s): %s",
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
