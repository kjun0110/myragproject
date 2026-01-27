"""
Player Repository - 데이터베이스 접근 계층

Neon 데이터베이스의 player 테이블에 대한 CRUD 작업을 담당합니다.
"""

import logging
from datetime import date
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from app.domains.v10.soccer.models.bases.players import Player

logger = logging.getLogger(__name__)


class PlayerRepository:
    """Player 데이터베이스 접근을 담당하는 Repository."""

    def __init__(self, session: AsyncSession):
        """
        PlayerRepository 초기화.

        Args:
            session: 비동기 데이터베이스 세션
        """
        self.session = session
        logger.info("[REPOSITORY] PlayerRepository 초기화")

    async def create(self, player_data: Dict[str, Any]) -> Player:
        """
        단일 Player 레코드를 생성합니다.

        Args:
            player_data: Player 데이터 딕셔너리

        Returns:
            생성된 Player 인스턴스

        Raises:
            IntegrityError: 중복 키 또는 제약 조건 위반 시
        """
        try:
            # birth_date 문자열을 date 객체로 변환
            if isinstance(player_data.get("birth_date"), str):
                player_data["birth_date"] = date.fromisoformat(player_data["birth_date"])

            # Player 인스턴스 생성
            player = Player(**player_data)

            self.session.add(player)
            await self.session.flush()  # ID를 얻기 위해 flush

            logger.debug(f"[REPOSITORY] Player 생성: id={player.id}, name={player.player_name}")

            return player

        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"[REPOSITORY] Player 생성 실패 (무결성 오류): {e}")
            raise
        except Exception as e:
            await self.session.rollback()
            logger.error(f"[REPOSITORY] Player 생성 실패: {e}")
            raise

    async def create_bulk(self, players_data: List[Dict[str, Any]]) -> List[Player]:
        """
        여러 Player 레코드를 일괄 생성합니다.

        Args:
            players_data: Player 데이터 딕셔너리 리스트

        Returns:
            생성된 Player 인스턴스 리스트
        """
        logger.info(f"[REPOSITORY] 일괄 생성 시작: {len(players_data)}개 레코드")

        created_players = []
        errors = []

        for idx, player_data in enumerate(players_data, 1):
            try:
                player = await self.create(player_data)
                created_players.append(player)
            except IntegrityError as e:
                # 중복 키 등의 경우 스킵하고 계속 진행
                error_msg = f"레코드 {idx} 생성 실패 (무결성 오류): {e}"
                logger.warning(f"[REPOSITORY] {error_msg}")
                errors.append({"index": idx, "data": player_data, "error": str(e)})
                continue
            except Exception as e:
                error_msg = f"레코드 {idx} 생성 실패: {e}"
                logger.error(f"[REPOSITORY] {error_msg}")
                errors.append({"index": idx, "data": player_data, "error": str(e)})
                continue

        logger.info(
            f"[REPOSITORY] 일괄 생성 완료: "
            f"성공 {len(created_players)}개, 실패 {len(errors)}개"
        )

        if errors:
            logger.warning(f"[REPOSITORY] 실패한 레코드: {errors[:5]}...")  # 처음 5개만 로그

        return created_players

    async def get_by_id(self, player_id: int) -> Optional[Player]:
        """
        ID로 Player를 조회합니다.

        Args:
            player_id: Player ID

        Returns:
            Player 인스턴스 또는 None
        """
        result = await self.session.execute(
            select(Player).where(Player.id == player_id)
        )
        return result.scalar_one_or_none()

    async def get_all(self, limit: Optional[int] = None, offset: int = 0) -> List[Player]:
        """
        모든 Player를 조회합니다.

        Args:
            limit: 조회할 최대 개수
            offset: 건너뛸 개수

        Returns:
            Player 인스턴스 리스트
        """
        query = select(Player).offset(offset)
        if limit:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def commit(self):
        """변경사항을 커밋합니다."""
        try:
            await self.session.commit()
            logger.info("[REPOSITORY] 변경사항 커밋 완료")
        except Exception as e:
            await self.session.rollback()
            logger.error(f"[REPOSITORY] 커밋 실패: {e}")
            raise
