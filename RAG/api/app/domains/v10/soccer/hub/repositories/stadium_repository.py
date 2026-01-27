"""
Stadium Repository - 데이터베이스 접근 계층

Neon 데이터베이스의 stadium 테이블에 대한 CRUD 작업을 담당합니다.
"""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from app.domains.v10.soccer.models.bases.stadiums import Stadium

logger = logging.getLogger(__name__)


class StadiumRepository:
    """Stadium 데이터베이스 접근을 담당하는 Repository."""
    
    def __init__(self, session: AsyncSession):
        """
        StadiumRepository 초기화.
        
        Args:
            session: 비동기 데이터베이스 세션
        """
        self.session = session
        logger.info("[REPOSITORY] StadiumRepository 초기화")
    
    async def create(self, stadium_data: Dict[str, Any]) -> Stadium:
        """
        단일 Stadium 레코드를 생성합니다.
        
        Args:
            stadium_data: Stadium 데이터 딕셔너리
            
        Returns:
            생성된 Stadium 인스턴스
            
        Raises:
            IntegrityError: 중복 키 또는 제약 조건 위반 시
        """
        try:
            stadium = Stadium(**stadium_data)
            
            self.session.add(stadium)
            await self.session.flush()
            
            logger.debug(f"[REPOSITORY] Stadium 생성: id={stadium.id}, name={stadium.stadium_name}")
            
            return stadium
            
        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"[REPOSITORY] Stadium 생성 실패 (무결성 오류): {e}")
            raise
        except Exception as e:
            await self.session.rollback()
            logger.error(f"[REPOSITORY] Stadium 생성 실패: {e}")
            raise
    
    async def create_bulk(self, stadiums_data: List[Dict[str, Any]]) -> List[Stadium]:
        """
        여러 Stadium 레코드를 일괄 생성합니다.
        
        Args:
            stadiums_data: Stadium 데이터 딕셔너리 리스트
            
        Returns:
            생성된 Stadium 인스턴스 리스트
        """
        logger.info(f"[REPOSITORY] 일괄 생성 시작: {len(stadiums_data)}개 레코드")
        
        created_stadiums = []
        errors = []
        
        for idx, stadium_data in enumerate(stadiums_data, 1):
            try:
                stadium = await self.create(stadium_data)
                created_stadiums.append(stadium)
            except IntegrityError as e:
                error_msg = f"레코드 {idx} 생성 실패 (무결성 오류): {e}"
                logger.warning(f"[REPOSITORY] {error_msg}")
                errors.append({"index": idx, "data": stadium_data, "error": str(e)})
                continue
            except Exception as e:
                error_msg = f"레코드 {idx} 생성 실패: {e}"
                logger.error(f"[REPOSITORY] {error_msg}")
                errors.append({"index": idx, "data": stadium_data, "error": str(e)})
                continue
        
        logger.info(
            f"[REPOSITORY] 일괄 생성 완료: "
            f"성공 {len(created_stadiums)}개, 실패 {len(errors)}개"
        )
        
        if errors:
            logger.warning(f"[REPOSITORY] 실패한 레코드: {errors[:5]}...")
        
        return created_stadiums
    
    async def get_by_id(self, stadium_id: int) -> Optional[Stadium]:
        """
        ID로 Stadium을 조회합니다.
        
        Args:
            stadium_id: Stadium ID
            
        Returns:
            Stadium 인스턴스 또는 None
        """
        result = await self.session.execute(
            select(Stadium).where(Stadium.id == stadium_id)
        )
        return result.scalar_one_or_none()
    
    async def get_all(self, limit: Optional[int] = None, offset: int = 0) -> List[Stadium]:
        """
        모든 Stadium을 조회합니다.
        
        Args:
            limit: 조회할 최대 개수
            offset: 건너뛸 개수
            
        Returns:
            Stadium 인스턴스 리스트
        """
        query = select(Stadium).offset(offset)
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
