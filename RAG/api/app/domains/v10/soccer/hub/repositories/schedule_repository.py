"""
Schedule Repository - 데이터베이스 접근 계층

Neon 데이터베이스의 schedule 테이블에 대한 CRUD 작업을 담당합니다.
"""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from app.domains.v10.soccer.models.bases.schedules import Schedule

logger = logging.getLogger(__name__)


class ScheduleRepository:
    """Schedule 데이터베이스 접근을 담당하는 Repository."""
    
    def __init__(self, session: AsyncSession):
        """
        ScheduleRepository 초기화.
        
        Args:
            session: 비동기 데이터베이스 세션
        """
        self.session = session
        logger.info("[REPOSITORY] ScheduleRepository 초기화")
    
    async def create(self, schedule_data: Dict[str, Any]) -> Schedule:
        """
        단일 Schedule 레코드를 생성합니다.
        
        Args:
            schedule_data: Schedule 데이터 딕셔너리
            
        Returns:
            생성된 Schedule 인스턴스
            
        Raises:
            IntegrityError: 중복 키 또는 제약 조건 위반 시
        """
        try:
            schedule = Schedule(**schedule_data)
            
            self.session.add(schedule)
            await self.session.flush()
            
            logger.debug(f"[REPOSITORY] Schedule 생성: id={schedule.id}, date={schedule.sche_date}")
            
            return schedule
            
        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"[REPOSITORY] Schedule 생성 실패 (무결성 오류): {e}")
            raise
        except Exception as e:
            await self.session.rollback()
            logger.error(f"[REPOSITORY] Schedule 생성 실패: {e}")
            raise
    
    async def create_bulk(self, schedules_data: List[Dict[str, Any]]) -> List[Schedule]:
        """
        여러 Schedule 레코드를 일괄 생성합니다.
        
        Args:
            schedules_data: Schedule 데이터 딕셔너리 리스트
            
        Returns:
            생성된 Schedule 인스턴스 리스트
        """
        logger.info(f"[REPOSITORY] 일괄 생성 시작: {len(schedules_data)}개 레코드")
        
        created_schedules = []
        errors = []
        
        for idx, schedule_data in enumerate(schedules_data, 1):
            try:
                schedule = await self.create(schedule_data)
                created_schedules.append(schedule)
            except IntegrityError as e:
                error_msg = f"레코드 {idx} 생성 실패 (무결성 오류): {e}"
                logger.warning(f"[REPOSITORY] {error_msg}")
                errors.append({"index": idx, "data": schedule_data, "error": str(e)})
                continue
            except Exception as e:
                error_msg = f"레코드 {idx} 생성 실패: {e}"
                logger.error(f"[REPOSITORY] {error_msg}")
                errors.append({"index": idx, "data": schedule_data, "error": str(e)})
                continue
        
        logger.info(
            f"[REPOSITORY] 일괄 생성 완료: "
            f"성공 {len(created_schedules)}개, 실패 {len(errors)}개"
        )
        
        if errors:
            logger.warning(f"[REPOSITORY] 실패한 레코드: {errors[:5]}...")
        
        return created_schedules
    
    async def get_by_id(self, schedule_id: int) -> Optional[Schedule]:
        """
        ID로 Schedule을 조회합니다.
        
        Args:
            schedule_id: Schedule ID
            
        Returns:
            Schedule 인스턴스 또는 None
        """
        result = await self.session.execute(
            select(Schedule).where(Schedule.id == schedule_id)
        )
        return result.scalar_one_or_none()
    
    async def get_all(self, limit: Optional[int] = None, offset: int = 0) -> List[Schedule]:
        """
        모든 Schedule을 조회합니다.
        
        Args:
            limit: 조회할 최대 개수
            offset: 건너뛸 개수
            
        Returns:
            Schedule 인스턴스 리스트
        """
        query = select(Schedule).offset(offset)
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
