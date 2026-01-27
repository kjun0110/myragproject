"""
Schedule 데이터 JSONL 파일 업로드 라우터

JSONL 파일을 받아서 첫 번째부터 다섯 번째 행까지만 출력합니다.
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse

# 프로젝트 경로 설정
current_file = Path(__file__).resolve()
router_dir = current_file.parent  # api/app/routers/v10/soccer/
routers_dir = router_dir.parent.parent  # api/app/routers/
app_dir = routers_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v10/soccer/schedule", tags=["Schedule"])


@router.post("/upload")
async def upload_schedule_jsonl(
    file: UploadFile = File(...),
):
    """
    Schedule JSONL 파일을 업로드하여 첫 5개 행을 출력합니다.
    
    Args:
        file: 업로드할 JSONL 파일
    
    Returns:
        첫 5개 행의 데이터
    """
    logger.info("[ROUTER] Schedule 업로드 라우터 도달")
    logger.info(f"[ROUTER] 파일명: {file.filename}")
    logger.info(f"[ROUTER] 파일 크기: {file.size if hasattr(file, 'size') else '알 수 없음'}")
    logger.info("[ROUTER] 파일 처리 시작")
    
    # 파일 확장자 확인
    if not file.filename.endswith(".jsonl"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="JSONL 파일만 업로드 가능합니다.",
        )
    
    try:
        # 파일 내용 읽기
        content = await file.read()
        text_content = content.decode("utf-8")
        
        # JSONL 파싱 (각 줄이 JSON 객체)
        records = []
        lines = text_content.strip().split("\n")
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"라인 {line_num} 파싱 실패: {e}")
                continue
        
        if not records:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="유효한 JSON 레코드가 없습니다.",
            )
        
        # 첫 5개 행을 로그로 출력
        first_five_records = records[:5]
        logger.info(f"[ROUTER] 총 {len(records)}개 레코드 중 첫 5개 레코드:")
        for idx, record in enumerate(first_five_records, 1):
            logger.info(f"[ROUTER] 레코드 {idx}: {json.dumps(record, ensure_ascii=False, indent=2)}")
        
        # Orchestrator를 통한 전략 패턴 처리
        logger.info("[ROUTER] Orchestrator를 통한 처리 시작")
        from app.domains.v10.soccer.hub.orchestrators.schedule_orchestrator import ScheduleOrchestrator
        
        orchestrator = ScheduleOrchestrator()
        result = await orchestrator.process(records)
        
        logger.info(f"[ROUTER] Schedule 파일 업로드 및 처리 완료: 총 {len(records)}개 레코드")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 업로드 처리 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"파일 처리 실패: {str(e)}",
        )
