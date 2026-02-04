"""
Player 데이터 JSONL 파일 업로드 라우터

JSONL 파일을 받아서 첫 번째부터 다섯 번째 행까지만 출력합니다.
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Request, status
from fastapi.responses import JSONResponse

# 프로젝트 경로 설정
current_file = Path(__file__).resolve()
router_dir = current_file.parent  # api/app/routers/v10/soccer/
routers_dir = router_dir.parent.parent  # api/app/routers/
app_dir = routers_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v10/soccer/player", tags=["Player"])


@router.post("/embed")
async def enqueue_player_embedding(request: Request) -> JSONResponse:
    """Player 임베딩 작업을 큐에 넣고 job_id를 반환합니다.

    - 큐에 job 추가 후 온디맨드 워커를 한 번 트리거. 워커는 대기 job을 모두 처리하고 큐가 비면 종료.
    - 프론트는 이 엔드포인트로 작업을 넣고, /embed/status/{job_id} 로 상태를 폴링합니다.
    """
    from app.routers.shared.embedding_queue import add_job_async

    job_id = await add_job_async("player", payload={})
    logger.info("[ROUTER] Player 임베딩 job 등록 job_id=%s", job_id)
    trigger = getattr(request.app.state, "_embedding_worker_trigger", None)
    if trigger:
        trigger(request.app)
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"success": True, "job_id": job_id},
    )


@router.get("/embed/status/{job_id}")
async def get_player_embedding_status(job_id: str) -> JSONResponse:
    """job_id의 임베딩 작업 상태를 반환합니다."""
    from app.routers.shared.embedding_queue import get_status_async

    data = await get_status_async(job_id)
    if data is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"success": False, "error": "job을 찾을 수 없습니다."},
        )
    return JSONResponse(status_code=status.HTTP_200_OK, content=data)


@router.post("/upload")
async def upload_player_jsonl(
    file: UploadFile = File(...),
):
    """
    Player JSONL 파일을 업로드하여 첫 5개 행을 출력합니다.
    
    Args:
        file: 업로드할 JSONL 파일
    
    Returns:
        첫 5개 행의 데이터
    """
    logger.info("[ROUTER] Player 업로드 라우터 도달")
    logger.info(f"[ROUTER] 파일명: {file.filename}")
    
    # 파일 확장자 확인
    if not file.filename.endswith(".jsonl"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="JSONL 파일만 업로드 가능합니다.",
        )
    
    try:
        # 파일 내용 읽기
        content = await file.read()
        file_size = len(content)
        logger.info(f"[ROUTER] 파일 크기: {file_size} bytes")
        logger.info("[ROUTER] 파일 처리 시작")
        
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
        from app.domains.v10.soccer.hub.orchestrators.player_orchestrator import PlayerOrchestrator
        
        orchestrator = PlayerOrchestrator()
        result = await orchestrator.process(records)
        
        logger.info(f"[ROUTER] Player 파일 업로드 및 처리 완료: 총 {len(records)}개 레코드")
        
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
