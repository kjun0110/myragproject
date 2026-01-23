"""GRI Social Contents 라우터.

GRI 사회 도메인 콘텐츠 관련 API 엔드포인트를 제공합니다.
"""

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
router_dir = current_file.parent.parent.parent  # api/app/routers/
app_dir = router_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

sys.path.insert(0, str(api_dir))

from app.domains.v1.esg.total.models.gri_soc_contents_model import (
    GRISocContentCreate,
    GRISocContentListResponse,
    GRISocContentResponse,
    GRISocContentUpdate,
)
from app.domains.v1.esg.total.orchestrator.gri_soc_contents_flow import (
    get_gri_soc_contents_orchestrator,
)

router = APIRouter()


@router.post("/", response_model=GRISocContentResponse)
async def create_gri_soc_content(request: GRISocContentCreate):
    """GRI 사회 도메인 콘텐츠 생성 엔드포인트."""
    try:
        orchestrator = get_gri_soc_contents_orchestrator()
        result = orchestrator.analyze(
            text=f"GRI 사회 도메인 콘텐츠 생성: 표준 ID {request.standard_id} - {request.disclosure_num}",
        )

        # TODO: 실제 데이터베이스 저장 로직 구현
        return GRISocContentResponse(
            id=1,
            standard_id=request.standard_id,
            disclosure_num=request.disclosure_num,
            content=request.content,
            metadata=request.metadata,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"GRI 사회 도메인 콘텐츠 생성 실패: {str(e)}"
        )


@router.get("/", response_model=GRISocContentListResponse)
async def list_gri_soc_contents(
    standard_id: int | None = None,
    disclosure_num: str | None = None,
):
    """GRI 사회 도메인 콘텐츠 목록 조회 엔드포인트."""
    try:
        query = "GRI 사회 도메인 콘텐츠 조회"
        if standard_id:
            query += f" 표준 ID: {standard_id}"
        if disclosure_num:
            query += f" 공개 번호: {disclosure_num}"

        orchestrator = get_gri_soc_contents_orchestrator()
        result = orchestrator.analyze(text=query)

        # TODO: 실제 데이터베이스 조회 로직 구현
        return GRISocContentListResponse(items=[], total=0)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"GRI 사회 도메인 콘텐츠 조회 실패: {str(e)}"
        )


@router.get("/{content_id}", response_model=GRISocContentResponse)
async def get_gri_soc_content(content_id: int):
    """GRI 사회 도메인 콘텐츠 상세 조회 엔드포인트."""
    try:
        query = f"GRI 사회 도메인 콘텐츠 ID {content_id} 상세 조회"
        orchestrator = get_gri_soc_contents_orchestrator()
        result = orchestrator.analyze(text=query)

        # TODO: 실제 데이터베이스 조회 로직 구현
        raise HTTPException(
            status_code=404, detail="GRI 사회 도메인 콘텐츠를 찾을 수 없습니다."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"GRI 사회 도메인 콘텐츠 조회 실패: {str(e)}"
        )


@router.put("/{content_id}", response_model=GRISocContentResponse)
async def update_gri_soc_content(content_id: int, request: GRISocContentUpdate):
    """GRI 사회 도메인 콘텐츠 수정 엔드포인트."""
    try:
        query = f"GRI 사회 도메인 콘텐츠 ID {content_id} 수정"
        orchestrator = get_gri_soc_contents_orchestrator()
        result = orchestrator.analyze(text=query)

        # TODO: 실제 데이터베이스 수정 로직 구현
        raise HTTPException(
            status_code=404, detail="GRI 사회 도메인 콘텐츠를 찾을 수 없습니다."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"GRI 사회 도메인 콘텐츠 수정 실패: {str(e)}"
        )


@router.delete("/{content_id}")
async def delete_gri_soc_content(content_id: int):
    """GRI 사회 도메인 콘텐츠 삭제 엔드포인트."""
    try:
        query = f"GRI 사회 도메인 콘텐츠 ID {content_id} 삭제"
        orchestrator = get_gri_soc_contents_orchestrator()
        result = orchestrator.analyze(text=query)

        # TODO: 실제 데이터베이스 삭제 로직 구현
        return {"message": f"GRI 사회 도메인 콘텐츠 ID {content_id}가 삭제되었습니다."}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"GRI 사회 도메인 콘텐츠 삭제 실패: {str(e)}"
        )
