"""GRI Standards 라우터.

GRI 표준 관련 API 엔드포인트를 제공합니다.
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

from app.domains.esg.total.models.gri_standards_model import (
    GRIStandardCreate,
    GRIStandardListResponse,
    GRIStandardResponse,
    GRIStandardUpdate,
)
from app.domains.esg.total.orchestrator.gri_standards_flow import (
    get_gri_standards_orchestrator,
)

router = APIRouter()


@router.post("/", response_model=GRIStandardResponse)
async def create_gri_standard(request: GRIStandardCreate):
    """GRI 표준 생성 엔드포인트."""
    try:
        orchestrator = get_gri_standards_orchestrator()
        result = orchestrator.analyze(
            text=f"GRI 표준 생성: {request.standard_code} - {request.standard_name}",
        )

        # TODO: 실제 데이터베이스 저장 로직 구현
        return GRIStandardResponse(
            id=1,
            standard_code=request.standard_code,
            standard_name=request.standard_name,
            category=request.category,
            published_year=request.published_year,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GRI 표준 생성 실패: {str(e)}")


@router.get("/", response_model=GRIStandardListResponse)
async def list_gri_standards(
    category: str | None = None,
    standard_code: str | None = None,
):
    """GRI 표준 목록 조회 엔드포인트."""
    try:
        query = f"GRI 표준 조회"
        if category:
            query += f" 카테고리: {category}"
        if standard_code:
            query += f" 코드: {standard_code}"

        orchestrator = get_gri_standards_orchestrator()
        result = orchestrator.analyze(text=query)

        # TODO: 실제 데이터베이스 조회 로직 구현
        return GRIStandardListResponse(items=[], total=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GRI 표준 조회 실패: {str(e)}")


@router.get("/{standard_id}", response_model=GRIStandardResponse)
async def get_gri_standard(standard_id: int):
    """GRI 표준 상세 조회 엔드포인트."""
    try:
        query = f"GRI 표준 ID {standard_id} 상세 조회"
        orchestrator = get_gri_standards_orchestrator()
        result = orchestrator.analyze(text=query)

        # TODO: 실제 데이터베이스 조회 로직 구현
        raise HTTPException(status_code=404, detail="GRI 표준을 찾을 수 없습니다.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GRI 표준 조회 실패: {str(e)}")


@router.put("/{standard_id}", response_model=GRIStandardResponse)
async def update_gri_standard(standard_id: int, request: GRIStandardUpdate):
    """GRI 표준 수정 엔드포인트."""
    try:
        query = f"GRI 표준 ID {standard_id} 수정"
        orchestrator = get_gri_standards_orchestrator()
        result = orchestrator.analyze(text=query)

        # TODO: 실제 데이터베이스 수정 로직 구현
        raise HTTPException(status_code=404, detail="GRI 표준을 찾을 수 없습니다.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GRI 표준 수정 실패: {str(e)}")


@router.delete("/{standard_id}")
async def delete_gri_standard(standard_id: int):
    """GRI 표준 삭제 엔드포인트."""
    try:
        query = f"GRI 표준 ID {standard_id} 삭제"
        orchestrator = get_gri_standards_orchestrator()
        result = orchestrator.analyze(text=query)

        # TODO: 실제 데이터베이스 삭제 로직 구현
        return {"message": f"GRI 표준 ID {standard_id}가 삭제되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GRI 표준 삭제 실패: {str(e)}")
