"""
Team Service - 규칙 기반 처리

단순한 규칙과 정형화된 로직을 처리합니다.
데이터 검증 및 변환 후 Repository를 통해 데이터베이스에 저장합니다.
"""

import logging
from typing import Any, Dict, List

from app.core.database.session import AsyncSessionLocal
from app.domains.v10.soccer.hub.repositories.team_repository import TeamRepository

logger = logging.getLogger(__name__)


class TeamService:
    """Team 데이터를 규칙 기반으로 처리하는 Service."""
    
    def __init__(self):
        logger.info("[SERVICE] TeamService 초기화")
    
    def _validate_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Team 레코드를 검증하고 변환합니다.
        
        Args:
            record: 검증할 레코드
            
        Returns:
            검증 및 변환된 레코드
            
        Raises:
            ValueError: 필수 필드가 없거나 유효하지 않은 경우
        """
        # 필수 필드 검증
        required_fields = ["id", "team_code", "region_name", "team_name"]
        for field in required_fields:
            if field not in record or record[field] is None:
                raise ValueError(f"필수 필드 '{field}'가 없거나 None입니다.")
        
        # 데이터 타입 변환 및 검증 (안전한 변환)
        def safe_int(value, field_name: str = ""):
            """안전하게 int로 변환. 실패 시 None 반환"""
            if value is None or value == "":
                return None
            try:
                if isinstance(value, str):
                    cleaned = value.strip()
                    if cleaned.isdigit():
                        return int(cleaned)
                    logger.warning(f"필드 '{field_name}'에 숫자가 아닌 값: '{value}', None으로 설정")
                    return None
                return int(value)
            except (ValueError, TypeError):
                logger.warning(f"필드 '{field_name}' 변환 실패: '{value}', None으로 설정")
                return None
        
        validated = {
            "id": safe_int(record["id"], "id"),
            "stadium_id": safe_int(record.get("stadium_id"), "stadium_id"),
            "team_code": str(record["team_code"]),
            "region_name": str(record["region_name"]),
            "team_name": str(record["team_name"]),
            "e_team_name": str(record["e_team_name"]) if record.get("e_team_name") else None,
            "orig_yyyy": str(record["orig_yyyy"]) if record.get("orig_yyyy") else None,
            "stadium_code": str(record["stadium_code"]) if record.get("stadium_code") else None,
            "zip_code1": str(record["zip_code1"]) if record.get("zip_code1") else None,
            "zip_code2": str(record["zip_code2"]) if record.get("zip_code2") else None,
            "address": str(record["address"]) if record.get("address") else None,
            "ddd": str(record["ddd"]) if record.get("ddd") else None,
            "tel": str(record["tel"]) if record.get("tel") else None,
            "fax": str(record["fax"]) if record.get("fax") else None,
            "homepage": str(record["homepage"]) if record.get("homepage") else None,
            "owner": str(record["owner"]) if record.get("owner") else None,
        }
        
        # id는 필수이므로 None이면 에러
        if validated["id"] is None:
            raise ValueError(f"필수 필드 'id'가 유효하지 않습니다: {record.get('id')}")
        
        return validated
    
    async def process(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Team 레코드들을 규칙 기반으로 처리하고 데이터베이스에 저장합니다.
        
        Args:
            records: 처리할 Team 레코드 리스트
            
        Returns:
            처리 결과 딕셔너리
        """
        logger.info(f"[SERVICE] 규칙 기반 처리 시작: {len(records)}개 레코드")
        
        # 데이터 검증 및 변환
        validated_records = []
        validation_errors = []
        
        for idx, record in enumerate(records, 1):
            try:
                validated = self._validate_record(record)
                validated_records.append(validated)
            except Exception as e:
                error_msg = f"레코드 {idx} 검증 실패: {e}"
                logger.warning(f"[SERVICE] {error_msg}")
                validation_errors.append({"index": idx, "error": str(e), "data": record})
        
        logger.info(
            f"[SERVICE] 검증 완료: "
            f"성공 {len(validated_records)}개, 실패 {len(validation_errors)}개"
        )
        
        if not validated_records:
            return {
                "success": False,
                "message": "유효한 레코드가 없습니다.",
                "total_records": len(records),
                "validated_records": 0,
                "validation_errors": validation_errors,
            }
        
        # 데이터베이스에 저장
        async with AsyncSessionLocal() as session:
            try:
                repository = TeamRepository(session)
                created_teams = await repository.create_bulk(validated_records)
                
                # commit 전에 id 추출 (세션 닫힌 후 접근 방지)
                saved_team_ids = [t.id for t in created_teams]
                
                await repository.commit()
                
                logger.info(
                    f"[SERVICE] 데이터베이스 저장 완료: "
                    f"{len(created_teams)}개 레코드 저장됨"
                )
                
                result = {
                    "success": True,
                    "message": "규칙 기반 처리 및 데이터베이스 저장 완료",
                    "total_records": len(records),
                    "validated_records": len(validated_records),
                    "saved_records": len(created_teams),
                    "validation_errors": validation_errors,
                    "saved_team_ids": saved_team_ids,
                }
                
                return result
                
            except Exception as e:
                logger.error(f"[SERVICE] 데이터베이스 저장 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "message": f"데이터베이스 저장 실패: {str(e)}",
                    "total_records": len(records),
                    "validated_records": len(validated_records),
                    "validation_errors": validation_errors,
                }
