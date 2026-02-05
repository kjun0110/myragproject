"""
임베딩 content 생성을 위한 스키마 기반 템플릿 시스템.

PK/FK를 제외한 테이블 스키마에서 컬럼 목록을 얻어 "필드명=값" 형태로 붙이고,
LLM에게 제약이 있는 프롬프트를 주어 환각 없이 짧은 한국어 문장을 생성하도록 합니다.
테이블이 늘어나도 이 파일 수정 없이 사용 가능합니다.
"""

from typing import Dict, Any, List, Optional, Union

# SQLAlchemy Table 또는 ORM 모델에서 스키마 정보를 얻기 위함
from sqlalchemy import Table


def _get_table(
    table_or_name: Union[str, type, Table],
    metadata: Optional[Any] = None,
) -> Table:
    """table_or_name + optional metadata 로부터 SQLAlchemy Table 객체를 반환."""
    if isinstance(table_or_name, Table):
        return table_or_name
    if isinstance(table_or_name, str):
        if metadata is None:
            raise ValueError("table_or_name이 문자열일 때 metadata는 필수입니다.")
        if table_or_name not in metadata.tables:
            raise ValueError(f"metadata에 테이블 '{table_or_name}'이 없습니다.")
        return metadata.tables[table_or_name]
    # ORM 모델 클래스 (__table__ 있음)
    if hasattr(table_or_name, "__table__"):
        return table_or_name.__table__
    raise TypeError(
        "table_or_name은 테이블명(str), SQLAlchemy Table, 또는 ORM 모델 클래스여야 합니다."
    )


def get_content_columns(
    table_or_name: Union[str, type, Table],
    metadata: Optional[Any] = None,
) -> List[str]:
    """
    PK·FK를 제외한 컬럼 이름 목록을 스키마 정의 순서대로 반환.

    Args:
        table_or_name: 테이블명(str), Table 객체, 또는 ORM 모델 클래스
        metadata: table_or_name이 str일 때 사용할 SQLAlchemy MetaData (예: Base.metadata)

    Returns:
        컬럼 이름(키) 리스트, 예: ["player_name", "position", "team_code", ...]
    """
    table = _get_table(table_or_name, metadata)
    return [
        c.key
        for c in table.columns
        if not c.primary_key and not c.foreign_keys
    ]


def generate_field_list(
    table_or_name: Union[str, type, Table],
    data: Dict[str, Any],
    metadata: Optional[Any] = None,
) -> str:
    """
    스키마 순서대로 "필드명=값" 형식의 문자열 생성. PK/FK 제외.

    Args:
        table_or_name: 테이블명(str), Table 객체, 또는 ORM 모델 클래스
        data: row 데이터 딕셔너리 (컬럼명 → 값)
        metadata: table_or_name이 str일 때 사용할 MetaData

    Returns:
        "player_name=홍길동, position=MF, team_code=K06, ..." 형식
    """
    columns = get_content_columns(table_or_name, metadata)
    parts = []
    for key in columns:
        value = data.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else "정보없음"


def get_embedding_prompt(
    table_or_name: Union[str, type, Table],
    data: Dict[str, Any],
    metadata: Optional[Any] = None,
) -> str:
    """
    임베딩 content 생성을 위한 LLM 프롬프트 생성.
    스키마에서 PK/FK 제외 컬럼만 사용하며, 제약이 있는 문장 생성을 요청합니다.

    Args:
        table_or_name: 테이블명(str), Table 객체, 또는 ORM 모델 클래스
        data: row 데이터 딕셔너리
        metadata: table_or_name이 str일 때 사용할 MetaData (예: Base.metadata)

    Returns:
        ExaOne 등 LLM에 넘길 프롬프트 문자열
    """
    field_list = generate_field_list(table_or_name, data, metadata)
    table = _get_table(table_or_name, metadata)
    table_name = table.name

    prompt = f"""제공된 정보만 사용해서 RAG 검색용 요약 문장을 **딱 한 문장만** 작성하세요.
예시 목록이나 (O)(X) 설명, 따옴표는 출력하지 말고, 요약 문장만 한 줄로 출력하세요.

형식: [입단년도] 입단한 [팀명]의 [포지션] [선수명]은 [등번호·키·체중·별명 등]. 팀의 핵심 [포지션]으로 활약하며 안정적인 수비/공격 라인 구축에 기여했다. (정보가 적으면 "[팀명] 소속 [포지션] [선수명]." 만)
팀: K01=울산 현대, K02=수원 삼성 블루윙즈, K03=일화천마, K04=전북 현대, K05=광주 FC, K06=성남 FC, K07=포항 스틸러스, K08=수원 FC, K09=FC 서울, K10=대전 하나 시티즌.
포지션: MF=미드필더, FW=공격수, DF=수비수, GK=골키퍼. 한국어만 사용.
금지: "태양", "Boltzmann", "에너자이져" 단어 사용 금지. 제공된 필드와 값만 사용.

제공된 정보:
{field_list}

요약 문장 (한 문장만):"""

    return prompt


# 사용 예시 (주석)
"""
# ORM 모델로 호출 (권장: 테이블 추가 시 이 파일 수정 불필요)
from app.domains.v10.shared.models.bases.embedding_content_template import get_embedding_prompt
from app.domains.v10.soccer.models.bases.players import Player

player_data = {"player_name": "홍길동", "position": "MF", "team_code": "K06", "back_no": 10, "height": 180}
prompt = get_embedding_prompt(Player, player_data)
content = await exaone_generate(prompt, max_new_tokens=200)

# 테이블명 + metadata로 호출
from app.domains.v10.shared.models.bases.base import Base
prompt = get_embedding_prompt("players", player_data, metadata=Base.metadata)
"""
