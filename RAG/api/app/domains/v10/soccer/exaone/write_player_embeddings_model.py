"""
EXAONE이 `player_embeddings.py` 파일을 직접 생성/작성하도록 하는 실행 스크립트.

요구사항:
- `players.py`(Player 테이블 정의)를 참고해서
- EXAONE이 `player_embeddings.py`(PlayerEmbedding 모델 파일) 내용을 생성
- 본 스크립트는 EXAONE 출력물을 받아 해당 파일에 저장(덮어쓰기)만 수행

실행 예시 (프로젝트 루트 RAG/ 기준):
  python -m api.app.domains.v10.soccer.exaone.write_player_embeddings_model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

# 프로젝트 경로 보정
current_file = Path(__file__).resolve()
app_dir = current_file.parents[4]  # api/app/
api_dir = app_dir.parent  # api/
sys.path.insert(0, str(api_dir))

from app.common.loaders import ModelLoader

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _strip_code_fences(s: str) -> str:
    text = s.strip()
    # NOTE:
    # - EXAONE 출력이 "---" (yaml frontmatter)로 시작하거나
    # - 코드펜스가 "중간"에 포함되는 경우가 있어서
    #   단순 startswith("```")만으로는 충분하지 않습니다.
    return text.strip()


def _clean_generated_output(raw: str) -> str:
    """
    EXAONE 출력에서 "순수 파이썬 코드"만 남기도록 정리합니다.

    처리:
    - YAML frontmatter(---)가 선두에 있으면 제거
    - ```python ... ``` 또는 ``` ... ``` 블록이 있으면 첫 블록의 내부만 추출
    - 남아있는 ``` 토큰이 있으면 제거(마지막 안전장치)
    """
    s = raw.strip()

    # 1) YAML frontmatter 선두 제거 (예: "---\n\n```python\n...")
    if s.startswith("---"):
        lines = s.splitlines()
        # 선두의 연속된 '---' 라인만 제거
        while lines and lines[0].strip() == "---":
            lines.pop(0)
        s = "\n".join(lines).lstrip()

    # 2) 코드펜스가 어디든 있으면, 첫 fenced block 내부만 추출
    if "```" in s:
        start = s.find("```")
        end = s.find("```", start + 3)
        if end != -1:
            inner = s[start + 3 : end]
            inner = inner.lstrip()
            # 첫 줄이 언어 태그(예: "python")이면 제거
            first_nl = inner.find("\n")
            if first_nl != -1:
                first_line = inner[:first_nl].strip().lower()
                if first_line in {"python", "py"}:
                    inner = inner[first_nl + 1 :]
            s = inner.strip()

    # 3) 남아있는 fence 토큰 제거 (최후의 안전장치)
    s = s.replace("```python", "").replace("```", "").strip()
    return s

def _preview(text: str, max_chars: int = 600) -> str:
    t = text.strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "\n... (truncated) ..."


def _validate_generated_python(code: str) -> Tuple[bool, str]:
    """
    너무 빡빡한 문자열 매칭 대신, 최소한의 구조가 있는지 가볍게 검증합니다.
    (EXAONE 출력이 약간 다른 스타일이어도 통과할 수 있게)
    """
    stripped = code.strip()
    if not stripped:
        return False, "출력이 비어 있습니다."
    if "```" in stripped:
        return False, "코드펜스(```)가 포함되어 있습니다."
    if stripped.startswith("---"):
        return False, "YAML frontmatter(---)가 포함되어 있습니다."
    if "class PlayerEmbedding" not in stripped:
        return False, "class PlayerEmbedding 정의가 없습니다."
    if "__tablename__" not in stripped or "players_embeddings" not in stripped:
        return False, "__tablename__ 또는 players_embeddings 지정이 없습니다."
    if "Vector" not in stripped:
        return False, "Vector 사용(또는 import)이 없습니다."
    if "ForeignKey" not in stripped or "players.id" not in stripped:
        return False, "players.id로의 ForeignKey가 없습니다."
    return True, "ok"


def generate_player_embeddings_py(
    players_py: str,
    dim: int,
    table_name: str,
    model_class_name: str,
    local_files_only: bool,
) -> str:
    """
    EXAONE에게 '파일 전체' 파이썬 코드만 출력하게 시킨다.
    """
    # EXAONE 출력 안정성을 위해 "템플릿을 그대로 채우게" 강제
    template = f'''"""PlayerEmbedding 모델 (자동 생성)

players 테이블을 기반으로 한 임베딩 테이블 모델입니다.
"""

from sqlalchemy import BigInteger, Column, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.types import TIMESTAMP

from pgvector.sqlalchemy import Vector

from app.domains.v10.shared.models.bases.base import Base


class {model_class_name}(Base):
    __tablename__ = "{table_name}"

    id = Column(BigInteger, primary_key=True, autoincrement=True, nullable=False)
    player_id = Column(BigInteger, ForeignKey("players.id", ondelete="CASCADE"), nullable=False)

    content = Column(Text, nullable=False)
    embedding = Column(Vector({dim}), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    player = relationship("Player", back_populates="embeddings")
'''

    prompt = (
        "너는 파이썬 백엔드 개발자다.\n"
        "아래 `players.py`의 Player 모델을 참고해서, pgvector 기반 임베딩 테이블 모델 파일을 작성해라.\n"
        "\n"
        "제약 조건:\n"
        "- 출력은 오직 `player_embeddings.py` 파일의 '전체 내용'만 출력한다.\n"
        "- 설명/마크다운/코드펜스(``` ) 금지. 파일 내용만.\n"
        "- 반드시 아래 템플릿을 그대로 사용하되, 필요한 경우에만 최소 수정한다.\n"
        "- 특히 import 경로, tablename, FK, Vector 차원은 유지한다.\n"
        "- SQLAlchemy ORM 스타일로 작성한다.\n"
        f"- 테이블명은 `{table_name}`\n"
        f"- 모델 클래스명은 `{model_class_name}`\n"
        "- 컬럼은 아래 스펙을 반드시 만족:\n"
        "  - id: BigInteger PK, autoincrement, not null\n"
        "  - player_id: BigInteger FK -> players.id, ondelete CASCADE, not null\n"
        "  - content: Text, not null\n"
        f"  - embedding: Vector({dim}), not null\n"
        "  - created_at: timezone-aware timestamp, server_default now(), not null\n"
        "- relationship: Player.embeddings <-> PlayerEmbedding.player back_populates 로 연결\n"
        "\n"
        "아래는 참고용 players.py 내용이다:\n"
        "----- players.py BEGIN -----\n"
        f"{players_py}\n"
        "----- players.py END -----\n"
        "\n"
        "----- TEMPLATE BEGIN -----\n"
        f"{template}\n"
        "----- TEMPLATE END -----\n"
        "\n"
        "이제 `player_embeddings.py` 파일 전체 내용을 출력해라.\n"
    )

    # 모델 로드 (generate용)
    model, tokenizer = ModelLoader.load_exaone_model(
        adapter_name=None,
        use_quantization=True,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model.eval()

    import torch  # type: ignore

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    if hasattr(model, "device"):
        device = model.device
    else:
        device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1600,
            do_sample=False,  # 파일 생성은 결정적으로(일관성) 생성
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return _clean_generated_output(_strip_code_fences(generated))


def main() -> None:
    parser = argparse.ArgumentParser(description="EXAONE이 player_embeddings.py 파일을 생성/작성")
    parser.add_argument("--verbose", action="store_true", help="디버그 로그 출력")
    parser.add_argument(
        "--dim",
        type=int,
        default=768,
        help="Vector 차원 (기본: 768)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일에 저장하지 않고 생성 결과만 검증",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="HuggingFace 네트워크 접근 없이 로컬 캐시만 사용(캐시가 없으면 실패)",
    )
    parser.add_argument(
        "--force-write",
        action="store_true",
        help="검증에 실패해도 player_embeddings.py를 덮어쓰기",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    players_path = app_dir / "domains" / "v10" / "soccer" / "models" / "bases" / "players.py"
    out_path = app_dir / "domains" / "v10" / "soccer" / "models" / "bases" / "player_embeddings.py"

    players_py = _read_text(players_path)

    if args.offline:
        # HuggingFace 라이브러리들이 네트워크 호출을 최소화하도록 힌트 제공
        # (이미 캐시되어 있어야 성공)
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    logger.info("[EXAONE] player_embeddings.py 생성 중...")
    code = generate_player_embeddings_py(
        players_py=players_py,
        dim=int(args.dim),
        table_name="players_embeddings",
        model_class_name="PlayerEmbedding",
        local_files_only=bool(args.offline),
    )

    ok, reason = _validate_generated_python(code)
    if not ok:
        # 생성 결과를 옆에 남겨서 디버깅 가능하게
        debug_path = out_path.with_suffix(".py.exaone_draft.txt")
        debug_path.write_text(code.rstrip() + "\n", encoding="utf-8")
        logger.error(
            "[ERROR] EXAONE 생성 결과가 기대 형식이 아닙니다. "
            f"사유: {reason} / 초안 저장: {debug_path}"
        )
        logger.error("생성 결과 미리보기:\n" + _preview(code))
        if not args.force_write:
            raise RuntimeError("EXAONE 생성 결과 검증 실패 (force-write로 강제 저장 가능)")

    if args.dry_run:
        logger.info("[DRY_RUN] 생성/검증만 완료 (파일 저장 생략)")
        return

    out_path.write_text(code.rstrip() + "\n", encoding="utf-8")
    logger.info(f"[OK] 파일 작성 완료: {out_path}")


if __name__ == "__main__":
    main()

