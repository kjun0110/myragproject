"""
EXAONE이 `schedule_embeddings.py` 파일을 직접 생성/작성하도록 하는 실행 스크립트.

실행 예시 (프로젝트 루트 RAG/ 기준):
  python -m api.app.domains.v10.soccer.exaone.write_schedule_embeddings_model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

# 프로젝트 경로 보정
current_file = Path(__file__).resolve()
app_dir = current_file.parents[4]  # api/app/
api_dir = app_dir.parent  # api/
sys.path.insert(0, str(api_dir))

from app.core.loaders import ModelLoader

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
    return s.strip()


def _clean_generated_output(raw: str) -> str:
    s = raw.strip()

    if s.startswith("---"):
        lines = s.splitlines()
        while lines and lines[0].strip() == "---":
            lines.pop(0)
        s = "\n".join(lines).lstrip()

    if "```" in s:
        start = s.find("```")
        end = s.find("```", start + 3)
        if end != -1:
            inner = s[start + 3 : end].lstrip()
            first_nl = inner.find("\n")
            if first_nl != -1:
                first_line = inner[:first_nl].strip().lower()
                if first_line in {"python", "py"}:
                    inner = inner[first_nl + 1 :]
            s = inner.strip()

    s = s.replace("```python", "").replace("```", "").strip()
    return s


def _preview(text: str, max_chars: int = 600) -> str:
    t = text.strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "\n... (truncated) ..."


def _validate_generated_python(code: str) -> Tuple[bool, str]:
    stripped = code.strip()
    if not stripped:
        return False, "출력이 비어 있습니다."
    if "```" in stripped:
        return False, "코드펜스(```)가 포함되어 있습니다."
    if stripped.startswith("---"):
        return False, "YAML frontmatter(---)가 포함되어 있습니다."
    if "class ScheduleEmbedding" not in stripped:
        return False, "class ScheduleEmbedding 정의가 없습니다."
    if "__tablename__" not in stripped or "schedules_embeddings" not in stripped:
        return False, "__tablename__ 또는 schedules_embeddings 지정이 없습니다."
    if "Vector" not in stripped:
        return False, "Vector 사용(또는 import)이 없습니다."
    if "ForeignKey" not in stripped or "schedules.id" not in stripped:
        return False, "schedules.id로의 ForeignKey가 없습니다."
    return True, "ok"


def generate_schedule_embeddings_py(
    schedules_py: str,
    dim: int,
    local_files_only: bool,
) -> str:
    template = f'''"""ScheduleEmbedding 모델 (자동 생성)

schedules 테이블을 기반으로 한 임베딩 테이블 모델입니다.
"""

from sqlalchemy import BigInteger, Column, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.types import TIMESTAMP

from pgvector.sqlalchemy import Vector

from app.domains.v10.shared.models.bases.base import Base


class ScheduleEmbedding(Base):
    __tablename__ = "schedules_embeddings"

    id = Column(BigInteger, primary_key=True, autoincrement=True, nullable=False)
    schedule_id = Column(BigInteger, ForeignKey("schedules.id", ondelete="CASCADE"), nullable=False)

    content = Column(Text, nullable=False)
    embedding = Column(Vector({dim}), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    schedule = relationship("Schedule", back_populates="embeddings")
'''

    prompt = (
        "너는 파이썬 백엔드 개발자다.\n"
        "아래 `schedules.py`의 Schedule 모델을 참고해서, pgvector 기반 임베딩 테이블 모델 파일을 작성해라.\n"
        "제약:\n"
        "- 출력은 오직 `schedule_embeddings.py` 파일의 전체 내용만.\n"
        "- 설명/마크다운/코드펜스(```) 금지.\n"
        "- 반드시 TEMPLATE를 그대로 사용하되 필요한 경우에만 최소 수정.\n"
        "\n"
        "----- schedules.py BEGIN -----\n"
        f"{schedules_py}\n"
        "----- schedules.py END -----\n"
        "\n"
        "----- TEMPLATE BEGIN -----\n"
        f"{template}\n"
        "----- TEMPLATE END -----\n"
        "\n"
        "이제 `schedule_embeddings.py` 파일 전체 내용을 출력해라.\n"
    )

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
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return _clean_generated_output(_strip_code_fences(generated))


def main() -> None:
    parser = argparse.ArgumentParser(description="EXAONE이 schedule_embeddings.py 파일을 생성/작성")
    parser.add_argument("--verbose", action="store_true", help="디버그 로그 출력")
    parser.add_argument("--dim", type=int, default=768, help="Vector 차원 (기본: 768)")
    parser.add_argument("--dry-run", action="store_true", help="파일 저장 없이 생성/검증만")
    parser.add_argument("--offline", action="store_true", help="로컬 캐시만 사용")
    parser.add_argument("--force-write", action="store_true", help="검증 실패해도 강제 덮어쓰기")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    in_path = app_dir / "domains" / "v10" / "soccer" / "models" / "bases" / "schedules.py"
    out_path = app_dir / "domains" / "v10" / "soccer" / "models" / "bases" / "schedule_embeddings.py"

    if args.offline:
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    logger.info("[EXAONE] schedule_embeddings.py 생성 중...")
    code = generate_schedule_embeddings_py(
        schedules_py=_read_text(in_path),
        dim=int(args.dim),
        local_files_only=bool(args.offline),
    )

    ok, reason = _validate_generated_python(code)
    if not ok:
        debug_path = out_path.with_suffix(".py.exaone_draft.txt")
        debug_path.write_text(code.rstrip() + "\n", encoding="utf-8")
        logger.error(f"[ERROR] 생성 결과 검증 실패: {reason} / 초안 저장: {debug_path}")
        logger.error("미리보기:\n" + _preview(code))
        if not args.force_write:
            raise RuntimeError("EXAONE 생성 결과 검증 실패 (force-write로 강제 저장 가능)")

    if args.dry_run:
        logger.info("[DRY_RUN] 생성/검증만 완료 (파일 저장 생략)")
        return

    out_path.write_text(code.rstrip() + "\n", encoding="utf-8")
    logger.info(f"[OK] 파일 작성 완료: {out_path}")


if __name__ == "__main__":
    main()

