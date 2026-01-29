"""
Stadium Agent - 정책 기반 처리

AI/ML 기반 복잡한 비즈니스 로직을 처리합니다.
ExaOne LLM을 로드하여 생성/분석에 사용합니다.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class StadiumAgent:
    """Stadium 데이터를 정책 기반으로 처리하는 Agent."""

    def __init__(self):
        self.exaone_model = None
        self.exaone_tokenizer = None
        self._load_exaone()
        self.mcp = self._build_mcp()
        logger.info("[AGENT] StadiumAgent 초기화")

    def _load_exaone(self) -> None:
        """ExaOne 모델을 로드합니다."""
        try:
            logger.info("[AGENT] ExaOne 모델 로드 시작...")
            from app.common.loaders import ModelLoader

            self.exaone_model, self.exaone_tokenizer = ModelLoader.load_exaone_model(
                adapter_name=None,
                use_quantization=True,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("[AGENT] ExaOne 모델 로드 완료")
        except Exception as e:
            logger.warning(f"[AGENT] ExaOne 모델 로드 실패 (정책 처리 시 미사용): {e}")
            self.exaone_model = None
            self.exaone_tokenizer = None

    def _build_mcp(self) -> Any:
        """FastMCP 서버를 생성하고 ExaOne MCP 툴만 등록합니다 (에이전트 단독용)."""
        from fastmcp import FastMCP

        mcp = FastMCP("Stadium Agent ExaOne")
        agent = self

        @mcp.tool
        def exaone_generate(prompt: str, max_new_tokens: int = 256) -> str:
            """ExaOne로 프롬프트에 대한 텍스트를 생성합니다."""
            return agent.generate(prompt, max_new_tokens=max_new_tokens)

        @mcp.tool
        def exaone_generate_with_fs_tools(prompt: str, max_steps: int = 5) -> str:
            """ExaOne이 파일/경로 도구(os/pathlib)를 사용해 답하도록 실행합니다."""
            result = agent.generate_with_fs_tools(prompt, max_steps=max_steps)
            return result.get("final", "")

        return mcp

    def get_mcp(self) -> Any:
        """FastMCP 서버를 반환합니다. ExaOne 툴이 연결되어 있습니다."""
        return self.mcp

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """ExaOne로 프롬프트에 대한 텍스트를 생성합니다."""
        if self.exaone_model is None or self.exaone_tokenizer is None:
            return ""
        try:
            import torch

            inputs = self.exaone_tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            if hasattr(self.exaone_model, "device"):
                device = self.exaone_model.device
            else:
                device = next(self.exaone_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.exaone_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.exaone_tokenizer.pad_token_id
                    or self.exaone_tokenizer.eos_token_id,
                )
            generated = self.exaone_tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            return generated.strip()
        except Exception as e:
            logger.error(f"[AGENT] ExaOne 생성 실패: {e}", exc_info=True)
            return ""

    # =========================
    # ExaOne + (os/pathlib) Tools
    # =========================
    def _safe_root(self) -> Path:
        """파일/경로 도구의 접근 허용 루트 디렉터리(api/)를 반환합니다."""
        current_file = Path(__file__).resolve()
        # stadium_agent.py 위치: api/app/domains/v10/soccer/spokes/agents/stadium_agent.py
        api_dir = current_file.parent.parent.parent.parent.parent.parent.parent  # api/
        return api_dir

    def _resolve_safe_path(self, path_str: str) -> Path:
        """사용자 입력 경로를 안전 루트 내부로만 해석합니다."""
        root = self._safe_root()
        raw = Path(path_str)
        candidate = raw if raw.is_absolute() else (root / raw)
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except Exception as e:
            raise ValueError(f"허용되지 않은 경로입니다 (api/ 밖 접근 금지): {resolved}") from e
        return resolved

    def _fs_tools(self) -> Dict[str, Any]:
        """os/pathlib 기반 도구 레지스트리."""

        def path_exists(path: str) -> bool:
            p = self._resolve_safe_path(path)
            return p.exists()

        def list_dir(path: str) -> List[str]:
            p = self._resolve_safe_path(path)
            if not p.exists():
                raise FileNotFoundError(f"경로가 존재하지 않습니다: {p}")
            if not p.is_dir():
                raise NotADirectoryError(f"디렉터리가 아닙니다: {p}")
            return sorted([x.name for x in p.iterdir()])

        def read_text(path: str, max_chars: int = 5000) -> str:
            p = self._resolve_safe_path(path)
            if not p.exists() or not p.is_file():
                raise FileNotFoundError(f"파일이 존재하지 않습니다: {p}")
            data = p.read_text(encoding="utf-8", errors="replace")
            return data[:max_chars]

        def join_path(*parts: str) -> str:
            return str(Path(*parts))

        def cwd() -> str:
            return os.getcwd()

        return {
            "path_exists": path_exists,
            "list_dir": list_dir,
            "read_text": read_text,
            "join_path": join_path,
            "cwd": cwd,
        }

    def generate_with_fs_tools(self, user_request: str, max_steps: int = 5) -> Dict[str, Any]:
        """ExaOne이 os/pathlib 도구를 호출하며 답하도록 실행합니다."""
        tools = self._fs_tools()
        tool_names = ", ".join(sorted(tools.keys()))
        steps: List[Dict[str, Any]] = []

        system = (
            "너는 파이썬 도구를 호출할 수 있는 어시스턴트다.\n"
            f"사용 가능한 도구: {tool_names}\n\n"
            "반드시 아래 JSON 중 하나로만 출력해라.\n"
            "1) 도구 호출:\n"
            '{"tool": "<tool_name>", "args": { ... }}\n'
            "2) 최종 답변:\n"
            '{"final": "<answer>"}\n\n'
            "도구 호출이 필요하면 1)로 출력하고, 도구 결과를 본 뒤 다시 판단해라.\n"
            "절대 JSON 밖의 텍스트를 출력하지 마라.\n"
        )

        transcript = f"[SYSTEM]\n{system}\n\n[USER]\n{user_request}\n"

        for _ in range(max_steps):
            raw = self.generate(transcript, max_new_tokens=256)
            raw_json = raw.strip()
            start = raw_json.find("{")
            end = raw_json.rfind("}")
            if start != -1 and end != -1 and end > start:
                raw_json = raw_json[start : end + 1]

            try:
                payload = json.loads(raw_json)
            except Exception:
                return {
                    "final": f"[ERROR] 모델 출력 JSON 파싱 실패: {raw.strip()}",
                    "steps": steps,
                }

            if isinstance(payload, dict) and "final" in payload:
                return {"final": str(payload.get("final", "")), "steps": steps}

            if not isinstance(payload, dict) or "tool" not in payload:
                return {"final": f"[ERROR] 잘못된 응답 형식: {payload}", "steps": steps}

            tool = str(payload.get("tool"))
            args = payload.get("args", {})
            if tool not in tools:
                return {"final": f"[ERROR] 알 수 없는 tool: {tool}", "steps": steps}
            if not isinstance(args, dict):
                return {"final": f"[ERROR] args는 dict여야 합니다: {args}", "steps": steps}

            try:
                result = tools[tool](**args)
            except Exception as e:
                result = {"error": str(e)}

            steps.append({"tool": tool, "args": args, "result": result})
            transcript += (
                f"\n[TOOL]\nname={tool}\nargs={json.dumps(args, ensure_ascii=False)}\n"
                f"result={json.dumps(result, ensure_ascii=False)}\n"
            )
            transcript += "\n[USER]\n도구 결과를 반영해서 계속 진행해.\n"

        return {"final": "[ERROR] max_steps 초과로 종료", "steps": steps}

    async def process(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Stadium 레코드들을 정책 기반으로 처리합니다.

        Args:
            records: 처리할 Stadium 레코드 리스트

        Returns:
            처리 결과 딕셔너리
        """
        logger.info(f"[AGENT] 정책 기반 처리 시작: {len(records)}개 레코드")

        processed_records = []
        for record in records:
            processed_record = {
                **record,
                "processed_by": "policy_based_agent",
                "processing_type": "ai_ml_based",
            }
            processed_records.append(processed_record)

        result = {
            "success": True,
            "message": "정책 기반 처리 완료",
            "total_records": len(records),
            "processed_records": len(processed_records),
            "data": processed_records,
        }

        logger.info(f"[AGENT] 정책 기반 처리 완료: {len(processed_records)}개 레코드 처리")

        return result
