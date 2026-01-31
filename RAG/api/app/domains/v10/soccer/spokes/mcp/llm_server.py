"""
LLM 전용 MCP 서버 (ExaOne)

목표:
- ExaOne 베이스 모델을 "이 서버에서만" 1회 로드해서 공유합니다.
- 다른 도메인(spoke) 서버들은 이 서버에 `call_tool`로 위임하여 중복 로드를 막습니다.

HTTP 실행 예시(권장: Streamable HTTP):
    cd api
    python -m uvicorn app.domains.v10.soccer.spokes.mcp.llm_server:app --host 0.0.0.0 --port 9100

클라이언트는 아래 URL로 접속합니다:
    http://127.0.0.1:9100/mcp
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from fastmcp.server.http import create_streamable_http_app

logger = logging.getLogger(__name__)

# HTTP 엔드포인트 경로 (Client는 URL에 이 경로를 포함해야 함)
MCP_PATH = os.getenv("SOCCER_LLM_MCP_PATH", "/mcp")


class ExaOneRuntime:
    """ExaOne 모델/토크나이저를 1회 로드해서 공유."""

    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None

    def ensure_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        try:
            logger.info("[LLM MCP] ExaOne 모델 로드 시작...")
            from app.core.loaders import ModelLoader

            self.model, self.tokenizer = ModelLoader.load_exaone_model(
                adapter_name=None,
                use_quantization=True,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("[LLM MCP] ExaOne 모델 로드 완료")
        except Exception as e:
            logger.exception("[LLM MCP] ExaOne 모델 로드 실패: %s", e)
            self.model = None
            self.tokenizer = None

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """ExaOne 텍스트 생성."""
        self.ensure_loaded()
        if self.model is None or self.tokenizer is None:
            return ""
        try:
            import torch

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            if hasattr(self.model, "device"):
                device = self.model.device
            else:
                device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            return generated.strip()
        except Exception as e:
            logger.exception("[LLM MCP] ExaOne 생성 실패: %s", e)
            return ""


runtime = ExaOneRuntime()


def _safe_root() -> Path:
    """파일/경로 도구의 접근 허용 루트 디렉터리(api/)를 반환합니다."""
    current_file = Path(__file__).resolve()
    # llm_server.py 위치: api/app/domains/v10/soccer/spokes/mcp/llm_server.py
    api_dir = current_file.parent.parent.parent.parent.parent.parent.parent  # api/
    return api_dir


def _resolve_safe_path(path_str: str) -> Path:
    """사용자 입력 경로를 안전 루트 내부로만 해석합니다."""
    root = _safe_root()
    raw = Path(path_str)
    candidate = raw if raw.is_absolute() else (root / raw)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except Exception as e:
        raise ValueError(f"허용되지 않은 경로입니다 (api/ 밖 접근 금지): {resolved}") from e
    return resolved


def _fs_tools() -> Dict[str, Any]:
    """os/pathlib 기반 도구 레지스트리."""

    def path_exists(path: str) -> bool:
        p = _resolve_safe_path(path)
        return p.exists()

    def list_dir(path: str) -> List[str]:
        p = _resolve_safe_path(path)
        if not p.exists():
            raise FileNotFoundError(f"경로가 존재하지 않습니다: {p}")
        if not p.is_dir():
            raise NotADirectoryError(f"디렉터리가 아닙니다: {p}")
        return sorted([x.name for x in p.iterdir()])

    def read_text(path: str, max_chars: int = 5000) -> str:
        p = _resolve_safe_path(path)
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


def _generate_with_fs_tools(user_request: str, max_steps: int = 5) -> Dict[str, Any]:
    """ExaOne이 os/pathlib 도구를 호출하며 답하도록 실행합니다.

    프롬프트 기반(LLM이 JSON으로 tool call을 출력) 간단 구현입니다.
    """
    tools = _fs_tools()
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
        raw = runtime.generate(transcript, max_new_tokens=256)
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


mcp = FastMCP("Soccer LLM MCP Server (ExaOne Only)")


@mcp.tool
def exaone_generate(prompt: str, max_new_tokens: int = 256) -> str:
    """ExaOne로 텍스트를 생성합니다(LLM 전용 서버)."""
    return runtime.generate(prompt, max_new_tokens=max_new_tokens)


@mcp.tool
def exaone_generate_with_fs_tools(prompt: str, max_steps: int = 5) -> str:
    """ExaOne이 파일/경로 도구(os/pathlib)를 사용해 답하도록 실행합니다(LLM 전용 서버)."""
    result = _generate_with_fs_tools(prompt, max_steps=max_steps)
    return result.get("final", "")


@mcp.tool
def llm_server_health() -> str:
    """LLM 서버 헬스체크 (모델 로드 여부 포함)."""
    loaded = runtime.model is not None and runtime.tokenizer is not None
    return json.dumps({"status": "ok", "model_loaded": loaded}, ensure_ascii=False)


# ASGI app (uvicorn으로 실행)
app = create_streamable_http_app(server=mcp, streamable_http_path=MCP_PATH)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9100"))
    uvicorn.run(
        "app.domains.v10.soccer.spokes.mcp.llm_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )

