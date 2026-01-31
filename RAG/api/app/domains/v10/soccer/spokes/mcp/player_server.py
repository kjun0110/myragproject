"""
Player Spoke MCP Server

원칙:
- 이 서버는 ExaOne 같은 무거운 모델을 직접 로드하지 않습니다.
- LLM이 필요하면 `SOCCER_LLM_MCP_URL`로 LLM 전용 서버에 call_tool 위임합니다.

HTTP 실행 예시:
    cd api
    python -m uvicorn app.domains.v10.soccer.spokes.mcp.player_server:app --host 0.0.0.0 --port 9001

클라이언트 URL:
    http://127.0.0.1:9001/mcp
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from fastmcp.client import Client
from fastmcp.server.http import create_streamable_http_app

logger = logging.getLogger(__name__)

MCP_PATH = os.getenv("SOCCER_PLAYER_MCP_PATH", "/mcp")
LLM_URL = os.getenv("SOCCER_LLM_MCP_URL", "http://127.0.0.1:9100/mcp")

mcp = FastMCP("Soccer Player Spoke MCP Server")


@mcp.tool
async def player_policy_process(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Player 정책 처리(업무 로직) 실행."""
    from app.domains.v10.soccer.spokes.agents.player_agent import PlayerAgent

    agent = PlayerAgent()
    return await agent.process(records)


@mcp.tool
async def player_exaone_generate(prompt: str, max_new_tokens: int = 256) -> str:
    """LLM 서버에 exaone_generate를 위임합니다."""
    async with Client(LLM_URL) as c:
        result = await c.call_tool(
            "exaone_generate",
            {"prompt": prompt, "max_new_tokens": max_new_tokens},
        )
        return "" if result.data is None else str(result.data)


@mcp.tool
async def player_exaone_generate_with_fs_tools(prompt: str, max_steps: int = 5) -> str:
    """LLM 서버에 exaone_generate_with_fs_tools를 위임합니다."""
    async with Client(LLM_URL) as c:
        result = await c.call_tool(
            "exaone_generate_with_fs_tools",
            {"prompt": prompt, "max_steps": max_steps},
        )
        return "" if result.data is None else str(result.data)


@mcp.tool
def player_server_health() -> str:
    return json.dumps({"status": "ok", "llm_url": LLM_URL}, ensure_ascii=False)


app = create_streamable_http_app(server=mcp, streamable_http_path=MCP_PATH)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9001"))
    uvicorn.run(
        "app.domains.v10.soccer.spokes.mcp.player_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )

