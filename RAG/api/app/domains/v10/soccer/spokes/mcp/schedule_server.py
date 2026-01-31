"""
Schedule Spoke MCP Server

- 무거운 LLM 로드는 하지 않음
- 필요 시 LLM 전용 MCP 서버로 call_tool 위임
"""

import json
import os
from typing import Any, Dict, List

from fastmcp import FastMCP
from fastmcp.client import Client
from fastmcp.server.http import create_streamable_http_app

MCP_PATH = os.getenv("SOCCER_SCHEDULE_MCP_PATH", "/mcp")
LLM_URL = os.getenv("SOCCER_LLM_MCP_URL", "http://127.0.0.1:9100/mcp")

mcp = FastMCP("Soccer Schedule Spoke MCP Server")


@mcp.tool
async def schedule_policy_process(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    from app.domains.v10.soccer.spokes.agents.schedule_agent import ScheduleAgent

    agent = ScheduleAgent()
    return await agent.process(records)


@mcp.tool
async def schedule_exaone_generate(prompt: str, max_new_tokens: int = 256) -> str:
    async with Client(LLM_URL) as c:
        result = await c.call_tool(
            "exaone_generate",
            {"prompt": prompt, "max_new_tokens": max_new_tokens},
        )
        return "" if result.data is None else str(result.data)


@mcp.tool
async def schedule_exaone_generate_with_fs_tools(prompt: str, max_steps: int = 5) -> str:
    async with Client(LLM_URL) as c:
        result = await c.call_tool(
            "exaone_generate_with_fs_tools",
            {"prompt": prompt, "max_steps": max_steps},
        )
        return "" if result.data is None else str(result.data)


@mcp.tool
def schedule_server_health() -> str:
    return json.dumps({"status": "ok", "llm_url": LLM_URL}, ensure_ascii=False)


app = create_streamable_http_app(server=mcp, streamable_http_path=MCP_PATH)

