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

# FunctionTool 실행 경로 패치 (fastmcp 과도기적 버그 우회). fastmcp 로드 직후 적용.
from app.domains.v10.soccer.spokes.mcp import mcp_functiontool_patch  # noqa: F401

logger = logging.getLogger(__name__)

MCP_PATH = os.getenv("SOCCER_PLAYER_MCP_PATH", "/mcp")
LLM_URL = os.getenv("SOCCER_LLM_MCP_URL", "http://127.0.0.1:9100/mcp")
# 9100 첫 호출 시 모델 로딩 1~2분 걸릴 수 있음
LLM_CALL_TIMEOUT = int(os.getenv("SOCCER_LLM_CALL_TIMEOUT", "180"))

mcp = FastMCP("Soccer Player Spoke MCP Server")


@mcp.tool
async def player_policy_process(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Player 정책 처리(업무 로직) 실행."""
    from app.domains.v10.soccer.spokes.agents.player_agent import PlayerAgent

    agent = PlayerAgent()
    return await agent.process(records)


@mcp.tool
async def player_exaone_generate(prompt: str, max_new_tokens: int = 256) -> str:
    """LLM 서버(9100)에 exaone_generate를 위임합니다."""
    logger.info("[Player MCP] 9100 호출 시작: LLM_URL=%s timeout=%s", LLM_URL, LLM_CALL_TIMEOUT)
    try:
        async with Client(LLM_URL, timeout=LLM_CALL_TIMEOUT) as c:
            result = await c.call_tool(
                "exaone_generate",
                {"prompt": prompt, "max_new_tokens": max_new_tokens},
                timeout=LLM_CALL_TIMEOUT,
            )
            logger.info("[Player MCP] 9100 호출 완료: result_data=%s", "있음" if result.data else "없음")
            return "" if result.data is None else str(result.data)
    except Exception as e:
        logger.exception("[Player MCP] 9100 호출 실패: %s", e)
        raise


@mcp.tool
async def player_exaone_generate_with_fs_tools(prompt: str, max_steps: int = 5) -> str:
    """LLM 서버에 exaone_generate_with_fs_tools를 위임합니다."""
    async with Client(LLM_URL, timeout=LLM_CALL_TIMEOUT) as c:
        result = await c.call_tool(
            "exaone_generate_with_fs_tools",
            {"prompt": prompt, "max_steps": max_steps},
            timeout=LLM_CALL_TIMEOUT,
        )
        return "" if result.data is None else str(result.data)


@mcp.tool
async def player_generate_embedding_content(player_data: Dict[str, Any]) -> str:
    """엑사원을 사용하여 선수 정보를 RAG 검색용 content로 생성합니다.

    Args:
        player_data: 선수 정보 딕셔너리 (player_name, position, team_code 등)

    Returns:
        엑사원이 생성한 검색용 content 문장
    """
    logger.info("[Player MCP] player_generate_embedding_content 호출됨")
    parts = []
    for k, v in (player_data or {}).items():
        if v is not None and str(v).strip():
            parts.append(f"{k}: {v}")
    raw = " | ".join(parts) if parts else str(player_data)
    prompt = f"""다음 축구 선수 정보를 RAG 검색에 적합한 한 문장으로 요약하세요.
검색 시 의미를 담은 자연어 문장으로 작성해주세요.

선수 정보:
{raw}

요약 문장:"""
    return await player_exaone_generate(prompt, max_new_tokens=256)


@mcp.tool
def player_server_health() -> str:
    return json.dumps({"status": "ok", "llm_url": LLM_URL}, ensure_ascii=False)


app = create_streamable_http_app(server=mcp, streamable_http_path=MCP_PATH)

# Startup 로그 (FastMCP lifespan을 건드리지 않고 바로 출력)
logger.info("=" * 70)
logger.info("[Player MCP 9001] ✓ 서버 준비")
logger.info("[Player MCP 9001] LLM 서버: %s (타임아웃: %ss)", LLM_URL, LLM_CALL_TIMEOUT)
logger.info("[Player MCP 9001] 엔드포인트: http://0.0.0.0:9001%s", MCP_PATH)
logger.info("=" * 70)


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

