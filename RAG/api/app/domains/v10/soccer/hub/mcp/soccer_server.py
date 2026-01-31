"""
Soccer MCP Server (Central MCP)

목표: 중앙 MCP 서버는 "라우팅만" 담당합니다.
- ExaOne/KoELECTRA 같은 무거운 모델을 중앙에서 로드/추론하지 않습니다.
- 질문을 받아서 어느 오케스트레이터(player/schedule/stadium/team)로 보낼지 결정만 합니다.

향후 확장:
- 실제 MCP 프로토콜(HTTP/stdio) 기반으로 spoke(각 도메인 서버)로 call_tool 프록시를 붙일 수 있습니다.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Literal

OrchestratorName = Literal["player", "schedule", "stadium", "team"]


@dataclass(frozen=True)
class RouteResult:
    orchestrator: OrchestratorName
    routed_by: str = "keywords"

    def to_dict(self) -> Dict[str, Any]:
        return {"orchestrator": self.orchestrator, "routed_by": self.routed_by}


class SoccerMCPServer:
    """중앙 MCP 서버.

    - 중앙에서 모델을 로드하지 않도록, 라우팅은 키워드 기반(경량)으로만 수행합니다.
    """

    def route(self, question: str) -> RouteResult:
        q = (question or "").lower()

        # 기존 ChatOrchestrator의 키워드 라우팅 로직과 같은 의도(경량 라우팅)
        player_keywords = [
            "선수",
            "player",
            "이름",
            "나이",
            "국적",
            "포지션",
            "키",
            "몸무게",
            "출생",
            "생년",
            "생일",
        ]
        schedule_keywords = ["일정", "schedule", "경기", "매치", "vs", "대전", "날짜", "시간"]
        stadium_keywords = ["경기장", "stadium", "구장", "아레나", "장소"]
        team_keywords = ["팀", "team", "클럽", "구단", "코드"]

        scores = {
            "player": sum(1 for k in player_keywords if k in q),
            "schedule": sum(1 for k in schedule_keywords if k in q),
            "stadium": sum(1 for k in stadium_keywords if k in q),
            "team": sum(1 for k in team_keywords if k in q),
        }

        max_score = max(scores.values()) if scores else 0
        if max_score == 0:
            selected: OrchestratorName = "player"
        else:
            selected = max(scores.keys(), key=lambda k: scores[k])  # type: ignore[assignment]

        return RouteResult(orchestrator=selected, routed_by="keywords")

    def build_mcp(self) -> Any:
        from fastmcp import FastMCP
        from fastmcp.client import Client

        mcp = FastMCP("Soccer MCP Server (Routing + Spoke Proxy)")
        server = self

        spoke_urls: Dict[str, str] = {
            "player": os.getenv("SOCCER_PLAYER_SPOKE_MCP_URL", "http://127.0.0.1:9001/mcp"),
            "schedule": os.getenv("SOCCER_SCHEDULE_SPOKE_MCP_URL", "http://127.0.0.1:9002/mcp"),
            "stadium": os.getenv("SOCCER_STADIUM_SPOKE_MCP_URL", "http://127.0.0.1:9003/mcp"),
            "team": os.getenv("SOCCER_TEAM_SPOKE_MCP_URL", "http://127.0.0.1:9004/mcp"),
        }

        @mcp.tool
        def soccer_route(question: str) -> str:
            """질문을 어느 오케스트레이터로 보낼지 결정합니다(라우팅만).

            Returns:
                JSON 문자열: {"orchestrator": "...", "routed_by": "keywords"}
            """

            result = server.route(question)
            return json.dumps(result.to_dict(), ensure_ascii=False)

        @mcp.tool
        async def soccer_call(orchestrator: OrchestratorName, tool: str, arguments: Dict[str, Any] | None = None) -> Any:
            """중앙 → spoke 로 `call_tool`을 프록시합니다.

            중앙은 무거운 추론을 하지 않고, spoke 서버만 호출합니다.
            """
            url = spoke_urls.get(orchestrator)
            if not url:
                return {"error": f"알 수 없는 orchestrator: {orchestrator}"}
            async with Client(url) as c:
                result = await c.call_tool(tool, arguments or {})
                if result.data is not None:
                    return result.data
                # data가 없으면 content를 그대로 반환(디버그용)
                return {"content": [getattr(x, "text", str(x)) for x in result.content], "is_error": result.is_error}

        @mcp.tool
        async def soccer_route_and_call(question: str, tool: str, arguments: Dict[str, Any] | None = None) -> Any:
            """라우팅 후 중앙 → spoke call_tool 프록시까지 한 번에 수행합니다."""
            rr = server.route(question)
            return await soccer_call(rr.orchestrator, tool, arguments)

        @mcp.tool
        def server_health() -> str:
            """중앙 MCP 서버 헬스체크."""
            return json.dumps(
                {"status": "ok", "role": "routing_and_spoke_proxy", "spokes": spoke_urls},
                ensure_ascii=False,
            )

        return mcp


def get_soccer_mcp_server() -> Any:
    """중앙 MCP 서버 인스턴스를 반환합니다."""
    return SoccerMCPServer().build_mcp()

