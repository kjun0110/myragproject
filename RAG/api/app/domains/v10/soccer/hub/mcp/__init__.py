"""
hub.mcp

중앙 MCP(게이트웨이/프록시) 관련 코드를 hub 아래에서 관리합니다.

현재 구현은 "얇은 프록시(thin proxy)" 전략의 첫 단계로,
중앙 MCP 서버가 무거운 모델 추론을 하지 않고(ExaOne/KoELECTRA 로드 X)
질문을 어느 도메인 오케스트레이터로 보낼지 라우팅 결과만 제공하도록 구성합니다.
"""

from .soccer_server import SoccerMCPServer, get_soccer_mcp_server

__all__ = ["SoccerMCPServer", "get_soccer_mcp_server"]

