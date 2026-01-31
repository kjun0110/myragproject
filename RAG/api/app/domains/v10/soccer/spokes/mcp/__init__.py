"""
spokes.mcp

- agents/: 도메인(업무) 로직
- mcp/: HTTP로 실행되는 MCP 서버 엔트리포인트(운영 단위)

이 폴더의 서버들은 무거운 모델을 "직접" 로드하지 않는 것을 기본 원칙으로 합니다.
예외적으로 `llm_server.py`만 ExaOne 베이스 모델을 1회 로드하여 공유합니다.
"""

from .llm_server import mcp as llm_mcp, app as llm_app

__all__ = ["llm_mcp", "llm_app"]

