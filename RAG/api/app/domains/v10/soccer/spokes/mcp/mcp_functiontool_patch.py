"""
MCP FunctionTool 실행 경로 패치.

fastmcp/MCP 서버가 툴 실행 시 FunctionTool 객체를 callable처럼 호출하려다
'FunctionTool' object is not callable 이 발생하는 과도기적 버그를 피하기 위해,
툴 클래스에 __call__을 추가하여 내부 실제 함수를 호출하도록 합니다.
HTTP 방식은 그대로 두고, 서버 내부 실행만 수정합니다.
fastmcp/MCP가 수정되면 이 패치 제거 가능.
"""

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _get_underlying_callable(obj: Any) -> Callable[..., Any] | None:
    """FunctionTool 등 툴 객체에서 실제 호출 가능한 함수를 찾습니다."""
    for attr in (
        "function", "fn", "_fn", "func", "_func", "callback", "_callable",
        "__wrapped__", "wrapped", "handler", "_handler",
    ):
        fn = getattr(obj, attr, None)
        if callable(fn) and not isinstance(fn, type):
            return fn
    # Pydantic/일반 __dict__ 에서 callable 필드 확인
    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            if callable(v) and not k.startswith("__") and k in (
                "function", "fn", "_fn", "func", "callback", "wrapped", "handler"
            ):
                return v
    return None


def _make_callable(tool_class: type) -> None:
    """툴 클래스에 __call__을 추가해 실제 함수를 호출하도록 합니다."""
    if not isinstance(tool_class, type):
        return
    if getattr(tool_class, "__call_patched__", False):
        return

    # 서버 쪽에서 tool(**args) 또는 await tool(**args) 형태로 부르므로
    # __call__은 실제 함수를 그대로 호출해 반환 (동기면 값, 비동기면 coroutine).
    def __call__(self: Any, *args: Any, **kwargs: Any) -> Any:
        fn = _get_underlying_callable(self)
        if fn is None:
            raise TypeError(
                f"{type(self).__name__!r} object is not callable "
                "(no underlying function found by patch)"
            )
        # 반환값을 그대로 넘김: 동기면 값, 비동기면 coroutine (await 가능)
        return fn(*args, **kwargs)

    tool_class.__call__ = __call__
    try:
        tool_class.__call_patched__ = True  # type: ignore[attr-defined]
    except Exception:
        pass
    logger.info("[MCP_PATCH] Applied __call__ to %s", tool_class.__name__)


def apply_functiontool_patch() -> None:
    """mcp / fastmcp 툴 클래스를 찾아 __call__ 패치를 적용합니다."""
    patched: list[str] = []

    # 1) mcp.types (공식 MCP Python SDK)
    try:
        from mcp.types import Tool  # type: ignore[import-not-found]

        if Tool is not None and hasattr(Tool, "__mro__"):
            for cls in Tool.__mro__:
                if cls is object:
                    continue
                name = getattr(cls, "__name__", "")
                if "Function" in name or "Tool" in name:
                    _make_callable(cls)
                    patched.append(f"mcp.types.{name}")
    except Exception as e:
        logger.debug("[MCP_PATCH] mcp.types skip: %s", e)

    # 2) fastmcp.tools
    for sub in ("tools", "tool", "function_tool"):
        try:
            mod = __import__(f"fastmcp.{sub}", fromlist=["FunctionTool", "Tool"])
            for attr in ("FunctionTool", "Tool", "ToolSpec"):
                cls = getattr(mod, attr, None)
                if isinstance(cls, type):
                    _make_callable(cls)
                    patched.append(f"fastmcp.{sub}.{attr}")
        except Exception as e:
            logger.debug("[MCP_PATCH] fastmcp.%s skip: %s", sub, e)

    # 3) fastmcp 내부에서 툴로 쓰는 클래스 이름으로 검색
    try:
        import fastmcp  # type: ignore[import-not-found]

        for name in dir(fastmcp):
            obj = getattr(fastmcp, name, None)
            if isinstance(obj, type) and "Tool" in name:
                _make_callable(obj)
                patched.append(f"fastmcp.{name}")
    except Exception as e:
        logger.debug("[MCP_PATCH] fastmcp dir skip: %s", e)

    # 4) sys.modules에서 FunctionTool 이름 클래스 검색 (mcp, fastmcp 하위)
    import sys

    for mod_name, mod in list(sys.modules.items()):
        if mod is None or (mod_name or "").startswith("_"):
            continue
        if "mcp" not in mod_name and "fastmcp" not in mod_name:
            continue
        try:
            cls = getattr(mod, "FunctionTool", None)
            if isinstance(cls, type):
                _make_callable(cls)
                patched.append(f"{mod_name}.FunctionTool")
        except Exception:
            continue

    if patched:
        logger.info("[MCP_PATCH] Patched tool classes: %s", patched)
    else:
        logger.warning("[MCP_PATCH] No tool class was patched (FunctionTool not found in mcp/fastmcp)")


# 모듈 로드 시 한 번 실행
apply_functiontool_patch()
