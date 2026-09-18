"""Reflex Framework Integrations.

Native, drop-in adapters and middleware for production AI stacks:
- mcp: Fail-closed Model Context Protocol (MCP) JSON-RPC safety proxy (< 1ms).
- fastapi: ASGI gateway middleware for sub-millisecond local routing & frontier escalation.
- langchain: Agent tool interceptor and callback handler with Ed25519 witness receipts.
"""

from __future__ import annotations

from system1.integrations.mcp import (
    ReflexMCPBlockedError,
    ReflexMCPProxy,
    wrap_mcp_tool,
)
from system1.integrations.fastapi import (
    ReflexGatewayMiddleware,
    add_reflex_gateway,
)
from system1.integrations.langchain import (
    ReflexGuardBlockedException,
    ReflexGuardCallbackHandler,
    ReflexToolInterceptor,
    wrap_langchain_tool,
)

__all__ = [
    # MCP
    "ReflexMCPBlockedError",
    "ReflexMCPProxy",
    "wrap_mcp_tool",
    # FastAPI / ASGI
    "ReflexGatewayMiddleware",
    "add_reflex_gateway",
    # LangChain / Agent
    "ReflexGuardBlockedException",
    "ReflexGuardCallbackHandler",
    "ReflexToolInterceptor",
    "wrap_langchain_tool",
]
