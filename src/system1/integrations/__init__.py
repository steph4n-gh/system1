"""Reflex Framework Integrations.

Native, drop-in adapters and middleware for production AI stacks:
- mcp: Fail-closed Model Context Protocol (MCP) JSON-RPC safety proxy (< 1ms).
- fastapi: ASGI gateway middleware for sub-millisecond local routing & frontier escalation.
- langchain: Agent tool interceptor and callback handler with Ed25519 witness receipts.
- observability: Prometheus-compatible /metrics exporter for decision engine telemetry.
- otel: OpenTelemetry tracing integration for decision spans.
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

# Optional: Prometheus observability (requires `prometheus_client`)
try:
    from system1.integrations.observability import ReflexMetricsExporter

    __all__.append("ReflexMetricsExporter")
except ImportError:
    pass

# Optional: OpenTelemetry tracing (requires `opentelemetry-api`)
try:
    from system1.integrations.otel import ReflexOTelInstrumentor

    __all__.append("ReflexOTelInstrumentor")
except ImportError:
    pass
