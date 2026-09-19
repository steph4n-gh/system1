"""System 1 Framework Integrations.

Native, drop-in adapters and middleware for production AI stacks:
- mcp: Fail-closed Model Context Protocol (MCP) JSON-RPC safety proxy (< 1ms).
- fastapi: ASGI gateway middleware for sub-millisecond local routing & frontier escalation.
- langchain: Agent tool interceptor and callback handler with Ed25519 witness receipts.
- observability: Prometheus-compatible /metrics exporter for decision engine telemetry.
- otel: OpenTelemetry tracing integration for decision spans.
"""

from __future__ import annotations

from system1.integrations.mcp import (
    SystemOneMCPBlockedError,
    SystemOneMCPProxy,
    wrap_mcp_tool,
)
from system1.integrations.fastapi import (
    SystemOneGatewayMiddleware,
    add_system1_gateway,
)
from system1.integrations.langchain import (
    SystemOneGuardBlockedException,
    SystemOneGuardCallbackHandler,
    SystemOneToolInterceptor,
    wrap_langchain_tool,
)

# Compatibility alias
System1MCPProxy = SystemOneMCPProxy

__all__ = [
    # MCP
    "SystemOneMCPBlockedError",
    "SystemOneMCPProxy",
    "System1MCPProxy",
    "wrap_mcp_tool",
    # FastAPI / ASGI
    "SystemOneGatewayMiddleware",
    "add_system1_gateway",
    # LangChain / Agent
    "SystemOneGuardBlockedException",
    "SystemOneGuardCallbackHandler",
    "SystemOneToolInterceptor",
    "wrap_langchain_tool",
]

# Optional: Prometheus observability (requires `prometheus_client`)
try:
    from system1.integrations.observability import SystemOneMetricsExporter

    System1MetricsExporter = SystemOneMetricsExporter
    __all__.extend(["SystemOneMetricsExporter", "System1MetricsExporter"])
except ImportError:
    pass

# Optional: OpenTelemetry tracing (requires `opentelemetry-api`)
try:
    from system1.integrations.otel import SystemOneOTelInstrumentor

    __all__.append("SystemOneOTelInstrumentor")
except ImportError:
    pass
