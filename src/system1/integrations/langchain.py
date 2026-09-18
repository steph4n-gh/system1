"""Reflex LangChain & Agent Framework Guard.

Provides fail-closed tool execution gating, sub-millisecond local safety checks,
and Ed25519 cryptographic audit receipts for LangChain and autonomous agent frameworks.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from system1.engine import DecisionResult, ReflexEngine
from system1.guard import (
    ActionProposal,
    DecisionOutcome,
    GuardInterceptionResult,
    PolicyDecision,
    ReflexGuardHook,
    RiskLevel,
)
from system1.ledger import ActionLedger

# Conditional import of LangChain BaseCallbackHandler if present; fallback to object
try:
    from langchain_core.callbacks.base import BaseCallbackHandler as _BaseCallbackHandler
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler as _BaseCallbackHandler
    except ImportError:
        class _BaseCallbackHandler:  # type: ignore[no-redef]
            """Fallback duck-typed callback handler when langchain is not installed."""
            pass


class ReflexGuardBlockedException(PermissionError):
    """Raised when an agent tool call is blocked or requires approval by Reflex Guard."""

    def __init__(self, interception: GuardInterceptionResult):
        self.interception = interception
        self.outcome = interception.outcome
        self.reason = interception.reason
        self.policy_decision = interception.policy_decision
        super().__init__(f"Reflex Guard Blocked Tool Execution: {self.outcome.value} - {self.reason}")


# Alias matching Technical Specification Section 7.3
ReflexSecurityException = ReflexGuardBlockedException


class ReflexGuardCallbackHandler(_BaseCallbackHandler):
    """LangChain CallbackHandler enforcing sub-millisecond Reflex safety before tool execution."""

    def __init__(
        self,
        guard: Optional[ReflexGuardHook] = None,
        *,
        ledger: Optional[ActionLedger] = None,
        alpha: float = 0.20,
        min_confidence: float = 0.50,
        tenant_id: str = "tenant_langchain",
        principal_id: str = "agent_langchain_runner",
    ):
        self.guard = guard or ReflexGuardHook(
            ledger=ledger,
            alpha=alpha,
            min_confidence=min_confidence,
        )
        self.tenant_id = tenant_id
        self.principal_id = principal_id
        self.interceptions: List[GuardInterceptionResult] = []

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Optional[Any] = None,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called before a tool is executed by the LangChain agent."""
        tool_name = serialized.get("name") or "unknown_tool"
        description = serialized.get("description") or f"Tool execution: {tool_name}"
        proposal = ActionProposal.create(
            tenant_id=self.tenant_id,
            principal_id=str(run_id or self.principal_id),
            scope="langchain:tools:exec",
            tool=tool_name,
            arguments={"input": input_str},
            purpose=description,
        )
        context = f"{description}. Action: {tool_name} with {str(input_str[:256])}."
        interception = self.guard.evaluate_proposal(proposal, context_prompt=context)
        self.interceptions.append(interception)

        if interception.outcome != DecisionOutcome.ALLOW:
            raise ReflexGuardBlockedException(interception)


class ReflexToolInterceptor:
    """Wraps a LangChain BaseTool or generic tool callable with fail-closed Reflex gating."""

    def __init__(
        self,
        tool: Any,
        guard: Optional[ReflexGuardHook] = None,
        *,
        alpha: float = 0.20,
        min_confidence: float = 0.50,
        tool_name: Optional[str] = None,
    ):
        self.tool = tool
        self.guard = guard or ReflexGuardHook(alpha=alpha, min_confidence=min_confidence)
        self.name = tool_name or getattr(tool, "name", getattr(tool, "__name__", "tool"))
        self.description = getattr(tool, "description", getattr(tool, "__doc__", ""))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.invoke(*args, **kwargs)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Synchronous invocation interceptor."""
        raw_args = dict(kwargs)
        if args:
            raw_args["_args"] = list(args)

        target = str(raw_args.get("input") or raw_args.get("query") or raw_args.get("path") or (args[0] if args else self.name))
        purpose = str(raw_args.get("purpose") or self.description or f"Tool invocation '{self.name}'")
        proposal = ActionProposal.create(
            tenant_id="tenant_reflex_tool",
            principal_id="agent_caller",
            scope="langchain:tool:invoke",
            tool=self.name,
            arguments=raw_args,
            purpose=purpose,
        )
        context = f"{purpose}. Tool: {self.name} on {target}."
        interception = self.guard.evaluate_proposal(proposal, context_prompt=context)
        if interception.outcome != DecisionOutcome.ALLOW:
            raise ReflexGuardBlockedException(interception)

        # Execute underlying tool
        if hasattr(self.tool, "invoke"):
            return self.tool.invoke(*args, **kwargs)
        elif hasattr(self.tool, "run"):
            return self.tool.run(*args, **kwargs)
        elif callable(self.tool):
            return self.tool(*args, **kwargs)
        raise TypeError(f"Target tool {self.tool!r} is neither callable nor implements invoke()/run()")

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        """Asynchronous invocation interceptor."""
        # For simplicity and sub-ms speed, evaluate guard synchronously on local metal
        raw_args = dict(kwargs)
        if args:
            raw_args["_args"] = list(args)

        target = str(raw_args.get("input") or raw_args.get("query") or raw_args.get("path") or (args[0] if args else self.name))
        purpose = str(raw_args.get("purpose") or self.description or f"Tool invocation '{self.name}'")
        proposal = ActionProposal.create(
            tenant_id="tenant_reflex_tool",
            principal_id="agent_caller",
            scope="langchain:tool:ainvoke",
            tool=self.name,
            arguments=raw_args,
            purpose=purpose,
        )
        context = f"{purpose}. Tool: {self.name} on {target}."
        interception = self.guard.evaluate_proposal(proposal, context_prompt=context)
        if interception.outcome != DecisionOutcome.ALLOW:
            raise ReflexGuardBlockedException(interception)

        if hasattr(self.tool, "ainvoke"):
            return await self.tool.ainvoke(*args, **kwargs)
        elif hasattr(self.tool, "arun"):
            return await self.tool.arun(*args, **kwargs)
        elif hasattr(self.tool, "invoke"):
            return self.tool.invoke(*args, **kwargs)
        elif callable(self.tool):
            return self.tool(*args, **kwargs)
        raise TypeError(f"Target tool {self.tool!r} is not callable asynchronously")


def wrap_langchain_tool(
    tool: Any,
    *,
    guard: Optional[ReflexGuardHook] = None,
    alpha: float = 0.20,
    min_confidence: float = 0.50,
    tool_name: Optional[str] = None,
) -> ReflexToolInterceptor:
    """Wraps any LangChain tool or callable function with Reflex fail-closed safety interceptor."""
    return ReflexToolInterceptor(tool, guard=guard, alpha=alpha, min_confidence=min_confidence, tool_name=tool_name)


__all__ = [
    "ReflexGuardBlockedException",
    "ReflexSecurityException",
    "ReflexGuardCallbackHandler",
    "ReflexToolInterceptor",
    "wrap_langchain_tool",
]
