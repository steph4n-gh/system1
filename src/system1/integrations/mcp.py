"""Reflex MCP (Model Context Protocol) Safety Proxy.

Intercepts MCP JSON-RPC tool calls on local metal in < 1ms with fail-closed
conformal gating and Ed25519 digital witness receipts.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

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
from system1.receipt import DecisionWitnessReceipt, create_decision_receipt


class ReflexMCPBlockedError(PermissionError):
    """Raised when an MCP tool call fails Reflex safety or conformal ambiguity checks."""

    def __init__(self, interception: GuardInterceptionResult):
        self.interception = interception
        self.outcome = interception.outcome
        self.reason = interception.reason
        self.policy_decision = interception.policy_decision
        super().__init__(f"Reflex MCP Security Interception: {self.outcome.value} - {self.reason}")

    def to_jsonrpc_error(self, request_id: Any = None) -> Dict[str, Any]:
        """Formats into standard JSON-RPC 2.0 error response with audit proof."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32000,
                "message": f"Reflex Safety Gate: {self.outcome.value} - {self.reason}",
                "data": {
                    "outcome": self.outcome.value,
                    "reason": self.reason,
                    "rule_id": self.policy_decision.rule_id if self.policy_decision else None,
                    "risk": self.policy_decision.risk.name if self.policy_decision else None,
                    "receipt_digest": (
                        self.interception.decision_result.schema_digest
                        if self.interception.decision_result
                        else None
                    ),
                },
            },
        }


class ReflexMCPProxy:
    """Fail-closed Reference Monitor for Model Context Protocol (MCP) tool execution.
    
    Evaluates MCP `tools/call` JSON-RPC requests on local metal in < 1ms.
    Enforces conformal coverage bounds, semantic safety, and creates Ed25519 digital receipts.
    """

    def __init__(
        self,
        guard: Optional[ReflexGuardHook] = None,
        *,
        ledger: Optional[ActionLedger] = None,
        alpha: float = 0.10,
        min_confidence: float = 0.50,
        tenant_id: str = "tenant_mcp_default",
        principal_id: str = "agent_mcp_client",
    ):
        self.guard = guard or ReflexGuardHook(
            ledger=ledger,
            alpha=alpha,
            min_confidence=min_confidence,
        )
        self.tenant_id = tenant_id
        self.principal_id = principal_id

    def evaluate_mcp_call(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        client_session_id: Optional[str] = None,
        context_prompt: Optional[str] = None,
    ) -> GuardInterceptionResult:
        """Evaluates an MCP tool call proposal against Reflex fail-closed gates."""
        target = str(arguments.get("path") or arguments.get("target") or arguments.get("command") or tool_name)
        purpose = str(arguments.get("purpose") or f"Execute tool {tool_name} on {target}")
        proposal = ActionProposal.create(
            tenant_id=self.tenant_id,
            principal_id=client_session_id or self.principal_id,
            scope="mcp:tools:call",
            tool=tool_name,
            arguments=dict(arguments),
            purpose=purpose,
        )
        context = context_prompt or f"{purpose}. Action: {tool_name} on {target}."
        return self.guard.evaluate_proposal(proposal, context_prompt=context)

    def intercept_jsonrpc(
        self,
        request: Union[Dict[str, Any], str, bytes],
        *,
        client_session_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[GuardInterceptionResult]]:
        """Intercepts a raw MCP JSON-RPC 2.0 message.
        
        Returns:
            (allowed: bool, jsonrpc_error_or_none: Optional[dict], result: GuardInterceptionResult)
        """
        if isinstance(request, (str, bytes)):
            try:
                data = json.loads(request)
            except Exception as e:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {e}"},
                }
                return False, err_resp, None
        else:
            data = request

        method = data.get("method")
        req_id = data.get("id")

        # Non-tool-call methods (e.g. initialize, tools/list, ping) pass through directly
        if method != "tools/call":
            return True, None, None

        params = data.get("params") or {}
        tool_name = params.get("name", "")
        arguments = params.get("arguments") or {}

        interception = self.evaluate_mcp_call(
            tool_name=tool_name,
            arguments=arguments,
            client_session_id=client_session_id,
        )

        if interception.outcome == DecisionOutcome.ALLOW:
            return True, None, interception
        else:
            err = ReflexMCPBlockedError(interception)
            return False, err.to_jsonrpc_error(request_id=req_id), interception

    def handle_call(
        self,
        request: Union[Dict[str, Any], str],
        executor: Callable[[str, Dict[str, Any]], Any],
        *,
        client_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """High-level dispatcher that intercepts, executes if allowed, and injects witness receipts."""
        allowed, err_resp, interception = self.intercept_jsonrpc(
            request, client_session_id=client_session_id
        )
        if not allowed:
            return err_resp or {}

        data = json.loads(request) if isinstance(request, str) else request
        req_id = data.get("id")
        params = data.get("params") or {}
        tool_name = params.get("name", "")
        arguments = params.get("arguments") or {}

        try:
            exec_output = executor(tool_name, arguments)
        except Exception as ex:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": f"Internal tool execution error: {ex}"},
            }

        # Structure standard MCP tool call result
        result_payload: Dict[str, Any]
        if isinstance(exec_output, dict) and "content" in exec_output:
            result_payload = exec_output
        else:
            result_payload = {
                "content": [{"type": "text", "text": str(exec_output)}],
            }

        # Inject cryptographic witness metadata
        if interception and interception.decision_result:
            meta = result_payload.setdefault("_meta", {})
            meta["reflex"] = {
                "status": interception.outcome.value,
                "latency_ms": interception.decision_result.latency_ms,
                "digest": interception.decision_result.schema_digest,
                "confidences": interception.decision_result.confidences,
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result_payload,
        }


def wrap_mcp_tool(
    func: Optional[Callable] = None,
    *,
    guard: Optional[ReflexGuardHook] = None,
    alpha: float = 0.20,
    min_confidence: float = 0.50,
    tool_name: Optional[str] = None,
) -> Callable:
    """Decorator to protect any Python function exposed as an MCP tool with Reflex fail-closed safety."""
    proxy = ReflexMCPProxy(guard=guard, alpha=alpha, min_confidence=min_confidence)

    def decorator(fn: Callable) -> Callable:
        resolved_name = tool_name or fn.__name__

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Map positional args and kwargs to argument dictionary
            arguments = dict(kwargs)
            if args:
                arguments["_args"] = list(args)
            if "purpose" not in arguments and fn.__doc__:
                arguments["purpose"] = fn.__doc__.strip()

            interception = proxy.evaluate_mcp_call(
                tool_name=resolved_name,
                arguments=arguments,
            )

            if interception.outcome != DecisionOutcome.ALLOW:
                raise ReflexMCPBlockedError(interception)

            return fn(*args, **kwargs)

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__reflex_proxy__ = proxy  # type: ignore[attr-defined]
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


__all__ = [
    "ReflexMCPBlockedError",
    "ReflexMCPProxy",
    "wrap_mcp_tool",
]
