"""Reflex MCP (Model Context Protocol) Safety Proxy.

Intercepts MCP JSON-RPC tool calls on local metal in < 1ms with fail-closed
conformal gating and Ed25519 digital witness receipts.
"""

from __future__ import annotations

import hashlib
import inspect
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
        if not isinstance(arguments, dict):
            arguments_dict: Dict[str, Any] = {"input": str(arguments)} if arguments is not None else {}
        else:
            arguments_dict = dict(arguments)
        target = str(
            arguments_dict.get("path")
            or arguments_dict.get("target")
            or arguments_dict.get("command")
            or arguments_dict.get("input")
            or arguments_dict.get("query")
            or arguments_dict.get("url")
            or arguments_dict.get("file")
            or tool_name
        )
        purpose = str(arguments_dict.get("purpose") or f"Execute tool {tool_name} on {target}")
        canonical_target = (
            f"sha256:{hashlib.sha256(target.encode('utf-8')).hexdigest()}"
            if len(target) > 4096
            else target
        )
        proposal = ActionProposal.create(
            tenant_id=self.tenant_id,
            principal_id=client_session_id or self.principal_id,
            scope="mcp:tools:call",
            tool=tool_name,
            arguments=arguments_dict,
            canonical_target=canonical_target,
            purpose=purpose,
        )
        base_context = f"{purpose}. Action: {tool_name} on {target}. Arguments: {arguments_dict}"
        context = context_prompt or base_context
        return self.guard.evaluate_proposal(proposal, context_prompt=context)

    def intercept_jsonrpc(
        self,
        request: Union[Dict[str, Any], str, bytes],
        *,
        client_session_id: Optional[str] = None,
        context_prompt: Optional[str] = None,
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

        if not isinstance(data, dict):
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request: payload must be a JSON object"},
            }
            return False, err_resp, None

        method = data.get("method")
        req_id = data.get("id")

        # Non-tool-call methods (e.g. initialize, tools/list, ping) pass through directly
        if method != "tools/call":
            return True, None, None

        params = data.get("params")
        if not isinstance(params, dict):
            err_resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "Invalid params: params must be a JSON object"},
            }
            return False, err_resp, None

        tool_name = params.get("name", "")
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        elif not isinstance(arguments, dict):
            err_resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "Invalid params: arguments must be an object"},
            }
            return False, err_resp, None

        if not tool_name:
            err_resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "Invalid params: missing tool name"},
            }
            return False, err_resp, None

        interception = self.evaluate_mcp_call(
            tool_name=tool_name,
            arguments=arguments,
            client_session_id=client_session_id,
            context_prompt=context_prompt,
        )

        if interception.outcome == DecisionOutcome.ALLOW:
            return True, None, interception
        else:
            err = ReflexMCPBlockedError(interception)
            return False, err.to_jsonrpc_error(request_id=req_id), interception

    def handle_call(
        self,
        request: Union[Dict[str, Any], str, bytes],
        executor: Callable[[str, Dict[str, Any]], Any],
        *,
        client_session_id: Optional[str] = None,
        context_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """High-level dispatcher that intercepts, executes if allowed, and injects witness receipts."""
        allowed, err_resp, interception = self.intercept_jsonrpc(
            request,
            client_session_id=client_session_id,
            context_prompt=context_prompt,
        )
        if not allowed:
            return err_resp or {}

        try:
            data = json.loads(request) if isinstance(request, (str, bytes)) else request
        except Exception:
            return err_resp or {}

        req_id = data.get("id")
        method = data.get("method")

        # Invariant 1: Only dispatch to executor when method == "tools/call".
        # For non-tool methods (e.g. ping, initialize), return standard protocol response with 0 executor calls.
        if method != "tools/call":
            if method == "ping":
                result = {}
            elif method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "reflex-mcp-guard", "version": "0.1.0"},
                    "capabilities": {"tools": {}},
                }
            elif method == "tools/list":
                result = {"tools": []}
            else:
                result = {}
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result,
            }

        params = data.get("params") or {}
        tool_name = params.get("name", "")
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        elif not isinstance(arguments, dict):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "Invalid params: arguments must be an object"},
            }

        # Two-phase execution outcome chaining
        ledger = self.guard.ledger
        action_id = (
            interception.policy_decision.action_id
            if (interception and interception.policy_decision)
            else str(req_id or "")
        )
        receipt_digest = ""
        if interception and interception.decision_result and interception.decision_result.receipt:
            receipt_digest = interception.decision_result.receipt.compute_digest()

        exec_output = None
        exec_error = None
        try:
            exec_output = executor(tool_name, arguments)
        except Exception as ex:
            exec_error = ex

        # Record outcome in ledger if active
        if ledger is not None and receipt_digest:
            outcome_status = "FAILED" if exec_error is not None else "SUCCEEDED"
            try:
                ledger.record_execution_outcome(
                    action_id=action_id,
                    receipt_digest=receipt_digest,
                    status=outcome_status,
                    result_payload={"result": str(exec_output)[:500]} if exec_output is not None else None,
                    error_message=str(exec_error) if exec_error is not None else None,
                    tenant_id=self.tenant_id,
                    principal_id=client_session_id or self.principal_id,
                    scope="mcp:tools:call:outcome",
                )
            except Exception as le:
                # If execution succeeded but outcome recording failed in fail-closed mode:
                if exec_error is None:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32001,
                            "message": f"Execution completed but outcome audit recording failed (status: INDETERMINATE): {le}",
                            "data": {"status": "INDETERMINATE", "raw_result": exec_output},
                        },
                    }

        if exec_error is not None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": f"Internal tool execution error: {exec_error}"},
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
    func: Optional[Any] = None,
    *,
    guard: Optional[ReflexGuardHook] = None,
    proxy: Optional[ReflexMCPProxy] = None,
    alpha: float = 0.20,
    min_confidence: float = 0.50,
    tool_name: Optional[str] = None,
) -> Callable:
    """Decorator to protect any Python function exposed as an MCP tool with Reflex fail-closed safety."""
    if isinstance(func, ReflexMCPProxy):
        proxy = func
        func = None
    resolved_proxy = proxy or ReflexMCPProxy(guard=guard, alpha=alpha, min_confidence=min_confidence)

    def decorator(fn: Callable) -> Callable:
        resolved_name = tool_name or getattr(fn, "__name__", "tool")
        try:
            sig = inspect.signature(fn)
        except Exception:
            sig = None

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Use inspect.signature to bind default parameter values and map positional *args into named arguments
            arguments: Dict[str, Any] = {}
            if sig is not None:
                try:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    arguments = dict(bound.arguments)
                except Exception:
                    arguments = dict(kwargs)
                    if args:
                        arguments["_args"] = list(args)
            else:
                arguments = dict(kwargs)
                if args:
                    arguments["_args"] = list(args)

            if "purpose" not in arguments and fn.__doc__:
                arguments["purpose"] = fn.__doc__.strip()

            interception = resolved_proxy.evaluate_mcp_call(
                tool_name=resolved_name,
                arguments=arguments,
            )

            if interception.outcome != DecisionOutcome.ALLOW:
                raise ReflexMCPBlockedError(interception)

            return fn(*args, **kwargs)

        wrapper.__name__ = getattr(fn, "__name__", "wrapper")
        wrapper.__doc__ = getattr(fn, "__doc__", "")
        wrapper.__reflex_proxy__ = resolved_proxy  # type: ignore[attr-defined]
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


__all__ = [
    "ReflexMCPBlockedError",
    "ReflexMCPProxy",
    "wrap_mcp_tool",
]
