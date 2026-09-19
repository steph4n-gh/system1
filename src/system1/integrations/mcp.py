"""System 1 MCP (Model Context Protocol) Safety Proxy.

Intercepts MCP JSON-RPC tool calls on local metal in < 1ms with fail-closed
conformal gating and Ed25519 digital witness receipts.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

from system1.engine import DecisionResult, SystemOneEngine
from system1.guard import (
    ActionProposal,
    DecisionOutcome,
    GuardInterceptionResult,
    PolicyDecision,
    SystemOneGuardHook,
    RiskLevel,
)
from system1.ledger import ActionLedger
from system1.receipt import DecisionWitnessReceipt, create_decision_receipt
from system1.integrations.langchain import SystemOneIndeterminateExecutionError


class SystemOneMCPBlockedError(PermissionError):
    """Raised when an MCP tool call fails System 1 safety or conformal ambiguity checks."""

    def __init__(self, interception: GuardInterceptionResult):
        self.interception = interception
        self.outcome = interception.outcome
        self.reason = interception.reason
        self.policy_decision = interception.policy_decision
        super().__init__(f"System 1 MCP Security Interception: {self.outcome.value} - {self.reason}")

    def to_jsonrpc_error(self, request_id: Any = None) -> Dict[str, Any]:
        """Formats into standard JSON-RPC 2.0 error response with audit proof."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32000,
                "message": f"System 1 Safety Gate: {self.outcome.value} - {self.reason}",
                "data": {
                    "outcome": self.outcome.value,
                    "reason": self.reason,
                    "rule_id": self.policy_decision.rule_id if self.policy_decision else None,
                    "risk": self.policy_decision.risk.name if self.policy_decision else None,
                    "receipt_digest": (
                        self.interception.receipt.compute_digest()
                        if self.interception.receipt is not None
                        else None
                    ),
                },
            },
        }


class SystemOneMCPProxy:
    """Fail-closed Reference Monitor for Model Context Protocol (MCP) tool execution.
    
    Evaluates MCP `tools/call` JSON-RPC requests on local metal in < 1ms.
    Enforces conformal coverage bounds, semantic safety, and creates Ed25519 digital receipts.
    """

    def __init__(
        self,
        guard: Optional[SystemOneGuardHook] = None,
        *,
        ledger: Optional[ActionLedger] = None,
        alpha: float = 0.10,
        min_confidence: float = 0.50,
        tenant_id: str = "tenant_mcp_default",
        principal_id: str = "agent_mcp_client",
    ):
        self.guard = guard or SystemOneGuardHook(
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
        executor: Optional[Callable] = None,
        client_session_id: Optional[str] = None,
        context_prompt: Optional[str] = None,
    ) -> GuardInterceptionResult:
        """Evaluates an MCP tool call proposal against System 1 fail-closed gates."""
        if executor is not None and inspect.iscoroutinefunction(executor):
            raise TypeError("evaluate_mcp_call cannot execute an async coroutine. Use async_handle_call instead.")
            
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
            err = SystemOneMCPBlockedError(interception)
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
        if inspect.iscoroutinefunction(executor):
            raise TypeError("handle_call cannot execute an async executor. Use async_handle_call instead.")
            
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
                    "serverInfo": {"name": "reflex-mcp-guard", "version": "0.2.2"},
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
        receipt = (
            interception.receipt
            or (interception.decision_result.receipt if interception.decision_result else None)
        )
        if receipt:
            receipt_digest = receipt.compute_digest()

        # Invariant 5: Dispatch exact frozen canonicalized proposal arguments
        dispatch_args = (
            dict(interception.proposal.arguments)
            if (interception and interception.proposal and interception.proposal.arguments is not None)
            else (arguments if isinstance(arguments, dict) else {})
        )
        dispatch_tool = (
            interception.proposal.tool
            if (interception and interception.proposal and interception.proposal.tool)
            else tool_name
        )

        exec_output = None
        exec_error = None
        try:
            exec_output = executor(dispatch_tool, dispatch_args)
            if inspect.isawaitable(exec_output):
                raise TypeError("Synchronous handle_call received an awaitable tool result. Use async_handle_call.")
        except Exception as ex:
            exec_error = ex

        # Record outcome in ledger if active
        if ledger is not None and receipt_digest:
            outcome_status = "FAILED" if exec_error is not None else "SUCCEEDED"
            pub_key = self.guard.engine.signing_key.public_key() if self.guard.engine.signing_key else None
            try:
                ledger.record_execution_outcome(
                    action_id=action_id,
                    receipt_digest=receipt_digest,
                    status=outcome_status,
                    result_payload={"result": str(exec_output)[:500]} if exec_output is not None else None,
                    error_message=str(exec_error) if exec_error is not None else None,
                    tenant_id=self.tenant_id,
                    principal_id=client_session_id or self.principal_id,
                    scope="mcp:tools:call",
                    trusted_public_key=pub_key,
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
                            "data": {
                                "status": "INDETERMINATE",
                                "raw_result": exec_output,
                                "action_id": action_id,
                                "receipt_digest": receipt_digest,
                            },
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
                "digest": receipt_digest,
                "confidences": interception.decision_result.confidences,
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result_payload,
        }


    async def async_handle_call(
        self,
        request: Union[Dict[str, Any], str, bytes],
        executor: Callable[[str, Dict[str, Any]], Any],
        *,
        client_session_id: Optional[str] = None,
        context_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """High-level async dispatcher that intercepts, executes if allowed, and injects witness receipts."""
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

        if method != "tools/call":
            if method == "ping":
                result = {}
            elif method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "reflex-mcp-guard", "version": "0.2.2"},
                    "capabilities": {"tools": {}},
                }
            elif method == "tools/list":
                result = {"tools": []}
            else:
                result = {}
            return {"jsonrpc": "2.0", "id": req_id, "result": result}

        params = data.get("params") or {}
        tool_name = params.get("name", "")
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        elif not isinstance(arguments, dict):
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params"}}

        ledger = self.guard.ledger
        action_id = interception.policy_decision.action_id if (interception and interception.policy_decision) else str(req_id or "")
        receipt_digest = ""
        receipt = interception.receipt or (interception.decision_result.receipt if interception.decision_result else None)
        if receipt:
            receipt_digest = receipt.compute_digest()

        dispatch_args = dict(interception.proposal.arguments) if (interception and interception.proposal and interception.proposal.arguments is not None) else (arguments if isinstance(arguments, dict) else {})
        dispatch_tool = interception.proposal.tool if (interception and interception.proposal and interception.proposal.tool) else tool_name

        exec_output = None
        exec_error = None
        try:
            res = executor(dispatch_tool, dispatch_args)
            if inspect.isawaitable(res):
                exec_output = await res
            else:
                exec_output = res
        except Exception as ex:
            exec_error = ex

        if ledger is not None and receipt_digest:
            outcome_status = "FAILED" if exec_error is not None else "SUCCEEDED"
            pub_key = self.guard.engine.signing_key.public_key() if self.guard.engine.signing_key else None
            try:
                ledger.record_execution_outcome(
                    action_id=action_id,
                    receipt_digest=receipt_digest,
                    status=outcome_status,
                    result_payload={"result": str(exec_output)[:500]} if exec_output is not None else None,
                    error_message=str(exec_error) if exec_error is not None else None,
                    tenant_id=self.tenant_id,
                    principal_id=client_session_id or self.principal_id,
                    scope="mcp:tools:call",
                    trusted_public_key=pub_key,
                )
            except Exception as le:
                if exec_error is None:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32001,
                            "message": f"Execution completed but outcome audit recording failed (status: INDETERMINATE): {le}",
                            "data": {
                                "status": "INDETERMINATE",
                                "raw_result": exec_output,
                                "action_id": action_id,
                                "receipt_digest": receipt_digest,
                            },
                        },
                    }

        if exec_error is not None:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32603, "message": f"Internal tool execution error: {exec_error}"}}

        result_payload: Dict[str, Any]
        if isinstance(exec_output, dict) and "content" in exec_output:
            result_payload = exec_output
        else:
            result_payload = {"content": [{"type": "text", "text": str(exec_output)}]}

        if interception and interception.decision_result:
            meta = result_payload.setdefault("_meta", {})
            meta["reflex"] = {
                "status": interception.outcome.value,
                "latency_ms": interception.decision_result.latency_ms,
                "digest": receipt_digest,
                "confidences": interception.decision_result.confidences,
            }

        return {"jsonrpc": "2.0", "id": req_id, "result": result_payload}


def wrap_mcp_tool(
    func: Optional[Any] = None,
    *,
    guard: Optional[SystemOneGuardHook] = None,
    proxy: Optional[SystemOneMCPProxy] = None,
    alpha: float = 0.20,
    min_confidence: float = 0.50,
    tool_name: Optional[str] = None,
) -> Callable:
    """Decorator to protect any Python function exposed as an MCP tool with System 1 fail-closed safety."""
    if isinstance(func, SystemOneMCPProxy):
        proxy = func
        func = None
    resolved_proxy = proxy or SystemOneMCPProxy(guard=guard, alpha=alpha, min_confidence=min_confidence)

    def decorator(fn: Callable) -> Callable:
        resolved_name = tool_name or getattr(fn, "__name__", "tool")
        try:
            sig = inspect.signature(fn)
        except Exception:
            sig = None

        if inspect.iscoroutinefunction(fn):
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                arguments: Dict[str, Any] = {}
                bound = None
                if sig is not None:
                    try:
                        bound = sig.bind(*args, **kwargs)
                        bound.apply_defaults()
                        arguments = dict(bound.arguments)
                    except Exception:
                        bound = None
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
                    raise SystemOneMCPBlockedError(interception)

                prop = getattr(interception, "proposal", None)
                if isinstance(prop, ActionProposal):
                    canon_args = dict(prop.arguments)
                elif isinstance(getattr(interception, "proposal", None), (dict, Mapping)):
                    canon_args = dict(interception.proposal)
                else:
                    canon_args = dict(arguments)

                if sig is not None and bound is not None:
                    for k in list(bound.arguments.keys()):
                        if k in canon_args:
                            bound.arguments[k] = canon_args[k]
                    for p_name, p in sig.parameters.items():
                        if p.kind == inspect.Parameter.VAR_KEYWORD and p_name in bound.arguments:
                            var_kw = dict(bound.arguments[p_name])
                            for k, v in canon_args.items():
                                if k not in sig.parameters and k != "_args" and k != "purpose":
                                    var_kw[k] = v
                            bound.arguments[p_name] = var_kw
                    call_args = bound.args
                    call_kwargs = bound.kwargs
                else:
                    call_args = tuple(canon_args.get("_args", [])) if "_args" in canon_args else args
                    call_kwargs = {k: v for k, v in canon_args.items() if k != "_args" and k != "purpose"}

                ledger = resolved_proxy.guard.ledger
                action_id = interception.policy_decision.action_id if (interception and interception.policy_decision) else ""
                receipt_digest = ""
                receipt = interception.receipt or (interception.decision_result.receipt if interception.decision_result else None)
                if receipt:
                    receipt_digest = receipt.compute_digest()

                exec_output = None
                exec_error = None
                try:
                    exec_output = await fn(*call_args, **call_kwargs)
                except Exception as ex:
                    exec_error = ex

                if ledger is not None and receipt_digest:
                    outcome_status = "FAILED" if exec_error is not None else "SUCCEEDED"
                    pub_key = resolved_proxy.guard.engine.signing_key.public_key() if resolved_proxy.guard.engine.signing_key else None
                    try:
                        ledger.record_execution_outcome(
                            action_id=action_id,
                            receipt_digest=receipt_digest,
                            status=outcome_status,
                            result_payload={"result": str(exec_output)[:500]} if exec_output is not None else None,
                            error_message=str(exec_error) if exec_error is not None else None,
                            tenant_id=resolved_proxy.tenant_id,
                            principal_id=resolved_proxy.principal_id,
                            scope=prop.scope if isinstance(prop, ActionProposal) else "mcp:tools:call",
                            trusted_public_key=pub_key,
                        )
                    except Exception as le:
                        raise SystemOneIndeterminateExecutionError(
                            f"MCP wrapped tool executed but outcome recording failed (status: INDETERMINATE): {le}",
                            action_id=action_id,
                            receipt_digest=receipt_digest,
                            raw_result=exec_output,
                            underlying_error=le,
                            exec_error=exec_error,
                        ) from le

                if exec_error is not None:
                    raise exec_error

                return exec_output
        else:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                arguments: Dict[str, Any] = {}
                bound = None
                if sig is not None:
                    try:
                        bound = sig.bind(*args, **kwargs)
                        bound.apply_defaults()
                        arguments = dict(bound.arguments)
                    except Exception:
                        bound = None
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
                    raise SystemOneMCPBlockedError(interception)

                prop = getattr(interception, "proposal", None)
                if isinstance(prop, ActionProposal):
                    canon_args = dict(prop.arguments)
                elif isinstance(getattr(interception, "proposal", None), (dict, Mapping)):
                    canon_args = dict(interception.proposal)
                else:
                    canon_args = dict(arguments)

                if sig is not None and bound is not None:
                    for k in list(bound.arguments.keys()):
                        if k in canon_args:
                            bound.arguments[k] = canon_args[k]
                    for p_name, p in sig.parameters.items():
                        if p.kind == inspect.Parameter.VAR_KEYWORD and p_name in bound.arguments:
                            var_kw = dict(bound.arguments[p_name])
                            for k, v in canon_args.items():
                                if k not in sig.parameters and k != "_args" and k != "purpose":
                                    var_kw[k] = v
                            bound.arguments[p_name] = var_kw
                    call_args = bound.args
                    call_kwargs = bound.kwargs
                else:
                    call_args = tuple(canon_args.get("_args", [])) if "_args" in canon_args else args
                    call_kwargs = {k: v for k, v in canon_args.items() if k != "_args" and k != "purpose"}

                ledger = resolved_proxy.guard.ledger
                action_id = interception.policy_decision.action_id if (interception and interception.policy_decision) else ""
                receipt_digest = ""
                receipt = interception.receipt or (interception.decision_result.receipt if interception.decision_result else None)
                if receipt:
                    receipt_digest = receipt.compute_digest()

                exec_output = None
                exec_error = None
                try:
                    exec_output = fn(*call_args, **call_kwargs)
                    if inspect.isawaitable(exec_output):
                        raise TypeError("Synchronous tool wrapper received an awaitable. Use async tool wrapper.")
                except Exception as ex:
                    exec_error = ex

                def _record(outcome_status: str, payload: Any, error: Optional[Exception]):
                    if ledger is not None and receipt_digest:
                        pub_key = resolved_proxy.guard.engine.signing_key.public_key() if resolved_proxy.guard.engine.signing_key else None
                        try:
                            ledger.record_execution_outcome(
                                action_id=action_id,
                                receipt_digest=receipt_digest,
                                status=outcome_status,
                                result_payload={"result": str(payload)[:500]} if payload is not None else None,
                                error_message=str(error) if error is not None else None,
                                tenant_id=resolved_proxy.tenant_id,
                                principal_id=resolved_proxy.principal_id,
                                scope=prop.scope if isinstance(prop, ActionProposal) else "mcp:tools:call",
                                trusted_public_key=pub_key,
                            )
                        except Exception as le:
                            raise SystemOneIndeterminateExecutionError(
                                f"MCP wrapped tool executed but outcome recording failed (status: INDETERMINATE): {le}",
                                action_id=action_id,
                                receipt_digest=receipt_digest,
                                raw_result=payload,
                                underlying_error=le,
                                exec_error=error,
                            ) from le

                if inspect.isawaitable(exec_output):
                    async def _async_record_wrapper(coro: Any) -> Any:
                        try:
                            res = await coro
                        except Exception as aex:
                            _record("FAILED", None, aex)
                            raise
                        _record("SUCCEEDED", res, None)
                        return res
                    return _async_record_wrapper(exec_output)
                else:
                    outcome_status = "FAILED" if exec_error is not None else "SUCCEEDED"
                    _record(outcome_status, exec_output, exec_error)

                if exec_error is not None:
                    raise exec_error

                return exec_output

        wrapper.__name__ = getattr(fn, "__name__", "wrapper")
        wrapper.__doc__ = getattr(fn, "__doc__", "")
        wrapper.__system1_proxy__ = resolved_proxy  # type: ignore[attr-defined]
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


__all__ = [
    "SystemOneIndeterminateExecutionError",
    "SystemOneMCPBlockedError",
    "SystemOneMCPProxy",
    "wrap_mcp_tool",
]
