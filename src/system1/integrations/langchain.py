"""System 1 LangChain & Agent Framework Guard.

Provides fail-closed tool execution gating, sub-millisecond local safety checks,
and Ed25519 cryptographic audit receipts for LangChain and autonomous agent frameworks.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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


class SystemOneGuardBlockedException(PermissionError):
    """Raised when an agent tool call is blocked or requires approval by System 1 Guard."""

    def __init__(self, interception: GuardInterceptionResult):
        self.interception = interception
        self.outcome = interception.outcome
        self.reason = interception.reason
        self.policy_decision = interception.policy_decision
        super().__init__(f"System 1 Guard Blocked Tool Execution: {self.outcome.value} - {self.reason}")


# Alias matching Technical Specification Section 7.3
SystemOneSecurityException = SystemOneGuardBlockedException


class SystemOneIndeterminateExecutionError(RuntimeError):
    """Raised when a tool executed with potential side effects, but durable outcome audit recording failed.

    Preserves reconciliation evidence (action_id, receipt_digest, raw_result) for governors.
    """

    def __init__(
        self,
        message: str,
        *,
        action_id: str,
        receipt_digest: str,
        raw_result: Any = None,
        underlying_error: Optional[Exception] = None,
        exec_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.status = "INDETERMINATE"
        self.action_id = str(action_id)
        self.receipt_digest = str(receipt_digest)
        self.raw_result = raw_result
        self.underlying_error = underlying_error
        self.exec_error = exec_error


class SystemOneGuardCallbackHandler(_BaseCallbackHandler):
    """LangChain CallbackHandler enforcing sub-millisecond System 1 safety before tool execution."""

    def __init__(
        self,
        guard: Optional[SystemOneGuardHook] = None,
        *,
        ledger: Optional[ActionLedger] = None,
        alpha: float = 0.20,
        min_confidence: float = 0.50,
        tenant_id: str = "tenant_langchain",
        principal_id: str = "agent_langchain_runner",
    ):
        self.guard = guard or SystemOneGuardHook(
            ledger=ledger,
            alpha=alpha,
            min_confidence=min_confidence,
        )
        self.tenant_id = tenant_id
        self.principal_id = principal_id
        self.interceptions: List[GuardInterceptionResult] = []
        self._active_runs: Dict[str, GuardInterceptionResult] = {}

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
        raw_input = str(input_str)
        canonical_target = (
            f"sha256:{hashlib.sha256(raw_input.encode('utf-8')).hexdigest()}"
            if len(raw_input) > 4096
            else raw_input
        )
        proposal = ActionProposal.create(
            tenant_id=self.tenant_id,
            principal_id=str(run_id or self.principal_id),
            scope="langchain:tools:exec",
            tool=tool_name,
            arguments={"input": input_str},
            canonical_target=canonical_target,
            purpose=description,
        )
        # Invariant 2: Full context without [:256] character truncation
        context = f"{description}. Action: {tool_name} with {str(input_str)}."
        interception = self.guard.evaluate_proposal(proposal, context_prompt=context)
        self.interceptions.append(interception)
        if run_id is not None:
            self._active_runs[str(run_id)] = interception

        if interception.outcome != DecisionOutcome.ALLOW:
            raise SystemOneGuardBlockedException(interception)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Optional[Any] = None,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Called after a tool finishes execution to chain outcome to the authorization receipt."""
        interception = (
            self._active_runs.pop(str(run_id), None)
            if run_id is not None
            else (self.interceptions[-1] if self.interceptions else None)
        )
        ledger = self.guard.ledger
        if ledger is not None and interception is not None:
            receipt = interception.receipt or (
                interception.decision_result.receipt if interception.decision_result else None
            )
            receipt_digest = receipt.compute_digest() if receipt else ""
            action_id = (
                interception.policy_decision.action_id
                if interception.policy_decision
                else (getattr(receipt, "decision_id", "") if receipt else str(run_id or ""))
            )
            if receipt_digest:
                try:
                    pub_key = self.guard.engine.signing_key.public_key() if self.guard.engine.signing_key else None
                    ledger.record_execution_outcome(
                        action_id=action_id,
                        receipt_digest=receipt_digest,
                        status="SUCCEEDED",
                        result_payload={"result": str(output)[:500]},
                        tenant_id=self.tenant_id,
                        principal_id=str(run_id or self.principal_id),
                        scope="langchain:tools:exec",
                        trusted_public_key=pub_key,
                    )
                except Exception as le:
                    raise SystemOneIndeterminateExecutionError(
                        f"LangChain tool executed but outcome recording failed (status: INDETERMINATE): {le}",
                        action_id=action_id,
                        receipt_digest=receipt_digest,
                        raw_result=output,
                        underlying_error=le,
                    ) from le

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[Any] = None,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool encounters an error during execution."""
        interception = (
            self._active_runs.pop(str(run_id), None)
            if run_id is not None
            else (self.interceptions[-1] if self.interceptions else None)
        )
        ledger = self.guard.ledger
        if ledger is not None and interception is not None:
            receipt = interception.receipt or (
                interception.decision_result.receipt if interception.decision_result else None
            )
            receipt_digest = receipt.compute_digest() if receipt else ""
            action_id = (
                interception.policy_decision.action_id
                if interception.policy_decision
                else (getattr(receipt, "decision_id", "") if receipt else str(run_id or ""))
            )
            if receipt_digest:
                try:
                    pub_key = self.guard.engine.signing_key.public_key() if self.guard.engine.signing_key else None
                    ledger.record_execution_outcome(
                        action_id=action_id,
                        receipt_digest=receipt_digest,
                        status="FAILED",
                        error_message=str(error),
                        tenant_id=self.tenant_id,
                        principal_id=str(run_id or self.principal_id),
                        scope="langchain:tools:exec",
                        trusted_public_key=pub_key,
                    )
                except Exception as le:
                    raise SystemOneIndeterminateExecutionError(
                        f"LangChain tool failed ({error}) AND outcome recording failed: {le}",
                        action_id=action_id,
                        receipt_digest=receipt_digest,
                        raw_result=None,
                        underlying_error=le,
                        exec_error=error if isinstance(error, Exception) else None,
                    ) from le



class SystemOneToolInterceptor:
    """Wraps a LangChain BaseTool or generic tool callable with fail-closed System 1 gating."""

    def __init__(
        self,
        tool: Any,
        guard: Optional[SystemOneGuardHook] = None,
        *,
        alpha: float = 0.20,
        min_confidence: float = 0.50,
        tool_name: Optional[str] = None,
    ):
        self.tool = tool
        self.guard = guard or SystemOneGuardHook(alpha=alpha, min_confidence=min_confidence)
        self.name = tool_name or getattr(tool, "name", getattr(tool, "__name__", "tool"))
        self.description = getattr(tool, "description", getattr(tool, "__doc__", ""))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.invoke(*args, **kwargs)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Synchronous invocation interceptor."""
        target_fn = getattr(self.tool, "func", None) or getattr(self.tool, "_run", None) or (self.tool if callable(self.tool) else getattr(self.tool, "invoke", None))
        try:
            sig = inspect.signature(target_fn) if target_fn is not None else None
        except Exception:
            sig = None

        raw_args: Dict[str, Any] = {}
        bound = None
        if sig is not None:
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                raw_args = dict(bound.arguments)
            except Exception:
                bound = None
                raw_args = dict(kwargs)
                if args:
                    raw_args["_args"] = list(args)
        else:
            raw_args = dict(kwargs)
            if args:
                raw_args["_args"] = list(args)

        target = str(raw_args.get("input") or raw_args.get("query") or raw_args.get("path") or raw_args.get("target") or (args[0] if args else self.name))
        purpose = str(raw_args.get("purpose") or self.description or f"Tool invocation '{self.name}'")
        canonical_target = (
            f"sha256:{hashlib.sha256(target.encode('utf-8')).hexdigest()}"
            if len(target) > 4096
            else target
        )
        proposal = ActionProposal.create(
            tenant_id="tenant_system1_tool",
            principal_id="agent_caller",
            scope="langchain:tool:invoke",
            tool=self.name,
            arguments=raw_args,
            canonical_target=canonical_target,
            purpose=purpose,
        )
        context = f"{purpose}. Tool: {self.name} on {target}. Arguments: {raw_args}"
        interception = self.guard.evaluate_proposal(proposal, context_prompt=context)
        if interception.outcome != DecisionOutcome.ALLOW:
            raise SystemOneGuardBlockedException(interception)

        ledger = self.guard.ledger
        action_id = (
            interception.policy_decision.action_id
            if (interception and interception.policy_decision)
            else ""
        )
        receipt_digest = ""
        receipt = (
            interception.receipt
            or (interception.decision_result.receipt if interception.decision_result else None)
        )
        if receipt:
            receipt_digest = receipt.compute_digest()

        prop = getattr(interception, "proposal", None)
        if isinstance(prop, ActionProposal):
            canon_args = dict(prop.arguments)
        elif isinstance(getattr(interception, "proposal", None), (dict, Mapping)):
            canon_args = dict(interception.proposal)
        else:
            canon_args = dict(raw_args)

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
            dispatch_args = bound.args
            dispatch_kwargs = bound.kwargs
        else:
            dispatch_args = tuple(canon_args.get("_args", [])) if "_args" in canon_args else args
            dispatch_kwargs = {k: v for k, v in canon_args.items() if k != "_args" and k != "purpose"}

        exec_output = None
        exec_error = None
        try:
            if hasattr(self.tool, "invoke"):
                try:
                    exec_output = self.tool.invoke(*dispatch_args, **dispatch_kwargs)
                except TypeError:
                    if dispatch_kwargs and not dispatch_args:
                        exec_output = self.tool.invoke(dispatch_kwargs)
                    else:
                        raise
            elif hasattr(self.tool, "run"):
                try:
                    exec_output = self.tool.run(*dispatch_args, **dispatch_kwargs)
                except TypeError:
                    if dispatch_kwargs and not dispatch_args:
                        exec_output = self.tool.run(dispatch_kwargs)
                    else:
                        raise
            elif callable(self.tool):
                exec_output = self.tool(*dispatch_args, **dispatch_kwargs)
            else:
                raise TypeError(f"Target tool {self.tool!r} is neither callable nor implements invoke()/run()")
        except Exception as ex:
            exec_error = ex

        # Two-phase outcome recording
        if ledger is not None and receipt_digest:
            outcome_status = "FAILED" if exec_error is not None else "SUCCEEDED"
            try:
                pub_key = self.guard.engine.signing_key.public_key() if self.guard.engine.signing_key else None
                ledger.record_execution_outcome(
                    action_id=action_id,
                    receipt_digest=receipt_digest,
                    status=outcome_status,
                    result_payload={"result": str(exec_output)[:500]} if exec_output is not None else None,
                    error_message=str(exec_error) if exec_error is not None else None,
                    tenant_id="tenant_system1_tool",
                    principal_id="agent_caller",
                    scope="langchain:tool:invoke",
                    trusted_public_key=pub_key,
                )
            except Exception as le:
                raise SystemOneIndeterminateExecutionError(
                    f"LangChain tool executed but outcome recording failed (status: INDETERMINATE): {le}",
                    action_id=action_id,
                    receipt_digest=receipt_digest,
                    raw_result=exec_output,
                    underlying_error=le,
                    exec_error=exec_error,
                ) from le

        if exec_error is not None:
            raise exec_error

        return exec_output

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        """Asynchronous invocation interceptor."""
        target_fn = getattr(self.tool, "coroutine", None) or getattr(self.tool, "afunc", None) or getattr(self.tool, "_arun", None) or getattr(self.tool, "func", None) or getattr(self.tool, "_run", None) or (self.tool if callable(self.tool) else getattr(self.tool, "ainvoke", None))
        try:
            sig = inspect.signature(target_fn) if target_fn is not None else None
        except Exception:
            sig = None

        raw_args: Dict[str, Any] = {}
        bound = None
        if sig is not None:
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                raw_args = dict(bound.arguments)
            except Exception:
                bound = None
                raw_args = dict(kwargs)
                if args:
                    raw_args["_args"] = list(args)
        else:
            raw_args = dict(kwargs)
            if args:
                raw_args["_args"] = list(args)

        target = str(raw_args.get("input") or raw_args.get("query") or raw_args.get("path") or raw_args.get("target") or (args[0] if args else self.name))
        purpose = str(raw_args.get("purpose") or self.description or f"Tool invocation '{self.name}'")
        canonical_target = (
            f"sha256:{hashlib.sha256(target.encode('utf-8')).hexdigest()}"
            if len(target) > 4096
            else target
        )
        proposal = ActionProposal.create(
            tenant_id="tenant_system1_tool",
            principal_id="agent_caller",
            scope="langchain:tool:ainvoke",
            tool=self.name,
            arguments=raw_args,
            canonical_target=canonical_target,
            purpose=purpose,
        )
        context = f"{purpose}. Tool: {self.name} on {target}. Arguments: {raw_args}"
        interception = self.guard.evaluate_proposal(proposal, context_prompt=context)
        if interception.outcome != DecisionOutcome.ALLOW:
            raise SystemOneGuardBlockedException(interception)

        ledger = self.guard.ledger
        action_id = (
            interception.policy_decision.action_id
            if (interception and interception.policy_decision)
            else ""
        )
        receipt_digest = ""
        receipt = (
            interception.receipt
            or (interception.decision_result.receipt if interception.decision_result else None)
        )
        if receipt:
            receipt_digest = receipt.compute_digest()

        prop = getattr(interception, "proposal", None)
        if isinstance(prop, ActionProposal):
            canon_args = dict(prop.arguments)
        elif isinstance(getattr(interception, "proposal", None), (dict, Mapping)):
            canon_args = dict(interception.proposal)
        else:
            canon_args = dict(raw_args)

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
            dispatch_args = bound.args
            dispatch_kwargs = bound.kwargs
        else:
            dispatch_args = tuple(canon_args.get("_args", [])) if "_args" in canon_args else args
            dispatch_kwargs = {k: v for k, v in canon_args.items() if k != "_args" and k != "purpose"}

        exec_output = None
        exec_error = None
        try:
            if hasattr(self.tool, "ainvoke"):
                try:
                    exec_output = await self.tool.ainvoke(*dispatch_args, **dispatch_kwargs)
                except TypeError:
                    if dispatch_kwargs and not dispatch_args:
                        exec_output = await self.tool.ainvoke(dispatch_kwargs)
                    else:
                        raise
            elif hasattr(self.tool, "arun"):
                try:
                    exec_output = await self.tool.arun(*dispatch_args, **dispatch_kwargs)
                except TypeError:
                    if dispatch_kwargs and not dispatch_args:
                        exec_output = await self.tool.arun(dispatch_kwargs)
                    else:
                        raise
            elif hasattr(self.tool, "invoke"):
                try:
                    exec_output = self.tool.invoke(*dispatch_args, **dispatch_kwargs)
                except TypeError:
                    if dispatch_kwargs and not dispatch_args:
                        exec_output = self.tool.invoke(dispatch_kwargs)
                    else:
                        raise
            elif callable(self.tool):
                res = self.tool(*dispatch_args, **dispatch_kwargs)
                if inspect.isawaitable(res):
                    exec_output = await res
                else:
                    exec_output = res
            else:
                raise TypeError(f"Target tool {self.tool!r} is not callable asynchronously")
        except Exception as ex:
            exec_error = ex

        if ledger is not None and receipt_digest:
            outcome_status = "FAILED" if exec_error is not None else "SUCCEEDED"
            try:
                pub_key = self.guard.engine.signing_key.public_key() if self.guard.engine.signing_key else None
                ledger.record_execution_outcome(
                    action_id=action_id,
                    receipt_digest=receipt_digest,
                    status=outcome_status,
                    result_payload={"result": str(exec_output)[:500]} if exec_output is not None else None,
                    error_message=str(exec_error) if exec_error is not None else None,
                    tenant_id="tenant_system1_tool",
                    principal_id="agent_caller",
                    scope="langchain:tool:ainvoke",
                    trusted_public_key=pub_key,
                )
            except Exception as le:
                raise SystemOneIndeterminateExecutionError(
                    f"Tool executed asynchronously but outcome recording failed (status: INDETERMINATE): {le}",
                    action_id=action_id,
                    receipt_digest=receipt_digest,
                    raw_result=exec_output,
                    underlying_error=le,
                    exec_error=exec_error,
                ) from le

        if exec_error is not None:
            raise exec_error

        return exec_output


def wrap_langchain_tool(
    tool: Any,
    *,
    guard: Optional[SystemOneGuardHook] = None,
    alpha: float = 0.20,
    min_confidence: float = 0.50,
    tool_name: Optional[str] = None,
) -> SystemOneToolInterceptor:
    """Wraps any LangChain tool or callable function with System 1 fail-closed safety interceptor."""
    return SystemOneToolInterceptor(tool, guard=guard, alpha=alpha, min_confidence=min_confidence, tool_name=tool_name)


__all__ = [
    "SystemOneGuardBlockedException",
    "SystemOneIndeterminateExecutionError",
    "SystemOneSecurityException",
    "SystemOneGuardCallbackHandler",
    "SystemOneToolInterceptor",
    "wrap_langchain_tool",
]
