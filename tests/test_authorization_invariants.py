"""Tests for Authorization Invariants (M1/M2).

Verifies:
- Invariant 1: MCP protocol dispatch & non-tool method handling (0 executor calls).
- Invariant 2: Elimination of [:256] truncation and inspect.signature parameter binding.
- Deterministic Policy Override: Pre-inference reference monitor policy engine veto.
- Harmless Sentinel Tool: Exactly 0 side effects / executor invocations across all failure modes.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from reflex import (
    ActionLedger,
    ActionProposal,
    DecisionOutcome,
    PolicyEngine,
    PolicyRule,
    SystemOneEngine,
    SystemOneGuardBlockedException,
    SystemOneGuardCallbackHandler,
    SystemOneGuardHook,
    SystemOneMCPProxy,
    SystemOneToolInterceptor,
    RiskLevel,
    wrap_langchain_tool,
    wrap_mcp_tool,
)


class SentinelExecutor:
    """Sentinel tool executor that tracks all calls and side effects."""

    def __init__(self):
        self.call_count = 0
        self.calls = []

    def __call__(self, name: str, arguments: dict):
        self.call_count += 1
        self.calls.append((name, arguments))
        return {"result": f"executed {name}", "call_count": self.call_count}

    def execute_tool(self, path: str, mode: str = "read", retries: int = 3):
        self.call_count += 1
        self.calls.append((path, mode, retries))
        return f"read {path} with mode={mode}"


# ---------------------------------------------------------------------------
# Invariant 1: MCP Protocol Dispatch & Non-tool Method Handling
# ---------------------------------------------------------------------------


def test_invariant_1_mcp_ping_does_not_invoke_executor():
    """Non-tool method 'ping' must return standard JSON-RPC response with 0 executor calls."""
    proxy = SystemOneMCPProxy()
    sentinel = SentinelExecutor()

    req = {
        "jsonrpc": "2.0",
        "id": "req-ping-1",
        "method": "ping",
        "params": {"some_arg": "value"},
    }

    response = proxy.handle_call(req, sentinel)

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == "req-ping-1"
    assert "result" in response
    assert response["result"] == {}
    assert sentinel.call_count == 0


def test_invariant_1_mcp_initialize_does_not_invoke_executor():
    """Non-tool method 'initialize' returns protocol handshake with 0 executor calls."""
    proxy = SystemOneMCPProxy()
    sentinel = SentinelExecutor()

    req = {
        "jsonrpc": "2.0",
        "id": "req-init-1",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    }

    response = proxy.handle_call(req, sentinel)

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == "req-init-1"
    assert response["result"]["protocolVersion"] == "2024-11-05"
    assert response["result"]["serverInfo"]["name"] == "reflex-mcp-guard"
    assert sentinel.call_count == 0


def test_invariant_1_mcp_tools_list_does_not_invoke_executor():
    """Non-tool method 'tools/list' returns tools catalog with 0 executor calls."""
    proxy = SystemOneMCPProxy()
    sentinel = SentinelExecutor()

    req = {
        "jsonrpc": "2.0",
        "id": "req-list-1",
        "method": "tools/list",
        "params": {},
    }

    response = proxy.handle_call(req, sentinel)

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == "req-list-1"
    assert "tools" in response["result"]
    assert sentinel.call_count == 0


def test_invariant_1_mcp_tools_call_dispatches_when_allowed():
    """Method 'tools/call' dispatches to executor when allowed by guard."""
    hook = SystemOneGuardHook(min_confidence=0.50, alpha=0.10)
    proxy = SystemOneMCPProxy(guard=hook)
    sentinel = SentinelExecutor()

    req = {
        "jsonrpc": "2.0",
        "id": "req-tool-1",
        "method": "tools/call",
        "params": {
            "name": "read_file",
            "arguments": {"target": "/Volumes/Storage/project/README.md", "mode": "read"},
        },
    }

    response = proxy.handle_call(
        req,
        sentinel,
        context_prompt="Inspect read-only project documentation in README.md",
    )

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == "req-tool-1"
    assert "result" in response
    assert sentinel.call_count == 1
    assert sentinel.calls[0] == ("read_file", {"target": "/Volumes/Storage/project/README.md", "mode": "read"})


# ---------------------------------------------------------------------------
# Invariant 2: No Argument Truncation & inspect.signature Parameter Binding
# ---------------------------------------------------------------------------


def test_invariant_2_langchain_callback_no_character_truncation():
    """Eliminates [:256] truncation in SystemOneGuardCallbackHandler.on_tool_start."""
    hook = SystemOneGuardHook(min_confidence=0.50, alpha=0.10)
    handler = SystemOneGuardCallbackHandler(guard=hook)

    # 500 character payload
    long_payload = "A" * 500
    serialized = {"name": "read_file", "description": "Inspect read-only project documentation"}

    try:
        handler.on_tool_start(serialized, long_payload)
    except SystemOneGuardBlockedException:
        pass

    assert len(handler.interceptions) == 1
    interception = handler.interceptions[0]
    # Verify the proposal argument preserves the entire 500 characters
    assert interception.proposal.arguments.get("input") == long_payload
    # Verify the canonical target is also the entire 500 characters
    assert interception.proposal.canonical_target == long_payload
    assert len(interception.proposal.arguments["input"]) == 500


def test_invariant_2_wrap_mcp_tool_signature_binding():
    """wrap_mcp_tool uses inspect.signature to bind positional args and defaults into named arguments."""
    proxy = SystemOneMCPProxy()

    captured_arguments = {}

    def mock_evaluate(tool_name, arguments, **kwargs):
        captured_arguments.update(arguments)
        mock_interception = MagicMock()
        mock_interception.outcome = DecisionOutcome.ALLOW
        mock_interception.reason = "Allowed by test"
        mock_interception.policy_decision = None
        mock_interception.decision_result = None
        return mock_interception

    proxy.evaluate_mcp_call = mock_evaluate

    sentinel = SentinelExecutor()
    wrapped = wrap_mcp_tool(proxy=proxy, tool_name="execute_tool")(sentinel.execute_tool)

    # Call with 1 positional arg; mode and retries should be bound from defaults
    res = wrapped("/tmp/data.csv")

    assert res == "read /tmp/data.csv with mode=read"
    assert sentinel.call_count == 1
    assert captured_arguments.get("path") == "/tmp/data.csv"
    assert captured_arguments.get("mode") == "read"
    assert captured_arguments.get("retries") == 3
    assert "_args" not in captured_arguments


def test_invariant_2_langchain_interceptor_signature_binding():
    """SystemOneToolInterceptor binds defaults and positional arguments into named dictionary."""
    hook = SystemOneGuardHook(min_confidence=0.50, alpha=0.10)
    sentinel = SentinelExecutor()
    interceptor = SystemOneToolInterceptor(tool=sentinel.execute_tool, guard=hook, tool_name="read_file")

    captured_proposals = []

    def intercept_eval(proposal, context_prompt=None):
        captured_proposals.append(proposal)
        mock_res = MagicMock()
        mock_res.outcome = DecisionOutcome.ALLOW
        mock_res.reason = "Allowed for signature test"
        mock_res.policy_decision = None
        mock_res.decision_result = None
        return mock_res

    hook.evaluate_proposal = intercept_eval

    # Call with positional args
    res = interceptor("/Volumes/Storage/project/README.md", "read")

    assert res == "read /Volumes/Storage/project/README.md with mode=read"
    assert sentinel.call_count == 1
    assert len(captured_proposals) == 1
    proposal = captured_proposals[0]
    assert proposal.arguments["path"] == "/Volumes/Storage/project/README.md"
    assert proposal.arguments["mode"] == "read"
    assert proposal.arguments["retries"] == 3


# ---------------------------------------------------------------------------
# Deterministic Policy Override: Reference Monitor Veto Authority
# ---------------------------------------------------------------------------


def test_deterministic_policy_override_vetoes_before_inference():
    """Deterministic policy vetoes dangerous targets regardless of model confidence or caching."""
    # Define a veto rule blocking access to /etc/shadow or root paths
    veto_rule = PolicyRule(
        rule_id="veto_root_access",
        effect=DecisionOutcome.DENY,
        target_pattern=r"^(/etc/.*|/root/.*|/bin/.*)$",
        reason="Veto: Access to root or sensitive system directories is strictly prohibited",
    )
    policy_engine = PolicyEngine(rules=[veto_rule])

    hook = SystemOneGuardHook(
        policy_engine=policy_engine,
        min_confidence=0.50,
        alpha=0.10,
    )

    # Even for a benign read_file tool that would normally be allowed
    proposal = ActionProposal.create(
        tenant_id="tenant_001",
        principal_id="principal_001",
        scope="system:read",
        tool="read_file",
        arguments={"path": "/etc/shadow"},
        canonical_target="/etc/shadow",
        purpose="Read file contents",
    )

    interception = hook.evaluate_proposal(
        proposal,
        context_prompt="Inspect read-only project documentation",
    )

    assert interception.outcome == DecisionOutcome.DENY
    assert "Veto: Access to root or sensitive system directories" in interception.reason
    assert interception.policy_decision.risk == RiskLevel.IRREVERSIBLE


def test_deterministic_policy_scoping_by_tenant_and_action():
    """Policy rules correctly scope by tenant, principal, and action regex."""
    rule_tenant = PolicyRule(
        rule_id="block_prod_writes",
        effect=DecisionOutcome.DENY,
        tenant_pattern=r"^prod-.*",
        action_pattern=r"^(delete|drop|wipe)",
        reason="Production destructive actions require approval/vetoed",
    )
    engine = PolicyEngine(rules=[rule_tenant])
    hook = SystemOneGuardHook(policy_engine=engine)

    # Tenant prod-01 attempting delete
    prop_prod = ActionProposal.create(
        tenant_id="prod-01",
        principal_id="user_admin",
        scope="db",
        tool="delete_table",
        arguments={"table": "users"},
        purpose="Delete table",
    )
    res_prod = hook.evaluate_proposal(prop_prod)
    assert res_prod.outcome == DecisionOutcome.DENY
    assert "Production destructive actions" in res_prod.reason

    # Tenant staging-01 attempting delete (not matched by rule)
    prop_staging = ActionProposal.create(
        tenant_id="staging-01",
        principal_id="user_admin",
        scope="db",
        tool="delete_table",
        arguments={"table": "users"},
        purpose="Delete table",
    )
    res_staging = hook.evaluate_proposal(prop_staging)
    # Staging does not match the deterministic policy veto, so it falls through to engine
    assert res_staging.policy_decision.rule_id != "block_prod_writes"


# ---------------------------------------------------------------------------
# Harmless Sentinel Tool: 0 Invocations Across All Failure Modes
# ---------------------------------------------------------------------------


def test_sentinel_zero_calls_on_deny():
    """Sentinel tool is never invoked when System 1 Guard returns DENY."""
    proxy = SystemOneMCPProxy()
    sentinel = SentinelExecutor()

    # Request to destroy system root
    req = {
        "jsonrpc": "2.0",
        "id": "req-deny-1",
        "method": "tools/call",
        "params": {
            "name": "bash_exec",
            "arguments": {"cmd": "rm -rf / --no-preserve-root"},
        },
    }

    response = proxy.handle_call(req, sentinel)

    assert "error" in response
    assert response["error"]["code"] == -32000
    assert "DENY" in response["error"]["message"]
    assert sentinel.call_count == 0


def test_sentinel_zero_calls_on_malformed_jsonrpc():
    """Sentinel tool is never invoked on malformed or non-dict JSON-RPC requests."""
    proxy = SystemOneMCPProxy()
    sentinel = SentinelExecutor()

    # 1. Non-dict request
    resp1 = proxy.handle_call(["invalid", "payload"], sentinel)
    assert resp1["error"]["code"] == -32600
    assert sentinel.call_count == 0

    # 2. Missing params or invalid params type
    resp2 = proxy.handle_call(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": "not-a-dict"},
        sentinel,
    )
    assert resp2["error"]["code"] == -32602
    assert sentinel.call_count == 0

    # 3. Missing tool name
    resp3 = proxy.handle_call(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"arguments": {}}},
        sentinel,
    )
    assert resp3["error"]["code"] == -32602
    assert sentinel.call_count == 0


def test_sentinel_zero_calls_on_langchain_blocked_exception():
    """LangChain wrapped tool throws exception and executes 0 times when guard blocks."""
    hook = SystemOneGuardHook()
    sentinel = SentinelExecutor()
    interceptor = wrap_langchain_tool(sentinel.execute_tool, guard=hook, tool_name="bash_exec")

    with pytest.raises(SystemOneGuardBlockedException):
        interceptor(path="/bin/sh", mode="rm -rf /")

    assert sentinel.call_count == 0
