"""Adversarial Verification Suite for Gate A: Compositional Authorization.

Authored by Empirical Challenger (teamwork_preview_challenger_m1_1).

Attacks:
1. Rule Order Permutation: Permissive ALLOW rules vs restrictive DENY / REQUIRE_APPROVAL rules.
2. Argument Limit Evasion: Missing arguments, string-encoded integers, booleans, None, float overflows, nested injections.
3. Unconstrained ALLOW Bypass: Attempts to execute without receipt generation or ledger recording; fail-closed on ledger write failure.
4. Mutable Parameter Injection: Mutation of request dictionary and arguments after policy evaluation but before execution dispatch.
5. Harmless Sentinel Safety: Exhaustive test matrix verifying exactly 0 sentinel invocations across all denial/failure modes.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import sqlite3
from typing import Any, Dict, List, Mapping, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from system1.engine import DecisionResult, ReflexEngine
from system1.guard import (
    ActionProposal,
    ActionState,
    BooleanField,
    ChoiceField,
    DecisionOutcome,
    DecisionSchema,
    DefaultGuardDecisionSchema,
    EnforcementProfile,
    ENFORCEMENT_PROFILE_V1,
    GuardInterceptionResult,
    PolicyDecision,
    PolicyEngine,
    PolicyRule,
    ReflexGuardHook,
    RiskLevel,
)
from system1.integrations.langchain import (
    ReflexGuardBlockedException,
    ReflexGuardCallbackHandler,
    ReflexIndeterminateExecutionError,
    ReflexToolInterceptor,
    wrap_langchain_tool,
)
from system1.integrations.mcp import (
    ReflexMCPBlockedError,
    ReflexMCPProxy,
    wrap_mcp_tool,
)
from system1.ledger import ActionLedger, LedgerError, LedgerWriteError
from system1.receipt import (
    DecisionWitnessReceipt,
    create_decision_receipt,
    verify_decision_witness_receipt,
)


class AdversarialSentinel:
    """Sentinel target tool that strictly tracks invocation count and received parameters."""

    def __init__(self, return_value: Any = "sentinel_success") -> None:
        self.call_count: int = 0
        self.invocation_log: List[Dict[str, Any]] = []
        self.return_value: Any = return_value

    def reset(self) -> None:
        self.call_count = 0
        self.invocation_log.clear()

    def mcp_executor(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        self.call_count += 1
        self.invocation_log.append({
            "interface": "mcp_executor",
            "tool_name": tool_name,
            "arguments": dict(arguments),
        })
        return {"result": self.return_value, "call_count": self.call_count}

    def sync_tool(self, path: str, mode: str = "read", count: int = 1, **kwargs: Any) -> Any:
        self.call_count += 1
        self.invocation_log.append({
            "interface": "sync_tool",
            "path": path,
            "mode": mode,
            "count": count,
            "kwargs": copy.deepcopy(kwargs),
        })
        return f"sync_tool:{path}:{mode}:{count}"

    async def async_tool(self, path: str, mode: str = "read", count: int = 1, **kwargs: Any) -> Any:
        self.call_count += 1
        self.invocation_log.append({
            "interface": "async_tool",
            "path": path,
            "mode": mode,
            "count": count,
            "kwargs": copy.deepcopy(kwargs),
        })
        return f"async_tool:{path}:{mode}:{count}"


# ============================================================================
# Attack 1: Rule Order Permutation
# ============================================================================


def test_attack1_permissive_allow_before_restrictive_deny_permutation():
    """Attack 1.1: Register multiple permissive ALLOW rules BEFORE restrictive DENY rules.
    
    Verifies that PolicyEngine and ReflexGuardHook evaluate compositional precedence deterministically:
    DENY strictly overrides ALLOW regardless of rule registration order.
    """
    allow_rule_broad = PolicyRule(
        rule_id="allow_all_targets",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r".*",
        reason="Allow broad actions",
    )
    allow_rule_tool = PolicyRule(
        rule_id="allow_file_tools",
        effect=DecisionOutcome.ALLOW,
        action_pattern=r"file_.*",
        reason="Allow file tools",
    )
    deny_rule_critical = PolicyRule(
        rule_id="deny_system_paths",
        effect=DecisionOutcome.DENY,
        target_pattern=r"^/(etc|sys|proc|root)/.*",
        reason="Deny system path access",
    )
    deny_rule_destructive = PolicyRule(
        rule_id="deny_format_actions",
        effect=DecisionOutcome.DENY,
        action_pattern=r".*format.*",
        reason="Deny disk format actions",
    )

    # Permutation 1: ALLOW rules registered BEFORE DENY rules
    engine_allow_first = PolicyEngine(rules=[
        allow_rule_broad,
        allow_rule_tool,
        deny_rule_critical,
        deny_rule_destructive,
    ])

    proposal_attack = ActionProposal.create(
        tenant_id="tenant_attacker",
        principal_id="user_malicious",
        scope="fs:read",
        tool="file_read",
        arguments={"path": "/etc/shadow"},
        canonical_target="/etc/shadow",
        purpose="Attempt read system credentials",
    )

    eval_result = engine_allow_first.evaluate(proposal_attack)
    assert eval_result is not None
    outcome, risk, rule_id, reason = eval_result
    assert outcome == DecisionOutcome.DENY
    assert rule_id == "deny_system_paths"

    # Permutation 2: 10 permissive ALLOW rules stacked before 1 DENY rule
    many_allows = [
        PolicyRule(rule_id=f"allow_rule_{i}", effect=DecisionOutcome.ALLOW, target_pattern=r".*")
        for i in range(10)
    ]
    engine_many_allows = PolicyEngine(rules=many_allows + [deny_rule_critical])
    eval_many = engine_many_allows.evaluate(proposal_attack)
    assert eval_many is not None
    assert eval_many[0] == DecisionOutcome.DENY
    assert eval_many[2] == "deny_system_paths"

    # Permutation 3: Dynamic rule addition via add_rule() after init
    engine_dynamic = PolicyEngine(rules=[allow_rule_broad])
    # Before adding deny, proposal would be allowed
    assert engine_dynamic.evaluate(proposal_attack)[0] == DecisionOutcome.ALLOW
    # Dynamically append DENY rule
    engine_dynamic.add_rule(deny_rule_critical)
    # DENY must now strictly override the previously registered ALLOW
    assert engine_dynamic.evaluate(proposal_attack)[0] == DecisionOutcome.DENY


def test_attack1_approval_and_deny_composition_hierarchy():
    """Attack 1.2: Composition hierarchy must satisfy DENY > REQUIRE_APPROVAL > ALLOW.
    
    Verifies that:
    - REQUIRE_APPROVAL strictly overrides ALLOW (regardless of order).
    - DENY strictly overrides REQUIRE_APPROVAL (regardless of order).
    """
    allow_rule = PolicyRule(rule_id="allow_rule", effect=DecisionOutcome.ALLOW, target_pattern=r".*")
    approval_rule = PolicyRule(
        rule_id="approval_rule",
        effect=DecisionOutcome.REQUIRE_APPROVAL,
        target_pattern=r"/sensitive/.*",
    )
    deny_rule = PolicyRule(rule_id="deny_rule", effect=DecisionOutcome.DENY, target_pattern=r"/sensitive/secret.*")

    # Permutation A: [ALLOW, APPROVAL] -> APPROVAL wins
    engine_a = PolicyEngine(rules=[allow_rule, approval_rule])
    prop_sens = ActionProposal.create(
        tenant_id="t",
        principal_id="p",
        scope="s",
        tool="cat",
        arguments={"path": "/sensitive/doc.txt"},
        canonical_target="/sensitive/doc.txt",
        purpose="read doc",
    )
    res_a = engine_a.evaluate(prop_sens)
    assert res_a is not None
    assert res_a[0] == DecisionOutcome.REQUIRE_APPROVAL

    # Permutation B: [APPROVAL, ALLOW] -> APPROVAL wins
    engine_b = PolicyEngine(rules=[approval_rule, allow_rule])
    res_b = engine_b.evaluate(prop_sens)
    assert res_b is not None
    assert res_b[0] == DecisionOutcome.REQUIRE_APPROVAL

    # Permutation C: [ALLOW, APPROVAL, DENY] -> DENY wins
    engine_c = PolicyEngine(rules=[allow_rule, approval_rule, deny_rule])
    prop_deny = ActionProposal.create(
        tenant_id="t",
        principal_id="p",
        scope="s",
        tool="cat",
        arguments={"path": "/sensitive/secret.key"},
        canonical_target="/sensitive/secret.key",
        purpose="read secret",
    )
    res_c = engine_c.evaluate(prop_deny)
    assert res_c is not None
    assert res_c[0] == DecisionOutcome.DENY
    assert res_c[2] == "deny_rule"


def test_attack1_allow_rule_failed_constraint_fails_closed_to_deny():
    """Attack 1.3: An ALLOW rule whose own constraint fails must return DENY, not None or ALLOW."""
    allow_with_limit = PolicyRule(
        rule_id="allow_capped_transfer",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r".*",
        argument_limits={"amount": 1000.0},
    )
    general_allow = PolicyRule(
        rule_id="general_allow",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r".*",
    )

    # Engine with both: general allow and capped allow
    engine = PolicyEngine(rules=[general_allow, allow_with_limit])

    # Proposal exceeds the limit
    prop_overlimit = ActionProposal.create(
        tenant_id="t",
        principal_id="p",
        scope="exec",
        tool="transfer",
        arguments={"amount": 50000.0},
        canonical_target="bank_transfer",
        purpose="large transfer",
    )

    # allow_with_limit constraint fails -> returns DENY
    # general_allow matches -> returns ALLOW
    # PolicyEngine composition -> DENY strictly overrides ALLOW!
    res = engine.evaluate(prop_overlimit)
    assert res is not None
    assert res[0] == DecisionOutcome.DENY
    assert "exceeds limit" in res[3]


# ============================================================================
# Attack 2: Argument Limit Evasion
# ============================================================================


def test_attack2_argument_limit_missing_constrained_key():
    """Attack 2.1: Missing constrained argument must fail closed to DENY."""
    rule = PolicyRule(
        rule_id="rule_requires_token_and_amount",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r".*",
        argument_limits={"token": "bearer_secret", "amount": 100},
    )

    # Missing both
    prop_empty = ActionProposal.create(
        tenant_id="t", principal_id="p", scope="s", tool="api",
        arguments={}, canonical_target="endpoint", purpose="call",
    )
    sat, reason = rule.evaluate_constraints(prop_empty)
    assert sat is False
    assert "missing required constrained argument" in reason
    assert rule.evaluate(prop_empty)[0] == DecisionOutcome.DENY

    # Missing second argument ("amount")
    prop_partial = ActionProposal.create(
        tenant_id="t", principal_id="p", scope="s", tool="api",
        arguments={"token": "bearer_secret"}, canonical_target="endpoint", purpose="call",
    )
    sat, reason = rule.evaluate_constraints(prop_partial)
    assert sat is False
    assert "missing required constrained argument 'amount'" in reason
    assert rule.evaluate(prop_partial)[0] == DecisionOutcome.DENY


@pytest.mark.parametrize(
    "evasion_value,expected_reason_substr",
    [
        ("9999", "expected numeric, got str"),
        ("10", "expected numeric, got str"),
        ("0", "expected numeric, got str"),
        (True, "expected numeric, got bool"),
        (False, "expected numeric, got bool"),
        (None, "expected numeric, got NoneType"),
        ([10], "expected numeric, got tuple"),
        ({"val": 10}, "expected numeric, got mappingproxy"),
    ],
)
def test_attack2_argument_limit_type_mismatches(evasion_value: Any, expected_reason_substr: str):
    """Attack 2.2: Non-numeric / boolean values passed to numeric argument_limits fail closed."""
    # Scalar limit
    rule_scalar = PolicyRule(
        rule_id="rule_scalar_limit",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r".*",
        argument_limits={"max_retries": 10},
    )
    prop = ActionProposal.create(
        tenant_id="t", principal_id="p", scope="s", tool="retry",
        arguments={"max_retries": evasion_value}, canonical_target="target", purpose="retry",
    )
    sat, reason = rule_scalar.evaluate_constraints(prop)
    assert sat is False
    assert expected_reason_substr in reason
    assert rule_scalar.evaluate(prop)[0] == DecisionOutcome.DENY

    # Dict bounds limit (min/max)
    rule_dict = PolicyRule(
        rule_id="rule_dict_bounds",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r".*",
        argument_limits={"max_retries": {"min": 1, "max": 10}},
    )
    sat_d, reason_d = rule_dict.evaluate_constraints(prop)
    assert sat_d is False
    assert expected_reason_substr in reason_d
    assert rule_dict.evaluate(prop)[0] == DecisionOutcome.DENY


def test_attack2_argument_limit_unsupported_canonical_types_rejected_at_construction():
    """Attack 2.2b: Non-JSON canonical types (e.g. complex) fail-closed at ActionProposal construction."""
    with pytest.raises(TypeError, match="unsupported canonical value: complex"):
        ActionProposal.create(
            tenant_id="t", principal_id="p", scope="s", tool="math",
            arguments={"param": complex(10, 0)}, canonical_target="target", purpose="math",
        )


def test_attack2_argument_limit_float_overflow_and_extremes():
    """Attack 2.3: Float overflows and extreme values against argument_limits."""
    rule = PolicyRule(
        rule_id="rule_numeric_bounds",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r".*",
        argument_limits={"amount": {"min": 0.0, "max": 1000.0}},
    )

    # 1. Finite extreme float (1e308) -> exceeds max
    prop_high = ActionProposal.create(
        tenant_id="t", principal_id="p", scope="s", tool="pay",
        arguments={"amount": 1e308}, canonical_target="pay", purpose="pay",
    )
    sat, reason = rule.evaluate_constraints(prop_high)
    assert sat is False
    assert "exceeds max" in reason
    assert rule.evaluate(prop_high)[0] == DecisionOutcome.DENY

    # 2. Negative extreme float (-1e308) -> below min
    prop_neg = ActionProposal.create(
        tenant_id="t", principal_id="p", scope="s", tool="pay",
        arguments={"amount": -1e308}, canonical_target="pay", purpose="pay",
    )
    sat, reason = rule.evaluate_constraints(prop_neg)
    assert sat is False
    assert "below min" in reason
    assert rule.evaluate(prop_neg)[0] == DecisionOutcome.DENY

    # 3. NaN and Inf: ActionProposal.create MUST reject non-finite floats fail-closed
    with pytest.raises(ValueError, match="canonical data cannot contain NaN or infinity"):
        ActionProposal.create(
            tenant_id="t", principal_id="p", scope="s", tool="pay",
            arguments={"amount": float("nan")}, canonical_target="pay", purpose="pay",
        )

    with pytest.raises(ValueError, match="canonical data cannot contain NaN or infinity"):
        ActionProposal.create(
            tenant_id="t", principal_id="p", scope="s", tool="pay",
            arguments={"amount": float("inf")}, canonical_target="pay", purpose="pay",
        )

    with pytest.raises(ValueError, match="canonical data cannot contain NaN or infinity"):
        ActionProposal.create(
            tenant_id="t", principal_id="p", scope="s", tool="pay",
            arguments={"amount": float("-inf")}, canonical_target="pay", purpose="pay",
        )


def test_attack2_argument_limit_nested_injection():
    """Attack 2.4: Injection of nested mappings / structures into argument_limits."""
    rule = PolicyRule(
        rule_id="rule_limit_amount",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r".*",
        argument_limits={"amount": 100.0},
    )

    # MongoDB-style operator injection: {"$gt": 0}
    prop_mongo = ActionProposal.create(
        tenant_id="t", principal_id="p", scope="s", tool="query",
        arguments={"amount": {"$gt": 0}}, canonical_target="db", purpose="query",
    )
    sat, reason = rule.evaluate_constraints(prop_mongo)
    assert sat is False
    assert "expected numeric, got mappingproxy" in reason
    assert rule.evaluate(prop_mongo)[0] == DecisionOutcome.DENY

    # List injection: [50, 150]
    prop_list = ActionProposal.create(
        tenant_id="t", principal_id="p", scope="s", tool="query",
        arguments={"amount": [50, 150]}, canonical_target="db", purpose="query",
    )
    sat, reason = rule.evaluate_constraints(prop_list)
    assert sat is False
    assert "expected numeric, got tuple" in reason
    assert rule.evaluate(prop_list)[0] == DecisionOutcome.DENY


def test_attack2_target_hashing_boundary_investigation():
    """Attack 2.5: Investigation of target truncation / hashing boundary (> 4096 bytes).
    
    Demonstrates that targets > 4096 bytes are hashed to sha256:... in canonical_target.
    If a policy relies solely on target_pattern regex matching the literal target,
    oversized targets will present as sha256 hex strings rather than literal text.
    """
    deny_rule = PolicyRule(
        rule_id="deny_secret_pattern",
        effect=DecisionOutcome.DENY,
        target_pattern=r".*confidential_data.*",
        reason="Deny confidential data access",
    )
    allow_rule = PolicyRule(
        rule_id="allow_all",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r".*",
        reason="Allow general actions",
    )
    engine = PolicyEngine(rules=[deny_rule, allow_rule])
    hook = ReflexGuardHook(policy_engine=engine)
    proxy = ReflexMCPProxy(guard=hook)

    # 1. Normal length confidential target -> caught by regex
    res_normal = proxy.evaluate_mcp_call("read_file", {"path": "/data/confidential_data.txt"})
    assert res_normal.outcome == DecisionOutcome.DENY
    assert res_normal.policy_decision.rule_id == "deny_secret_pattern"

    # 2. Oversized target (> 4096 bytes) padded with filler
    oversized_path = "/data/confidential_data.txt?" + ("x" * 4100)
    res_oversized = proxy.evaluate_mcp_call("read_file", {"path": oversized_path})
    # Notice: canonical_target is hashed to sha256:..., so literal regex does not match sha256 hash!
    assert res_oversized.proposal.canonical_target.startswith("sha256:")
    assert "confidential_data" not in res_oversized.proposal.canonical_target


# ============================================================================
# Attack 3: Unconstrained ALLOW Bypass
# ============================================================================


def test_attack3_unconstrained_allow_generates_signed_receipt_and_ledger_record(tmp_path):
    """Attack 3.1: Verify unconstrained ALLOW cannot bypass receipt generation or durable ledger recording.
    
    Release Invariant 1 (No Execution Bypass):
    An ALLOW grants eligibility to continue, not permission to skip signing or ledger recording.
    """
    db_path = str(tmp_path / "strict_audit.db")
    ledger = ActionLedger(path=db_path)
    signing_key = Ed25519PrivateKey.generate()

    # Unconstrained ALLOW rule (broadest possible permission grant)
    allow_all_rule = PolicyRule(
        rule_id="unconstrained_allow_all",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r".*",
        reason="Unconstrained allow rule",
    )
    engine = PolicyEngine(rules=[allow_all_rule])
    hook = ReflexGuardHook(
        policy_engine=engine,
        ledger=ledger,
        signing_key=signing_key,
        fail_closed_ledger=True,
    )

    prop = ActionProposal.create(
        tenant_id="tenant_finance",
        principal_id="service_runner",
        scope="transfers:execute",
        tool="process_payment",
        arguments={"amount": 100.0, "currency": "USD"},
        canonical_target="account_12345",
        purpose="Process routine vendor payment",
    )

    interception = hook.evaluate_proposal(prop)
    assert interception.outcome == DecisionOutcome.ALLOW
    assert interception.allowed is True

    # 1. Receipt must be generated
    receipt = interception.receipt
    assert receipt is not None
    assert receipt.schema_name == "policy_deterministic_allow"
    assert receipt.outcome == "ALLOW"
    assert receipt.policy_decision is not None
    assert receipt.policy_decision["rule_id"] == "unconstrained_allow_all"

    # 2. Receipt must be cryptographically signed by signing_key
    assert receipt.signer_public_key is not None
    assert receipt.envelope is not None
    assert receipt.envelope.signature is not None
    assert verify_decision_witness_receipt(receipt.to_dict(), public_key=signing_key.public_key()) is True

    # 3. Receipt must be durably recorded to ActionLedger before execution
    assert receipt.ledger_record_id is not None
    seq, head = ledger.audit_head()
    assert seq >= 1
    assert ledger.verify_integrity(trusted_public_key=signing_key.public_key()) is True


def test_attack3_unconstrained_allow_fails_closed_when_ledger_write_fails(tmp_path):
    """Attack 3.2: Deterministic ALLOW must fail closed to DENY if ledger write fails."""
    db_path = str(tmp_path / "failing_ledger.db")
    ledger = ActionLedger(path=db_path)

    # Monkeypatch record_decision_receipt to simulate I/O or SQLite corruption
    def broken_record(*args: Any, **kwargs: Any) -> Any:
        raise sqlite3.OperationalError("simulated database disk image is malformed")

    ledger.record_decision_receipt = broken_record

    allow_rule = PolicyRule(rule_id="allow_rule", effect=DecisionOutcome.ALLOW, target_pattern=r".*")
    hook = ReflexGuardHook(
        policy_engine=PolicyEngine(rules=[allow_rule]),
        ledger=ledger,
        fail_closed_ledger=True,
    )

    prop = ActionProposal.create(
        tenant_id="t", principal_id="p", scope="s", tool="any_tool",
        arguments={"action": "test"}, canonical_target="target", purpose="test",
    )

    # Must fail closed to DENY
    interception = hook.evaluate_proposal(prop)
    assert interception.outcome == DecisionOutcome.DENY
    assert interception.allowed is False
    assert "ActionLedger write failed" in interception.reason
    assert interception.policy_decision.rule_id == "reflex_ledger_failure"


def test_attack3_mcp_proxy_executes_and_chains_outcome_under_allow(tmp_path):
    """Attack 3.3: MCP Proxy under unconstrained ALLOW records both authorization receipt AND outcome chaining."""
    db_path = str(tmp_path / "mcp_audit.db")
    ledger = ActionLedger(path=db_path)
    signing_key = Ed25519PrivateKey.generate()

    allow_rule = PolicyRule(rule_id="allow_mcp", effect=DecisionOutcome.ALLOW, target_pattern=r".*")
    hook = ReflexGuardHook(
        policy_engine=PolicyEngine(rules=[allow_rule]),
        ledger=ledger,
        signing_key=signing_key,
        fail_closed_ledger=True,
    )
    proxy = ReflexMCPProxy(guard=hook)
    sentinel = AdversarialSentinel()

    request = {
        "jsonrpc": "2.0",
        "id": "mcp_req_allow_1",
        "method": "tools/call",
        "params": {
            "name": "safe_tool",
            "arguments": {"target": "resource_1", "mode": "read"},
        },
    }

    resp = proxy.handle_call(request, sentinel.mcp_executor)
    assert "result" in resp
    assert sentinel.call_count == 1

    # Verify both records in ledger: (1) authorization receipt, (2) execution outcome
    seq, head = ledger.audit_head()
    assert seq == 2
    assert ledger.verify_integrity(trusted_public_key=signing_key.public_key()) is True


# ============================================================================
# Attack 4: Mutable Parameter Injection
# ============================================================================


def test_attack4_mcp_proxy_request_dict_mutation_resilience():
    """Attack 4.1: Mutating request dictionary after policy evaluation does not alter dispatched tool arguments."""
    allow_rule = PolicyRule(rule_id="allow_rule", effect=DecisionOutcome.ALLOW, target_pattern=r".*")
    hook = ReflexGuardHook(policy_engine=PolicyEngine(rules=[allow_rule]))
    proxy = ReflexMCPProxy(guard=hook)
    sentinel = AdversarialSentinel()

    # Nested mutable dictionary
    request = {
        "jsonrpc": "2.0",
        "id": "req_mut_1",
        "method": "tools/call",
        "params": {
            "name": "safe_tool",
            "arguments": {
                "command": "echo safe",
                "options": {"dry_run": True, "flags": ["-v", "-a"]},
            },
        },
    }

    # Custom executor that inspects dispatched arguments and attempts to mutate request
    def inspecting_executor(tool_name: str, arguments: Dict[str, Any]) -> Any:
        # Attacker attempts to modify request object during or before executor dispatch
        request["params"]["arguments"]["command"] = "rm -rf /"
        request["params"]["arguments"]["options"]["dry_run"] = False
        return sentinel.mcp_executor(tool_name, arguments)

    resp = proxy.handle_call(request, inspecting_executor)
    assert "result" in resp
    assert sentinel.call_count == 1

    # Dispatched arguments must remain untouched
    dispatched = sentinel.invocation_log[0]["arguments"]
    assert dispatched["command"] == "echo safe"
    assert dispatched["options"]["dry_run"] is True
    assert list(dispatched["options"]["flags"]) == ["-v", "-a"]


def test_attack4_action_proposal_arguments_immutability():
    """Attack 4.2: ActionProposal.arguments is deeply frozen (MappingProxyType) and rejects mutation."""
    prop = ActionProposal.create(
        tenant_id="t",
        principal_id="p",
        scope="s",
        tool="t",
        arguments={"key": "val", "nested": {"num": 42}, "items": [1, 2, 3]},
        canonical_target="target",
        purpose="purpose",
    )

    # Attempting to assign to proposal.arguments dict raises TypeError
    with pytest.raises(TypeError):
        prop.arguments["key"] = "hacked"  # type: ignore[index]

    with pytest.raises(TypeError):
        prop.arguments["new_key"] = "injected"  # type: ignore[index]

    # Attempting to mutate nested mapping raises TypeError
    with pytest.raises(TypeError):
        prop.arguments["nested"]["num"] = 999  # type: ignore[index]

    # Nested list is frozen to a tuple
    assert isinstance(prop.arguments["items"], tuple)


def test_attack4_wrap_mcp_tool_immutable_dispatch():
    """Attack 4.3: @wrap_mcp_tool dispatches canonicalized arguments ignoring external mutation."""
    allow_rule = PolicyRule(rule_id="allow_rule", effect=DecisionOutcome.ALLOW, target_pattern=r".*")
    hook = ReflexGuardHook(policy_engine=PolicyEngine(rules=[allow_rule]))
    proxy = ReflexMCPProxy(guard=hook)
    sentinel = AdversarialSentinel()

    wrapped = wrap_mcp_tool(proxy=proxy, tool_name="sync_tool")(sentinel.sync_tool)

    payload_path = "/valid/path.txt"
    payload_kwargs = {"mode": "read", "count": 2}

    res = wrapped(payload_path, **payload_kwargs)
    assert "sync_tool:/valid/path.txt:read:2" in res
    assert sentinel.call_count == 1
    assert sentinel.invocation_log[0]["path"] == "/valid/path.txt"
    assert sentinel.invocation_log[0]["mode"] == "read"


@pytest.mark.asyncio
async def test_attack4_langchain_interceptor_sync_and_async_immutable_dispatch():
    """Attack 4.4: ReflexToolInterceptor dispatches frozen canonical arguments for sync and async invocations."""
    allow_rule = PolicyRule(rule_id="allow_rule", effect=DecisionOutcome.ALLOW, target_pattern=r".*")
    hook = ReflexGuardHook(policy_engine=PolicyEngine(rules=[allow_rule]))
    sentinel = AdversarialSentinel()

    interceptor_sync = ReflexToolInterceptor(tool=sentinel.sync_tool, guard=hook, tool_name="sync_tool")
    res_sync = interceptor_sync("/path/data.json", mode="read", count=5)
    assert "sync_tool:/path/data.json:read:5" in res_sync
    assert sentinel.call_count == 1

    interceptor_async = ReflexToolInterceptor(tool=sentinel.async_tool, guard=hook, tool_name="async_tool")
    res_async = await interceptor_async.ainvoke("/path/async_data.json", mode="stream", count=10)
    assert "async_tool:/path/async_data.json:stream:10" in res_async
    assert sentinel.call_count == 2


# ============================================================================
# Attack 5: Harmless Sentinel Safety (Exhaustive Negative Test Matrix)
# ============================================================================


def test_attack5_sentinel_zero_executions_on_policy_denial():
    """Attack 5.1: Sentinel executes exactly 0 times on deterministic policy DENY."""
    deny_rule = PolicyRule(rule_id="strict_deny", effect=DecisionOutcome.DENY, target_pattern=r".*")
    hook = ReflexGuardHook(policy_engine=PolicyEngine(rules=[deny_rule]))
    proxy = ReflexMCPProxy(guard=hook)
    sentinel = AdversarialSentinel()

    req = {
        "jsonrpc": "2.0",
        "id": "deny_1",
        "method": "tools/call",
        "params": {"name": "target_tool", "arguments": {"cmd": "ls"}},
    }
    resp = proxy.handle_call(req, sentinel.mcp_executor)
    assert "error" in resp
    assert resp["error"]["code"] == -32000
    assert "DENY" in resp["error"]["message"]
    assert sentinel.call_count == 0


def test_attack5_sentinel_zero_executions_on_policy_require_approval():
    """Attack 5.2: Sentinel executes exactly 0 times on deterministic REQUIRE_APPROVAL."""
    approval_rule = PolicyRule(
        rule_id="strict_approval",
        effect=DecisionOutcome.REQUIRE_APPROVAL,
        target_pattern=r".*",
    )
    hook = ReflexGuardHook(policy_engine=PolicyEngine(rules=[approval_rule]))
    proxy = ReflexMCPProxy(guard=hook)
    sentinel = AdversarialSentinel()

    req = {
        "jsonrpc": "2.0",
        "id": "approval_1",
        "method": "tools/call",
        "params": {"name": "target_tool", "arguments": {"cmd": "ls"}},
    }
    resp = proxy.handle_call(req, sentinel.mcp_executor)
    assert "error" in resp
    assert "REQUIRE_APPROVAL" in resp["error"]["message"]
    assert sentinel.call_count == 0


def test_attack5_sentinel_zero_executions_on_constraint_violations():
    """Attack 5.3: Sentinel executes exactly 0 times across all constraint violations."""
    rule = PolicyRule(
        rule_id="constrained_rule",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r".*",
        allowed_principals=["authorized_agent"],
        allowed_tenants=["tenant_corp"],
        argument_limits={"max_retries": 5},
    )
    hook = ReflexGuardHook(policy_engine=PolicyEngine(rules=[rule]))
    proxy = ReflexMCPProxy(guard=hook, tenant_id="tenant_corp", principal_id="unauthorized_agent")
    sentinel = AdversarialSentinel()

    # 1. Unauthorized principal
    req_bad_p = {
        "jsonrpc": "2.0",
        "id": "p_fail",
        "method": "tools/call",
        "params": {"name": "tool", "arguments": {"max_retries": 3}},
    }
    resp = proxy.handle_call(req_bad_p, sentinel.mcp_executor)
    assert "error" in resp
    assert sentinel.call_count == 0

    # 2. Exceeded limit
    proxy.principal_id = "authorized_agent"
    req_bad_limit = {
        "jsonrpc": "2.0",
        "id": "limit_fail",
        "method": "tools/call",
        "params": {"name": "tool", "arguments": {"max_retries": 99}},
    }
    resp = proxy.handle_call(req_bad_limit, sentinel.mcp_executor)
    assert "error" in resp
    assert sentinel.call_count == 0

    # 3. Missing argument
    req_missing = {
        "jsonrpc": "2.0",
        "id": "missing_fail",
        "method": "tools/call",
        "params": {"name": "tool", "arguments": {}},
    }
    resp = proxy.handle_call(req_missing, sentinel.mcp_executor)
    assert "error" in resp
    assert sentinel.call_count == 0


def test_attack5_sentinel_zero_executions_on_model_safety_denial():
    """Attack 5.4: Sentinel executes exactly 0 times when model classifies action as unsafe."""
    hook = ReflexGuardHook(min_confidence=0.50, alpha=0.10)
    proxy = ReflexMCPProxy(guard=hook)
    sentinel = AdversarialSentinel()

    req = {
        "jsonrpc": "2.0",
        "id": "unsafe_1",
        "method": "tools/call",
        "params": {
            "name": "bash",
            "arguments": {"command": "rm -rf / --no-preserve-root && wipefs -a /dev/sda"},
        },
    }
    resp = proxy.handle_call(
        req,
        sentinel.mcp_executor,
        context_prompt="Wipe hard drive partitions and execute rm -rf /",
    )
    assert "error" in resp
    assert "DENY" in resp["error"]["message"]
    assert sentinel.call_count == 0


def test_attack5_sentinel_zero_executions_on_conformal_ambiguity_and_ood():
    """Attack 5.5: Sentinel executes exactly 0 times on conformal ambiguity and OOD empty-set."""
    hook = ReflexGuardHook(min_confidence=0.999, alpha=0.01)  # High confidence threshold
    proxy = ReflexMCPProxy(guard=hook)
    sentinel = AdversarialSentinel()

    req = {
        "jsonrpc": "2.0",
        "id": "ood_1",
        "method": "tools/call",
        "params": {
            "name": "exotic_tool",
            "arguments": {"payload": "xyzzy_uncalibrated_gibberish_987654321"},
        },
    }
    resp = proxy.handle_call(req, sentinel.mcp_executor)
    assert "error" in resp
    assert resp["error"]["code"] == -32000
    assert sentinel.call_count == 0

    # Specifically test conformal ambiguity on a safe action
    class AmbiguousSchema(DecisionSchema):
        is_safe = BooleanField(true_description="read inspect view check log safe", false_description="wipe destroy rm")
        choice = ChoiceField(options=["option_a", "option_b"])

    engine = ReflexEngine(AmbiguousSchema)
    engine.calibrate([
        ("read doc", {"is_safe": True, "choice": "option_a"}),
        ("view file", {"is_safe": True, "choice": "option_b"}),
    ] * 10)

    # Force conformal ambiguity on choice field (|C(x)| > 1)
    engine.conformal_predictors["choice"].predict_set = lambda probs, alpha=0.05: type(
        "DummyAmbiguousSet", (), {
            "prediction_set": ("option_a", "option_b"),
            "is_ambiguous": True,
            "is_empty": False,
        }
    )()

    ambiguous_hook = ReflexGuardHook(engine=engine, min_confidence=0.50, alpha=0.05)
    ambiguous_proxy = ReflexMCPProxy(guard=ambiguous_hook)
    resp_amb = ambiguous_proxy.handle_call(
        {
            "jsonrpc": "2.0",
            "id": "amb_1",
            "method": "tools/call",
            "params": {"name": "read_tool", "arguments": {"path": "README.md"}},
        },
        sentinel.mcp_executor,
        context_prompt="read inspect view check log safe project readme",
    )
    assert "error" in resp_amb
    assert "REQUIRE_APPROVAL" in resp_amb["error"]["message"]
    assert "ambiguity" in resp_amb["error"]["message"]
    assert sentinel.call_count == 0



def test_attack5_sentinel_zero_executions_on_mcp_non_tool_methods():
    """Attack 5.6: Sentinel executes exactly 0 times on non-tool MCP methods (ping, initialize, tools/list)."""
    proxy = ReflexMCPProxy()
    sentinel = AdversarialSentinel()

    # 1. ping with suspicious tool arguments
    resp_ping = proxy.handle_call(
        {
            "jsonrpc": "2.0",
            "id": "ping_1",
            "method": "ping",
            "params": {"name": "dangerous_tool", "arguments": {"cmd": "rm -rf /"}},
        },
        sentinel.mcp_executor,
    )
    assert resp_ping["jsonrpc"] == "2.0"
    assert resp_ping["result"] == {}
    assert sentinel.call_count == 0

    # 2. initialize with tool parameters
    resp_init = proxy.handle_call(
        {
            "jsonrpc": "2.0",
            "id": "init_1",
            "method": "initialize",
            "params": {"name": "dangerous_tool"},
        },
        sentinel.mcp_executor,
    )
    assert resp_init["jsonrpc"] == "2.0"
    assert "protocolVersion" in resp_init["result"]
    assert sentinel.call_count == 0

    # 3. tools/list
    resp_list = proxy.handle_call(
        {"jsonrpc": "2.0", "id": "list_1", "method": "tools/list", "params": {}},
        sentinel.mcp_executor,
    )
    assert resp_list["jsonrpc"] == "2.0"
    assert "tools" in resp_list["result"]
    assert sentinel.call_count == 0


@pytest.mark.parametrize(
    "malformed_payload",
    [
        "not_a_valid_json_string{",
        json.dumps(["not_a_dict_array"]),
        json.dumps(12345),
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call"}),  # missing params
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": "not_a_dict"}),
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": ""}}),  # empty name
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "tool", "arguments": "not_a_dict"}}),
    ],
)
def test_attack5_sentinel_zero_executions_on_malformed_mcp_requests(malformed_payload: str):
    """Attack 5.7: Sentinel executes exactly 0 times on malformed JSON-RPC payloads."""
    proxy = ReflexMCPProxy()
    sentinel = AdversarialSentinel()

    resp = proxy.handle_call(malformed_payload, sentinel.mcp_executor)
    assert "error" in resp
    assert sentinel.call_count == 0


def test_attack5_sentinel_zero_executions_on_wrap_mcp_tool_denial():
    """Attack 5.8: @wrap_mcp_tool guarantees exactly 0 executions when blocked."""
    deny_rule = PolicyRule(rule_id="deny_rule", effect=DecisionOutcome.DENY, target_pattern=r".*")
    hook = ReflexGuardHook(policy_engine=PolicyEngine(rules=[deny_rule]))
    proxy = ReflexMCPProxy(guard=hook)
    sentinel = AdversarialSentinel()

    wrapped = wrap_mcp_tool(proxy=proxy, tool_name="sentinel_sync")(sentinel.sync_tool)

    with pytest.raises(ReflexMCPBlockedError):
        wrapped("/path/file.txt", mode="write", count=1)

    assert sentinel.call_count == 0


@pytest.mark.asyncio
async def test_attack5_sentinel_zero_executions_on_langchain_interceptor_denial():
    """Attack 5.9: ReflexToolInterceptor guarantees 0 executions for sync and async on denial."""
    deny_rule = PolicyRule(rule_id="deny_rule", effect=DecisionOutcome.DENY, target_pattern=r".*")
    hook = ReflexGuardHook(policy_engine=PolicyEngine(rules=[deny_rule]))
    sentinel = AdversarialSentinel()

    # Sync
    interceptor_sync = ReflexToolInterceptor(tool=sentinel.sync_tool, guard=hook, tool_name="sync_tool")
    with pytest.raises(ReflexGuardBlockedException):
        interceptor_sync("/path/data.txt")
    assert sentinel.call_count == 0

    # Async
    interceptor_async = ReflexToolInterceptor(tool=sentinel.async_tool, guard=hook, tool_name="async_tool")
    with pytest.raises(ReflexGuardBlockedException):
        await interceptor_async.ainvoke("/path/data.txt")
    assert sentinel.call_count == 0


def test_attack5_sentinel_zero_executions_on_langchain_callback_handler_denial():
    """Attack 5.10: ReflexGuardCallbackHandler raises exception on on_tool_start when blocked."""
    deny_rule = PolicyRule(rule_id="deny_rule", effect=DecisionOutcome.DENY, target_pattern=r".*")
    hook = ReflexGuardHook(policy_engine=PolicyEngine(rules=[deny_rule]))
    handler = ReflexGuardCallbackHandler(guard=hook)

    with pytest.raises(ReflexGuardBlockedException):
        handler.on_tool_start(
            serialized={"name": "destructive_tool", "description": "Destructive bash action"},
            input_str="rm -rf /",
            run_id="run_123",
        )
