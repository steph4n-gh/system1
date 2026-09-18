"""Comprehensive Verification Suite for Milestone 1: Gate A & Gate B.

Covers:
- Gate A: Compositional Authorization (P0)
  1. Deterministic Rule Composition in PolicyEngine (DENY > REQUIRE_APPROVAL > ALLOW regardless of order).
  2. Decouple Rule Applicability from Constraint Evaluation in PolicyRule.
  3. Fail-Closed Constrained Arguments in argument_limits (missing keys, type mismatches, boolean values).
  4. Release Invariant 1: No Execution Bypass on deterministic ALLOW (signed receipt + durable ledger entry).
  5. Immutable Canonical Action Dispatch in ReflexMCPProxy, wrap_mcp_tool, and ReflexToolInterceptor.
  6. Harmless Sentinel Execution (0 invocations on all rejections and ledger failures).

- Gate B: Authenticated Final Authorization & Durability (P0)
  1. Explicit EnforcementProfile (reflex.witness.enforcement.v1 / strict_attested_durable_v1).
  2. Comprehensive Authorization Receipt Binding (PolicyDecision, arguments, principal, tenant, scope, risk, epoch).
  3. Fail-Closed Ledger Durability (reject :memory:, verify_integrity with trusted_public_key).
  4. Complete Two-Phase Outcome Chaining & Structured INDETERMINATE across MCP and LangChain adapters.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import sqlite3
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from system1.guard import (
    ActionProposal,
    DecisionOutcome,
    EnforcementProfile,
    ENFORCEMENT_PROFILE_V1,
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
    create_decision_receipt,
    verify_decision_witness_receipt,
)


class SentinelTarget:
    """Mock target that records invocations and arguments."""

    def __init__(self, return_value: str = "sentinel_ok"):
        self.call_count = 0
        self.last_tool = None
        self.last_args = None
        self.last_kwargs = None
        self.return_value = return_value

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        self.last_args = args
        self.last_kwargs = kwargs
        return self.return_value

    def mcp_executor(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        self.call_count += 1
        self.last_tool = tool_name
        self.last_args = (tool_name, arguments)
        self.last_kwargs = arguments
        return {"result": self.return_value}

    def execute_file(self, path: str, mode: str = "read", retries: int = 3) -> str:
        self.call_count += 1
        self.last_args = (path, mode, retries)
        self.last_kwargs = {"path": path, "mode": mode, "retries": retries}
        return f"file:{path}:{mode}:{retries}"

    async def aexecute_file(self, path: str, mode: str = "read", retries: int = 3) -> str:
        self.call_count += 1
        self.last_args = (path, mode, retries)
        self.last_kwargs = {"path": path, "mode": mode, "retries": retries}
        return f"afile:{path}:{mode}:{retries}"


# ===========================================================================
# Gate A.1: Deterministic Rule Composition in PolicyEngine
# ===========================================================================


def test_gate_a_rule_composition_deny_overrides_allow_regardless_of_order():
    """DENY rule strictly overrides ALLOW rule regardless of registration order."""
    allow_rule = PolicyRule(
        rule_id="allow_all_tools",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r".*",
        reason="Allow broad actions",
    )
    deny_rule = PolicyRule(
        rule_id="deny_sensitive_path",
        effect=DecisionOutcome.DENY,
        target_pattern=r"/etc/.*",
        reason="Deny sensitive path",
    )

    # Order 1: ALLOW registered before DENY
    engine_allow_first = PolicyEngine(rules=[allow_rule, deny_rule])
    proposal = ActionProposal.create(
        tenant_id="test_t",
        principal_id="test_p",
        scope="exec",
        tool="cat",
        arguments={"path": "/etc/shadow"},
        canonical_target="/etc/shadow",
        purpose="Read system credentials",
    )
    res = engine_allow_first.evaluate(proposal)
    assert res is not None
    outcome, risk, rule_id, reason = res
    assert outcome == DecisionOutcome.DENY
    assert rule_id == "deny_sensitive_path"

    # Order 2: DENY registered before ALLOW
    engine_deny_first = PolicyEngine(rules=[deny_rule, allow_rule])
    res2 = engine_deny_first.evaluate(proposal)
    assert res2 is not None
    assert res2[0] == DecisionOutcome.DENY
    assert res2[2] == "deny_sensitive_path"


def test_gate_a_rule_composition_approval_overrides_allow():
    """REQUIRE_APPROVAL strictly overrides ALLOW when both match."""
    allow_rule = PolicyRule(
        rule_id="allow_read",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r"/data/.*",
        reason="Allow data access",
    )
    approval_rule = PolicyRule(
        rule_id="approval_high_volume",
        effect=DecisionOutcome.REQUIRE_APPROVAL,
        target_pattern=r".*",
        reason="Require human approval for all accesses",
    )
    engine = PolicyEngine(rules=[allow_rule, approval_rule])
    proposal = ActionProposal.create(
        tenant_id="test_t",
        principal_id="test_p",
        scope="exec",
        tool="cat",
        arguments={"path": "/data/records.csv"},
        canonical_target="/data/records.csv",
        purpose="Inspect data file",
    )
    res = engine.evaluate(proposal)
    assert res is not None
    assert res[0] == DecisionOutcome.REQUIRE_APPROVAL
    assert res[2] == "approval_high_volume"


# ===========================================================================
# Gate A.2 & A.3: Decoupled Rule Applicability and Fail-Closed Constraints
# ===========================================================================


def test_gate_a_decouple_applicability_and_evaluate_constraints():
    """Rule applicability (applies_to) does not short-circuit constraints (evaluate_constraints)."""
    rule = PolicyRule(
        rule_id="bounded_numeric_limit",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r"/api/transfer.*",
        allowed_principals=["finance_agent"],
        argument_limits={"amount": 1000.0},
    )

    # 1. Matching target pattern and matching principal, valid amount -> applies and satisfies constraints
    prop_valid = ActionProposal.create(
        tenant_id="t1",
        principal_id="finance_agent",
        scope="exec",
        tool="transfer",
        arguments={"amount": 500.0},
        canonical_target="/api/transfer_funds",
        purpose="Process valid transfer",
    )
    assert rule.applies_to(prop_valid) is True
    assert rule.evaluate_constraints(prop_valid) == (True, "")
    assert rule.evaluate(prop_valid) is not None
    assert rule.evaluate(prop_valid)[0] == DecisionOutcome.ALLOW

    # 2. Unauthorized principal: applies_to is True (matches target), but evaluate_constraints fails closed
    prop_bad_principal = ActionProposal.create(
        tenant_id="t1",
        principal_id="unauthorized_actor",
        scope="exec",
        tool="transfer",
        arguments={"amount": 500.0},
        canonical_target="/api/transfer_funds",
        purpose="Unauthorized transfer attempt",
    )
    assert rule.applies_to(prop_bad_principal) is True
    satisfied, reason = rule.evaluate_constraints(prop_bad_principal)
    assert satisfied is False
    assert "not in allowed_principals" in reason
    # Rule evaluate fails closed to DENY on constraint failure
    eval_res = rule.evaluate(prop_bad_principal)
    assert eval_res is not None
    assert eval_res[0] == DecisionOutcome.DENY


def test_gate_a_fail_closed_constrained_arguments():
    """Missing arguments, type mismatches, and booleans in argument_limits fail closed."""
    rule = PolicyRule(
        rule_id="strict_numeric_limits",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r".*",
        argument_limits={"max_count": 10, "threshold": 0.95},
    )

    # Missing constrained argument
    prop_missing = ActionProposal.create(
        tenant_id="t1",
        principal_id="p1",
        scope="exec",
        tool="batch",
        arguments={"threshold": 0.80},  # missing max_count
        canonical_target="job_1",
        purpose="Run batch job",
    )
    sat, reason = rule.evaluate_constraints(prop_missing)
    assert sat is False
    assert "missing required constrained argument 'max_count'" in reason
    assert rule.evaluate(prop_missing)[0] == DecisionOutcome.DENY

    # Type mismatch: string passed for numeric limit
    prop_str = ActionProposal.create(
        tenant_id="t1",
        principal_id="p1",
        scope="exec",
        tool="batch",
        arguments={"max_count": "100", "threshold": 0.80},
        canonical_target="job_1",
        purpose="Run batch job",
    )
    sat, reason = rule.evaluate_constraints(prop_str)
    assert sat is False
    assert "expected numeric" in reason
    assert rule.evaluate(prop_str)[0] == DecisionOutcome.DENY

    # Type mismatch: boolean passed for numeric limit (in Python, isinstance(True, int) is True)
    prop_bool = ActionProposal.create(
        tenant_id="t1",
        principal_id="p1",
        scope="exec",
        tool="batch",
        arguments={"max_count": True, "threshold": 0.80},
        canonical_target="job_1",
        purpose="Run batch job",
    )
    sat, reason = rule.evaluate_constraints(prop_bool)
    assert sat is False
    assert "expected numeric" in reason and "bool" in reason
    assert rule.evaluate(prop_bool)[0] == DecisionOutcome.DENY


# ===========================================================================
# Gate A.4: Release Invariant 1: No Execution Bypass on Deterministic ALLOW
# ===========================================================================


def test_gate_a_deterministic_allow_issues_signed_receipt_and_ledger_entry(tmp_path):
    """Deterministic ALLOW issues authenticated witness receipt and records to durable ledger before execution."""
    db_path = str(tmp_path / "allow_audit.db")
    ledger = ActionLedger(path=db_path)
    signing_key = Ed25519PrivateKey.generate()

    allow_rule = PolicyRule(
        rule_id="trusted_read_allow",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r"^/safe/.*",
        reason="Trusted read path allowed",
    )
    policy_engine = PolicyEngine(rules=[allow_rule])

    hook = ReflexGuardHook(
        policy_engine=policy_engine,
        ledger=ledger,
        signing_key=signing_key,
        fail_closed_ledger=True,
    )

    prop = ActionProposal.create(
        tenant_id="corp_tenant",
        principal_id="agent_reader",
        scope="fs:read",
        tool="read_file",
        arguments={"path": "/safe/data.json"},
        canonical_target="/safe/data.json",
        purpose="Read safe customer record",
    )

    result = hook.evaluate_proposal(prop)
    assert result.outcome == DecisionOutcome.ALLOW
    assert result.allowed is True

    # Invariant 1: Decision has authenticated receipt
    receipt = result.receipt
    assert receipt is not None
    assert receipt.signer_public_key is not None
    assert receipt.policy_decision is not None
    assert receipt.policy_decision["outcome"] == "ALLOW"
    assert receipt.policy_decision["rule_id"] == "trusted_read_allow"
    assert receipt.outcome == "ALLOW"

    # Invariant 1: Durably recorded in ledger before execution
    assert receipt.ledger_record_id is not None
    assert ledger.verify_integrity(trusted_public_key=signing_key.public_key()) is True
    seq, head = ledger.audit_head()
    assert seq == 1


def test_gate_a_deterministic_allow_fails_closed_if_ledger_write_fails(tmp_path):
    """Deterministic ALLOW fails closed to DENY if durable ledger recording fails."""
    db_path = str(tmp_path / "failing_audit.db")
    ledger = ActionLedger(path=db_path)

    # Simulate catastrophic disk/sqlite failure on append
    def failing_record(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error on commit")

    ledger.record_decision_receipt = failing_record

    allow_rule = PolicyRule(
        rule_id="trusted_read_allow",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r".*",
        reason="Allow all",
    )
    policy_engine = PolicyEngine(rules=[allow_rule])

    hook = ReflexGuardHook(
        policy_engine=policy_engine,
        ledger=ledger,
        fail_closed_ledger=True,
    )

    prop = ActionProposal.create(
        tenant_id="corp_tenant",
        principal_id="agent_reader",
        scope="fs:read",
        tool="read_file",
        arguments={"path": "/safe/data.json"},
        canonical_target="/safe/data.json",
        purpose="Read data",
    )

    result = hook.evaluate_proposal(prop)
    # Must fail closed to DENY
    assert result.outcome == DecisionOutcome.DENY
    assert result.allowed is False
    assert "ActionLedger write failed" in result.reason
    assert result.policy_decision.rule_id == "reflex_ledger_failure"


# ===========================================================================
# Gate A.5 & A.6: Immutable Canonical Action Dispatch & Harmless Sentinel
# ===========================================================================


def test_gate_a_mcp_proxy_canonical_immutable_dispatch():
    """ReflexMCPProxy dispatches exact frozen canonical arguments, ignoring caller mutations."""
    allow_rule = PolicyRule(rule_id="allow_safe", effect=DecisionOutcome.ALLOW, target_pattern=r".*", reason="Allow test")
    hook = ReflexGuardHook(policy_engine=PolicyEngine(rules=[allow_rule]))
    sentinel = SentinelTarget()
    proxy = ReflexMCPProxy(guard=hook)

    request_dict = {
        "jsonrpc": "2.0",
        "id": "mcp_immutable_1",
        "method": "tools/call",
        "params": {
            "name": "safe_tool",
            "arguments": {"target": "benign_input", "mode": "read"},
        },
    }

    # Intercept and execute via handle_call
    resp = proxy.handle_call(request_dict, sentinel.mcp_executor)
    assert "result" in resp
    assert sentinel.call_count == 1
    assert sentinel.last_kwargs == {"target": "benign_input", "mode": "read"}

    # Mutate the input request dictionary
    request_dict["params"]["arguments"]["target"] = "MALICIOUS_INJECTION"
    # Sentinel last kwargs should NOT be mutated
    assert sentinel.last_kwargs["target"] == "benign_input"


def test_gate_a_wrap_mcp_tool_immutable_dispatch_and_sentinel():
    """@wrap_mcp_tool dispatches canonicalized arguments and guarantees 0 executions on denial."""
    # Create rule allowing safe paths, denying write paths
    rules = [
        PolicyRule(rule_id="deny_writes", effect=DecisionOutcome.DENY, target_pattern=r".*write.*", reason="Veto writes"),
        PolicyRule(rule_id="allow_reads", effect=DecisionOutcome.ALLOW, target_pattern=r".*", reason="Allow reads"),
    ]
    hook = ReflexGuardHook(policy_engine=PolicyEngine(rules=rules))
    proxy = ReflexMCPProxy(guard=hook)

    sentinel = SentinelTarget()
    wrapped_fn = wrap_mcp_tool(proxy=proxy, tool_name="execute_file")(sentinel.execute_file)

    # 1. Allowed call
    res = wrapped_fn("/safe/path.txt", mode="read", retries=2)
    assert res == "file:/safe/path.txt:read:2"
    assert sentinel.call_count == 1

    # 2. Denied call -> 0 executions
    with pytest.raises(ReflexMCPBlockedError):
        wrapped_fn("/safe/write_payload.txt", mode="write", retries=1)

    assert sentinel.call_count == 1  # call count did not increase


def test_gate_a_langchain_interceptor_immutable_dispatch_and_sentinel():
    """ReflexToolInterceptor dispatches canonicalized arguments and guarantees 0 executions on denial."""
    rules = [
        PolicyRule(rule_id="deny_deletions", effect=DecisionOutcome.DENY, target_pattern=r".*delete.*", reason="Veto deletions"),
        PolicyRule(rule_id="allow_reads", effect=DecisionOutcome.ALLOW, target_pattern=r".*", reason="Allow reads"),
    ]
    hook = ReflexGuardHook(policy_engine=PolicyEngine(rules=rules))

    sentinel = SentinelTarget()
    interceptor = ReflexToolInterceptor(tool=sentinel.execute_file, guard=hook, tool_name="file_tool")

    # 1. Allowed call
    res = interceptor("/safe/data.txt", mode="read", retries=5)
    assert res == "file:/safe/data.txt:read:5"
    assert sentinel.call_count == 1

    # 2. Denied call
    with pytest.raises(ReflexGuardBlockedException):
        interceptor("/data/delete_all.txt", mode="delete")

    assert sentinel.call_count == 1


# ===========================================================================
# Gate B.1 & B.3: Explicit EnforcementProfile & Fail-Closed Ledger Durability
# ===========================================================================


def test_gate_b_enforcement_profile_validation(tmp_path):
    """EnforcementProfile rejects missing trusted signer or in-memory (:memory:) ledgers."""
    profile = EnforcementProfile(
        profile_id=ENFORCEMENT_PROFILE_V1,
        require_signer=True,
        require_durable_ledger=True,
    )

    # 1. In-memory ledger rejected in enforcement mode
    mem_ledger = ActionLedger(path=":memory:")
    signing_key = Ed25519PrivateKey.generate()
    with pytest.raises(ValueError, match="rejects in-memory ledger|requires a durable disk-backed ActionLedger"):
        profile.validate_configuration(signing_key=signing_key, ledger=mem_ledger)

    # 2. Missing signing key rejected in enforcement mode
    disk_ledger = ActionLedger(path=str(tmp_path / "valid_audit.db"))
    with pytest.raises(ValueError, match="requires a configured trusted Ed25519 signing key"):
        profile.validate_configuration(signing_key=None, ledger=disk_ledger)

    # 3. Valid configuration passes validation
    profile.validate_configuration(signing_key=signing_key, ledger=disk_ledger)


def test_gate_b_action_ledger_durability_flag():
    """ActionLedger.is_durable is False for :memory: and True for disk paths."""
    mem_ledger = ActionLedger(path=":memory:")
    assert mem_ledger.is_durable is False

    # Setting require_durable=True raises on :memory:
    with pytest.raises((ValueError, LedgerError)):
        ActionLedger(path=":memory:", require_durable=True)


def test_gate_b_action_ledger_verify_integrity_with_trusted_public_key(tmp_path):
    """ActionLedger.verify_integrity(trusted_public_key) cryptographically verifies entry signatures."""
    signing_key = Ed25519PrivateKey.generate()
    pub_key = signing_key.public_key()
    wrong_key = Ed25519PrivateKey.generate().public_key()

    db_path = str(tmp_path / "integrity_audit.db")
    ledger = ActionLedger(path=db_path)

    # Create signed receipt and record
    receipt = create_decision_receipt(
        schema_name="TestSchema",
        schema_digest="a" * 64,
        prompt="Inspect financial transaction",
        values={"is_safe": True},
        confidences={"is_safe": 0.99},
        conformal_sets={"is_safe": ["True"]},
        probabilities={"is_safe": {"True": 0.99, "False": 0.01}},
        latency_ms=0.5,
        is_ambiguous=False,
        signing_key=signing_key,
    )
    rec_id = ledger.record_decision_receipt(receipt)
    assert rec_id is not None

    # 1. Verified against correct public key -> passes
    assert ledger.verify_integrity(trusted_public_key=pub_key) is True

    # 2. Verified against wrong public key -> fails
    assert ledger.verify_integrity(trusted_public_key=wrong_key) is False

    # 3. Tampered entry payload in database -> fails
    with ledger._transaction() as conn:
        conn.execute(
            "UPDATE audit_entries SET payload_json=? WHERE sequence=1",
            (json.dumps({"tampered": True}),),
        )

    assert ledger.verify_integrity(trusted_public_key=pub_key) is False


# ===========================================================================
# Gate B.2: Comprehensive Authorization Receipt Binding
# ===========================================================================


def test_gate_b_authorization_receipt_binding_and_tamper_detection():
    """DecisionWitnessReceipt binds Gate B policy decision and claims, failing on any tampering."""
    signing_key = Ed25519PrivateKey.generate()
    pub_key = signing_key.public_key()

    prop = ActionProposal.create(
        tenant_id="tenant_alpha",
        principal_id="principal_beta",
        scope="scope_gamma",
        tool="transfer_assets",
        arguments={"amount": 500, "dest": "vault_9"},
        canonical_target="vault_9",
        purpose="Transfer assets to secure vault",
    )
    pol_decision = PolicyDecision._issue(
        action_id=prop.action_id,
        outcome=DecisionOutcome.ALLOW,
        reason="Verified authorization under policy rule 42",
        rule_id="rule_42",
        risk=RiskLevel.EXTERNAL,
        normalized_arguments=prop.arguments,
        policy_id="policy_fin_v1",
        policy_epoch="2026_q3",
    )

    receipt = create_decision_receipt(
        schema_name="GateB_Receipt_Test",
        schema_digest="f" * 64,
        prompt="Execute asset transfer",
        values={"is_safe": True},
        confidences={"is_safe": 0.999},
        conformal_sets={"is_safe": ["True"]},
        probabilities={"is_safe": {"True": 0.999, "False": 0.001}},
        latency_ms=0.45,
        is_ambiguous=False,
        signing_key=signing_key,
        policy_decision=pol_decision,
        action_proposal=prop,
        profile=ENFORCEMENT_PROFILE_V1,
        effective_alpha=0.05,
    )

    # 1. Unsigned payload and receipt dict include all Gate B claims
    r_dict = receipt.to_dict()
    assert r_dict["policy_decision"] is not None
    assert r_dict["policy_decision"]["rule_id"] == "rule_42"
    assert r_dict["policy_decision"]["policy_id"] == "policy_fin_v1"
    assert r_dict["policy_decision"]["policy_epoch"] == "2026_q3"
    assert r_dict["principal_id"] == "principal_beta"
    assert r_dict["tenant_id"] == "tenant_alpha"
    assert r_dict["scope"] == "scope_gamma"
    assert r_dict["effective_alpha"] == 0.05

    # 2. Genuine verification succeeds
    assert verify_decision_witness_receipt(r_dict, public_key=pub_key) is True

    # 3. Tampering policy_decision invalidates verification
    tampered_1 = copy.deepcopy(r_dict)
    tampered_1["policy_decision"]["rule_id"] = "fraudulent_rule"
    assert verify_decision_witness_receipt(tampered_1, public_key=pub_key) is False

    # 4. Tampering principal_id invalidates verification
    tampered_2 = copy.deepcopy(r_dict)
    tampered_2["principal_id"] = "impersonator"
    assert verify_decision_witness_receipt(tampered_2, public_key=pub_key) is False


# ===========================================================================
# Gate B.4: Two-Phase Execution Outcome Chaining & Structured INDETERMINATE
# ===========================================================================


def test_gate_b_mcp_handle_call_outcome_chaining_and_indeterminate(tmp_path):
    """ReflexMCPProxy chains execution outcome and returns structured INDETERMINATE error on audit failure."""
    db_path = str(tmp_path / "mcp_chain_audit.db")
    ledger = ActionLedger(path=db_path)
    allow_rule = PolicyRule(rule_id="allow_doc", effect=DecisionOutcome.ALLOW, target_pattern=r".*", reason="Allow docs")
    hook = ReflexGuardHook(policy_engine=PolicyEngine(rules=[allow_rule]), ledger=ledger, min_confidence=0.50, alpha=0.10)
    sentinel = SentinelTarget(return_value="mcp_success")
    proxy = ReflexMCPProxy(guard=hook)

    # 1. Normal execution chains SUCCEEDED outcome to ledger
    req = {
        "jsonrpc": "2.0",
        "id": "mcp-req-101",
        "method": "tools/call",
        "params": {"name": "read_doc", "arguments": {"path": "/safe/readme.md"}},
    }
    resp = proxy.handle_call(req, sentinel.mcp_executor)
    assert "result" in resp
    assert sentinel.call_count == 1

    # Verify ledger captured decision receipt (seq 1) + execution outcome (seq 2)
    seq, head = ledger.audit_head()
    assert seq == 2

    # 2. Simulate outcome recording failure -> structured INDETERMINATE response
    def failing_outcome(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error on outcome record")

    ledger.record_execution_outcome = failing_outcome

    req2 = {
        "jsonrpc": "2.0",
        "id": "mcp-req-102",
        "method": "tools/call",
        "params": {"name": "read_doc", "arguments": {"path": "/safe/license.txt"}},
    }
    resp2 = proxy.handle_call(req2, sentinel.mcp_executor)
    assert "error" in resp2
    assert resp2["error"]["code"] == -32001
    err_data = resp2["error"]["data"]
    assert err_data["status"] == "INDETERMINATE"
    assert "action_id" in err_data
    assert "receipt_digest" in err_data
    assert err_data["raw_result"] == {"result": "mcp_success"}


def test_gate_b_wrap_mcp_tool_outcome_chaining_and_indeterminate(tmp_path):
    """@wrap_mcp_tool chains execution outcome and raises ReflexIndeterminateExecutionError on audit failure."""
    db_path = str(tmp_path / "wrap_chain_audit.db")
    ledger = ActionLedger(path=db_path)
    allow_rule = PolicyRule(rule_id="allow_sample", effect=DecisionOutcome.ALLOW, target_pattern=r".*", reason="Allow sample")
    hook = ReflexGuardHook(policy_engine=PolicyEngine(rules=[allow_rule]), ledger=ledger, min_confidence=0.50, alpha=0.10)
    proxy = ReflexMCPProxy(guard=hook)

    sentinel = SentinelTarget()
    wrapped_fn = wrap_mcp_tool(proxy=proxy, tool_name="execute_file")(sentinel.execute_file)

    # 1. Successful execution chains outcome
    res = wrapped_fn("/safe/sample.txt")
    assert res == "file:/safe/sample.txt:read:3"
    seq, head = ledger.audit_head()
    assert seq == 2

    # 2. Failure on outcome recording raises ReflexIndeterminateExecutionError with evidence
    def failing_outcome(*args, **kwargs):
        raise sqlite3.OperationalError("ledger WAL locked")

    ledger.record_execution_outcome = failing_outcome

    with pytest.raises(ReflexIndeterminateExecutionError) as exc_info:
        wrapped_fn("/safe/sample2.txt")

    err = exc_info.value
    assert "INDETERMINATE" in str(err)
    assert err.action_id != ""
    assert err.receipt_digest != ""
    assert err.raw_result == "file:/safe/sample2.txt:read:3"
    assert isinstance(err.underlying_error, sqlite3.OperationalError)


def test_gate_b_langchain_interceptor_sync_and_async_outcome_chaining(tmp_path):
    """ReflexToolInterceptor chains sync and async outcomes and raises ReflexIndeterminateExecutionError."""
    db_path = str(tmp_path / "lc_chain_audit.db")
    ledger = ActionLedger(path=db_path)
    allow_rule = PolicyRule(rule_id="allow_lc", effect=DecisionOutcome.ALLOW, target_pattern=r".*", reason="Allow LC")
    hook = ReflexGuardHook(policy_engine=PolicyEngine(rules=[allow_rule]), ledger=ledger, min_confidence=0.50, alpha=0.10)

    sentinel = SentinelTarget()
    interceptor = ReflexToolInterceptor(tool=sentinel.execute_file, guard=hook, tool_name="file_tool")

    # 1. Synchronous invocation chains outcome
    res = interceptor("/safe/sync.txt")
    assert res == "file:/safe/sync.txt:read:3"
    seq, head = ledger.audit_head()
    assert seq == 2

    # 2. Async invocation chains outcome
    ainterceptor = ReflexToolInterceptor(tool=sentinel.aexecute_file, guard=hook, tool_name="afile_tool")
    ares = asyncio.run(ainterceptor.ainvoke("/safe/async.txt"))
    assert ares == "afile:/safe/async.txt:read:3"
    seq2, head2 = ledger.audit_head()
    assert seq2 == 4

    # 3. Simulate outcome recording failure on sync invoke
    def failing_outcome(*args, **kwargs):
        raise sqlite3.OperationalError("disk error")

    ledger.record_execution_outcome = failing_outcome

    with pytest.raises(ReflexIndeterminateExecutionError) as exc_info:
        interceptor("/safe/sync_fail.txt")

    err = exc_info.value
    assert err.action_id != ""
    assert err.receipt_digest != ""
    assert err.raw_result == "file:/safe/sync_fail.txt:read:3"


def test_gate_b_langchain_callback_handler_outcome_chaining(tmp_path):
    """ReflexGuardCallbackHandler chains on_tool_end and on_tool_error to ActionLedger."""
    db_path = str(tmp_path / "cb_chain_audit.db")
    ledger = ActionLedger(path=db_path)
    allow_rule = PolicyRule(rule_id="allow_cb", effect=DecisionOutcome.ALLOW, target_pattern=r".*", reason="Allow CB")
    hook = ReflexGuardHook(policy_engine=PolicyEngine(rules=[allow_rule]), ledger=ledger, min_confidence=0.50, alpha=0.10)
    handler = ReflexGuardCallbackHandler(guard=hook)

    serialized = {"name": "search_tool", "description": "Search index"}

    # 1. on_tool_start evaluates and records receipt
    handler.on_tool_start(serialized, "query term", run_id="run_1001")
    seq1, _ = ledger.audit_head()
    assert seq1 == 1

    # 2. on_tool_end records outcome
    handler.on_tool_end("found 10 items", run_id="run_1001")
    seq2, _ = ledger.audit_head()
    assert seq2 == 2

    # 3. Tool error records FAILED outcome
    handler.on_tool_start(serialized, "failing query", run_id="run_1002")
    handler.on_tool_error(ValueError("Invalid query syntax"), run_id="run_1002")
    seq3, _ = ledger.audit_head()
    assert seq3 == 4
