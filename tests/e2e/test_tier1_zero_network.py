"""Tier 1.3: E2E Requirement Tests for Zero External Network Egress.

Authoritative Invariants:
1. System 1 / System 1 executes 100% on local metal with ZERO outbound socket connections.
2. Decision evaluation, conformal gating, cryptographic signing, SQLite ledger,
   TypeSafe client, calibration, and compilation MUST complete without attempting
   any remote network I/O or DNS lookups.
3. Any outbound network call is an immediate security and architectural violation.
"""

from __future__ import annotations

import socket
from pathlib import Path
import pytest

import reflex
from reflex import (
    ActionLedger,
    ActionProposal,
    Choice,
    DefaultGuardDecisionSchema,
    SystemOneCompiler,
    SystemOneEngine,
    SystemOneGuardHook,
    TypeSafeClient,
)
from e2e.conftest import BlockedNetworkCallError


def test_socket_block_actually_traps_network(enforce_zero_network):
    """Adversarial check: Ensure the network guard fixture actually blocks socket connections."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(BlockedNetworkCallError, match="Zero-network violation"):
        s.connect(("1.1.1.1", 80))


def test_zero_network_engine_decision(enforce_zero_network, triage_schema):
    """Verify SystemOneEngine.decide executes completely locally under hard socket block."""
    engine = SystemOneEngine(triage_schema)
    res = engine.decide("Perform local fast telemetry observation")
    assert res is not None
    assert "route" in res.values
    assert res.latency_ms < 20.0


def test_zero_network_guard_hook_evaluation(enforce_zero_network, temp_ledger):
    """Verify SystemOneGuardHook proposal evaluation and ledger recording require zero network calls."""
    guard = SystemOneGuardHook(ledger=temp_ledger, auto_calibrate=False)
    proposal = ActionProposal.create(
        tenant_id="local_tenant",
        principal_id="local_agent",
        scope="fs:read",
        tool="cat",
        arguments={"file": "/var/log/audit.log"},
        purpose="Audit local events",
    )
    int_res = guard.evaluate_proposal(proposal)
    assert int_res.outcome is not None
    assert int_res.policy_decision is not None
    seq, head = temp_ledger.audit_head()
    assert seq >= 1
    assert len(head) == 64


def test_zero_network_action_ledger_operations(enforce_zero_network, temp_ledger):
    """Verify ActionLedger append, chaining, and verification operate with zero network calls."""
    res = reflex.decide("Sample decision for ledger testing", schema=DefaultGuardDecisionSchema)
    entry_hash = temp_ledger.record_decision_receipt(res.receipt)
    assert len(entry_hash) == 64
    assert temp_ledger.verify_integrity() is True


def test_zero_network_typesafe_client_offline(enforce_zero_network):
    """Verify TypeSafeClient systemone non-autoregressive mode runs offline."""
    client = TypeSafeClient(api_key="sk-offline-dummy")
    resp = client.systemone(
        "Safe read-only check of documentation",
        {"route": Choice("Routing choice", criteria=["local_fast", "cloud_slow"])},
    )
    assert resp is not None
    assert resp.answers.route.choice in ("local_fast", "cloud_slow")
    assert resp.local_execution is True


def test_zero_network_calibration_and_compilation(enforce_zero_network, triage_schema, tmp_path: Path):
    """Verify temperature calibration and binary compiler operate with zero network calls."""
    engine = SystemOneEngine(triage_schema)
    dataset = [
        ("Read local disk", {"route": "local_system1", "is_safe": True, "confidence_score": 0.9}),
        ("Destroy partitions", {"route": "human_escalation", "is_safe": False, "confidence_score": 0.95}),
    ] * 4
    metrics = engine.calibrate(dataset, n_bins=3)
    assert len(metrics) > 0

    model_path = tmp_path / "offline_model.s1m"
    compiler = SystemOneCompiler(triage_schema)
    compiler.compile_and_save(model_path, samples_per_choice=3)
    assert model_path.is_file()
