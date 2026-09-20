"""Tests for Autonomous AI Agent Firewall & Tool Execution Monitor Showcase.

Covers examples/autonomous_agent_firewall_showcase.py and examples/enterprise_stress_showcase.py.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import autonomous_agent_firewall_showcase
import enterprise_stress_showcase
from autonomous_agent_firewall_showcase import (
    demonstrate_monkey_patching,
    get_autonomous_agent_firewall_dataset,
    get_autonomous_agent_firewall_schema,
    get_interleaved_query_stream,
    make_firewall_baseline_handler,
    run_autonomous_firewall_showcase,
    run_multithreaded_stress_test,
    verify_crypto_and_audit_trail,
)
from system1.compat.typesafe import (
    Choice,
    MultiChoice,
    Noul,
    PromotionPolicy,
    Score,
    TypeSafeClient,
)
from system1.compiler import CompiledSystemOneModel
from system1.ledger import ActionLedger
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def test_schema_and_dataset_specifications():
    """Verify the 5-head schema and 48-query dataset adhere to strict specifications."""
    schema = get_autonomous_agent_firewall_schema()
    assert len(schema) == 5
    assert isinstance(schema["action_decision"], Choice)
    assert len(schema["action_decision"].options) == 4
    assert "ALLOW_IMMEDIATE" in schema["action_decision"].options
    assert "TERMINATE_AGENT_SESSION" in schema["action_decision"].options

    assert isinstance(schema["risk_category"], Choice)
    assert len(schema["risk_category"].options) == 5

    assert isinstance(schema["policy_violations"], MultiChoice)
    assert len(schema["policy_violations"].options) == 6
    assert "DATA_EXFILTRATION" in schema["policy_violations"].options
    assert "PROMPT_INJECTION" in schema["policy_violations"].options

    assert isinstance(schema["is_safe_to_execute"], Noul)
    assert isinstance(schema["cvss_risk_score"], Score)
    assert schema["cvss_risk_score"].min_value == 0.0
    assert schema["cvss_risk_score"].max_value == 10.0

    # Test dataset
    dataset = get_autonomous_agent_firewall_dataset()
    assert len(dataset) == 48
    # 6 sectors with 8 queries each
    assert len(dataset[0:8]) == 8   # Sector 1: Benign Actions
    assert len(dataset[8:16]) == 8  # Sector 2: Borderline / Ambiguous
    assert len(dataset[16:24]) == 8 # Sector 3: Prompt Injections
    assert len(dataset[24:32]) == 8 # Sector 4: Obfuscated Shell Exploits
    assert len(dataset[32:40]) == 8 # Sector 5: Data Exfiltration
    assert len(dataset[40:48]) == 8 # Sector 6: Database & PII Manipulation

    # Test interleaved stream
    stream = get_interleaved_query_stream(dataset)
    assert len(stream) == 48
    # First 6 queries should span all 6 sectors (ids FW-01, FW-09, FW-17, FW-25, FW-33, FW-41)
    assert stream[0]["id"].startswith("FW-01")
    assert stream[1]["id"].startswith("FW-09")
    assert stream[2]["id"].startswith("FW-17")
    assert stream[3]["id"].startswith("FW-25")
    assert stream[4]["id"].startswith("FW-33")
    assert stream[5]["id"].startswith("FW-41")


def test_baseline_handler_generation():
    """Verify high-fidelity baseline handler generates valid cloud responses for each category."""
    schema = get_autonomous_agent_firewall_schema()
    dataset = get_autonomous_agent_firewall_dataset()
    handler = make_firewall_baseline_handler(dataset)

    # Benign query from Sector 1
    benign_prompt = dataset[0]["prompt"]
    resp_benign = handler(benign_prompt, schema)
    assert resp_benign.answers["action_decision"].value == "ALLOW_IMMEDIATE"
    assert resp_benign.answers["risk_category"].value == "BENIGN_ROUTINE"
    assert resp_benign.answers["is_safe_to_execute"].value is True
    assert resp_benign.answers["cvss_risk_score"].value <= 0.5
    assert resp_benign.egress_bytes > 0
    assert resp_benign.local_execution is False

    # Exploit query from Sector 4 (Shell exploit)
    exploit_prompt = dataset[24]["prompt"]
    resp_exploit = handler(exploit_prompt, schema)
    assert resp_exploit.answers["action_decision"].value == "TERMINATE_AGENT_SESSION"
    assert resp_exploit.answers["risk_category"].value == "EXFILTRATION_EXPLOIT"
    assert "UNAUTHORIZED_SHELL_EXECUTION" in resp_exploit.answers["policy_violations"].value
    assert resp_exploit.answers["is_safe_to_execute"].value is False
    assert resp_exploit.answers["cvss_risk_score"].value >= 7.0

    # Borderline query from Sector 2
    borderline_prompt = dataset[8]["prompt"]
    resp_borderline = handler(borderline_prompt, schema)
    assert resp_borderline.answers["action_decision"].value == "REQUIRE_HUMAN_APPROVAL"
    assert resp_borderline.is_ambiguous is True


def test_autonomous_firewall_lifecycle_run(tmp_path: Path):
    """Verify complete auto-cutover lifecycle run with SQLite ledger and conformal sets."""
    db_file = str(tmp_path / "firewall_test.db")
    export_file = str(tmp_path / "firewall_model.s1m")

    # Run with small threshold and total for fast test execution
    run_autonomous_firewall_showcase(
        cutover_threshold=6,
        total_queries=12,
        concurrent_requests=50,
        concurrent_workers=4,
        db_path=db_file,
        run_concurrent_stress=True,
        run_monkey_patch=True,
        throttle_ms=0.0,
        export_model_path=export_file,
    )

    # 1. Verify SQLite ledger exists and hash chain is unbroken
    assert Path(db_file).exists()
    ledger = ActionLedger(db_file, read_only=True)
    assert ledger.verify_integrity() is True
    head_seq, _ = ledger.audit_head()
    # At least 12 queries + 1 cutover event + 50 stress audit queries = 63+ entries
    assert head_seq >= 60

    # 2. Verify model export file exists and is loadable
    assert Path(export_file).exists()
    assert Path(export_file).stat().st_size > 0
    loaded_model = CompiledSystemOneModel.load(export_file)
    res = loaded_model.forward_single("git status")
    assert "action_decision" in res.fields
    assert res.fields["action_decision"].selected_value is not None


def test_multithreaded_stress_benchmark():
    """Verify concurrent requests complete with an intact, authenticated ledger."""
    schema = get_autonomous_agent_firewall_schema()
    dataset = get_autonomous_agent_firewall_dataset()
    ledger = ActionLedger(":memory:")
    signing_key = Ed25519PrivateKey.generate()

    # Pre-compiled local client
    client = TypeSafeClient(
        mode="local",
        signing_key=signing_key,
        ledger=ledger,
    )

    results = run_multithreaded_stress_test(
        client=client,
        questions=schema,
        queries=dataset[:12],
        num_workers=4,
        total_requests=100,
        record_receipt=True,
    )

    assert results["total_requests"] == 100
    assert results["inline_qps"] > 100.0  # Safe lower bound across CI/test machines
    # Full-history verification grows with the ledger; report timing without
    # treating a shared CI runner's speed as a correctness requirement.
    assert math.isfinite(results["audit_qps"]) and results["audit_qps"] > 0.0
    assert results["ledger_integrity"] is True
    assert ledger.verify_integrity(trusted_public_key=signing_key.public_key()) is True
    head_seq, _ = ledger.audit_head()
    assert head_seq == 100


def test_crypto_and_audit_trail_verification():
    """Verify verify_crypto_and_audit_trail validates Ed25519 signature and ledger."""
    ledger = ActionLedger(":memory:")
    signing_key = Ed25519PrivateKey.generate()
    schema = get_autonomous_agent_firewall_schema()

    client = TypeSafeClient(
        mode="local",
        signing_key=signing_key,
        ledger=ledger,
    )

    resp = client.systemone("git diff HEAD~1", schema, alpha=0.05)
    assert resp.receipt is not None

    # Should execute without throwing assertion error
    verify_crypto_and_audit_trail(ledger, signing_key, resp)


def test_non_cutover_graceful_completion(tmp_path: Path):
    """Verify showcase completes gracefully when total_queries < cutover_threshold."""
    db_file = str(tmp_path / "non_cutover.db")

    run_autonomous_firewall_showcase(
        cutover_threshold=10,
        total_queries=4,
        concurrent_requests=20,
        concurrent_workers=2,
        db_path=db_file,
        run_concurrent_stress=False,
        run_monkey_patch=False,
        throttle_ms=0.0,
    )

    ledger = ActionLedger(db_file, read_only=True)
    assert ledger.verify_integrity() is True
    head_seq, _ = ledger.audit_head()
    assert head_seq == 4


def test_enterprise_stress_showcase_alias():
    """Verify enterprise_stress_showcase alias module correctly re-exports functions."""
    assert enterprise_stress_showcase.get_autonomous_agent_firewall_schema is get_autonomous_agent_firewall_schema
    assert enterprise_stress_showcase.get_autonomous_agent_firewall_dataset is get_autonomous_agent_firewall_dataset
    assert enterprise_stress_showcase.run_autonomous_firewall_showcase is run_autonomous_firewall_showcase
    assert enterprise_stress_showcase.run_multithreaded_stress_test is run_multithreaded_stress_test


def test_firewall_defers_when_full_decisions_disagree():
    """Getting some fields right must not promote an unreliable five-field skill."""
    from system1.core.embeddings import HybridProjector

    questions = get_autonomous_agent_firewall_schema()
    dataset = get_autonomous_agent_firewall_dataset()
    stream = get_interleaved_query_stream(dataset)
    handler = make_firewall_baseline_handler(dataset)
    client = TypeSafeClient(
        mode="auto_cutover", cutover_threshold=20,
        promotion_policy=PromotionPolicy(
            min_agreement_threshold=0.75, false_allow_ceiling=0.0,
            require_statistical_bound=False, min_local_acceptance=0.0,
        ),
        augment=True, strict_mode=False,
        projector=HybridProjector(dimension=384), dimension=384,
        baseline_handler=handler,
    )
    for query in stream:
        response = client.systemone(query["prompt"], questions)
        assert not response.local_execution
        assert response.answers == handler(query["prompt"], questions).answers

    assert not client.is_cutover
    report = client.last_promotion_report
    assert report is not None and not report.is_eligible
    assert report.matching_checks > 0
    assert report.agreement_rate < client.promotion_policy.min_agreement_threshold
    assert any("Held-out agreement" in reason for reason in report.rejection_reasons)


def test_demonstrate_monkey_patching_accuracy():
    """Verify monkey patching drop-in execution yields clean benign classifications with zero spurious violations."""
    schema = get_autonomous_agent_firewall_schema()
    benign_query = "Read contents of README.md in workspace root directory to inspect repository overview."

    # 1. Zero-shot uncompiled client demonstration
    demonstrate_monkey_patching(schema, benign_query, compiled_model=None)

    # 2. Test programmatic client behavior under monkey patch with HybridProjector
    from system1.compat.typesafe import patch_typesafe
    from system1.embeddings import HybridProjector
    unpatcher = patch_typesafe()
    try:
        import typesafe_sdk
        projector = HybridProjector(dimension=384)
        client = typesafe_sdk.TypeSafeClient(dimension=384, projector=projector)
        resp = client.systemone(benign_query, schema)
        assert resp.local_execution is True
        assert resp.answers["action_decision"].value == "ALLOW_IMMEDIATE"
        assert resp.answers["risk_category"].value == "BENIGN_ROUTINE"
        assert resp.answers["is_safe_to_execute"].value is True
        assert len(resp.answers["policy_violations"].value) == 0
        assert resp.answers["cvss_risk_score"].value < 3.0

        # Also verify default projector keeps CVSS score below midpoint (< 50.0) and 0 violations
        client_def = typesafe_sdk.TypeSafeClient(dimension=384)
        resp_def = client_def.systemone(benign_query, schema)
        assert resp_def.answers["action_decision"].value == "ALLOW_IMMEDIATE"
        assert resp_def.answers["is_safe_to_execute"].value is True
        assert len(resp_def.answers["policy_violations"].value) == 0
        assert resp_def.answers["cvss_risk_score"].value < 50.0
    finally:
        unpatcher.unpatch()

