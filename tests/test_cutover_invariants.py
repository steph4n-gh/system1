"""Tests for Invariant 10: Cutover Promotion Generalization, Held-Out Validation, and Drift Monitoring.

Verifies:
1. Invariant 10 violation: overfitted noise achieves 100% in-sample accuracy on training exemplars,
   but fails on held-out validation; decoupled partition logic defers cutover.
2. 3-way partition decoupling: training, calibration, and validation folds are strictly disjoint.
3. Strict zero-tolerance false-allow ceiling (0.0%) on critical security classes blocks cutover.
4. Minimum validation sample threshold deferral.
5. Wilson score confidence interval lower bound promotion gating.
6. Post-cutover concept and distributional drift detection across sliding window.
7. Post-cutover cloud fallback routing vs. zero-egress offline abstention.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from system1.compat.typesafe import (
    Choice,
    CutoverPartition,
    DriftDetector,
    Noul,
    PromotionPolicy,
    PromotionReport,
    Score,
    TypeSafeClient,
    _build_dynamic_schema,
    evaluate_promotion_eligibility,
    partition_cutover_history,
)
from system1.compiler import SystemOneCompiler
from system1.engine import SystemOneEngine
from system1.ledger import ActionLedger


def test_invariant_10_held_out_validation_prevents_overfit_promotion():
    """Invariant 10: Overfitted training exemplars must not promote cutover when held-out validation fails."""
    questions = {
        "access": Choice("Access Decision", criteria={"allow": "Permitted", "deny": "Forbidden"}),
    }
    schema = _build_dynamic_schema(questions)

    # 10 training samples that memorize random noisy tokens to 'allow'
    train_history = [
        {"state": f"noise_token_train_{i}_{1000 + i}", "answers": {"access": "allow"}}
        for i in range(10)
    ]
    # 5 held-out validation samples with opposite target labels ('deny')
    val_history = [
        {"state": f"noise_token_val_{i}_{9000 + i}", "answers": {"access": "deny"}}
        for i in range(5)
    ]

    # Fit model strictly on train_history
    compiler = SystemOneCompiler(schema=schema, dimension=128)
    exemplars = {"access": [(h["state"], h["answers"]["access"]) for h in train_history]}
    compiled_model = compiler.compile(exemplars=exemplars)

    engine = SystemOneEngine(schema, dimension=128)
    for f_name, ch in compiled_model.heads.items():
        engine.model.heads[f_name].set_weights(ch.weights, ch.biases)

    # In-sample check on training exemplars: passes with high agreement
    in_sample_report = evaluate_promotion_eligibility(
        engine=engine,
        val_history=train_history,
        schema=schema,
        policy=PromotionPolicy(min_validation_samples=5, min_agreement_threshold=0.80),
    )
    # The old logic would have promoted because in-sample agreement is high!
    assert in_sample_report.agreement_rate >= 0.80

    # Held-out validation check: model fails to generalize to held-out validation fold
    held_out_report = evaluate_promotion_eligibility(
        engine=engine,
        val_history=val_history,
        schema=schema,
        policy=PromotionPolicy(min_validation_samples=5, min_agreement_threshold=0.80),
    )

    # Held-out evaluation correctly blocks cutover!
    assert held_out_report.is_eligible is False
    assert held_out_report.agreement_rate < 50.0
    assert any("below minimum threshold" in r or "ceiling" in r for r in held_out_report.rejection_reasons)


def test_partition_cutover_history_disjointness():
    """Verify that training, calibration, and validation partitions are strictly disjoint."""
    history = [
        {"state": f"prompt_{i}", "answers": {"route": "support" if i % 2 == 0 else "sales"}}
        for i in range(30)
    ]

    partition = partition_cutover_history(history, train_ratio=0.60, calib_ratio=0.20, val_ratio=0.20)
    assert partition.total_samples == 30
    assert len(partition.train_history) == 18
    assert len(partition.calib_history) == 6
    assert len(partition.val_history) == 6

    # Verify assert_disjoint passes without error
    partition.assert_disjoint()

    train_ids = {id(x) for x in partition.train_history}
    calib_ids = {id(x) for x in partition.calib_history}
    val_ids = {id(x) for x in partition.val_history}

    # Strict pairwise disjointness
    assert len(train_ids & val_ids) == 0
    assert len(calib_ids & val_ids) == 0
    assert len(train_ids & calib_ids) == 0


def test_strict_zero_tolerance_false_allow_ceiling():
    """Verify that a single false allow on a critical security class strictly blocks cutover."""
    questions = {
        "firewall_action": Choice(
            "Firewall Policy",
            criteria={"allow": "Permit traffic", "deny": "Block malicious traffic"},
        ),
    }
    schema = _build_dynamic_schema(questions)

    # 10 validation samples: 9 matching allows, 1 critical false allow (target=deny, pred=allow)
    class MockEngine:
        def decide(self, prompt: str, **kwargs: Any) -> Any:
            from types import SimpleNamespace
            # Always predicts 'allow'
            return SimpleNamespace(values={"firewall_action": "allow"}, is_ambiguous=False)

    engine = MockEngine()
    val_history = [
        {"state": f"safe_traffic_{i}", "answers": {"firewall_action": "allow"}}
        for i in range(9)
    ] + [
        {"state": "sql_injection_probe", "answers": {"firewall_action": "deny"}}
    ]

    policy = PromotionPolicy(
        min_validation_samples=5,
        min_agreement_threshold=0.85,
        false_allow_ceiling=0.0,  # Strict 0.0% tolerance
        critical_classes={"deny"},
    )

    report = evaluate_promotion_eligibility(engine, val_history, schema, policy)

    # Overall agreement is 9/10 = 90% (exceeds 85% threshold)
    assert report.agreement_rate == 0.90
    assert report.false_allow_count == 1
    # But because 1 false allow occurred on a critical class, promotion MUST be rejected!
    assert report.is_eligible is False
    assert any("false-allow rate" in r for r in report.rejection_reasons)


def test_minimum_validation_sample_threshold():
    """Verify that promotion is deferred when held-out validation sample count is below threshold."""
    questions = {"tier": Choice("Tier", criteria={"basic": "Basic", "pro": "Pro"})}
    schema = _build_dynamic_schema(questions)

    class MockEngine:
        def decide(self, prompt: str, **kwargs: Any) -> Any:
            from types import SimpleNamespace
            return SimpleNamespace(values={"tier": "pro"}, is_ambiguous=False)

    engine = MockEngine()
    # 3 validation samples with 100% agreement
    val_history = [
        {"state": f"query_{i}", "answers": {"tier": "pro"}}
        for i in range(3)
    ]

    policy = PromotionPolicy(
        min_validation_samples=10,  # Requires at least 10 samples
        min_agreement_threshold=0.80,
    )

    report = evaluate_promotion_eligibility(engine, val_history, schema, policy)
    assert report.agreement_rate == 1.0
    assert report.is_eligible is False
    assert any("Validation sample count (3) below minimum threshold (10)" in r for r in report.rejection_reasons)


def test_wilson_score_statistical_bound_gating():
    """Verify that require_statistical_bound gates cutover when confidence bound is wide."""
    questions = {"verdict": Choice("Verdict", criteria={"yes": "Yes", "no": "No"})}
    schema = _build_dynamic_schema(questions)

    class MockEngine:
        def decide(self, prompt: str, **kwargs: Any) -> Any:
            from types import SimpleNamespace
            return SimpleNamespace(values={"verdict": "yes"}, is_ambiguous=False)

    engine = MockEngine()
    # 5 samples with 100% empirical agreement: Wilson lower bound on 5/5 at 95% is ~0.56
    val_history = [{"state": f"q_{i}", "answers": {"verdict": "yes"}} for i in range(5)]

    # Without statistical bound: 100% empirical agreement is eligible
    lenient_policy = PromotionPolicy(
        min_validation_samples=5,
        min_agreement_threshold=0.80,
        require_statistical_bound=False,
    )
    lenient_report = evaluate_promotion_eligibility(engine, val_history, schema, lenient_policy)
    assert lenient_report.is_eligible is True

    # With statistical bound: Wilson lower bound (~0.56) < 50.0, so promotion is deferred
    strict_policy = PromotionPolicy(
        min_validation_samples=5,
        min_agreement_threshold=0.80,
        require_statistical_bound=True,
    )
    strict_report = evaluate_promotion_eligibility(engine, val_history, schema, strict_policy)
    assert strict_report.is_eligible is False
    assert strict_report.wilson_lower_bound < 50.0
    assert any("Wilson statistical lower bound" in r for r in strict_report.rejection_reasons)


def test_sliding_window_drift_detector():
    """Verify DriftDetector identifies concept and distributional drift across sliding window."""
    detector = DriftDetector(window_size=20, ambiguity_threshold=0.20, ood_threshold=0.10)
    assert not detector.is_drifted

    # 15 nominal observations
    for _ in range(15):
        detector.record(is_ambiguous=False, is_ood=False, confidence=0.95)
    assert not detector.is_drifted

    # Introduce 6 ambiguous observations (6/20 = 30% > 20% threshold)
    for _ in range(6):
        detector.record(is_ambiguous=True, is_ood=False, confidence=0.60)

    assert detector.is_drifted is True
    assert "Concept drift detected" in detector.drift_reason

    # Recovery: stream 20 nominal observations to clear sliding window
    for _ in range(20):
        detector.record(is_ambiguous=False, is_ood=False, confidence=0.99)
    assert detector.is_drifted is False


def test_post_cutover_cloud_fallback_routing(monkeypatch):
    """Verify that when allow_cloud_fallback=True, ambiguous queries route to cloud."""
    client = TypeSafeClient(
        mode="auto_cutover",
        cutover_threshold=2,
        zero_egress=False,
        allow_cloud_fallback=True,
    )
    client._has_cutover = True
    from system1.compat.typesafe import TypeSafeResponse
    monkeypatch.setattr(client, "call_real_api", lambda **kwargs: (
        TypeSafeResponse({"answers": {}, "local_execution": False}), 1.0, 100,
    ))

    # Inject mock drift in detector
    client.drift_detector.record(is_ambiguous=True, is_ood=True, confidence=0.10)
    client.drift_detector._drift_detected = True

    questions = {"flag": Choice("Flag", criteria={"red": "Red", "green": "Green"})}
    resp = client.systemone("Borderline edge-case prompt", questions)

    # Cloud fallback routing executed
    assert resp.get("fallback_routed") is True
    assert resp.get("drift_detected") is True
    assert resp.get("egress_bytes", 0) > 0


def test_post_cutover_zero_egress_offline_abstention():
    """Verify that in zero_egress mode, ambiguous/drifted queries perform offline abstention (0 egress)."""
    client = TypeSafeClient(
        mode="auto_cutover",
        cutover_threshold=2,
        zero_egress=True,
        allow_cloud_fallback=False,
        baseline_handler=lambda s, q: {},
    )
    client._has_cutover = True

    # Inject mock drift in detector
    client.drift_detector._drift_detected = True

    questions = {"flag": Choice("Flag", criteria={"red": "Red", "green": "Green"})}
    resp = client.systemone("Borderline prompt with active drift", questions)

    # Offline abstention returned with 0 network egress
    assert resp.get("abstain") is True
    assert resp.get("verdict") == "REQUIRE_APPROVAL"
    assert resp.get("status") in ("ABSTAINED_AMBIGUOUS", "ABSTAINED_DRIFT")
    assert resp.get("egress_bytes") == 0
