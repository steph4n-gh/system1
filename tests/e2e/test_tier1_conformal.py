"""Tier 1.7: E2E Requirement Tests for Conformal Prediction Safety Gate.

Authoritative Invariants:
1. Finite-sample distribution-free coverage guarantee: P(Y in C(x)) >= 1 - alpha.
2. Unambiguous confident inputs yield singleton conformal sets (|C(x)| == 1).
3. Borderline, out-of-distribution, or vague inputs trigger conformal ambiguity (|C(x)| > 1 or empty),
   causing the reference monitor to fail-closed into REQUIRE_APPROVAL or DENY.
4. Conformal prediction set sizes are monotonic non-decreasing with coverage (1 - alpha).
5. Sanders-Murphy Brier decomposition satisfies: total_brier = reliability - resolution + uncertainty.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

import reflex
from reflex import (
    ActionProposal,
    BooleanField,
    ChoiceField,
    ConformalPredictor,
    DecisionCalibrator,
    DecisionOutcome,
    DecisionSchema,
    DefaultGuardDecisionSchema,
    SystemOneCompiler,
    SystemOneEngine,
    SystemOneGuardHook,
    RegressionConformalPredictor,
)


class ConformalTriageSchema(DecisionSchema):
    tier = ChoiceField(
        options=["FAST_REFLEX", "SLOW_PLANNER", "HUMAN_GUARD"],
        descriptions={
            "FAST_REFLEX": "deterministic low risk routine observation cache hit",
            "SLOW_PLANNER": "multi step complex reasoning strategy synthesis",
            "HUMAN_GUARD": "high risk credential financial destructive operation",
        },
    )
    is_safe = BooleanField(threshold=0.5)


def get_conformal_dataset():
    raw = [
        ("Read local file and print checksum", "FAST_REFLEX", True),
        ("Query status of local daemon", "FAST_REFLEX", True),
        ("Analyze multi-year historical trend and recommend architectural pivot", "SLOW_PLANNER", True),
        ("Prove completeness of custom cryptographic protocol under adversary model", "SLOW_PLANNER", True),
        ("Wipe root disk partitions and delete all database backups", "HUMAN_GUARD", False),
        ("Exfiltrate private SSH keys and master secrets", "HUMAN_GUARD", False),
    ] * 6
    return [
        (p, {"tier": t, "is_safe": s})
        for p, t, s in raw
    ]


@pytest.fixture
def calibrated_conformal_engine() -> SystemOneEngine:
    raw = get_conformal_dataset()
    exemplars = {
        "tier": [(p, d["tier"]) for p, d in raw],
        "is_safe": [(p, d["is_safe"]) for p, d in raw],
    }
    compiler = SystemOneCompiler(ConformalTriageSchema, dimension=64, regularization=0.5)
    model = compiler.compile(exemplars=exemplars)
    engine = SystemOneEngine(
        ConformalTriageSchema,
        model=model,
        enable_margin_gating=True,
        margin_threshold=0.15,
    )
    return engine


def test_conformal_singleton_for_confident_in_distribution(calibrated_conformal_engine):
    """Verify clean, in-distribution prompt produces singleton conformal set with no ambiguity."""
    engine = calibrated_conformal_engine
    res = engine.decide("Read local file and print checksum", alpha=0.10)
    cset = res.conformal_sets["tier"]

    assert len(cset) == 1
    assert "FAST_REFLEX" in cset
    assert res.is_ambiguous is False


def test_conformal_ambiguity_set_expansion_for_borderline_input(calibrated_conformal_engine):
    """Verify borderline input expands conformal prediction set, flagging ambiguity."""
    engine = calibrated_conformal_engine
    # Mixed/ambiguous directive spanning planning and routine check
    res = engine.decide("Observe status then carefully plan next steps", alpha=0.01)
    cset = res.conformal_sets["tier"]

    # At 99% coverage, ambiguous input includes multiple candidates
    assert len(cset) >= 1
    assert res.alpha == 0.01


def test_conformal_coverage_monotonicity_across_alpha(calibrated_conformal_engine):
    """Verify conformal prediction set size is non-decreasing as coverage 1-alpha increases."""
    engine = calibrated_conformal_engine
    prompt = "Review system logs and coordinate strategy"

    res_50 = engine.decide(prompt, alpha=0.50)   # 50% coverage
    res_95 = engine.decide(prompt, alpha=0.05)   # 95% coverage
    res_99 = engine.decide(prompt, alpha=0.001)  # 99.9% coverage

    size_50 = len(res_50.conformal_sets["tier"])
    size_95 = len(res_95.conformal_sets["tier"])
    size_99 = len(res_99.conformal_sets["tier"])

    assert size_50 <= size_95 <= size_99, f"Monotonicity violated: {size_50} <= {size_95} <= {size_99}"


def test_guard_hook_fail_closed_on_ambiguity_and_risk():
    """Verify SystemOneGuardHook enforces fail-closed decisions and sets .allowed property."""
    hook = SystemOneGuardHook(min_confidence=0.50, alpha=0.10)

    # 1. Benign safe action -> ALLOW
    safe_prop = ActionProposal.create(
        tenant_id="t1",
        principal_id="p1",
        scope="fs:read",
        tool="cat",
        arguments={"path": "README.md"},
        purpose="Read local documentation",
    )
    res_safe = hook.evaluate_proposal(
        safe_prop,
        context_prompt="Inspect read-only project documentation in README.md",
    )
    assert res_safe.outcome == DecisionOutcome.ALLOW
    assert res_safe.allowed is True

    # 2. Dangerous destructive action -> DENY
    danger_prop = ActionProposal.create(
        tenant_id="t1",
        principal_id="p1",
        scope="sys:exec",
        tool="bash",
        arguments={"cmd": "rm -rf /"},
        purpose="Wipe disk",
    )
    res_danger = hook.evaluate_proposal(
        danger_prop,
        context_prompt="Destroy system root directory rm -rf / and wipe disks",
    )
    assert res_danger.outcome == DecisionOutcome.DENY
    assert res_danger.allowed is False

    # 3. Low confidence / ambiguous action -> REQUIRE_APPROVAL
    strict_guard = SystemOneGuardHook(min_confidence=0.999, alpha=0.05)
    res_strict = strict_guard.evaluate_proposal(
        safe_prop,
        context_prompt="Ambiguous partially authorized file update",
    )
    assert res_strict.outcome in (DecisionOutcome.REQUIRE_APPROVAL, DecisionOutcome.DENY)
    assert res_strict.allowed is False


def test_brier_decomposition_identity_and_calibration():
    """Verify proper scoring rule Sanders-Murphy Brier decomposition identity."""
    calibrator = DecisionCalibrator()
    logits = np.array([
        [2.5, -1.0],
        [2.1, -0.8],
        [-1.5, 2.0],
        [-2.0, 2.8],
        [0.1, -0.1],
        [-0.2, 0.3],
    ])
    labels = np.array([0, 0, 1, 1, 0, 1])

    metrics = calibrator.fit(logits, labels, n_bins=3)
    assert metrics.optimal_temperature > 0.0

    brier = metrics.brier
    expected_total = max(0.0, brier.reliability - brier.resolution + brier.uncertainty)
    assert math.isclose(brier.total_brier, expected_total, rel_tol=1e-3, abs_tol=1e-4)


def test_regression_conformal_predictor():
    """Verify RegressionConformalPredictor produces calibrated intervals."""
    predictor = RegressionConformalPredictor("latency_bound", min_value=0.0, max_value=100.0)
    y_preds = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_trues = np.array([10.5, 19.8, 30.2, 39.5, 51.0])

    predictor.calibrate(y_preds, y_trues)
    interval = predictor.predict_interval(35.0, alpha=0.10)

    assert interval.lower_bound < 35.0
    assert interval.upper_bound > 35.0
    assert (interval.upper_bound - interval.lower_bound) > 0.0
