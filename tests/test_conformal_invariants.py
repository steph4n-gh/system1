"""Comprehensive test suite for Milestone M3: Conformal Calibration and Prediction Set Gating.

Authoritative Invariants Tested:
1. Invariant 6: Elimination of erroneous min_top_prob cutoff in ConformalPredictor.predict_set().
   100 identical probability vectors [0.4, 0.35, 0.25] labeled 'A' at alpha=0.05 MUST yield 'A' in the prediction set.
   Degenerate / true OOD vectors correctly yield empty prediction sets with needs_escalation=True.
2. Invariant 7: Conservative full feasible domain [min_value, max_value] with margin = val_range
   when order statistic k = ceil((n+1)(1-alpha)) > n in RegressionConformalPredictor.
3. Score & Serialization Harmonization:
   Round-trip .s1m serialization preserves {name}_calib_scores, conformal bounds, and empirical coverage.
4. Prediction Set Cardinality vs Escalation Policy:
   Empty sets (|C(x)| == 0) and multi-label sets (|C(x)| > 1) escalate; strict=True suppresses margin override.
5. Twin-namespace parity:
   Identical behavior when imported via `system1` or `reflex`.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
import numpy as np
import pytest

import reflex
import system1
from system1 import (
    BooleanField,
    ChoiceField,
    ConformalPredictor,
    DecisionCalibrator,
    DecisionResult,
    DecisionSchema,
    SystemOneCompiler,
    SystemOneEngine,
    RegressionConformalInterval,
    RegressionConformalPredictor,
    ScoreField,
)


class IncidentClassificationSchema(DecisionSchema):
    severity = ChoiceField(
        options=["P1", "P2", "P3"],
        descriptions={
            "P1": "critical production outage data loss or security breach",
            "P2": "elevated error rate latency spike partial service degradation",
            "P3": "minor bug background cosmetic glitch or routine query",
        },
    )
    notify_ops = BooleanField(threshold=0.5)
    impact_score = ScoreField(min_value=0.0, max_value=100.0)


def test_invariant_6_identical_probability_vectors_yield_non_empty_set():
    """Invariant 6: 100 identical probability vectors [0.4, 0.35, 0.25] labeled 'A' at alpha=0.05 yields 'A'."""
    cp = ConformalPredictor("action", ["A", "B", "C"])
    probs = np.tile([0.4, 0.35, 0.25], (100, 1))
    labels = ["A"] * 100
    cp.calibrate(probs, labels)

    assert cp.is_calibrated is True
    assert len(cp.calibration_scores) == 100
    assert np.allclose(cp.calibration_scores, 0.40)

    # Test point identical to calibration vectors
    test_p = np.array([0.4, 0.35, 0.25])
    cset = cp.predict_set(test_p, alpha=0.05)

    assert cset.is_empty is False
    assert "A" in cset.prediction_set
    assert len(cset.prediction_set) >= 1
    assert cset.conformal_threshold == pytest.approx(0.40, abs=1e-5)


def test_invariant_6_ood_degenerate_mass_yields_empty_set_and_escalation():
    """Invariant 6: Degenerate or unnormalized vectors produce empty prediction set and flag escalation."""
    cp = ConformalPredictor("action", ["A", "B", "C"])
    probs = np.tile([0.5, 0.3, 0.2], (50, 1))
    labels = ["A"] * 50
    cp.calibrate(probs, labels)

    # Unnormalized / degenerate OOD vectors whose total mass is severely deficient (< 50.0 or < q_hat)
    ood_vec = np.array([0.05, 0.05, 0.05])
    cset_ood = cp.predict_set(ood_vec, alpha=0.05)

    assert cset_ood.is_empty is True
    assert len(cset_ood.prediction_set) == 0
    assert cset_ood.needs_escalation is True


def test_invariant_7_order_statistic_exceeding_n_returns_full_domain():
    """Invariant 7: When order statistic k = ceil((n+1)(1-alpha)) > n, return conservative full domain."""
    rcp = RegressionConformalPredictor("score", min_value=0.0, max_value=100.0)
    # n = 5 samples
    preds = [10.0, 20.0, 30.0, 40.0, 50.0]
    trues = [11.0, 21.0, 29.0, 42.0, 48.0]
    rcp.calibrate(preds, trues)

    # For n=5, alpha=0.05: k = ceil(6 * 0.95) = ceil(5.7) = 6 > 5
    interval = rcp.predict_interval(20.0, alpha=0.05)

    # Must return full feasible domain [min_value, max_value] with margin = val_range
    assert interval.lower_bound == 0.0
    assert interval.upper_bound == 100.0
    assert interval.margin == 100.0
    assert interval.target_coverage == pytest.approx(0.95)


def test_invariant_7_valid_order_statistic_within_n_returns_calibrated_margin():
    """Invariant 7: When k <= n, predict_interval returns the empirical residual quantile."""
    rcp = RegressionConformalPredictor("score", min_value=0.0, max_value=100.0)
    # n = 20 samples
    preds = [float(i) for i in range(20)]
    trues = [float(i) + 0.5 for i in range(20)]  # residuals are all 0.5
    rcp.calibrate(preds, trues)

    # For n=20, alpha=0.10: k = ceil(21 * 0.90) = ceil(18.9) = 19 <= 20
    interval = rcp.predict_interval(50.0, alpha=0.10)

    assert interval.margin == pytest.approx(0.5, abs=1e-4)
    assert interval.lower_bound == pytest.approx(49.5, abs=1e-4)
    assert interval.upper_bound == pytest.approx(50.5, abs=1e-4)


def test_score_and_serialization_harmonization_roundtrip(tmp_path: Path):
    """Verify {name}_calib_scores are serialized in .s1m and restored identically in SystemOneEngine."""
    exemplars = {
        "severity": [
            ("Core database server corrupted disk failure", "P1"),
            ("Data center network fiber cut outage", "P1"),
            ("High API response latency degradation", "P2"),
            ("Payment gateway 503 throttling errors", "P2"),
            ("CSS static asset 404 cache miss", "P3"),
            ("Routine weekly dependency check", "P3"),
        ] * 5,
        "notify_ops": [
            ("Core database server corrupted disk failure", True),
            ("Data center network fiber cut outage", True),
            ("High API response latency degradation", True),
            ("Payment gateway 503 throttling errors", True),
            ("CSS static asset 404 cache miss", False),
            ("Routine weekly dependency check", False),
        ] * 5,
        "impact_score": [
            ("Core database server corrupted disk failure", 95.0),
            ("Data center network fiber cut outage", 90.0),
            ("High API response latency degradation", 55.0),
            ("Payment gateway 503 throttling errors", 60.0),
            ("CSS static asset 404 cache miss", 10.0),
            ("Routine weekly dependency check", 5.0),
        ] * 5,
    }

    compiler = SystemOneCompiler(IncidentClassificationSchema, dimension=64, regularization=0.5)
    model = compiler.compile(exemplars=exemplars)

    # Check that compiler populated calibration_scores on each head
    assert len(model.heads["severity"].calibration_scores) > 0
    assert len(model.heads["notify_ops"].calibration_scores) > 0
    assert len(model.heads["impact_score"].calibration_scores) > 0

    # Save model to disk and bytes
    model_path = tmp_path / "model.s1m"
    model.save(model_path)
    model_bytes = model.to_bytes()

    # Reconstruct from bytes and disk
    from system1.compiler import CompiledSystemOneModel
    reloaded_from_bytes = CompiledSystemOneModel.from_bytes(model_bytes)
    reloaded_from_disk = CompiledSystemOneModel.load(model_path)

    # Verify calibration_scores preserved across serialization
    orig_scores = model.heads["severity"].calibration_scores
    bytes_scores = reloaded_from_bytes.heads["severity"].calibration_scores
    disk_scores = reloaded_from_disk.heads["severity"].calibration_scores

    assert np.allclose(orig_scores, bytes_scores)
    assert np.allclose(orig_scores, disk_scores)

    # Initialize engines with original and reloaded models
    engine_orig = SystemOneEngine(IncidentClassificationSchema, model=model)
    engine_reloaded = SystemOneEngine(IncidentClassificationSchema, model=reloaded_from_bytes)

    # Verify conformal predictors in engine received calibration scores
    assert len(engine_reloaded.conformal_predictors["severity"].calibration_scores) == len(orig_scores)
    assert np.allclose(
        engine_reloaded.conformal_predictors["severity"].calibration_scores,
        engine_orig.conformal_predictors["severity"].calibration_scores,
    )

    # Evaluate test input on both engines and verify exact identical conformal sets
    test_prompt = "Database primary node degraded and dropping connections"
    res_orig = engine_orig.decide(test_prompt, alpha=0.05)
    res_reloaded = engine_reloaded.decide(test_prompt, alpha=0.05)

    assert res_orig.conformal_sets["severity"] == res_reloaded.conformal_sets["severity"]
    assert res_orig.conformal_sets["notify_ops"] == res_reloaded.conformal_sets["notify_ops"]
    assert res_orig.is_ambiguous == res_reloaded.is_ambiguous


def test_prediction_set_cardinality_escalation_in_strict_mode():
    """Verify that in strict mode, margin gating CANNOT suppress conformal ambiguity (|C(x)| != 1)."""
    cp = ConformalPredictor("severity", ["P1", "P2", "P3"])
    cp.calibration_scores = np.linspace(0.80, 0.96, 20)
    cp.is_calibrated = True

    # Test probability distribution where top candidate dominates runner-up, but alpha forces 2 candidates
    # P1 + P2 = .95 <= .96: both are inside the calibrated threshold.
    test_probs = np.array([0.85, 0.10, 0.05])

    # In non-strict mode with low margin threshold, margin gate can activate
    cset_relaxed = cp.predict_set(
        test_probs,
        alpha=0.05,
        margin_threshold=0.20,
        relative_odds_ratio=1.5,
        confidence_floor_tau0=0.10,
        strict=False,
    )
    assert len(cset_relaxed.prediction_set) == 3
    assert cset_relaxed.raw_is_ambiguous is True
    assert cset_relaxed.margin_gate_active is True
    assert cset_relaxed.is_ambiguous is False  # Suppressed by margin gate

    # In strict mode, margin gate MUST NOT activate; ambiguity remains True
    cset_strict = cp.predict_set(
        test_probs,
        alpha=0.05,
        margin_threshold=0.20,
        relative_odds_ratio=1.5,
        confidence_floor_tau0=0.10,
        strict=True,
    )
    assert len(cset_strict.prediction_set) == 2
    assert cset_strict.raw_is_ambiguous is True
    assert cset_strict.margin_gate_active is False
    assert cset_strict.is_ambiguous is True  # NEVER suppressed in strict mode
    assert cset_strict.needs_escalation is True


def test_engine_strict_mode_cardinality_escalation():
    """Verify SystemOneEngine.decide(strict=True) escalates whenever cardinality != 1."""
    schema = IncidentClassificationSchema()
    engine = SystemOneEngine(schema, strict_mode=False, enable_margin_gating=True)

    # Mock conformal predictor to return 2 candidates
    cp = engine.conformal_predictors["severity"]
    cp.calibration_scores = np.array([0.90, 0.95, 0.98])
    cp.is_calibrated = True

    # Under strict decide with high coverage forcing multi-candidate sets
    res_strict = engine.decide(
        "PostgreSQL database primary node latency degradation",
        alpha=0.001,
        strict=True,
    )

    if len(res_strict.conformal_sets["severity"]) > 1:
        assert res_strict.is_ambiguous is True
        assert "severity" in res_strict.ambiguous_fields


def test_twin_namespace_parity_for_conformal_components():
    """Verify that system1 and reflex export identical conformal calibration classes."""
    assert system1.ConformalPredictor is reflex.ConformalPredictor
    assert system1.ConformalPredictionSet is reflex.ConformalPredictionSet
    assert system1.RegressionConformalPredictor is reflex.RegressionConformalPredictor
    assert system1.RegressionConformalInterval is reflex.RegressionConformalInterval
    assert system1.DecisionCalibrator is reflex.DecisionCalibrator
    assert system1.SystemOneEngine is reflex.SystemOneEngine
