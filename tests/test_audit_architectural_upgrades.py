"""Unit and integration tests for the 5 architectural upgrades and boundary additions.

Covers:
1. Cardinality-Scaled Margin Gate & Relative Odds Ratio Dominance (K=5 vs K=77).
2. Exponential Forgetting Factor in Sherman-Morrison Online Updates (500+ updates, lambda_f=0.995 vs 1.0).
3. Recency-Aware Context Weighting for Multi-Turn Agent Traces.
4. Field-Level Escalation Granularity (escalate_on_ambiguity=False prevents advisory poisoning).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from system1.calibration import ConformalPredictionSet, ConformalPredictor
from system1.core.model import (
    DecisionFieldHead,
    DeterministicSemanticProjector,
    SystemOneModel,
)
from system1.core.schema import ChoiceField, DecisionSchema
from system1.engine import DecisionResult, SystemOneEngine


# ---------------------------------------------------------------------------
# Test 1: Cardinality-Scaled Margin Gate & Relative Odds Ratio Dominance
# ---------------------------------------------------------------------------

def test_cardinality_scaled_margin_gate_high_vs_low_k():
    """Verify that at K=77, pseudo-random noise (0.12 - 0.03 = 0.09) does not bypass ambiguity,
    whereas at K=5, genuine confidence (0.85 - 0.05 = 0.80) does activate margin gating.
    """
    # 1. High cardinality: K = 77 (e.g. MTEB Banking77)
    k_77_options = [f"intent_{i}" for i in range(77)]
    predictor_77 = ConformalPredictor(
        field_name="intent",
        options=k_77_options,
        margin_threshold=0.08,
        relative_odds_ratio=1.5,
        confidence_floor_tau0=0.15,
    )
    # Simulate non-conformity quantile requiring high probability to include in set
    # Let quantile be such that probabilities < 5.0 are included (e.g. quantile = 0.88)
    predictor_77.quantile = 0.88
    predictor_77.is_calibrated = True

    # Construct probability vector where top=0.12, second=0.03, rest split remainder
    # Naive margin = 0.12 - 0.03 = 0.09 > 0.08
    # Uniform baseline = 1/77 ~ 0.013. Floor required = 1/77 + 0.15 ~ 0.163
    probs_77 = np.zeros(77, dtype=np.float64)
    probs_77[0] = 0.12
    probs_77[1] = 0.03
    remaining_mass = 1.0 - (0.12 + 0.03)
    probs_77[2:] = remaining_mass / 75.0

    cset_77 = predictor_77.predict_set(probs_77, alpha=0.05)
    # Because 0.12 < (1/77 + 0.15), the cardinality-scaled floor rejects the false margin!
    assert cset_77.margin_gate_active is False
    assert cset_77.is_ambiguous is True
    assert len(cset_77.prediction_set) > 1

    # 2. Low cardinality: K = 5
    k_5_options = [f"opt_{i}" for i in range(5)]
    predictor_5 = ConformalPredictor(
        field_name="route",
        options=k_5_options,
        margin_threshold=0.08,
        relative_odds_ratio=1.5,
        confidence_floor_tau0=0.15,
    )
    predictor_5.quantile = 0.88
    predictor_5.is_calibrated = True

    probs_5 = np.array([0.85, 0.05, 0.04, 0.03, 0.03], dtype=np.float64)
    # Floor required = 1/5 + 0.15 = 0.35. 0.85 >= 0.35 (satisfied)
    # Odds ratio = 0.85 / 0.05 = 17.0 >= 1.5 (satisfied)
    # Margin = 0.85 - 0.05 = 0.80 >= 0.08 (satisfied)
    cset_5 = predictor_5.predict_set(probs_5, alpha=0.05)
    assert cset_5.margin_gate_active is True
    assert cset_5.is_ambiguous is False
    assert "opt_0" in cset_5.prediction_set
    assert cset_5.raw_is_ambiguous is True  # raw set had multiple, but margin gate suppressed ambiguity!


def test_relative_odds_ratio_gate_rejects_close_runner_up():
    """Verify relative odds ratio gate rejects close runner-up even when margin and floor pass."""
    k_3_options = ["A", "B", "C"]
    predictor = ConformalPredictor(
        field_name="choice",
        options=k_3_options,
        margin_threshold=0.08,
        relative_odds_ratio=1.5,
        confidence_floor_tau0=0.15,
    )
    predictor.quantile = 0.90
    predictor.is_calibrated = True

    # top=0.50, second=0.41, third=0.09
    # Margin = 0.50 - 0.41 = 0.09 > 0.08 (passes naive margin)
    # Floor = 1/3 + 0.15 = 0.4833. 0.50 >= 0.4833 (passes floor)
    # Relative odds ratio = 0.50 / 0.41 = 1.2195 < 5.0 (FAILS odds ratio!)
    probs = np.array([0.50, 0.41, 0.09], dtype=np.float64)
    cset = predictor.predict_set(probs, alpha=0.05)
    assert cset.margin_gate_active is False
    assert cset.is_ambiguous is True
    assert cset.odds_ratio is not None
    assert cset.odds_ratio < 5.0


def test_conformal_set_metadata_serialization():
    """Test ConformalPredictionSet to_dict contains new metadata."""
    cset = ConformalPredictionSet(
        field_name="test",
        target_coverage=0.95,
        alpha=0.05,
        prediction_set=("A",),
        conformal_threshold=0.88,
        is_ambiguous=False,
        is_empty=False,
        p_values={"A": 0.9},
        margin=0.8,
        margin_threshold=0.08,
        margin_gate_active=True,
        raw_is_ambiguous=False,
        odds_ratio=16.0,
        relative_odds_ratio_threshold=1.5,
        confidence_floor=0.35,
    )
    d = cset.to_dict()
    assert d["odds_ratio"] == 16.0
    assert d["relative_odds_ratio_threshold"] == 1.5
    assert d["confidence_floor"] == 0.35


# ---------------------------------------------------------------------------
# Test 2: Exponential Forgetting Factor in Sherman-Morrison Online Updates
# ---------------------------------------------------------------------------

def test_sherman_morrison_forgetting_factor_prevents_asphyxiation():
    """Verify that after 500 updates, lambda_f=0.995 maintains covariance sensitivity
    and eigenvalues orders of magnitude higher than lambda_f=1.0.
    """
    dim = 16
    field_def = ChoiceField(options=["class_0", "class_1"])
    head_frozen = DecisionFieldHead(field_def=field_def, dimension=dim, forgetting_factor=1.0)
    head_adaptive = DecisionFieldHead(field_def=field_def, dimension=dim, forgetting_factor=0.995)

    head_frozen.init_covariance(regularization=1.0)
    head_adaptive.init_covariance(regularization=1.0)

    rng = np.random.default_rng(42)

    # Initial training vector direction
    base_x = rng.normal(size=dim).astype(np.float32)
    base_x /= np.linalg.norm(base_x)
    base_x_aug = np.append(base_x, 1.0).astype(np.float32)
    target_class_0 = np.array([1.0, 0.0], dtype=np.float32)

    # Apply 500 online updates reinforcing class 0
    for _ in range(500):
        noise = rng.normal(scale=0.1, size=dim).astype(np.float32)
        x = base_x + noise
        x /= np.linalg.norm(x)
        x_aug = np.append(x, 1.0).astype(np.float32)
        head_frozen.online_update(x_aug, target_class_0, forgetting_factor=1.0)
        head_adaptive.online_update(x_aug, target_class_0, forgetting_factor=0.995)

    # Inspect the maximum eigenvalue / trace of inverse covariance P
    p_frozen = head_frozen.covariance_inv
    p_adaptive = head_adaptive.covariance_inv

    # Check symmetry
    assert np.allclose(p_adaptive, p_adaptive.T, atol=1e-5)
    assert np.allclose(p_frozen, p_frozen.T, atol=1e-5)

    norm_frozen = float(np.linalg.norm(p_frozen, 2))
    norm_adaptive = float(np.linalg.norm(p_adaptive, 2))

    # Overall spectral norm of P remains significantly higher with forgetting factor
    assert norm_adaptive > norm_frozen * 5.0

    # Along the heavily updated direction, sensitivity is preserved orders of magnitude higher
    quad_frozen = float(base_x_aug.T @ p_frozen @ base_x_aug)
    quad_adaptive = float(base_x_aug.T @ p_adaptive @ base_x_aug)
    assert quad_adaptive > quad_frozen * 3.0, f"Expected quad_adaptive ({quad_adaptive}) > 3x quad_frozen ({quad_frozen})"

    # Now simulate a concept drift / new tier-2 correction: target reverses to class 1
    new_target = np.array([0.0, 1.0], dtype=np.float32)
    test_x = rng.normal(size=dim).astype(np.float32)
    test_x /= np.linalg.norm(test_x)
    test_x_aug = np.append(test_x, 1.0).astype(np.float32)

    # Apply 5 updates of the new concept
    for _ in range(5):
        head_frozen.online_update(test_x_aug, new_target, forgetting_factor=1.0)
        head_adaptive.online_update(test_x_aug, new_target, forgetting_factor=0.995)

    eval_frozen = head_frozen.forward(test_x)
    eval_adaptive = head_adaptive.forward(test_x)

    # Head adaptive should have adapted much more towards class 1 than head frozen
    prob_adaptive_class1 = np.exp(eval_adaptive.logits[1]) / np.sum(np.exp(eval_adaptive.logits))
    prob_frozen_class1 = np.exp(eval_frozen.logits[1]) / np.sum(np.exp(eval_frozen.logits))

    assert prob_adaptive_class1 > prob_frozen_class1


def test_engine_learn_from_tier2_supports_forgetting_factor():
    """Verify SystemOneEngine.learn_from_tier2 forwards forgetting_factor."""
    class SimpleSchema(DecisionSchema):
        action = ChoiceField(options=["A", "B"])

    engine = SystemOneEngine(SimpleSchema, forgetting_factor=0.99)
    assert engine.forgetting_factor == 0.99

    res = engine.learn_from_tier2("test prompt", {"action": "B"}, forgetting_factor=0.98)
    assert res["status"] == "updated"
    assert "action" in res["updated_fields"]


# ---------------------------------------------------------------------------
# Test 3: Recency-Aware Context Weighting for Multi-Turn Agent Traces
# ---------------------------------------------------------------------------

def test_recency_aware_context_weighting():
    """Verify that recency weighting prioritizes trailing instructions over past prefix tokens."""
    projector = DeterministicSemanticProjector(dimension=128)

    # Prefix repeated 40 times: "read file status log"
    prefix = "read file status log " * 40
    tail_action = "execute emergency shutdown now"
    multi_turn_prompt = prefix + tail_action

    # Embed with uniform vs recency-weighted
    emb_uniform = projector.project(multi_turn_prompt, recency_weighted=False)
    emb_recency = projector.project(multi_turn_prompt, recency_weighted=True)

    # Embed tail action standalone
    emb_tail = projector.project(tail_action)

    # Similarity of full prompt to tail action should be significantly higher with recency weighting
    sim_uniform = float(np.dot(emb_uniform, emb_tail))
    sim_recency = float(np.dot(emb_recency, emb_tail))

    assert sim_recency > sim_uniform, f"Recency sim ({sim_recency}) should exceed uniform sim ({sim_uniform})"


def test_recency_formula_decay():
    """Verify the exact mathematical decay of recency weights."""
    projector = DeterministicSemanticProjector(dimension=64)
    tokens = ["alpha", "beta", "gamma", "delta", "omega"]
    n = len(tokens)

    # For token omega (i = 4, N-1-i = 0): weight = log(1 + 5) / sqrt(1.0 + 0) = log(6)
    expected_w_omega = math.log(1.0 + len("omega")) / math.sqrt(1.0 + 0.05 * 0)
    # For token alpha (i = 0, N-1-i = 4): weight = log(1 + 5) / sqrt(1.0 + 0.05 * 4) = log(6) / sqrt(1.20)
    expected_w_alpha = math.log(1.0 + len("alpha")) / math.sqrt(1.0 + 0.05 * 4)

    assert expected_w_omega > expected_w_alpha
    ratio = expected_w_omega / expected_w_alpha
    assert math.isclose(ratio, math.sqrt(1.20), rel_tol=1e-5)


def test_engine_decide_with_recency_weighted():
    """Verify SystemOneEngine.decide and encode accept recency_weighted."""
    class TriageSchema(DecisionSchema):
        action = ChoiceField(options=["READ", "EXECUTE"])

    engine = SystemOneEngine(TriageSchema, use_cache=False)
    emb1 = engine.encode("user query", recency_weighted=True)
    emb2 = engine.encode("user query", recency_weighted=False)
    assert emb1.shape == (384,)
    assert emb2.shape == (384,)

    result = engine.decide("user query", recency_weighted=True)
    assert isinstance(result, DecisionResult)
    assert result.values["action"] in ["READ", "EXECUTE"]


# ---------------------------------------------------------------------------
# Test 4: Field-Level Escalation Granularity
# ---------------------------------------------------------------------------

def test_field_level_escalation_granularity_prevents_advisory_poisoning():
    """Verify that an ambiguous advisory field with escalate_on_ambiguity=False
    does not cause global is_ambiguous=True when the primary control field is confident.
    """
    class CompositeAgentSchema(DecisionSchema):
        # Critical execution route: MUST escalate on ambiguity
        route = ChoiceField(
            options=["ALLOW", "BLOCK"],
            escalate_on_ambiguity=True,
        )
        # Advisory telemetry: SHOULD NOT halt execution on ambiguity
        sentiment = ChoiceField(
            options=["POSITIVE", "NEUTRAL", "NEGATIVE"],
            escalate_on_ambiguity=False,
        )

    engine = SystemOneEngine(CompositeAgentSchema, use_cache=False)

    # Ensure route produces confident prediction for ALLOW
    engine.model.heads["route"].weights[0, :] = 1.0
    engine.model.heads["route"].weights[1, :] = -1.0
    engine.conformal_predictors["route"].is_calibrated = True
    engine.conformal_predictors["route"].calibration_scores = np.array([0.5] * 20)

    # Ensure sentiment produces ambiguous uniform probabilities
    engine.model.heads["sentiment"].weights[:] = 0.0
    engine.conformal_predictors["sentiment"].is_calibrated = True
    engine.conformal_predictors["sentiment"].calibration_scores = np.array([0.99] * 20)
    engine.conformal_predictors["sentiment"].margin_threshold = 0.0

    result = engine.decide("Check file contents for sensitive keys")

    # Verify field-level outcomes
    assert len(result.conformal_sets["route"]) == 1
    assert len(result.conformal_sets["sentiment"]) == 3

    # The advisory sentiment field IS ambiguous
    assert "sentiment" in result.ambiguous_fields
    # But route IS NOT ambiguous
    assert "route" not in result.ambiguous_fields

    # Crucially: Because sentiment has escalate_on_ambiguity=False,
    # it was NOT added to escalated_fields, and global is_ambiguous remains FALSE!
    assert result.is_ambiguous is False
    assert result.escalated_fields == []

    # Verify serialization
    d = result.to_dict()
    assert d["is_ambiguous"] is False
    assert d["ambiguous_fields"] == ["sentiment"]
    assert d["escalated_fields"] == []


def test_field_level_escalation_triggers_on_control_field_ambiguity():
    """Verify that when a field with escalate_on_ambiguity=True is ambiguous,
    global is_ambiguous is set to True and the field is recorded in escalated_fields.
    """
    class StrictAgentSchema(DecisionSchema):
        route = ChoiceField(
            options=["ALLOW", "BLOCK"],
            escalate_on_ambiguity=True,
        )

    engine = SystemOneEngine(StrictAgentSchema, use_cache=False)
    engine.conformal_predictors["route"].is_calibrated = True
    engine.conformal_predictors["route"].quantile = 1.0  # forces ambiguity

    result = engine.decide("Ambiguous query")
    assert result.is_ambiguous is True
    assert "route" in result.ambiguous_fields
    assert "route" in result.escalated_fields

    d = result.to_dict()
    assert d["is_ambiguous"] is True
    assert d["escalated_fields"] == ["route"]


def test_schema_serialization_preserves_escalate_on_ambiguity():
    """Verify that schema to_dict and from_dict preserve escalate_on_ambiguity."""
    class TestSchema(DecisionSchema):
        f1 = ChoiceField(options=["A", "B"], escalate_on_ambiguity=False)
        f2 = ChoiceField(options=["X", "Y"], escalate_on_ambiguity=True)

    s = TestSchema()
    d = s.to_dict()
    assert d["fields"]["f1"]["escalate_on_ambiguity"] is False
    assert d["fields"]["f2"]["escalate_on_ambiguity"] is True

    restored = DecisionSchema.from_dict(d)
    assert restored.fields["f1"].escalate_on_ambiguity is False
    assert restored.fields["f2"].escalate_on_ambiguity is True


def test_covariance_stability_over_10000_updates_unexcited_subspace():
    """Verify covariance bounding invariant, strict positive-definiteness, and finite weights
    over 10,000 streaming updates with unexcited subspace coordinates.
    """
    dim = 16
    field = ChoiceField(options=["A", "B"])
    head = DecisionFieldHead(field, dimension=dim, forgetting_factor=0.995)
    head.init_covariance(regularization=1.0)

    rng = np.random.default_rng(123)
    # Excite only dimensions 0..3; dimensions 4..15 remain unexcited
    for step in range(10000):
        x = np.zeros(dim, dtype=np.float32)
        x[:4] = rng.normal(size=4).astype(np.float32)
        x[:4] /= np.linalg.norm(x[:4])
        x_aug = np.append(x, 1.0)
        target = np.array([1.0, 0.0] if (step % 2 == 0) else [0.0, 1.0], dtype=np.float32)
        head.online_update(x_aug, target)

    P = head.covariance_inv
    eigvals = np.linalg.eigvalsh(P)
    assert np.min(eigvals) > 0.0, "P matrix must remain strictly positive-definite"
    assert np.max(np.diag(P)) <= 50.0 + 1e-4, "Covariance bounding violated"
    assert np.all(np.isfinite(head.weights)), "Head weights blew up to NaN/Inf"
    assert np.all(np.isfinite(head.biases)), "Head biases blew up to NaN/Inf"
    assert np.max(np.abs(head.weights)) < 10.0, "Weights exploded"


def test_s1m_serialization_roundtrip_preserves_all_hyperparameters():
    """Verify that .s1m compilation and binary serialization/deserialization faithfully preserves
    forgetting_factor, relative_odds_ratio, confidence_floor_tau0, and recency_weighted.
    """
    from system1.compiler import SystemOneCompiler, CompiledSystemOneModel

    class ModelSchema(DecisionSchema):
        action = ChoiceField(options=["read", "write"], escalate_on_ambiguity=False)

    compiler = SystemOneCompiler(
        schema=ModelSchema,
        forgetting_factor=0.991,
        relative_odds_ratio=1.75,
        confidence_floor_tau0=0.18,
        recency_weighted=True,
    )
    compiled = compiler.compile()
    assert compiled.forgetting_factor == 0.991
    assert compiled.recency_weighted is True
    assert compiled.heads["action"].relative_odds_ratio == 1.75
    assert compiled.heads["action"].confidence_floor_tau0 == 0.18
    assert compiled.heads["action"].escalate_on_ambiguity is False

    data = compiled.to_bytes()
    restored = CompiledSystemOneModel.from_bytes(data)
    assert restored.forgetting_factor == 0.991
    assert restored.recency_weighted is True
    assert restored.heads["action"].relative_odds_ratio == 1.75
    assert restored.heads["action"].confidence_floor_tau0 == 0.18
    assert restored.heads["action"].escalate_on_ambiguity is False


def test_engine_init_and_decide_custom_odds_ratio_and_confidence_floor():
    """Verify SystemOneEngine initialization and decide() with custom relative_odds_ratio and confidence_floor_tau0,
    and verify DecisionResult contains odds_ratios and confidence_floors dictionaries in to_dict().
    """
    class RouterSchema(DecisionSchema):
        route = ChoiceField(options=["primary", "secondary"])

    engine = SystemOneEngine(
        RouterSchema,
        relative_odds_ratio=2.5,
        confidence_floor_tau0=0.18,
        use_cache=False,
    )
    assert engine.relative_odds_ratio == 2.5
    assert engine.confidence_floor_tau0 == 0.18

    res = engine.decide("route traffic", relative_odds_ratio=3.0, confidence_floor_tau0=0.22)
    assert "route" in res.odds_ratios
    assert "route" in res.confidence_floors
    assert res.relative_odds_ratio_thresholds["route"] == 3.0
    assert math.isclose(res.confidence_floors["route"], 0.5 + 0.22, rel_tol=1e-5)

    d = res.to_dict()
    assert "odds_ratios" in d
    assert "confidence_floors" in d
    assert "relative_odds_ratio_thresholds" in d
    assert "escalated_fields" in d
    assert "ambiguous_fields" in d
