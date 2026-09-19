"""Tier 5 Adversarial Coverage Hardening & Edge-Case Stress Test Suite.

Probes white-box boundary conditions across:
1. Conformal prediction gating, margin checks, calibration stability (calibration.py)
2. Sherman-Morrison rank-1 distillation, vector fusion, projectors (core/model.py, core/telemetry.py)
3. Tier 0 Semantic System 1 Cache boundary churn and thread safety (cache.py)
4. SystemOneEngine runtime robustness and fail-safe guarantees (engine.py)
"""

import concurrent.futures
import math
import numpy as np
import pytest

from system1.calibration import (
    ConformalPredictor,
    DecisionCalibrator,
    RegressionConformalPredictor,
    _stable_softmax,
    compute_brier_decomposition,
    compute_ece_and_bins,
    compute_nll,
)
from system1.cache import SemanticSystemOneCache
from system1.core.model import (
    DecisionFieldHead,
    DeterministicSemanticProjector,
    SystemOneModel,
    _stable_sigmoid,
)
from system1.core.schema import (
    BooleanField,
    ChoiceField,
    DecisionSchema,
    MultiChoiceField,
    ScoreField,
)
from system1.core.telemetry import TelemetryProjector
from system1.engine import DecisionResult, SystemOneEngine


# ==============================================================================
# Test Schemas for Stress Testing
# ==============================================================================

class MultiFieldAdversarialSchema(DecisionSchema):
    """Schema spanning all field types for rigorous edge-case testing."""
    route = ChoiceField(
        options=["ALLOW", "REVIEW", "BLOCK"],
        descriptions={
            "ALLOW": "Grant immediate low-risk access",
            "REVIEW": "Flag for human supervisor review",
            "BLOCK": "Terminate transaction immediately",
        },
    )
    is_anomaly = BooleanField(
        description="Whether heuristic anomaly detector triggered",
        threshold=0.5,
    )
    risk_score = ScoreField(
        min_value=0.0,
        max_value=1.0,
        description="Continuous risk score",
    )
    flags = MultiChoiceField(
        options=["NEW_DEVICE", "FOREIGN_IP", "RAPID_FIRE", "ROOTED_ENV"],
        threshold=0.5,
    )


# ==============================================================================
# 1. Numerical Stability & Calibration Boundary Tests
# ==============================================================================

def test_stable_softmax_and_sigmoid_extreme_boundary_conditions():
    """Verify softmax and sigmoid under extreme logits, zero/negative/infinite temperature."""
    # 1. Zero and negative temperatures must clamp safely (>= 1e-4) without ZeroDivisionError
    logits = np.array([2.0, 1.0, 0.0], dtype=np.float64)
    p_zero_temp = _stable_softmax(logits, temperature=0.0)
    assert np.all(np.isfinite(p_zero_temp))
    assert np.isclose(np.sum(p_zero_temp), 1.0)
    assert p_zero_temp[0] == 1.0
    assert p_zero_temp[1] == 0.0
    assert p_zero_temp[2] == 0.0

    p_neg_temp = _stable_softmax(logits, temperature=-5.0)
    assert np.all(np.isfinite(p_neg_temp))
    assert np.isclose(np.sum(p_neg_temp), 1.0)

    # 2. Extremely high temperature -> uniform distribution
    p_high_temp = _stable_softmax(logits, temperature=1e8)
    assert np.all(np.isfinite(p_high_temp))
    assert np.allclose(p_high_temp, 1.0 / 3.0, atol=1e-5)

    # 3. Extreme logits (+1000, -1000) without numerical overflow/underflow NaN
    extreme_logits = np.array([1000.0, -1000.0, 0.0], dtype=np.float64)
    p_extreme = _stable_softmax(extreme_logits, temperature=1.0)
    assert np.all(np.isfinite(p_extreme))
    assert np.isclose(p_extreme[0], 1.0)
    assert np.isclose(p_extreme[1], 0.0)
    assert np.isclose(np.sum(p_extreme), 1.0)

    # 4. All identical logits
    equal_logits = np.array([42.0, 42.0, 42.0, 42.0], dtype=np.float64)
    p_equal = _stable_softmax(equal_logits, temperature=1.0)
    assert np.allclose(p_equal, 0.25)

    # 5. Sigmoid extreme boundary conditions
    sig_zero_temp = _stable_sigmoid(np.array([10.0, -10.0, 0.0]), temperature=0.0)
    assert np.all(np.isfinite(sig_zero_temp))
    assert sig_zero_temp[0] == 1.0
    assert sig_zero_temp[1] == 0.0
    assert sig_zero_temp[2] == 0.5

    sig_extreme = _stable_sigmoid(np.array([10000.0, -10000.0]), temperature=1.0)
    assert sig_extreme[0] == 1.0
    assert sig_extreme[1] == 0.0


def test_conformal_predictor_uncalibrated_and_alpha_bounds():
    """Verify split conformal prediction set logic on uncalibrated states and boundary alphas."""
    options = ["alpha", "beta", "gamma"]
    cp = ConformalPredictor("token", options)

    assert not cp.is_calibrated

    probs = np.array([0.70, 0.20, 0.10], dtype=np.float64)

    # Uncalibrated fallback: q_hat = 1 - alpha = 0.90
    cset = cp.predict_set(probs, alpha=0.10)
    assert cset.conformal_threshold == pytest.approx(0.90)
    assert "alpha" in cset.prediction_set
    assert "beta" in cset.prediction_set
    assert cset.p_values["alpha"] == pytest.approx(0.70)

    # Alpha extreme near 0 and near 1
    cset_tight = cp.predict_set(probs, alpha=1e-4)
    assert set(cset_tight.prediction_set) == set(options)  # Requires virtually 100% coverage

    # When alpha=0.50 (q_hat=0.50), top candidate (prob=0.70 >= 0.50) satisfies coverage
    cset_mid = cp.predict_set(probs, alpha=0.50)
    assert len(cset_mid.prediction_set) == 1
    assert cset_mid.prediction_set == ("alpha",)

    # When alpha=0.99 (q_hat=0.01), top candidate (prob=0.70 >= 0.01) satisfies coverage
    cset_loose = cp.predict_set(probs, alpha=0.99)
    assert cset_loose.is_empty is False
    assert cset_loose.prediction_set == ("alpha",)

    # Invalid alpha values must raise ValueError
    with pytest.raises(ValueError, match="Significance level alpha must be in"):
        cp.predict_set(probs, alpha=0.0)
    with pytest.raises(ValueError, match="Significance level alpha must be in"):
        cp.predict_set(probs, alpha=1.0)
    with pytest.raises(ValueError, match="Significance level alpha must be in"):
        cp.predict_set(probs, alpha=-0.5)
    with pytest.raises(ValueError, match="Significance level alpha must be in"):
        cp.predict_set(probs, alpha=1.5)

    # Shape mismatch validation
    with pytest.raises(ValueError, match="does not match options"):
        cp.predict_set(np.array([0.5, 0.5]), alpha=0.05)


def test_conformal_single_option_and_ood_empty_set():
    """Verify edge case of 1-option schema field and out-of-distribution empty set detection."""
    cp_single = ConformalPredictor("lone_field", ["SOLO"])
    p_single = np.array([1.0], dtype=np.float64)
    cset_single = cp_single.predict_set(p_single, alpha=0.05)
    assert cset_single.prediction_set == ("SOLO",)
    assert cset_single.margin == 1.0
    assert not cset_single.is_ambiguous

    # Out-of-distribution (OOD) test: all probabilities tiny, degenerate sum < 0.5
    cp_multi = ConformalPredictor("multi", ["A", "B", "C"])
    tiny_probs = np.array([1e-7, 1e-7, 1e-7], dtype=np.float64)
    cset_ood = cp_multi.predict_set(tiny_probs, alpha=0.05)
    assert cset_ood.is_empty is True
    assert len(cset_ood.prediction_set) == 0
    assert not cset_ood.is_ambiguous


def test_regression_conformal_predictor_boundaries():
    """Verify continuous regression conformal predictor edge cases and uncalibrated state."""
    rcp = RegressionConformalPredictor("score", min_value=0.0, max_value=100.0)
    assert not rcp.is_calibrated

    # Uncalibrated fallback interval: range = 100, margin = 100 * (1 - 0.10) / 2 = 45.0
    interval = rcp.predict_interval(50.0, alpha=0.10)
    assert interval.lower_bound == pytest.approx(5.0)
    assert interval.upper_bound == pytest.approx(95.0)
    assert interval.margin == pytest.approx(45.0)
    assert interval.to_interval_string() == "[5.0000, 95.0000]"

    # Clamping at domain boundaries
    interval_low = rcp.predict_interval(10.0, alpha=0.10)
    assert interval_low.lower_bound == 0.0  # Clamped to min_value
    assert interval_low.upper_bound == pytest.approx(55.0)

    # Invariant 7: Order statistic exceeding sample count (k > n) returns conservative full feasible domain
    rcp.calibrate([10.0, 20.0, 30.0, 40.0, 50.0], [11.0, 21.0, 29.0, 42.0, 48.0])
    interval_k_gt_n = rcp.predict_interval(20.0, alpha=0.05)
    assert interval_k_gt_n.lower_bound == 0.0
    assert interval_k_gt_n.upper_bound == 100.0
    assert interval_k_gt_n.margin == 100.0

    # Invalid alpha values
    with pytest.raises(ValueError):
        rcp.predict_interval(50.0, alpha=0.0)
    with pytest.raises(ValueError):
        rcp.predict_interval(50.0, alpha=1.0)

    # Calibration validation errors
    with pytest.raises(ValueError, match="0 samples"):
        rcp.calibrate([], [])
    with pytest.raises(ValueError, match="mismatch"):
        rcp.calibrate([1.0], [1.0, 2.0])


def test_margin_gating_adversarial_thresholds():
    """Stress-test Lever 3 margin-based conformal gating under adversarial thresholds."""
    cp = ConformalPredictor("decision", ["A", "B", "C"])
    cp.is_calibrated = True
    cp.calibration_scores = np.array([0.50, 0.60, 0.70])

    # Near tie: top=0.42, runner-up=0.40 => margin = 0.02
    tie_probs = np.array([0.42, 0.40, 0.18])

    # Negative threshold -> margin gate must NOT activate
    cset_neg = cp.predict_set(tie_probs, alpha=0.05, margin_threshold=-0.1)
    assert cset_neg.margin_gate_active is False
    assert cset_neg.is_ambiguous is True

    # Zero threshold -> margin gate must NOT activate
    cset_zero = cp.predict_set(tie_probs, alpha=0.05, margin_threshold=0.0)
    assert cset_zero.margin_gate_active is False
    assert cset_zero.is_ambiguous is True

    # Impossible threshold (> 1.0) -> can never be satisfied
    dominant_probs = np.array([0.90, 0.08, 0.02])  # Margin = 0.82
    cset_imp = cp.predict_set(dominant_probs, alpha=0.05, margin_threshold=1.5)
    assert cset_imp.margin_gate_active is False

    # Standard dominance: margin 0.82 >= 0.50 suppresses ambiguity
    cset_dom = cp.predict_set(dominant_probs, alpha=0.05, margin_threshold=0.50)
    assert cset_dom.margin == pytest.approx(0.82)
    assert cset_dom.margin_gate_active is True
    assert cset_dom.is_ambiguous is False
    assert cset_dom.raw_is_ambiguous is True


def test_proper_scoring_rules_and_single_sample_calibration():
    """Verify calibration diagnostics (ECE, NLL, Brier) under single-sample and zero-sample cases."""
    # Zero samples
    ece0, mce0, accs0, confs0, counts0 = compute_ece_and_bins(np.empty((0, 2)), np.empty(0))
    assert ece0 == 0.0 and mce0 == 0.0

    brier0 = compute_brier_decomposition(np.empty((0, 2)), np.empty(0))
    assert brier0.total_brier == 0.0

    assert compute_nll(np.empty((0, 2)), np.empty(0)) == 0.0

    # Single sample
    probs1 = np.array([[0.95, 0.05]])
    labels1 = np.array([0])
    ece1, mce1, accs1, confs1, counts1 = compute_ece_and_bins(probs1, labels1, n_bins=5)
    assert np.isfinite(ece1)
    assert counts1[-1] == 1  # Falls in highest confidence bin

    brier1 = compute_brier_decomposition(probs1, labels1, n_bins=5)
    assert np.isfinite(brier1.total_brier)
    assert np.isfinite(brier1.reliability)
    assert np.isfinite(brier1.resolution)
    assert np.isfinite(brier1.uncertainty)


# ==============================================================================
# 2. Sherman-Morrison Distillation & Core Model Boundary Tests
# ==============================================================================

def test_sherman_morrison_500_sequential_updates_drift_and_symmetry():
    """Stress-test Sherman-Morrison rank-1 update with 500 sequential steps.
    
    Verifies that repeated rank-1 subtraction maintains positive semi-definiteness,
    perfect symmetry, and exhibits zero NaN or Inf drift.
    """
    dim = 32
    f_def = ChoiceField(options=["PASS", "FAIL"])
    head = DecisionFieldHead(f_def, dimension=dim, contrastive_whitening=False)
    head.init_covariance(regularization=1.0)

    rng = np.random.default_rng(2026)

    for step in range(500):
        # Generate varied synthetic input embeddings
        x = rng.standard_normal(dim).astype(np.float32)
        x /= np.linalg.norm(x)
        x_aug = np.append(x, 1.0).astype(np.float32)

        target = np.array([1.0, 0.0] if (step % 2 == 0) else [0.0, 1.0], dtype=np.float32)
        dt_ms = head.online_update(x_aug, target)
        assert dt_ms >= 0.0

    # Inspect final covariance inverse P
    P = head.covariance_inv
    assert P is not None
    assert np.all(np.isfinite(P)), "P contains NaN or Inf after 500 updates"
    assert np.all(np.isfinite(head.weights)), "Weights contain NaN or Inf"
    assert np.all(np.isfinite(head.biases)), "Biases contain NaN or Inf"

    # Symmetry check: P == P.T
    assert np.allclose(P, P.T, atol=1e-5), "Covariance inverse lost symmetry"

    # Positive semi-definiteness check: all eigenvalues >= -1e-6
    eigvals = np.linalg.eigvalsh(P)
    assert np.all(eigvals >= -1e-6), f"Covariance inverse lost PSD property: min eigval = {np.min(eigvals)}"


def test_sherman_morrison_adversarial_inputs_and_singularity():
    """Verify input validation and numerical singularity protection in Sherman-Morrison update."""
    dim = 16
    f_def = BooleanField(description="test")
    head = DecisionFieldHead(f_def, dimension=dim)
    head.init_covariance(regularization=1.0)

    # 1. Non-finite inputs must raise ValueError
    nan_vec = np.zeros(dim, dtype=np.float32)
    nan_vec[0] = np.nan
    with pytest.raises(ValueError, match="x_aug contains NaN or Inf"):
        head.online_update(nan_vec, np.array([1.0]))

    inf_target = np.array([np.inf], dtype=np.float32)
    with pytest.raises(ValueError, match="y_target contains NaN or Inf"):
        head.online_update(np.zeros(dim), inf_target)

    # 2. Dimension mismatch
    with pytest.raises(ValueError, match="Expected x_aug of length"):
        head.online_update(np.zeros(dim - 2), np.array([1.0]))

    with pytest.raises(ValueError, match="Expected target vector y of length"):
        head.online_update(np.zeros(dim), np.array([1.0, 2.0]))

    # 3. Denominator singularity safeguard (denom <= 1e-12)
    # Artificially set P such that 1 + x^T P x <= 0
    head.covariance_inv = -1.0 * np.eye(dim + 1, dtype=np.float32)
    unit_x = np.zeros(dim + 1, dtype=np.float32)
    unit_x[0] = 1.0  # x^T P x = -1.0 => denom = 1.0 + (-1.0) = 0.0
    # Must preserve internal state cleanly and return without throwing
    elapsed = head.online_update(unit_x, np.array([1.0]))
    assert elapsed >= 0.0


def test_semantic_projector_adversarial_inputs():
    """Verify DeterministicSemanticProjector on adversarial text strings and boundary cases."""
    proj = DeterministicSemanticProjector(dimension=128)

    # 1. Empty string & whitespace only -> returns default unit vector [1, 0, 0, ...]
    v_empty = proj.project("")
    assert np.isclose(np.linalg.norm(v_empty), 1.0)
    assert v_empty[0] == 1.0 and np.all(v_empty[1:] == 0.0)

    v_ws = proj.project("   \t\n  \r ")
    assert np.allclose(v_empty, v_ws)

    # 2. Pure punctuation / non-alphanumeric
    v_punct = proj.project("!@#$%^&*()_+=-{}[]:;'<>,.?/")
    assert np.all(np.isfinite(v_punct))
    assert np.isclose(np.linalg.norm(v_punct), 1.0)

    # 3. Unicode and emojis
    v_emoji = proj.project("🚀 System 1 System 1 Engine 🧠 Apple Silicon ⚡️")
    assert np.all(np.isfinite(v_emoji))
    assert np.isclose(np.linalg.norm(v_emoji), 1.0)

    # 4. Massive string (> 10,000 characters) with truncation guarantee
    massive_text = "system security breach " * 1000
    v_massive = proj.project(massive_text)
    assert np.all(np.isfinite(v_massive))
    assert np.isclose(np.linalg.norm(v_massive), 1.0)

    # 5. Type validation
    with pytest.raises(TypeError, match="Expected text to be a string"):
        proj.project(None)  # type: ignore
    with pytest.raises(TypeError, match="Expected text to be a string"):
        proj.project(12345)  # type: ignore

    # 6. Batched projection edge cases
    batch_empty = proj.project_batch([])
    assert batch_empty.shape == (0, 128)

    with pytest.raises(TypeError, match="All elements in texts must be strings"):
        proj.project_batch(["valid", 42])  # type: ignore


def test_telemetry_projector_extreme_fusions():
    """Verify continuous telemetry normalization and vector fusion under edge conditions."""
    proj = TelemetryProjector(embedding_dim=128, fusion_weight=0.5)
    sem_vec = np.ones(128, dtype=np.float32) / math.sqrt(128)

    # 1. Empty telemetry mappings and sequences
    v_dict_empty = proj.project({})
    assert np.isclose(np.linalg.norm(v_dict_empty), 1.0)
    assert v_dict_empty[0] == 1.0

    v_seq_empty = proj.project([])
    assert np.allclose(v_dict_empty, v_seq_empty)

    # 2. Extreme continuous values (+/- 1e6) under robust tanh normalization
    extreme_telem = {"cpu_burst": 1e6, "mem_leak": -1e6, "ping_ms": 0.0}
    norm_telem = proj.normalize(extreme_telem)
    assert np.all(np.isfinite(norm_telem))
    assert np.all(np.abs(norm_telem) <= 1.0)

    # 3. High-dimensional telemetry within embedding bound (e.g. 64 features for 128 dim)
    mid_telem = np.random.randn(64).astype(np.float32)
    fused_mid = proj.fuse(sem_vec, mid_telem, mode="project")
    assert fused_mid.shape == (128,)
    assert np.isclose(np.linalg.norm(fused_mid), 1.0)

    # Telemetry exceeding embedding dimension (200 features > 128 dim) triggers matrix dimension mismatch
    over_dim_telem = np.random.randn(200).astype(np.float32)
    with pytest.raises(ValueError, match="mismatch in its core dimension"):
        proj.fuse(sem_vec, over_dim_telem, mode="project")

    # 4. Zero semantic embedding
    fused_zero_sem = proj.fuse(np.zeros(128, dtype=np.float32), [1.0, 2.0], mode="project")
    assert np.all(np.isfinite(fused_zero_sem))
    assert np.isclose(np.linalg.norm(fused_zero_sem), 1.0)

    # 5. Invalid mode and normalization rejection
    with pytest.raises(ValueError, match="Invalid normalization"):
        TelemetryProjector(normalization="invalid_norm")
    with pytest.raises(ValueError, match="Invalid mode"):
        TelemetryProjector(mode="invalid_mode")


# ==============================================================================
# 3. Tier 0 Semantic System 1 Cache Boundary & Thread-Safety Tests
# ==============================================================================

def test_cache_capacity_1_churn_and_zero_norm():
    """Verify cache LRU eviction with capacity=1 and zero-norm embedding handling."""
    cache = SemanticSystemOneCache(capacity=1, similarity_threshold=0.98)

    # Put entry B -> evicts entry A immediately
    cache.put("prompt_B", {"res": "B"}, embedding=np.array([0.0, 1.0]))
    assert cache.size == 1
    assert cache.get("prompt_A", embedding=np.array([1.0, 0.0])) is None
    hit_b, _ = cache.get("prompt_B", embedding=np.array([0.0, 1.0]))
    assert hit_b.result["res"] == "B"

    # Put entry C with zero-norm embedding (should not crash, handled gracefully)
    cache.put("prompt_C", {"res": "C"}, embedding=np.zeros(2))
    assert cache.size == 1
    assert cache.get("prompt_B", embedding=np.array([0.0, 1.0])) is None
    hit_c, sim_c = cache.get("prompt_C", embedding=np.zeros(2))
    assert hit_c.result["res"] == "C"

    # Zero-norm embedding query must not raise ZeroDivisionError
    zero_emb = np.zeros(2, dtype=np.float32)
    miss_zero = cache.get("unknown_query", embedding=zero_emb)
    assert miss_zero is None


def test_cache_telemetry_isolation():
    """Verify identical prompts with different telemetry never collide in cache."""
    cache = SemanticSystemOneCache(capacity=10)

    p = "transfer funds to external routing account"
    telem_low = {"amount": 50.0}
    telem_high = {"amount": 5000000.0}

    cache.put(p, "ALLOW_LOW", telemetry=telem_low)
    cache.put(p, "REQUIRE_MFA_HIGH", telemetry=telem_high)

    hit_low, _ = cache.get(p, telemetry=telem_low)
    assert hit_low.result == "ALLOW_LOW"

    hit_high, _ = cache.get(p, telemetry=telem_high)
    assert hit_high.result == "REQUIRE_MFA_HIGH"

    # Query without telemetry should miss
    assert cache.get(p) is None


def test_cache_multithreaded_concurrency_stress():
    """Verify thread-safety of SemanticSystemOneCache under concurrent parallel reads and writes."""
    cache = SemanticSystemOneCache(capacity=50, similarity_threshold=0.95)
    num_threads = 8
    ops_per_thread = 50

    def worker(thread_id: int):
        rng = np.random.default_rng(thread_id)
        for i in range(ops_per_thread):
            key = f"key_{thread_id}_{i % 10}"
            emb = rng.standard_normal(32).astype(np.float32)
            emb /= np.linalg.norm(emb)

            if i % 2 == 0:
                cache.put(key, f"val_{thread_id}_{i}", embedding=emb)
            else:
                cache.get(key, embedding=emb)
                cache.stats()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, t) for t in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert cache.size <= 50
    st = cache.stats()
    assert st["total_queries"] > 0


# ==============================================================================
# 4. SystemOneEngine Runtime Robustness & Integration Stress
# ==============================================================================

def test_engine_uncalibrated_baseline_execution():
    """Verify SystemOneEngine operates reliably in a completely uncalibrated state."""
    engine = SystemOneEngine(MultiFieldAdversarialSchema, enable_margin_gating=True)

    # Decide without calling engine.calibrate()
    res = engine.decide("Suspicious packet detected on gateway", alpha=0.05)
    assert isinstance(res, DecisionResult)
    assert res.values["route"] in ("ALLOW", "REVIEW", "BLOCK")
    assert isinstance(res.values["is_anomaly"], bool)
    assert 0.0 <= res.values["risk_score"] <= 1.0
    assert isinstance(res.values["flags"], tuple)

    # Conformal sets must be populated
    assert len(res.conformal_sets["route"]) >= 1
    assert len(res.conformal_sets["is_anomaly"]) >= 1
    assert len(res.conformal_sets["risk_score"]) == 1


def test_engine_empty_prompt_and_batch_mismatch():
    """Verify SystemOneEngine with empty prompt and mismatched batch telemetry."""
    engine = SystemOneEngine(MultiFieldAdversarialSchema)

    # 1. Empty string prompt
    res_empty = engine.decide("")
    assert isinstance(res_empty, DecisionResult)
    assert res_empty.prompt == ""

    # 2. Batch with 0 prompts
    assert engine.decide_batch([]) == []

    # 3. Batch with telemetry length mismatch (3 prompts, 1 telemetry item)
    # Must fallback gracefully to evaluating without crashing
    batch_res = engine.decide_batch(
        ["p1", "p2", "p3"],
        telemetry=[{"risk": 0.1}],
    )
    assert len(batch_res) == 3


def test_engine_learn_from_tier2_unknown_field_graceful():
    """Verify learn_from_tier2 ignores unknown schema fields gracefully."""
    engine = SystemOneEngine(MultiFieldAdversarialSchema)
    stats = engine.learn_from_tier2(
        "anomalous request payload",
        target={"non_existent_field": "UNKNOWN", "route": "BLOCK"},
    )
    assert stats["status"] == "updated"
    assert "route" in stats["updated_fields"]
    assert "non_existent_field" not in stats["updated_fields"]


def test_engine_rapid_sequential_stress_loop():
    """Stress test: 50 sequential rapid decisions verifying zero state leakage and sub-2ms speed."""
    import statistics

    engine = SystemOneEngine(MultiFieldAdversarialSchema, use_cache=True)
    prompts = [
        "Normal ping check",
        "Elevated traffic from user agent",
        "SQL injection attempt inside login parameter",
        "Privilege escalation probe detected",
        "Routine backup cron job running",
    ]

    latencies = []
    for i in range(50):
        p = prompts[i % len(prompts)]
        res = engine.decide(p)
        assert res.values["route"] in ("ALLOW", "REVIEW", "BLOCK")
        assert res.latency_ms < 30.0  # Safe frame ceiling against OS virtualization preemption
        latencies.append(res.latency_ms)

    median_lat = statistics.median(latencies)
    assert median_lat < 2.0, f"Expected median latency < 2.0ms, got {median_lat:.3f}ms"


def test_engine_benchmark_minimal_warmup_and_custom_prompts():
    """Verify benchmark utility with minimal warmup and custom prompt set."""
    engine = SystemOneEngine(MultiFieldAdversarialSchema)
    report = engine.benchmark(
        prompts=["custom prompt alpha", "custom prompt beta"],
        iterations=5,
        warmup=1,
    )
    assert report.total_decisions == 5
    assert report.mean_latency_ms > 0.0
    assert report.speedup_factor > 0.0
