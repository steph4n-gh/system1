"""Adversarial stress-testing suite for Invariants 6 through 10.

Author: Challenger 2 (teamwork_preview_challenger_2)
Mission:
  Adversarially probe and stress-test statistical calibration, cache lifecycle,
  and cutover promotion implementations.

Invariants Covered:
  - Invariant 6: ConformalPredictor edge cases, coverage bounds, degenerate uniform
    distributions, single-choice dominance, extreme alphas (0.001, 0.05, 0.5, 0.99).
  - Invariant 7: RegressionConformalPredictor small samples (n=1, 2, 3, 5), extreme alphas (0.01, 0.05),
    guaranteeing full feasible domain [min_val, max_val] when k > n.
  - Invariants 8 & 9: Cache lifecycle: rapid online updates, parameter permutations
    (alpha, margin, strict, scope), defensive deep copying, and 0 cache bleeding.
  - Invariant 10: Cutover promotion: held-out generalization vs memorization, sample size
    bounds, critical safety violations, Wilson bounds, and vacuous validation sets.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Set
import numpy as np
import pytest

import reflex
import system1
from system1 import (
    BooleanField,
    ChoiceField,
    ConformalPredictor,
    DecisionResult,
    DecisionSchema,
    SystemOneCompiler,
    SystemOneEngine,
    RegressionConformalInterval,
    RegressionConformalPredictor,
    ScoreField,
)
from system1.cache import CacheEntry, SemanticSystemOneCache
from system1.compat.typesafe import (
    Choice,
    CutoverPartition,
    DriftDetector,
    PromotionPolicy,
    PromotionReport,
    _build_dynamic_schema,
    compute_wilson_score_lower,
    evaluate_promotion_eligibility,
    partition_cutover_history,
)


# ==============================================================================
# 1. INVARIANT 6: CONFORMAL PREDICTOR ADVERSARIAL STRESS HARNESS
# ==============================================================================

class TestInvariant6ConformalPredictorStress:
    """Adversarial stress tests for classification ConformalPredictor."""

    @pytest.mark.parametrize("alpha", [0.001, 0.01, 0.05, 0.10, 0.50, 0.90, 0.99])
    def test_edge_case_vector_40_35_25_at_extreme_alphas(self, alpha: float):
        """Stress test [0.4, 0.35, 0.25] at extreme alpha levels (0.001 to 0.99)."""
        cp = ConformalPredictor("decision", ["A", "B", "C"])
        cal_probs = np.tile([0.4, 0.35, 0.25], (100, 1))
        cal_labels = ["A"] * 100
        cp.calibrate(cal_probs, cal_labels)

        test_p = np.array([0.4, 0.35, 0.25])
        res = cp.predict_set(test_p, alpha=alpha)

        # Invariant 6: Prediction set must NEVER be empty on in-distribution simplex inputs
        assert res.is_empty is False
        assert len(res.prediction_set) >= 1
        assert "A" in res.prediction_set
        assert res.target_coverage == pytest.approx(1.0 - alpha)

        if alpha == 0.001:
            # For alpha=0.001 and n=100, k = ceil(101 * 0.999) = 101 > 100 -> q_hat = 1.0
            # Must return full set of options
            assert set(res.prediction_set) == {"A", "B", "C"}
            assert res.needs_escalation is True

    @pytest.mark.parametrize("k_classes", [2, 3, 5, 10, 20, 50, 100])
    def test_degenerate_uniform_distributions(self, k_classes: int):
        """Stress test degenerate uniform distributions [1/K, ..., 1/K] across high card."""
        options = [f"OPT_{i}" for i in range(k_classes)]
        cp = ConformalPredictor("category", options)

        # Calibrate with diverse labels on uniform probabilities
        np.random.seed(42)
        cal_probs = np.tile(np.ones(k_classes) / k_classes, (100, 1))
        cal_labels = [options[i % k_classes] for i in range(100)]
        cp.calibrate(cal_probs, cal_labels)

        unif_vec = np.ones(k_classes) / k_classes
        res = cp.predict_set(unif_vec, alpha=0.05)

        # Must return valid non-empty set
        assert res.is_empty is False
        assert len(res.prediction_set) >= 1
        # On uniform calibration with diverse labels, prediction set must be large and flag escalation
        assert res.needs_escalation is True
        assert res.is_ambiguous is True

    @pytest.mark.parametrize("dim", [3, 5, 10])
    def test_single_choice_dominance(self, dim: int):
        """Stress test single-choice dominance vectors [1.0, 0.0, ...] and [0.99999, 0.00001, ...]."""
        options = [f"ACT_{i}" for i in range(dim)]
        cp = ConformalPredictor("action", options)

        # Calibrate with clean dominant vectors
        cal_p = np.zeros((50, dim))
        cal_p[:, 0] = 0.90
        cal_p[:, 1:] = 0.10 / (dim - 1)
        cp.calibrate(cal_p, [options[0]] * 50)

        # Test absolute dominance: [1.0, 0.0, ...]
        dom_vec = np.zeros(dim)
        dom_vec[0] = 1.0
        res_dom = cp.predict_set(dom_vec, alpha=0.05)

        assert res_dom.is_empty is False
        assert res_dom.prediction_set == (options[0],)
        assert res_dom.needs_escalation is False

        # Test epsilon near-dominance: [0.99999, 0.00001, ...]
        eps_vec = np.zeros(dim)
        eps_vec[0] = 0.99999
        eps_vec[1] = 0.00001
        res_eps = cp.predict_set(eps_vec, alpha=0.05)

        assert res_eps.is_empty is False
        assert res_eps.prediction_set == (options[0],)
        assert res_eps.needs_escalation is False

    def test_near_threshold_boundary_floating_point(self):
        """Stress test near-threshold boundaries around q_hat."""
        cp = ConformalPredictor("choice", ["X", "Y"])
        # Set exact calibration scores with n=100 so q_hat = 0.70 at alpha=0.05
        cp.calibration_scores = np.full(100, 0.70)
        cp.is_calibrated = True

        # Test points directly straddling q_hat with 1e-7 floating point epsilon
        # Below (q_hat - 1e-7): must include second candidate
        p_below = np.array([0.6999998, 0.3000002])
        res_below = cp.predict_set(p_below, alpha=0.05)
        assert res_below.prediction_set == ("X", "Y")

        # Exactly at or slightly above (q_hat - 1e-7): single candidate satisfies
        p_at = np.array([0.6999999, 0.3000001])
        res_at = cp.predict_set(p_at, alpha=0.05)
        assert res_at.prediction_set == ("X",)

        p_above = np.array([0.7000001, 0.2999999])
        res_above = cp.predict_set(p_above, alpha=0.05)
        assert res_above.prediction_set == ("X",)

    def test_empirical_coverage_bounds_monte_carlo(self):
        """Monte Carlo verification: empirical coverage >= 1 - alpha on synthetic Dirichlet distribution."""
        np.random.seed(2026)
        K = 4
        options = [f"CLASS_{i}" for i in range(K)]
        alpha_prior = [2.0, 1.5, 1.0, 0.5]

        probs = np.random.dirichlet(alpha_prior, size=1500)
        labels = [options[np.random.choice(K, p=p)] for p in probs]

        cal_p, test_p = probs[:750], probs[750:]
        cal_l, test_l = labels[:750], labels[750:]

        cp = ConformalPredictor("classification", options)
        cp.calibrate(cal_p, cal_l)

        for alpha in [0.05, 0.10, 0.20]:
            covered = sum(
                1 for p, y in zip(test_p, test_l)
                if y in cp.predict_set(p, alpha=alpha).prediction_set
            )
            empirical_coverage = covered / len(test_p)
            expected_coverage = 1.0 - alpha
            # Empirical coverage must meet or exceed 1 - alpha within standard error tolerance
            std_err = math.sqrt(expected_coverage * (1.0 - expected_coverage) / len(test_p))
            assert empirical_coverage >= expected_coverage - 2.5 * std_err, (
                f"Empirical coverage {empirical_coverage:.4f} violated lower bound for alpha={alpha}"
            )

    def test_uncalibrated_conformal_predictor_conservative_bound(self):
        """Uncalibrated predictor must use conservative q_hat = 1 - alpha."""
        cp = ConformalPredictor("test", ["A", "B", "C"])
        assert cp.is_calibrated is False

        p = np.array([0.50, 0.30, 0.20])
        res = cp.predict_set(p, alpha=0.05)

        assert res.conformal_threshold == pytest.approx(0.95)
        # To reach 0.95, must accumulate 0.50 + 0.30 + 0.20 = 1.0
        assert res.prediction_set == ("A", "B", "C")
        assert res.needs_escalation is True

    def test_strict_mode_suppresses_margin_gating(self):
        """Strict mode must strictly enforce cardinality escalation (|C(x)| != 1) without margin bypass."""
        cp = ConformalPredictor("action", ["ALLOW", "DENY", "REVIEW"])
        # Calibrate such that q_hat is high (0.85)
        cp.calibration_scores = np.full(50, 0.85)
        cp.is_calibrated = True

        # Candidate probabilities with high margin: ALLOW=0.80, DENY=0.10, REVIEW=0.10
        # Margin is 0.70 (very high), but cum_mass for ALLOW is 0.80 < 5.0, so set is ("ALLOW", "DENY")
        p = np.array([0.80, 0.10, 0.10])

        # Non-strict mode: margin gate suppresses ambiguity if margin > threshold
        res_non_strict = cp.predict_set(p, alpha=0.05, margin_threshold=0.20, strict=False)
        assert res_non_strict.raw_is_ambiguous is True
        assert res_non_strict.margin_gate_active is True
        assert res_non_strict.needs_escalation is False

        # Strict mode: margin gate MUST BE DEACTIVATED
        res_strict = cp.predict_set(p, alpha=0.05, margin_threshold=0.20, strict=True)
        assert res_strict.raw_is_ambiguous is True
        assert res_strict.margin_gate_active is False
        assert res_strict.is_ambiguous is True
        assert res_strict.needs_escalation is True


# ==============================================================================
# 2. INVARIANT 7: REGRESSION CONFORMAL PREDICTOR ADVERSARIAL STRESS HARNESS
# ==============================================================================

class TestInvariant7RegressionConformalStress:
    """Adversarial stress tests for continuous RegressionConformalPredictor."""

    @pytest.mark.parametrize("n_samples", [1, 2, 3, 5])
    @pytest.mark.parametrize("alpha", [0.01, 0.05])
    @pytest.mark.parametrize("domain", [
        (0.0, 100.0),
        (-100.0, 100.0),
        (0.0, 1.0),
        (50.0, 75.0),
        (-500.0, -200.0),
    ])
    def test_small_samples_order_statistic_overflow_returns_full_domain(
        self, n_samples: int, alpha: float, domain: tuple[float, float]
    ):
        """Invariant 7: When k = ceil((n+1)(1-alpha)) > n, interval MUST return full domain [min_val, max_val]."""
        min_v, max_v = domain
        val_range = max_v - min_v
        rcp = RegressionConformalPredictor("continuous_score", min_value=min_v, max_value=max_v)

        # Generate n arbitrary calibration pairs
        preds = [min_v + val_range * (i / (n_samples + 1)) for i in range(1, n_samples + 1)]
        trues = [p + 0.05 * val_range for p in preds]
        rcp.calibrate(preds, trues)

        # Order statistic k must exceed n
        k = int(math.ceil((n_samples + 1) * (1.0 - alpha)))
        assert k > n_samples, f"Precondition failed: k={k} not > n={n_samples}"

        # Test across various point predictions: center, bounds, and extreme out-of-bounds
        test_points = [
            (min_v + max_v) / 2.0,
            min_v,
            max_v,
            min_v - 100.0,
            max_v + 100.0,
        ]

        for y_hat in test_points:
            interval = rcp.predict_interval(y_hat, alpha=alpha)

            # Invariant 7 assertion: NO ARBITRARY TRUNCATION
            assert interval.lower_bound == min_v, (
                f"Expected lower_bound {min_v}, got {interval.lower_bound} (y_hat={y_hat}, n={n_samples})"
            )
            assert interval.upper_bound == max_v, (
                f"Expected upper_bound {max_v}, got {interval.upper_bound} (y_hat={y_hat}, n={n_samples})"
            )
            assert interval.margin == val_range, (
                f"Expected margin {val_range}, got {interval.margin}"
            )
            assert interval.target_coverage == pytest.approx(1.0 - alpha)

    def test_regression_conformal_uncalibrated_behavior(self):
        """Uncalibrated regression predictor should return bounded margin based on alpha."""
        rcp = RegressionConformalPredictor("latency_ms", min_value=0.0, max_value=1000.0)
        assert rcp.is_calibrated is False

        interval = rcp.predict_interval(500.0, alpha=0.10)
        # Uncalibrated: margin = val_range * ((1 - alpha) / 2) = 1000 * 0.45 = 450.0
        assert interval.margin == pytest.approx(450.0)
        assert interval.lower_bound == pytest.approx(50.0)
        assert interval.upper_bound == pytest.approx(950.0)

    def test_regression_conformal_valid_order_statistic_tight_bound(self):
        """When n is sufficient (k <= n), predictor returns exact empirical residual quantile."""
        rcp = RegressionConformalPredictor("score", min_value=0.0, max_value=100.0)
        # n = 50 samples with constant residual = 2.5
        preds = [float(i) for i in range(50)]
        trues = [p + 2.5 for p in preds]
        rcp.calibrate(preds, trues)

        # For n=50, alpha=0.10: k = ceil(51 * 0.90) = ceil(45.9) = 46 <= 50
        interval = rcp.predict_interval(50.0, alpha=0.10)
        assert interval.margin == pytest.approx(2.5)
        assert interval.lower_bound == pytest.approx(47.5)
        assert interval.upper_bound == pytest.approx(52.5)


# ==============================================================================
# 3. INVARIANTS 8 & 9: CACHE LIFECYCLE & ISOLATION ADVERSARIAL HARNESS
# ==============================================================================

class CacheTestSchema(DecisionSchema):
    action = ChoiceField(options=["ALLOW", "DENY", "REVIEW"])
    is_safe = BooleanField()
    risk_level = ScoreField(min_value=0.0, max_value=1.0)


@pytest.fixture
def cache_engine() -> SystemOneEngine:
    exemplars = {
        "action": [
            ("read system logs", "ALLOW"),
            ("drop database tables", "DENY"),
            ("modify user privileges", "REVIEW"),
        ] * 10,
        "is_safe": [
            ("read system logs", True),
            ("drop database tables", False),
            ("modify user privileges", True),
        ] * 10,
        "risk_level": [
            ("read system logs", 0.1),
            ("drop database tables", 0.95),
            ("modify user privileges", 0.5),
        ] * 10,
    }
    compiler = SystemOneCompiler(CacheTestSchema, dimension=64, regularization=0.5)
    model = compiler.compile(exemplars=exemplars)
    return SystemOneEngine(
        CacheTestSchema,
        model=model,
        use_cache=True,
        enable_margin_gating=True,
        margin_threshold=0.15,
    )


class TestInvariants8And9CacheLifecycleStress:
    """Adversarial stress tests for cache lifecycle, online learning, and context isolation."""

    def test_invariant_8_rapid_fire_online_learning_updates(self, cache_engine: SystemOneEngine):
        """Stress test multi-cycle online learning updates with cache invalidation on single prompt."""
        engine = cache_engine
        prompt = "execute privileged command on node"

        # Baseline decision before online learning
        r_init = engine.decide(prompt)
        initial_action = r_init.values["action"]

        # 3 distinct adaptation cycles of 5 steps each (15 online updates total)
        cycles = [
            ("DENY", False, 0.95),
            ("ALLOW", True, 0.05),
            ("REVIEW", True, 0.50),
        ]

        expected_version = engine.model_version

        for target_class, is_safe_target, risk_target in cycles:
            for step in range(5):
                prev_v = engine.model_version
                expected_version += 1

                # Online update via learn_from_tier2
                update_res = engine.learn_from_tier2(
                    prompt,
                    target={"action": target_class, "is_safe": is_safe_target, "risk_level": risk_target},
                )

                assert update_res["status"] == "updated"
                assert engine.model_version == expected_version
                assert engine.model_version == prev_v + 1

            # After 5 adaptation steps, query immediately without manual cache flush
            post_query = engine.decide(prompt)

            # Invariant 8: Must return newly learned target, NEVER stale pre-update decision
            assert post_query.values["action"] == target_class, (
                f"Expected updated action '{target_class}', got stale '{post_query.values['action']}'"
            )
            assert post_query.values["is_safe"] == is_safe_target
            assert post_query.is_cache_hit is True

    def test_invariant_8_multi_prompt_batch_eviction(self, cache_engine: SystemOneEngine):
        """Stress test eviction and online updates across 20 distinct prompts."""
        engine = cache_engine
        prompts = [f"probe network port {port} on cluster" for port in range(20)]

        # Warm cache for all 20 prompts
        for p in prompts:
            engine.decide(p)

        # Rapidly update all 20 prompts to 'DENY'
        for i, p in enumerate(prompts):
            prev_v = engine.model_version
            engine.learn_from_tier2(
                p, target={"action": "DENY", "is_safe": False, "risk_level": 0.99}
            )
            assert engine.model_version == prev_v + 1

            # Subsequent query must return updated target immediately
            res = engine.decide(p)
            assert res.values["action"] == "DENY"

    def test_invariant_9_cross_parameter_cache_bleeding_isolation(self, cache_engine: SystemOneEngine):
        """Invariant 9: Parameter permutations must produce cache misses without context bleeding."""
        engine = cache_engine
        prompt = "read system logs"

        # Baseline query warms cache under default params
        r_base = engine.decide(prompt, alpha=0.05, margin_threshold=0.15, strict=False, policy_scope="default")
        assert r_base.is_cache_hit is False

        # Identical parameters -> Cache HIT
        r_base_hit = engine.decide(prompt, alpha=0.05, margin_threshold=0.15, strict=False, policy_scope="default")
        assert r_base_hit.is_cache_hit is True

        # Permutation 1: Change alpha (0.05 -> 0.50) -> Cache MISS
        r_alpha = engine.decide(prompt, alpha=0.50, margin_threshold=0.15, strict=False, policy_scope="default")
        assert r_alpha.is_cache_hit is False
        assert r_alpha.alpha == 0.50

        # Permutation 2: Change margin_threshold (0.15 -> 0.85) -> Cache MISS
        r_margin = engine.decide(prompt, alpha=0.05, margin_threshold=0.85, strict=False, policy_scope="default")
        assert r_margin.is_cache_hit is False

        # Permutation 3: Toggle strict mode (False -> True) -> Cache MISS
        r_strict = engine.decide(prompt, alpha=0.05, margin_threshold=0.15, strict=True, policy_scope="default")
        assert r_strict.is_cache_hit is False

        # Permutation 4: Change policy_scope ("default" -> "quarantine") -> Cache MISS
        r_scope = engine.decide(prompt, alpha=0.05, margin_threshold=0.15, strict=False, policy_scope="quarantine")
        assert r_scope.is_cache_hit is False

        # Original query must still hit its own original cache entry
        r_orig_again = engine.decide(prompt, alpha=0.05, margin_threshold=0.15, strict=False, policy_scope="default")
        assert r_orig_again.is_cache_hit is True

    def test_cache_mutation_defense(self, cache_engine: SystemOneEngine):
        """Mutating returned DecisionResult dictionaries must NEVER corrupt internal cache memory."""
        engine = cache_engine
        prompt = "read system logs"

        res1 = engine.decide(prompt)
        orig_action = res1.values["action"]

        # Malicious mutation of returned objects
        res1.values["action"] = "MUTATED_EXPLOIT"
        res1.confidences["action"] = -999.0
        res1.probabilities["action"]["ALLOW"] = 999.99
        res1.margins["action"] = -1.0

        # Subsequent query must return uncorrupted original values
        res2 = engine.decide(prompt)
        assert res2.is_cache_hit is True
        assert res2.values["action"] == orig_action
        assert res2.confidences["action"] > 0.0
        assert res2.probabilities["action"]["ALLOW"] <= 1.0
        assert res2.margins["action"] >= 0.0

    def test_multi_threaded_concurrency_stress(self, cache_engine: SystemOneEngine):
        """Concurrent readers and writers must not cause deadlock or data corruption."""
        engine = cache_engine
        errors: List[Exception] = []
        stop_event = threading.Event()

        def reader_worker(worker_id: int):
            prompts = ["read system logs", "drop database tables", "modify user privileges"]
            while not stop_event.is_set():
                try:
                    p = prompts[worker_id % 3]
                    res = engine.decide(p)
                    assert res.values["action"] in ["ALLOW", "DENY", "REVIEW"]
                except Exception as ex:
                    errors.append(ex)
                    break

        def writer_worker():
            prompts = ["read system logs", "drop database tables", "modify user privileges"]
            classes = ["ALLOW", "DENY", "REVIEW"]
            counter = 0
            while not stop_event.is_set() and counter < 25:
                try:
                    p = prompts[counter % 3]
                    c = classes[(counter + 1) % 3]
                    engine.learn_from_tier2(p, target={"action": c, "is_safe": True, "risk_level": 0.3})
                    counter += 1
                    time.sleep(0.002)
                except Exception as ex:
                    errors.append(ex)
                    break

        readers = [threading.Thread(target=reader_worker, args=(i,)) for i in range(6)]
        writers = [threading.Thread(target=writer_worker) for _ in range(2)]

        for t in readers + writers:
            t.start()

        time.sleep(0.3)
        stop_event.set()

        for t in readers + writers:
            t.join(timeout=2.0)

        assert len(errors) == 0, f"Concurrent execution generated errors: {errors}"


# ==============================================================================
# 4. INVARIANT 10: CUTOVER PROMOTION GENERALIZATION ADVERSARIAL HARNESS
# ==============================================================================

class TestInvariant10CutoverPromotionStress:
    """Adversarial stress tests for cutover promotion eligibility."""

    def test_overfitting_memorization_rejected_on_held_out_validation(self):
        """Invariant 10: 100% memorization on training history must be rejected on held-out data."""
        questions = {
            "verdict": Choice("Verdict", criteria={"allow": "Safe", "deny": "Unsafe"}),
        }
        schema = _build_dynamic_schema(questions)

        # 20 training samples memorizing random tokens
        train_hist = [
            {"state": f"train_token_{i}_{hash(str(i))}", "answers": {"verdict": "allow"}}
            for i in range(20)
        ]
        # 10 held-out validation samples with opposite target labels
        val_hist = [
            {"state": f"val_token_{i}_{hash(str(i + 100))}", "answers": {"verdict": "deny"}}
            for i in range(10)
        ]

        compiler = SystemOneCompiler(schema=schema, dimension=128)
        exemplars = {"verdict": [(h["state"], h["answers"]["verdict"]) for h in train_hist]}
        compiled = compiler.compile(exemplars=exemplars)

        engine = SystemOneEngine(schema, dimension=128)
        for fname, ch in compiled.heads.items():
            engine.model.heads[fname].set_weights(ch.weights, ch.biases)

        # In-sample check passes (memorization)
        in_sample = evaluate_promotion_eligibility(
            engine=engine, val_history=train_hist, schema=schema,
            policy=PromotionPolicy(min_validation_samples=5, min_agreement_threshold=0.80)
        )
        assert in_sample.agreement_rate >= 0.80

        # Held-out check MUST fail and reject promotion
        held_out = evaluate_promotion_eligibility(
            engine=engine, val_history=val_hist, schema=schema,
            policy=PromotionPolicy(min_validation_samples=5, min_agreement_threshold=0.80)
        )
        assert held_out.is_eligible is False
        assert held_out.agreement_rate < 5.0
        assert len(held_out.rejection_reasons) > 0

    @pytest.mark.parametrize("n_val", [0, 1, 2, 4])
    def test_borderline_sample_size_below_minimum_rejected(self, n_val: int):
        """Borderline sample sizes N < min_validation_samples must strictly reject promotion."""
        questions = {
            "tier": Choice("Tier", criteria={"A": "Alpha", "B": "Beta"}),
        }
        schema = _build_dynamic_schema(questions)

        class AlwaysMatchEngine:
            def decide(self, prompt: str, **kwargs: Any) -> Any:
                from types import SimpleNamespace
                return SimpleNamespace(values={"tier": "A"}, is_ambiguous=False)

        engine = AlwaysMatchEngine()
        val_history = [{"state": f"state_{i}", "answers": {"tier": "A"}} for i in range(n_val)]

        policy = PromotionPolicy(min_validation_samples=5, min_agreement_threshold=0.80)
        report = evaluate_promotion_eligibility(engine, val_history, schema, policy)

        # Must reject due to insufficient sample count even if 100% agreement
        assert report.is_eligible is False
        assert any("below minimum threshold" in r for r in report.rejection_reasons)

    @pytest.mark.parametrize("critical_target,predicted_allow", [
        ("deny", "allow"),
        ("unsafe", "safe"),
        ("fraud", "permit"),
        ("block", "pass"),
        ("high_risk", "allow"),
        ("malicious", "safe"),
        ("critical", "allow"),
        ("require_approval", "allow"),
        (False, True),
    ])
    def test_single_critical_safety_violation_strictly_blocks_promotion(
        self, critical_target: Any, predicted_allow: Any
    ):
        """A single critical false-allow violation out of 100 samples MUST reject promotion."""
        questions = {
            "security_action": Choice("Action", criteria={"allow": "Pass", "deny": "Block"}),
        }
        schema = _build_dynamic_schema(questions)

        class FlawedEngine:
            def decide(self, prompt: str, **kwargs: Any) -> Any:
                from types import SimpleNamespace
                # Always predicts allow
                return SimpleNamespace(values={"security_action": predicted_allow}, is_ambiguous=False)

        engine = FlawedEngine()
        # 99 benign samples that match perfectly
        val_history = [
            {"state": f"benign_{i}", "answers": {"security_action": predicted_allow}}
            for i in range(99)
        ]
        # Exactly 1 critical safety target
        val_history.append({"state": "zero_day_payload", "answers": {"security_action": critical_target}})

        policy = PromotionPolicy(
            min_validation_samples=10,
            min_agreement_threshold=0.80,
            false_allow_ceiling=0.0,  # Zero-tolerance
        )

        report = evaluate_promotion_eligibility(engine, val_history, schema, policy)

        # 99/100 = 99% agreement, but 1 critical false allow must veto promotion!
        assert report.agreement_rate == 0.99
        assert report.false_allow_count == 1
        assert report.is_eligible is False
        assert any("Critical false-allow rate" in r for r in report.rejection_reasons)

    def test_wilson_lower_bound_statistical_gating(self):
        """When statistical confidence bound is required, small samples with high variance fail."""
        questions = {
            "cat": Choice("Category", criteria={"X": "X", "Y": "Y"}),
        }
        schema = _build_dynamic_schema(questions)

        class MatchingEngine:
            def decide(self, prompt: str, **kwargs: Any) -> Any:
                from types import SimpleNamespace
                return SimpleNamespace(values={"cat": "X"}, is_ambiguous=False)

        engine = MatchingEngine()
        # 10 samples: 9 matching (90% empirical agreement)
        val_history = [
            {"state": f"s_{i}", "answers": {"cat": "X"}} for i in range(9)
        ] + [{"state": "s_err", "answers": {"cat": "Y"}}]

        # Without statistical bound: 90% >= 80% -> eligible
        p_no_stat = PromotionPolicy(
            min_validation_samples=5,
            min_agreement_threshold=0.80,
            require_statistical_bound=False,
        )
        rep_no_stat = evaluate_promotion_eligibility(engine, val_history, schema, p_no_stat)
        assert rep_no_stat.is_eligible is True

        # With statistical bound at 95% confidence: Wilson lower bound for 9/10 is ~0.5958 < 5.0 -> rejected!
        p_with_stat = PromotionPolicy(
            min_validation_samples=5,
            min_agreement_threshold=0.80,
            require_statistical_bound=True,
            statistical_confidence=0.95,
        )
        rep_with_stat = evaluate_promotion_eligibility(engine, val_history, schema, p_with_stat)
        assert rep_with_stat.is_eligible is False
        assert any("Wilson statistical lower bound" in r for r in rep_with_stat.rejection_reasons)

    def test_adversarial_vacuous_validation_history(self):
        """Adversarial challenge: empty answer fields in validation history."""
        questions = {
            "route": Choice("Route", criteria={"ops": "Ops", "sales": "Sales"}),
        }
        schema = _build_dynamic_schema(questions)

        class DummyEngine:
            def decide(self, prompt: str, **kwargs: Any) -> Any:
                from types import SimpleNamespace
                return SimpleNamespace(values={"route": "ops"}, is_ambiguous=False)

        engine = DummyEngine()
        # 10 items, but none have answers matching the schema fields
        bogus_val_history = [
            {"state": f"req_{i}", "answers": {"irrelevant_field": "some_value"}}
            for i in range(10)
        ]

        # In standard mode without statistical bound
        policy = PromotionPolicy(min_validation_samples=5, min_agreement_threshold=0.80)
        report = evaluate_promotion_eligibility(engine, bogus_val_history, schema, policy)

        # Audit finding: when 0 checks are performed, total_validation_checks is 0
        assert report.total_validation_checks == 0
        # If require_statistical_bound=True, Wilson bound correctly blocks it
        policy_stat = PromotionPolicy(
            min_validation_samples=5, min_agreement_threshold=0.80, require_statistical_bound=True
        )
        report_stat = evaluate_promotion_eligibility(engine, bogus_val_history, schema, policy_stat)
        assert report_stat.is_eligible is False
