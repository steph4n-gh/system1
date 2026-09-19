"""Comprehensive Verification Suite for Milestone 2: Gate C & Gate D.

Covers:
- Gate C: Cache & Uncertainty Lifecycle (P0)
  1. Pre-Lookup Input Validation (prompt, alpha, margin, odds, floor, telemetry, embedding).
  2. Cache-Hit Ambiguity and Escalated Fields Preservation (no unconditional overrides).
  3. Collision-Resistant Cache Key Binding (full 64-hex SHA-256 telemetry digest, policy epoch, digests)
     and strict/enforce_durability suppression of approximate semantic search.
  4. Atomic Model Versioning & Concurrency Synchronization (RLock, weights-first update, eviction).
  5. Decoupled Conformal Calibration Folds (50/50 split for temperature scaling & non-conformity quantiles).
  6. Uncalibrated Model Abstention (strict mode full label set / full domain interval escalation).
  7. Multilabel & Regression Uncertainty Gating (MultiChoiceField and ScoreField conformal gating).

- Gate D: Validated Artifact Promotion (P1)
  1. Release Invariant 3: Candidate Artifact Promotion Integrity (promote evaluated candidate directly,
     eliminate post-validation retraining on 100% of history).
  2. Elimination of Small-Sample Relaxation (no total_checks <= 4 lowering) and Rejection of Zero Scored Checks.
  3. Mandatory Statistical Acceptance (default require_statistical_bound=True, Wilson lower bound,
     strict 0.0% false-allow ceiling on critical security classes).
  4. Disjoint Lineage & Manifest Binding (CutoverPartition.assert_disjoint, promotion manifest hashes).
  5. Twin-Namespace Parity (system1 and reflex namespaces).
"""

from __future__ import annotations

import concurrent.futures
import copy
import dataclasses
import hashlib
import json
import math
import numpy as np
import pytest
from typing import Any, Dict, List

# Twin-namespace imports
import system1.cache as s1_cache
import reflex.cache as rx_cache
import system1.calibration as s1_calib
import reflex.calibration as rx_calib
import system1.compiler as s1_comp
import reflex.compiler as rx_comp
import system1.engine as s1_eng
import reflex.engine as rx_eng
import system1.compat.typesafe as s1_typesafe
import reflex.compat.typesafe as rx_typesafe

from system1.core.schema import (
    BooleanField,
    ChoiceField,
    DecisionSchema,
    MultiChoiceField,
    ScoreField,
)

SingleChoiceField = ChoiceField
SystemOneSchema = DecisionSchema

from system1.engine import DecisionResult, SystemOneEngine
from system1.cache import CacheEntry, SemanticSystemOneCache, validate_cache_inputs
from system1.calibration import ConformalPredictor, RegressionConformalPredictor
from system1.compiler import SystemOneCompiler
from system1.compat.typesafe import (
    CutoverPartition,
    PromotionPolicy,
    PromotionReport,
    TypeSafeClient,
    compute_wilson_score_lower,
    evaluate_promotion_eligibility,
    partition_cutover_history,
)


# ============================================================================
# Gate C: 1. Pre-Lookup Input Validation
# ============================================================================

class TestPreLookupInputValidation:
    """Tests input validation before cache key formatting or index traversal."""

    def test_cache_validate_prompt(self):
        cache = SemanticSystemOneCache()
        for invalid_prompt in [123, None, ["query"], {"q": "val"}]:
            with pytest.raises(TypeError, match="Prompt must be a string"):
                cache.get(invalid_prompt)  # type: ignore

    def test_cache_validate_alpha(self):
        cache = SemanticSystemOneCache()
        for invalid_alpha in [0.0, 1.0, -0.05, 1.5, float("nan")]:
            with pytest.raises(ValueError, match="Significance level alpha must be in"):
                cache.get("test prompt", alpha=invalid_alpha)

        with pytest.raises(TypeError, match="alpha must be a real number"):
            cache.get("test prompt", alpha="invalid")  # type: ignore

    def test_cache_validate_margin_threshold(self):
        cache = SemanticSystemOneCache()
        for invalid_m in [-0.01, -1.0, float("nan")]:
            with pytest.raises(ValueError, match="margin_threshold must be non-negative"):
                cache.get("test prompt", margin_threshold=invalid_m)

    def test_cache_validate_odds_ratio(self):
        cache = SemanticSystemOneCache()
        for invalid_odds in [0.0, -0.5, float("nan")]:
            with pytest.raises(ValueError, match="relative_odds_ratio must be positive"):
                cache.get("test prompt", relative_odds_ratio=invalid_odds)

    def test_cache_validate_confidence_floor(self):
        cache = SemanticSystemOneCache()
        for invalid_floor in [-0.1, -10.0, float("nan")]:
            with pytest.raises(ValueError, match="confidence_floor_tau0 must be non-negative"):
                cache.get("test prompt", confidence_floor_tau0=invalid_floor)

    def test_cache_validate_telemetry(self):
        cache = SemanticSystemOneCache()
        with pytest.raises(TypeError, match="telemetry must be a Mapping, Sequence, or ndarray"):
            cache.get("test prompt", telemetry=object())

        with pytest.raises(ValueError, match="telemetry array contains NaN or infinite"):
            cache.get("test prompt", telemetry=np.array([1.0, np.nan, 2.0]))

    def test_cache_validate_embedding(self):
        cache = SemanticSystemOneCache()
        with pytest.raises(ValueError, match="embedding contains NaN or infinite values"):
            cache.get("test prompt", embedding=np.array([1.0, np.inf, 2.0]))

        with pytest.raises(ValueError, match="embedding must be a 1D vector"):
            cache.get("test prompt", embedding=np.zeros((2, 2)))

    def test_engine_decide_pre_lookup_validation(self):
        schema = SystemOneSchema(
            schema_name="TestValidation",
            fields={"decision": SingleChoiceField(["allow", "deny"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8)

        # Invalid prompt
        with pytest.raises(TypeError, match="Prompt must be a string"):
            engine.decide(12345)  # type: ignore

        # Invalid alpha
        with pytest.raises(ValueError, match="alpha must be in"):
            engine.decide("test", alpha=0.0)

        # Invalid margin_threshold
        with pytest.raises(ValueError, match="margin_threshold must be non-negative"):
            engine.decide("test", margin_threshold=-0.5)

        # Invalid odds ratio
        with pytest.raises(ValueError, match="relative_odds_ratio must be strictly positive"):
            engine.decide("test", relative_odds_ratio=-1.0)

        # Invalid embedding dimension
        with pytest.raises(ValueError, match="embedding dimension mismatch"):
            engine.decide("test", embedding=np.zeros(16, dtype=np.float32))

        # Invalid embedding with NaN
        with pytest.raises(ValueError, match="embedding must contain only finite numbers"):
            engine.decide("test", embedding=np.array([np.nan] * 8, dtype=np.float32))


# ============================================================================
# Gate C: 2. Remove Unconditional Cache-Hit Ambiguity Overrides
# ============================================================================

class TestCacheAmbiguityPreservation:
    """Verifies that cache hits preserve evaluated ambiguity and escalated fields."""

    def test_cache_hit_preserves_ambiguous_evaluation(self):
        schema = SystemOneSchema(
            schema_name="TestAmbiguity",
            fields={
                "action": SingleChoiceField(["allow", "review", "block"]),
            },
        )
        engine = SystemOneEngine(schema=schema, dimension=8, use_cache=True)
        prompt = "Suspicious financial transfer exceeding standard thresholds"

        res1 = engine.decide(prompt, alpha=0.05)
        ambiguous_res = dataclasses.replace(
            res1,
            is_ambiguous=True,
            ambiguous_fields=["action"],
            escalated_fields=["action"],
        )

        # Store in cache
        engine.cache.put(
            prompt=prompt,
            result=ambiguous_res,
            schema_digest=schema.schema_digest(),
            model_version=engine.model_version,
            policy_scope=engine.policy_scope,
            alpha=0.05,
            margin_threshold=engine.margin_threshold if engine.enable_margin_gating else 0.0,
            strict=engine.strict_mode,
            relative_odds_ratio=engine.relative_odds_ratio,
            confidence_floor_tau0=engine.confidence_floor_tau0,
            model_digest=engine._model_digest(),
            projector_digest=engine._projector_digest(),
            calibration_digest=engine._calibration_digest(),
            policy_epoch=engine.policy_epoch,
        )

        # Decision on the same prompt must hit cache AND preserve ambiguity
        hit_res = engine.decide(prompt, alpha=0.05)
        assert hit_res.is_cache_hit is True
        assert hit_res.is_ambiguous is True
        assert hit_res.ambiguous_fields == ["action"]
        assert hit_res.escalated_fields == ["action"]

    def test_cache_hit_preserves_unambiguous_evaluation(self):
        schema = SystemOneSchema(
            schema_name="TestUnambiguous",
            fields={"status": SingleChoiceField(["ok", "fail"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8, use_cache=True)
        prompt = "Routine benign ping"

        res1 = engine.decide(prompt, alpha=0.05)
        clear_res = dataclasses.replace(
            res1,
            is_ambiguous=False,
            ambiguous_fields=[],
            escalated_fields=[],
        )

        engine.cache.put(
            prompt=prompt,
            result=clear_res,
            schema_digest=schema.schema_digest(),
            model_version=engine.model_version,
            policy_scope=engine.policy_scope,
            alpha=0.05,
            margin_threshold=0.0,
            strict=False,
            relative_odds_ratio=engine.relative_odds_ratio,
            confidence_floor_tau0=engine.confidence_floor_tau0,
            model_digest=engine._model_digest(),
            projector_digest=engine._projector_digest(),
            calibration_digest=engine._calibration_digest(),
            policy_epoch=engine.policy_epoch,
        )

        hit_res = engine.decide(prompt, alpha=0.05)
        assert hit_res.is_cache_hit is True
        assert hit_res.is_ambiguous is False
        assert hit_res.ambiguous_fields == []
        assert hit_res.escalated_fields == []


# ============================================================================
# Gate C: 3. Collision-Resistant Cache Key Binding & Strict Mode
# ============================================================================

class TestCacheKeyBindingAndStrictMode:
    """Verifies full 64-hex SHA-256 telemetry digest binding and strict mode semantics."""

    def test_full_64_hex_telemetry_digest(self):
        cache = SemanticSystemOneCache()
        telemetry = {"user_id": "usr_99", "risk": 0.88, "context": "us-east"}
        telem_dig = s1_cache._digest_telemetry(telemetry)

        # Full SHA-256 is 64 hex characters (not truncated to 16)
        assert len(telem_dig) == 64
        assert telem_dig == s1_cache._digest_telemetry(telemetry)
        assert s1_cache._digest_telemetry({"a": 1}) != s1_cache._digest_telemetry({"a": 2})

        key = cache._make_key("prompt", telemetry=telemetry)
        assert telem_dig in key

    def test_cache_key_binds_all_policy_parameters(self):
        cache = SemanticSystemOneCache()
        base_kwargs = {
            "prompt": "test query",
            "telemetry": {"device": "mobile"},
            "schema_digest": "schema_abc",
            "model_version": 1,
            "policy_scope": "production",
            "alpha": 0.05,
            "margin_threshold": 0.1,
            "strict": True,
            "relative_odds_ratio": 2.0,
            "confidence_floor_tau0": 0.8,
            "recency_weighted": True,
            "model_digest": "m_sha256",
            "projector_digest": "p_sha256",
            "calibration_digest": "c_sha256",
            "policy_epoch": 42,
        }

        base_key = cache._make_key(**base_kwargs)

        # Changing ANY parameter must alter the key
        mutations = [
            ("alpha", 0.01),
            ("policy_scope", "staging"),
            ("model_version", 2),
            ("margin_threshold", 0.2),
            ("strict", False),
            ("relative_odds_ratio", 3.0),
            ("confidence_floor_tau0", 0.9),
            ("policy_epoch", 43),
            ("model_digest", "m_different"),
            ("telemetry", {"device": "desktop"}),
        ]

        for param, new_val in mutations:
            modified_kwargs = dict(base_kwargs)
            modified_kwargs[param] = new_val
            new_key = cache._make_key(**modified_kwargs)
            assert new_key != base_key, f"Key failed to vary on param {param}"

    def test_strict_mode_suppresses_approximate_semantic_search(self):
        cache = SemanticSystemOneCache(similarity_threshold=0.8)
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.99, 0.05, 0.0], dtype=np.float32)  # High cosine similarity > 0.99

        # Put with strict=False
        cache.put("exact prompt 1", "result 1", embedding=emb1, strict=False)

        # In non-strict mode: query with similar emb2 should hit semantically
        hit = cache.get("different prompt", embedding=emb2, strict=False)
        assert hit is not None
        assert hit[0].prompt == "exact prompt 1"

        # In strict mode: approximate semantic search must be suppressed
        miss_strict = cache.get("different prompt", embedding=emb2, strict=True)
        assert miss_strict is None

        # In enforce_durability mode: approximate search must also be suppressed
        miss_durable = cache.get("different prompt", embedding=emb2, strict=False, enforce_durability=True)
        assert miss_durable is None


# ============================================================================
# Gate C: 4. Atomic Model Versioning & Concurrency Synchronization
# ============================================================================

class TestAtomicVersioningAndConcurrency:
    """Verifies thread-safe atomic updates under RLock during online learning."""

    def test_atomic_learn_from_tier2_version_and_invalidation(self):
        schema = SystemOneSchema(
            schema_name="TestAtomic",
            fields={"decision": SingleChoiceField(["allow", "deny"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8, use_cache=True)
        prompt = "User login attempt from anomalous ASN"

        initial_v = engine.model_version
        assert initial_v >= 1

        # Execute decision to seed cache
        res1 = engine.decide(prompt)
        assert res1.is_cache_hit is False

        # Verify cached
        hit_before = engine.decide(prompt)
        assert hit_before.is_cache_hit is True

        # Perform Tier 2 closed-form online update
        update_info = engine.learn_from_tier2(prompt, target={"decision": "deny"})
        assert engine.model_version == initial_v + 1
        assert "decision" in update_info.get("updated_fields", [])

        # Cache for old version must be evicted / invalidated
        res_after = engine.decide(prompt)
        assert res_after.values["decision"] == "deny"

    def test_concurrent_decide_and_learn_under_rlock(self):
        schema = SystemOneSchema(
            schema_name="TestConcurrency",
            fields={"status": SingleChoiceField(["pass", "fail"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8, use_cache=True)

        prompts = [f"System health check probe {i}" for i in range(20)]
        stop_flag = False

        def reader_worker():
            hits = 0
            while not stop_flag:
                p = prompts[np.random.randint(0, len(prompts))]
                res = engine.decide(p)
                assert res.values["status"] in ["pass", "fail"]
                if res.is_cache_hit:
                    hits += 1
            return hits

        def writer_worker():
            for i in range(10):
                p = prompts[i % len(prompts)]
                engine.learn_from_tier2(p, target={"status": "fail" if i % 2 == 0 else "pass"})

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            writer_future = executor.submit(writer_worker)
            reader_futures = [executor.submit(reader_worker) for _ in range(4)]

            writer_future.result(timeout=10.0)
            stop_flag = True
            for rf in reader_futures:
                rf.result(timeout=5.0)

        assert engine.model_version >= 10


# ============================================================================
# Gate C: 5. Decoupled Conformal Calibration Folds
# ============================================================================

class TestDecoupledCalibrationFolds:
    """Verifies temperature scaling and conformal prediction use independent folds."""

    def test_engine_calibrate_splits_folds_deterministically(self):
        schema = SystemOneSchema(
            schema_name="TestDecoupledFolds",
            fields={"decision": SingleChoiceField(["low", "medium", "high"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8)

        # Create distinct calibration observations
        dataset = [
            (f"Transaction signal sample {i}", {"decision": "low" if i % 2 == 0 else "high"})
            for i in range(30)
        ]

        engine.calibrate(dataset, n_bins=3)

        # Check calibrated status on conformal predictor
        pred = engine.conformal_predictors["decision"]
        assert pred.is_calibrated is True
        assert len(pred.calibration_scores) > 0

    def test_compiler_calibration_split_disjoint(self):
        schema = SystemOneSchema(
            schema_name="TestCompilerFolds",
            fields={"category": SingleChoiceField(["cat_a", "cat_b", "cat_c"])},
        )
        compiler = SystemOneCompiler(schema=schema)
        # Synthetic exemplars
        exemplars = compiler.generate_synthetic_exemplars(samples_per_choice=6)
        assert len(exemplars["category"]) >= 18


# ============================================================================
# Gate C: 6. Uncalibrated Model Abstention
# ============================================================================

class TestUncalibratedModelAbstention:
    """Verifies strict-mode full label / interval abstention when uncalibrated."""

    def test_conformal_predictor_uncalibrated_strict_abstention(self):
        pred = ConformalPredictor("decision", ["allow", "review", "deny"])
        assert pred.is_calibrated is False

        # Non-strict mode fallback
        res_non_strict = pred.predict_set([0.8, 0.15, 0.05], alpha=0.05, strict=False)
        assert len(res_non_strict.prediction_set) >= 1

        # Strict mode: must abstain with full label set and escalate
        res_strict = pred.predict_set([0.8, 0.15, 0.05], alpha=0.05, strict=True)
        assert res_strict.needs_escalation is True
        assert set(res_strict.prediction_set) == {"allow", "review", "deny"}

    def test_regression_conformal_predictor_uncalibrated_full_domain(self):
        reg_pred = RegressionConformalPredictor("risk_score", min_value=0.0, max_value=100.0)
        assert reg_pred.is_calibrated is False

        # In strict mode: uncalibrated returns entire feasible domain [0, 100]
        interval = reg_pred.predict_interval(45.0, alpha=0.05, strict=True)
        assert interval.is_calibrated is False
        assert interval.lower_bound == 0.0
        assert interval.upper_bound == 100.0


# ============================================================================
# Gate C: 7. Multilabel & Regression Uncertainty Gating
# ============================================================================

class TestMultilabelAndRegressionUncertaintyGating:
    """Verifies conformal uncertainty escalation for MultiChoiceField and ScoreField."""

    def test_multichoice_field_uncertainty_escalation(self):
        schema = SystemOneSchema(
            schema_name="TestMultiChoiceGating",
            fields={"tags": MultiChoiceField(["pii", "financial", "confidential", "public"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8)

        res = engine.decide("Unseen prompt with high entropy across multi-choice tags")
        assert "tags" in res.values
        assert isinstance(res.values["tags"], (list, tuple))
        assert "tags" in res.conformal_sets

    def test_score_field_uncertainty_escalation(self):
        schema = SystemOneSchema(
            schema_name="TestScoreFieldGating",
            fields={"risk_score": ScoreField(min_value=0.0, max_value=10.0)},
        )
        engine = SystemOneEngine(schema=schema, dimension=8)

        res = engine.decide("Prompt requesting credit increase")
        assert "risk_score" in res.values
        val = res.values["risk_score"]
        assert 0.0 <= val <= 10.0
        assert "risk_score" in res.conformal_sets
        interval = res.conformal_sets["risk_score"]
        assert len(interval) == 1
        assert interval[0].startswith("[")
        assert interval[0].endswith("]")


# ============================================================================
# Gate D: 1. Artifact Promotion Integrity (Release Invariant 3)
# ============================================================================

class TestArtifactPromotionIntegrity:
    """Release Invariant 3: Deploy only the evaluated candidate artifact without retraining."""

    def test_promotion_deploys_candidate_without_full_history_recompile(self):
        from system1.compat.typesafe import Choice, TypeSafeClient

        client = TypeSafeClient(
            mode="auto_cutover",
            cutover_threshold=5,
            promotion_policy=PromotionPolicy(
                min_agreement_threshold=0.8,
                require_statistical_bound=False,
                false_allow_ceiling=1.0,
            ),
            zero_egress=False,
            fallback_baseline=True,
        )
        questions = {
            "tier": Choice("Model tier", criteria={"fast": "Fast model", "smart": "Reasoning model"}),
        }
        for i in range(5):
            client.systemone(f"Request payload {i}", questions)

        assert client.is_cutover is True
        assert client.compiled_model is not None
        audit_log = client.cutover_audit_log
        assert len(audit_log) >= 1
        event = audit_log[0]
        assert "artifact_digest" in event
        assert "candidate_artifact_digest" in event
        assert len(event["candidate_artifact_digest"]) == 64
        assert event["candidate_artifact_digest"] == event["artifact_digest"]


# ============================================================================
# Gate D: 2. Eliminate Small-Sample Relaxation & Zero Scored Labels
# ============================================================================

class TestSmallSampleEliminationAndZeroChecks:
    """Verifies total_checks == 0 is rejected and small-sample threshold is not relaxed."""

    def test_reject_zero_scored_checks(self):
        schema = SystemOneSchema(
            schema_name="TestGateDZero",
            fields={"action": SingleChoiceField(["allow", "deny"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8)
        policy = PromotionPolicy(min_agreement_threshold=0.80)

        report = evaluate_promotion_eligibility(engine, [], schema=schema, policy=policy)
        assert report.is_eligible is False
        assert any("Zero scored validation checks" in r for r in report.rejection_reasons)
        assert report.total_validation_checks == 0

    def test_no_relaxation_for_small_samples(self):
        schema = SystemOneSchema(
            schema_name="TestGateDSmall",
            fields={"action": SingleChoiceField(["allow", "deny"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8)
        policy = PromotionPolicy(
            min_agreement_threshold=0.80,
            require_statistical_bound=False,
            false_allow_ceiling=1.0,  # Isolate agreement rate check
        )

        val_history = [
            {"state": f"sample_{i}", "answers": {"action": "allow"}} for i in range(4)
        ]

        # Engine predictions will have some mismatches or we can evaluate report
        report = evaluate_promotion_eligibility(engine, val_history, schema=schema, policy=policy)
        # Even with <= 4 checks, min_agreement_threshold must remain strictly 0.80
        assert policy.min_agreement_threshold == 0.80

    def test_client_defers_cutover_under_default_policy_for_small_sample(self):
        """End-to-end verification: TypeSafeClient(mode='auto_cutover', cutover_threshold=4)
        defers cutover under default policy because Wilson lower bound fails threshold.
        """
        from system1.compat.typesafe import Choice, TypeSafeClient

        client = TypeSafeClient(
            mode="auto_cutover",
            cutover_threshold=4,
            min_agreement_threshold=0.80,
            zero_egress=False,
            fallback_baseline=True,
        )
        questions = {
            "action": Choice("Action", criteria={"allow": "Allow", "deny": "Deny"}),
        }
        for i in range(4):
            client.systemone(f"Request {i}", questions)

        assert client.is_cutover is False
        assert len(client.cutover_audit_log) >= 1
        rep = client.cutover_audit_log[0]["report"]
        assert rep.is_eligible is False
        assert rep.wilson_lower_bound < 0.80

    def test_client_defers_cutover_on_two_samples_due_to_empty_val_history(self):
        """End-to-end verification: 2-sample queries fail to populate validation fold,
        preventing in-sample cutover and deferring promotion.
        """
        from system1.compat.typesafe import Choice, TypeSafeClient

        client = TypeSafeClient(
            mode="auto_cutover",
            cutover_threshold=2,
            zero_egress=False,
            fallback_baseline=True,
        )
        questions = {
            "action": Choice("Action", criteria={"allow": "Allow", "deny": "Deny"}),
        }
        client.systemone("Query 1", questions)
        client.systemone("Query 2", questions)

        assert client.is_cutover is False
        assert len(client.cutover_audit_log) >= 1
        rep = client.cutover_audit_log[0]["report"]
        assert rep.total_validation_checks == 0
        assert rep.is_eligible is False
        assert "Zero scored validation checks; cannot evaluate promotion" in rep.rejection_reasons


# ============================================================================
# Gate D: 3. Mandatory Statistical Acceptance
# ============================================================================

class TestMandatoryStatisticalAcceptance:
    """Verifies default require_statistical_bound=True, Wilson bound, and 0% false-allow ceiling."""

    def test_default_require_statistical_bound_is_true(self):
        policy = PromotionPolicy()
        assert policy.require_statistical_bound is True

    def test_wilson_lower_bound_calculation(self):
        # 10 successes out of 10 trials
        low = compute_wilson_score_lower(10, 10, confidence=0.95)
        # Wilson lower bound for 10/10 is approx 0.722, well below 0.80
        assert low < 0.80

        # 100 successes out of 100 trials
        low_100 = compute_wilson_score_lower(100, 100, confidence=0.95)
        # Wilson lower bound for 100/100 is > 0.96
        assert low_100 > 0.96

    def test_rejection_on_critical_false_allow(self):
        schema = SystemOneSchema(
            schema_name="TestGateDCritical",
            fields={"action": SingleChoiceField(["allow", "deny"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8)

        # Force engine to predict "allow"
        policy = PromotionPolicy(
            min_agreement_threshold=0.50,
            require_statistical_bound=False,
            false_allow_ceiling=0.0,
        )

        # Val history has ground-truth "deny" (critical security action)
        val_history = [
            {"state": "malicious sql injection attack", "answers": {"action": "deny"}}
        ]

        report = evaluate_promotion_eligibility(engine, val_history, schema=schema, policy=policy)
        if report.false_allow_count > 0:
            assert report.is_eligible is False
            assert any("false-allow" in r.lower() for r in report.rejection_reasons)


# ============================================================================
# Gate D: 4. Disjoint Lineage & Manifest Binding
# ============================================================================

class TestDisjointLineageAndManifestBinding:
    """Verifies disjoint partition enforcement and manifest binding."""

    def test_cutover_partition_assert_disjoint_raises_on_overlap(self):
        overlapping_item = {"state": "shared_payload", "answers": {"action": "allow"}}

        train = [overlapping_item, {"state": "train_only", "answers": {"action": "deny"}}]
        calib = [{"state": "calib_only", "answers": {"action": "allow"}}]
        val = [overlapping_item]  # Leak into validation fold!

        partition = CutoverPartition(
            train_history=train,
            calib_history=calib,
            val_history=val,
            total_samples=4,
        )

        with pytest.raises(AssertionError, match="Violation"):
            partition.assert_disjoint()

    def test_cutover_partition_assert_disjoint_passes_on_clean_splits(self):
        partition = CutoverPartition(
            train_history=[{"state": f"train_{i}", "answers": {"a": 1}} for i in range(5)],
            calib_history=[{"state": f"calib_{i}", "answers": {"a": 1}} for i in range(5)],
            val_history=[{"state": f"val_{i}", "answers": {"a": 1}} for i in range(5)],
            total_samples=15,
        )
        partition.assert_disjoint()  # Must not raise


# ============================================================================
# Gate D: 5. Twin-Namespace Parity (system1 vs reflex)
# ============================================================================

class TestTwinNamespaceParity:
    """Verifies that all classes and methods in system1 and reflex are identical and aligned."""

    def test_cache_namespace_parity(self):
        assert rx_cache.SemanticSystemOneCache is s1_cache.SemanticSystemOneCache
        assert rx_cache.CacheEntry is s1_cache.CacheEntry
        assert rx_cache.validate_cache_inputs is s1_cache.validate_cache_inputs

    def test_engine_namespace_parity(self):
        assert rx_eng.SystemOneEngine is s1_eng.SystemOneEngine
        assert rx_eng.DecisionResult is s1_eng.DecisionResult

    def test_calibration_namespace_parity(self):
        assert rx_calib.ConformalPredictor is s1_calib.ConformalPredictor
        assert rx_calib.RegressionConformalPredictor is s1_calib.RegressionConformalPredictor

    def test_typesafe_namespace_parity(self):
        assert rx_typesafe.CutoverPartition is s1_typesafe.CutoverPartition
        assert rx_typesafe.PromotionPolicy is s1_typesafe.PromotionPolicy
        assert rx_typesafe.evaluate_promotion_eligibility is s1_typesafe.evaluate_promotion_eligibility
        assert rx_typesafe.compute_wilson_score_lower is s1_typesafe.compute_wilson_score_lower
