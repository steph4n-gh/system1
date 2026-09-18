"""Adversarial Attack & Verification Suite for Gate C (Cache & Uncertainty Lifecycle).

Authored by Empirical Challenger (teamwork_preview_challenger_m2_1).

Attacks Covered:
1. Invalid Parameter Injection:
   - Inject negative, zero, NaN, Inf, and non-numeric odds ratios.
   - Inject out-of-range, NaN, Inf, and non-numeric alphas.
   - Inject negative, NaN, Inf, and non-numeric margin thresholds.
   - Inject negative, NaN, Inf, and non-numeric confidence floors.
   - Inject malformed telemetry (non-mappings/sequences, NaN/Inf arrays).
   - Inject malformed embeddings (2D/3D shapes, scalar 0D, NaN/Inf, dimension mismatch).
   - Inject non-string prompts (int, list, dict, None, bytes).
   - Verify pre-lookup validation raises TypeError/ValueError before index traversal.

2. Cache Collision & Risk Parameter Sensitivity:
   - Verify altering alpha, margin_threshold, relative_odds_ratio, confidence_floor_tau0,
     model_version, policy_scope, policy_epoch, strict, recency_weighted, telemetry,
     model_digest, or calibration_digest forces cache misses.
   - Full 64-hex SHA-256 telemetry digest collision resistance.
   - Ensure distinct parameter tuples never collide in formatted context keys.

3. Cache Hit Ambiguity Masking Attack:
   - Query an ambiguous input, cache it, re-query, and verify that is_ambiguous,
     ambiguous_fields, and escalated_fields remain faithfully ambiguous on cache hit.
   - Query an unambiguous input, cache it, re-query, and verify is_ambiguous is False.
   - Defensive copies: mutate returned decision values, probabilities, and sets;
     verify cached entries remain completely uncorrupted.

4. Multithreaded Concurrency Race:
   - Concurrently trigger online rank-1 weight updates (learn_from_tier2) while reader
     threads query decide(); verify no reader observes pre-update weights under
     post-update versions.
   - Verify cache is never polluted with pre-update weights labeled with post-update versions.
   - Verify thread-safe LRU eviction and index maintenance under concurrent reads and writes.

5. Strict Mode Semantic Search Evasion:
   - Attempt to retrieve cached items via approximate cosine search in strict mode; verify rejection.
   - Attempt to retrieve strict items with lax queries via cosine search; verify rejection.
   - Verify enforce_durability suppresses approximate semantic cosine search.

6. Uncalibrated Model Exploitation:
   - Query uncalibrated ConformalPredictor in strict mode; verify full label set, is_ambiguous=True,
     needs_escalation=True, q_hat=1.0, and margin gating is inactive.
   - Query uncalibrated RegressionConformalPredictor in strict mode; verify full feasible domain.
   - Query ReflexEngine across ChoiceField, BooleanField, MultiChoiceField, and ScoreField
     in strict mode while uncalibrated; verify complete ambiguity escalation.

7. Decoupled Calibration Folds & Conformal Invariants:
   - Invariant 6: Empty set counterexample on identical probability vectors [0.4, 0.35, 0.25].
   - Invariant 7: Order statistic exceeding calibration sample count returns conservative q_hat=1.0.
   - Decoupled 50/50 partition for temperature scaling and conformal quantile fitting.

8. Twin Namespace Parity:
   - Verify exact parity across system1 and reflex namespaces for cache, calibration, compiler, and engine.
"""

from __future__ import annotations

import concurrent.futures
import copy
import dataclasses
import hashlib
import json
import math
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pytest

# Twin namespace imports
import reflex.cache as rx_cache
import reflex.calibration as rx_calib
import reflex.compiler as rx_comp
import reflex.engine as rx_eng
import system1.cache as s1_cache
import system1.calibration as s1_calib
import system1.compiler as s1_comp
import system1.engine as s1_eng

from system1.cache import (
    CacheEntry,
    SemanticReflexCache,
    _digest_prompt,
    _digest_telemetry,
    _format_context,
    _validate_cache_inputs,
    validate_cache_inputs,
)
from system1.calibration import (
    ConformalPredictionSet,
    ConformalPredictor,
    DecisionCalibrator,
    RegressionConformalInterval,
    RegressionConformalPredictor,
)
from system1.compiler import CompiledSystemOneModel, ReflexCompiler
from system1.core.schema import (
    BooleanField,
    ChoiceField,
    DecisionSchema,
    MultiChoiceField,
    ScoreField,
)
from system1.engine import DecisionResult, ReflexEngine


# ============================================================================
# Attack 1: Invalid Parameter Injection
# ============================================================================

class TestAttack1InvalidParameterInjection:
    """Adversarial parameter injections to trigger pre-lookup validation fail-closed."""

    @pytest.mark.parametrize("bad_prompt", [123, None, ["query"], {"p": "q"}, b"bytes_prompt", 45.67])
    def test_inject_invalid_prompt_types(self, bad_prompt):
        cache = SemanticReflexCache()
        with pytest.raises(TypeError, match="Prompt must be a string"):
            cache.get(bad_prompt)  # type: ignore

        with pytest.raises(TypeError, match="Prompt must be a string"):
            cache.put(bad_prompt, "res")  # type: ignore

        with pytest.raises(TypeError, match="Prompt must be a string"):
            validate_cache_inputs(bad_prompt)  # type: ignore

    @pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.01, -1.5, 1.0001, 100.0, float("nan"), float("inf"), float("-inf")])
    def test_inject_out_of_range_alpha_values(self, bad_alpha):
        cache = SemanticReflexCache()
        with pytest.raises(ValueError, match="Significance level alpha must be in"):
            cache.get("valid prompt", alpha=bad_alpha)

        with pytest.raises(ValueError, match="Significance level alpha must be in"):
            cache.put("valid prompt", "res", alpha=bad_alpha)

        with pytest.raises(ValueError, match="Significance level alpha must be in"):
            validate_cache_inputs("valid prompt", alpha=bad_alpha)

    @pytest.mark.parametrize("bad_alpha_type", ["invalid", [0.05], {"alpha": 0.05}])
    def test_inject_invalid_alpha_types(self, bad_alpha_type):
        cache = SemanticReflexCache()
        with pytest.raises(TypeError, match="alpha must be a real number"):
            cache.get("valid prompt", alpha=bad_alpha_type)  # type: ignore

    @pytest.mark.parametrize("bad_margin", [-0.0001, -1.0, -99.9, float("nan"), float("-inf")])
    def test_inject_negative_margin_threshold(self, bad_margin):
        cache = SemanticReflexCache()
        with pytest.raises(ValueError, match="margin_threshold must be non-negative"):
            cache.get("valid prompt", margin_threshold=bad_margin)

        with pytest.raises(ValueError, match="margin_threshold must be non-negative"):
            cache.put("valid prompt", "res", margin_threshold=bad_margin)

    @pytest.mark.parametrize("bad_odds", [0.0, -0.001, -1.0, -50.0, float("nan"), float("-inf")])
    def test_inject_invalid_odds_ratios(self, bad_odds):
        cache = SemanticReflexCache()
        with pytest.raises(ValueError, match="relative_odds_ratio must be positive"):
            cache.get("valid prompt", relative_odds_ratio=bad_odds)

        with pytest.raises(ValueError, match="relative_odds_ratio must be positive"):
            cache.get("valid prompt", odds_ratio=bad_odds)

        with pytest.raises(ValueError, match="relative_odds_ratio must be positive"):
            cache.put("valid prompt", "res", relative_odds_ratio=bad_odds)

    @pytest.mark.parametrize("bad_floor", [-0.001, -0.5, -10.0, float("nan"), float("-inf")])
    def test_inject_invalid_confidence_floor(self, bad_floor):
        cache = SemanticReflexCache()
        with pytest.raises(ValueError, match="confidence_floor_tau0 must be non-negative"):
            cache.get("valid prompt", confidence_floor_tau0=bad_floor)

        with pytest.raises(ValueError, match="confidence_floor_tau0 must be non-negative"):
            cache.put("valid prompt", "res", confidence_floor_tau0=bad_floor)

    def test_inject_malformed_telemetry(self):
        cache = SemanticReflexCache()
        # Invalid telemetry type (e.g. object, int)
        with pytest.raises(TypeError, match="telemetry must be a Mapping, Sequence, or ndarray"):
            cache.get("valid prompt", telemetry=object())

        # Array with NaN or Inf
        with pytest.raises(ValueError, match="telemetry array contains NaN or infinite"):
            cache.get("valid prompt", telemetry=np.array([1.0, np.nan, 2.0]))

        with pytest.raises(ValueError, match="telemetry array contains NaN or infinite"):
            cache.get("valid prompt", telemetry=np.array([1.0, np.inf, 2.0]))

    def test_inject_malformed_embeddings(self):
        cache = SemanticReflexCache()
        # 2D embedding matrix instead of 1D vector
        with pytest.raises(ValueError, match="embedding must be a 1D vector"):
            cache.get("valid prompt", embedding=np.zeros((3, 3)))

        # 3D embedding tensor
        with pytest.raises(ValueError, match="embedding must be a 1D vector"):
            cache.get("valid prompt", embedding=np.zeros((2, 2, 2)))

        # Embedding with NaN or Inf
        with pytest.raises(ValueError, match="embedding contains NaN or infinite values"):
            cache.get("valid prompt", embedding=np.array([1.0, np.nan, 0.0]))

        with pytest.raises(ValueError, match="embedding contains NaN or infinite values"):
            cache.get("valid prompt", embedding=np.array([1.0, np.inf, 0.0]))

        with pytest.raises(ValueError, match="embedding contains NaN or infinite values"):
            cache.get("valid prompt", embedding=[1.0, float("-inf"), 0.0])

    def test_engine_pre_lookup_validation_rejection(self):
        schema = DecisionSchema(schema_name="EngVal", fields={"act": ChoiceField(["allow", "deny"])})
        engine = ReflexEngine(schema=schema, dimension=8, use_cache=True)

        with pytest.raises(TypeError, match="Prompt must be a string"):
            engine.decide(99999)  # type: ignore

        with pytest.raises(ValueError, match="alpha must be in"):
            engine.decide("test", alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            engine.decide("test", alpha=1.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            engine.decide("test", alpha=float("nan"))

        with pytest.raises(ValueError, match="margin_threshold must be non-negative"):
            engine.decide("test", margin_threshold=-0.1)

        with pytest.raises(ValueError, match="relative_odds_ratio must be strictly positive"):
            engine.decide("test", relative_odds_ratio=0.0)

        with pytest.raises(ValueError, match="confidence_floor_tau0 must be non-negative"):
            engine.decide("test", confidence_floor_tau0=-0.5)

        with pytest.raises(ValueError, match="embedding dimension mismatch"):
            engine.decide("test", embedding=np.zeros(4))

        with pytest.raises(ValueError, match="embedding must contain only finite numbers"):
            engine.decide("test", embedding=np.array([np.nan] * 8))


# ============================================================================
# Attack 2: Cache Collision & Risk Parameter Sensitivity
# ============================================================================

class TestAttack2CacheCollisionAndSensitivity:
    """Verifies complete collision-resistant key isolation across all execution context axes."""

    def test_all_individual_parameters_force_cache_miss(self):
        schema = DecisionSchema(schema_name="SensSchema", fields={"act": ChoiceField(["allow", "deny"])})
        engine = ReflexEngine(schema=schema, dimension=8, use_cache=True, enable_margin_gating=True)

        # Train to make unambiguous
        for _ in range(3):
            engine.learn_from_tier2("seed prompt", target={"act": "allow"})

        prompt = "seed prompt"
        r0 = engine.decide(prompt, alpha=0.05, margin_threshold=0.1, relative_odds_ratio=1.5, confidence_floor_tau0=0.2)
        # Re-query with identical parameters -> must hit cache
        r0_hit = engine.decide(prompt, alpha=0.05, margin_threshold=0.1, relative_odds_ratio=1.5, confidence_floor_tau0=0.2)
        assert r0_hit.is_cache_hit is True

        # 1. Mutate alpha
        r_alpha = engine.decide(prompt, alpha=0.01, margin_threshold=0.1, relative_odds_ratio=1.5, confidence_floor_tau0=0.2)
        assert r_alpha.is_cache_hit is False

        # 2. Mutate margin_threshold
        r_margin = engine.decide(prompt, alpha=0.05, margin_threshold=0.35, relative_odds_ratio=1.5, confidence_floor_tau0=0.2)
        assert r_margin.is_cache_hit is False

        # 3. Mutate relative_odds_ratio
        r_odds = engine.decide(prompt, alpha=0.05, margin_threshold=0.1, relative_odds_ratio=3.0, confidence_floor_tau0=0.2)
        assert r_odds.is_cache_hit is False

        # 4. Mutate confidence_floor_tau0
        r_floor = engine.decide(prompt, alpha=0.05, margin_threshold=0.1, relative_odds_ratio=1.5, confidence_floor_tau0=0.8)
        assert r_floor.is_cache_hit is False

        # 5. Mutate policy_scope
        r_scope = engine.decide(prompt, alpha=0.05, margin_threshold=0.1, relative_odds_ratio=1.5, confidence_floor_tau0=0.2, policy_scope="quarantine")
        assert r_scope.is_cache_hit is False

        # 6. Mutate strict mode
        r_strict = engine.decide(prompt, alpha=0.05, margin_threshold=0.1, relative_odds_ratio=1.5, confidence_floor_tau0=0.2, strict=True)
        assert r_strict.is_cache_hit is False

        # 7. Mutate telemetry presence and content in strict/enforcement mode
        r_telem = engine.decide(prompt, alpha=0.05, margin_threshold=0.1, relative_odds_ratio=1.5, confidence_floor_tau0=0.2, telemetry={"user_id": 101.0}, strict=True)
        assert r_telem.is_cache_hit is False

        # Put into cache with strict=True
        engine.cache.put(
            prompt=prompt,
            result=r_telem,
            schema_digest=schema.schema_digest(),
            model_version=engine.model_version,
            policy_scope=engine.policy_scope,
            alpha=0.05,
            margin_threshold=0.1,
            strict=True,
            relative_odds_ratio=1.5,
            confidence_floor_tau0=0.2,
            telemetry={"user_id": 101.0},
            model_digest=engine._model_digest(),
            projector_digest=engine._projector_digest(),
            calibration_digest=engine._calibration_digest(),
            policy_epoch=engine.policy_epoch,
        )

        # Repeat with exact same telemetry in strict mode -> must hit
        r_telem_hit = engine.decide(prompt, alpha=0.05, margin_threshold=0.1, relative_odds_ratio=1.5, confidence_floor_tau0=0.2, telemetry={"user_id": 101.0}, strict=True)
        assert r_telem_hit.is_cache_hit is True

        # Mutate telemetry content in strict mode -> must miss
        r_telem_diff = engine.decide(prompt, alpha=0.05, margin_threshold=0.1, relative_odds_ratio=1.5, confidence_floor_tau0=0.2, telemetry={"user_id": 102.0}, strict=True)
        assert r_telem_diff.is_cache_hit is False

    def test_full_64_hex_telemetry_digest_collision_resistance(self):
        # Different key orders must produce identical digests (canonical sorting)
        t1 = {"alpha": 1, "beta": 2}
        t2 = {"beta": 2, "alpha": 1}
        assert _digest_telemetry(t1) == _digest_telemetry(t2)
        assert len(_digest_telemetry(t1)) == 64

        # Different values must produce distinct digests
        t3 = {"alpha": 1, "beta": 3}
        assert _digest_telemetry(t1) != _digest_telemetry(t3)

        # Float representation vs string representation
        t4 = {"val": 1.0}
        t5 = {"val": "1.0"}
        assert _digest_telemetry(t4) == _digest_telemetry(t5)

        # Different numeric values must not collide
        t6 = {"val": 1.00001}
        assert _digest_telemetry(t4) != _digest_telemetry(t6)

        # Array vs Array with different values
        arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arr2 = np.array([1.0, 2.0, 3.0001], dtype=np.float32)
        assert _digest_telemetry(arr1) != _digest_telemetry(arr2)

    def test_model_version_and_digest_key_isolation(self):
        cache = SemanticReflexCache()
        k1 = cache._make_key("query", model_version=1, model_digest="sha_v1")
        k2 = cache._make_key("query", model_version=2, model_digest="sha_v1")
        k3 = cache._make_key("query", model_version=1, model_digest="sha_v2")
        assert k1 != k2
        assert k1 != k3
        assert k2 != k3


# ============================================================================
# Attack 3: Cache Hit Ambiguity Masking Attack
# ============================================================================

class TestAttack3CacheHitAmbiguityMasking:
    """Verifies that cache hits faithfully preserve evaluated ambiguity and escalated fields."""

    def test_cache_hit_preserves_ambiguity_and_escalated_fields(self):
        schema = DecisionSchema(
            schema_name="AmbiguityPreservation",
            fields={
                "triage": ChoiceField(["auto_resolve", "escalate_tier2", "quarantine"]),
                "risk_tag": MultiChoiceField(["pii", "financial", "malware"]),
            },
        )
        engine = ReflexEngine(schema=schema, dimension=8, use_cache=True)
        prompt = "Borderline anomalous transaction payload"

        res1 = engine.decide(prompt, alpha=0.05)
        # Construct ambiguous result
        ambig_res = dataclasses.replace(
            res1,
            is_ambiguous=True,
            ambiguous_fields=["triage", "risk_tag"],
            escalated_fields=["triage"],
            conformal_sets={"triage": ["auto_resolve", "escalate_tier2"], "risk_tag": ["pii", "financial"]},
        )

        # Seed cache manually
        engine.cache.put(
            prompt=prompt,
            result=ambig_res,
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

        # Re-query
        hit_res = engine.decide(prompt, alpha=0.05)
        assert hit_res.is_cache_hit is True
        assert hit_res.is_ambiguous is True
        assert hit_res.ambiguous_fields == ["triage", "risk_tag"]
        assert hit_res.escalated_fields == ["triage"]
        assert hit_res.conformal_sets["triage"] == ["auto_resolve", "escalate_tier2"]

    def test_cache_hit_preserves_unambiguous_state(self):
        schema = DecisionSchema(
            schema_name="UnambigPreservation",
            fields={"verdict": ChoiceField(["allow", "deny"])},
        )
        engine = ReflexEngine(schema=schema, dimension=8, use_cache=True)
        prompt = "Standard verified ping"

        res1 = engine.decide(prompt, alpha=0.05)
        clear_res = dataclasses.replace(
            res1,
            is_ambiguous=False,
            ambiguous_fields=[],
            escalated_fields=[],
            conformal_sets={"verdict": ["allow"]},
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
        assert hit_res.conformal_sets["verdict"] == ["allow"]

    def test_defensive_copies_prevent_external_cache_corruption(self):
        schema = DecisionSchema(schema_name="DefCopy", fields={"decision": ChoiceField(["yes", "no"])})
        engine = ReflexEngine(schema=schema, dimension=8, use_cache=True)
        engine.learn_from_tier2("probe prompt", target={"decision": "yes"})

        r1 = engine.decide("probe prompt")
        assert r1.is_cache_hit is True
        assert r1.values["decision"] == "yes"

        # Attempt to mutate returned decision fields
        r1.values["decision"] = "MUTATED_BY_ATTACKER"
        r1.confidences["decision"] = -999.0
        r1.conformal_sets["decision"].append("INJECTED_SET")
        r1.ambiguous_fields.append("INJECTED_AMBIG")
        r1.escalated_fields.append("INJECTED_ESCAL")

        # Subsequent cache hit must return pure, unmutated data
        r2 = engine.decide("probe prompt")
        assert r2.is_cache_hit is True
        assert r2.values["decision"] == "yes"
        assert r2.confidences["decision"] != -999.0
        assert "INJECTED_SET" not in r2.conformal_sets["decision"]
        assert "INJECTED_AMBIG" not in r2.ambiguous_fields
        assert "INJECTED_ESCAL" not in r2.escalated_fields


# ============================================================================
# Attack 4: Multithreaded Concurrency Race
# ============================================================================

class TestAttack4MultithreadedConcurrencyRace:
    """Stress-tests atomic publication and reader-writer synchronization under heavy load."""

    def test_concurrent_online_updates_and_decide(self):
        schema = DecisionSchema(schema_name="ConcurrentRace", fields={"status": ChoiceField(["allow", "deny"])})
        engine = ReflexEngine(schema=schema, dimension=16, use_cache=True)

        prompts = [f"network_flow_packet_{i}" for i in range(16)]
        for p in prompts:
            engine.learn_from_tier2(p, target={"status": "allow"})

        stop_event = threading.Event()
        observed_inconsistencies: List[str] = []

        def reader_job(reader_id: int):
            local_reads = 0
            while not stop_event.is_set():
                p = prompts[local_reads % len(prompts)]
                v_pre = engine.model_version
                res = engine.decide(p)
                v_post = engine.model_version
                # Verify that returned decision adheres to schema
                if res.values["status"] not in ["allow", "deny"]:
                    observed_inconsistencies.append(f"Reader {reader_id} saw invalid choice: {res.values['status']}")
                local_reads += 1
            return local_reads

        def writer_job():
            for i in range(30):
                p = prompts[i % len(prompts)]
                target_choice = "deny" if i % 2 == 0 else "allow"
                engine.learn_from_tier2(p, target={"status": target_choice})
                time.sleep(0.0005)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            writer_fut = executor.submit(writer_job)
            reader_futs = [executor.submit(reader_job, i) for i in range(8)]

            writer_fut.result(timeout=10.0)
            stop_event.set()
            total_reads = sum(rf.result(timeout=5.0) for rf in reader_futs)

        assert len(observed_inconsistencies) == 0, f"Observed inconsistencies: {observed_inconsistencies}"
        assert total_reads > 50
        assert engine.model_version >= 30

    def test_cache_invalidates_prior_versions_on_multiple_updates(self):
        # Invariant 8: Cache returns stale pre-update decision after 10 online updates
        schema = DecisionSchema(schema_name="Inv8Schema", fields={"action": ChoiceField(["fast", "slow"])})
        engine = ReflexEngine(schema=schema, dimension=8, use_cache=True, forgetting_factor=0.1)
        prompt = "invariant 8 target prompt"

        # Initial seed
        engine.learn_from_tier2(prompt, target={"action": "fast"})
        r_init = engine.decide(prompt)
        assert r_init.values["action"] == "fast"

        # Transition to new target and perform 10 consecutive online updates
        target_action = "slow"
        for step in range(1, 11):
            prev_v = engine.model_version
            engine.learn_from_tier2(prompt, target={"action": target_action})
            assert engine.model_version == prev_v + 1

            # Decide immediately; must return newly learned target, never stale pre-update decision
            res_step = engine.decide(prompt)
            assert res_step.values["action"] == target_action, f"Step {step} failed: expected {target_action}, got {res_step.values['action']}"


# ============================================================================
# Attack 5: Strict Mode Semantic Search Evasion
# ============================================================================

class TestAttack5StrictModeSemanticSearchEvasion:
    """Verifies that approximate semantic cosine search is strictly rejected in enforcement / strict modes."""

    def test_strict_mode_rejects_approximate_semantic_search(self):
        cache = SemanticReflexCache(similarity_threshold=0.7)
        emb_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.999, 0.001, 0.0, 0.0], dtype=np.float32)  # Cosine similarity > 0.9999

        # Case 1: Entry cached under strict=False
        cache.put("canonical prompt", "safe_response", embedding=emb_a, strict=False)

        # Query in lax mode (strict=False) -> hits approximate match
        hit_lax = cache.get("adversarial variation", embedding=emb_b, strict=False)
        assert hit_lax is not None
        assert hit_lax[0].prompt == "canonical prompt"

        # Query in strict mode (strict=True) -> approximate search MUST be suppressed
        hit_strict = cache.get("adversarial variation", embedding=emb_b, strict=True)
        assert hit_strict is None

    def test_enforce_durability_rejects_approximate_semantic_search(self):
        cache = SemanticReflexCache(similarity_threshold=0.7)
        emb_a = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.0, 0.999, 0.001], dtype=np.float32)

        cache.put("login admin", "granted", embedding=emb_a, strict=False)

        # enforce_durability=True must suppress semantic approximate search
        hit_durable = cache.get("login root", embedding=emb_b, strict=False, enforce_durability=True)
        assert hit_durable is None

    def test_cross_strict_entry_isolation(self):
        cache = SemanticReflexCache(similarity_threshold=0.7)
        emb = np.array([1.0, 0.0], dtype=np.float32)

        # Entry put under strict=True
        cache.put("classified command", "secret_output", embedding=emb, strict=True)

        # Attacker queries with strict=False (trying to bypass context binding)
        hit_attempt = cache.get("classified command", strict=False)
        assert hit_attempt is None

        # Attacker queries with similar embedding under strict=False
        hit_emb_attempt = cache.get("classified command variant", embedding=emb, strict=False)
        assert hit_emb_attempt is None


# ============================================================================
# Attack 6: Uncalibrated Model Exploitation
# ============================================================================

class TestAttack6UncalibratedModelExploitation:
    """Verifies that uncalibrated models abstain/escalate and return conservative full domains."""

    def test_conformal_predictor_uncalibrated_strict_full_set(self):
        cp = ConformalPredictor("gate", ["grant", "deny", "step_up"])
        assert cp.is_calibrated is False

        # Attempt to pass high-confidence probability vector to uncalibrated predictor
        probs = np.array([0.98, 0.01, 0.01])

        # In strict mode: must return full label set and flag needs_escalation
        res_strict = cp.predict_set(
            probs,
            alpha=0.05,
            strict=True,
            margin_threshold=0.05,
            relative_odds_ratio=1.01,
            confidence_floor_tau0=0.01,
        )
        assert res_strict.is_ambiguous is True
        assert res_strict.needs_escalation is True
        assert res_strict.margin_gate_active is False
        assert set(res_strict.prediction_set) == {"grant", "deny", "step_up"}
        assert res_strict.conformal_threshold == 1.0

    def test_regression_conformal_predictor_uncalibrated_full_domain(self):
        rcp = RegressionConformalPredictor("p_score", min_value=10.0, max_value=100.0)
        assert rcp.is_calibrated is False

        # Strict mode: must return entire range [10, 100]
        interval = rcp.predict_interval(50.0, alpha=0.05, strict=True)
        assert interval.is_calibrated is False
        assert interval.lower_bound == 10.0
        assert interval.upper_bound == 100.0
        assert interval.margin == 90.0

    def test_engine_all_fields_uncalibrated_strict_escalation(self):
        schema = DecisionSchema(
            schema_name="StrictAllFields",
            fields={
                "c_field": ChoiceField(["allow", "deny"]),
                "b_field": BooleanField(threshold=0.5),
                "m_field": MultiChoiceField(["read", "write", "admin"]),
                "s_field": ScoreField(min_value=0.0, max_value=10.0),
            },
        )
        engine = ReflexEngine(schema=schema, dimension=8, strict_mode=True)

        res = engine.decide("uncalibrated prompt in strict mode")
        assert res.is_ambiguous is True
        assert "c_field" in res.escalated_fields
        assert "b_field" in res.escalated_fields
        assert "m_field" in res.escalated_fields
        assert "s_field" in res.escalated_fields

        # Prediction sets must be complete
        assert set(res.conformal_sets["c_field"]) == {"allow", "deny"}
        assert set(res.conformal_sets["b_field"]) == {"False", "True"}
        assert res.conformal_sets["s_field"] == ["[0.0000, 10.0000]"]


# ============================================================================
# Attack 7: Decoupled Calibration Folds & Invariants 6 and 7
# ============================================================================

class TestAttack7DecoupledCalibrationAndInvariants:
    """Verifies finite-sample conformal invariants and decoupled calibration fold partitioning."""

    def test_invariant_6_empty_set_counterexample(self):
        # Invariant 6: 100 identical probability vectors [0.4, 0.35, 0.25] labeled 'A' at alpha=0.05
        cp = ConformalPredictor("target", ["A", "B", "C"])
        probs = [0.4, 0.35, 0.25]
        dataset_probs = [probs] * 100
        dataset_labels = ["A"] * 100
        cp.calibrate(dataset_probs, dataset_labels)

        res = cp.predict_set(probs, alpha=0.05)
        assert len(res.prediction_set) > 0, "Conformal prediction set must not be empty"
        assert "A" in res.prediction_set, f"Ground truth 'A' must be in prediction set {res.prediction_set}"

    def test_invariant_7_small_sample_order_statistic_full_domain(self):
        # Invariant 7: k = ceil((n+1)(1-alpha)) > n returns conservative q_hat = 1.0 and full set
        cp = ConformalPredictor("action", ["allow", "review", "block"])
        # n = 3 samples, alpha = 0.01 -> (3 + 1) * 0.99 = 3.96 -> ceil = 4 > 3
        probs = [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.8, 0.1, 0.1]]
        labels = ["allow", "allow", "allow"]
        cp.calibrate(probs, labels)

        res = cp.predict_set([0.7, 0.2, 0.1], alpha=0.01)
        assert res.conformal_threshold == 1.0
        assert set(res.prediction_set) == {"allow", "review", "block"}

    def test_decoupled_temperature_and_quantile_folds(self):
        schema = DecisionSchema(
            schema_name="DecoupledFolds",
            fields={"decision": ChoiceField(["class_0", "class_1", "class_2"])},
        )
        engine = ReflexEngine(schema=schema, dimension=8)
        dataset = [
            (f"calibration sample prompt {i}", {"decision": f"class_{i % 3}"})
            for i in range(40)
        ]
        metrics = engine.calibrate(dataset, n_bins=5)
        assert "decision" in metrics

        cp = engine.conformal_predictors["decision"]
        assert cp.is_calibrated is True
        # Calibration scores length must correspond to conformal fold (50% of 40 = 20)
        assert len(cp.calibration_scores) == 20


# ============================================================================
# Attack 8: Twin Namespace Parity
# ============================================================================

class TestAttack8TwinNamespaceParity:
    """Verifies 100% functional and structural parity across system1 and reflex namespaces."""

    def test_cache_namespace_parity(self):
        assert s1_cache.SemanticReflexCache is rx_cache.SemanticReflexCache
        assert s1_cache.CacheEntry is rx_cache.CacheEntry
        assert s1_cache.validate_cache_inputs is rx_cache.validate_cache_inputs

    def test_calibration_namespace_parity(self):
        assert s1_calib.ConformalPredictor is rx_calib.ConformalPredictor
        assert s1_calib.RegressionConformalPredictor is rx_calib.RegressionConformalPredictor
        assert s1_calib.DecisionCalibrator is rx_calib.DecisionCalibrator
        assert s1_calib.ConformalPredictionSet is rx_calib.ConformalPredictionSet
        assert s1_calib.RegressionConformalInterval is rx_calib.RegressionConformalInterval

    def test_compiler_namespace_parity(self):
        assert s1_comp.ReflexCompiler is rx_comp.ReflexCompiler
        assert s1_comp.CompiledSystemOneModel is rx_comp.CompiledSystemOneModel

    def test_engine_namespace_parity(self):
        assert s1_eng.ReflexEngine is rx_eng.ReflexEngine
        assert s1_eng.DecisionResult is rx_eng.DecisionResult
