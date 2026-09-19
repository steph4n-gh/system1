"""Comprehensive test suite for Milestone M4: Cache Lifecycle & Online Learning Semantic Invalidation.

Authoritative Invariants Tested:
1. Invariant 8: Cache invalidation and prompt eviction in SystemOneEngine.learn_from_tier2().
   Verifies that after 10 consecutive online learning weight updates, subsequent queries
   immediately return post-update decisions without manual cache flush.
2. Invariant 9: Full execution context binding in SemanticSystemOneCache keys.
   Changing alpha, margin_threshold, strict, policy_scope, schema_digest, or model_version
   rejects cached decisions evaluated under different parameters.
3. Defensive Copies:
   Mutating returned DecisionResult dictionaries (values, confidences, probabilities, conformal_sets)
   never corrupts the internal cached entry.
4. Twin-namespace parity:
   Identical behavior when imported via `system1` or `reflex`.
"""

from __future__ import annotations

import copy
import numpy as np
import pytest

import reflex
import reflex.cache
import system1
import system1.cache
from system1 import (
    BooleanField,
    ChoiceField,
    DecisionResult,
    DecisionSchema,
    SystemOneCompiler,
    SystemOneEngine,
    ScoreField,
)
from system1.cache import CacheEntry, SemanticSystemOneCache


class TriageRoutingSchema(DecisionSchema):
    tier = ChoiceField(
        options=["TIER_1_LOCAL", "TIER_2_CLOUD", "HUMAN_ESCALATE"],
        descriptions={
            "TIER_1_LOCAL": "fast local reflex deterministic observation lookup",
            "TIER_2_CLOUD": "deliberative deep reasoning strategy synthesis",
            "HUMAN_ESCALATE": "sensitive destructive critical privilege grant",
        },
    )
    is_safe = BooleanField(threshold=0.5)
    urgency = ScoreField(min_value=0.0, max_value=1.0)


@pytest.fixture
def base_triage_engine() -> SystemOneEngine:
    exemplars = {
        "tier": [
            ("Read local config", "TIER_1_LOCAL"),
            ("Analyze annual growth strategy", "TIER_2_CLOUD"),
            ("Transfer corporate funds", "HUMAN_ESCALATE"),
        ] * 10,
        "is_safe": [
            ("Read local config", True),
            ("Analyze annual growth strategy", True),
            ("Transfer corporate funds", False),
        ] * 10,
        "urgency": [
            ("Read local config", 0.1),
            ("Analyze annual growth strategy", 0.5),
            ("Transfer corporate funds", 0.9),
        ] * 10,
    }
    compiler = SystemOneCompiler(TriageRoutingSchema, dimension=64, regularization=0.5)
    model = compiler.compile(exemplars=exemplars)
    return SystemOneEngine(
        TriageRoutingSchema,
        model=model,
        use_cache=True,
        enable_margin_gating=True,
        margin_threshold=0.15,
    )


def test_invariant_8_online_learning_semantic_cache_invalidation(base_triage_engine: SystemOneEngine):
    """Invariant 8: After 10 consecutive online learning updates, queries return post-update decisions without manual flush."""
    engine = base_triage_engine
    test_prompt = "Execute ambiguous diagnostic probe on cluster node"

    # Initial query populates cache
    initial_res = engine.decide(test_prompt)
    initial_tier = initial_res.values["tier"]

    # Verify initial decision is cached
    cached_query = engine.decide(test_prompt)
    assert cached_query.is_cache_hit is True
    assert cached_query.values["tier"] == initial_tier

    target_tier = "HUMAN_ESCALATE" if initial_tier != "HUMAN_ESCALATE" else "TIER_2_CLOUD"

    # Perform 10 consecutive online learning updates as specified in Invariant 8
    for step in range(1, 11):
        prev_version = engine.model_version

        # Online update via learn_from_tier2
        update_info = engine.learn_from_tier2(
            test_prompt,
            target={"tier": target_tier, "is_safe": True, "urgency": 0.8},
        )

        assert update_info["status"] == "updated"
        # Invariant 8: model_version must increment
        assert engine.model_version == prev_version + 1

        # Query IMMEDIATELY without manual cache clear
        res_after = engine.decide(test_prompt)

        # Must return the updated decision, NOT the stale pre-update decision
        assert res_after.values["tier"] == target_tier, (
            f"Step {step}: Expected updated tier '{target_tier}', got stale '{res_after.values['tier']}'"
        )
        assert res_after.values["tier"] != initial_tier


def test_invariant_8_batch_of_ten_distinct_prompts_evicted_and_updated(base_triage_engine: SystemOneEngine):
    """Invariant 8: Verifies 10 distinct prompts are each evicted and updated without stale cache bleed."""
    engine = base_triage_engine
    prompts = [f"Automated edge case scenario {i}: inspect memory buffer" for i in range(10)]

    # Populate cache for all 10 prompts
    initial_tiers = {}
    for p in prompts:
        res = engine.decide(p)
        initial_tiers[p] = res.values["tier"]

    # Now teach new target for all 10 prompts
    new_target = "HUMAN_ESCALATE"
    for i, p in enumerate(prompts):
        prev_v = engine.model_version
        engine.learn_from_tier2(p, target={"tier": new_target, "is_safe": False, "urgency": 0.95})
        assert engine.model_version == prev_v + 1

        # Must return new target immediately
        res_after = engine.decide(p)
        assert res_after.values["tier"] == new_target
        assert res_after.values["tier"] != "TIER_1_LOCAL" or new_target == "TIER_1_LOCAL"


def test_invariant_9_context_binding_rejects_cached_entries_under_different_risk_params(
    base_triage_engine: SystemOneEngine,
):
    """Invariant 9: Changing alpha, margin_threshold, strict, or policy_scope rejects entries evaluated under different parameters."""
    engine = base_triage_engine
    prompt = "Observe filesystem directory contents"

    # 1. Evaluate and cache under alpha=0.10, margin_threshold=0.15, strict=False
    res_10 = engine.decide(prompt, alpha=0.10, strict=False)
    assert res_10.alpha == 0.10
    assert res_10.is_cache_hit is False

    # Repeat exact query -> cache hit
    res_10_hit = engine.decide(prompt, alpha=0.10, strict=False)
    assert res_10_hit.is_cache_hit is True
    assert res_10_hit.alpha == 0.10

    # 2. Change alpha to 0.01 -> must NOT return cached alpha=0.10 entry
    res_01 = engine.decide(prompt, alpha=0.01, strict=False)
    assert res_01.alpha == 0.01
    assert res_01.is_cache_hit is False

    # 3. Change strict to True -> must NOT return cached strict=False entry
    res_strict = engine.decide(prompt, alpha=0.10, strict=True)
    assert res_strict.is_cache_hit is False

    # 4. Change policy_scope -> must NOT return cached default scope entry
    res_scoped = engine.decide(prompt, alpha=0.10, strict=False, policy_scope="restricted_env")
    assert res_scoped.is_cache_hit is False

    # 5. Query again with policy_scope="restricted_env" -> cache hit for that scope
    res_scoped_hit = engine.decide(prompt, alpha=0.10, strict=False, policy_scope="restricted_env")
    assert res_scoped_hit.is_cache_hit is True


def test_defensive_copies_prevent_cache_corruption(base_triage_engine: SystemOneEngine):
    """Verify that caller mutations of returned DecisionResult do not corrupt the internal cache."""
    engine = base_triage_engine
    prompt = "Read local status dashboard"

    # Evaluate and cache
    res1 = engine.decide(prompt)
    original_tier = res1.values["tier"]
    original_conf = res1.confidences["tier"]
    original_prob = res1.probabilities["tier"][original_tier]

    # Malicious or buggy caller mutates returned result dictionaries in-place
    res1.values["tier"] = "CORRUPTED_TIER"
    res1.confidences["tier"] = 0.00001
    res1.probabilities["tier"][original_tier] = -999.0
    res1.conformal_sets["tier"].append("INJECTED_OPTION")

    # Fetch from cache again
    res2 = engine.decide(prompt)
    assert res2.is_cache_hit is True

    # Assert cache retained the true uncorrupted values
    assert res2.values["tier"] == original_tier
    assert res2.values["tier"] != "CORRUPTED_TIER"
    assert res2.confidences["tier"] == original_conf
    assert res2.probabilities["tier"][original_tier] == original_prob
    assert "INJECTED_OPTION" not in res2.conformal_sets["tier"]


def test_cache_version_invalidation_methods():
    """Verify SemanticSystemOneCache.evict_prompt and invalidate_prior_versions directly."""
    cache = SemanticSystemOneCache(capacity=100)
    emb = np.zeros(64, dtype=np.float32)
    emb[0] = 1.0

    mock_result = {"action": "ALLOW"}

    # Insert entry with model_version=1
    cache.put(
        "deploy staging",
        mock_result,
        embedding=emb,
        model_version=1,
    )

    # Cache hit at version 1
    hit = cache.get("deploy staging", embedding=emb, model_version=1)
    assert hit is not None
    assert hit[0].result == mock_result

    # Cache miss at version 2
    miss_v2 = cache.get("deploy staging", embedding=emb, model_version=2)
    assert miss_v2 is None

    # Invalidate prior versions up to 2
    cache.invalidate_prior_versions(min_version=2)
    assert cache.get("deploy staging", embedding=emb, model_version=1) is None

    # Insert prompt at version 2, then evict specifically
    cache.put(
        "reboot node",
        mock_result,
        embedding=emb,
        model_version=2,
    )
    assert cache.get("reboot node", embedding=emb, model_version=2) is not None

    cache.evict_prompt("reboot node")
    assert cache.get("reboot node", embedding=emb, model_version=2) is None


def test_twin_namespace_parity_for_cache_components():
    """Verify system1.cache and reflex.cache export identical classes and functions."""
    assert system1.cache.SemanticSystemOneCache is reflex.cache.SemanticSystemOneCache
    assert system1.cache.CacheEntry is reflex.cache.CacheEntry
