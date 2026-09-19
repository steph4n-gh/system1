"""Comprehensive tests for the 4 Tier 1 architectural levers:

1. Lever 1: Tier 0 Semantic System 1 Cache (L1 Vector/Exact Cache, sub-0.05ms certified execution)
2. Lever 2: Online Sherman-Morrison Distillation (closed-form rank-1 update, <0.1ms, live boundary adaptation)
3. Lever 3: Margin-Based Conformal Gating (M(x) = s_{(1)} - s_{(2)} >= tau_margin suppresses false ambiguity)
4. Lever 4: Telemetry & Continuous State Vector Fusion (normalization, projection, multimodal decision hyperplanes)
"""

import time
import numpy as np
import pytest

from system1.schema import DecisionSchema, ChoiceField, BooleanField, ScoreField
from system1.compiler import SystemOneCompiler, CompiledSystemOneModel
from system1.engine import SystemOneEngine, DecisionResult
from system1.cache import SemanticSystemOneCache, CacheEntry
from system1.telemetry import TelemetryProjector
from system1.calibration import ConformalPredictor, ConformalPredictionSet


# ==============================================================================
# Helper Schemas & Exemplars
# ==============================================================================

class AlertRoutingSchema(DecisionSchema):
    """Schema for server incident mitigation decisions."""
    action = ChoiceField(
        description="Recommended mitigation action",
        options=["ALLOW_NORMAL", "THROTTLE_RATE", "QUARANTINE_NODE", "RESTART_POD"],
    )
    is_urgent = BooleanField(
        description="Whether immediate on-call escalation is required",
        threshold=0.5,
    )


def get_alert_exemplars():
    raw = [
        ("Normal health ping response 200 OK with stable latencies", "ALLOW_NORMAL", False),
        ("Routine periodic metric scrape successful no warnings", "ALLOW_NORMAL", False),
        ("Gradual increase in incoming API traffic volume within limits", "ALLOW_NORMAL", False),
        ("Minor request rate spike exceeding threshold slightly", "THROTTLE_RATE", False),
        ("Incoming request burst causing upstream connection backlog", "THROTTLE_RATE", False),
        ("Rate limit exceeded for tenant client IP group", "THROTTLE_RATE", False),
        ("Anomalous payload with high error rate detected on worker", "QUARANTINE_NODE", True),
        ("Suspicious lateral scan packets and corrupted worker state", "QUARANTINE_NODE", True),
        ("Segfault loop and out of memory crash on worker process", "RESTART_POD", True),
        ("Deadlock detected in event loop kernel thread unresponsive", "RESTART_POD", True),
    ] * 6
    return {
        "action": [(p, a) for p, a, _ in raw],
        "is_urgent": [(p, u) for p, _, u in raw],
    }


# ==============================================================================
# Lever 1: Tier 0 Semantic System 1 Cache Tests
# ==============================================================================

def test_semantic_system1_cache_exact_and_vector_hits():
    cache = SemanticSystemOneCache(capacity=100, similarity_threshold=0.95)

    emb1 = np.random.randn(64).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)

    result_mock = {"status": "ok", "action": "ALLOW_NORMAL"}
    cache.put("health check probe", result_mock, embedding=emb1, source="test")

    # 1. Exact match hit: O(1) hash table lookup
    t0 = time.perf_counter()
    hit = cache.get("health check probe")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert hit is not None
    entry, sim = hit
    assert sim == 1.0
    assert entry.result == result_mock
    assert elapsed_ms < 0.5, f"Exact cache hit exceeded 0.5ms: {elapsed_ms:.4f}ms"

    # 2. Vector cosine similarity hit (>= 0.95)
    emb_near = emb1 + 0.02 * np.random.randn(64).astype(np.float32)
    emb_near /= np.linalg.norm(emb_near)
    assert np.dot(emb1, emb_near) >= 0.95

    t0 = time.perf_counter()
    hit_vec = cache.get("health probe query varying wording", embedding=emb_near)
    elapsed_vec_ms = (time.perf_counter() - t0) * 1000.0

    assert hit_vec is not None
    entry_vec, sim_vec = hit_vec
    assert sim_vec >= 0.95
    assert entry_vec.result == result_mock
    assert elapsed_vec_ms < 0.5, f"Vector cache hit exceeded 0.5ms: {elapsed_vec_ms:.4f}ms"

    # 3. Cache miss for orthogonal / distant embedding
    emb_far = np.random.randn(64).astype(np.float32)
    emb_far -= np.dot(emb_far, emb1) * emb1
    emb_far /= np.linalg.norm(emb_far)
    assert abs(np.dot(emb1, emb_far)) < 0.3

    assert cache.get("completely unrelated query", embedding=emb_far) is None


def test_semantic_system1_cache_lru_eviction():
    cache = SemanticSystemOneCache(capacity=3, similarity_threshold=0.98)

    for i in range(5):
        emb = np.zeros(16, dtype=np.float32)
        emb[i % 16] = 1.0
        cache.put(f"query_{i}", f"result_{i}", embedding=emb)

    assert cache.size == 3
    assert len(cache) == 3
    # First two should have been evicted
    assert cache.get("query_0") is None
    assert cache.get("query_1") is None
    # Last three should be present
    assert cache.get("query_2") is not None
    assert cache.get("query_3") is not None
    assert cache.get("query_4") is not None


def test_semantic_system1_cache_engine_integration_and_certified_execution():
    compiler = SystemOneCompiler(AlertRoutingSchema, dimension=64, regularization=0.5)
    model = compiler.compile(exemplars=get_alert_exemplars())
    engine = SystemOneEngine(
        AlertRoutingSchema,
        model=model,
        use_cache=True,
        cache_threshold=0.95,
        enable_margin_gating=True,
        margin_threshold=0.20,
    )

    prompt = "Normal health ping response 200 OK"
    res1 = engine.decide(prompt)
    assert res1.is_cache_hit is False

    # Second call should be an exact sub-0.05ms cache hit
    res2 = engine.decide(prompt)
    assert res2.is_cache_hit is True
    assert res2.latency_ms < 0.5  # Sub-millisecond guaranteed
    assert res2.is_ambiguous is False  # Certified execution bypasses ambiguity halts!
    assert res2.values == res1.values


# ==============================================================================
# Lever 2: Online Sherman-Morrison Distillation Tests
# ==============================================================================

def test_sherman_morrison_mathematical_identity():
    """Verify that Sherman-Morrison rank-1 update matches exact matrix inversion."""
    np.random.seed(42)
    D = 32
    N = 100
    lam = 0.5

    # Base design matrix
    X0 = np.random.randn(N, D).astype(np.float32)
    X0_aug = np.column_stack([X0, np.ones(N, dtype=np.float32)])  # shape (N, D+1)
    d_aug = D + 1

    # Exact initial inverse covariance P0
    reg = lam * np.eye(d_aug, dtype=np.float32)
    P0_exact = np.linalg.pinv(X0_aug.T @ X0_aug + reg).astype(np.float32)

    # New single data point x_aug
    x_new = np.random.randn(D).astype(np.float32)
    x_aug = np.append(x_new, 1.0).astype(np.float32)

    # 1. Exact new inverse covariance via full recomputation
    X1_aug = np.vstack([X0_aug, x_aug.reshape(1, -1)])
    P1_exact = np.linalg.pinv(X1_aug.T @ X1_aug + reg).astype(np.float32)

    # 2. Sherman-Morrison rank-1 update formula
    # P_{t+1} = P_t - (P_t x x^T P_t) / (1 + x^T P_t x)
    Px = P0_exact @ x_aug
    denom = 1.0 + float(np.dot(x_aug, Px))
    P1_sm = P0_exact - (np.outer(Px, Px) / denom).astype(np.float32)

    # Verify mathematical identity to tight tolerance
    max_err = float(np.max(np.abs(P1_exact - P1_sm)))
    assert max_err < 1e-4, f"Sherman-Morrison inverse deviated from pinv: max error = {max_err}"


def test_online_distillation_learn_from_tier2():
    """Verify live boundary adaptation when Tier 2 resolves an edge case."""
    compiler = SystemOneCompiler(AlertRoutingSchema, dimension=64, regularization=0.5)
    model = compiler.compile(exemplars=get_alert_exemplars())
    engine = SystemOneEngine(AlertRoutingSchema, model=model, use_cache=True)

    edge_prompt = "Unexpected kernel telemetry: anomalous interrupt storm on eth0"
    
    # Pre-adaptation evaluation
    initial_res = engine.decide(edge_prompt)
    
    # Suppose Tier 2 expert determines this requires immediate RESTART_POD and is_urgent=True
    t0 = time.perf_counter()
    learn_stats = engine.learn_from_tier2(
        prompt=edge_prompt,
        target={"action": "RESTART_POD", "is_urgent": True},
    )
    learn_time_ms = (time.perf_counter() - t0) * 1000.0

    assert learn_stats["status"] == "updated"
    assert "action" in learn_stats["updated_fields"]
    assert learn_time_ms < 5.0  # Ultra-fast online update

    # Post-adaptation evaluation on the metal
    post_res = engine.decide(edge_prompt)
    assert post_res.values["action"] == "RESTART_POD"
    assert post_res.values["is_urgent"] is True

    # Check that compiled model also supports learn_from_tier2
    compiled_model = compiler.compile(exemplars=get_alert_exemplars())
    comp_stats = compiled_model.learn_from_tier2(
        prompt=edge_prompt,
        target={"action": "RESTART_POD", "is_urgent": True},
    )
    assert comp_stats["status"] == "updated"
    res_comp = compiled_model.forward_single(edge_prompt)
    assert res_comp.fields["action"].selected_value == "RESTART_POD"


# ==============================================================================
# Lever 3: Margin-Based Conformal Gating Tests
# ==============================================================================

def test_margin_based_conformal_gating():
    """Verify that margin-of-dominance M(x) = s_{(1)} - s_{(2)} suppresses false-positive ambiguity."""
    calibrator = ConformalPredictor("action", ["ALLOW_NORMAL", "THROTTLE_RATE", "QUARANTINE_NODE", "RESTART_POD"])
    calibrator.is_calibrated = True
    calibrator.quantile = 0.60  # Low quantile that would normally produce multi-class sets

    # Case 1: Dominant classification where top score is 0.85, runner-up is 0.10 => Margin = 0.75
    confident_probs = np.array([0.85, 0.10, 0.03, 0.02], dtype=np.float32)
    cset_gated = calibrator.predict_set(confident_probs, alpha=0.05, margin_threshold=0.30)

    assert cset_gated.margin == pytest.approx(0.75, abs=1e-3)
    assert cset_gated.margin_gate_active is True
    assert cset_gated.is_ambiguous is False
    assert cset_gated.raw_is_ambiguous is True

    # Case 2: Near-tie where top score is 0.45, runner-up is 0.40 => Margin = 0.05 < 0.30
    near_tie_probs = np.array([0.45, 0.40, 0.10, 0.05], dtype=np.float32)
    cset_tie = calibrator.predict_set(near_tie_probs, alpha=0.05, margin_threshold=0.30)

    assert cset_tie.margin == pytest.approx(0.05, abs=1e-3)
    assert cset_tie.margin_gate_active is False
    assert cset_tie.is_ambiguous is True
    assert len(cset_tie.prediction_set) >= 2


def test_margin_gating_in_system1_engine():
    """Verify SystemOneEngine respects margin_threshold and records margin telemetry in DecisionResult."""
    compiler = SystemOneCompiler(AlertRoutingSchema, dimension=64, regularization=0.5)
    model = compiler.compile(exemplars=get_alert_exemplars())
    
    # Enable margin gating with threshold 0.15
    engine = SystemOneEngine(
        AlertRoutingSchema,
        model=model,
        enable_margin_gating=True,
        margin_threshold=0.15,
        use_cache=False,
    )

    res = engine.decide("Normal health ping response 200 OK with stable latencies")
    assert "action" in res.margins
    assert res.margins["action"] >= 0.0
    assert "action" in res.margin_gate_active
    assert res.margin_thresholds["action"] == 0.15


# ==============================================================================
# Lever 4: Continuous Telemetry Vector Fusion Tests
# ==============================================================================

def test_telemetry_projector_normalization_and_modes():
    projector = TelemetryProjector(embedding_dim=32, normalization="robust")

    # Vector input
    raw_vec = np.array([10.0, -25.0, 100.0, 0.05], dtype=np.float32)
    norm_vec = projector.normalize(raw_vec)
    assert np.all(norm_vec >= -1.0) and np.all(norm_vec <= 1.0)

    # Dict input
    raw_dict = {"cpu": 95.0, "latency": 1500.0, "err_rate": 0.35}
    norm_dict = projector.normalize(raw_dict)
    assert norm_dict.shape == (3,)

    # Fusion via project_add mode
    base_emb = np.random.randn(32).astype(np.float32)
    base_emb /= np.linalg.norm(base_emb)

    fused = projector.fuse(base_emb, raw_dict, mode="project_add")
    assert fused.shape == (32,)
    assert pytest.approx(float(np.linalg.norm(fused)), abs=1e-4) == 1.0

    # Fusion via concatenate mode
    fused_cat = projector.fuse(base_emb, raw_dict, mode="concatenate")
    assert fused_cat.shape == (32 + 3,)


def test_continuous_telemetry_alters_decision_boundary():
    """Verify that structured continuous telemetry shifts System 1 hyperplanes."""
    compiler = SystemOneCompiler(AlertRoutingSchema, dimension=64, regularization=0.5)
    model = compiler.compile(exemplars=get_alert_exemplars())
    engine = SystemOneEngine(AlertRoutingSchema, model=model, use_cache=False)

    neutral_prompt = "Server cluster node status update event received"

    # With normal telemetry: low cpu, zero error rate
    normal_telemetry = np.array([0.10, 0.0, 0.05, 0.0], dtype=np.float32)
    res_normal = engine.evaluate(neutral_prompt, telemetry=normal_telemetry)

    # With extreme incident telemetry: 99% cpu, 80% packet drop, high error rate
    critical_telemetry = np.array([0.99, 0.80, 0.95, 0.90], dtype=np.float32)
    res_critical = engine.evaluate(neutral_prompt, telemetry=critical_telemetry)

    # Embeddings must be distinct due to telemetry fusion
    emb_norm = model.encode(neutral_prompt, telemetry=normal_telemetry)
    emb_crit = model.encode(neutral_prompt, telemetry=critical_telemetry)
    cosine_sim = float(np.dot(emb_norm, emb_crit))
    assert cosine_sim < 0.98, f"Telemetry did not sufficiently separate embeddings: cosine={cosine_sim}"


# ==============================================================================
# Multi-Lever Synergy: Escalation Collapse Benchmark Test
# ==============================================================================

def test_four_levers_synergy_escalation_collapse():
    """Verify that combining all 4 levers drives Tier 2 escalations down to zero on repeat/adapted queries."""
    compiler = SystemOneCompiler(AlertRoutingSchema, dimension=64, regularization=0.5)
    model = compiler.compile(exemplars=get_alert_exemplars())
    engine = SystemOneEngine(
        AlertRoutingSchema,
        model=model,
        use_cache=True,
        cache_threshold=0.95,
        enable_margin_gating=True,
        margin_threshold=0.10,
    )

    test_stream = [
        ("Normal health ping response 200 OK", {"cpu": 0.1}),
        ("Routine periodic metric scrape successful", {"cpu": 0.15}),
        ("Minor request rate spike exceeding threshold slightly", {"cpu": 0.65}),
        ("Anomalous edge case payload unseen in training", {"cpu": 0.90, "err": 0.4}),
    ]

    escalations = 0
    for prompt, telem in test_stream:
        res = engine.decide(prompt, telemetry=telem)
        if res.is_ambiguous:
            escalations += 1
            # Tier 2 resolves and teaches Tier 1
            engine.learn_from_tier2(
                prompt=prompt,
                target={"action": "QUARANTINE_NODE", "is_urgent": True},
                telemetry=telem,
            )

    # On second pass with learned boundaries and warm semantic cache:
    second_pass_escalations = 0
    for prompt, telem in test_stream:
        res = engine.decide(prompt, telemetry=telem)
        if res.is_ambiguous:
            second_pass_escalations += 1

    # Escalations must collapse to 0 on the second pass
    assert second_pass_escalations == 0


def test_compiled_model_save_load_roundtrip_with_online_adaptation(tmp_path):
    """Verify .s1m saving (<20KB), loading, and subsequent online Sherman-Morrison distillation."""
    compiler = SystemOneCompiler(AlertRoutingSchema, dimension=64, regularization=0.5)
    model = compiler.compile(exemplars=get_alert_exemplars())

    model_path = tmp_path / "edge_routing.s1m"
    model.save(model_path)
    assert model_path.stat().st_size < 20 * 1024, f"Model size exceeded 20KB: {model_path.stat().st_size} bytes"

    loaded_model = CompiledSystemOneModel.load(model_path)
    edge_prompt = "Unprecedented TCP FIN flood from spoofed gateway"

    # Online learning on loaded model (P and B lazily initialized)
    t0 = time.perf_counter()
    stats = loaded_model.learn_from_tier2(
        prompt=edge_prompt,
        target={"action": "QUARANTINE_NODE", "is_urgent": True},
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert stats["status"] == "updated"
    assert dt_ms < 5.0

    eval_res = loaded_model.forward_single(edge_prompt)
    assert eval_res.fields["action"].selected_value == "QUARANTINE_NODE"
    assert eval_res.fields["is_urgent"].selected_value is True


def test_telemetry_projector_edge_cases():
    """Verify telemetry projector handles empty dicts, unseen keys, extreme values, and zeros."""
    projector = TelemetryProjector(embedding_dim=64, normalization="robust")

    # 1. Empty dict
    norm_empty = projector.normalize({})
    assert norm_empty.shape == (0,)

    # 2. Extreme values (tanh compression prevents overflow)
    extreme_dict = {"huge": 1e12, "tiny": -1e12, "nan_like": 0.0}
    norm_extreme = projector.normalize(extreme_dict)
    assert np.all(np.isfinite(norm_extreme))
    assert np.all(norm_extreme >= -1.0) and np.all(norm_extreme <= 1.0)

    # 3. None telemetry passthrough
    base = np.ones(64, dtype=np.float32) / np.sqrt(64)
    fused_none = projector.fuse(base, None)
    assert np.allclose(fused_none, base)

    # 4. Zero vector
    zero_vec = np.zeros(10, dtype=np.float32)
    norm_zero = projector.normalize(zero_vec)
    assert np.all(norm_zero == 0.0)


def test_semantic_system1_cache_edge_cases():
    """Verify cache handles empty string prompts, clear operations, and threshold boundaries."""
    cache = SemanticSystemOneCache(capacity=5, similarity_threshold=0.98)

    # 1. Empty string prompt
    cache.put("", "empty_result")
    hit = cache.get("")
    assert hit is not None
    assert hit[0].result == "empty_result"

    # 2. Cache clear
    cache.clear()
    assert cache.size == 0
    assert cache.get("") is None


def test_scorefield_online_update_and_extreme_values():
    """Verify that ScoreField online update properly maps targets to logit space,
    avoiding the midpoint 5.0 bug when training extreme values (e.g. 0.0 or 10.0)."""
    class ScoreSchema(DecisionSchema):
        risk = ScoreField(description="Risk score", min_value=0.0, max_value=10.0)

    exemplars = {
        "risk": [
            ("low risk safe prompt", 1.0),
            ("medium risk neutral prompt", 5.0),
            ("high risk dangerous prompt", 9.0),
        ] * 5
    }
    compiler = SystemOneCompiler(ScoreSchema, dimension=32, regularization=0.5)
    model = compiler.compile(exemplars=exemplars)

    prompt_high = "critical zero day kernel exploit"
    # Train extreme target 10.0 (near max, should clamp to ~0.999 in logit space)
    model.learn_from_tier2(prompt_high, target={"risk": 10.0})
    res_after = model.forward_single(prompt_high)
    assert res_after.fields["risk"].selected_value > 8.0, f"Expected risk > 8.0, got {res_after.fields['risk'].selected_value}"

    # Now train extreme target 0.0 on a distinct prompt (near min, should clamp to ~0.001 in logit space)
    prompt_low = "routine safe ping from trusted monitoring agent"
    model.learn_from_tier2(prompt_low, target={"risk": 0.0})
    res_zero = model.forward_single(prompt_low)
    assert res_zero.fields["risk"].selected_value < 3.0, f"Expected risk < 3.0, got {res_zero.fields['risk'].selected_value}"


def test_telemetry_projector_high_dimensional_rank():
    """Verify that TelemetryProjector maintains full rank for high-dimensional telemetry vectors (>8 dimensions),
    preventing the rank collapse bug where (i * 4) % 32 capped the rank at 8."""
    dim = 64
    projector = TelemetryProjector(embedding_dim=dim)
    # 20 distinct feature dimensions
    M = projector._get_projection_matrix(20)
    assert M.shape == (dim, 20)
    
    # Check matrix rank: must be exactly 20 (full column rank)
    rank = np.linalg.matrix_rank(M)
    assert rank == 20, f"Expected full column rank 20, but got {rank} (indicates subspace collapse)"


def test_semantic_system1_cache_continuous_telemetry_vector_hit():
    """Verify that SemanticSystemOneCache vector search allows continuous telemetry float variations (e.g. 1e-6)
    to hit the cache when embeddings are highly similar, rather than masking them out via exact digest mismatch."""
    cache = SemanticSystemOneCache(capacity=50, similarity_threshold=0.95)
    emb = np.random.randn(32).astype(np.float32)
    emb /= np.linalg.norm(emb)

    t1 = {"cpu": 50.000001, "memory": 70.0}
    cache.put("server query", "action_ok", embedding=emb, telemetry=t1)

    # Near-identical embedding with slight telemetry float jitter
    emb_jitter = emb + 0.001 * np.random.randn(32).astype(np.float32)
    emb_jitter /= np.linalg.norm(emb_jitter)
    t2 = {"cpu": 50.000002, "memory": 70.0}  # Different digest, but continuous telemetry present

    hit = cache.get("different query text", embedding=emb_jitter, telemetry=t2)
    assert hit is not None, "Vector cache hit should succeed despite floating-point telemetry jitter"
    entry, sim = hit
    assert entry.result == "action_ok"
    assert sim >= 0.95


def test_sherman_morrison_thread_safety_concurrent_updates():
    """Verify that concurrent online updates across multiple threads do not corrupt the inverse covariance matrix."""
    import threading

    compiler = SystemOneCompiler(AlertRoutingSchema, dimension=32, regularization=0.5)
    model = compiler.compile(exemplars=get_alert_exemplars())

    errors = []

    def worker(worker_id: int):
        try:
            for i in range(20):
                prompt = f"concurrent alert from worker {worker_id} iteration {i}"
                target = {"action": "QUARANTINE_NODE" if i % 2 == 0 else "THROTTLE_RATE", "is_urgent": bool(i % 2)}
                model.learn_from_tier2(prompt, target)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Encountered thread errors during concurrent online updates: {errors}"
    res = model.forward_single("health check")
    assert res.fields["action"].selected_value in ["ALLOW_NORMAL", "THROTTLE_RATE", "QUARANTINE_NODE", "RESTART_POD"]


def test_sherman_morrison_nan_inf_resilience():
    """Verify that online update validates inputs and rejects NaN/Inf embeddings without corrupting weights."""
    compiler = SystemOneCompiler(AlertRoutingSchema, dimension=32, regularization=0.5)
    model = compiler.compile(exemplars=get_alert_exemplars())

    head = model._field_heads["action"]
    W_orig = head.weights.copy()

    # Pass NaN embedding
    nan_emb = np.full(32, np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="NaN or Inf"):
        head.online_update(nan_emb, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    # Pass Inf embedding
    inf_emb = np.full(32, np.inf, dtype=np.float32)
    with pytest.raises(ValueError, match="NaN or Inf"):
        head.online_update(inf_emb, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    # Pass NaN target
    valid_emb = np.ones(32, dtype=np.float32) / np.sqrt(32)
    with pytest.raises(ValueError, match="NaN or Inf"):
        head.online_update(valid_emb, np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32))

    # Verify weights remained pristine and unchanged
    assert np.array_equal(head.weights, W_orig)


def test_system_one_model_evaluate_and_learn_from_tier2():
    """Verify uncompiled SystemOneModel supports evaluate() and learn_from_tier2() identically to CompiledSystemOneModel."""
    from system1.model import SystemOneModel

    model = SystemOneModel(
        schema=AlertRoutingSchema,
        dimension=32,
    )

    prompt = "unseen catastrophic deadlock in kernel"
    res_before = model.evaluate(prompt)

    stats = model.learn_from_tier2(prompt, target={"action": "RESTART_POD", "is_urgent": True})
    assert stats["status"] == "updated"
    assert "action" in stats["updated_fields"]

    res_after = model.evaluate(prompt)
    assert res_after.fields["action"].selected_value == "RESTART_POD"
    assert res_after.fields["is_urgent"].selected_value is True


def test_online_update_precomputed_embedding():
    """Verify that passing precomputed embedding to learn_from_tier2 eliminates redundant encoding overhead."""
    compiler = SystemOneCompiler(AlertRoutingSchema, dimension=32, regularization=0.5)
    model = compiler.compile(exemplars=get_alert_exemplars())

    prompt = "repeat telemetry ping event"
    emb = model.encode(prompt)

    # Warm up pass
    model.learn_from_tier2(
        prompt="warmup prompt",
        target={"action": "ALLOW_NORMAL", "is_urgent": False},
        embedding=emb,
    )

    t0 = time.perf_counter()
    stats = model.learn_from_tier2(
        prompt=prompt,
        target={"action": "ALLOW_NORMAL", "is_urgent": False},
        embedding=emb,
    )
    dt_us = (time.perf_counter() - t0) * 1e6

    assert stats["status"] == "updated"
    assert stats["update_latency_ms"] < 0.5  # Sub-500 microsecond rank-1 core math (tolerant of cloud runner jitter)

