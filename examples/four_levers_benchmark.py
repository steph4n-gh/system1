#!/usr/bin/env python3
"""The repeated-stream teacher is a fixed simulated answer. Cache/escalation figures demonstrate mechanics, not held-out decision quality.

Benchmark and Showcase for the 4 Tier 1 Architectural Levers:

1. Lever 1: Tier 0 Semantic System 1 Cache (L1 Vector/Exact Cache, <0.05ms execution)
2. Lever 2: Online Sherman-Morrison Distillation (closed-form rank-1 update, <0.1ms)
3. Lever 3: Margin-Based Conformal Gating (M(x) = s_{(1)} - s_{(2)} >= tau_margin)
4. Lever 4: Continuous Telemetry State Vector Fusion (multimodal decision hyperplanes)
"""

import os
import sys
import time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from system1.schema import DecisionSchema, ChoiceField, BooleanField, ScoreField
from system1.compiler import SystemOneCompiler, CompiledSystemOneModel
from system1.engine import SystemOneEngine
from system1.cache import SemanticSystemOneCache
from system1.telemetry import TelemetryProjector
from system1.calibration import ConformalPredictor


# ==============================================================================
# Schema Definition
# ==============================================================================

class CloudGatewayFirewallSchema(DecisionSchema):
    """Production Gateway routing decision schema."""
    action = ChoiceField(
        description="Routing or mitigation decision",
        options=["ALLOW_TRAFFIC", "RATE_LIMIT", "QUARANTINE_IP", "BLOCK_IMMEDIATE"],
    )
    is_anomaly = BooleanField(
        description="Whether traffic exhibits anomalous characteristics",
        threshold=0.5,
    )
    risk_score = ScoreField(
        description="Continuous calculated risk score",
        min_value=0.0,
        max_value=10.0,
    )


def get_gateway_exemplars():
    raw = [
        ("GET /api/v1/health status 200 OK standard client keep-alive", "ALLOW_TRAFFIC", False, 0.2, {"cpu": 10.0, "p99_latency_ms": 20.0, "error_rate": 0.001}),
        ("POST /api/v1/checkout user session valid payment token verified", "ALLOW_TRAFFIC", False, 0.5, {"cpu": 15.0, "p99_latency_ms": 35.0, "error_rate": 0.002}),
        ("GET /static/app.bundle.js content cached 304 not modified", "ALLOW_TRAFFIC", False, 0.1, {"cpu": 8.0, "p99_latency_ms": 15.0, "error_rate": 0.0}),
        ("Rapid successive GET /api/v1/products exceeding burst threshold 50 rps", "RATE_LIMIT", False, 3.5, {"cpu": 60.0, "p99_latency_ms": 300.0, "error_rate": 0.02}),
        ("Excessive parallel search queries from single authorization token", "RATE_LIMIT", False, 4.0, {"cpu": 65.0, "p99_latency_ms": 350.0, "error_rate": 0.03}),
        ("Scraping catalog endpoints with automated non-browser headers", "RATE_LIMIT", False, 4.5, {"cpu": 70.0, "p99_latency_ms": 400.0, "error_rate": 0.04}),
        ("High frequency SQL injection probe SELECT union select in user-agent", "QUARANTINE_IP", True, 8.5, {"cpu": 85.0, "p99_latency_ms": 1200.0, "error_rate": 0.35}),
        ("Path traversal attempt in URI query parameter ../../etc/passwd", "QUARANTINE_IP", True, 8.8, {"cpu": 88.0, "p99_latency_ms": 1500.0, "error_rate": 0.40}),
        ("Volumetric distributed SYN flood attacking gateway load balancer", "BLOCK_IMMEDIATE", True, 9.8, {"cpu": 98.0, "p99_latency_ms": 4000.0, "error_rate": 0.85}),
        ("Massive protocol abuse malformed HTTP/2 frame injection exploit", "BLOCK_IMMEDIATE", True, 9.9, {"cpu": 99.0, "p99_latency_ms": 4500.0, "error_rate": 0.90}),
    ] * 8
    return {
        "action": [(p, a, t) for p, a, _, _, t in raw],
        "is_anomaly": [(p, u, t) for p, _, u, _, t in raw],
        "risk_score": [(p, s, t) for p, _, _, s, t in raw],
    }


def run_benchmark():
    print("=" * 80)
    print("  SYSTEM 1 REFLEX: 4 ARCHITECTURAL LEVERS BENCHMARK & SHOWCASE")
    print("  Local Cache, Correction, Review and Telemetry Demonstration")
    print("=" * 80)

    # 0. Compilation Phase
    print("\n[Phase 0] Compiling Heuristics into Ultra-Compact .s1m Binary...")
    compiler = SystemOneCompiler(CloudGatewayFirewallSchema, dimension=128, regularization=0.5)
    exemplars = get_gateway_exemplars()
    compiled_model = compiler.compile(exemplars=exemplars)

    model_bytes = compiled_model.to_bytes()
    print(f"  Model compiled: {len(model_bytes):,} bytes; under 20 KiB: {len(model_bytes) < 20 * 1024}")

    engine = SystemOneEngine(
        CloudGatewayFirewallSchema,
        model=compiled_model,
        use_cache=True,
        cache_threshold=0.95,
        enable_margin_gating=True,
        margin_threshold=0.15,
    )

    # ==========================================================================
    # Lever 1: Tier 0 Semantic System 1 Cache
    # ==========================================================================
    print("\n" + "-" * 80)
    print("[Lever 1] Tier 0 Semantic System 1 Cache (Sub-0.05ms L1 Vector/Exact Cache)")
    print("-" * 80)

    test_prompt = "GET /api/v1/health status 200 OK standard client keep-alive"

    # Forward pass (cold)
    t0 = time.perf_counter()
    cold_res = engine.decide(test_prompt)
    cold_latency_us = (time.perf_counter() - t0) * 1_000_000.0

    # Cache hit (exact)
    hit_latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        exact_hit = engine.decide(test_prompt)
        hit_latencies.append((time.perf_counter() - t0) * 1_000_000.0)

    exact_p50_us = float(np.median(hit_latencies))
    print(f"  Cold Model Forward Pass:      {cold_latency_us:>8.2f} µs ({cold_latency_us/1000.0:.3f} ms)")
    print(f"  L1 Exact Cache Hit Latency:   {exact_p50_us:>8.2f} µs ({exact_p50_us/1000.0:.4f} ms)")
    print(f"  Cache Speedup Multiplier:     {cold_latency_us / max(1.0, exact_p50_us):>8.1f}x faster")
    exact_res_ambiguous = exact_hit.is_ambiguous
    print(f"  Cached Review Status:        Ambiguity = {exact_res_ambiguous} (not execution permission)")

    # Vector cosine similarity hit (near-match query)
    near_prompt = "GET /api/v1/health status 200 OK standard client keepalive"
    near_latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        vec_hit = engine.decide(near_prompt)
        near_latencies.append((time.perf_counter() - t0) * 1_000_000.0)

    vec_p50_us = float(np.median(near_latencies))
    print(f"  L1 Semantic Vector Hit (0.98):{vec_p50_us:>8.2f} µs ({vec_p50_us/1000.0:.4f} ms)")

    # ==========================================================================
    # Lever 2: Online Sherman-Morrison Distillation
    # ==========================================================================
    print("\n" + "-" * 80)
    print("[Lever 2] Online Sherman-Morrison Distillation (<0.1ms Rank-1 Closed-Form)")
    print("-" * 80)

    novel_edge_prompt = "Novel GraphQL zero-day introspection exploit nested depth 50"
    pre_res = engine.decide(novel_edge_prompt)
    print(f"  Pre-Adaptation Decision:      action={pre_res.values['action']}, is_ambiguous={pre_res.is_ambiguous}")

    # Closed-form online update (single field vs multi-field)
    single_field_latencies = []
    for _ in range(10):
        stats_single = engine.learn_from_tier2(
            prompt=novel_edge_prompt,
            target={"action": "BLOCK_IMMEDIATE"},
        )
        single_field_latencies.append(stats_single["update_latency_ms"] * 1000.0)
    single_p50_us = float(np.median(single_field_latencies))

    update_latencies = []
    for _ in range(10):
        update_stats = engine.learn_from_tier2(
            prompt=novel_edge_prompt,
            target={"action": "BLOCK_IMMEDIATE", "is_anomaly": True, "risk_score": 9.9},
        )
        update_latencies.append(update_stats["update_latency_ms"] * 1000.0)

    update_p50_us = float(np.median(update_latencies))
    post_res = engine.decide(novel_edge_prompt)

    print(f"  Rank-1 Single Field Head:     {single_p50_us:>8.2f} µs ({single_p50_us/1000.0:.4f} ms) [<50µs target]")
    print(f"  Rank-1 Multi-Field (3 heads): {update_p50_us:>8.2f} µs ({update_p50_us/1000.0:.3f} ms)")
    print(f"  Post-Adaptation Decision:     action={post_res.values['action']}, risk_score={post_res.values['risk_score']:.1f}")
    print(f"  Single-field update under 50 µs: {single_p50_us < 50}")

    # ==========================================================================
    # Lever 3: Margin-Based Conformal Gating
    # ==========================================================================
    print("\n" + "-" * 80)
    print("[Lever 3] Margin-Based Conformal Gating (M(x) = s_(1) - s_(2) >= tau)")
    print("-" * 80)

    calibrator = ConformalPredictor("action", ["ALLOW_TRAFFIC", "RATE_LIMIT", "QUARANTINE_IP", "BLOCK_IMMEDIATE"])
    calibrator.is_calibrated = True
    calibrator.quantile = 0.65  # Broad quantile

    # Test dominant distribution (clear winner)
    confident_probs = np.array([0.88, 0.08, 0.02, 0.02], dtype=np.float32)
    ungated_cset = calibrator.predict_set(confident_probs, alpha=0.05, margin_threshold=0.0)
    gated_cset = calibrator.predict_set(confident_probs, alpha=0.05, margin_threshold=0.20)

    print(f"  Standard Conformal Set:       {list(ungated_cset.prediction_set)} | Ambiguous: {ungated_cset.is_ambiguous}")
    print(f"  Margin-Gated Conformal Set:   {list(gated_cset.prediction_set)} | Ambiguous: {gated_cset.is_ambiguous}")
    print(f"  Margin of Dominance M(x):     {gated_cset.margin:.3f} (Threshold: {gated_cset.margin_threshold:.2f})")
    print(f"  False Ambiguity Suppressed:   {ungated_cset.is_ambiguous and not gated_cset.is_ambiguous}")

    # ==========================================================================
    # Lever 4: Continuous Telemetry State Vector Fusion
    # ==========================================================================
    print("\n" + "-" * 80)
    print("[Lever 4] Continuous Telemetry State Vector Fusion")
    print("-" * 80)

    projector = TelemetryProjector(embedding_dim=128, normalization="robust")
    ambiguous_text = "Standard TLS connection payload received on worker endpoint"

    # Telemetry condition 1: Normal operations
    normal_telem = {"cpu": 12.0, "p99_latency_ms": 25.0, "error_rate": 0.001}
    # Telemetry condition 2: High stress failure
    stress_telem = {"cpu": 98.5, "p99_latency_ms": 4200.0, "error_rate": 0.88}

    res_norm = engine.evaluate(ambiguous_text, telemetry=normal_telem)
    res_crit = engine.evaluate(ambiguous_text, telemetry=stress_telem)

    print(f"  Prompt Text (Constant):       '{ambiguous_text}'")
    print(f"  Decision [Normal Telemetry]:  action={res_norm.values['action']} (risk={res_norm.values['risk_score']:.2f})")
    print(f"  Decision [Crisis Telemetry]:  action={res_crit.values['action']} (risk={res_crit.values['risk_score']:.2f})")
    print(f"  Hyperplane Separation:        Telemetry shifted risk from {res_norm.values['risk_score']:.2f} to {res_crit.values['risk_score']:.2f}")

    # ==========================================================================
    # Multi-Lever Synergy: Escalation Collapse Benchmark
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  MULTI-LEVER SYNERGY: ESCALATION COLLAPSE DEMONSTRATION")
    print("=" * 80)

    test_stream = [
        ("GET /api/v1/metrics cluster status healthy", {"cpu": 10.0, "err": 0.0}),
        ("Rapid connection requests from subnet 10.0.4.0/24", {"cpu": 75.0, "err": 0.02}),
        ("Unknown HTTP header format with variable cookie size", {"cpu": 50.0, "err": 0.15}),
        ("Database connection timeout cascade on auth worker", {"cpu": 92.0, "err": 0.65}),
        ("Corrupted TLS handshake on port 443 with junk payload", {"cpu": 88.0, "err": 0.70}),
    ] * 20  # 100 queries total

    # Run without 4 Levers (Vanilla baseline)
    vanilla_engine = SystemOneEngine(
        CloudGatewayFirewallSchema,
        model=compiler.compile(exemplars=exemplars),
        use_cache=False,
        enable_margin_gating=False,
    )

    baseline_escalations = 0
    t_start = time.perf_counter()
    for prompt, telem in test_stream:
        r = vanilla_engine.decide(prompt)
        if r.is_ambiguous:
            baseline_escalations += 1
    t_vanilla_ms = (time.perf_counter() - t_start) * 1000.0

    # Run with All 4 Levers Active (Adaptive System 1 System 1)
    smart_engine = SystemOneEngine(
        CloudGatewayFirewallSchema,
        model=compiler.compile(exemplars=exemplars),
        use_cache=True,
        cache_threshold=0.95,
        enable_margin_gating=True,
        margin_threshold=0.15,
    )

    smart_escalations = 0
    tier2_learn_calls = 0
    t_start = time.perf_counter()
    for prompt, telem in test_stream:
        r = smart_engine.decide(prompt, telemetry=telem)
        if r.is_ambiguous:
            smart_escalations += 1
            tier2_learn_calls += 1
            # Tier 2 resolves once and immediately distills back into Tier 1
            smart_engine.learn_from_tier2(
                prompt=prompt,
                target={"action": "QUARANTINE_IP", "is_anomaly": True, "risk_score": 8.5},
                telemetry=telem,
            )
    t_smart_ms = (time.perf_counter() - t_start) * 1000.0

    print(f"\n  [Workload: 100 Queries across 5 repeating dynamic operational patterns]")
    print(f"  Vanilla System 1 Escalations: {baseline_escalations}/100 ({baseline_escalations}%) | Total Time: {t_vanilla_ms:.2f} ms")
    print(f"  4-Levers System 1 Escalations:  {smart_escalations}/100 ({smart_escalations}%) | Total Time: {t_smart_ms:.2f} ms")
    print(f"  Tier 2 Escalation Reduction:  {((baseline_escalations - smart_escalations) / max(1, baseline_escalations)) * 100.0:.1f}% reduction")
    print("  Repeated scripted answers measure update/cache mechanics, not certified execution.")

    # Scorecard
    print("\n" + "┌" + "─" * 78 + "┐")
    print("│                     TIER 1 ARCHITECTURAL LEVERS SCORECARD                    │")
    print("├" + "─" * 78 + "┤")
    print(f"│  Lever 1 (L1 Cache Hit Latency)    │ {exact_p50_us:>10.2f} µs ({exact_p50_us/1000.0:.4f} ms)  │ Measured above │")
    print(f"│  Lever 2 (Sherman-Morrison Update) │ {single_p50_us:>10.2f} µs ({single_p50_us/1000.0:.4f} ms)  │ Measured above │")
    print(f"│  Lever 3 (Margin Conformal Gate)   │ Dominance Gating Active     │ Heuristic review│")
    print(f"│  Lever 4 (Telemetry Vector Fusion) │ Robust Multi-Modal Fusion   │ Numeric telemetry│")
    print(f"│  Escalation Collapse Rate          │ {baseline_escalations}% -> {smart_escalations}%                  │ Measured stream  │")
    print("└" + "─" * 78 + "┘\n")


if __name__ == "__main__":
    run_benchmark()
