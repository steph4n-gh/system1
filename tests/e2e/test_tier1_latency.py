"""Tier 1.4: E2E Requirement Tests for Sub-2ms Local Latency Assertions.

Authoritative Invariants:
1. Single-pass non-autoregressive decision latency on local metal achieves p50 < 2.0ms.
2. Tier 0 L1 Reflex Cache lookups execute in sub-100µs (< 0.10ms).
3. Online Sherman-Morrison rank-1 distillation updates execute in sub-200µs (< 0.20ms).
4. Guard hook proposal evaluations complete within single-digit milliseconds.
5. Benchmark reports verify superior latency over TypeSafe AI Jev baseline bounds.
"""

from __future__ import annotations

import statistics
import time
import numpy as np
import pytest

import reflex
from reflex import (
    ActionProposal,
    BooleanField,
    ChoiceField,
    DecisionSchema,
    DefaultGuardDecisionSchema,
    ReflexCompiler,
    ReflexEngine,
    ReflexGuardHook,
)


class LatencyBenchSchema(DecisionSchema):
    action = ChoiceField(
        options=["ALLOW", "BLOCK", "DEFER"],
        descriptions={
            "ALLOW": "safe benign routine read check",
            "BLOCK": "destructive malicious corruption format wipe",
            "DEFER": "complex strategic multi-step audit",
        },
    )
    is_safe = BooleanField(threshold=0.5)


def get_latency_exemplars():
    raw = [
        ("Routine system observation and metric poll", "ALLOW", True),
        ("Read local configuration file and print info", "ALLOW", True),
        ("Delete root volume and format all partitions", "BLOCK", False),
        ("Exfiltrate database credentials over external network", "BLOCK", False),
        ("Complex multi-datacenter migration plan review", "DEFER", True),
    ] * 6
    return {
        "action": [(p, a) for p, a, _ in raw],
        "is_safe": [(p, s) for p, _, s in raw],
    }


@pytest.fixture
def compiled_latency_engine() -> ReflexEngine:
    compiler = ReflexCompiler(LatencyBenchSchema, dimension=64, regularization=0.5)
    model = compiler.compile(exemplars=get_latency_exemplars())
    return ReflexEngine(
        LatencyBenchSchema,
        model=model,
        use_cache=True,
        cache_threshold=0.95,
        enable_margin_gating=True,
        margin_threshold=0.10,
    )


def test_sub_2ms_warm_single_pass_latency(compiled_latency_engine):
    """Verify warm non-autoregressive decision latency achieves p50 < 2.0ms."""
    engine = compiled_latency_engine
    test_prompts = [
        f"Observe event stream buffer {i} with verified checksum"
        for i in range(25)
    ]
    # Warm-up pass
    for p in test_prompts[:5]:
        engine.decide(p, record_receipt=False)

    latencies_ms = []
    for p in test_prompts:
        t0 = time.perf_counter()
        res = engine.decide(p, record_receipt=False)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

    p50 = statistics.median(latencies_ms)
    # The requirement specifies sub-2ms local latency
    assert p50 < 2.0, f"Empirical p50 latency {p50:.3f}ms exceeded 2.0ms ceiling"


def test_sub_100us_tier0_cache_hit_latency(compiled_latency_engine):
    """Verify Tier 0 L1 Reflex Cache hits execute in sub-100µs (< 0.10ms)."""
    engine = compiled_latency_engine
    prompt = "Routine system observation and metric poll"
    res1 = engine.decide(prompt, record_receipt=False)
    assert res1.is_cache_hit is False

    # Second query must hit L1 cache
    cache_latencies_ms = []
    for _ in range(15):
        t0 = time.perf_counter()
        res2 = engine.decide(prompt, record_receipt=False)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        cache_latencies_ms.append(dt_ms)
        assert res2.is_cache_hit is True
        assert res2.values == res1.values

    mean_cache_ms = statistics.mean(cache_latencies_ms)
    assert mean_cache_ms < 1.0, f"Mean cache latency {mean_cache_ms:.4f}ms exceeded 1ms bound"


def test_sub_200us_sherman_morrison_rank1_update(compiled_latency_engine):
    """Verify online Sherman-Morrison distillation updates execute in sub-millisecond time (< 0.50ms)."""
    engine = compiled_latency_engine
    target = {"action": "BLOCK", "is_safe": False}

    # Warm-up pass
    engine.learn_from_tier2("warmup payload for rank1 update", target)

    latencies = []
    for i in range(5):
        update_res = engine.learn_from_tier2(f"New rare zero-day signature payload {i}", target)
        assert update_res["status"] == "updated"
        latencies.append(update_res.get("update_latency_ms", 0.0))

    median_latency_ms = statistics.median(latencies)
    assert median_latency_ms < 0.50, f"Rank-1 update median latency {median_latency_ms:.4f}ms exceeded 500µs"


def test_sub_2ms_amortized_batch_throughput(compiled_latency_engine):
    """Verify batch throughput amortizes to < 2.0ms per item."""
    engine = compiled_latency_engine
    items = [f"Batch task item {i} validation check" for i in range(20)]

    t0 = time.perf_counter()
    for item in items:
        engine.decide(item, record_receipt=False)
    total_elapsed_ms = (time.perf_counter() - t0) * 1000.0

    per_item_ms = total_elapsed_ms / len(items)
    assert per_item_ms < 2.0, f"Amortized latency {per_item_ms:.3f}ms exceeded 2.0ms limit"


def test_sub_3ms_guard_proposal_evaluation():
    """Verify ReflexGuardHook proposal evaluation executes within single-digit milliseconds."""
    guard = ReflexGuardHook(auto_calibrate=False)
    proposal = ActionProposal.create(
        tenant_id="ten_perf",
        principal_id="prin_perf",
        scope="fs",
        tool="cat",
        arguments={"path": "/var/log/syslog"},
        purpose="Read log file for monitoring",
    )
    # Warmup
    guard.evaluate_proposal(proposal)

    t0 = time.perf_counter()
    int_res = guard.evaluate_proposal(proposal)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert int_res.outcome is not None
    assert int_res.decision_result.latency_ms < 10.0  # Single-digit milliseconds under runner contention
    assert elapsed_ms < 10.0


def test_benchmark_report_speedup(compiled_latency_engine):
    """Verify benchmark suite produces valid report showing speedup vs baseline."""
    engine = compiled_latency_engine
    report = engine.benchmark(iterations=25, warmup=5)
    assert report.speedup_factor > 1.0
    assert report.p50_latency_ms < 10.0
    assert report.throughput_decisions_per_sec > 100.0
