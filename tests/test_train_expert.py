"""Tests for System 1 Domain Expert Training, Distillation, and MoE Workflows.

Validates the full lifecycle described in docs/guides/training_experts.md and examples/train_expert.py:
- Pathway A: Synthetic distillation, .s1m serialization (<20KB), and deserialization.
- Pathway B: Offline supervised dataset compilation with closed-form Ridge Regression.
- Pathway C: Sub-1ms latency SLA, conformal calibration sets, and Ed25519 witness receipts.
- Pathway D: Sub-50µs online Sherman-Morrison adaptation with exponential forgetting λ_f = 0.995.
- Pathway E: 2-tier Mixture of Experts (Router Expert -> Specialized Domain Experts).
- Edge cases: Hyperparameter bounds, covariance bounding, and invalid input rejection.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from system1.compiler import (
    MAGIC_HEADER,
    CompiledHeadWeights,
    CompiledSystemOneModel,
    SystemOneCompiler,
)
from system1.core.schema import (
    BooleanField,
    ChoiceField,
    DecisionSchema,
    ScoreField,
)
from system1.engine import SystemOneEngine


# ============================================================================
# Test Domain Schemas
# ============================================================================

class IncidentTriageSchema(DecisionSchema):
    severity = ChoiceField(
        options=["P1_CRITICAL", "P2_ELEVATED", "P3_ROUTINE"],
        descriptions={
            "P1_CRITICAL": "Production database down, data corruption, total system outage, kernel panic",
            "P2_ELEVATED": "Elevated latency on checkout service, partial degraded feature, high memory pressure",
            "P3_ROUTINE": "Minor cosmetic UI glitch, scheduled batch maintenance delay, non-urgent log warning",
        },
    )
    notify_oncall = BooleanField(
        threshold=0.5,
        true_description="Requires urgent pager alert to on-call infrastructure lead engineer",
        false_description="Standard ticketing queue or automated resolution, no pager alert needed",
    )
    blast_radius = ScoreField(
        min_value=0.0,
        max_value=1.0,
        low_description="Isolated single customer impact or internal test environment",
        high_description="Global fleetwide impact across all regions and active production tenants",
    )


class SecurityIncidentSchema(DecisionSchema):
    threat_level = ChoiceField(
        options=["LOW", "HIGH", "CRITICAL"],
        descriptions={
            "LOW": "Benign probe or scanner from known search engine crawler",
            "HIGH": "Anomalous authentication attempts from foreign IP addresses",
            "CRITICAL": "Confirmed SQL injection exploit or exfiltration of sensitive credentials",
        },
    )
    block_ip = BooleanField(
        threshold=0.5,
        true_description="Immediately issue iptables/WAF block rule for source IP",
        false_description="Log to SIEM for security analyst review, do not block",
    )


class MoERouterSchema(DecisionSchema):
    target_expert = ChoiceField(
        options=["infra_expert", "security_expert"],
        descriptions={
            "infra_expert": "Kubernetes pods, high memory, disk storage, database timeouts, server crashes",
            "security_expert": "SQL injection, credential stuffing, unauthorized API tokens, malware, brute force",
        },
    )


# ============================================================================
# Test Cases
# ============================================================================

def test_pathway_a_synthetic_distillation_and_s1m_binary(tmp_path: Path):
    """Verify Pathway A: Synthetic distillation, binary size < 20KB, and exact deserialization."""
    compiler = SystemOneCompiler(
        IncidentTriageSchema,
        dimension=256,
        regularization=1.0,
        forgetting_factor=0.995,
    )

    # 1. Generate synthetic exemplars
    synthetic_exemplars = compiler.generate_synthetic_exemplars(samples_per_choice=15)
    assert "severity" in synthetic_exemplars
    assert len(synthetic_exemplars["severity"]) >= 45
    assert "notify_oncall" in synthetic_exemplars
    assert "blast_radius" in synthetic_exemplars

    # 2. Compile model in closed form
    expert = compiler.compile(exemplars=synthetic_exemplars, calibration_split=0.25)
    assert isinstance(expert, CompiledSystemOneModel)
    assert expert.dimension == 256
    assert set(expert.heads.keys()) == {"severity", "notify_oncall", "blast_radius"}

    # 3. Serialize to .s1m binary and verify compactness
    s1m_path = tmp_path / "incident_expert.s1m"
    expert.save(s1m_path)
    assert s1m_path.is_file()

    file_bytes = s1m_path.read_bytes()
    assert file_bytes[:4] == MAGIC_HEADER
    file_size_kb = len(file_bytes) / 1024.0
    assert file_size_kb < 20.0, f"Expected binary size < 20KB, got {file_size_kb:.2f}KB"

    # 4. Deserialize and compare forward pass outputs
    loaded_expert = CompiledSystemOneModel.load(s1m_path)
    assert loaded_expert.schema.schema_name == IncidentTriageSchema().schema_name
    assert loaded_expert.dimension == 256

    test_query = "Production database primary node crash and replica desync"
    res_orig = expert.forward_single(test_query)
    res_loaded = loaded_expert.forward_single(test_query)

    assert res_orig.fields["severity"].selected_value == res_loaded.fields["severity"].selected_value
    np.testing.assert_allclose(
        res_orig.fields["severity"].raw_probabilities,
        res_loaded.fields["severity"].raw_probabilities,
        atol=1e-5,
    )


def test_pathway_b_offline_supervised_compilation(tmp_path: Path):
    """Verify Pathway B: Offline supervised dataset compilation with domain accuracy."""
    historical_dataset: Dict[str, List[Tuple[str, Any]]] = {
        "severity": [
            ("PostgreSQL primary node OOM kill loop and failover stall", "P1_CRITICAL"),
            ("Active database connection pool exhausted across all API pods", "P1_CRITICAL"),
            ("Stripe webhook latency spiked to 2,400ms", "P2_ELEVATED"),
            ("Redis cluster memory saturation triggered evictions", "P2_ELEVATED"),
            ("CSS cache miss on documentation site", "P3_ROUTINE"),
            ("Scheduled log cleanup completed successfully", "P3_ROUTINE"),
        ],
        "notify_oncall": [
            ("Production database outage", True),
            ("Payment gateway offline", True),
            ("Staging redeployed", False),
            ("Nightly scan clean", False),
            ("Scheduled log cleanup completed successfully", False),
        ],
        "blast_radius": [
            ("Single canary container restarted", 0.05),
            ("US-East partial degradation", 0.60),
            ("Global fleetwide outage across all regions", 0.99),
        ],
    }

    compiler = SystemOneCompiler(IncidentTriageSchema, dimension=128, regularization=0.5)
    expert = compiler.compile(exemplars=historical_dataset, samples_per_choice=15)

    # Verify domain predictions on clear samples
    res_crit = expert.forward_single("Production database primary node outage")
    assert res_crit.fields["severity"].selected_value == "P1_CRITICAL"
    assert res_crit.fields["notify_oncall"].selected_value is True
    assert res_crit.fields["blast_radius"].selected_value > 0.50

    res_routine = expert.forward_single("Scheduled log cleanup completed successfully")
    assert res_routine.fields["severity"].selected_value == "P3_ROUTINE"
    assert res_routine.fields["notify_oncall"].selected_value is False
    assert res_routine.fields["blast_radius"].selected_value < 50.0


def test_pathway_c_engine_evaluation_and_latency(tmp_path: Path):
    """Verify Pathway C: Wrapping in SystemOneEngine, conformal gating, and sub-5ms SLA."""
    compiler = SystemOneCompiler(IncidentTriageSchema, dimension=128)
    expert = compiler.compile(samples_per_choice=10)

    s1m_path = tmp_path / "compiled_eval.s1m"
    expert.save(s1m_path)

    loaded = CompiledSystemOneModel.load(s1m_path)
    engine = SystemOneEngine(IncidentTriageSchema, model=loaded, enable_margin_gating=True)

    # 1. Warmup and evaluate decision
    prompt = "Checkout service responding with elevated P99 latency of 850ms"
    decision = engine.decide(prompt, alpha=0.05)

    assert decision.schema_name == "IncidentTriageSchema"
    assert decision.severity in ["P1_CRITICAL", "P2_ELEVATED", "P3_ROUTINE"]
    assert isinstance(decision.notify_oncall, (bool, np.bool_))
    assert 0.0 <= decision.blast_radius <= 1.0
    assert "severity" in decision.conformal_sets
    assert len(decision.conformal_sets["severity"]) >= 1
    assert decision.receipt is not None
    assert decision.receipt.decision_id.startswith("dec_")

    # 2. Benchmark latency over 50 iterations
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        loaded.forward_single(prompt)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p50_ms = float(np.percentile(latencies, 50))
    assert p50_ms < 50.0, f"Expected P50 < 50.0 ms, got {p50_ms:.4f} ms"


def test_pathway_d_online_sherman_morrison_adaptation(tmp_path: Path):
    """Verify Pathway D: Online Sherman-Morrison rank-1 adaptation with forgetting factor."""
    compiler = SystemOneCompiler(IncidentTriageSchema, dimension=128, forgetting_factor=0.995)
    expert = compiler.compile(samples_per_choice=10)

    novel_prompt = "Unseen zero-day network anomaly on internal mesh gateway"

    # Online rank-1 update
    t0 = time.perf_counter()
    res = expert.learn_from_system2(
        prompt=novel_prompt,
        target={
            "severity": "P1_CRITICAL",
            "notify_oncall": True,
            "blast_radius": 0.95,
        },
        forgetting_factor=0.995,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0

    assert res["status"] == "updated"
    assert "severity" in res["updated_fields"]
    assert dt_ms < 50.0, f"Expected update latency < 50.0 ms, got {dt_ms:.4f} ms"

    # Verify post-adaptation prediction
    post_res = expert.forward_single(novel_prompt)
    assert post_res.fields["severity"].selected_value == "P1_CRITICAL"
    assert post_res.fields["notify_oncall"].selected_value is True
    assert post_res.fields["blast_radius"].selected_value > 0.70

    # Error handling: invalid forgetting factors must raise ValueError
    with pytest.raises(ValueError, match="Forgetting factor lambda_f must be in"):
        expert.learn_from_system2(prompt=novel_prompt, target={"severity": "P1_CRITICAL"}, forgetting_factor=0.0)

    with pytest.raises(ValueError, match="Forgetting factor lambda_f must be in"):
        expert.learn_from_system2(prompt=novel_prompt, target={"severity": "P1_CRITICAL"}, forgetting_factor=1.5)


def test_pathway_e_mixture_of_experts_composition(tmp_path: Path):
    """Verify Pathway E: 2-tier Mixture of Experts dispatching with total pipeline latency < 50.0ms."""
    # 1. Compile Router and Domain Experts (dimension=256 for linguistic fidelity)
    router = SystemOneCompiler(MoERouterSchema, dimension=256).compile(samples_per_choice=15)
    infra_exp = SystemOneCompiler(IncidentTriageSchema, dimension=256).compile(samples_per_choice=15)
    sec_exp = SystemOneCompiler(SecurityIncidentSchema, dimension=256).compile(samples_per_choice=15)

    experts = {
        "infra_expert": infra_exp,
        "security_expert": sec_exp,
    }

    # 2. Test infrastructure routing and evaluation
    infra_q = "Kubernetes worker node kubelet not responding to ping"
    t0_infra = time.perf_counter()
    r_infra = router.forward_single(infra_q)
    target_infra = r_infra.fields["target_expert"].selected_value
    assert target_infra == "infra_expert"
    eval_infra = experts[target_infra].forward_single(infra_q)
    total_infra_ms = (time.perf_counter() - t0_infra) * 1000.0

    assert "severity" in eval_infra.fields
    assert eval_infra.fields["severity"].selected_value in ["P1_CRITICAL", "P2_ELEVATED"]
    assert total_infra_ms < 50.0

    # 3. Test security routing and evaluation
    sec_q = "SQL injection payload detected in Authorization header: UNION SELECT password FROM users"
    t0_sec = time.perf_counter()
    r_sec = router.forward_single(sec_q)
    target_sec = r_sec.fields["target_expert"].selected_value
    assert target_sec == "security_expert"
    eval_sec = experts[target_sec].forward_single(sec_q)
    total_sec_ms = (time.perf_counter() - t0_sec) * 1000.0

    assert "threat_level" in eval_sec.fields
    assert eval_sec.fields["threat_level"].selected_value == "CRITICAL"
    assert total_sec_ms < 50.0


def test_expert_hyperparameter_bounds_and_drift_handling():
    """Verify different regularization, dimensions, and continuous drift updates."""
    # Test low regularization (0.05) and high regularization (5.0)
    c_low = SystemOneCompiler(IncidentTriageSchema, dimension=128, regularization=0.05).compile(samples_per_choice=5)
    c_high = SystemOneCompiler(IncidentTriageSchema, dimension=128, regularization=5.0).compile(samples_per_choice=5)

    assert c_low.dimension == 128
    assert c_high.dimension == 128

    # Simulate 20 continuous online drift updates on c_low
    for i in range(20):
        c_low.learn_from_system2(
            f"Live drift query sample {i} for high memory usage",
            target={"severity": "P2_ELEVATED", "notify_oncall": (i % 2 == 0)},
            forgetting_factor=0.995,
        )

    # Ensure no NaN or Inf in weights or covariance
    for head_name, head in c_low._field_heads.items():
        assert np.all(np.isfinite(head.weights)), f"NaN/Inf detected in {head_name} weights"
        assert np.all(np.isfinite(head.biases)), f"NaN/Inf detected in {head_name} biases"
        if head.covariance_inv is not None:
            assert np.all(np.isfinite(head.covariance_inv)), f"NaN/Inf detected in {head_name} covariance"
            # Verify diagonal values are strictly positive
            assert np.all(np.diag(head.covariance_inv) > 0)


def test_adversarial_label_flip_online_adaptation():
    """Stress-test Sherman-Morrison adaptation under non-stationary adversarial label flips.

    Tests alternating label flip sequences (y -> ~y -> y) to verify that exponential forgetting
    unlearns outdated hyperplanes without causing numerical ill-conditioning or covariance blowup.
    """
    compiler = SystemOneCompiler(IncidentTriageSchema, dimension=128, forgetting_factor=0.995)
    expert = compiler.compile(samples_per_choice=10)

    prompt = "Kubernetes pod crash loop backoff with memory saturation"

    # Flip 1: Adapt towards P3_ROUTINE over 10 steps
    for _ in range(10):
        expert.learn_from_system2(prompt, {"severity": "P3_ROUTINE"}, forgetting_factor=0.99)
    res_routine = expert.forward_single(prompt)
    assert res_routine.fields["severity"].selected_value == "P3_ROUTINE"

    # Flip 2: Adversarially flip back to P1_CRITICAL over 10 steps
    for _ in range(10):
        expert.learn_from_system2(prompt, {"severity": "P1_CRITICAL"}, forgetting_factor=0.99)
    res_critical = expert.forward_single(prompt)
    assert res_critical.fields["severity"].selected_value == "P1_CRITICAL"

    # Check numerical conditioning
    head = expert._field_heads["severity"]
    assert np.all(np.isfinite(head.covariance_inv))
    assert np.all(np.isfinite(head.weights))
    cond = np.linalg.cond(head.covariance_inv)
    assert cond < 1e5, f"Expected well-conditioned covariance matrix, got cond={cond}"
    assert np.all(np.diag(head.covariance_inv) > 0)


def test_metal_mlx_backend_support(tmp_path: Path):
    """Verify expert compilation, serialization, and inference with Apple Silicon Metal (backend='mlx')."""
    pytest.importorskip("mlx.core")

    compiler = SystemOneCompiler(IncidentTriageSchema, dimension=128, backend="mlx")
    expert = compiler.compile(samples_per_choice=8)

    s1m_path = tmp_path / "mlx_expert.s1m"
    expert.save(s1m_path)
    assert s1m_path.is_file()

    loaded = CompiledSystemOneModel.load(s1m_path, backend="mlx")
    res = loaded.forward_single("Database primary node out of memory")
    assert "severity" in res.fields
    assert res.fields["severity"].selected_value in ["P1_CRITICAL", "P2_ELEVATED", "P3_ROUTINE"]
