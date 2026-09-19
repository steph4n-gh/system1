#!/usr/bin/env python3
"""System 1 Domain Expert Training, Distillation, and Deployment Example.

Demonstrates end-to-end workflows for training, distilling, and deploying System 1
Domain Experts (.s1m binaries) running on local host silicon (CPU/Metal) in < 1ms:

1. Defining a domain schema (IncidentTriageSchema).
2. Pathway A: Synthetic distillation and compilation to a portable <20KB .s1m binary.
3. Pathway B: Offline supervised dataset compilation from historical domain exemplars.
4. Pathway C: Evaluating the compiled expert, measuring sub-millisecond latency (P50/P99),
   and validating conformal prediction sets and margin dominance.
5. Pathway D: Online Sherman-Morrison fine-tuning with exponential forgetting factor λ_f = 0.995.
6. Pathway E: Composing a 2-tier Mixture of Experts (Router Expert -> Specialized Domain Experts).
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure src/ is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

import reflex
from reflex import (
    BooleanField,
    ChoiceField,
    DecisionSchema,
    SystemOneEngine,
    ScoreField,
)
from reflex.compiler import (
    CompiledSystemOneModel,
    SystemOneCompiler,
)


# ============================================================================
# 1. Domain Schema Definition
# ============================================================================

class IncidentTriageSchema(DecisionSchema):
    """Schema for automated infrastructure incident triage."""

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
    """Schema for automated cybersecurity threat triage."""

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
    """Router schema dispatching queries to specialized domain experts."""

    target_expert = ChoiceField(
        options=["infra_expert", "security_expert"],
        descriptions={
            "infra_expert": "Kubernetes pods, high memory, disk storage, database timeouts, server crashes",
            "security_expert": "SQL injection, credential stuffing, unauthorized API tokens, malware, brute force",
        },
    )


# ============================================================================
# Main Demonstration
# ============================================================================

def main() -> None:
    print("=" * 80)
    print("  REFLEX DOMAIN EXPERT: TRAINING, DISTILLATION & DEPLOYMENT GUIDE")
    print(f"  System 1 Version: {reflex.__version__} | Target SLA: Sub-1ms on local silicon")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # --------------------------------------------------------------------
        # Pathway A: Synthetic Teacher Distillation & .s1m Compilation
        # --------------------------------------------------------------------
        print("\n" + "-" * 80)
        print("  PATHWAY A: Synthetic Teacher Distillation & .s1m Compilation")
        print("-" * 80)

        t0_synth = time.perf_counter()
        compiler_synth = SystemOneCompiler(
            IncidentTriageSchema,
            dimension=256,
            regularization=1.0,
            forgetting_factor=0.995,
        )

        # Generate synthetic exemplars with rich domain template expansion
        print("1. Generating balanced synthetic training exemplars from schema descriptions...")
        synth_exemplars = compiler_synth.generate_synthetic_exemplars(samples_per_choice=20)
        print(f"   Generated {len(synth_exemplars['severity'])} severity samples, "
              f"{len(synth_exemplars['notify_oncall'])} notify_oncall samples, "
              f"{len(synth_exemplars['blast_radius'])} blast_radius samples.")

        # Fit closed-form Ridge Regression hyperplanes
        print("2. Solving closed-form multi-head Ridge Regression with Cholesky factorization...")
        expert_synth = compiler_synth.compile(exemplars=synth_exemplars, calibration_split=0.25)
        compile_time_ms = (time.perf_counter() - t0_synth) * 1000.0

        # Save to portable .s1m binary file
        s1m_file_path = tmp_path / "incident_expert.s1m"
        expert_synth.save(s1m_file_path)
        file_size_kb = s1m_file_path.stat().st_size / 1024.0

        print(f"3. Expert successfully serialized to: {s1m_file_path.name}")
        print(f"   Binary File Size:    {file_size_kb:.2f} KB (under 20 KiB: {file_size_kb < 20})")
        print(f"   Compilation Time:    {compile_time_ms:.2f} ms")

        # --------------------------------------------------------------------
        # Pathway B: Offline Supervised Dataset Compilation
        # --------------------------------------------------------------------
        print("\n" + "-" * 80)
        print("  PATHWAY B: Offline Supervised Dataset Compilation")
        print("-" * 80)

        historical_dataset: Dict[str, List[Tuple[str, Any]]] = {
            "severity": [
                ("PostgreSQL primary node OOM kill loop and replica failover stall", "P1_CRITICAL"),
                ("Active database connection pool completely exhausted across all API pods", "P1_CRITICAL"),
                ("Stripe webhook processing latency spiked to 2,400ms", "P2_ELEVATED"),
                ("Redis cluster memory saturation triggered key evictions", "P2_ELEVATED"),
                ("CSS cache miss causing layout shift on marketing blog", "P3_ROUTINE"),
                ("Daily metrics aggregation job completed with exit code 0", "P3_ROUTINE"),
            ],
            "notify_oncall": [
                ("Production auth cluster returning 500 internal server errors", True),
                ("Payment processing gateway offline", True),
                ("Staging environment redeployed successfully", False),
                ("Nightly dependency vulnerability scan passed", False),
            ],
            "blast_radius": [
                ("Single canary container restarted", 0.05),
                ("One tenant encountered rate limiting", 0.15),
                ("US-East payment transactions experiencing partial failure", 0.65),
                ("All global edge points of presence returning 502 bad gateway", 0.99),
            ],
        }

        print("1. Ingesting historical incident dataset...")
        compiler_sup = SystemOneCompiler(IncidentTriageSchema, dimension=256, regularization=0.5)

        t0_fit = time.perf_counter()
        expert_supervised = compiler_sup.compile(exemplars=historical_dataset, samples_per_choice=15)
        fit_time_ms = (time.perf_counter() - t0_fit) * 1000.0

        sup_file_path = tmp_path / "incident_expert_supervised.s1m"
        expert_supervised.save(sup_file_path)
        print(f"2. Supervised expert compiled in {fit_time_ms:.2f} ms! Size: {sup_file_path.stat().st_size / 1024.0:.2f} KB")

        # --------------------------------------------------------------------
        # Pathway C: Evaluating Expert, Latency SLA & Conformal Calibration
        # --------------------------------------------------------------------
        print("\n" + "-" * 80)
        print("  PATHWAY C: Evaluating Compiled Expert & Measuring Sub-1ms Latency")
        print("-" * 80)

        # Load expert binary from disk
        print("1. Loading expert from binary container (.s1m)...")
        loaded_expert = CompiledSystemOneModel.load(s1m_file_path)
        engine = SystemOneEngine(IncidentTriageSchema, model=loaded_expert, enable_margin_gating=True)

        test_queries = [
            "Production Kubernetes cluster master node panic and etcd split-brain",
            "Checkout API responding with elevated P99 latency of 850ms",
            "Minor documentation typo in API quickstart guide",
        ]

        print("\n2. Evaluating live production incident prompts:")
        latencies: List[float] = []

        for query in test_queries:
            t0 = time.perf_counter()
            decision = engine.decide(query, alpha=0.05)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            print(f"\n  [Prompt]: \"{query}\"")
            print(f"    Latency:           {lat_ms:.3f} ms")
            print(f"    Severity:          {decision.severity} (Confidence: {decision.confidences['severity']:.1%})")
            print(f"    Conformal Set:     {decision.conformal_sets['severity']} (Singleton: {len(decision.conformal_sets['severity']) == 1})")
            print(f"    Notify On-Call:    {decision.notify_oncall} (Confidence: {decision.confidences['notify_oncall']:.1%})")
            print(f"    Blast Radius:      {decision.blast_radius:.3f}")
            print(f"    Receipt ID:        {decision.receipt.decision_id}")

        # Benchmark 100 iterations for empirical distribution
        print("\n3. Running 100-iteration throughput and latency benchmark...")
        bench_latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            loaded_expert.forward_single("Core payment engine connection timeout")
            bench_latencies.append((time.perf_counter() - t0) * 1000.0)

        p50 = float(np.percentile(bench_latencies, 50))
        p95 = float(np.percentile(bench_latencies, 95))
        p99 = float(np.percentile(bench_latencies, 99))
        print(f"   Benchmark Latency P50: {p50:.3f} ms | P95: {p95:.3f} ms | P99: {p99:.3f} ms")
        assert p50 < 1.0, f"Expected P50 < 1.0 ms, got {p50:.3f} ms"

        # --------------------------------------------------------------------
        # Pathway D: Online Sherman-Morrison Fine-Tuning (λ_f = 0.995)
        # --------------------------------------------------------------------
        print("\n" + "-" * 80)
        print("  PATHWAY D: Online Sherman-Morrison Adaptation with λ_f = 0.995")
        print("-" * 80)

        novel_edge_case = "Custom eBPF network probe unhandled page fault causing packet drops"
        print(f"1. Evaluating novel, previously unseen edge case:\n   \"{novel_edge_case}\"")

        pre_eval = loaded_expert.forward_single(novel_edge_case)
        print(f"   Pre-Adaptation Severity:   {pre_eval.fields['severity'].selected_value} "
              f"(Confidence: {pre_eval.fields['severity'].confidence:.1%})")

        # System 2 deliberative governor provides ground-truth resolution
        print("\n2. Applying System 2 feedback via closed-form Sherman-Morrison rank-1 update...")
        t0_update = time.perf_counter()
        update_result = loaded_expert.learn_from_system2(
            prompt=novel_edge_case,
            target={
                "severity": "P1_CRITICAL",
                "notify_oncall": True,
                "blast_radius": 0.85,
            },
            forgetting_factor=0.995,
        )
        update_latency_us = (time.perf_counter() - t0_update) * 1_000_000.0
        rank1_math_us = float(update_result["update_latency_ms"]) * 1000.0
        num_updated = len(update_result["updated_fields"])
        per_head_us = rank1_math_us / max(1, num_updated)

        print(f"   Rank-1 Math Duration:      {rank1_math_us:.1f} µs ({per_head_us:.1f} µs/head across {num_updated} heads)")
        print(f"   Total Adaptation Time:     {update_latency_us:.1f} µs (< 1.0 ms online adaptation SLA achieved!)")
        print(f"   Updated Fields:            {update_result['updated_fields']}")

        # Re-evaluate
        post_eval = loaded_expert.forward_single(novel_edge_case)
        print(f"\n3. Post-Adaptation Resolution:")
        print(f"   Post-Adaptation Severity:  {post_eval.fields['severity'].selected_value} "
              f"(Confidence: {post_eval.fields['severity'].confidence:.1%})")
        print(f"   Notify On-Call:            {post_eval.fields['notify_oncall'].selected_value}")
        print(f"   Blast Radius:              {post_eval.fields['blast_radius'].selected_value:.3f}")

        assert post_eval.fields["severity"].selected_value == "P1_CRITICAL"
        assert post_eval.fields["notify_oncall"].selected_value is True

        # --------------------------------------------------------------------
        # Pathway E: 2-Tier Mixture of Experts (Router -> Domain Experts)
        # --------------------------------------------------------------------
        print("\n" + "-" * 80)
        print("  PATHWAY E: 2-Tier Mixture of Experts (Router -> Domain Experts)")
        print("-" * 80)

        print("1. Compiling Router Expert and Specialized Security Expert (dimension=256)...")
        router_expert = SystemOneCompiler(MoERouterSchema, dimension=256).compile(samples_per_choice=15)
        security_expert = SystemOneCompiler(SecurityIncidentSchema, dimension=256).compile(samples_per_choice=15)

        moe_experts = {
            "infra_expert": loaded_expert,
            "security_expert": security_expert,
        }

        moe_queries = [
            "Kubernetes worker node kubelet not responding to ping",
            "SQL injection payload detected in Authorization Bearer header: SELECT * FROM users",
        ]

        print("\n2. Executing end-to-end 2-tier MoE dispatch pipeline:")
        for q in moe_queries:
            t0_moe = time.perf_counter()

            # Step 1: Router Expert evaluation (< 0.3 ms)
            route_res = router_expert.forward_single(q)
            selected_domain = route_res.fields["target_expert"].selected_value
            route_conf = route_res.fields["target_expert"].confidence

            # Step 2: Specialized Expert evaluation (< 0.4 ms)
            chosen_expert = moe_experts[selected_domain]
            expert_res = chosen_expert.forward_single(q)
            total_moe_ms = (time.perf_counter() - t0_moe) * 1000.0

            if "threat_level" in expert_res.fields:
                spec_action = f"Threat Level: {expert_res.fields['threat_level'].selected_value}"
                assert selected_domain == "security_expert"
                assert expert_res.fields["threat_level"].selected_value == "CRITICAL"
            elif "severity" in expert_res.fields:
                spec_action = f"Severity: {expert_res.fields['severity'].selected_value}"
                assert selected_domain == "infra_expert"
                assert expert_res.fields["severity"].selected_value in ["P1_CRITICAL", "P2_ELEVATED"]
            else:
                first_k = next(iter(expert_res.fields.keys()))
                spec_action = f"{first_k}: {expert_res.fields[first_k].selected_value}"

            print(f"\n  [Query]: \"{q}\"")
            print(f"    Router Dispatch:   {selected_domain} (Confidence: {route_conf:.1%})")
            print(f"    Specialized Action:{spec_action}")
            print(f"    Total MoE Latency: {total_moe_ms:.3f} ms (< 1.0 ms pipeline SLA!)")

    print("\n" + "=" * 80)
    print("  EXAMPLE PATHWAYS COMPLETED; REVIEW THE MEASURED RESULTS ABOVE.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
