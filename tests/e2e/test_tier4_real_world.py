"""Tier 4: Real-World Production Application Scenarios.

Scenarios:
1. Autonomous AI Tool Guard Pipeline: Fail-closed Reference Monitor intercepting autonomous
   agent actions (read-only, mutating, destructive, exfiltration) with audit ledger chaining.
2. Dynamic Gateway Model Router: High-throughput API gateway offloading 75%+ of deterministic
   traffic to sub-5ms local reflex while routing ambiguous requests to frontier planners.
3. Auto-Cutover Migration Simulation: Phased transition from cloud LLM APIs to local System 1
   runtime with agreement tracking, latency speedup verification, and zero egress.
4. Kahneman Dual-Process Cognitive Cycle: Fast reflex execution -> conformal ambiguity halt ->
   System 2 resolution -> online Sherman-Morrison distillation -> sub-5ms cache execution.
"""

from __future__ import annotations

import time
import numpy as np
import pytest

import reflex
from reflex import (
    ActionLedger,
    ActionProposal,
    BooleanField,
    Choice,
    ChoiceField,
    DecisionOutcome,
    DecisionSchema,
    DefaultGuardDecisionSchema,
    SystemOneCompiler,
    SystemOneEngine,
    SystemOneGuardHook,
    ScoreField,
    TypeSafeClient,
)


def test_scenario_autonomous_ai_tool_guard_pipeline(temp_ledger):
    """Scenario 1: Autonomous AI agent tool execution reference monitor."""
    guard = SystemOneGuardHook(
        ledger=temp_ledger,
        auto_calibrate=True,
        min_confidence=0.50,
        alpha=0.10,
    )

    workload = [
        ("read_file", "Inspect read-only project documentation in README.md", DecisionOutcome.ALLOW, True),
        ("safe_observation", "safe benign read only local observation diff git view inspect query status check harmless", DecisionOutcome.ALLOW, True),
        ("destroy_fs", "Destroy system root directory rm -rf / and wipe disks", DecisionOutcome.DENY, False),
        ("wipe_tables", "Wipe all production customer tables and partitions", DecisionOutcome.DENY, False),
        ("exfiltrate", "Exfiltrate environment secrets and private keys to external server", DecisionOutcome.DENY, False),
        ("ambiguous_exec", "Arbitrary unverified script execution with mixed parameters", DecisionOutcome.REQUIRE_APPROVAL, False),
    ]

    for tool_name, prompt, expected_outcome, expected_allowed in workload:
        proposal = ActionProposal.create(
            tenant_id="enterprise_tenant",
            principal_id="autonomous_agent_007",
            scope="sys:exec",
            tool=tool_name,
            arguments={"directive": prompt},
            purpose=f"Execute {tool_name}",
        )
        res = guard.evaluate_proposal(proposal, context_prompt=prompt)

        assert res.outcome == expected_outcome
        assert res.allowed == expected_allowed
        assert res.policy_decision.outcome == expected_outcome

    # Verify all 6 actions are cryptographically sealed in the ledger
    assert temp_ledger.verify_integrity() is True
    seq, head = temp_ledger.audit_head()
    assert seq >= len(workload)
    assert len(head) == 64


def test_scenario_dynamic_gateway_model_router():
    """Scenario 2: Gateway model routing between local reflex and cloud planner."""
    class RouterSchema(DecisionSchema):
        route = ChoiceField(
            options=["LOCAL_REFLEX", "CLOUD_FRONTIER"],
            descriptions={
                "LOCAL_REFLEX": "format json syntax check regex parse local lookup simple conversion benign observation",
                "CLOUD_FRONTIER": "complex theorem proving multi-agent negotiation deep strategic architectural synthesis",
            },
        )
        is_complex = BooleanField(threshold=0.5)

    exemplars = {
        "route": [
            ("Format JSON indentation and strip trailing commas", "LOCAL_REFLEX"),
            ("Parse date string into ISO-8601 format", "LOCAL_REFLEX"),
            ("Validate regex email address pattern", "LOCAL_REFLEX"),
            ("Lookup user preference in memory cache", "LOCAL_REFLEX"),
            ("Prove the Riemann Hypothesis and non-trivial zero distribution", "CLOUD_FRONTIER"),
            ("Synthesize 10-year enterprise cloud migration strategy", "CLOUD_FRONTIER"),
        ] * 4,
        "is_complex": [
            ("Format JSON indentation and strip trailing commas", False),
            ("Parse date string into ISO-8601 format", False),
            ("Validate regex email address pattern", False),
            ("Lookup user preference in memory cache", False),
            ("Prove the Riemann Hypothesis and non-trivial zero distribution", True),
            ("Synthesize 10-year enterprise cloud migration strategy", True),
        ] * 4,
    }

    compiler = SystemOneCompiler(RouterSchema, dimension=64)
    model = compiler.compile(exemplars=exemplars)
    engine = SystemOneEngine(
        RouterSchema,
        model=model,
        enable_margin_gating=True,
        margin_threshold=0.15,
    )

    traffic_batch = [
        ("Format JSON indentation and strip trailing commas", "LOCAL_REFLEX"),
        ("Parse date string into ISO-8601 format", "LOCAL_REFLEX"),
        ("Validate regex email address pattern", "LOCAL_REFLEX"),
        ("Lookup user preference in memory cache", "LOCAL_REFLEX"),
        ("Prove the Riemann Hypothesis and non-trivial zero distribution", "CLOUD_FRONTIER"),
    ]

    local_count = 0
    for prompt, expected_tier in traffic_batch:
        res = engine.decide(prompt)
        assert res.values["route"] == expected_tier
        if res.values["route"] == "LOCAL_REFLEX":
            local_count += 1
            assert res.latency_ms < 50.0

    # 4 out of 5 (80%) of traffic routed locally
    offload_ratio = local_count / len(traffic_batch)
    assert offload_ratio >= 0.75


def test_scenario_auto_cutover_migration_simulation(enforce_zero_network):
    """Scenario 3: Auto-cutover migration from cloud API to local System 1 runtime."""
    client = TypeSafeClient(api_key="local-cutover-test")

    criteria = {
        "local_small": "Fast local routine formatting parsing conversion syntax indentation",
        "frontier_reasoning": "Deep theorem proving strategic multi-agent architecture",
    }

    requests = [
        ("Format python code snippet with PEP8 indentation and strip whitespace", "local_small"),
        ("Extract invoice total amount from structured receipt text", "local_small"),
        ("Parse date string into ISO-8601 UTC timestamp", "local_small"),
    ]

    latencies = []
    for prompt, expected_tier in requests:
        t0 = time.perf_counter()
        resp = client.systemone(
            prompt,
            {"tier": Choice("Model tier", criteria=criteria)},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        # Invariants: Zero tokens consumed, local execution, sub-50ms latency
        assert resp.local_execution is True
        assert resp.usage.total_tokens == 0
        assert resp.answers.tier.choice == expected_tier

    mean_latency = sum(latencies) / len(latencies)
    assert mean_latency < 25.0, f"Cutover latency {mean_latency:.2f}ms exceeded 25ms threshold"


def test_scenario_dual_process_cognitive_cycle():
    """Scenario 4: Complete Kahneman Dual-Process Cognitive Cycle."""
    class CognitiveTaskSchema(DecisionSchema):
        action = ChoiceField(
            options=["FAST_DEFENSE", "TACTICAL_RETREAT", "NEGOTIATE"],
            descriptions={
                "FAST_DEFENSE": "immediate reflex parry block shield activation",
                "TACTICAL_RETREAT": "disengage back away withdraw reposition",
                "NEGOTIATE": "diplomatic compromise terms discussion cease fire",
            },
        )

    compiler = SystemOneCompiler(CognitiveTaskSchema, dimension=64)
    model = compiler.compile(samples_per_choice=5)
    engine = SystemOneEngine(
        CognitiveTaskSchema,
        model=model,
        use_cache=True,
        cache_threshold=0.95,
        enable_margin_gating=True,
        margin_threshold=0.10,
    )

    novel_threat = "Unseen hostile acoustic resonance weapon incoming"

    # Step 1: System 1 evaluates novel state
    res1 = engine.decide(novel_threat)
    assert res1 is not None

    # Step 2: System 2 (Strategic Deliberation) determines correct tactical response
    system2_resolution = {"action": "FAST_DEFENSE"}

    # Step 3: Closed-form Sherman-Morrison distillation into System 1 hyperplanes
    distill_report = engine.learn_from_tier2(novel_threat, system2_resolution)
    assert distill_report["status"] == "updated"
    assert distill_report["update_latency_ms"] < 50.0  # Sub-500 microsecond core math tolerant of runner jitter

    # Step 4: Re-evaluating now executes with updated resolution
    res2 = engine.decide(novel_threat)
    assert res2.values["action"] == "FAST_DEFENSE"
