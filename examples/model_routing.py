#!/usr/bin/env python3
"""System 1 Standalone Example: Dynamic Model Gateway & Routing.

Demonstrates using System 1 as a low-latency (<2ms) front-line router to classify
prompts and route them between local small models, cloud chat models, and expensive
frontier reasoning models, with calibrated confidences and conformal prediction sets.

Showcases three integration paradigms:
1. Native System 1 Declarative Schema (DecisionSchema + SystemOneEngine)
2. TypeSafe SDK Drop-in Import Swap (from system1.compat.typesafe import TypeSafeClient, Choice, Noul, Score)
3. Zero-Code-Change Monkey Patching (patch_typesafe() -> import typesafe_sdk)
4. Live Side-by-Side Comparison vs TypeSafe Cloud (or realistic WAN baseline fallback)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure src/ is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import system1
from system1 import (
    BooleanField,
    ChoiceField,
    DecisionSchema,
    SystemOneEngine,
    ScoreField,
)
from system1.compat.typesafe import (
    Choice,
    Noul,
    Score,
    TypeSafeClient,
    patch_typesafe,
)


# ============================================================================
# 1. Native System 1 Schema
# ============================================================================

class ModelRouterSchema(DecisionSchema):
    """Schema for AI Gateway routing decisions."""

    target_tier = ChoiceField(
        options=["local_small", "standard_chat", "frontier_reasoning"],
        descriptions={
            "local_small": "Fast local on-device small model / System 1 reflex for sub-millisecond transforms and formatting",
            "standard_chat": "Standard cloud chat model for conversational queries and summaries (e.g. Gemini 3.8 Flash, GPT-5.6 Luna)",
            "frontier_reasoning": "Frontier reasoning model for advanced math, logic, and architecture (e.g. OpenAI Astra, GPT-6, GPT-5.6 Sol/Terra, Anthropic Opus 5, Fable 5.1, Mythos 5, Gemini 3.1 Pro, xAI Grok)",
        },
    )
    task_category = ChoiceField(
        options=["code_syntax", "math_reasoning", "general_qa", "creative"],
        descriptions={
            "code_syntax": "Code formatting, linting, syntax transformation",
            "math_reasoning": "Mathematical proofs, formal logic, theorem proving",
            "general_qa": "General knowledge question and answer",
            "creative": "Creative writing, email drafts, marketing copy",
        },
    )
    requires_deep_search = BooleanField(
        threshold=0.5,
        true_description="Requires deep recursive web search or literature review",
        false_description="Direct answer without external retrieval",
    )
    complexity_score = ScoreField(
        min_value=0.0,
        max_value=1.0,
        low_description="Simple mechanical conversion or trivia query",
        high_description="Extreme multi-step architectural or formal proof complexity",
    )


def run_native_system1_mode(test_queries: list[str]):
    print("\n" + "=" * 76)
    print("  MODE 1: NATIVE REFLEX DECISION ENGINE (DecisionSchema + SystemOneEngine)")
    print("=" * 76)

    engine = SystemOneEngine(ModelRouterSchema, backend="auto")

    # Calibration dataset
    calibration_data = [
        ("Convert timestamp 1711929600 to ISO 8601 in Python", {"target_tier": "local_small", "task_category": "code_syntax", "requires_deep_search": False, "complexity_score": 0.12}),
        ("Prettify this JSON payload with 2-space indentation", {"target_tier": "local_small", "task_category": "code_syntax", "requires_deep_search": False, "complexity_score": 0.08}),
        ("What is the speed of light in vacuum?", {"target_tier": "local_small", "task_category": "general_qa", "requires_deep_search": False, "complexity_score": 0.05}),
        ("Draft a warm introductory message to our new VP of Engineering", {"target_tier": "standard_chat", "task_category": "creative", "requires_deep_search": False, "complexity_score": 0.35}),
        ("Explain the trade-offs between Raft and Paxos consensus", {"target_tier": "standard_chat", "task_category": "general_qa", "requires_deep_search": False, "complexity_score": 0.60}),
        ("Prove that all non-trivial zeros of the Riemann zeta function have real part 1/2", {"target_tier": "frontier_reasoning", "task_category": "math_reasoning", "requires_deep_search": True, "complexity_score": 0.99}),
        ("Formally verify a lock-free ring buffer in TLA+ under weakly consistent memory", {"target_tier": "frontier_reasoning", "task_category": "code_syntax", "requires_deep_search": True, "complexity_score": 0.96}),
    ] * 6

    print(f"Calibrating temperature scaling & conformal sets on {len(calibration_data)} examples...")
    engine.calibrate(calibration_data)

    print("\nEvaluating Live Routing Decisions:")
    for query in test_queries:
        t0 = time.perf_counter()
        decision = engine.decide(query, alpha=0.05)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        print(f"\nPrompt: \"{query}\"")
        print(f"  Latency:             {latency_ms:.3f} ms")
        print(f"  Target Tier:         {decision.target_tier} (confidence: {decision.confidences['target_tier']:.1%})")
        print(f"  Conformal Set (95%): {decision.conformal_sets['target_tier']}")
        print(f"  Task Category:       {decision.task_category} (confidence: {decision.confidences['task_category']:.1%})")
        print(f"  Requires Deep Search:{decision.requires_deep_search}")
        print(f"  Complexity Score:    {decision.complexity_score:.3f}")
        print(f"  Receipt ID:          {decision.receipt.receipt_id}")


# ============================================================================
# 2. TypeSafe SDK Drop-in Import Swap
# ============================================================================

def run_typesafe_sdk_dropin_mode(test_queries: list[str]):
    print("\n" + "=" * 76)
    print("  MODE 2: TYPESAFE SDK DROP-IN IMPORT SWAP")
    print("  (from system1.compat.typesafe import TypeSafeClient, Choice, Noul, Score)")
    print("=" * 76)

    # Define questions matching TypeSafe AI's exact SDK syntax
    typesafe_questions = {
        "target_tier": Choice(
            "Which execution tier should handle this query?",
            criteria={
                "local_small": "Fast local small model for simple transforms and formatting",
                "standard_chat": "Standard cloud chat model for conversational queries and summaries",
                "frontier_reasoning": "Frontier reasoning model for advanced math, logic, and architecture",
            },
        ),
        "task_category": Choice(
            "What is the category of the task?",
            criteria={
                "code_syntax": "Code formatting, linting, syntax transformation",
                "math_reasoning": "Mathematical proofs, formal logic, theorem proving",
                "general_qa": "General knowledge question and answer",
                "creative": "Creative writing, email drafts, marketing copy",
            },
        ),
        "requires_deep_search": Noul(
            "Requires deep recursive web search or literature review?",
            criteria={"true": "Requires deep external search", "false": "Direct answer without retrieval"},
        ),
        "complexity_score": Score(
            "Rate query complexity from 0 (trivial) to 1 (frontier proof)",
            min_value=0.0,
            max_value=1.0,
        ),
    }

    client = TypeSafeClient()

    print("\nEvaluating via TypeSafeClient.systemone():")
    for query in test_queries:
        t0 = time.perf_counter()
        resp = client.systemone(query, typesafe_questions, alpha=0.05)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        print(f"\nPrompt: \"{query}\"")
        print(f"  Latency:             {latency_ms:.3f} ms (TypeSafe client on-device)")
        print(f"  Target Tier:         {resp.answers.target_tier.choice} (conf: {resp.answers.target_tier.confidence:.1%})")
        print(f"  Conformal Set (95%): {resp.answers.target_tier.conformal_set}")
        print(f"  Task Category:       {resp.answers.task_category.choice}")
        print(f"  Requires Deep Search:{resp.answers.requires_deep_search.value} (p={resp.answers.requires_deep_search.noul:.2f})")
        print(f"  Complexity Score:    {resp.answers.complexity_score.score:.3f}")
        print(f"  Receipt ID:          {resp.receipt.receipt_id}")


# ============================================================================
# 3. Monkey Patching Mode (Zero-Code-Change for existing typesafe_sdk code)
# ============================================================================

def run_monkey_patch_mode(test_queries: list[str]):
    print("\n" + "=" * 76)
    print("  MODE 3: ZERO-CODE-CHANGE MONKEY PATCHING (patch_typesafe())")
    print("=" * 76)

    with patch_typesafe():
        import typesafe_sdk
        from typesafe_sdk import Choice as JevChoice, Noul as JevNoul, Score as JevScore, TypeSafeClient as JevClient

        print(f"Imported typesafe_sdk successfully: {typesafe_sdk.__name__}")
        client = JevClient()

        questions = {
            "tier": JevChoice("Target tier", criteria=["local_small", "standard_chat", "frontier_reasoning"]),
            "hard": JevNoul("Is the question hard?"),
            "score": JevScore("Complexity", min_value=0.0, max_value=3.0),
        }

        for query in test_queries[:2]:
            res = client.systemone(query, questions)
            print(f"  • Prompt: \"{query[:45]}...\"")
            print(f"    -> Tier: {res.answers.tier.choice}, Hard: {res.answers.hard.value}, Latency: {res.latency_ms:.3f} ms")


# ============================================================================
# 4. Live Side-by-Side Comparison (Local vs TypeSafe Cloud / Fallback)
# ============================================================================

def run_side_by_side_comparison(test_queries: list[str]):
    print("\n" + "=" * 76)
    print("  MODE 4: LIVE HEAD-TO-HEAD COMPARISON (System 1 System 1 vs TypeSafe Cloud)")
    print("=" * 76)

    api_key = os.environ.get("TYPESAFE_API_KEY", "") or os.environ.get("JEV_API_KEY", "")
    if api_key:
        print(f"Targeting real TypeSafe API with key: [***{api_key[-4:]}]")
    else:
        print("No TYPESAFE_API_KEY found. Measuring WAN socket RTT and realistic baseline profile.")

    client = TypeSafeClient(api_key=api_key, zero_egress=False)

    questions = {
        "target_tier": Choice("Model tier", criteria=["local_small", "standard_chat", "frontier_reasoning"]),
        "needs_deep": Noul("Needs deep reasoning?"),
        "complexity": Score("Complexity", min_value=0.0, max_value=1.0),
    }

    print("\n" + "-" * 92)
    print(f"{'Query':<42} | {'Local (ms)':<12} | {'Cloud (ms)':<12} | {'Speedup':<10} | {'Egress Diff'}")
    print("-" * 92)

    for q in test_queries:
        comp = client.compare(q, questions)
        short_q = q[:40] + ".." if len(q) > 42 else q
        print(
            f"{short_q:<42} | "
            f"{comp.local_latency_ms:>8.3f} ms | "
            f"{comp.cloud_latency_ms:>8.2f} ms | "
            f"{comp.speedup_factor:>7.1f}x  | "
            f"0 B vs {comp.cloud_egress_bytes} B"
        )
    print("-" * 92)


def main():
    test_queries = [
        "Fix my syntax error: missing colon at end of def statement",
        "How do transformers handle positional encodings compared to RoPE?",
        "Prove Godel's incompleteness theorems from first principles using Peano arithmetic",
    ]

    print("#" * 76)
    print("  REFLEX SYSTEM 1: DYNAMIC MODEL GATEWAY & ROUTING SHOWCASE")
    print("#" * 76)

    run_native_system1_mode(test_queries)
    run_typesafe_sdk_dropin_mode(test_queries)
    run_monkey_patch_mode(test_queries)
    run_side_by_side_comparison(test_queries)

    print("\n" + "=" * 76)
    print("  MODEL ROUTING DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
