#!/usr/bin/env python3
"""Reflex Standalone Example: Dynamic Model Gateway & Routing.

Demonstrates using Reflex as a low-latency (<2ms) front-line router to classify
prompts and route them between local small models, cloud chat models, and expensive
frontier reasoning models, with calibrated confidences and conformal prediction sets.
"""

import time
from reflex import (
    BooleanField,
    ChoiceField,
    DecisionSchema,
    ReflexEngine,
    ScoreField,
)


class ModelRouterSchema(DecisionSchema):
    """Schema for AI Gateway routing decisions."""

    target_tier = ChoiceField(
        options=["local_small", "standard_chat", "frontier_reasoning"],
        descriptions={
            "local_small": "Fast local on-device small model for simple transforms and formatting",
            "standard_chat": "Standard cloud chat model for conversational queries and summaries",
            "frontier_reasoning": "Frontier reasoning model for advanced math, logic, and architecture",
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


def main():
    print("Initializing Reflex Model Gateway Router...")
    engine = ReflexEngine(ModelRouterSchema, backend="auto")

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

    test_queries = [
        "Fix my syntax error: missing colon at end of def statement",
        "How do transformers handle positional encodings compared to RoPE?",
        "Prove Godel's incompleteness theorems from first principles using Peano arithmetic",
    ]

    print("\nEvaluating Live Routing Decisions:")
    print("-" * 70)

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


if __name__ == "__main__":
    main()
