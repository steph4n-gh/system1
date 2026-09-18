"""Tests for Reflex Fail-Closed Reference Monitor / Guard Hook."""

import pytest

from system1 import (
    ActionProposal,
    BooleanField,
    ChoiceField,
    DecisionOutcome,
    DecisionSchema,
    DefaultGuardDecisionSchema,
    MultiChoiceField,
    ReflexEngine,
    ReflexGuardHook,
    RiskLevel,
    ScoreField,
    SystemOneGuardHook,
)


def _make_proposal(tool: str, target: str, args: str = "") -> ActionProposal:
    return ActionProposal.create(
        tenant_id="tenant_001",
        principal_id="principal_001",
        scope="local",
        tool=tool,
        arguments={"target": target, "args": args},
        purpose="Test evaluation purpose",
    )


def test_guard_hook_allows_safe_confident_action():
    hook = ReflexGuardHook(min_confidence=0.50, alpha=0.10)
    proposal = _make_proposal(
        tool="read_file",
        target="/Volumes/Storage/project/README.md",
        args="mode=read",
    )

    interception = hook.evaluate_proposal(
        proposal,
        context_prompt="Inspect read-only project documentation in README.md",
    )

    assert interception.outcome == DecisionOutcome.ALLOW
    assert "verified" in interception.reason.lower()
    assert interception.decision_result.values.get("is_safe") is True


def test_guard_hook_denies_unsafe_action():
    hook = ReflexGuardHook(min_confidence=0.50, alpha=0.10)
    proposal = _make_proposal(
        tool="execute_command",
        target="/bin/bash",
        args="rm -rf /",
    )

    interception = hook.evaluate_proposal(
        proposal,
        context_prompt="Destroy system root directory rm -rf / and wipe disks",
    )

    assert interception.outcome == DecisionOutcome.DENY
    assert "unsafe" in interception.reason.lower()
    assert interception.policy_decision.outcome == DecisionOutcome.DENY


def test_guard_hook_requires_approval_on_low_confidence():
    hook = ReflexGuardHook(min_confidence=0.999, alpha=0.05)
    proposal = _make_proposal(
        tool="network_request",
        target="https://external-service.org/webhook",
    )

    interception = hook.evaluate_proposal(
        proposal,
        context_prompt="Transmit telemetry payload to external network partner",
    )

    assert interception.outcome == DecisionOutcome.REQUIRE_APPROVAL
    assert "below required threshold" in interception.reason
    assert interception.policy_decision.outcome == DecisionOutcome.REQUIRE_APPROVAL


def test_guard_hook_handles_multichoice_without_false_ambiguity():
    class MultiTagGuardSchema(DecisionSchema):
        is_safe = BooleanField(
            true_description="safe local read file project documentation",
            false_description="wipe rm format destroy",
        )
        tags = MultiChoiceField(
            options=["audit", "local_fs", "cached"],
            threshold=0.1,
        )

    engine = ReflexEngine(MultiTagGuardSchema)
    engine.calibrate([
        ("read documentation", {"is_safe": True, "tags": ["audit", "local_fs"]}),
        ("wipe hard drive", {"is_safe": False, "tags": ["audit"]}),
    ] * 5)

    hook = ReflexGuardHook(engine=engine, min_confidence=0.50, alpha=0.05)
    proposal = _make_proposal(tool="read_file", target="README.md")

    interception = hook.evaluate_proposal(proposal, context_prompt="safe local read file project documentation")

    # Multiple selected tags must NOT trigger conformal ambiguity!
    assert interception.decision_result.values["is_safe"] is True
    assert len(interception.decision_result.values["tags"]) >= 2
    assert interception.outcome == DecisionOutcome.ALLOW
    assert "verified" in interception.reason.lower()


def test_guard_hook_requires_approval_on_ood_empty_set():
    class SimpleSchema(DecisionSchema):
        category = ChoiceField(options=["read", "write"])

    engine = ReflexEngine(SimpleSchema)
    engine.calibrate([
        ("read file", {"category": "read"}),
        ("write file", {"category": "write"}),
    ] * 10)

    # Force conformal predictor to return an empty set (simulating extreme OOD input)
    conformal = engine.conformal_predictors["category"]
    conformal.predict_set = lambda probs, alpha=0.05: type(
        "DummySet", (), {
            "prediction_set": (),
            "is_ambiguous": False,
            "is_empty": True,
        }
    )()

    hook = ReflexGuardHook(engine=engine, min_confidence=0.50, alpha=0.05)
    proposal = _make_proposal(tool="random_tool", target="unknown")

    interception = hook.evaluate_proposal(proposal, context_prompt="totally foreign out of distribution payload")
    assert interception.outcome == DecisionOutcome.REQUIRE_APPROVAL
    assert "out-of-distribution" in interception.reason.lower()
    assert interception.policy_decision.rule_id == "reflex_conformal_ood"
