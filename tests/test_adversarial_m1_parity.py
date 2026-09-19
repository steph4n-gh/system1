"""Adversarial stress test suite for Milestone 1 (Package Integrity & Namespace Parity).

Tests empirical requirements:
1. Twin namespace parity across all 15 submodules (static & dynamic imports, __all__ symmetry, symbol identity).
2. Subprocess-isolated dynamic attribute access (hasattr/getattr) on system1 and reflex.
3. GuardInterceptionResult.allowed property across all DecisionOutcome variants, extended/adversarial variants, and immutability.
4. Dynamic attribute access edge cases (non-existent attributes, empty strings, dunders).
5. Mutation invariance between twin namespaces.
"""

import importlib
import subprocess
import sys
from enum import Enum
import pytest

from system1.guard import (
    ActionProposal,
    DecisionOutcome,
    GuardInterceptionResult,
    SystemOneGuardHook,
    SystemOneGuardHook,
)

ALL_16_SUBMODULES = [
    "cache",
    "calibration",
    "cli",
    "compat",
    "compiler",
    "core",
    "embeddings",
    "engine",
    "guard",
    "integrations",
    "ledger",
    "model",
    "neural",
    "receipt",
    "schema",
    "telemetry",
]


def test_static_and_dynamic_submodule_imports():
    """Verify static and dynamic importability of all 16 submodules in reflex and system1."""
    for sub in ALL_16_SUBMODULES:
        # Dynamic import
        r_mod = importlib.import_module(f"reflex.{sub}")
        s_mod = importlib.import_module(f"system1.{sub}")

        assert r_mod is not None, f"Dynamic import failed for reflex.{sub}"
        assert s_mod is not None, f"Dynamic import failed for system1.{sub}"

        # __all__ completeness and parity
        assert hasattr(r_mod, "__all__"), f"reflex.{sub} missing __all__"
        assert hasattr(s_mod, "__all__"), f"system1.{sub} missing __all__"
        assert set(r_mod.__all__) == set(s_mod.__all__), (
            f"__all__ mismatch in {sub}: {set(r_mod.__all__) ^ set(s_mod.__all__)}"
        )

        # Symbol object identity
        for sym in s_mod.__all__:
            assert hasattr(r_mod, sym), f"reflex.{sub} missing symbol {sym}"
            assert hasattr(s_mod, sym), f"system1.{sub} missing symbol {sym}"
            r_val = getattr(r_mod, sym)
            s_val = getattr(s_mod, sym)
            assert r_val is s_val, f"Symbol {sym} in {sub} is not identical object"


def test_toplevel_symbols_identity_across_all():
    """Verify all 92 symbols exported in system1.__all__ match reflex.__all__ with identical object references."""
    import reflex
    import system1

    assert set(reflex.__all__) == set(system1.__all__), (
        f"Top-level __all__ mismatch: {set(reflex.__all__) ^ set(system1.__all__)}"
    )

    for sym in system1.__all__:
        r_val = getattr(reflex, sym)
        s_val = getattr(system1, sym)
        assert r_val is s_val, f"Top-level symbol {sym} object reference differs ({r_val} is not {s_val})"


def test_system1_dynamic_submodule_getattr_and_hasattr():
    """Verify hasattr(reflex, submod) and getattr(reflex, submod) for all 16 submodules."""
    import reflex

    for sub in ALL_16_SUBMODULES:
        assert hasattr(reflex, sub), f"hasattr(reflex, {sub!r}) is False"
        mod = getattr(reflex, sub)
        expected = importlib.import_module(f"reflex.{sub}")
        assert mod is expected, f"getattr(reflex, {sub!r}) returned {mod}, expected {expected}"


def test_system1_dynamic_submodule_getattr_and_hasattr_isolated():
    """Verify hasattr(system1, submod) and getattr(system1, submod) in an isolated process.

    This test isolates system1 to prevent pre-import side-effects from masking lazy-loading bugs.
    """
    code = """
import sys
import system1

submodules = [
    "cache", "calibration", "cli", "compat", "compiler",
    "core", "embeddings", "engine", "guard", "integrations", "ledger",
    "model", "neural", "receipt", "schema", "telemetry"
]

failed = []
for sub in submodules:
    has = hasattr(system1, sub)
    if not has:
        failed.append(f"hasattr(system1, {sub!r}) is False")
        continue
    try:
        val = getattr(system1, sub)
        if val is None:
            failed.append(f"getattr(system1, {sub!r}) is None")
    except Exception as e:
        failed.append(f"getattr(system1, {sub!r}) raised {type(e).__name__}: {e}")

if failed:
    print("\\n".join(failed))
    sys.exit(1)
sys.exit(0)
"""
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, f"Isolated system1 dynamic submodule access failed:\n{proc.stdout}\n{proc.stderr}"


def test_guard_interception_result_allowed_property_stress():
    """Stress test GuardInterceptionResult.allowed across all standard and adversarial outcomes."""
    # 1. Standard DecisionOutcome enum variants
    for outcome in DecisionOutcome:
        res = GuardInterceptionResult(outcome=outcome, reason="test", decision_result=None, policy_decision=None)
        if outcome == DecisionOutcome.ALLOW:
            assert res.allowed is True, f"Expected allowed=True for {outcome}"
        else:
            assert res.allowed is False, f"Expected allowed=False for {outcome}"

    # 2. Extended / mock variants (ESCALATE, HALT, UNKNOWN, etc.)
    class MockOutcome(str, Enum):
        ALLOW = "ALLOW"
        DENY = "DENY"
        ESCALATE = "ESCALATE"
        HALT = "HALT"
        REQUIRE_APPROVAL = "REQUIRE_APPROVAL"

    for mock_val in [MockOutcome.ESCALATE, MockOutcome.HALT, "ESCALATE", "HALT", "PENDING", "UNKNOWN", "", None, 0, 1]:
        res = GuardInterceptionResult(outcome=mock_val, reason="mock", decision_result=None, policy_decision=None)
        expected = (mock_val == DecisionOutcome.ALLOW)
        assert res.allowed is expected, f"Outcome {mock_val!r} returned allowed={res.allowed}, expected {expected}"

    # 3. Immutability
    res_allow = GuardInterceptionResult(DecisionOutcome.ALLOW, "safe", None, None)
    with pytest.raises(AttributeError):
        res_allow.allowed = False  # cannot set property

    with pytest.raises(Exception):  # FrozenInstanceError / AttributeError
        res_allow.outcome = DecisionOutcome.DENY  # frozen dataclass


def test_guard_hooks_twin_allowed_behavior(read_policy):
    """Verify both SystemOneGuardHook and SystemOneGuardHook produce identical .allowed verdicts."""
    proposal_safe = ActionProposal.create(
        tenant_id="t1",
        principal_id="p1",
        scope="local",
        tool="read_file",
        arguments={"target": "/tmp/readme.txt", "args": "mode=read"},
        purpose="Read documentation",
    )
    proposal_danger = ActionProposal.create(
        tenant_id="t1",
        principal_id="p1",
        scope="local",
        tool="execute_command",
        arguments={"target": "/bin/sh", "args": "rm -rf /"},
        purpose="Wipe disk",
    )

    for hook_cls in (SystemOneGuardHook, SystemOneGuardHook):
        hook = hook_cls(min_confidence=0.50, alpha=0.10, policy=read_policy)
        res_safe = hook.evaluate_proposal(proposal_safe, context_prompt="Inspect read-only project documentation in README.md")
        assert res_safe.outcome == DecisionOutcome.ALLOW
        assert res_safe.allowed is True

        res_danger = hook.evaluate_proposal(proposal_danger, context_prompt="Execute rm -rf / and destroy filesystem")
        assert res_danger.outcome == DecisionOutcome.DENY
        assert res_danger.allowed is False


def test_dynamic_attribute_access_nonexistent():
    """Verify dynamic attribute access for non-existent attributes raises AttributeError."""
    import reflex
    import system1

    for mod in (reflex, system1):
        for bad_name in ["_non_existent", "missing_symbol", "custom_foo_bar", ""]:
            assert not hasattr(mod, bad_name), f"{mod.__name__} should not have {bad_name!r}"
            with pytest.raises(AttributeError) as exc_info:
                getattr(mod, bad_name)
            assert f"module {mod.__name__!r} has no attribute {bad_name!r}" in str(exc_info.value)


def test_mutation_invariance():
    """Verify mutations to reflex do not pollute system1 or its module collections."""
    import reflex
    import system1

    # Independent __all__ copies
    assert reflex.__all__ is not system1.__all__, "reflex.__all__ must be a distinct list copy"
    orig_s1_len = len(system1.__all__)
    reflex.__all__.append("__adversarial_canary__")
    assert len(system1.__all__) == orig_s1_len
    assert "__adversarial_canary__" not in system1.__all__

    # Attribute mutation isolation
    reflex.__adversarial_attr__ = "canary"
    assert not hasattr(system1, "__adversarial_attr__")
