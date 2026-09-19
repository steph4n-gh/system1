"""Tier 1.1: E2E Requirement Tests for Dual Namespace Imports (reflex vs system1).

Authoritative Invariants:
1. Complete namespace symmetry: import reflex and import system1 export identical public interfaces.
2. Version alignment: reflex.__version__ == system1.__version__ == '0.2.2'.
3. Object identity: exported classes, functions, and schemas are identical objects in memory.
4. Drop-in interoperability: decision engines, guard hooks, ledgers, and SDK clients operate interchangeably.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import pytest

import reflex
import system1


def test_top_level_package_exports_parity():
    """Verify top-level symbols, __all__, and __version__ are completely symmetric."""
    assert reflex.__version__ == system1.__version__ == "0.2.2"
    
    # Both packages should export the exact same set of public symbols
    reflex_all = set(reflex.__all__)
    system1_all = set(system1.__all__)
    assert reflex_all == system1_all, f"Export mismatch: diff={reflex_all ^ system1_all}"

    # Verify identity for core exports
    core_symbols = [
        "SystemOneEngine",
        "System1Engine",
        "DecisionResult",
        "BenchmarkReport",
        "decide",
        "DecisionSchema",
        "DecisionField",
        "ChoiceField",
        "BooleanField",
        "MultiChoiceField",
        "ScoreField",
        "DecisionCalibrator",
        "ConformalPredictor",
        "DecisionWitnessReceipt",
        "create_decision_receipt",
        "verify_decision_witness_receipt",
        "ActionLedger",
        "SystemOneGuardHook",
        "DefaultGuardDecisionSchema",
        "GuardInterceptionResult",
        "ActionProposal",
        "PolicyDecision",
        "EvidenceRef",
        "DecisionOutcome",
        "RiskLevel",
        "SystemOneCompiler",
        "SemanticSystemOneCache",
        "TypeSafeClient",
        "patch_typesafe",
    ]
    for sym in core_symbols:
        assert hasattr(reflex, sym), f"reflex missing symbol: {sym}"
        assert hasattr(system1, sym), f"system1 missing symbol: {sym}"
        assert getattr(reflex, sym) is getattr(system1, sym), f"Identity mismatch for {sym}"


def test_submodule_existence_and_importability():
    """Verify all submodules can be resolved and exported via reflex and system1."""
    submodules = [
        "schema",
        "cli",
        "compiler",
        "cache",
        "telemetry",
        "embeddings",
        "proto",
        "grpc_server",
        "integrations",
        "compat",
    ]
    for submod in submodules:
        mod_reflex = importlib.import_module(f"reflex.{submod}")
        mod_system1 = importlib.import_module(f"system1.{submod}")
        assert mod_reflex is not None
        assert mod_system1 is not None

        # Check key functions/classes in each submodule
        if hasattr(mod_reflex, "__all__") and hasattr(mod_system1, "__all__"):
            assert set(mod_reflex.__all__) == set(mod_system1.__all__)


def test_functional_runtime_decision_parity(triage_schema):
    """Verify executing decisions via reflex.SystemOneEngine and system1.System1Engine produces identical results."""
    engine_reflex = reflex.SystemOneEngine(triage_schema)
    engine_system1 = system1.System1Engine(triage_schema)

    test_prompt = "Perform read-only query on customer SQLite database"
    res_reflex = engine_reflex.decide(test_prompt)
    res_system1 = engine_system1.decide(test_prompt)

    assert res_reflex.values == res_system1.values
    assert res_reflex.schema_name == res_system1.schema_name
    assert set(res_reflex.conformal_sets.keys()) == set(res_system1.conformal_sets.keys())
    assert set(res_reflex.confidences.keys()) == set(res_system1.confidences.keys())

    # Convenient decide() helper function parity
    res_helper_reflex = reflex.decide(test_prompt, schema=triage_schema)
    res_helper_system1 = system1.decide(test_prompt, schema=triage_schema)
    assert res_helper_reflex.values == res_helper_system1.values


def test_typesafe_sdk_twin_parity():
    """Verify TypeSafe SDK drop-in client symbols and methods have identical twins."""
    assert reflex.TypeSafeClient is system1.TypeSafeClient
    assert reflex.AsyncTypeSafeClient is system1.AsyncTypeSafeClient
    assert reflex.patch_typesafe is system1.patch_typesafe
    assert reflex.system_one is system1.system_one
    assert reflex.batch_system_one is system1.batch_system_one

    client_r = reflex.TypeSafeClient(api_key="local-test-dummy")
    client_s = system1.TypeSafeClient(api_key="local-test-dummy")
    assert type(client_r) is type(client_s)


def test_guard_hook_and_ledger_twin_parity(tmp_path: Path):
    """Verify ActionLedger and SystemOneGuardHook operate seamlessly across imports."""
    db_r = tmp_path / "ledger_reflex.db"
    db_s = tmp_path / "ledger_system1.db"

    ledger_r = reflex.ActionLedger(db_r)
    ledger_s = system1.ActionLedger(db_s)

    guard_r = reflex.SystemOneGuardHook(ledger=ledger_r, auto_calibrate=False)
    guard_s = system1.SystemOneGuardHook(ledger=ledger_s, auto_calibrate=False)

    proposal = reflex.ActionProposal.create(
        tenant_id="t1",
        principal_id="p1",
        scope="read",
        tool="file_reader",
        arguments={"path": "/etc/hosts"},
        purpose="Inspect local configuration",
    )

    int_r = guard_r.evaluate_proposal(proposal)
    int_s = guard_s.evaluate_proposal(proposal)

    assert int_r.outcome == int_s.outcome
    assert int_r.allowed == int_s.allowed
    assert int_r.policy_decision.outcome == int_s.policy_decision.outcome

    ledger_r.close()
    ledger_s.close()


def test_compiler_and_cache_twin_parity(triage_schema):
    """Verify SystemOneCompiler and SemanticSystemOneCache share identity and behavior."""
    assert reflex.SystemOneCompiler is system1.SystemOneCompiler
    assert reflex.SemanticSystemOneCache is system1.SemanticSystemOneCache
    assert reflex.CacheEntry is system1.CacheEntry

    cache = reflex.SemanticSystemOneCache(similarity_threshold=0.98)
    assert cache.capacity > 0

    compiler = reflex.SystemOneCompiler(triage_schema)
    assert isinstance(compiler.schema, triage_schema)
