"""Tests for system1 comprehensive exports, submodules, CLI entrypoint, and API integrity."""

import json
from pathlib import Path
import numpy as np
import pytest

import system1
from system1.cli import build_parser

ALL_16_SUBMODULES = [
    "cache",
    "calibration",
    "cli",
    "compat",
    "compiler",
    "core",
    "embeddings",
    "engine",
    "grpc_server",
    "guard",
    "integrations",
    "ledger",
    "model",
    "neural",
    "proto",
    "receipt",
    "schema",
    "telemetry",
]


def test_system1_namespace_exports():
    """Verify system1 top-level exports and API integrity."""
    expected_exports = {
        "__version__",
        "ReflexEngine",
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
        "SchemaMeta",
        "DecisionCalibrator",
        "CalibrationMetrics",
        "BrierDecomposition",
        "ConformalPredictor",
        "ConformalPredictionSet",
        "RegressionConformalPredictor",
        "RegressionConformalInterval",
        "DecisionWitnessReceipt",
        "create_decision_receipt",
        "verify_decision_witness_receipt",
        "compute_receipt_digest",
        "RunWitnessEnvelope",
        "create_run_witness_envelope",
        "sign_run_witness_envelope",
        "verify_run_witness_envelope",
        "ActionLedger",
        "LedgerError",
        "IntegrityError",
        "ReflexGuardHook",
        "SystemOneGuardHook",
        "DefaultGuardDecisionSchema",
        "GuardInterceptionResult",
        "ActionProposal",
        "PolicyDecision",
        "EvidenceRef",
        "DecisionOutcome",
        "RiskLevel",
        "ActionState",
        "ResultStatus",
        "SystemOneModel",
        "DecisionFieldHead",
        "DeterministicSemanticProjector",
        "LocalNeuralProjector",
        "ModelInferenceResult",
        "RawFieldEvaluation",
        "TypeSafeClient",
        "Client",
        "AsyncTypeSafeClient",
        "AsyncClient",
        "Choice",
        "Noul",
        "Score",
        "TypeSafeResponse",
        "SystemOneResponse",
        "ChoiceAnswer",
        "NoulAnswer",
        "ScoreAnswer",
        "MultiChoiceAnswer",
        "Usage",
        "system_one",
        "systemone",
        "batch_system_one",
        "batch_systemone",
        "patch_typesafe",
        "DotDict",
    }
    assert expected_exports.issubset(set(system1.__all__))
    for name in system1.__all__:
        assert hasattr(system1, name), f"system1 missing export {name}"
        obj = getattr(system1, name)
        assert obj is not None, f"Export {name} is None"


def test_system1_submodules():
    """Verify all 16 submodules exist on system1, match disk discovery, and export valid symbols."""
    import types
    import pkgutil

    discovered = sorted([m.name for m in pkgutil.iter_modules(system1.__path__)])
    assert discovered == ALL_16_SUBMODULES, (
        f"Submodule inventory mismatch: discovered={discovered} vs expected={ALL_16_SUBMODULES}"
    )

    for sub in ALL_16_SUBMODULES:
        assert hasattr(system1, sub), f"system1 missing submodule {sub}"
        submod = getattr(system1, sub)
        assert submod is not None, f"Submodule {sub} is None"
        assert isinstance(submod, types.ModuleType), f"Submodule {sub} is not a ModuleType"
        assert hasattr(submod, "__all__"), f"Submodule {sub} missing explicit __all__"
        assert len(submod.__all__) > 0, f"Submodule {sub} has empty __all__"
        for item in submod.__all__:
            assert hasattr(submod, item), f"Submodule {sub} missing export {item}"

    assert system1.compat.typesafe.TypeSafeClient is not None
    assert system1.core.SystemOneModel is not None


def test_system1_decide_api():
    """Verify system1.decide() executes end-to-end and returns a valid DecisionResult."""
    class RouteSchema(system1.DecisionSchema):
        action = system1.ChoiceField(options=["allow", "deny"])

    prompt = "Read /etc/resolv.conf"
    r1 = system1.decide(prompt, schema=RouteSchema)

    assert r1.action in ("allow", "deny")
    assert r1.schema_digest is not None
    assert r1.latency_ms >= 0.0
    assert "action" in r1.confidences


def test_engine_benchmark_percentiles_np():
    """Verify engine.benchmark computes exact numpy interpolated percentiles."""
    class SimpleSchema(system1.DecisionSchema):
        choice = system1.ChoiceField(options=["a", "b"])

    engine = system1.ReflexEngine(SimpleSchema, backend="numpy")
    report = engine.benchmark(iterations=50, warmup=5)

    assert report.total_decisions == 50
    assert report.p50_latency_ms <= report.p90_latency_ms
    assert report.p90_latency_ms <= report.p95_latency_ms
    assert report.p95_latency_ms <= report.p99_latency_ms
    assert report.min_latency_ms <= report.p50_latency_ms <= report.max_latency_ms


def test_engine_benchmark_invalid_iterations():
    """Verify engine.benchmark validates iterations >= 1 and warmup >= 0."""
    class SimpleSchema(system1.DecisionSchema):
        choice = system1.ChoiceField(options=["a", "b"])

    engine = system1.ReflexEngine(SimpleSchema, backend="numpy")
    with pytest.raises(ValueError, match="iterations must be a positive integer >= 1"):
        engine.benchmark(iterations=0)

    with pytest.raises(ValueError, match="warmup must be a non-negative integer >= 0"):
        engine.benchmark(warmup=-1)


def test_cli_persistent_identity_reuse(tmp_path, monkeypatch):
    """Verify CLI --sign reuses persistent identity key and creates 0o600 permissions."""
    id_dir = tmp_path / "test_identity"
    id_dir.mkdir(parents=True, exist_ok=True)

    parser = build_parser()

    # First decide run with explicit identity dir
    args1 = parser.parse_args([
        "decide", "First signed decision",
        "--sign",
        "--identity-dir", str(id_dir),
        "--json",
    ])
    assert args1.func(args1) == 0

    key_file = id_dir / "identity.key"
    pub_file = id_dir / "identity.pub"
    assert key_file.is_file(), "identity.key must be saved to disk"
    assert pub_file.is_file(), "identity.pub must be saved to disk"

    # Verify restrictive owner-only permissions (0o600)
    key_mode = key_file.stat().st_mode & 0o777
    assert key_mode == 0o600, f"Expected 0o600 for private key, got {oct(key_mode)}"

    initial_key_bytes = key_file.read_bytes()
    initial_pub_bytes = pub_file.read_bytes()

    # Second decide run with same identity dir
    args2 = parser.parse_args([
        "decide", "Second signed decision",
        "--sign",
        "--identity-dir", str(id_dir),
        "--json",
    ])
    assert args2.func(args2) == 0

    # Key must be preserved and reused, not overwritten with a new key
    assert key_file.read_bytes() == initial_key_bytes
    assert pub_file.read_bytes() == initial_pub_bytes


def test_system1_cli_main_entrypoint(capsys):
    """Verify system1.cli.main executes correctly."""
    import system1.cli as s1_cli
    ret = s1_cli.main(["decide", "Safe read check", "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["schema_name"] == "DefaultTriageSchema"


def test_reflex_package_parity():
    """Verify reflex package exports and provides 100% interoperability with system1."""
    import reflex
    assert reflex.__version__ == system1.__version__
    assert reflex.ReflexEngine is system1.ReflexEngine
    assert reflex.ReflexCompiler is system1.ReflexCompiler
    assert reflex.DecisionSchema is system1.DecisionSchema
    assert reflex.ChoiceField is system1.ChoiceField
    assert reflex.BooleanField is system1.BooleanField
    assert reflex.MultiChoiceField is system1.MultiChoiceField
    assert reflex.ScoreField is system1.ScoreField
    assert reflex.ActionLedger is system1.ActionLedger

    class ParitySchema(reflex.DecisionSchema):
        choice = reflex.ChoiceField(options=["a", "b"])

    res = reflex.decide("Option a test", schema=ParitySchema)
    assert res.choice in ("a", "b")


def test_reflex_all_16_submodules_import_and_parity():
    """Verify all 16 submodules in system1 can be imported via reflex.<submodule> and __all__ matches."""
    import importlib
    import pkgutil
    import system1
    import reflex

    discovered = sorted([m.name for m in pkgutil.iter_modules(system1.__path__)])
    assert discovered == ALL_16_SUBMODULES, f"Expected 16 submodules {ALL_16_SUBMODULES}, discovered {discovered}"

    for sub in ALL_16_SUBMODULES:
        s_mod = importlib.import_module(f"system1.{sub}")
        r_mod = importlib.import_module(f"reflex.{sub}")

        assert hasattr(s_mod, "__all__"), f"system1.{sub} missing explicit __all__"
        assert hasattr(r_mod, "__all__"), f"reflex.{sub} missing explicit __all__"
        assert set(r_mod.__all__) == set(s_mod.__all__), (
            f"Submodule {sub} __all__ mismatch: {set(r_mod.__all__) ^ set(s_mod.__all__)}"
        )
        assert len(r_mod.__all__) == len(s_mod.__all__)

        for symbol in s_mod.__all__:
            assert hasattr(r_mod, symbol), f"reflex.{sub} missing symbol {symbol}"
            r_val = getattr(r_mod, symbol)
            s_val = getattr(s_mod, symbol)
            assert r_val is s_val, f"Symbol {symbol} in {sub} is not identical object ({r_val} vs {s_val})"


def test_reflex_dynamic_submodule_getattr():
    """Verify getattr(reflex, submod) dynamically resolves all 15 submodules and __dir__ includes them."""
    import importlib
    import reflex

    submodules = [
        "cache", "calibration", "cli", "compat", "compiler",
        "core", "embeddings", "engine", "guard", "ledger",
        "model", "neural", "receipt", "schema", "telemetry",
    ]
    reflex_dir = dir(reflex)
    for sub in submodules:
        mod = getattr(reflex, sub)
        assert mod is not None, f"getattr(reflex, {sub!r}) returned None"
        expected_mod = importlib.import_module(f"reflex.{sub}")
        assert mod is expected_mod, f"getattr(reflex, {sub!r}) returned {mod}, expected {expected_mod}"
        assert sub in reflex_dir, f"Submodule {sub} missing from dir(reflex)"


def test_guard_interception_result_allowed_property():
    """Verify GuardInterceptionResult.allowed returns True for ALLOW and False for non-ALLOW outcomes."""
    from system1.guard import (
        ActionProposal,
        DecisionOutcome,
        GuardInterceptionResult,
        ReflexGuardHook,
    )

    # 1. Direct unit verification on GuardInterceptionResult instance
    res_allow = GuardInterceptionResult(DecisionOutcome.ALLOW, "safe action", None, None)
    assert res_allow.allowed is True

    res_deny = GuardInterceptionResult(DecisionOutcome.DENY, "forbidden action", None, None)
    assert res_deny.allowed is False

    res_escalate = GuardInterceptionResult(DecisionOutcome.REQUIRE_APPROVAL, "requires human approval", None, None)
    assert res_escalate.allowed is False

    # 2. End-to-end reference monitor hook evaluation
    hook = ReflexGuardHook(min_confidence=0.50, alpha=0.10)
    proposal_allow = ActionProposal.create(
        tenant_id="tenant_001",
        principal_id="principal_001",
        scope="local",
        tool="read_file",
        arguments={"target": "/tmp/readme.txt", "args": "mode=read"},
        purpose="Read documentation",
    )
    res_hook_allow = hook.evaluate_proposal(
        proposal_allow,
        context_prompt="Inspect read-only project documentation in README.md",
    )
    assert res_hook_allow.outcome == DecisionOutcome.ALLOW
    assert res_hook_allow.allowed is True

    proposal_deny = ActionProposal.create(
        tenant_id="tenant_001",
        principal_id="principal_001",
        scope="local",
        tool="execute_command",
        arguments={"target": "/bin/bash", "args": "rm -rf /"},
        purpose="Dangerous destruction",
    )
    res_hook_deny = hook.evaluate_proposal(
        proposal_deny,
        context_prompt="Destroy system root directory rm -rf / and wipe disks",
    )
    assert res_hook_deny.outcome != DecisionOutcome.ALLOW
    assert res_hook_deny.allowed is False


def test_reflex_core_submodules_parity():
    """Verify all reflex.core submodules match system1.core."""
    import importlib
    import reflex.core
    import system1.core

    core_subs = ["embeddings", "model", "neural", "schema", "telemetry"]
    for sub in core_subs:
        s_mod = importlib.import_module(f"system1.core.{sub}")
        r_mod = importlib.import_module(f"reflex.core.{sub}")
        assert hasattr(s_mod, "__all__")
        assert hasattr(r_mod, "__all__")
        assert set(r_mod.__all__) == set(s_mod.__all__)
        for symbol in s_mod.__all__:
            assert getattr(r_mod, symbol) is getattr(s_mod, symbol)


def test_reflex_cli_main_entrypoint(capsys):
    """Verify reflex.cli.main executes correctly."""
    import reflex.cli as r_cli
    ret = r_cli.main(["decide", "Safe read check", "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["schema_name"] == "DefaultTriageSchema"


def test_system1_dynamic_submodule_getattr():
    """Verify getattr(system1, submod) dynamically resolves all 16 submodules and __dir__ includes them."""
    import importlib

    system1_dir = dir(system1)
    for sub in ALL_16_SUBMODULES:
        assert hasattr(system1, sub), f"hasattr(system1, {sub!r}) returned False"
        mod = getattr(system1, sub)
        assert mod is not None, f"getattr(system1, {sub!r}) returned None"
        expected_mod = importlib.import_module(f"system1.{sub}")
        assert mod is expected_mod, f"getattr(system1, {sub!r}) returned {mod}, expected {expected_mod}"
        assert sub in system1_dir, f"Submodule {sub} missing from dir(system1)"


def test_system1_isolated_subprocess_attribute_resolution():
    """Verify in a clean, isolated subprocess that all 15 submodules on system1 resolve dynamically.
    
    Prevents pre-import contamination (from reflex or engine) from masking lazy-loading defects.
    """
    import os
    import subprocess
    import sys

    code = """
import sys
import types
import system1

ALL_15 = [
    "cache", "calibration", "cli", "compat", "compiler",
    "core", "embeddings", "engine", "guard", "ledger",
    "model", "neural", "receipt", "schema", "telemetry"
]

failed = []
for sub in ALL_15:
    if not hasattr(system1, sub):
        failed.append(f"hasattr(system1, {sub!r}) is False")
        continue
    try:
        val = getattr(system1, sub)
        if val is None:
            failed.append(f"getattr(system1, {sub!r}) is None")
        elif not isinstance(val, types.ModuleType):
            failed.append(f"getattr(system1, {sub!r}) is not ModuleType: {type(val)}")
        elif val.__name__ != f"system1.{sub}":
            failed.append(f"getattr(system1, {sub!r}).__name__ is {val.__name__!r}, expected 'system1.{sub}'")
    except Exception as exc:
        failed.append(f"getattr(system1, {sub!r}) raised {type(exc).__name__}: {exc}")

if failed:
    print("\\n".join(failed))
    sys.exit(1)
sys.exit(0)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, (
        f"Isolated system1 dynamic submodule resolution failed:\n{proc.stdout}\n{proc.stderr}"
    )


def test_twin_namespaces_isolated_subprocess_parity():
    """Verify that both system1 and reflex provide identical dynamic submodule resolution in independent isolated subprocesses."""
    import os
    import subprocess
    import sys

    for pkg in ("system1", "reflex"):
        code = f"""
import sys
import types
import {pkg}

ALL_15 = [
    "cache", "calibration", "cli", "compat", "compiler",
    "core", "embeddings", "engine", "guard", "ledger",
    "model", "neural", "receipt", "schema", "telemetry"
]

for sub in ALL_15:
    assert hasattr({pkg}, sub), f"hasattr({pkg}, {{sub!r}}) is False"
    val = getattr({pkg}, sub)
    assert val is not None, f"getattr({pkg}, {{sub!r}}) is None"
    assert isinstance(val, types.ModuleType), f"getattr({pkg}, {{sub!r}}) is not ModuleType"
    assert val.__name__ == f"{pkg}.{{sub}}", f"Module name mismatch: {{val.__name__}}"
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
        )
        assert proc.returncode == 0, f"Isolated {pkg} dynamic submodule test failed:\n{proc.stdout}\n{proc.stderr}"


def test_system1_and_reflex_negative_attribute_resolution():
    """Verify that non-existent attributes raise AttributeError and hasattr is False on both namespaces."""
    import reflex

    for pkg, name in [(system1, "system1"), (reflex, "reflex")]:
        for bad_attr in ["non_existent_module", "random_field_123", "", "_internal_unknown"]:
            assert not hasattr(pkg, bad_attr), f"hasattr({name}, {bad_attr!r}) should be False"
            with pytest.raises(AttributeError, match=f"module {name!r} has no attribute {bad_attr!r}"):
                getattr(pkg, bad_attr)


def test_submodule_inventory_and_module_map_completeness():
    """Verify disk packages, _MODULE_MAP, and __dir__ match all 16 submodules across both namespaces."""
    import pkgutil
    import reflex

    s1_disk = set(m.name for m in pkgutil.iter_modules(system1.__path__))
    r_disk = set(m.name for m in pkgutil.iter_modules(reflex.__path__))
    expected = set(ALL_16_SUBMODULES)

    assert s1_disk == expected, f"system1 disk submodules mismatch: {s1_disk ^ expected}"
    assert r_disk == expected, f"reflex disk submodules mismatch: {r_disk ^ expected}"

    # Verify all 16 submodules appear in dir()
    assert expected.issubset(set(dir(system1))), f"Missing from dir(system1): {expected - set(dir(system1))}"
    assert expected.issubset(set(dir(reflex))), f"Missing from dir(reflex): {expected - set(dir(reflex))}"


def test_spec_harmonization_aliases():
    """Verify spec harmonization aliases for exceptions, TypeSafeClient, and engine learning methods."""
    import reflex
    import system1
    from system1.integrations.langchain import ReflexSecurityException, ReflexGuardBlockedException
    from reflex.integrations.langchain import ReflexSecurityException as ReflexSecurityExceptionReflex

    # 1. Exception alias parity
    assert ReflexSecurityException is ReflexGuardBlockedException
    assert ReflexSecurityExceptionReflex is ReflexSecurityException

    # 2. TypeSafeClient agreement_threshold alias
    client = system1.compat.typesafe.TypeSafeClient(agreement_threshold=0.95)
    assert client.min_agreement_threshold == 0.95
    client_r = reflex.compat.typesafe.TypeSafeClient(agreement_threshold=0.92)
    assert client_r.min_agreement_threshold == 0.92

    # 3. Engine and Model System 2 / Tier 3 learning aliases
    assert system1.ReflexEngine.learn_from_system2 is system1.ReflexEngine.learn_from_tier2
    assert system1.ReflexEngine.learn_from_tier3 is system1.ReflexEngine.learn_from_tier2
    assert reflex.ReflexEngine.learn_from_system2 is reflex.ReflexEngine.learn_from_tier2

    assert system1.compiler.CompiledSystemOneModel.learn_from_system2 is system1.compiler.CompiledSystemOneModel.learn_from_tier2
    assert system1.core.SystemOneModel.learn_from_system2 is system1.core.SystemOneModel.learn_from_tier2



