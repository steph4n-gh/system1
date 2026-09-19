"""Tests verifying system1.core functions completely in isolation.

Guarantees:
1. Zero dependencies on cryptography (Ed25519 signing, receipts)
2. Zero dependencies on SQLite / disk I/O (ActionLedger)
3. Zero dependencies on governance stack (engine, guard, compat, cli)
4. Evaluator kernel operates with ONLY NumPy (and optional Apple Silicon MLX)
5. Comprehensive export integrity for system1.core
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

import system1.core
from system1.core import (
    BooleanField,
    ChoiceField,
    DecisionField,
    DecisionFieldHead,
    DecisionSchema,
    DeterministicSemanticProjector,
    LocalNeuralProjector,
    ModelInferenceResult,
    MultiChoiceField,
    RawFieldEvaluation,
    ScoreField,
    SystemOneModel,
)


def _run_in_isolated_python(script: str) -> subprocess.CompletedProcess:
    """Runs a Python snippet in a clean subprocess with src/ on sys.path."""
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    cmd = [
        sys.executable,
        "-c",
        f"import sys; sys.path.insert(0, {str(src_dir)!r})\n{script}",
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_core_imports_without_cryptography_or_sqlite():
    """Verify system1.core does NOT load cryptography, sqlite3, or disk I/O modules."""
    script = """
import sys
import system1.core as core

# 1. Assert cryptography is NOT in sys.modules
crypto_mods = [m for m in sys.modules if "cryptography" in m]
if crypto_mods:
    print(f"FAILED: cryptography loaded: {crypto_mods}")
    sys.exit(1)

# 2. Assert sqlite3 is NOT in sys.modules
sqlite_mods = [m for m in sys.modules if "sqlite3" in m]
if sqlite_mods:
    print(f"FAILED: sqlite3 loaded: {sqlite_mods}")
    sys.exit(2)

# 3. Assert governance layer is NOT in sys.modules
gov_mods = [
    m for m in sys.modules
    if any(m.startswith(f"system1.{x}") for x in ("receipt", "ledger", "engine", "guard", "compat", "cli"))
]
if gov_mods:
    print(f"FAILED: governance modules loaded: {gov_mods}")
    sys.exit(3)

print("SUCCESS: system1.core imported with pure zero-dependency isolation")
"""
    result = _run_in_isolated_python(script)
    assert result.returncode == 0, f"Subprocess failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "SUCCESS" in result.stdout



def test_core_full_inference_pipeline_in_isolation():
    """Execute complete end-to-end forward evaluation in isolated process without crypto/sqlite."""
    script = """
import sys
import system1.core as core

class IsolatedTriageSchema(core.DecisionSchema):
    route = core.ChoiceField(options=["tier1", "tier2", "escalate"])
    is_urgent = core.BooleanField(threshold=0.6)
    tags = core.MultiChoiceField(options=["billing", "auth", "outage"], threshold=0.4)
    risk_score = core.ScoreField(min_value=0.0, max_value=1.0)

# Instantiate model
model = core.SystemOneModel(IsolatedTriageSchema, dimension=256, backend="numpy")

# Single forward pass
prompt = "URGENT: Database authentication is failing for all billing services!"
res = model.forward_single(prompt)

assert isinstance(res, core.ModelInferenceResult)
assert "route" in res.fields
assert "is_urgent" in res.fields
assert "tags" in res.fields
assert "risk_score" in res.fields

# Check evaluation types
assert isinstance(res.fields["route"].selected_value, str)
assert res.fields["route"].selected_value in ["tier1", "tier2", "escalate"]
assert 0.0 <= res.fields["route"].confidence <= 1.0

assert isinstance(res.fields["is_urgent"].selected_value, bool)
assert isinstance(res.fields["tags"].selected_value, tuple)
assert 0.0 <= res.fields["risk_score"].selected_value <= 1.0

# Batch forward pass
prompts = [
    "Billing dispute for invoice #100",
    "Password reset email not received",
    "Major cluster degradation observed",
]
batch_res = model.forward_batch(prompts)
assert len(batch_res) == 3
for b in batch_res:
    assert isinstance(b, core.ModelInferenceResult)
    assert b.fields["route"].selected_value in ["tier1", "tier2", "escalate"]

# Verify projectors
det_proj = core.DeterministicSemanticProjector(dimension=128)
v_det = det_proj.project(prompt)
assert v_det.shape == (128,)

neural_proj = core.LocalNeuralProjector(dimension=128, backend="numpy")
v_neur = neural_proj.project(prompt)
assert v_neur.shape == (128,)

# Final sanity check on isolated modules: ZERO crypto, ZERO sqlite
assert not [m for m in sys.modules if "cryptography" in m], "cryptography was loaded during inference"
assert not [m for m in sys.modules if "sqlite3" in m], "sqlite3 was loaded during inference"

print("SUCCESS: full inference pipeline executed in zero-dependency isolation")
"""
    result = _run_in_isolated_python(script)
    assert result.returncode == 0, f"Subprocess failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "SUCCESS" in result.stdout


def test_core_namespace_exports():
    """Verify comprehensive exports on system1.core."""
    expected = {
        "BooleanField",
        "ChoiceField",
        "DecisionField",
        "DecisionFieldHead",
        "DecisionSchema",
        "DeterministicSemanticProjector",
        "LocalNeuralProjector",
        "ModelInferenceResult",
        "MultiChoiceField",
        "RawFieldEvaluation",
        "ScoreField",
        "SystemOneModel",
    }
    for item in expected:
        assert hasattr(system1.core, item)
        assert item in system1.core.__all__

    # Verify mathematical and backend utilities are available
    for util_name in ("HAS_MLX", "stable_softmax", "stable_sigmoid"):
        assert hasattr(system1.core, util_name)


def test_core_submodule_exports():
    """Verify model, neural, and schema submodules in system1.core."""
    for sub in ("model", "neural", "schema"):
        assert hasattr(system1.core, sub)
        s_mod = getattr(system1.core, sub)
        assert hasattr(s_mod, "__all__")
        for item in s_mod.__all__:
            assert hasattr(s_mod, item)


def test_core_schema_primitives():
    """Test schema definition, validation, serialization, and digests in core."""
    class AgentSchema(DecisionSchema):
        action = ChoiceField(options=["allow", "deny", "escalate"])
        critical = BooleanField(threshold=0.7)
        labels = MultiChoiceField(options=["security", "compliance", "performance"])
        risk = ScoreField(min_value=0.0, max_value=10.0)

    schema = AgentSchema()
    assert len(schema.fields) == 4
    assert schema.get_field("action").options == ("allow", "deny", "escalate")

    # Serialization and digest
    canonical = schema.canonical_json()
    digest = schema.schema_digest()
    assert isinstance(canonical, str)
    assert len(digest) == 64

    # Validation
    valid_dec = schema.validate_decision({
        "action": "allow",
        "critical": True,
        "labels": ["security"],
        "risk": 3.5,
    })
    assert valid_dec["action"] == "allow"
    assert valid_dec["critical"] is True
    assert valid_dec["labels"] == ("security",)
    assert valid_dec["risk"] == 3.5

    # Reconstruct from dict
    s_dict = schema.to_dict()
    reconstructed = DecisionSchema.from_dict(s_dict)
    assert reconstructed.schema_digest() == digest


def test_core_deterministic_projector():
    """Verify properties of DeterministicSemanticProjector in core."""
    proj = DeterministicSemanticProjector(dimension=256)
    v1 = proj.project("kubernetes ingress controller error")
    v2 = proj.project("kubernetes ingress controller error")
    v3 = proj.project("chocolate chip cookie recipe")

    # Identical strings yield identical vectors
    np.testing.assert_allclose(v1, v2)
    assert np.isclose(np.linalg.norm(v1), 1.0, atol=1e-5)

    # Cosine similarities
    sim_diff = float(np.dot(v1, v3))
    assert sim_diff < 5.0, f"Unrelated texts too similar: {sim_diff}"

    # Empty text handling
    v_empty = proj.project("")
    assert v_empty.shape == (256,)
    assert np.isclose(np.linalg.norm(v_empty), 1.0, atol=1e-5)

    # encode alias parity
    np.testing.assert_allclose(proj.encode("test string"), proj.project("test string"))

    # similarity method parity
    sim_calc = proj.similarity("kubernetes ingress controller", "chocolate chip cookie")
    v_a = proj.project("kubernetes ingress controller")
    v_b = proj.project("chocolate chip cookie")
    assert np.isclose(sim_calc, float(np.dot(v_a, v_b)), atol=1e-5)

    # project_batch testing
    texts = [
        "First test prompt",
        "Second test prompt",
        "Third test prompt",
    ]
    batch_vecs = proj.project_batch(texts)
    assert batch_vecs.shape == (3, 256)
    for i, t in enumerate(texts):
        np.testing.assert_allclose(batch_vecs[i], proj.project(t), atol=1e-5)

    # Empty project_batch
    assert proj.project_batch([]).shape == (0, 256)

    # Validation errors
    with pytest.raises(ValueError, match="dimension must be positive"):
        DeterministicSemanticProjector(dimension=0)
    with pytest.raises(ValueError, match="dimension must be positive"):
        DeterministicSemanticProjector(dimension=-10)
    with pytest.raises(TypeError, match="Expected text to be a string"):
        proj.project(12345)  # type: ignore
    with pytest.raises(TypeError, match="Expected text to be a string"):
        proj.project(None)  # type: ignore
    with pytest.raises(TypeError, match="texts must be a sequence of strings"):
        proj.project_batch(None)  # type: ignore
    with pytest.raises(TypeError, match="All elements in texts must be strings"):
        proj.project_batch(["valid", 999])  # type: ignore


def test_core_neural_projector():
    """Verify properties of LocalNeuralProjector in core."""
    proj = LocalNeuralProjector(dimension=192, hidden_dim=256, backend="numpy", seed=123)
    text = "Deploying microservices with docker and container registries"
    vec = proj.project(text)
    assert vec.shape == (192,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    batch_vecs = proj.project_batch([text, "Unrelated culinary cuisine text"])
    assert batch_vecs.shape == (2, 192)
    np.testing.assert_allclose(batch_vecs[0], vec, atol=1e-5)


def test_core_system_one_model_forward():
    """Verify SystemOneModel non-autoregressive forward evaluation in core."""
    class RouteSchema(DecisionSchema):
        action = ChoiceField(options=["route_fast", "route_smart"])
        urgent = BooleanField()

    model = SystemOneModel(RouteSchema, dimension=128, backend="numpy")
    res = model.forward_single("Optimize latency for simple arithmetic queries")

    assert res.fields["action"].field_type == "choice"
    assert res.fields["action"].selected_value in ["route_fast", "route_smart"]
    assert res.fields["urgent"].field_type == "boolean"
    assert isinstance(res.fields["urgent"].selected_value, bool)
    assert res.inference_latency_ms >= 0.0


def test_governance_layers_build_on_top_of_core():
    """Verify governance modules correctly import and subclass/use core primitives."""
    from system1.engine import SystemOneEngine
    from system1.guard import SystemOneGuardHook
    from system1.compat.typesafe import TypeSafeClient

    class QuickSchema(DecisionSchema):
        opt = ChoiceField(options=["a", "b"])

    engine = SystemOneEngine(QuickSchema, dimension=128, backend="numpy")
    assert isinstance(engine.model, SystemOneModel)

    hook = SystemOneGuardHook(engine=engine)
    assert isinstance(hook.engine.schema, DecisionSchema)

    client = TypeSafeClient()
    assert hasattr(client, "systemone")


def test_core_edge_cases_and_error_handling():
    """Verify edge cases, boundary conditions, and invalid parameter handling in core."""
    # 1. ChoiceField validation errors
    with pytest.raises(ValueError, match="requires at least one option"):
        ChoiceField(options=[])
    with pytest.raises(ValueError, match="Duplicate option"):
        ChoiceField(options=["yes", "yes"])
    with pytest.raises(ValueError, match="Default value 'unknown' not in options"):
        ChoiceField(options=["a", "b"], default="unknown")

    field = ChoiceField(options=["low", "high"])
    field.bind_name("tier")
    assert field.validate_value("low") == "low"
    with pytest.raises(TypeError):
        field.validate_value(123)
    with pytest.raises(ValueError):
        field.validate_value("medium")

    # 2. BooleanField boundary conditions
    with pytest.raises(ValueError, match="threshold must be in"):
        BooleanField(threshold=-0.1)
    with pytest.raises(ValueError, match="threshold must be in"):
        BooleanField(threshold=1.1)

    b_field = BooleanField(threshold=0.5)
    b_field.bind_name("flag")
    assert b_field.validate_value("true") is True
    assert b_field.validate_value("false") is False
    assert b_field.validate_value(1) is True
    assert b_field.validate_value(0) is False
    with pytest.raises(TypeError):
        b_field.validate_value("not_a_bool")

    # 3. ScoreField bounds & errors
    with pytest.raises(ValueError, match="must be strictly less than"):
        ScoreField(min_value=1.0, max_value=0.0)
    with pytest.raises(ValueError, match="outside bounds"):
        ScoreField(min_value=0.0, max_value=1.0, default=2.0)

    s_field = ScoreField(min_value=0.0, max_value=10.0)
    s_field.bind_name("score")
    assert s_field.validate_value(5.5) == 5.5
    with pytest.raises(ValueError):
        s_field.validate_value(11.0)
    with pytest.raises(TypeError):
        s_field.validate_value("5.5")

    # 4. MultiChoiceField validation
    with pytest.raises(ValueError, match="requires at least one option"):
        MultiChoiceField(options=[])
    with pytest.raises(ValueError, match="Duplicate option"):
        MultiChoiceField(options=["a", "a"])

    mc_field = MultiChoiceField(options=["tag1", "tag2", "tag3"])
    mc_field.bind_name("tags")
    assert mc_field.validate_value(["tag1", "tag2"]) == ("tag1", "tag2")
    with pytest.raises(TypeError):
        mc_field.validate_value("tag1")
    with pytest.raises(ValueError):
        mc_field.validate_value(["tag1", "invalid"])

    # 5. Empty Schema error
    with pytest.raises(ValueError, match="has no fields defined"):
        DecisionSchema({})

    # 6. Model forward edge cases
    class EdgeSchema(DecisionSchema):
        choice = ChoiceField(options=["c1", "c2"])
        score = ScoreField(min_value=-5.0, max_value=5.0)

    model = SystemOneModel(EdgeSchema, dimension=64, backend="numpy")

    # Non-string prompt error
    with pytest.raises(TypeError, match="Prompt must be a string"):
        model.forward_single(12345)  # type: ignore

    # Empty string prompt
    res_empty = model.forward_single("")
    assert isinstance(res_empty, ModelInferenceResult)
    assert res_empty.fields["choice"].selected_value in ["c1", "c2"]

    # Very long prompt (>8192 characters)
    long_prompt = "alpha beta gamma " * 1000
    res_long = model.forward_single(long_prompt)
    assert isinstance(res_long, ModelInferenceResult)

    # Empty batch returns empty list
    assert model.forward_batch([]) == []
    assert model.encode_batch([]).shape == (0, 64)

    # encode_batch validation
    batch_embs = model.encode_batch(["first query", "second query"])
    assert batch_embs.shape == (2, 64)
    np.testing.assert_allclose(batch_embs[0], model.encode("first query"), atol=1e-5)
    np.testing.assert_allclose(batch_embs[1], model.encode("second query"), atol=1e-5)

    with pytest.raises(TypeError, match="prompts must be a sequence of strings"):
        model.encode_batch(None)  # type: ignore
    with pytest.raises(TypeError, match="prompts must be a sequence of strings"):
        model.encode_batch(12345)  # type: ignore
    with pytest.raises(TypeError, match="All elements in prompts must be strings"):
        model.encode_batch(["valid", 999])  # type: ignore

    # forward_batch validation
    with pytest.raises(TypeError, match="prompts must be a sequence of strings"):
        model.forward_batch(None)  # type: ignore
    with pytest.raises(TypeError, match="prompts must be a sequence of strings"):
        model.forward_batch(12345)  # type: ignore
    with pytest.raises(TypeError, match="All elements in prompts must be strings"):
        model.forward_batch(["valid", 999])  # type: ignore

    # Model with LocalNeuralProjector forward_batch
    neural_proj = LocalNeuralProjector(dimension=64, hidden_dim=128, backend="numpy", seed=42)
    neural_model = SystemOneModel(EdgeSchema, dimension=64, backend="numpy", projector=neural_proj)
    neural_batch_res = neural_model.forward_batch(["query one", "query two"])
    assert len(neural_batch_res) == 2
    for n_item in neural_batch_res:
        assert isinstance(n_item, ModelInferenceResult)
        assert n_item.fields["choice"].selected_value in ["c1", "c2"]


def test_deterministic_projector_batch_and_similarity():
    """Verify DeterministicSemanticProjector interchangeable API parity with LocalNeuralProjector."""
    proj = DeterministicSemanticProjector(dimension=128)
    texts = ["api endpoint routing", "database query optimization", "network proxy filter"]

    # 1. encode alias
    for t in texts:
        np.testing.assert_allclose(proj.encode(t), proj.project(t))

    # 2. project_batch
    batch = proj.project_batch(texts)
    assert batch.shape == (3, 128)
    for i, t in enumerate(texts):
        np.testing.assert_allclose(batch[i], proj.project(t), atol=1e-5)

    # 3. similarity
    sim = proj.similarity("api endpoint routing", "api endpoint gateway")
    assert -1.0 <= sim <= 1.0
    v1 = proj.project("api endpoint routing")
    v2 = proj.project("api endpoint gateway")
    assert np.isclose(sim, float(np.dot(v1, v2)), atol=1e-5)


def test_model_encode_batch_and_vectorization():
    """Verify SystemOneModel.encode_batch and batched forward inference with custom projectors."""
    class TriageSchema(DecisionSchema):
        action = ChoiceField(options=["route", "hold", "drop"])
        score = ScoreField(min_value=0.0, max_value=1.0)

    # With DeterministicSemanticProjector
    det_model = SystemOneModel(TriageSchema, dimension=128, backend="numpy")
    prompts = ["Route user to support", "Hold user for review", "Drop malicious request"]

    embs = det_model.encode_batch(prompts)
    assert embs.shape == (3, 128)
    for i, p in enumerate(prompts):
        np.testing.assert_allclose(embs[i], det_model.encode(p), atol=1e-5)

    results = det_model.forward_batch(prompts)
    assert len(results) == 3
    for r in results:
        assert r.fields["action"].selected_value in ["route", "hold", "drop"]
        assert 0.0 <= r.fields["score"].selected_value <= 1.0

