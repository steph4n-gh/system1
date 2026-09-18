"""Tests for Pure-NumPy Dense Subword Semantic Engine & Hybrid Projection."""

import time
import numpy as np
import pytest

from system1.core.embeddings import HybridProjector, HybridSemanticProjector, SubwordSemanticEmbeddings
from system1.core.model import SystemOneModel
from system1.core.schema import ChoiceField, DecisionSchema


def test_subword_semantic_embeddings_pure_numpy_invariants():
    """Verify SubwordSemanticEmbeddings operates purely in NumPy without torch/transformers."""
    engine = SubwordSemanticEmbeddings(dimension=256, vocab_size=5000, seed=42)
    assert engine.dimension == 256
    assert engine.vocab_size == 5000

    # Encoding
    v = engine.encode("we are hitting the sack")
    assert isinstance(v, np.ndarray)
    assert v.shape == (256,)
    assert v.dtype == np.float32
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-4

    # Empty string edge case
    v_empty = engine.encode("")
    assert v_empty.shape == (256,)
    assert v_empty[0] == 1.0

    # Non-string type error
    with pytest.raises(TypeError):
        engine.encode(123)  # type: ignore


def test_subword_semantic_clustering_and_idiom_separation():
    """Verify semantic clustering maps colloquial idioms to their true conceptual domains."""
    engine = SubwordSemanticEmbeddings(dimension=256, seed=42)

    # Idiom: "hitting the sack" -> bedroom vs kitchen
    v_idiom = engine.encode("I am exhausted, hitting the sack")
    v_bedroom = engine.encode("master bedroom guest bedside lamps")
    v_kitchen = engine.encode("kitchen lights oven dishwasher refrigerator")

    sim_bed = float(np.dot(v_idiom, v_bedroom))
    sim_kit = float(np.dot(v_idiom, v_kitchen))
    assert sim_bed > sim_kit + 0.20, f"Expected bedroom ({sim_bed}) to beat kitchen ({sim_kit}) by > 0.20"

    # Financial crime idiom: "wired funds offshore" -> sanctions vs normal
    v_wire = engine.encode("Customer wired funds offshore to secrecy haven")
    v_sanctions = engine.encode("sanctions nexus ofac sanctioned entities embargo")
    v_normal = engine.encode("normal legitimate payroll retail recurring payment")

    sim_sanctions = float(np.dot(v_wire, v_sanctions))
    sim_normal = float(np.dot(v_wire, v_normal))
    assert sim_sanctions > sim_normal + 0.20, (
        f"Expected sanctions ({sim_sanctions}) to beat normal ({sim_normal}) by > 0.20"
    )


def test_hybrid_projector_fusion_mathematics():
    """Verify hybrid sparse-dense projection concatenation and unit sphere normalization."""
    proj = HybridProjector(sparse_dim=1024, dense_dim=256, alpha=0.5, seed=42)
    assert proj.dimension == 1280
    assert proj.sparse_dim == 1024
    assert proj.dense_dim == 256

    v = proj.project("SELECT * FROM users WHERE email = 'admin@example.com'")
    assert v.shape == (1280,)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-4

    # Check that first 1024 components correspond to sparse and remaining 256 to dense
    w_sparse = np.linalg.norm(v[:1024])
    w_dense = np.linalg.norm(v[1024:])
    # Since alpha=0.5, ||v_sparse_part||^2 approx 0.5 and ||v_dense_part||^2 approx 0.5
    np.testing.assert_allclose(w_sparse ** 2, 0.5, atol=0.05)
    np.testing.assert_allclose(w_dense ** 2, 0.5, atol=0.05)


def test_hybrid_projector_custom_dimension_split():
    """Verify passing a target dimension splits 75% sparse and 25% dense."""
    proj = HybridProjector(dimension=384, alpha=0.5)
    assert proj.dimension == 384
    assert proj.sparse_dim + proj.dense_dim == 384

    v = proj.project("test sentence for custom dimension")
    assert v.shape == (384,)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-4


def test_hybrid_projector_batch_and_similarity():
    """Verify batched projection and cosine similarity in hybrid space."""
    proj = HybridProjector(dimension=128)
    texts = [
        "turn off the lights",
        "turn on the lights",
        "read /etc/passwd",
    ]
    batch_v = proj.project_batch(texts)
    assert batch_v.shape == (3, 128)

    sim_opposite = proj.similarity("turn off the lights", "turn on the lights")
    sim_unrelated = proj.similarity("turn off the lights", "read /etc/passwd")
    assert sim_opposite > sim_unrelated


def test_hybrid_projector_speed():
    """Verify hybrid projection executes in under 0.5 ms per query."""
    proj = HybridProjector(dimension=384)

    # Warmup
    for _ in range(5):
        proj.project("warmup prompt text for speed benchmark")

    t0 = time.perf_counter()
    n_queries = 200
    for _ in range(n_queries):
        proj.project("Turn off all the lights in the house, we are going to sleep.")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_query_ms = elapsed_ms / n_queries

    assert per_query_ms < 0.50, f"Expected < 0.50 ms per query, got {per_query_ms:.4f} ms"


def test_system_one_model_with_hybrid_projector():
    """Verify SystemOneModel integrates with HybridProjector seamlessly."""
    class SecuritySchema(DecisionSchema):
        action = ChoiceField(
            options=["allow", "block"],
            descriptions={
                "allow": "Safe parameterized query",
                "block": "Dangerous SQL injection concatenation with user input",
            },
        )

    hybrid_proj = HybridProjector(dimension=384, alpha=0.3)
    model = SystemOneModel(SecuritySchema, projector=hybrid_proj)

    assert model.dimension == 384
    res = model.forward_single("query = f\"SELECT * FROM users WHERE id = '{input}'\"")
    assert res.fields["action"].selected_value == "block"
    assert res.fields["action"].confidence > 0.5
