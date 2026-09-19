"""Tests for LocalNeuralProjector in system1."""

import numpy as np
import pytest

import system1.core.neural as core_neural
import system1.neural as system1_neural
from system1.neural import LocalNeuralProjector
from system1 import SystemOneEngine, DecisionSchema, ChoiceField


def test_neural_projector_parity():
    """Verify system1.neural re-exports the exact class from system1.core.neural."""
    assert system1_neural.LocalNeuralProjector is core_neural.LocalNeuralProjector


def test_neural_projector_output_shape_and_normalization():
    """Verify projector outputs normalized vectors of exact specified dimension."""
    proj = LocalNeuralProjector(dimension=384, hidden_dim=256, backend="numpy", seed=42)

    vec = proj.project("Analyze high-frequency telemetry data for anomalies")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert vec.dtype == np.float32

    # Vector must be unit L2-norm
    norm = float(np.linalg.norm(vec))
    assert pytest.approx(norm, rel=1e-5) == 1.0


def test_neural_projector_batch_forward():
    """Verify batched projection output matches single projections."""
    proj = LocalNeuralProjector(dimension=128, hidden_dim=256, backend="numpy", seed=123)

    texts = [
        "Read configuration file from disk",
        "Send database query across network socket",
        "Execute bash shell subprocess",
    ]

    batch_vecs = proj.project_batch(texts)
    assert batch_vecs.shape == (3, 128)

    for i, t in enumerate(texts):
        single_vec = proj.project(t)
        np.testing.assert_allclose(batch_vecs[i], single_vec, rtol=1e-5, atol=1e-5)


def test_neural_projector_semantic_discrimination():
    """Verify semantically similar sentences have higher cosine similarity than unrelated ones."""
    proj = LocalNeuralProjector(dimension=384, backend="numpy", seed=42)

    sim_synonym = proj.similarity(
        "Read local file from disk system",
        "Inspect file content on local storage system",
    )

    sim_unrelated = proj.similarity(
        "Read local file from disk system",
        "Bake chocolate chip cookies with fresh vanilla extract and flour",
    )

    assert sim_synonym > sim_unrelated


def test_neural_projector_empty_and_long_inputs():
    """Verify edge cases: empty strings, pure whitespace, very long text."""
    proj = LocalNeuralProjector(dimension=384, backend="numpy")

    v_empty = proj.project("")
    assert np.isfinite(v_empty).all()
    assert pytest.approx(float(np.linalg.norm(v_empty)), rel=1e-5) == 1.0

    v_spaces = proj.project("     \n\t  ")
    assert np.isfinite(v_spaces).all()
    assert pytest.approx(float(np.linalg.norm(v_spaces)), rel=1e-5) == 1.0

    long_text = "word " * 5000
    v_long = proj.project(long_text)
    assert np.isfinite(v_long).all()
    assert pytest.approx(float(np.linalg.norm(v_long)), rel=1e-5) == 1.0


def test_neural_projector_engine_integration():
    """Verify LocalNeuralProjector can be plugged into SystemOneEngine."""
    class TriageSchema(DecisionSchema):
        route = ChoiceField(
            options=["code", "db", "chat"],
            descriptions={
                "code": "python code syntax and indentation errors",
                "db": "database SQL queries",
                "chat": "conversational chat",
            },
        )

    proj = LocalNeuralProjector(dimension=384, seed=99)
    engine = SystemOneEngine(TriageSchema, projector=proj, backend="numpy")

    result = engine.decide("Fix python indent error in line 25")
    assert result.route == "code"
    assert result.latency_ms < 25.0


def test_neural_projector_mlx_numpy_parity():
    """Verify numerical parity between Apple Silicon MLX and NumPy backends across all activations."""
    from system1.neural import HAS_MLX
    if not HAS_MLX:
        pytest.skip("MLX not installed on this system")

    for act in ["gelu", "relu", "tanh"]:
        proj_np = LocalNeuralProjector(dimension=256, hidden_dim=384, backend="numpy", activation=act, seed=777)
        proj_mx = LocalNeuralProjector(dimension=256, hidden_dim=384, backend="mlx", activation=act, seed=777)

        sample_prompts = [
            "Inspect system performance logs and kernel panics",
            "SELECT * FROM accounts WHERE balance > 1000",
            "Bake artisan sourdough bread with whole wheat flour",
        ]

        for p in sample_prompts:
            v_np = proj_np.project(p)
            v_mx = proj_mx.project(p)
            np.testing.assert_allclose(
                v_np, v_mx, atol=1e-4, rtol=1e-4,
                err_msg=f"MLX vs NumPy parity failure for activation={act!r} on prompt={p!r}",
            )

        # Batch parity test
        batch_np = proj_np.project_batch(sample_prompts)
        batch_mx = proj_mx.project_batch(sample_prompts)
        np.testing.assert_allclose(
            batch_np, batch_mx, atol=1e-4, rtol=1e-4,
            err_msg=f"MLX vs NumPy batch parity failure for activation={act!r}",
        )


def test_neural_projector_parameter_validation():
    """Verify bounds checking and type validation on constructor and methods."""
    with pytest.raises(ValueError, match="dimension must be positive"):
        LocalNeuralProjector(dimension=0)

    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        LocalNeuralProjector(hidden_dim=-1)

    with pytest.raises(ValueError, match="feature_dim must be positive"):
        LocalNeuralProjector(feature_dim=0)

    with pytest.raises(ValueError, match="Unsupported activation"):
        LocalNeuralProjector(activation="sigmoid_unknown")

    proj = LocalNeuralProjector(dimension=128)
    with pytest.raises(TypeError, match="Expected text to be a string"):
        proj.project(12345)  # type: ignore

    with pytest.raises(TypeError, match="All elements in texts must be strings"):
        proj.project_batch(["valid text", 42])  # type: ignore

