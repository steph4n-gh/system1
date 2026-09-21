"""Authored transform checks without fitting or optional benchmark dependencies."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.fixture
def shallow(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks/quality/n8n_gauntlet"))
    import shallow_features
    return shallow_features


class Encoder:
    dimension = 2
    calls = 0

    def project(self, text):
        self.calls += 1
        return np.asarray([.6, .8], dtype=np.float32)

    def projector_digest(self):
        return "authored-fixed-encoder"


def test_taught_transform_preserves_semantics_and_uses_one_encoder(shallow):
    encoder = Encoder()
    projector = shallow.ShallowFeatures(encoder, [[2, -1], [0, 1]], [-1, 0])
    vector = projector.project("sample")
    np.testing.assert_allclose(vector, [.6 / np.sqrt(2), .8 / np.sqrt(2), .5, .5], atol=1e-6)
    assert encoder.calls == 1 and np.linalg.norm(vector) == pytest.approx(1)
    empty = shallow.ShallowFeatures(encoder, -np.ones((2, 2)), [-1, -1]).project("sample")
    np.testing.assert_array_equal(empty, np.asarray([.6, .8, 0, 0], dtype=np.float32))


def test_transform_identity_survives_safe_serialization(shallow, tmp_path):
    original = shallow.ShallowFeatures(Encoder(), [[2, -1], [0, 1]], [-1, 0])
    path = tmp_path / "transform.npz"
    np.savez_compressed(path, weights=original.weights, bias=original.bias)
    restored = shallow.ShallowFeatures.load(Encoder(), path)
    assert restored.projector_digest() == original.projector_digest()
    np.testing.assert_array_equal(restored.project("sample"), original.project("sample"))
    changed = shallow.ShallowFeatures(Encoder(), [[2, -1], [0, 2]], [-1, 0])
    assert changed.projector_digest() != original.projector_digest()
    with pytest.raises(ValueError):
        original.weights[0, 0] = 3


def test_transform_rejects_bad_parameters_inputs_and_recency(shallow):
    for weights, bias in [([1, 2], [0]), ([[1], [2]], [0, 1]), ([[np.nan], [2]], [0])]:
        with pytest.raises(ValueError, match="Invalid taught"):
            shallow.ShallowFeatures(Encoder(), weights, bias)
    projector = shallow.ShallowFeatures(Encoder(), [[1], [2]], [0])
    with pytest.raises(ValueError, match="semantic"):
        projector.transform([1, 2, 3])
    with pytest.raises(ValueError, match="semantic"):
        projector.transform([1, np.inf])
    with pytest.raises(ValueError, match="recency"):
        projector.project("sample", recency_weighted=True)


def test_transform_import_and_inference_without_fitting_libraries():
    folder = Path(__file__).resolve().parents[1] / "benchmarks/quality/n8n_gauntlet"
    script = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'threadpoolctl', 'scipy', 'sklearn', 'onnxruntime', 'tokenizers'}:
            raise ImportError('Optional dependency: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
sys.path.insert(0, sys.argv[1])
from shallow_features import ShallowFeatures
class Encoder:
    dimension = 2
    def project(self, text): return [1., 0.]
    def projector_digest(self): return 'authored'
assert ShallowFeatures(Encoder(), [[1.], [0.]], [0.]).project('sample').shape == (3,)
"""
    subprocess.run([sys.executable, "-c", script, str(folder)], check=True, capture_output=True, text=True)
