"""Feature composition preserves meanings and the saved feature-space identity."""
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from system1.core.text import TfidfProjector


@pytest.fixture
def fused(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks/quality/n8n_gauntlet"))
    import fused_features
    return fused_features


class FakeEncoder:
    dimension = 2

    def project(self, text):
        return np.asarray([.6, .8], dtype=np.float32)

    def projector_digest(self):
        return "fixed-authored-encoder"


def test_fusion_preserves_both_components_and_unknown_words(fused):
    lexical = TfidfProjector.fit(["reset password", "billing refund"])
    projector = fused.FusedFeatures(FakeEncoder(), lexical)
    vector = projector.project("reset password")
    np.testing.assert_allclose(vector[:2], [.6 / np.sqrt(2), .8 / np.sqrt(2)], atol=1e-7)
    np.testing.assert_allclose(vector[2:], lexical.project("reset password") / np.sqrt(2), atol=1e-7)
    unknown = projector.project("unseenword")
    np.testing.assert_allclose(unknown[:2], [.6, .8], atol=1e-7)
    assert np.count_nonzero(unknown[2:]) == 0 and np.linalg.norm(unknown) == pytest.approx(1.)


def test_saved_lexical_config_preserves_projection_and_binds_identity(fused):
    lexical = TfidfProjector.fit(["reset password", "billing refund"])
    original = fused.FusedFeatures(FakeEncoder(), lexical)
    restored = fused.FusedFeatures(FakeEncoder(), TfidfProjector.from_config(json.loads(json.dumps(lexical.to_config()))))
    assert original.projector_digest() == restored.projector_digest()
    np.testing.assert_array_equal(original.project("refund"), restored.project("refund"))
    changed = fused.FusedFeatures(FakeEncoder(), TfidfProjector.fit(["reset password", "reset password", "billing refund"]))
    assert changed.projector_digest() != original.projector_digest()
    with pytest.raises(ValueError, match="recency"):
        original.project("refund", recency_weighted=True)


def test_fusion_has_no_optional_library_dependency():
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
import fused_features
assert len(fused_features.join_features([1.,0.], [0.,1.])) == 4
"""
    subprocess.run([sys.executable, "-c", script, str(folder)], check=True, capture_output=True, text=True)
