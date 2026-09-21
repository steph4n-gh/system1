"""Context review serialization must preserve a dense or sparse taught score."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.fixture
def scores(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks/quality/n8n_gauntlet"))
    import context_scores
    from polynomial_scores import load_review
    return context_scores, load_review


@pytest.mark.parametrize("sparse_input", [False, True])
def test_context_json_scores_match_fitted_model(scores, sparse_input):
    linear = pytest.importorskip("sklearn.linear_model")
    preprocessing = pytest.importorskip("sklearn.preprocessing")
    sparse = pytest.importorskip("scipy.sparse")
    module, load_review = scores
    rng = np.random.default_rng(71)
    signals, context = rng.normal(size=(250, 7)), rng.normal(size=(250, 12))
    context /= np.linalg.norm(context, axis=1, keepdims=True)
    category = rng.integers(0, 3, size=len(signals))
    target = signals[:, 0] - signals[:, 2] + 5 * context[:, 1] + category * .2 > 0
    scaler = preprocessing.StandardScaler().fit(signals)
    matrix = np.column_stack([scaler.transform(signals), np.eye(3)[category], context])
    if sparse_input:
        matrix = sparse.csr_matrix(matrix)
    fitted = linear.LogisticRegression(C=.1, max_iter=1000).fit(matrix, target)
    manifest = json.loads(json.dumps(dict(quadratic=False, context_dimension=12,
        context_weights=fitted.coef_[0, 10:].tolist(), review_parameters=dict(mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(), weights=fitted.coef_[0, :10].tolist(), bias=float(fitted.intercept_[0])))))
    parameters, weights = load_review(manifest, 3), module.load_context(manifest, 12)
    input_vectors = sparse.csr_matrix(context) if sparse_input else context
    logits = np.asarray(input_vectors @ weights).ravel()
    actual = [module.score(row, int(chosen), float(value), parameters)
              for row, chosen, value in zip(signals, category, logits, strict=True)]
    np.testing.assert_allclose(actual, fitted.predict_proba(matrix)[:, 1], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("manifest", [dict(context_dimension=True, context_weights=[0.]),
    dict(context_dimension=2, context_weights=[0.]), dict(context_dimension=1, context_weights=[float("nan")]),
    dict(context_dimension=1, context_weights=[]), dict(context_dimension=0, context_weights=[])])
def test_invalid_context_rejected(scores, manifest):
    module, _ = scores
    with pytest.raises(ValueError):
        module.load_context(manifest, 1)


def test_context_helper_imports_without_optional_libraries():
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
import context_scores
assert context_scores.load_context(dict(context_dimension=1, context_weights=[2.]), 1).tolist() == [2.]
"""
    subprocess.run([sys.executable, "-c", script, str(folder)], check=True, capture_output=True, text=True)
