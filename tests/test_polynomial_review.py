"""The portable review score must match the teaching library after JSON reload."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def test_numerical_scores_import_without_optional_benchmark_libraries():
    folder = Path(__file__).resolve().parents[1] / "benchmarks/quality/n8n_gauntlet"
    script = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'threadpoolctl', 'scipy', 'sklearn', 'onnxruntime', 'tokenizers'}:
            raise ImportError('Optional benchmark dependency: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
sys.path.insert(0, sys.argv[1])
import polynomial_scores
assert polynomial_scores.score([0.] * 7, 0, False,
    polynomial_scores.load_review(dict(quadratic=False, review_parameters=dict(
        mean=[0.] * 7, scale=[1.] * 7, weights=[0.] * 8, bias=0.)), 1)) == .5
"""
    subprocess.run([sys.executable, "-c", script, str(folder)], check=True, capture_output=True, text=True)


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks/quality/n8n_gauntlet"))
    import polynomial_scores
    return polynomial_scores


@pytest.mark.parametrize("quadratic", [False, True])
def test_json_reload_matches_sklearn_review_scores(runtime, quadratic):
    preprocessing = pytest.importorskip("sklearn.preprocessing")
    linear = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(41)
    rows = rng.normal(size=(250, 7))
    categories = rng.integers(0, 3, size=len(rows))
    target = rows[:, 0] * rows[:, 1] + .5 * rows[:, 2] ** 2 - rows[:, 3] + .2 * categories > 0
    expected = preprocessing.PolynomialFeatures(degree=2 if quadratic else 1, include_bias=False).fit_transform(rows)
    np.testing.assert_array_equal(runtime.expand(rows, quadratic), expected)
    scaler = preprocessing.StandardScaler().fit(expected)
    x = np.column_stack([scaler.transform(expected), np.eye(3)[categories]])
    model = linear.LogisticRegression(C=.1, max_iter=1000).fit(x, target)
    manifest = json.loads(json.dumps(dict(quadratic=quadratic, review_parameters=dict(
        mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), weights=model.coef_[0].tolist(), bias=float(model.intercept_[0])))))
    parameters = runtime.load_review(manifest, 3)
    actual = [runtime.score(row, int(category), quadratic, parameters) for row, category in zip(rows, categories, strict=True)]
    np.testing.assert_allclose(actual, model.predict_proba(x)[:, 1], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("change", ["zero_scale", "nan_weight", "wrong_dimension", "non_boolean_flag"])
def test_invalid_numerical_artifacts_are_rejected(runtime, change):
    manifest = dict(quadratic=True, review_parameters=dict(mean=[0.] * 35, scale=[1.] * 35,
                                                         weights=[0.] * 38, bias=0.))
    if change == "zero_scale":
        manifest["review_parameters"]["scale"][0] = 0.
    elif change == "nan_weight":
        manifest["review_parameters"]["weights"][0] = float("nan")
    elif change == "wrong_dimension":
        manifest["review_parameters"]["weights"].pop()
    else:
        manifest["quadratic"] = 1
    with pytest.raises(ValueError):
        runtime.load_review(manifest, 3)
