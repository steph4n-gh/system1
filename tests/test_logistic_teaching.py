"""Cross-entropy teaching must preserve the existing portable runtime contract."""
import builtins
from types import SimpleNamespace

import numpy as np
import pytest

from system1 import ChoiceField, CompiledSystemOneModel, DecisionSchema, System1Engine, SystemOneCompiler
from system1.core.text import TfidfProjector


def test_logistic_two_choice_bias_recovers_observed_frequency():
    # With no signal, the maximum-likelihood answer is the observed base rate.
    x = np.zeros((100, 3))
    y = np.eye(2)[[0] * 80 + [1] * 20]
    weights, biases = SystemOneCompiler._solve_logistic(x, y, .1)
    probabilities = np.exp((biases - biases.max()) / .25)
    probabilities /= probabilities.sum()
    np.testing.assert_allclose(probabilities, [.8, .2], atol=1e-5)
    np.testing.assert_array_equal(weights, np.zeros((2, 3)))


def test_logistic_saved_skill_needs_no_scipy_and_preserves_review(monkeypatch):
    class Fruit(DecisionSchema):
        intent = ChoiceField(options=["z", "a", "m"])

    names = [("apple", "z"), ("banana", "a"), ("grape", "m")]
    fit = [(f"{fruit} order {i}", label) for fruit, label in names for i in range(20)]
    checks = [(f"{fruit} delivery {i}", label) for fruit, label in names for i in range(20)]
    projector = TfidfProjector.fit([t for t, _ in fit])
    compiler = SystemOneCompiler(Fruit, projector=projector, regularization=.1, choice_solver="logistic")
    skill = compiler.compile({"intent": fit}, augment=False, calibration_exemplars={"intent": checks})
    assert skill.metadata["choice_solver"] == "logistic"
    assert len(skill.heads["intent"].calibration_scores) == 30
    engine = System1Engine(Fruit, model=skill, strict_mode=True, use_cache=False)
    prompts = [f"order {fruit} please" for fruit, _ in names] + ["completely unfamiliar"]
    before = [engine.decide(text, record_receipt=False) for text in prompts]
    assert [r.values["intent"] for r in before[:3]] == [label for _, label in names]
    original_import = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("Teaching dependency unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scipy)
    restored = CompiledSystemOneModel.from_bytes(skill.to_bytes())
    after = System1Engine(restored.schema, model=restored, strict_mode=True, use_cache=False)
    for text, expected in zip(prompts, before):
        actual = after.decide(text, record_receipt=False)
        assert actual.values == expected.values
        assert actual.probabilities == expected.probabilities
        assert actual.conformal_sets == expected.conformal_sets
        assert actual.is_ambiguous == expected.is_ambiguous
    assert before[-1].is_ambiguous
    with pytest.raises(ImportError, match=r"system1\[teaching\]"):
        compiler.compile({"intent": fit}, augment=False, calibration_exemplars={"intent": checks})


def test_logistic_rejects_missing_choices_and_failed_optimizer(monkeypatch):
    import scipy.optimize

    with pytest.raises(ValueError, match="every choice"):
        SystemOneCompiler._solve_logistic(np.ones((3, 2)), np.array([[1, 0]] * 3), .1)
    monkeypatch.setattr(scipy.optimize, "minimize", lambda *a, **k: SimpleNamespace(success=False))
    with pytest.raises(RuntimeError, match="did not converge"):
        SystemOneCompiler._solve_logistic(np.ones((3, 2)), np.array([[1, 0], [0, 1], [1, 0]]), .1)


def test_choice_solver_is_explicit_and_validated():
    class Answer(DecisionSchema):
        choice = ChoiceField(options=["yes", "no"])

    assert SystemOneCompiler(Answer).choice_solver == "ridge"
    with pytest.raises(ValueError, match="choice_solver"):
        SystemOneCompiler(Answer, choice_solver="unknown")
