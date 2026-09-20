"""Learned text features must be portable, bounded, and isolated from calibration."""
from dataclasses import FrozenInstanceError
import json
import struct

import numpy as np
import pytest

from system1 import ChoiceField, DecisionSchema, System1Engine, SystemOneCompiler, TfidfProjector
from system1.compiler import CompiledSystemOneModel


class Route(DecisionSchema):
    category = ChoiceField(options=["billing", "support"])


def test_known_tfidf_math_and_word_pairs():
    projector = TfidfProjector.fit(["red red blue", "blue green"])
    assert projector.to_config()["terms"] == ["blue", "blue green", "green", "red", "red blue", "red red"]
    np.testing.assert_allclose(projector.to_config()["idf"], [1, *([np.log(3 / 2) + 1] * 5)])
    expected = np.array([1, 0, 0, (1 + np.log(2)) * (1 + np.log(1.5)), 1 + np.log(1.5), 1 + np.log(1.5)])
    np.testing.assert_allclose(projector.project("red red blue"), expected / np.linalg.norm(expected), rtol=1e-6)
    np.testing.assert_array_equal(projector.project("BLUE red"), projector.project("blue RED"))
    assert not projector.project("unseen vocabulary").any()
    assert projector.project_batch([]).shape == (0, projector.dimension)


def test_features_are_deterministic_immutable_and_isolated():
    texts = ["same common first", "same common second"]
    first = TfidfProjector.fit(texts, max_features=3)
    second = TfidfProjector.fit(reversed(texts), max_features=3)
    assert first.to_config() == second.to_config()
    before = first.to_config()
    first.project("heldout special vocabulary")
    assert first.to_config() == before
    with pytest.raises(FrozenInstanceError):
        first._idf = (1, 1, 1)
    before["terms"][0] = "mutated copy"
    assert first.to_config() == second.to_config()
    with pytest.raises(ValueError, match="recency"):
        first.project("text", recency_weighted=True)


@pytest.mark.parametrize("config", [
    {"type": "tfidf", "terms": [], "idf": []},
    {"type": "tfidf", "terms": ["word", "word"], "idf": [1, 1]},
    {"type": "tfidf", "terms": ["word"], "idf": [float("nan")]},
    {"type": "tfidf", "terms": ["word"], "idf": [float("inf")]},
    {"type": "tfidf", "terms": ["word"], "idf": [True]},
    {"type": "tfidf", "terms": ["word"], "idf": [0]},
    {"type": "tfidf", "terms": ["word"], "idf": [[1]]},
    {"type": "tfidf", "terms": ["x" * 129], "idf": [1]},
    {"type": "tfidf", "terms": [str(i) for i in range(4097)], "idf": [1] * 4097},
    {"type": "tfidf", "terms": ["word"], "idf": [1], "tokenizer": "import.me"},
])
def test_invalid_saved_features_are_rejected(config):
    with pytest.raises(ValueError):
        TfidfProjector.from_config(config)


@pytest.fixture
def skill():
    teach = [("refund payment invoice", "billing"), ("software crash error", "support"),
             ("credit card charge", "billing"), ("broken software bug", "support")]
    projector = TfidfProjector.fit([text for text, _ in teach])
    return SystemOneCompiler(Route, projector=projector, regularization=.1).compile(
        {"category": teach}, augment=False,
        calibration_exemplars={"category": [(f"{text} calibration{i}", label) for i in range(10) for text, label in teach]},
    )


def test_saved_skill_carries_its_features_and_review_policy(skill):
    loaded = CompiledSystemOneModel.from_bytes(skill.to_bytes())
    assert isinstance(loaded.projector, TfidfProjector)
    assert loaded.projector.to_config() == skill.projector.to_config()
    before = System1Engine(Route, model=skill, use_cache=False, strict_mode=True)
    after = System1Engine(Route, model=loaded, use_cache=False, strict_mode=True)
    for text in ("refund invoice", "software error", "unseen vocabulary"):
        a, b = before.decide(text, record_receipt=False), after.decide(text, record_receipt=False)
        assert a.values == b.values and a.probabilities == b.probabilities
        assert a.conformal_sets == b.conformal_sets and a.is_ambiguous == b.is_ambiguous
    assert after.decide("unseen vocabulary", record_receipt=False).is_ambiguous
    # Even a strongly biased head may not turn unknown vocabulary into acceptance.
    loaded._field_heads["category"].biases[:] = [100, -100]
    assert after.decide("unseen vocabulary", record_receipt=False).is_ambiguous
    assert not any("calibration" in word for word in loaded.projector.to_config()["terms"])


def test_saved_feature_dimension_and_supplied_projector_must_match(skill):
    data = skill.to_bytes()
    size = struct.unpack(">I", data[4:8])[0]
    header = json.loads(data[8:8 + size])
    header["projector"]["terms"].pop()
    header["projector"]["idf"].pop()
    altered = json.dumps(header).encode()
    with pytest.raises(ValueError, match="dimension"):
        CompiledSystemOneModel.from_bytes(data[:4] + struct.pack(">I", len(altered)) + altered + data[8 + size:])
    with pytest.raises(ValueError, match="feature space"):
        CompiledSystemOneModel.from_bytes(data, projector=TfidfProjector.fit(["different terms"]))
