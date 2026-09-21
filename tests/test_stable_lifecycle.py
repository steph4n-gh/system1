"""Regression checks for calibration, correction, and portable skill evidence."""

import numpy as np
import pytest

from system1 import BooleanField, ChoiceField, DecisionSchema, MultiChoiceField, ScoreField, System1Engine
from system1.calibration import ConformalPredictor
from system1.compiler import CompiledHeadWeights, CompiledSystemOneModel


class Outputs(DecisionSchema):
    route = ChoiceField(options=["billing", "support"])
    active = BooleanField()
    tags = MultiChoiceField(options=["urgent", "external"])
    score = ScoreField(min_value=0, max_value=10)


def constant_skill():
    heads = {}
    for name, definition in Outputs().fields.items():
        bias = {"route": [1, -1], "active": [1], "tags": [-1, -1], "score": [0]}[name]
        options = getattr(definition, "options", ("False", "True") if name == "active" else ())
        heads[name] = CompiledHeadWeights(
            name, definition.field_type, np.zeros((len(bias), 16), dtype=np.float32),
            np.asarray(bias, dtype=np.float32), options=options,
            score_method="lac" if name in ("route", "active") else "aps",
        )
    return CompiledSystemOneModel(Outputs(), heads, dimension=16, use_cache=False)


def test_lac_accepts_high_confidence_and_defers_boundary():
    predictor = ConformalPredictor("route", ["a", "b"], score_method="lac")
    predictor.calibrate(np.tile([0.9, 0.1], (100, 1)), ["a"] * 100)
    assert predictor.predict_set([0.99, 0.01], strict=True).prediction_set == ("a",)
    assert predictor.predict_set([0.5, 0.5], strict=True).needs_escalation
    # The finite-sample correction still requires all labels with too little evidence.
    predictor.calibrate(np.tile([0.9, 0.1], (3, 1)), ["a"] * 3)
    assert predictor.predict_set([0.99, 0.01], strict=True).prediction_set == ("a", "b")


def test_calibration_and_correction_survive_serialization():
    skill = constant_skill()
    engine = System1Engine(skill.schema, model=skill, strict_mode=True, use_cache=False)
    labels = {"route": "billing", "active": True, "tags": [], "score": 5.0}
    engine.calibrate([(f"calibration case {i}", labels) for i in range(100)])
    original = engine.decide("fresh request", record_receipt=False)
    assert not original.is_ambiguous
    assert original.values["tags"] == ()
    assert original.conformal_sets["tags"] == []

    restored = CompiledSystemOneModel.from_bytes(skill.to_bytes())
    fresh = System1Engine(restored.schema, model=restored, strict_mode=True, use_cache=False)
    reloaded = fresh.decide("fresh request", record_receipt=False)
    assert original.values == reloaded.values
    assert original.probabilities == reloaded.probabilities
    assert original.conformal_sets == reloaded.conformal_sets
    assert original.is_ambiguous == reloaded.is_ambiguous

    engine.learn_from_tier2("fresh request", {"route": "support", "tags": ["urgent"]})
    for name in ("route", "tags"):
        assert skill.heads[name].calibration_scores == ()
    updated = CompiledSystemOneModel.from_bytes(skill.to_bytes())
    updated_engine = System1Engine(updated.schema, model=updated, strict_mode=True)
    assert updated_engine.decide("fresh request", record_receipt=False).is_ambiguous
    assert not updated_engine.conformal_predictors["route"].is_calibrated
    assert not updated_engine.conformal_predictors["tags"].is_calibrated
    assert updated_engine.conformal_predictors["active"].is_calibrated

    revised_labels = {**labels, "route": "support", "tags": ["urgent"]}
    engine.calibrate([(f"new calibration {i}", revised_labels) for i in range(100)])
    revised = CompiledSystemOneModel.from_bytes(skill.to_bytes())
    reopened = System1Engine(revised.schema, model=revised, strict_mode=True, use_cache=False)
    assert reopened.conformal_predictors["route"].is_calibrated
    before = engine.decide("new request", record_receipt=False)
    after = reopened.decide("new request", record_receipt=False)
    assert before.probabilities == after.probabilities
    assert before.conformal_sets == after.conformal_sets
    assert before.is_ambiguous == after.is_ambiguous


def test_repeated_calibration_prompts_are_one_evidence_unit():
    engine = System1Engine(Outputs, dimension=16)
    examples = [("Invoice request", {"route": "billing"}), ("Error request", {"route": "support"})]
    engine.calibrate(examples * 30)
    assert len(engine.conformal_predictors["route"].calibration_scores) == 1
    assert sum(engine.calibrators["route"].metrics.bin_counts) == 1
    with pytest.raises(ValueError, match="Conflicting"):
        engine.calibrate(examples + [("  INVOICE   REQUEST ", {"route": "support"})])


def test_multilabel_review_uses_every_binary_assignment():
    skill = constant_skill()
    skill.heads["tags"].calibration_scores = (0.1,) * 100
    engine = System1Engine(skill.schema, model=skill, strict_mode=True, use_cache=False)
    result = engine.decide("No tags apply", record_receipt=False)
    assert "tags" not in result.ambiguous_fields
    engine.conformal_predictors["tags"].calibration_scores = np.full(100, 1.0)
    uncertain = engine.decide("No tags apply", record_receipt=False)
    assert "tags" in uncertain.ambiguous_fields
    assert uncertain.conformal_sets["tags"] == ["urgent", "external"]


def test_compiled_cache_cannot_replace_a_nearby_distinct_input():
    skill = constant_skill()
    skill.use_cache = True
    head = skill._field_heads["route"]
    weights = np.zeros((2, 16), dtype=np.float32)
    weights[:, 1] = [1000, -1000]
    head.set_weights(weights, np.zeros(2, dtype=np.float32))
    a, b = np.zeros(16, dtype=np.float32), np.zeros(16, dtype=np.float32)
    a[0] = b[0] = 1
    a[1], b[1] = .001, -.001
    first = skill.forward_single("first request", embedding=a)
    second = skill.forward_single("different request", embedding=b)
    assert first.fields["route"].selected_value == "billing"
    assert second.fields["route"].selected_value == "support"


def test_cache_receipts_bind_the_current_request_and_respect_opt_out():
    skill = constant_skill()
    engine = System1Engine(skill.schema, model=skill, use_cache=True, cache_threshold=0.01)
    embedding = np.ones(16, dtype=np.float32)
    first = engine.decide("first request", embedding=embedding)
    second = engine.decide("different request", embedding=embedding)
    assert second.is_cache_hit
    assert second.receipt.prompt == "different request"
    assert second.receipt.decision_id != first.receipt.decision_id
    from system1.receipt import check_decision_receipt_integrity
    assert check_decision_receipt_integrity(second.receipt.to_dict())
    assert engine.decide("different request", embedding=embedding, record_receipt=False).receipt is None


def test_uncached_receipt_opt_out_skips_signing_and_cache_digests(monkeypatch):
    import json
    import system1.engine as engine_module

    skill = constant_skill()
    engine = System1Engine(skill.schema, model=skill, use_cache=False)

    def unexpected(*args, **kwargs):
        raise AssertionError("Disabled receipt/cache work was executed")

    monkeypatch.setattr(engine, "_model_digest", unexpected)
    monkeypatch.setattr(engine, "_projector_digest", unexpected)
    monkeypatch.setattr(engine, "_calibration_digest", unexpected)
    monkeypatch.setattr(engine_module, "create_decision_receipt", unexpected)
    result = engine.decide("fresh request", record_receipt=False)
    assert result.receipt is None
    assert result.to_dict()["receipt"] is None
    assert json.loads(result.to_json())["receipt"] is None
    assert result.values["route"] == "billing"


@pytest.mark.parametrize("method", ["lac", "aps"])
def test_conformal_tail_counts_preserve_ties_and_option_order(method):
    predictor = ConformalPredictor("route", ["last", "first", "middle"], score_method=method)
    predictor.calibration_scores = np.array([0, .25, .25, .5, .75, .75, 1.0])
    predictor.is_calibrated = True
    for probabilities in ([.25, .5, .25], [0, 1, 0], [1 / 3] * 3):
        order = np.argsort(-np.asarray(probabilities))
        cumulative = {}
        mass = 0.0
        for i in order:
            mass += probabilities[i]
            cumulative[i] = mass
        expected = {}
        for i, option in enumerate(predictor.options):
            score = 1 - probabilities[i] if method == "lac" else cumulative[i]
            expected[option] = (1 + sum(value >= score for value in predictor.calibration_scores)) / 8
        assert predictor.predict_set(probabilities, strict=True).p_values == expected


def test_new_artifacts_record_scoring_format_and_legacy_aps_still_loads():
    import json
    import struct
    skill = constant_skill()
    for head in skill.heads.values():
        head.score_method = "aps"
    data = skill.to_bytes()
    assert data[:4] == b"S1M\x02"
    length = struct.unpack(">I", data[4:8])[0]
    meta = json.loads(data[8:8 + length])
    meta["version"] = 1
    for head in meta["heads"].values():
        head.pop("score_method")
    header = json.dumps(meta).encode()
    legacy = b"S1M\x01" + struct.pack(">I", len(header)) + header + data[8 + length:]
    restored = CompiledSystemOneModel.from_bytes(legacy)
    assert all(head.score_method == "aps" for head in restored.heads.values())


@pytest.mark.parametrize("corrupt", ["version", "schema_digest", "temperature", "score_method", "dimension"])
def test_saved_skill_rejects_incompatible_metadata(corrupt):
    import json
    import struct
    data = constant_skill().to_bytes()
    length = struct.unpack(">I", data[4:8])[0]
    meta = json.loads(data[8:8 + length])
    if corrupt in ("temperature", "score_method"):
        meta["heads"]["route"][corrupt] = float("nan") if corrupt == "temperature" else "unknown"
    else:
        meta[corrupt] = {"version": 99, "schema_digest": "wrong", "dimension": 17}[corrupt]
    header = json.dumps(meta).encode()
    with pytest.raises(ValueError):
        CompiledSystemOneModel.from_bytes(data[:4] + struct.pack(">I", len(header)) + header + data[8 + length:])


def test_promotion_counts_complete_independent_decisions():
    from types import SimpleNamespace
    from system1.compat.typesafe import PromotionPolicy, evaluate_promotion_eligibility
    class TwoFields(DecisionSchema):
        a = ChoiceField(options=["yes", "no"])
        b = ChoiceField(options=["yes", "no"])
    class Engine:
        def decide(self, prompt, **kwargs):
            return SimpleNamespace(values={"a": "yes", "b": "no"}, is_ambiguous=prompt != "review")
    rows = [{"state": "review", "answers": {"a": "yes", "b": "yes"}}] * 50
    rows.append({"state": "uncertain", "answers": {"a": "yes", "b": "yes"}})
    report = evaluate_promotion_eligibility(Engine(), rows, TwoFields(), PromotionPolicy())
    assert report.independent_validation_groups == 2
    assert report.agreement_rate == 0  # One correct field cannot hide the wrong field.
    assert report.local_acceptance_rate == .5  # Repeated accepted answers are one unit.
    assert not report.is_eligible


def test_exact_promotion_bound_and_attempt_budget():
    from system1.compat.typesafe import _binomial_lower_bound
    assert _binomial_lower_bound(10, 10, .025) == pytest.approx(.025 ** .1)
    assert _binomial_lower_bound(5, 10, .025) == pytest.approx(.1870860284)
    assert _binomial_lower_bound(5, 10, .001) < _binomial_lower_bound(5, 10, .025)


def test_drift_evidence_is_isolated_by_schema():
    from system1.compat.typesafe import Client, DriftDetector
    client = Client(drift_detector=DriftDetector(window_size=10))
    for _ in range(10):
        client._record_drift("uncertain-schema", True, True, .1)
    assert client.drift_detector.is_drifted
    assert not client._record_drift("clear-schema", False, False, .99)
    assert client._record_drift("uncertain-schema", True, True, .1)


def test_bad_teacher_responses_never_become_teaching_evidence():
    from system1.compat.typesafe import Client, Choice, TypeSafeResponse
    for answers in ({}, {"q": {"type": "choice", "choice": "unknown"}}):
        client = Client(mode="auto_cutover", baseline_handler=lambda *_: {"answers": answers})
        with pytest.raises(ValueError):
            client.systemone("request", {"q": Choice(criteria=["yes", "no"])})
        assert client.teacher_sample_count == 0
    with pytest.raises(ValueError):
        TypeSafeResponse({"answers": {"q": {"type": "noul", "noul": float("nan")}}})


def test_repeated_promotion_cannot_reuse_a_validation_block():
    from system1.compat.typesafe import Client, Choice, PromotionPolicy
    questions = {"route": Choice(criteria=["billing", "support"])}
    client = Client(
        mode="auto_cutover", dimension=16, cutover_threshold=50,
        promotion_policy=PromotionPolicy(min_agreement_threshold=1.0),
        baseline_handler=lambda state, _: {"answers": {
            "route": {"type": "choice", "choice": state["topic"]}}},
    )
    for i in range(80):
        client.systemone({"topic": "billing" if i % 2 else "support", "id": i}, questions,
                         record_receipt=False)
    assert not client.is_cutover
    attempts = dict(client._validation_attempts)
    assert sum(attempts.values()) > 0
    for _ in range(3):
        assert not client.distill_and_cutover(questions)
    assert dict(client._validation_attempts) == attempts


def test_refitting_calibrator_clears_old_binary_and_calibrated_state():
    from system1.calibration import DecisionCalibrator
    calibrator = DecisionCalibrator()
    calibrator.fit(np.array([-1.0, 1.0]), np.array([0, 1]))
    assert calibrator.is_binary and calibrator.is_calibrated
    calibrator.fit(np.array([[1., -1.], [-1., 1.]]), np.array([0, 1]))
    assert not calibrator.is_binary and calibrator.is_calibrated
    calibrator.fit(np.empty((0, 2)), np.array([]))
    assert not calibrator.is_calibrated and calibrator.temperature == 1


def test_correction_cannot_change_weights_during_uncertainty_evaluation(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    skill = constant_skill()
    engine = System1Engine(skill.schema, model=skill, use_cache=False)
    entered, release, correcting = Event(), Event(), Event()
    calibrator = engine.calibrators["route"]
    original = calibrator.calibrate_logits

    def paused(logits):
        entered.set()
        assert release.wait(5)
        return original(logits)

    def correct():
        correcting.set()
        return skill.learn_from_tier2("request", {"route": "support"})

    monkeypatch.setattr(calibrator, "calibrate_logits", paused)
    with ThreadPoolExecutor(max_workers=2) as pool:
        decision = pool.submit(engine.decide, "request", record_receipt=False)
        try:
            assert entered.wait(5)
            correction = pool.submit(correct)
            assert correcting.wait(5)
            from concurrent.futures import TimeoutError
            with pytest.raises(TimeoutError):
                correction.result(timeout=.05)
        finally:
            release.set()
        assert decision.result(timeout=5).values["route"] == "billing"
        assert "route" in correction.result(timeout=5)["updated_fields"]
    # An engine also notices direct corrections to its shared model.
    assert engine.decide("request", record_receipt=False, strict=True).is_ambiguous
