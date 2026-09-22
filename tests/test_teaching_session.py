"""A correction must not silently disable or replace the approved skill."""
import hashlib
import json

import pytest

from system1 import BooleanField, ChoiceField, DecisionSchema, TeachingSession
from system1.compiler import CompiledSystemOneModel, SystemOneCompiler


class Filing(DecisionSchema):
    folder = ChoiceField(options=["billing", "support"])


def lessons():
    rows = []
    for split, count in (("teach", 8), ("calibrate", 60), ("evaluate", 12)):
        for index in range(count):
            label = "billing" if index % 2 == 0 else "support"
            phrase = "invoice payment billing " if label == "billing" else "software crash debug "
            rows.append({"input": phrase * 4 + f"{split} example {index}",
                         "label": label, "split": split})
    return rows


@pytest.fixture
def session(tmp_path):
    result = TeachingSession(tmp_path, Filing)
    result.record_many(lessons())
    return result


def test_replacement_disjointness_atomic_import_and_reopening(session):
    original = session.records[0]
    text = "  " + original["input"].upper().replace(" ", "  ") + "\n"
    session.record(text, "support")
    assert len(session.records) == 80
    assert session.records[0]["label"] == "support"
    before = session.session_path.read_bytes()
    with pytest.raises(ValueError, match="disjoint"):
        session.record(original["input"], "billing", split="evaluate")
    with pytest.raises(ValueError, match="supplied label"):
        session.record_many([
            {"input": "new document", "label": "billing", "split": "teach"},
            {"input": "invalid document", "label": "unknown", "split": "teach"},
        ])
    with pytest.raises(ValueError, match="supplied label"):
        session.record_many([
            {"input": "invalid document", "label": "unknown", "split": "teach"},
            {"input": "invalid document", "label": "billing", "split": "teach"},
        ])
    assert session.session_path.read_bytes() == before
    reopened = TeachingSession(session.directory)
    assert reopened.records == session.records
    reopened.remove(original["input"])
    assert len(reopened.records) == 79
    with pytest.raises(ValueError, match="No recorded"):
        reopened.remove(original["input"])


def test_real_assessment_adoption_and_reload_preserve_exact_artifact(session, monkeypatch):
    monkeypatch.setattr(SystemOneCompiler, "generate_synthetic_exemplars",
                        lambda *args, **kwargs: pytest.fail("Invented lessons"))
    report = session.assess()
    assert report["passed"]
    assert report["counts"] == {"teach": 8, "calibrate": 60, "evaluate": 12}
    assert report["candidate"]["raw_correct"] == report["candidate"]["accepted_correct"] == 12
    assert report["calibration_counts"] == {"fit": 8, "calibration": 60, "temperature": 30, "generated": 0}
    assert report["incumbent"] is None
    assert session.engine is None
    candidate_bytes = session.candidate_path.read_bytes()
    assert hashlib.sha256(candidate_bytes).hexdigest() == report["identity"]["candidate"]
    session.adopt()
    assert session.current_path.read_bytes() == candidate_bytes
    assert session.snapshot()["adopted"]
    assert not session.snapshot()["candidate_stale"]
    reopened = TeachingSession(session.directory)
    for row in [item for item in lessons() if item["split"] == "evaluate"]:
        first = session.predict(row["input"])
        second = reopened.predict(row["input"])
        assert first.values == second.values == {"folder": row["label"]}
        assert first.conformal_sets == second.conformal_sets
        assert first.probabilities == second.probabilities
    assert len(session.records) == 80  # predictions are never lessons


def test_assessment_and_bad_corrections_keep_approved_engine_serving(session):
    session.assess()
    session.adopt()
    before = session.current_path.read_bytes()
    engine = session.engine
    known_text = session.records[0]["input"]
    for row in session.records:
        if row["split"] == "teach":
            session.record(row["input"], "support" if row["label"] == "billing" else "billing")
    assert session.engine is engine
    assert session.predict(known_text).values == {"folder": "billing"}
    report = session.assess()
    assert not report["passed"]
    assert report["candidate"]["raw_accuracy"] == 0
    assert report["incumbent"]["raw_accuracy"] == 1
    assert report["raw_regressions"] == report["regressions"] == 12
    assert any("before review" in message for message in report["diagnostics"])
    with pytest.raises(ValueError, match="did not pass"):
        session.adopt()
    assert session.current_path.read_bytes() == before
    assert session.engine is engine


@pytest.mark.parametrize("changed", ["teach", "calibrate", "evaluate", "candidate", "current", "settings"])
def test_each_assessment_identity_must_still_match(session, changed):
    session.assess()
    session.adopt()
    session.assess()
    if changed in ("teach", "calibrate", "evaluate"):
        row = next(row for row in session.records if row["split"] == changed)
        session.record(row["input"], "support" if row["label"] == "billing" else "billing", split=changed)
    elif changed in ("candidate", "current"):
        path = session.candidate_path if changed == "candidate" else session.current_path
        model = CompiledSystemOneModel.load(path)
        model.heads["folder"].biases[0] += .01
        model.save(path)
    else:
        data = json.loads(session.session_path.read_text())
        data["regularization"] = .2
        session.session_path.write_text(json.dumps(data))
    before = session.current_path.read_bytes()
    with pytest.raises(ValueError, match="stale"):
        session.adopt()
    assert session.current_path.read_bytes() == before
    assert session.snapshot()["candidate_stale"]


def test_changed_report_cannot_relax_original_gate(session):
    session.assess()
    report = session.report
    report["thresholds"]["min_accuracy"] = 0
    session.report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="settings changed"):
        session.adopt()
    report = session.assess()
    report["candidate"]["raw_correct"] -= 1
    session.report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="assessment changed"):
        session.adopt()
    assert not session.current_path.exists()


def test_review_cannot_hide_loss_of_previously_accepted_correct_checks(session, monkeypatch):
    session.assess()
    session.adopt()
    original = session._measure

    def with_review(engine, checks):
        metrics, outcomes = original(engine, checks)
        # Only candidate evaluation changes: all raw choices remain correct.
        if engine.model.metadata.get("teaching_session", {}).get("current"):
            metrics.update(accepted=0, accepted_correct=0, accepted_errors=0,
                           review=12, review_rate=1., coverage=0.)
            outcomes = [{**outcome, "review": True} for outcome in outcomes]
        return metrics, outcomes

    monkeypatch.setattr(session, "_measure", with_review)
    report = session.assess(min_coverage=0)
    assert report["candidate"]["raw_accuracy"] == 1
    assert report["candidate"]["accepted_errors"] == 0
    assert report["raw_regressions"] == 0
    assert report["regressions"] == 12
    assert not report["passed"]
    assert any("required review" in message for message in report["reasons"])


@pytest.mark.parametrize("kwargs", [
    {"min_accuracy": float("nan")}, {"min_coverage": float("inf")},
    {"min_accuracy": -1}, {"min_coverage": 2}, {"max_regressions": .5},
    {"max_accepted_errors": -1}, {"max_regressions": True},
])
def test_invalid_gate_parameters_fail_before_compiling(session, kwargs):
    with pytest.raises(ValueError):
        session.assess(**kwargs)
    assert not session.candidate_path.exists()


def test_missing_evaluation_calibration_or_classes_cannot_be_assessed(tmp_path):
    for missing in ("evaluate", "calibrate", "class"):
        session = TeachingSession(tmp_path / missing, Filing)
        session.record_many([row for row in lessons() if row["split"] != missing and not (
            missing == "class" and row["split"] == "teach" and row["label"] == "support")])
        with pytest.raises(ValueError, match="evaluate|calibrate|every choice"):
            session.assess()
        with pytest.raises(ValueError, match="Assess"):
            session.adopt()


def test_only_supported_schemas_and_valid_explicit_labels(tmp_path):
    class BooleanSkill(DecisionSchema):
        ready = BooleanField()

    class Multiple(DecisionSchema):
        first = ChoiceField(["a", "b"])
        second = ChoiceField(["a", "b"])

    for schema in (None, BooleanSkill, Multiple):
        with pytest.raises(ValueError, match="ChoiceField"):
            TeachingSession(tmp_path / str(schema), schema)
    session = TeachingSession(tmp_path / "valid", Filing)
    for text, label in (("", "billing"), ("text", None), ("text", "unknown")):
        with pytest.raises(ValueError):
            session.record(text, label)
    with pytest.raises(ValueError, match="schema differs"):
        TeachingSession(session.directory, BooleanSkill)


def test_new_session_cannot_disable_review_on_uncertain_choices(tmp_path):
    class NoReview(DecisionSchema):
        folder = ChoiceField(["billing", "support"], escalate_on_ambiguity=False)

    with pytest.raises(ValueError, match="requires escalate_on_ambiguity=True"):
        TeachingSession(tmp_path, NoReview)
    assert not (tmp_path / "session.json").exists()
    assert not (tmp_path / "candidate.s1m").exists()


def test_reopened_session_cannot_disable_review_or_replace_approved_artifact(session):
    session.assess()
    session.adopt()
    unknown = session.predict("zebras migrate")
    assert unknown.is_ambiguous
    assert unknown.conformal_sets["folder"] == []
    current = session.current_path.read_bytes()
    candidate = session.candidate_path.read_bytes()
    state = json.loads(session.session_path.read_text())
    state["schema"]["fields"]["folder"]["escalate_on_ambiguity"] = False
    session.session_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="requires escalate_on_ambiguity=True"):
        TeachingSession(session.directory)
    with pytest.raises(ValueError, match="requires escalate_on_ambiguity=True"):
        session.assess()
    with pytest.raises(ValueError, match="requires escalate_on_ambiguity=True"):
        session.adopt()
    assert session.current_path.read_bytes() == current
    assert session.candidate_path.read_bytes() == candidate


def test_identical_record_import_does_not_stale_assessment(session):
    before = session.assess()
    session.record_many(session.records)
    assert not session.snapshot()["candidate_stale"]
    assert session.adopt() == before


def test_one_corrected_lesson_improves_unseen_cases_without_losing_old_skill(session):
    phrase = "subscription renewal charge " * 4
    wrong_lesson = phrase + "teach specialty"
    session.record(wrong_lesson, "support")
    # Split names and single-digit IDs are absent from the fitted vocabulary.
    # A known billing cue keeps the unseen checks off the calibration-score tie,
    # where batch/single BLAS rounding could change the prediction set by one ULP.
    session.record_many([{"input": ("invoice " if split == "evaluate" else "") + phrase + f"{split} specialty {index}",
                          "label": "billing", "split": split}
                         for split, count in (("calibrate", 6), ("evaluate", 2)) for index in range(count)])
    initial = session.assess()
    assert initial["passed"]
    assert initial["candidate"]["raw_correct"] == 12
    assert initial["candidate"]["review"] == 2
    session.adopt()
    incumbent = session.current_path.read_bytes()
    count = len(session.records)
    session.record(wrong_lesson, "billing")
    assert len(session.records) == count
    assert session.current_path.read_bytes() == incumbent
    corrected = session.assess()
    assert corrected["passed"]
    assert corrected["incumbent"]["raw_correct"] == 12
    assert corrected["candidate"]["raw_correct"] == corrected["candidate"]["accepted_correct"] == 14
    assert corrected["regressions"] == 0
    improved = [case for case in corrected["cases"] if not case["incumbent"]["correct"]]
    assert len(improved) == 2
    assert all(case["candidate"]["correct"] and not case["candidate"]["review"] for case in improved)
    assert session.current_path.read_bytes() == incumbent
    session.adopt()
    assert session.current_path.read_bytes() == session.candidate_path.read_bytes()
    assert session.current_path.read_bytes() != incumbent


def test_long_text_cannot_hide_cross_split_overlap_beyond_projector_limit(session):
    for suffix in ("teaching", "evaluation"):
        with pytest.raises(ValueError, match="8192"):
            session.record("invoice " * 1024 + suffix, "billing")
    with pytest.raises(ValueError, match="8192"):
        session.predict("x" * 8193)
