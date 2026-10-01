"""Release contracts for qualification, recovery and existing session reuse."""
import builtins
import hashlib
import json
import shutil

import pytest

import reflex
import system1
from system1 import ChoiceField, DecisionSchema, TeachingSession


class Filing(DecisionSchema):
    folder = ChoiceField(options=["billing", "support"])


def _rows(split, count, *, group=None):
    rows = []
    for index in range(count):
        label = "billing" if index % 2 == 0 else "support"
        words = "invoice payment billing " if label == "billing" else "software crash debug "
        row = {"input": words * 4 + f"{split} example {index}", "label": label, "split": split}
        if group is not None:
            row["group"] = group
        rows.append(row)
    return rows


def _qualification_rows(count=120, *, group=None):
    return [{key: value for key, value in row.items() if key != "split"}
            for row in _rows("qualification", count, group=group)]


@pytest.fixture
def session(tmp_path):
    result = TeachingSession(tmp_path, Filing)
    result.record_many(_rows("teach", 8) + _rows("calibrate", 60) + _rows("evaluate", 12))
    return result


def _adopt(session, *, require_qualification=False):
    report = session.assess(require_qualification=require_qualification)
    assert report["passed"]
    if require_qualification:
        qualified = session.qualify(_qualification_rows(), source="independent release cohort")
        assert qualified["passed"]
    session.adopt()
    return session.current_path.read_bytes()


def test_legacy_version_one_session_opens_without_rewriting(tmp_path):
    # Version 1.1.0 wrote precisely these keys, without choice_solver or groups.
    original = {"version": 1, "schema": Filing().to_dict(),
                "records": [{**row, "source": "human"} for row in _rows("teach", 8)],
                "regularization": .1, "max_features": 1024}
    path = tmp_path / "session.json"
    path.write_text(json.dumps(original, indent=2) + "\n")
    before = path.read_bytes()
    reopened = TeachingSession(tmp_path)
    assert reopened.records == original["records"]
    assert reopened.snapshot()["settings"]["choice_solver"] == "ridge"
    assert path.read_bytes() == before
    assert reopened.history == []
    assert reflex.TeachingSession is system1.TeachingSession


def test_approved_revisions_rollback_and_reopen_keep_exact_artifacts(session):
    first = _adopt(session)
    first_revision = hashlib.sha256(first).hexdigest()
    # A supplied additional lesson creates a distinct passing revision.
    session.record("invoice payment billing " * 4 + "additional lesson", "billing")
    second = _adopt(session)
    second_revision = hashlib.sha256(second).hexdigest()
    assert first_revision != second_revision
    history = session.history
    assert {item["revision"] for item in history} == {first_revision, second_revision}
    assert [item["revision"] for item in history if item["current"]] == [second_revision]
    assert all(item["approved_at"] for item in history)

    restored = session.rollback(first_revision)
    assert restored["revision"] == first_revision
    assert session.current_path.read_bytes() == first
    assert session.snapshot()["candidate_stale"]
    reopened = TeachingSession(session.directory)
    assert reopened.current_path.read_bytes() == first
    text = "invoice payment billing " * 4 + "fresh current request"
    assert reopened.predict(text).values == session.predict(text).values == {"folder": "billing"}
    assert [item["revision"] for item in reopened.history if item["current"]] == [first_revision]

    before = reopened.current_path.read_bytes()
    with pytest.raises(ValueError, match="stale"):
        reopened.adopt()
    assert reopened.current_path.read_bytes() == before
    assert reopened.assess()["identity"]["current"] == first_revision
    reopened.adopt()
    assert reopened.current_path.read_bytes() == reopened.candidate_path.read_bytes()


def test_failed_candidate_is_not_approved_and_keeps_current(session):
    approved = _adopt(session)
    history = session.history
    for row in session.records:
        if row["split"] == "teach":
            session.record(row["input"], "support" if row["label"] == "billing" else "billing")
    assert not session.assess()["passed"]
    assert session.history == history
    with pytest.raises(ValueError, match="did not pass"):
        session.adopt()
    assert session.current_path.read_bytes() == approved


def test_existing_legacy_current_is_retained_for_recovery_on_first_new_adoption(session):
    legacy = _adopt(session)
    revision = hashlib.sha256(legacy).hexdigest()
    # An old workspace has current.s1m, no revision directory, and no solver key.
    shutil.rmtree(session.revisions_path)
    state = json.loads(session.session_path.read_text())
    state.pop("choice_solver")
    session.session_path.write_text(json.dumps(state))
    reopened = TeachingSession(session.directory)
    assert reopened.history == []
    assert reopened.predict("invoice payment billing " * 4).values == {"folder": "billing"}
    reopened.record("invoice payment billing " * 4 + "new release lesson", "billing")
    updated = _adopt(reopened)
    assert updated != legacy
    assert revision in {item["revision"] for item in reopened.history}
    reopened.rollback(revision)
    assert reopened.current_path.read_bytes() == legacy
    assert TeachingSession(reopened.directory).predict("invoice payment billing " * 4).values == {"folder": "billing"}


@pytest.mark.parametrize("revision", ["../current", "a" * 64, "not-a-revision"])
def test_unknown_rollback_cannot_change_current(session, revision):
    approved = _adopt(session)
    with pytest.raises(ValueError):
        session.rollback(revision)
    assert session.current_path.read_bytes() == approved


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_damaged_archived_revision_cannot_replace_current(session, damage):
    first = _adopt(session)
    first_revision = hashlib.sha256(first).hexdigest()
    session.record("invoice payment billing " * 4 + "second approved lesson", "billing")
    current = _adopt(session)
    archives = [path for path in session.directory.rglob("*.s1m")
                if path not in (session.current_path, session.candidate_path) and
                hashlib.sha256(path.read_bytes()).hexdigest() == first_revision]
    assert archives, "Approved revisions need retained restorable artifacts"
    for archive in archives:
        if damage == "missing":
            archive.unlink()
        else:
            archive.write_bytes(first + b"changed")
    with pytest.raises(ValueError):
        session.rollback(first_revision)
    assert session.current_path.read_bytes() == current


@pytest.mark.parametrize("changed", ["revision", "schema", "approval"])
def test_changed_approval_record_cannot_authorize_rollback(session, changed):
    first = _adopt(session)
    revision = hashlib.sha256(first).hexdigest()
    session.record("invoice payment billing " * 4 + "second approved lesson", "billing")
    current = _adopt(session)
    path = session.revisions_path / f"{revision}.json"
    record = json.loads(path.read_text())
    if changed == "approval":
        record["assessment"]["passed"] = False
    else:
        record[changed] = "0" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="integrity"):
        session.rollback(revision)
    assert session.current_path.read_bytes() == current


@pytest.mark.parametrize("changed", ["missing", "failed", "source", "request", "required_policy"])
def test_required_qualification_is_rechecked_before_restoring_approved_revision(session, changed):
    first = _adopt(session, require_qualification=True)
    revision = hashlib.sha256(first).hexdigest()
    session.record("invoice payment billing " * 4 + "second approved lesson", "billing")
    current = _adopt(session)
    path = session.revisions_path / f"{revision}.json"
    record = json.loads(path.read_text())
    if changed == "request":
        request_path = session.qualifications_path / f"{revision}.request.json"
        request = json.loads(request_path.read_text())
        request["source"] = "altered archived provenance"
        request_path.write_text(json.dumps(request))
    else:
        if changed == "missing":
            record["qualification"] = {}
        elif changed == "failed":
            record["qualification"]["passed"] = False
        elif changed == "source":
            record["qualification"]["source"] = "altered archived provenance"
        else:
            record["assessment"]["thresholds"]["require_qualification"] = False
            record["qualification"] = {}
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="integrity|qualification|Qualification|changed"):
        session.rollback(revision)
    assert session.current_path.read_bytes() == current


def test_required_qualification_blocks_adoption_until_fresh_checks_pass(session):
    assert session.assess(require_qualification=True)["passed"]
    assert not session.snapshot()["can_adopt"]
    with pytest.raises(ValueError, match="qualification|qualify"):
        session.adopt()
    assert not session.current_path.exists()
    qualified = session.qualify(_qualification_rows(), source="independent release cohort")
    assert qualified["passed"]
    assert session.snapshot()["can_adopt"]
    session.adopt()
    assert session.qualification is not None
    assert session.current_path.read_bytes() == session.candidate_path.read_bytes()
    assert session.history[0]["qualification"]
    reopened = TeachingSession(session.directory)
    assert reopened.qualification == qualified


def test_small_error_free_cohort_does_not_establish_qualification(session):
    session.assess(require_qualification=True)
    report = session.qualify(_qualification_rows(12), source="small independent cohort")
    assert not report["passed"]
    assert not session.snapshot()["can_adopt"]
    with pytest.raises(ValueError, match="qualification|qualify"):
        session.adopt()
    assert not session.current_path.exists()


def test_qualification_uses_joint_exact_bounds_and_retains_class_evidence(session):
    session.assess(require_qualification=True)
    report = session.qualify(_qualification_rows(), confidence=.9, source="independent cohort")
    metrics = report["candidate"]
    assert metrics["accepted"] == metrics["accepted_correct"] == 120
    # The first attempt receives half of the session error budget, split again
    # between correctness and coverage. Later attempts retain their own share.
    expected = .025 ** (1 / 120)
    assert metrics["accepted_correctness_lower_bound"] == pytest.approx(expected, abs=1e-10)
    assert metrics["coverage_lower_bound"] == pytest.approx(expected, abs=1e-10)
    assert metrics["accepted_accuracy"] == 1
    assert set(metrics["per_class"]) == {"billing", "support"}
    assert all(item["count"] == item["accepted"] == 60 and item["accepted_errors"] == 0
               for item in metrics["per_class"].values())
    assert len(report["cases"]) == 120
    assert report["evidence_scope"] == "supplied_independent_qualification"
    assert "independent" in report["sampling_assumption"]


def test_fresh_second_candidate_spends_session_confidence_budget(session):
    session.assess(require_qualification=True)
    first = session.qualify(_qualification_rows(), confidence=.9, source="first independent cohort")
    assert first["passed"]
    session.adopt()
    # Update recorded provenance so the candidate identity changes while this
    # confidence-allocation test keeps the same fitted features and choices.
    original = session.records[0]
    session.record(original["input"], original["label"], source="independently rechecked lesson")
    session.assess(require_qualification=True)
    fresh = [{**row, "input": row["input"].replace("qualification", "anotherfreshcohort")}
             for row in _qualification_rows()]
    second = session.qualify(fresh, confidence=.9, source="second independent cohort")
    assert second["attempt"] == 2
    assert first["attempt"] == 1
    # Sum over attempts of 1/(k*(k+1)) is 1; each claim gets half.
    expected = (.1 / (2 * 2 * 3)) ** (1 / 120)
    assert second["candidate"]["accepted_correctness_lower_bound"] == pytest.approx(expected, abs=1e-10)
    assert second["candidate"]["coverage_lower_bound"] == pytest.approx(expected, abs=1e-10)
    assert second["confidence_scope"] == first["confidence_scope"]
    assert second["passed"]
    session.adopt()


def test_later_qualification_cannot_change_fixed_session_confidence(session):
    session.assess(require_qualification=True)
    session.qualify(_qualification_rows(12), confidence=.9, source="first independent cohort")
    session.record("invoice payment billing " * 4 + "second candidate lesson", "billing")
    session.assess(require_qualification=True)
    fresh = [{**row, "input": row["input"].replace("qualification", "second-independent-cohort")}
             for row in _qualification_rows()]
    before = {str(path.relative_to(session.directory)): path.read_bytes()
              for path in session.directory.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="confidence"):
        session.qualify(fresh, confidence=.95, source="second independent cohort")
    after = {str(path.relative_to(session.directory)): path.read_bytes()
             for path in session.directory.rglob("*") if path.is_file()}
    assert after == before


def test_no_accepted_answers_report_missing_correctness_evidence(session):
    session.assess(require_qualification=True)
    rows = [{"input": f"zebras migrate wilderness habitat {index}", "label": "billing"}
            for index in range(120)]
    report = session.qualify(rows, source="independent unfamiliar requests")
    metrics = report["candidate"]
    assert metrics["accepted"] == 0
    assert metrics["accepted_accuracy"] is None
    assert metrics["accepted_correctness_lower_bound"] is None
    assert metrics["coverage_lower_bound"] == 0
    assert not report["passed"]
    assert len(report["cases"]) == 120


def test_qualification_retains_accepted_errors_and_their_class(session):
    session.assess(require_qualification=True)
    rows = _qualification_rows()
    rows[0]["label"] = "support"
    report = session.qualify(rows, min_accepted_accuracy=.99, source="independently checked labels")
    assert not report["passed"]
    assert report["candidate"]["accepted_errors"] == 1
    assert report["candidate"]["per_class"]["support"]["accepted_errors"] == 1
    errors = [case for case in report["cases"] if not case["candidate"]["correct"]]
    assert len(errors) == 1
    assert errors[0]["input"] == rows[0]["input"]
    assert errors[0]["label"] == "support"
    assert errors[0]["candidate"]["predicted"] == "billing"
    assert errors[0]["candidate"]["review"] is False


@pytest.mark.parametrize("split", ["teach", "calibrate", "evaluate"])
def test_qualification_rejects_normalized_overlap_with_every_recorded_split(session, split):
    session.assess(require_qualification=True)
    known = next(row for row in session.records if row["split"] == split)
    overlap = {"input": "  " + known["input"].upper().replace(" ", "  ") + "\n", "label": known["label"]}
    with pytest.raises(ValueError, match="disjoint|overlap"):
        session.qualify(_qualification_rows() + [overlap], source="overlapping cohort")
    assert session.qualification is None
    assert not session.current_path.exists()


def test_qualification_rejects_related_groups_and_accepts_separate_groups(tmp_path):
    session = TeachingSession(tmp_path, Filing)
    session.record_many(_rows("teach", 8, group="conversation-a") +
                        _rows("calibrate", 60, group="conversation-b") +
                        _rows("evaluate", 12, group="conversation-c"))
    session.assess(require_qualification=True)
    with pytest.raises(ValueError, match="group|disjoint"):
        session.qualify(_qualification_rows(group="conversation-a"), source="related messages")
    independent = [{**row, "group": f"fresh-conversation-{index}"}
                   for index, row in enumerate(_qualification_rows())]
    report = session.qualify(independent, source="independent messages")
    assert report["passed"]


def test_invalid_qualification_batch_is_not_committed_or_scored(session, monkeypatch):
    session.assess(require_qualification=True)
    before = {str(path.relative_to(session.directory)): path.read_bytes()
              for path in session.directory.rglob("*") if path.is_file()}
    monkeypatch.setattr(session, "_qualification_report", lambda *args, **kwargs: pytest.fail("Invalid labels reached qualification scoring"))
    rows = _qualification_rows()
    rows[-1]["label"] = "invented"
    with pytest.raises(ValueError):
        session.qualify(rows, source="invalid cohort")
    after = {str(path.relative_to(session.directory)): path.read_bytes()
             for path in session.directory.rglob("*") if path.is_file()}
    assert after == before


def test_forged_development_pass_cannot_consume_independent_qualification(session, monkeypatch):
    row = next(row for row in session.records if row["split"] == "evaluate")
    session.record(row["input"], "support", split="evaluate")
    assessment = session.assess(require_qualification=True)
    assert not assessment["passed"]
    assessment["passed"] = True
    session.report_path.write_text(json.dumps(assessment))
    before = {str(path.relative_to(session.directory)): path.read_bytes()
              for path in session.directory.rglob("*") if path.is_file()}
    monkeypatch.setattr(session, "_qualification_report", lambda *args, **kwargs: pytest.fail("Unapproved candidate reached qualification scoring"))
    with pytest.raises(ValueError, match="assessment|Assessment|development"):
        session.qualify(_qualification_rows(), source="reserved independent cohort")
    after = {str(path.relative_to(session.directory)): path.read_bytes()
             for path in session.directory.rglob("*") if path.is_file()}
    assert after == before
    assert not list(session.qualifications_path.glob("*.request.json"))


def test_qualification_policy_and_cohort_cannot_be_replaced_after_scoring(session):
    session.assess(require_qualification=True)
    session.qualify(_qualification_rows(12), source="reserved cohort")
    with pytest.raises(ValueError, match="frozen|already|replace|qualification"):
        session.qualify(_qualification_rows(120), min_accepted_accuracy=.5,
                        min_coverage=.1, source="replacement cohort")
    assert not session.snapshot()["can_adopt"]


def test_previously_scored_cohort_cannot_be_reused_for_a_new_candidate(session):
    session.assess(require_qualification=True)
    session.qualify(_qualification_rows(12), source="first reserved cohort")
    session.record("invoice payment billing " * 4 + "new candidate lesson", "billing")
    session.assess(require_qualification=True)
    with pytest.raises(ValueError, match="disjoint|used|qualification"):
        session.qualify(_qualification_rows(12), source="same inspected cohort reused")


@pytest.mark.parametrize("changed", ["metrics", "cases", "thresholds", "request", "source"])
def test_changed_qualification_evidence_cannot_authorize_adoption(session, changed):
    session.assess(require_qualification=True)
    report = session.qualify(_qualification_rows(), source="independent cohort")
    revision = report["identity"]["candidate"]
    report_path = session.qualifications_path / f"{revision}.json"
    if changed == "request":
        path = session.qualifications_path / f"{revision}.request.json"
        request = json.loads(path.read_text())
        request["source"] = "changed provenance"
        path.write_text(json.dumps(request))
    else:
        if changed == "metrics":
            report["candidate"]["raw_correct"] -= 1
        elif changed == "cases":
            report["cases"][0]["label"] = "support"
        elif changed == "thresholds":
            report["thresholds"]["min_accepted_accuracy"] = .1
        else:
            report["source"] = "changed provenance"
        report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Qualification|qualification"):
        session.adopt()
    assert not session.current_path.exists()


def test_forged_pass_flag_cannot_override_insufficient_qualification(session):
    session.assess(require_qualification=True)
    report = session.qualify(_qualification_rows(12), source="small independent cohort")
    assert not report["passed"]
    report["passed"] = True
    path = session.qualifications_path / f"{report['identity']['candidate']}.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Qualification|qualification"):
        session.adopt()
    assert not session.current_path.exists()


@pytest.mark.parametrize("passed", [False, True])
def test_optional_qualification_is_validated_before_adoption(session, passed):
    session.assess()
    report = session.qualify(_qualification_rows(120 if passed else 12), source="optional independent cohort")
    assert report["passed"] is passed
    if passed:
        report["source"] = "changed independent provenance"
    else:
        report["passed"] = True
    path = session.qualifications_path / f"{report['identity']['candidate']}.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Qualification|qualification"):
        session.adopt()
    assert not session.current_path.exists()


def test_valid_failed_optional_qualification_allows_development_approval_and_recovery(session):
    session.assess()
    qualification = session.qualify(_qualification_rows(12), source="optional independent cohort")
    assert not qualification["passed"]
    session.adopt()
    artifact = session.current_path.read_bytes()
    revision = hashlib.sha256(artifact).hexdigest()
    assert session.history[0]["qualification"] is False
    session.record("invoice payment billing " * 4 + "later development lesson", "billing")
    _adopt(session)
    restored = session.rollback(revision)
    assert restored["qualification"] is False
    assert session.current_path.read_bytes() == artifact


@pytest.mark.parametrize("passed", [False, True])
def test_saved_optional_qualification_is_validated_before_rollback(session, passed):
    session.assess()
    qualification = session.qualify(_qualification_rows(120 if passed else 12), source="optional independent cohort")
    assert qualification["passed"] is passed
    session.adopt()
    revision = hashlib.sha256(session.current_path.read_bytes()).hexdigest()
    session.record("invoice payment billing " * 4 + "later development lesson", "billing")
    current = _adopt(session)
    path = session.revisions_path / f"{revision}.json"
    approval = json.loads(path.read_text())
    approval["qualification"]["source"] = "changed archived provenance"
    path.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="qualification|Qualification|changed"):
        session.rollback(revision)
    assert session.current_path.read_bytes() == current


def test_interrupted_qualification_consumes_its_reserved_cohort(session, monkeypatch):
    session.assess(require_qualification=True)
    original = session._qualification_report

    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted scoring")

    monkeypatch.setattr(session, "_qualification_report", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        session.qualify(_qualification_rows(12), source="reserved interrupted cohort")
    monkeypatch.setattr(session, "_qualification_report", original)
    assert session.qualification is None
    with pytest.raises(ValueError):
        session.qualify(_qualification_rows(12), source="repeated interrupted cohort")
    with pytest.raises(ValueError, match="disjoint"):
        session.record(_qualification_rows(12)[0]["input"], "billing")
    assert not session.current_path.exists()


def test_repeated_qualification_group_cannot_inflate_sample_evidence(session):
    session.assess(require_qualification=True)
    with pytest.raises(ValueError, match="one row per group"):
        session.qualify(_qualification_rows(group="one-conversation"), source="correlated conversation")
    assert session.qualification is None


def test_lesson_change_stales_qualification_before_adoption(session):
    session.assess(require_qualification=True)
    assert session.qualify(_qualification_rows(), source="independent cohort")["passed"]
    session.record("invoice payment billing " * 4 + "new lesson after qualification", "billing")
    with pytest.raises(ValueError, match="stale"):
        session.adopt()
    assert not session.current_path.exists()
    assert not session.snapshot()["can_adopt"]


def test_decision_details_explain_review_without_recording_a_label(session):
    approved = _adopt(session)
    before = session.session_path.read_bytes()
    ambiguous = session.decision_details("zebras migrate between wilderness habitats")
    assert ambiguous["review"]
    assert ambiguous["review_reasons"]
    assert ambiguous["prediction_set"] == []
    assert ambiguous["revision"] == hashlib.sha256(approved).hexdigest()
    known = session.decision_details("invoice payment billing " * 4 + "new decision")
    assert known["values"] == {"folder": "billing"}
    assert known["review"] is False
    assert known["review_reasons"] == []
    assert session.session_path.read_bytes() == before


def test_explicit_logistic_setting_persists_and_cannot_silently_change(tmp_path):
    session = TeachingSession(tmp_path, Filing, choice_solver="logistic")
    assert session.snapshot()["settings"]["choice_solver"] == "logistic"
    assert TeachingSession(tmp_path).snapshot()["settings"]["choice_solver"] == "logistic"
    with pytest.raises(ValueError, match="choice_solver.*differs"):
        TeachingSession(tmp_path, choice_solver="ridge")


def test_default_session_qualification_and_recovery_need_no_scipy(session, monkeypatch):
    original_import = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("Optional teaching dependency unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scipy)
    artifact = _adopt(session, require_qualification=True)
    revision = hashlib.sha256(artifact).hexdigest()
    session.rollback(revision)
    assert session.predict("invoice payment billing " * 4).values == {"folder": "billing"}


def test_logistic_session_passes_setting_to_compiler_and_reuses_saved_skill(tmp_path, monkeypatch):
    from system1.compiler import CompiledSystemOneModel

    session = TeachingSession(tmp_path, Filing, choice_solver="logistic")
    session.record_many(_rows("teach", 8) + _rows("calibrate", 60) + _rows("evaluate", 12))
    artifact = _adopt(session)
    assert CompiledSystemOneModel.from_bytes(artifact).metadata["choice_solver"] == "logistic"
    expected = session.predict("invoice payment billing " * 4)
    original_import = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("Optional teaching dependency unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scipy)
    reopened = TeachingSession(tmp_path)
    actual = reopened.predict("invoice payment billing " * 4)
    assert actual.values == expected.values
    assert actual.conformal_sets == expected.conformal_sets
    assert actual.probabilities == expected.probabilities
