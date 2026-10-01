"""Exercise the public correction CLI through real saved skills and records."""
import json

import pytest

from system1 import ChoiceField, DecisionSchema
from system1.cli import main


class Route(DecisionSchema):
    team = ChoiceField(options=["billing", "support"])


def setup_files(tmp_path):
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(Route().to_dict()))
    dataset = tmp_path / "lessons.json"
    data = {
        split: [
            [f"{phrase} {split} example {index}", label]
            for label, phrase in (("billing", "invoice payment refund"), ("support", "software crash bug"))
            for index in range(count)
        ]
        for split, count in (("teach", 12), ("calibrate", 24), ("evaluate", 6))
    }
    dataset.write_text(json.dumps(data))
    return schema, dataset


def test_teach_cli_roundtrip_and_stale_candidate_preserves_current(tmp_path, capsys):
    from system1 import TeachingSession

    schema, dataset = setup_files(tmp_path)
    directory = tmp_path / "skill"
    prefix = ["teach", str(directory)]
    assert main(prefix + ["--schema", str(schema), "--dataset", str(dataset), "--json"]) == 0
    json.loads(capsys.readouterr().out)
    assert main(prefix + ["--assess", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] and report["candidate"]["accepted_errors"] == 0
    assert report["incumbent"] is None
    session = TeachingSession(directory)
    assert not session.current_path.exists()
    candidate = session.candidate_path.read_bytes()
    assert main(prefix + ["--adopt", "--json"]) == 0
    json.loads(capsys.readouterr().out)
    assert session.current_path.read_bytes() == candidate

    assert main(prefix + ["--assess"]) == 0
    assert "Previously accepted correct cases lost: 0" in capsys.readouterr().out
    assert main(prefix + ["--record", "Invoice payment refund teach example 0", "--label", "support", "--json"]) == 0
    capsys.readouterr()
    reopened = TeachingSession(directory)
    assert len(reopened.records) == 84
    assert main(prefix + ["--adopt", "--json"]) == 1
    assert "error" in json.loads(capsys.readouterr().out)
    assert reopened.current_path.read_bytes() == candidate
    assert reopened.predict("invoice payment refund").values["team"] == "billing"


def test_teach_cli_bulk_import_is_atomic(tmp_path, capsys):
    from system1 import TeachingSession

    schema, dataset = setup_files(tmp_path)
    directory = tmp_path / "skill"
    prefix = ["teach", str(directory)]
    assert main(prefix + ["--schema", str(schema), "--dataset", str(dataset), "--json"]) == 0
    capsys.readouterr()
    before = TeachingSession(directory).records
    dataset.write_text(json.dumps({"teach": [["a new invoice", "billing"], ["invalid answer", "unknown"]]}))
    assert main(prefix + ["--dataset", str(dataset), "--json"]) == 1
    assert "error" in json.loads(capsys.readouterr().out)
    assert TeachingSession(directory).records == before
    assert main(prefix + ["--record", "  INVOICE payment refund TEACH example 0  ", "--label", "billing", "--split", "evaluate", "--json"]) == 1
    assert "error" in json.loads(capsys.readouterr().out)
    assert TeachingSession(directory).records == before


def test_teach_cli_requests_reassessment_after_correcting_an_adopted_skill(tmp_path, capsys):
    from system1 import TeachingSession

    schema, dataset = setup_files(tmp_path)
    directory = tmp_path / "skill"
    prefix = ["teach", str(directory)]
    assert main(prefix + ["--schema", str(schema), "--dataset", str(dataset)]) == 0
    assert main(prefix + ["--assess"]) == 0
    assert main(prefix + ["--adopt"]) == 0
    capsys.readouterr()
    session = TeachingSession(directory)
    approved = session.current_path.read_bytes()
    assert main(prefix + ["--record", "Invoice payment refund teach example 0", "--label", "support"]) == 0
    message = capsys.readouterr().out
    assert "assess again before adopting" in message
    assert "The assessed candidate is the current skill" not in message
    state = session.snapshot()
    assert state["adopted"] and state["candidate_stale"]
    assert session.current_path.read_bytes() == approved
    assert main(prefix) == 0
    assert "assess again before adopting" in capsys.readouterr().out


@pytest.mark.parametrize("flags", [
    ["--record", "unlabelled"],
    ["--label", "billing"],
    ["--split", "evaluate"],
    ["--min-coverage", "0.9"],
    ["--source", "unused cohort"],
    ["--confidence", "0.9"],
    ["--min-accepted-accuracy", "0.9"],
    ["--require-qualification"],
    ["--qualification", "unused.json"],
    ["--group", "unused group"],
])
def test_teach_cli_rejects_unused_or_incomplete_options(tmp_path, capsys, flags):
    assert main(["teach", str(tmp_path / "skill"), *flags, "--json"]) == 1
    assert "error" in json.loads(capsys.readouterr().out)
    assert not (tmp_path / "skill").exists()


def test_teach_cli_failed_assessment_has_distinct_exit_and_no_current(tmp_path, capsys):
    from system1 import TeachingSession

    schema, dataset = setup_files(tmp_path)
    data = json.loads(dataset.read_text())
    data["evaluate"] = [[text, "support" if label == "billing" else "billing"] for text, label in data["evaluate"]]
    dataset.write_text(json.dumps(data))
    directory = tmp_path / "skill"
    prefix = ["teach", str(directory)]
    assert main(prefix + ["--schema", str(schema), "--dataset", str(dataset), "--json"]) == 0
    capsys.readouterr()
    assert main(prefix + ["--assess", "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert not report["passed"] and report["candidate"]["accepted_errors"] > 0
    assert not TeachingSession(directory).current_path.exists()


def test_teach_cli_qualified_adoption_history_prediction_and_rollback(tmp_path, capsys):
    from system1 import TeachingSession

    schema, dataset = setup_files(tmp_path)
    directory = tmp_path / "skill"
    prefix = ["teach", str(directory)]
    assert main(prefix + ["--schema", str(schema), "--choice-solver", "logistic", "--dataset", str(dataset), "--json"]) == 0
    state = json.loads(capsys.readouterr().out)
    assert state["settings"]["choice_solver"] == "logistic"
    assert main(prefix + ["--choice-solver", "ridge", "--json"]) == 1
    assert "differs" in json.loads(capsys.readouterr().out)["error"]
    assert main(prefix + ["--assess", "--require-qualification", "--json"]) == 0
    assessment = json.loads(capsys.readouterr().out)
    assert assessment["candidate"]["accepted_correctness_lower_bound"] < .95
    assert set(assessment["candidate"]["per_class"]) == {"billing", "support"}
    assert main(prefix + ["--adopt", "--json"]) == 1
    assert "qualification" in json.loads(capsys.readouterr().out)["error"]

    cohort = tmp_path / "qualification.json"
    # Synthetic independent test fixture exercises artifact binding and gates;
    # it makes no claim to represent actual incoming requests.
    cohort.write_text(json.dumps([
        [f"{phrase} qualification request {index}", label]
        for label, phrase in (("billing", "invoice payment refund"), ("support", "software crash bug"))
        for index in range(60)
    ]))
    assert main(prefix + ["--qualification", str(cohort), "--source", "Synthetic CLI integration fixture", "--json"]) == 0
    qualification = json.loads(capsys.readouterr().out)
    assert qualification["passed"]
    assert qualification["identity"] == assessment["identity"]
    assert main(prefix + ["--qualification", str(cohort), "--source", "Attempted reuse", "--json"]) == 1
    assert "frozen" in json.loads(capsys.readouterr().out)["error"]
    assert main(prefix + ["--adopt", "--json"]) == 0
    capsys.readouterr()
    session = TeachingSession(directory)
    first = session.current_path.read_bytes()
    first_revision = session.history[0]["revision"]
    assert main(prefix + ["--predict", "xyzzynonexistent blorbunknown", "--json"]) == 0
    decision = json.loads(capsys.readouterr().out)
    assert decision["review"] and "unknown_vocabulary" in decision["review_reasons"]
    assert decision["revision"] == first_revision

    assert main(prefix + ["--record", "invoice payment refund additional lesson", "--label", "billing", "--json"]) == 0
    capsys.readouterr()
    assert main(prefix + ["--assess", "--json"]) == 0
    capsys.readouterr()
    assert main(prefix + ["--adopt", "--json"]) == 0
    capsys.readouterr()
    assert main(prefix + ["--history", "--json"]) == 0
    history = json.loads(capsys.readouterr().out)
    assert len(history) == 2 and sum(item["current"] for item in history) == 1
    assert next(item for item in history if item["revision"] == first_revision)["qualification"]
    records = TeachingSession(directory).records
    assert main(prefix + ["--rollback", first_revision, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["current"]
    reopened = TeachingSession(directory)
    assert reopened.current_path.read_bytes() == first and reopened.records == records
    assert main(prefix + ["--adopt", "--json"]) == 1
    assert "stale" in json.loads(capsys.readouterr().out)["error"]


def test_compile_cli_exposes_existing_logistic_solver(tmp_path, capsys):
    from system1 import CompiledSystemOneModel

    schema, dataset = setup_files(tmp_path)
    lessons = json.loads(dataset.read_text())
    dataset.write_text(json.dumps({"team": lessons["teach"]}))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"team": lessons["calibrate"]}))
    model_path = tmp_path / "skill.s1m"
    assert main(["compile", "--schema", str(schema), "--dataset", str(dataset),
                 "--calibration-dataset", str(calibration), "--choice-solver", "logistic",
                 "--output", str(model_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["choice_solver"] == "logistic"
    assert CompiledSystemOneModel.load(model_path).metadata["choice_solver"] == "logistic"


def test_teach_cli_preserves_group_isolation_and_failed_import(tmp_path, capsys):
    from system1 import TeachingSession

    schema, dataset = setup_files(tmp_path)
    dataset.write_text(json.dumps({'teach': [
        {'input': 'Invoice from one conversation', 'label': 'billing', 'group': 'conversation-one'},
        {'input': 'Bug from another conversation', 'label': 'support', 'group': 'conversation-two'},
    ]}))
    directory = tmp_path / 'skill'
    prefix = ['teach', str(directory)]
    assert main(prefix + ['--schema', str(schema), '--dataset', str(dataset), '--json']) == 0
    capsys.readouterr()
    before = TeachingSession(directory).records
    assert before[0]['group'] == 'conversation-one'
    assert main(prefix + ['--record', 'Different wording from the same conversation', '--label', 'billing',
                          '--split', 'evaluate', '--group', 'conversation-one', '--json']) == 1
    assert 'groups' in json.loads(capsys.readouterr().out)['error']
    assert TeachingSession(directory).records == before
