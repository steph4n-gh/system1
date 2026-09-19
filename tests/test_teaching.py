"""Teaching one local skill must preserve examples, evidence, and portability."""

import json

import numpy as np
import pytest

from system1 import BooleanField, ChoiceField, DecisionSchema, MultiChoiceField, ScoreField, System1Engine
from system1.cli import main
from system1.compiler import CompiledSystemOneModel, SystemOneCompiler


class Route(DecisionSchema):
    team = ChoiceField(options=["billing", "support"])


EXAMPLES = {"team": [
    ("Refund my payment", "billing"), ("Correct the invoice", "billing"),
    ("Subscription charged twice", "billing"), ("Payment receipt requested", "billing"),
    ("Debug this exception", "support"), ("Fix a software crash", "support"),
    ("API request fails", "support"), ("Investigate error logs", "support"),
]}


def test_examples_only_never_generate_synthetic_data(monkeypatch):
    compiler = SystemOneCompiler(Route, dimension=128)
    monkeypatch.setattr(compiler, "generate_synthetic_exemplars", lambda **_: pytest.fail("Invented examples"))
    skill = compiler.compile(EXAMPLES, augment=False, calibration_split=0)
    assert skill.metadata["teaching_mode"] == "examples"
    assert skill.metadata["sample_counts"]["team"]["fit"] == 8
    assert skill.metadata["sample_counts"]["team"]["generated"] == 0
    assert skill.forward_single("Refund my payment").fields["team"].selected_value == "billing"


@pytest.mark.parametrize("examples", [
    {"team": [("Invoice dispute", "billng")]},
    {"tema": [("Invoice dispute", "billing")]},
    {"team": [("", "billing")]}, {"team": [("Invoice dispute", None)]},
    {"team": ["Invoice dispute"]},
    {"team": None}, [],
])
def test_bad_examples_are_rejected_instead_of_relabelled(examples):
    with pytest.raises(ValueError):
        SystemOneCompiler(Route).compile(examples)


def test_all_field_types_validate_labels():
    class Fields(DecisionSchema):
        active = BooleanField()
        tags = MultiChoiceField(options=["urgent", "routine"])
        score = ScoreField(min_value=0, max_value=10)

    compiler = SystemOneCompiler(Fields, dimension=32)
    for field, label in [("active", "maybe"), ("tags", ["unknown"]), ("score", float("nan")), ("score", 11)]:
        with pytest.raises(ValueError, match="Invalid label"):
            compiler.compile({field: [("example", label)]})
    validated = compiler._validate_exemplars({"active": [("negative example", "false")]})
    assert validated["active"][0][1] is False


def test_calibration_never_changes_learned_weights():
    compiler = SystemOneCompiler(Route, dimension=64)
    calibration = {"team": [("Held-out invoice", "billing"), ("Held-out error", "support")]}
    first = compiler.compile(EXAMPLES, augment=False, calibration_exemplars=calibration)
    flipped = {"team": [(prompt, "support" if label == "billing" else "billing") for prompt, label in calibration["team"]]}
    second = compiler.compile(EXAMPLES, augment=False, calibration_exemplars=flipped)
    np.testing.assert_array_equal(first.heads["team"].weights, second.heads["team"].weights)
    np.testing.assert_array_equal(first.heads["team"].biases, second.heads["team"].biases)


def test_duplicate_prompts_cannot_cross_calibration_boundary():
    compiler = SystemOneCompiler(Route, dimension=64)
    with pytest.raises(ValueError, match="disjoint"):
        compiler.compile(EXAMPLES, augment=False, calibration_exemplars={"team": [("  REFUND  my payment ", "billing")]})
    repeated = EXAMPLES["team"] * 3
    fit, held_out = compiler._split_samples(repeated, Route().fields["team"], 0.25)
    key = compiler._prompt_key
    assert {key(row[0]) for row in fit}.isdisjoint(key(row[0]) for row in held_out)
    temp, conformal = compiler._split_samples(held_out, None, 0.5)
    assert {key(row[0]) for row in temp}.isdisjoint(key(row[0]) for row in conformal)


def test_unchecked_skill_remains_uncalibrated_after_loading(tmp_path):
    path = tmp_path / "route.s1m"
    skill = SystemOneCompiler(Route, dimension=128).compile(EXAMPLES, augment=False, calibration_split=0)
    skill.save(path)
    restored = CompiledSystemOneModel.load(path)
    assert restored.heads["team"].calibration_scores == ()
    assert restored.heads["team"].conformal_quantile == 0
    engine = System1Engine(restored.schema, model=restored, strict_mode=True)
    decision = engine.decide("Refund my payment")
    assert decision.is_ambiguous
    assert set(decision.conformal_sets["team"]) == {"billing", "support"}
    np.testing.assert_array_equal(skill.heads["team"].weights, restored.heads["team"].weights)


@pytest.mark.parametrize("split", [-0.1, 1.0, float("nan")])
def test_invalid_calibration_split_is_rejected(split):
    with pytest.raises(ValueError, match="calibration_split"):
        SystemOneCompiler(Route).compile(EXAMPLES, augment=False, calibration_split=split)


def test_examples_must_cover_every_class():
    with pytest.raises(ValueError, match="every class"):
        SystemOneCompiler(Route).compile({"team": [("Invoice", "billing")]}, augment=False)


def test_cli_teach_save_and_decide(tmp_path, capsys):
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(Route().to_dict()))
    examples = tmp_path / "examples.json"
    examples.write_text(json.dumps(EXAMPLES))
    model = tmp_path / "route.s1m"
    assert main(["compile", "--schema", str(schema), "--dataset", str(examples),
                 "--dimension", "128", "--output", str(model), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["teaching_mode"] == "examples"
    assert report["sample_counts"]["team"]["generated"] == 0
    assert main(["decide", "Refund my payment", "--model", str(model), "--json"]) == 0
    decision = json.loads(capsys.readouterr().out)
    assert decision["schema_name"] == "Route"
    assert decision["values"]["team"] == "billing"
    assert decision["receipt"]


@pytest.mark.parametrize("content", [None, "[]", '{}', '[{"prompt":"invoice"}]', '{"team":[["invoice","typo"]]}'])
def test_cli_bad_dataset_does_not_fall_back_to_synthetic(tmp_path, capsys, content):
    dataset, output = tmp_path / "examples.json", tmp_path / "model.s1m"
    if content is not None:
        dataset.write_text(content)
    assert main(["compile", "--schema", "triage", "--dataset", str(dataset), "--output", str(output)]) == 1
    assert not output.exists()
    assert "ERROR" in capsys.readouterr().err


def test_evaluation_keeps_repeated_prompts_in_one_fold():
    from benchmarks.quality.evaluate_teaching import make_folds

    rows = [{"prompt": prompt, "label": label} for prompt, label in EXAMPLES["team"]]
    rows.append({"prompt": "  REFUND   MY PAYMENT ", "label": "billing"})
    folds = make_folds(rows, "label")
    assert folds[0] == folds[-1]
    assert len(folds) == len(rows)
