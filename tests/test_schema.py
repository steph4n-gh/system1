"""Tests for System 1 Decision Schema definitions."""

import pytest

from system1 import (
    BooleanField,
    ChoiceField,
    DecisionField,
    DecisionSchema,
    MultiChoiceField,
    ScoreField,
)


def test_choice_field_validation():
    field = ChoiceField(options=["route_a", "route_b", "route_c"], default="route_a")
    field.bind_name("route")

    assert field.validate_value("route_b") == "route_b"
    assert field.validate_value(None) == "route_a"

    with pytest.raises(ValueError, match="not permitted"):
        field.validate_value("unknown_route")

    with pytest.raises(TypeError):
        field.validate_value(123)


def test_choice_field_invalid_initialization():
    with pytest.raises(ValueError, match="at least one option"):
        ChoiceField(options=[])

    with pytest.raises(ValueError, match="Duplicate option"):
        ChoiceField(options=["a", "b", "a"])

    with pytest.raises(ValueError, match="Default value"):
        ChoiceField(options=["a", "b"], default="c")


def test_boolean_field_validation():
    b = BooleanField(threshold=0.6, default=True)
    b.bind_name("is_admin")

    assert b.validate_value(True) is True
    assert b.validate_value(False) is False
    assert b.validate_value(1) is True
    assert b.validate_value(0) is False
    assert b.validate_value("true") is True
    assert b.validate_value("deny") is False
    assert b.validate_value(None) is True

    with pytest.raises(TypeError):
        b.validate_value("invalid_bool_string")


def test_multi_choice_field_validation():
    mc = MultiChoiceField(options=["opt1", "opt2", "opt3"], default=["opt1"])
    mc.bind_name("tags")

    assert mc.validate_value(["opt1", "opt2"]) == ("opt1", "opt2")
    assert mc.validate_value(None) == ("opt1",)

    with pytest.raises(ValueError, match="Invalid option"):
        mc.validate_value(["opt1", "bad_opt"])

    with pytest.raises(TypeError):
        mc.validate_value("not_a_sequence")


def test_score_field_validation():
    sf = ScoreField(min_value=0.0, max_value=10.0, default=5.0)
    sf.bind_name("rating")

    assert sf.validate_value(7.5) == 7.5
    assert sf.validate_value(0) == 0.0
    assert sf.validate_value(10.0) == 10.0
    assert sf.validate_value(None) == 5.0

    with pytest.raises(ValueError, match="outside bounds"):
        sf.validate_value(15.0)

    with pytest.raises(ValueError, match="outside bounds"):
        sf.validate_value(-1.0)

    with pytest.raises(TypeError):
        sf.validate_value(True)


def test_declarative_schema_subclass():
    class TestPipelineSchema(DecisionSchema):
        action = ChoiceField(options=["deploy", "test", "rollback"])
        dry_run = BooleanField(default=False)
        confidence_cutoff = ScoreField(min_value=0.0, max_value=1.0)
        flags = MultiChoiceField(options=["skip_ci", "notify_slack"])

    schema = TestPipelineSchema()
    assert len(schema.fields) == 4
    assert schema.get_field("action").name == "action"

    validated = schema.validate_decision({
        "action": "deploy",
        "confidence_cutoff": 0.95,
        "flags": ["skip_ci"],
    })

    assert validated["action"] == "deploy"
    assert validated["dry_run"] is False
    assert validated["confidence_cutoff"] == 0.95
    assert validated["flags"] == ("skip_ci",)

    with pytest.raises(ValueError, match="Required field 'confidence_cutoff' missing"):
        schema.validate_decision({"action": "deploy"})


def test_schema_serialization_and_digest_stability():
    class S1(DecisionSchema):
        f1 = ChoiceField(options=["alpha", "beta"])
        f2 = BooleanField()

    s = S1()
    digest1 = s.schema_digest()
    assert isinstance(digest1, str)
    assert len(digest1) == 64

    # Serialization to dict and reconstruction
    d = s.to_dict()
    reconstructed = DecisionSchema.from_dict(d)
    digest2 = reconstructed.schema_digest()

    assert digest1 == digest2
    assert "f1" in reconstructed.fields
    assert "f2" in reconstructed.fields

    # JSON schema export
    js = s.to_json_schema()
    assert js["type"] == "object"
    assert "f1" in js["properties"]
    assert js["properties"]["f1"]["enum"] == ["alpha", "beta"]
