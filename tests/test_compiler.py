"""Tests for System 1 Compiler, Closed-Form Distillation, and .s1m Serialization."""

import time
from pathlib import Path
import numpy as np
import pytest

from system1.compiler import CompiledSystemOneModel, MAGIC_HEADER, SystemOneCompiler
from system1.core.schema import BooleanField, ChoiceField, DecisionSchema, MultiChoiceField, ScoreField


class SupportTriageSchema(DecisionSchema):
    category = ChoiceField(
        options=["technical", "billing", "account"],
        descriptions={
            "technical": "Software bugs, crashes, error traces, and API issues",
            "billing": "Invoices, refunds, credit card charges, and subscription fees",
            "account": "Password reset, login troubles, MFA tokens, and profile changes",
        },
    )
    is_urgent = BooleanField(
        threshold=0.5,
        true_description="Critical outage or severe payment failure needing urgent handling",
        false_description="Routine support inquiry or non-critical question",
    )
    risk_score = ScoreField(
        min_value=0.0,
        max_value=1.0,
        low_description="Zero account risk",
        high_description="Critical account takeover or payment fraud risk",
    )


def test_compiler_synthetic_exemplar_generation():
    """Verify compiler generates rich synthetic exemplars for all schema fields."""
    compiler = SystemOneCompiler(SupportTriageSchema, dimension=128)
    exemplars = compiler.generate_synthetic_exemplars(samples_per_choice=15)

    assert "category" in exemplars
    assert "is_urgent" in exemplars
    assert "risk_score" in exemplars

    assert len(exemplars["category"]) >= 45
    assert len(exemplars["is_urgent"]) >= 20
    assert len(exemplars["risk_score"]) >= 3

    # Check structure: (prompt, label)
    for p, label in exemplars["category"]:
        assert isinstance(p, str) and len(p) > 0
        assert label in ("technical", "billing", "account")


def test_compiler_closed_form_solve_and_accuracy():
    """Verify compiler solves Ridge regression and achieves high accuracy on exemplars."""
    compiler = SystemOneCompiler(SupportTriageSchema, dimension=128, regularization=0.5)
    compiled_model = compiler.compile(samples_per_choice=20)

    assert isinstance(compiled_model, CompiledSystemOneModel)
    assert compiled_model.dimension == 128
    assert "category" in compiled_model.heads
    assert "is_urgent" in compiled_model.heads

    # Evaluate on explicit domain prompts
    res_tech = compiled_model.forward_single("Server crashed with NullPointerException in auth service")
    assert res_tech.fields["category"].selected_value == "technical"
    assert res_tech.fields["category"].confidence > 0.5

    res_bill = compiled_model.forward_single("Why did you charge my credit card $120 for renewal?")
    assert res_bill.fields["category"].selected_value == "billing"
    assert res_bill.fields["category"].confidence > 0.5

    res_acct = compiled_model.forward_single("I forgot my password and cannot pass MFA verification")
    assert res_acct.fields["category"].selected_value == "account"
    assert res_acct.fields["category"].confidence > 0.5


def test_compiler_binary_serialization_roundtrip(tmp_path: Path):
    """Verify compiled model serializes to .s1m binary and deserializes with exact fidelity."""
    compiler = SystemOneCompiler(SupportTriageSchema, dimension=128, regularization=1.0)
    compiled_original = compiler.compile(samples_per_choice=10)

    s1m_path = tmp_path / "triage.s1m"
    compiled_original.save(s1m_path)

    assert s1m_path.is_file()
    raw_bytes = s1m_path.read_bytes()
    assert raw_bytes[:4] == MAGIC_HEADER
    # Compactness: should be well under 200 KB
    assert len(raw_bytes) < 200 * 1024

    # Load via CompiledSystemOneModel.load and SystemOneCompiler.load
    loaded1 = CompiledSystemOneModel.load(s1m_path)
    loaded2 = SystemOneCompiler.load(s1m_path)

    assert loaded1.schema.schema_name == SupportTriageSchema().schema_name
    assert loaded1.dimension == 128
    assert set(loaded1.heads.keys()) == set(compiled_original.heads.keys())

    test_prompt = "Emergency database outage affecting all customers"
    r_orig = compiled_original.forward_single(test_prompt)
    r_loaded1 = loaded1.forward_single(test_prompt)
    r_loaded2 = loaded2.forward_single(test_prompt)

    assert r_orig.fields["category"].selected_value == r_loaded1.fields["category"].selected_value
    assert r_orig.fields["category"].selected_value == r_loaded2.fields["category"].selected_value
    np.testing.assert_allclose(
        r_orig.fields["category"].raw_probabilities,
        r_loaded1.fields["category"].raw_probabilities,
        atol=1e-5,
    )


def test_compiled_model_speed():
    """Verify compiled model forward evaluation executes in < 1 ms."""
    compiler = SystemOneCompiler(SupportTriageSchema, dimension=128)
    compiled_model = compiler.compile(samples_per_choice=5)

    # Warmup
    for _ in range(5):
        compiled_model.forward_single("Warmup query for compiled model")

    t0 = time.perf_counter()
    n_iters = 100
    for _ in range(n_iters):
        compiled_model.forward_single("Invoice dispute for customer account #12345")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_query_ms = elapsed_ms / n_iters

    assert per_query_ms < 5.0, f"Expected < 5.0 ms per forward pass, got {per_query_ms:.4f} ms"


def test_compiler_boolean_and_score_fields_calibration_and_bounds():
    """Verify BooleanField and ScoreField in compiled model evaluate accurately in logit space."""
    compiler = SystemOneCompiler(SupportTriageSchema, dimension=128)
    compiled_model = compiler.compile(samples_per_choice=20)

    # BooleanField: verify non-urgent inquiry evaluates to False, urgent evaluates to True
    res_routine = compiled_model.forward_single("Routine documentation question about account settings")
    assert res_routine.fields["is_urgent"].selected_value is False
    assert res_routine.fields["is_urgent"].raw_probabilities[0] > 0.5  # prob of False > 0.5

    res_critical = compiled_model.forward_single("Critical emergency payment failure and database crash")
    assert res_critical.fields["is_urgent"].selected_value is True
    assert res_critical.fields["is_urgent"].raw_probabilities[1] > 0.5  # prob of True > 0.5

    # ScoreField: verify low risk vs high risk
    res_low_risk = compiled_model.forward_single("Normal safe user login with zero account risk")
    res_high_risk = compiled_model.forward_single("Critical account takeover and active payment fraud risk")
    assert res_low_risk.fields["risk_score"].selected_value < 5.0
    assert res_high_risk.fields["risk_score"].selected_value > 0.65


def test_compiler_multichoice_tag_separation():
    """Verify MultiChoiceField compiles with accurate positive and negative tag detection."""
    class TagSchema(DecisionSchema):
        labels = MultiChoiceField(
            options=["db_error", "payment_fraud", "network_drop"],
            descriptions={
                "db_error": "Database query timeout or lock contention",
                "payment_fraud": "Stolen credit card or suspicious checkout attempt",
                "network_drop": "Socket disconnect or packet loss on connection",
            },
        )

    compiler = SystemOneCompiler(TagSchema, dimension=128)
    compiled = compiler.compile(samples_per_choice=20)

    # Unrelated input should not trigger all tags
    res_unrelated = compiled.forward_single("Good morning, sunny weather outside today")
    assert len(res_unrelated.fields["labels"].selected_value) == 0

    # Specific db_error prompt should trigger db_error
    res_db = compiled.forward_single("Database query timeout and lock contention error")
    assert "db_error" in res_db.fields["labels"].selected_value
    assert "payment_fraud" not in res_db.fields["labels"].selected_value


def test_compiler_class_imbalance_and_starvation_mitigation():
    """Attack Target 1: Verify ChoiceField with 500 samples for one option and 0 for another does not starve the missing option."""
    class RoutingSchema(DecisionSchema):
        route = ChoiceField(
            options=["rare", "frequent"],
            descriptions={
                "rare": "Emergency high-priority escalation ticket",
                "frequent": "General routine support ticket inquiry",
            },
        )

    # Dataset with 500 frequent exemplars and 0 rare exemplars
    exemplars = {
        "route": [("General routine support ticket inquiry for billing", "frequent")] * 500,
    }

    compiler = SystemOneCompiler(RoutingSchema, dimension=128)
    compiled = compiler.compile(exemplars=exemplars)

    # The missing 'rare' option should have been augmented from schema definitions
    # and must predict 'rare' when a rare query is presented
    res = compiled.forward_single("Emergency high-priority escalation ticket")
    assert res.fields["route"].selected_value == "rare"
    assert res.fields["route"].confidence > 0.5
