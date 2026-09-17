"""Tests for Reflex Engine and Developer API."""

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import reflex
from reflex import (
    ActionLedger,
    BooleanField,
    ChoiceField,
    DecisionResult,
    DecisionSchema,
    MultiChoiceField,
    ReflexEngine,
    ScoreField,
    SystemOneEngine,
    decide,
)


class CompleteServiceSchema(DecisionSchema):
    tier = ChoiceField(
        options=["free", "pro", "enterprise"],
        descriptions={
            "free": "Basic tier with standard rate limits",
            "pro": "Professional tier with priority queues",
            "enterprise": "Enterprise tier with dedicated VPC and SLA",
        },
    )
    is_compliant = BooleanField(
        description="Whether request complies with governance and privacy policies",
        threshold=0.5,
    )
    sla_score = ScoreField(
        min_value=0.0,
        max_value=1.0,
        description="Priority score for SLA dispatch",
    )
    capabilities = MultiChoiceField(
        options=["gpu", "high_memory", "persistent_storage"],
    )


def test_engine_decide_and_dynamic_attributes():
    signing_key = Ed25519PrivateKey.generate()
    engine = ReflexEngine(CompleteServiceSchema, signing_key=signing_key)

    res = engine.decide(
        "Request dedicated enterprise cluster with GPU acceleration and compliance review",
        alpha=0.05,
        record_receipt=True,
    )

    assert isinstance(res, DecisionResult)
    assert res.schema_name == "CompleteServiceSchema"

    # Test dynamic attribute access
    assert res.tier in ("free", "pro", "enterprise")
    assert isinstance(res.is_compliant, bool)
    assert 0.0 <= res.sla_score <= 1.0
    assert isinstance(res.capabilities, tuple)

    # Test confidences and conformal sets
    assert "tier" in res.confidences
    assert 0.0 <= res.confidences["tier"] <= 1.0
    assert "tier" in res.conformal_sets
    assert len(res.conformal_sets["tier"]) >= 1

    # Receipt verification
    assert res.receipt is not None
    assert res.receipt.envelope is not None
    assert res.receipt.envelope.signature is not None

    # Performance
    assert res.latency_ms < 20.0


def test_top_level_reflex_decide_api():
    result = reflex.decide(
        "Deploy to free staging environment without enterprise SLA",
        schema=CompleteServiceSchema,
    )

    assert isinstance(result, DecisionResult)
    assert result.tier in ("free", "pro", "enterprise")
    assert result.latency_ms < 20.0


def test_engine_calibration_workflow():
    engine = ReflexEngine(CompleteServiceSchema)

    calib_dataset = [
        ("Free tier user ping", {"tier": "free", "is_compliant": True}),
        ("Pro user background worker", {"tier": "pro", "is_compliant": True}),
        ("Enterprise cluster with dedicated resources", {"tier": "enterprise", "is_compliant": True}),
        ("Malicious injection attack", {"tier": "free", "is_compliant": False}),
        ("Enterprise customer security audit", {"tier": "enterprise", "is_compliant": True}),
    ]

    metrics = engine.calibrate(calib_dataset, n_bins=5)
    assert "tier" in metrics
    assert "is_compliant" in metrics

    m_tier = metrics["tier"]
    assert m_tier.optimal_temperature > 0.0
    assert m_tier.expected_calibration_error >= 0.0
    assert m_tier.brier.total_brier >= 0.0

    res = engine.decide("Enterprise customer audit request")
    assert res.tier == "enterprise"
    assert "enterprise" in res.conformal_sets["tier"]


def test_engine_benchmark_beats_jev():
    engine = ReflexEngine(CompleteServiceSchema)
    prompts = [
        "Read file system metadata",
        "Enterprise workload allocation",
        "Free tier healthcheck",
    ]

    report = engine.benchmark(prompts, iterations=30, warmup=5)

    assert report.total_decisions == 30
    assert report.p50_latency_ms < 10.0
    assert report.p95_latency_ms < 20.0
    assert report.beats_jev is True
    assert report.speedup_factor_vs_jev_p50 > 5.0
    assert report.throughput_decisions_per_sec > 50.0


def test_engine_action_ledger_integration(tmp_path):
    ledger_db = tmp_path / "test_ledger.sqlite"
    ledger = ActionLedger(str(ledger_db))

    signing_key = Ed25519PrivateKey.generate()
    engine = ReflexEngine(CompleteServiceSchema, signing_key=signing_key, ledger=ledger)

    # First decision
    res = engine.decide(
        "Authorize enterprise modification",
        record_receipt=True,
    )

    # Initial head is 64 zeros
    assert res.receipt.truth_ledger_head == "0000000000000000000000000000000000000000000000000000000000000000"
    assert res.receipt.envelope.truth_ledger_head == res.receipt.truth_ledger_head
    assert res.receipt.ledger_record_id is not None
    assert len(res.receipt.ledger_record_id) == 64
    assert ledger.audit_head()[0] == 1
    assert ledger.audit_head()[1] == res.receipt.ledger_record_id
    assert ledger.verify_integrity() is True

    # Second decision: should chain to first decision's record ID as head
    res2 = engine.decide(
        "Inspect pro cluster health metrics",
        record_receipt=True,
    )
    assert res2.receipt.truth_ledger_head == res.receipt.ledger_record_id
    assert res2.receipt.ledger_record_id is not None
    assert res2.receipt.ledger_record_id != res.receipt.ledger_record_id
    assert ledger.audit_head()[0] == 2
    assert ledger.audit_head()[1] == res2.receipt.ledger_record_id
    assert ledger.verify_integrity() is True


def test_engine_multichoice_and_score_calibration():
    engine = ReflexEngine(CompleteServiceSchema)
    calib_dataset = [
        ("Deploy GPU enterprise service", {
            "tier": "enterprise",
            "is_compliant": True,
            "sla_score": 0.95,
            "capabilities": ["gpu", "persistent_storage"],
        }),
        ("Free tier basic task", {
            "tier": "free",
            "is_compliant": True,
            "sla_score": 0.10,
            "capabilities": [],
        }),
        ("Pro high memory task", {
            "tier": "pro",
            "is_compliant": True,
            "sla_score": 0.55,
            "capabilities": ["high_memory"],
        }),
    ]

    metrics = engine.calibrate(calib_dataset, n_bins=2)
    assert "capabilities" in metrics
    assert "sla_score" in metrics
    assert metrics["capabilities"].optimal_temperature > 0.0
    assert metrics["sla_score"].optimal_temperature > 0.0
    assert engine.regression_conformal_predictors["sla_score"].is_calibrated is True

    res = engine.decide("Enterprise cluster with GPU")
    assert isinstance(res.capabilities, tuple)
    assert isinstance(res.sla_score, float)
    assert len(res.conformal_sets["capabilities"]) >= 1
    assert len(res.conformal_sets["sla_score"]) == 1


def test_engine_prompt_type_validation():
    engine = ReflexEngine(CompleteServiceSchema)
    with pytest.raises(TypeError, match="Prompt must be a string"):
        engine.decide(None)  # type: ignore

    with pytest.raises(TypeError, match="Prompt must be a string"):
        reflex.decide(None, schema=CompleteServiceSchema)  # type: ignore


def test_engine_invalid_choice_label_in_calibration():
    engine = ReflexEngine(CompleteServiceSchema)
    invalid_dataset = [("Deploy service", {"tier": "invalid_tier_option"})]
    with pytest.raises(ValueError, match="not permitted"):
        engine.calibrate(invalid_dataset)
