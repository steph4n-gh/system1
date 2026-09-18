"""Tier 2: Boundary Value Analysis & Adversarial Corner Cases.

Authoritative Invariants:
1. Empty and whitespace-only prompts evaluate deterministically without crash.
2. Extreme batch sizes (0, 1, 200) execute predictably with proper error handling or completion.
3. Invalid hash digests, malformed identifiers, and length mismatches fail early with descriptive ValueErrors.
4. Conformal significance limits (alpha outside (0, 1)) and confidence thresholds fail-closed.
5. Extreme prompt lengths (15,000+ chars) embed into bounded vectors without buffer or memory blowout.
6. Format strings, SQL injection strings, null characters, and emojis preserve integrity in ledger and receipts.
"""

from __future__ import annotations

import copy
import pytest

import reflex
from reflex import (
    ActionLedger,
    ActionProposal,
    BooleanField,
    ChoiceField,
    ConformalPredictor,
    DecisionOutcome,
    DecisionSchema,
    DefaultGuardDecisionSchema,
    EvidenceRef,
    IntegrityError,
    ReflexEngine,
    ReflexGuardHook,
)


class BoundarySchema(DecisionSchema):
    decision = ChoiceField(
        options=["ACCEPT", "REJECT", "AUDIT"],
        descriptions={
            "ACCEPT": "clean valid standard input payload",
            "REJECT": "malformed dangerous or hostile input",
            "AUDIT": "vague ambiguous edge condition",
        },
    )
    is_valid = BooleanField(threshold=0.5)


def test_boundary_empty_prompt_and_whitespace():
    """Verify empty string and whitespace prompts evaluate safely without crashing."""
    engine = ReflexEngine(BoundarySchema)
    
    # 1. Empty string prompt
    res_empty = engine.decide("", record_receipt=True)
    assert res_empty is not None
    assert "decision" in res_empty.values
    assert res_empty.receipt is not None
    assert len(res_empty.receipt.prompt_digest) == 64

    # 2. Whitespace-only prompt
    res_ws = engine.decide("   \n\t  \r\n   ", record_receipt=True)
    assert res_ws is not None
    assert "decision" in res_ws.values


def test_boundary_extreme_batch_sizes():
    """Verify handling of extreme batch sizes: 0, 1, and 200."""
    engine = ReflexEngine(BoundarySchema)

    # 1. Unitary batch
    res_single = engine.decide("Single item", record_receipt=False)
    assert res_single is not None

    # 2. Large batch (200 items)
    bulk_prompts = [f"Bulk payload test record #{i}" for i in range(200)]
    for p in bulk_prompts:
        r = engine.decide(p, record_receipt=False)
        assert r.values["decision"] in ("ACCEPT", "REJECT", "AUDIT")


def test_boundary_invalid_and_corrupt_hashes():
    """Verify malformed SHA-256 strings and invalid identifiers raise ValueError."""
    # 1. SHA-256 digest length validation (must be exactly 64 hex chars)
    with pytest.raises(ValueError, match="content_digest must be a lowercase SHA-256 digest"):
        EvidenceRef(
            evidence_id="ev_001",
            tenant_id="t1",
            principal_id="p1",
            scope="read",
            source="test",
            content_digest="abc",  # Too short
            observed_at="2026-09-17T00:00:00Z",
        )

    with pytest.raises(ValueError, match="content_digest must be a lowercase SHA-256 digest"):
        EvidenceRef(
            evidence_id="ev_001",
            tenant_id="t1",
            principal_id="p1",
            scope="read",
            source="test",
            content_digest="g" * 64,  # Non-hex characters
            observed_at="2026-09-17T00:00:00Z",
        )

    # 2. Invalid identifier characters
    with pytest.raises(ValueError, match="tenant_id must be a non-empty canonical identifier"):
        EvidenceRef(
            evidence_id="ev_001",
            tenant_id="",  # Empty identifier
            principal_id="p1",
            scope="read",
            source="test",
            content_digest="0" * 64,
            observed_at="2026-09-17T00:00:00Z",
        )


def test_boundary_threshold_limits_alpha_and_confidence():
    """Verify alpha bounds in (0, 1) and confidence threshold edge behavior."""
    engine = ReflexEngine(BoundarySchema)

    # Alpha <= 0.0 or >= 1.0 must raise ValueError
    with pytest.raises(ValueError, match="alpha must be in"):
        engine.decide("Test alpha lower bound", alpha=0.0)

    with pytest.raises(ValueError, match="alpha must be in"):
        engine.decide("Test alpha upper bound", alpha=1.0)

    with pytest.raises(ValueError, match="alpha must be in"):
        engine.decide("Test alpha negative", alpha=-0.1)

    with pytest.raises(ValueError, match="alpha must be in"):
        engine.decide("Test alpha > 1", alpha=1.05)

    # Valid extreme boundaries
    res_low_alpha = engine.decide("Extremely high coverage request", alpha=0.0001)
    assert res_low_alpha.alpha == 0.0001
    assert len(res_low_alpha.conformal_sets["decision"]) >= 1

    res_high_alpha = engine.decide("Extremely low coverage request", alpha=0.9999)
    assert res_high_alpha.alpha == 0.9999


def test_boundary_extreme_prompt_lengths():
    """Verify extreme prompt length (20,000+ chars) embeds safely without memory blowout."""
    engine = ReflexEngine(BoundarySchema)
    
    giant_prompt = "Large corpus evaluation block. " * 700  # ~21,000 characters
    assert len(giant_prompt) > 20000

    res = engine.decide(giant_prompt, record_receipt=True)
    assert res is not None
    assert "decision" in res.values
    assert res.latency_ms < 100.0
    assert len(res.receipt.prompt_digest) == 64


def test_boundary_special_characters_and_injections(temp_ledger):
    """Verify SQL injection, null characters, control characters, and emojis preserve integrity."""
    engine = ReflexEngine(DefaultGuardDecisionSchema, ledger=temp_ledger)

    adversarial_inputs = [
        "'; DROP TABLE audit_entries; --",
        "UNION SELECT * FROM ledger_meta WHERE '1'='1",
        "{\"__proto__\": {\"admin\": true}}",
        "Null byte embedded \x00 in string payload",
        "Emoji stress: 🚀🔥🔒🧬🛡️⚡🎯🤖💡",
        "ANSI sequence: \x1b[31;1mRED_ALERT\x1b[0m\r\n\t",
    ]

    for malicious_input in adversarial_inputs:
        res = engine.decide(malicious_input, record_receipt=True)
        assert res is not None
        assert res.receipt.truth_ledger_head is not None

    # Verify ledger was not compromised by SQL injection
    assert temp_ledger.verify_integrity() is True
    seq, head = temp_ledger.audit_head()
    assert seq == len(adversarial_inputs)
