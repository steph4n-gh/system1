"""Tests for Cryptographic Attestation Invariants (M1/M2).

Verifies:
- Invariant 3: Rejection of signature=None when public_key supplied or profile is product_signed_v1,
               and rejection of unauthenticated key substitution.
- Invariant 4: Canonical probabilities inclusion in receipts, digest computation, and verification
               (tampering with probabilities causes verification failure).
- Invariant 5: Strict fail-closed ledger mode raising LedgerWriteError on write failure,
               and reference monitor denying action on unrecorded/failed ledger writes.
- Execution Outcome Chaining: Cryptographic linkage of execution status (SUCCEEDED, FAILED, INDETERMINATE)
                              to pre-execution authorization receipts in ActionLedger.
"""

from __future__ import annotations

import copy
import sqlite3
from unittest.mock import MagicMock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from reflex import (
    ActionLedger,
    ActionProposal,
    DecisionOutcome,
    DefaultGuardDecisionSchema,
    LedgerError,
    LedgerWriteError,
    SystemOneEngine,
    SystemOneGuardHook,
    SystemOneMCPProxy,
    create_decision_receipt,
    verify_decision_witness_receipt,
)
from reflex.receipt import public_key_bytes


# ---------------------------------------------------------------------------
# Invariant 3: Cryptographic Signature Verification & Key Authenticity
# ---------------------------------------------------------------------------


def test_invariant_3_rejects_missing_signature_when_public_key_supplied():
    """verify_decision_witness_receipt must reject signature=None when public_key is provided."""
    signing_key = Ed25519PrivateKey.generate()
    pub_key = signing_key.public_key()

    receipt = create_decision_receipt(
        schema_name="TestSecurity",
        schema_digest="a" * 64,
        prompt="Safe read-only prompt",
        values={"is_safe": True},
        confidences={"is_safe": 0.99},
        conformal_sets={"is_safe": ["True"]},
        probabilities={"is_safe": {"True": 0.99, "False": 0.01}},
        latency_ms=1.2,
        is_ambiguous=False,
        signing_key=signing_key,
    )

    receipt_dict = receipt.to_dict()
    assert verify_decision_witness_receipt(receipt_dict, public_key=pub_key) is True

    # Strip signature from envelope
    tampered_dict = copy.deepcopy(receipt_dict)
    tampered_dict["envelope"]["signature"] = None

    # Must reject when public_key is provided
    assert verify_decision_witness_receipt(tampered_dict, public_key=pub_key) is False


def test_invariant_3_rejects_missing_signature_for_product_signed_profile():
    """verify_decision_witness_receipt rejects signature=None when profile == 'product_signed_v1'."""
    signing_key = Ed25519PrivateKey.generate()

    receipt = create_decision_receipt(
        schema_name="TestSecurity",
        schema_digest="b" * 64,
        prompt="Safe read-only prompt",
        values={"is_safe": True},
        confidences={"is_safe": 0.99},
        conformal_sets={"is_safe": ["True"]},
        probabilities={"is_safe": {"True": 0.99, "False": 0.01}},
        latency_ms=1.2,
        is_ambiguous=False,
        signing_key=signing_key,
    )

    receipt_dict = receipt.to_dict()
    assert receipt_dict["envelope"]["profile"] == "product_signed_v1"

    # Strip signature
    tampered_dict = copy.deepcopy(receipt_dict)
    tampered_dict["envelope"]["signature"] = None

    # Must reject even if public_key=None is passed
    assert verify_decision_witness_receipt(tampered_dict, public_key=None) is False


def test_invariant_3_rejects_unauthenticated_key_substitution():
    """Rejects unauthenticated key substitution where attacker replaces signer_public_key."""
    legit_key = Ed25519PrivateKey.generate()
    attacker_key = Ed25519PrivateKey.generate()

    receipt = create_decision_receipt(
        schema_name="SecurityGate",
        schema_digest="c" * 64,
        prompt="Delete production database records",
        values={"is_safe": False},
        confidences={"is_safe": 0.99},
        conformal_sets={"is_safe": ["False"]},
        probabilities={"is_safe": {"True": 0.01, "False": 0.99}},
        latency_ms=1.5,
        is_ambiguous=False,
        signing_key=legit_key,
    )

    receipt_dict = receipt.to_dict()
    # Verified against legitimate public key
    assert verify_decision_witness_receipt(receipt_dict, public_key=legit_key.public_key()) is True

    # Attacker tries to verify receipt against their own key
    assert verify_decision_witness_receipt(receipt_dict, public_key=attacker_key.public_key()) is False

    # Attacker attempts key substitution inside the receipt and envelope
    tampered_dict = copy.deepcopy(receipt_dict)
    attacker_pub_hex = public_key_bytes(attacker_key.public_key()).hex()
    tampered_dict["signer_public_key"] = attacker_pub_hex
    tampered_dict["envelope"]["effect_observation"]["signer_public_key"] = attacker_pub_hex

    # Fails against legit key due to declared key mismatch
    assert verify_decision_witness_receipt(tampered_dict, public_key=legit_key.public_key()) is False
    # Fails against attacker key due to signature mismatch
    assert verify_decision_witness_receipt(tampered_dict, public_key=attacker_key.public_key()) is False


# ---------------------------------------------------------------------------
# Invariant 4: Canonical Probabilities Inclusion & Verification
# ---------------------------------------------------------------------------


def test_invariant_4_canonical_probabilities_in_receipt_and_digest():
    """Canonical probabilities are included in receipt unsigned payload and affect the digest."""
    signing_key = Ed25519PrivateKey.generate()
    probs = {"is_safe": {"True": 0.95, "False": 0.05}}

    receipt1 = create_decision_receipt(
        schema_name="TestSchema",
        schema_digest="d" * 64,
        prompt="Evaluate action",
        values={"is_safe": True},
        confidences={"is_safe": 0.95},
        conformal_sets={"is_safe": ["True"]},
        probabilities=probs,
        latency_ms=2.0,
        is_ambiguous=False,
        signing_key=signing_key,
    )

    payload1 = receipt1.unsigned_payload()
    assert "probabilities" in payload1
    assert payload1["probabilities"] == probs
    digest1 = receipt1.compute_digest()

    # Different probabilities produce a different digest
    probs_altered = {"is_safe": {"True": 0.80, "False": 0.20}}
    receipt2 = create_decision_receipt(
        schema_name="TestSchema",
        schema_digest="d" * 64,
        prompt="Evaluate action",
        values={"is_safe": True},
        confidences={"is_safe": 0.95},
        conformal_sets={"is_safe": ["True"]},
        probabilities=probs_altered,
        latency_ms=2.0,
        is_ambiguous=False,
        signing_key=signing_key,
    )

    digest2 = receipt2.compute_digest()
    assert digest1 != digest2


def test_invariant_4_mutating_probabilities_breaks_verification():
    """Mutating probabilities in a signed receipt causes cryptographic verification to fail."""
    signing_key = Ed25519PrivateKey.generate()
    pub_key = signing_key.public_key()

    receipt = create_decision_receipt(
        schema_name="TestSchema",
        schema_digest="e" * 64,
        prompt="Query user accounts",
        values={"is_safe": True},
        confidences={"is_safe": 0.98},
        conformal_sets={"is_safe": ["True"]},
        probabilities={"is_safe": {"True": 0.98, "False": 0.02}},
        latency_ms=1.1,
        is_ambiguous=False,
        signing_key=signing_key,
    )

    receipt_dict = receipt.to_dict()
    assert verify_decision_witness_receipt(receipt_dict, public_key=pub_key) is True

    # Mutate probabilities in receipt top-level
    tampered1 = copy.deepcopy(receipt_dict)
    tampered1["probabilities"] = {"is_safe": {"True": 0.02, "False": 0.98}}
    assert verify_decision_witness_receipt(tampered1, public_key=pub_key) is False

    # Mutate probabilities inside envelope guard_receipt
    tampered2 = copy.deepcopy(receipt_dict)
    tampered2["envelope"]["guard_receipt"]["probabilities"] = {"is_safe": {"True": 0.50, "False": 0.50}}
    assert verify_decision_witness_receipt(tampered2, public_key=pub_key) is False


# ---------------------------------------------------------------------------
# Invariant 5: Fail-Closed Ledger Mode
# ---------------------------------------------------------------------------


def test_invariant_5_engine_decide_raises_ledger_write_error_when_fail_closed(tmp_path):
    """SystemOneEngine.decide() with fail_closed_ledger=True raises LedgerWriteError on append failure."""
    db_path = str(tmp_path / "test_ledger.db")
    ledger = ActionLedger(path=db_path)

    engine = SystemOneEngine(
        DefaultGuardDecisionSchema,
        ledger=ledger,
        fail_closed_ledger=True,
    )

    # Corrupt or mock ledger append to simulate disk full / database locked error
    def broken_append(*args, **kwargs):
        raise sqlite3.OperationalError("database disk image is malformed / disk full")

    ledger.append = broken_append

    with pytest.raises(LedgerWriteError) as excinfo:
        engine.decide(
            "Inspect safe project documentation",
            fail_closed_ledger=True,
        )

    assert "Fail-closed ledger recording failed" in str(excinfo.value)


def test_invariant_5_guard_hook_fails_closed_on_ledger_error(tmp_path):
    """SystemOneGuardHook denies tool execution and sentinel is called 0 times on ledger failure."""
    db_path = str(tmp_path / "guard_ledger.db")
    ledger = ActionLedger(path=db_path)

    hook = SystemOneGuardHook(ledger=ledger, fail_closed_ledger=True)

    # Force ledger append failure
    def broken_append(*args, **kwargs):
        raise LedgerWriteError("Simulated durable ledger write failure")

    ledger.append = broken_append

    proposal = ActionProposal.create(
        tenant_id="tenant_001",
        principal_id="principal_001",
        scope="system:read",
        tool="read_file",
        arguments={"path": "/Volumes/Storage/project/README.md"},
        canonical_target="/Volumes/Storage/project/README.md",
        purpose="Read project documentation",
    )

    interception = hook.evaluate_proposal(
        proposal,
        context_prompt="Inspect read-only project documentation in README.md",
    )

    assert interception.outcome == DecisionOutcome.DENY
    assert "Fail-closed Reference Monitor" in interception.reason
    assert not interception.allowed


def test_invariant_5_guard_hook_denies_on_unrecorded_receipt():
    """If ledger is active but receipt fails to record a ledger_record_id, reference monitor denies."""
    mock_engine = MagicMock()
    mock_decision = MagicMock()
    mock_receipt = MagicMock()
    # Unrecorded: ledger_record_id is None
    mock_receipt.ledger_record_id = None
    mock_decision.receipt = mock_receipt
    mock_decision.values = {"is_safe": True}
    mock_decision.conformal_sets = {"is_safe": ["True"]}
    mock_decision.confidences = {"is_safe": 0.99}
    mock_engine.decide.return_value = mock_decision

    fake_ledger = MagicMock()
    hook = SystemOneGuardHook(engine=mock_engine, ledger=fake_ledger, fail_closed_ledger=True)

    proposal = ActionProposal.create(
        tenant_id="tenant_001",
        principal_id="principal_001",
        scope="system:read",
        tool="read_file",
        arguments={"path": "/test/file.txt"},
        purpose="Read safe file",
    )

    interception = hook.evaluate_proposal(proposal)

    assert interception.outcome == DecisionOutcome.DENY
    assert "ActionLedger failed to durably record decision receipt" in interception.reason


# ---------------------------------------------------------------------------
# Execution Outcome Chaining
# ---------------------------------------------------------------------------


def test_execution_outcome_chaining_success(tmp_path):
    """ActionLedger chains post-execution SUCCEEDED status to prior authorization receipt."""
    db_path = str(tmp_path / "chain_ledger.db")
    ledger = ActionLedger(path=db_path)

    # 1. Record pre-execution authorization receipt
    signing_key = Ed25519PrivateKey.generate()
    receipt = create_decision_receipt(
        schema_name="ExecutionGate",
        schema_digest="f" * 64,
        prompt="Execute verified operation",
        values={"is_safe": True},
        confidences={"is_safe": 0.99},
        conformal_sets={"is_safe": ["True"]},
        probabilities={"is_safe": {"True": 0.99, "False": 0.01}},
        latency_ms=1.0,
        is_ambiguous=False,
        signing_key=signing_key,
        action_id="act_exec_001",
    )
    receipt_hash = ledger.append(
        receipt,
        tenant_id="tenant_001",
        principal_id="worker_001",
        scope="fs:read",
    )
    assert receipt_hash
    receipt_digest = receipt.compute_digest()

    # 2. Record post-execution outcome chained to receipt digest
    outcome_hash = ledger.record_execution_outcome(
        action_id="act_exec_001",
        receipt_digest=receipt_digest,
        status="SUCCEEDED",
        result_payload={"bytes_read": 1024, "status": "ok"},
        tenant_id="tenant_001",
        principal_id="worker_001",
        scope="fs:read",
    )
    assert outcome_hash

    # 3. Verify ledger entries and hash chain
    entries = ledger.entries()
    assert len(entries) == 2
    assert entries[0]["event_type"] == "system1_decision"
    assert entries[1]["event_type"] == "execution_outcome"
    assert entries[1]["previous_hash"] == entries[0]["entry_hash"]
    assert entries[1]["payload"]["status"] == "SUCCEEDED"
    assert entries[1]["payload"]["prior_receipt_digest"] == receipt_digest
    assert entries[1]["payload"]["action_id"] == "act_exec_001"

    # 4. Verify overall cryptographic integrity
    assert ledger.verify_integrity() is True


def test_execution_outcome_chaining_failure(tmp_path):
    """ActionLedger chains post-execution FAILED status with error details to receipt."""
    db_path = str(tmp_path / "chain_fail_ledger.db")
    ledger = ActionLedger(path=db_path)

    signing_key = Ed25519PrivateKey.generate()
    receipt = create_decision_receipt(
        schema_name="ExecutionGate",
        schema_digest="0" * 64,
        prompt="Execute failing operation",
        values={"is_safe": True},
        confidences={"is_safe": 0.99},
        conformal_sets={"is_safe": ["True"]},
        probabilities={"is_safe": {"True": 0.99, "False": 0.01}},
        latency_ms=1.0,
        is_ambiguous=False,
        signing_key=signing_key,
        action_id="act_exec_fail_002",
    )
    ledger.append(
        receipt,
        tenant_id="tenant_001",
        principal_id="worker_001",
        scope="execution_outcome",
    )
    receipt_digest = receipt.compute_digest()

    outcome_hash = ledger.record_execution_outcome(
        action_id="act_exec_fail_002",
        receipt_digest=receipt_digest,
        status="FAILED",
        error_message="FileNotFoundError: /etc/fake_file does not exist",
        tenant_id="tenant_001",
        principal_id="worker_001",
    )
    assert outcome_hash

    entries = ledger.entries()
    assert len(entries) == 2
    outcome_entry = entries[1]
    assert outcome_entry["payload"]["status"] == "FAILED"
    assert "FileNotFoundError" in outcome_entry["payload"]["error_message"]
    assert outcome_entry["payload"]["prior_receipt_digest"] == receipt_digest
    assert ledger.verify_integrity() is True


def test_mcp_handle_call_chains_outcome_to_ledger(tmp_path):
    """SystemOneMCPProxy.handle_call automatically logs post-execution outcome to ActionLedger."""
    db_path = str(tmp_path / "mcp_chain_ledger.db")
    ledger = ActionLedger(path=db_path)
    hook = SystemOneGuardHook(ledger=ledger, min_confidence=0.50, alpha=0.10)
    proxy = SystemOneMCPProxy(guard=hook)

    def mock_executor(name: str, args: dict):
        return {"data": "file contents", "size": 13}

    req = {
        "jsonrpc": "2.0",
        "id": "mcp-exec-chain-1",
        "method": "tools/call",
        "params": {
            "name": "read_file",
            "arguments": {"target": "/Volumes/Storage/project/README.md", "mode": "read"},
        },
    }

    res = proxy.handle_call(
        req,
        mock_executor,
        context_prompt="Inspect read-only project documentation in README.md",
    )

    assert "result" in res

    # Verify ledger has both the decision receipt AND the execution outcome record
    entries = ledger.entries()
    assert len(entries) >= 2
    outcome_records = [r for r in entries if r["event_type"] == "execution_outcome"]
    assert len(outcome_records) >= 1
    latest_outcome = outcome_records[-1]
    assert latest_outcome["payload"]["status"] == "SUCCEEDED"
    assert latest_outcome["payload"]["prior_receipt_digest"]
    assert ledger.verify_integrity() is True
