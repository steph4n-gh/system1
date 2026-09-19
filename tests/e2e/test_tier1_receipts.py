"""Tier 1.5: E2E Requirement Tests for Cryptographic Witness Receipts.

Authoritative Invariants:
1. Proof-carrying decisions generate Ed25519 digital signatures.
2. Receipts are cryptographically verifiable offline with zero dependencies on the runtime.
3. Any alteration to decision values, confidences, schemas, or latencies invalidates the receipt digest and signature.
4. RunWitnessEnvelope binds mutation intent, guard decisions, and truth ledger heads into a verifiable chain.
5. Key generation, serialization (PEM/raw), and fingerprinting conform to SHA-256/Ed25519 standards.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

import reflex
from reflex import (
    ActionLedger,
    DecisionResult,
    DecisionSchema,
    DecisionWitnessReceipt,
    DefaultGuardDecisionSchema,
    SystemOneEngine,
    RunWitnessEnvelope,
    canonical_bytes,
    canonical_json,
    compute_receipt_digest,
    create_decision_receipt,
    create_run_witness_envelope,
    load_private_key,
    load_public_key,
    public_key_fingerprint,
    save_keypair,
    save_private_key,
    save_public_key,
    sign_payload,
    verify_decision_witness_receipt,
    verify_payload,
    verify_run_witness_envelope,
)


def test_ed25519_signature_creation_and_offline_verification(triage_schema):
    """Verify signed decision receipt can be verified offline with Ed25519 public key."""
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()

    engine = SystemOneEngine(triage_schema, signing_key=priv)
    res = engine.decide("Check network status for host", record_receipt=True)
    receipt = res.receipt

    assert receipt is not None
    assert receipt.envelope is not None
    assert receipt.envelope.signature is not None
    assert len(receipt.envelope.signature) == 88  # Base64-encoded 64-byte Ed25519 signature

    # Offline verification with public key
    receipt_dict = receipt.to_dict()
    assert verify_decision_witness_receipt(receipt_dict, public_key=pub) is True


def test_tamper_evident_payload_detection(triage_schema):
    """Verify mutating any field in a signed receipt causes cryptographic verification to fail."""
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()

    engine = SystemOneEngine(triage_schema, signing_key=priv)
    res = engine.decide("Perform non-destructive telemetry check", record_receipt=True)
    original_dict = res.receipt.to_dict()

    # 1. Tamper with decision values
    tampered_values = copy.deepcopy(original_dict)
    tampered_values["values"]["is_safe"] = not tampered_values["values"]["is_safe"]
    assert verify_decision_witness_receipt(tampered_values, public_key=pub) is False

    # 2. Tamper with confidence score
    tampered_conf = copy.deepcopy(original_dict)
    first_field = next(iter(tampered_conf["confidences"].keys()))
    tampered_conf["confidences"][first_field] = 0.9999
    assert verify_decision_witness_receipt(tampered_conf, public_key=pub) is False

    # 3. Tamper with decision ID
    tampered_id = copy.deepcopy(original_dict)
    tampered_id["decision_id"] = "dec_FORGED_0000000000000000"
    assert verify_decision_witness_receipt(tampered_id, public_key=pub) is False

    # 4. Tamper with signature
    tampered_sig = copy.deepcopy(original_dict)
    tampered_sig["envelope"]["signature"] = "A" * 88
    assert verify_decision_witness_receipt(tampered_sig, public_key=pub) is False


def test_canonical_json_and_digest_determinism():
    """Verify canonical serialization enforces strict sorting, compact formatting, and deterministic digests."""
    data = {
        "zeta": 1,
        "alpha": "test",
        "nested": {"gamma": [3, 2, 1], "beta": True},
    }
    c_str1 = canonical_json(data)
    c_str2 = canonical_json(data)
    assert c_str1 == c_str2
    assert ' ' not in c_str1  # strictly compact separators (',', ':')

    c_bytes = canonical_bytes(data)
    assert isinstance(c_bytes, bytes)
    assert c_bytes == c_str1.encode("utf-8")

    digest1 = compute_receipt_digest(data)
    digest2 = compute_receipt_digest(data)
    assert digest1 == digest2
    assert len(digest1) == 64


def test_run_witness_envelope_binding():
    """Verify RunWitnessEnvelope seals lifecycle artifacts and signer identity."""
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    pub_fp = public_key_fingerprint(priv)

    envelope = create_run_witness_envelope(
        mutation_intent={"action": "update_cache"},
        guard_receipt={"allowed": True, "risk": "READ_ONLY"},
        effect_observation={"status": "success"},
        readback_observation={"status": "confirmed"},
        truth_ledger_head="0" * 64,
        artifact_checksums={"bundle": "a" * 64},
        closure_receipt_fingerprint="f" * 64,
        signing_key=priv,
    )

    assert envelope.signer_fingerprint == pub_fp
    assert envelope.signature is not None
    is_valid, msg = verify_run_witness_envelope(envelope)
    assert is_valid is True
    assert msg == "valid"


def test_keypair_persistence_and_loading(tmp_path: Path):
    """Verify key saving, PEM serialization, loading, and cross-signing."""
    original_priv = Ed25519PrivateKey.generate()
    priv_file, pub_file = save_keypair(original_priv, tmp_path)

    loaded_priv = load_private_key(priv_file)
    loaded_pub = load_public_key(pub_file)

    message = {"audit_id": "audit_123", "verified": True}
    sig = sign_payload(message, loaded_priv)
    assert isinstance(sig, str)
    assert len(sig) == 88

    assert verify_payload(message, sig, loaded_pub) is True
    assert verify_payload(message, sig, original_priv.public_key()) is True


def test_truth_ledger_head_binding(temp_ledger, triage_schema):
    """Verify decision receipts record the tip of the SQLite ledger."""
    priv = Ed25519PrivateKey.generate()
    engine = SystemOneEngine(triage_schema, signing_key=priv, ledger=temp_ledger)

    res1 = engine.decide("First transaction prompt", record_receipt=True)
    head1 = temp_ledger.head_hash()
    assert res1.receipt.truth_ledger_head == "0" * 64  # Prior to first transaction ledger was at genesis
    assert res1.receipt.ledger_record_id is not None

    res2 = engine.decide("Second transaction prompt", record_receipt=True)
    # The second decision's truth_ledger_head should bind to the previous entry
    assert res2.receipt.truth_ledger_head == head1
    assert temp_ledger.verify_integrity() is True
