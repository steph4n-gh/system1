"""Tests for System 1 Cryptographic Decision Receipts and RunWitnessEnvelope."""

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from system1.receipt import (
    create_decision_receipt,
    check_decision_receipt_integrity,
    public_key_bytes,
    verify_decision_witness_receipt,
)


def test_unsigned_decision_receipt():
    receipt = create_decision_receipt(
        schema_name="TestSchema",
        schema_digest="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        prompt="Sample test query",
        values={"action": "read"},
        confidences={"action": 0.98},
        conformal_sets={"action": ["read"]},
        probabilities={"action": {"read": 0.98, "write": 0.02}},
        latency_ms=3.45,
        is_ambiguous=False,
    )

    assert receipt.decision_id.startswith("dec_")
    assert receipt.values["action"] == "read"
    assert receipt.envelope is not None
    assert receipt.envelope.profile == "diagnostic_local"

    # Unsigned diagnostics have integrity, never authenticated provenance.
    assert check_decision_receipt_integrity(receipt.to_dict()) is True
    assert verify_decision_witness_receipt(receipt.to_dict()) is False


def test_signed_decision_receipt_and_verification():
    signing_key = Ed25519PrivateKey.generate()
    pub_key = signing_key.public_key()

    receipt = create_decision_receipt(
        schema_name="SecuritySchema",
        schema_digest="aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899",
        prompt="Delete production database records",
        values={"is_safe": False, "risk": 0.99},
        confidences={"is_safe": 0.999, "risk": 0.99},
        conformal_sets={"is_safe": ["False"]},
        probabilities={"is_safe": {"True": 0.001, "False": 0.999}},
        latency_ms=2.15,
        is_ambiguous=False,
        truth_ledger_head="fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
        signing_key=signing_key,
    )

    assert receipt.envelope is not None
    assert receipt.envelope.profile == "product_signed_v1"
    assert receipt.envelope.signature is not None
    assert receipt.envelope.signer_fingerprint is not None

    receipt_dict = receipt.to_dict()

    # 1. Independent offline verification succeeds with correct public key
    assert verify_decision_witness_receipt(receipt_dict, public_key=pub_key) is True

    # 2. Verification fails with wrong public key
    wrong_key = Ed25519PrivateKey.generate().public_key()
    assert verify_decision_witness_receipt(receipt_dict, public_key=wrong_key) is False

    # 3. Tampering with mutation intent / values is detected
    tampered_dict = dict(receipt_dict)
    tampered_envelope = dict(tampered_dict["envelope"])
    tampered_mutation = dict(tampered_envelope["mutation_intent"])
    tampered_mutation["values"] = {"is_safe": True, "risk": 0.01}
    tampered_envelope["mutation_intent"] = tampered_mutation
    tampered_dict["envelope"] = tampered_envelope

    assert verify_decision_witness_receipt(tampered_dict, public_key=pub_key) is False

    # 4. Tampering with signature is detected
    tampered_dict2 = dict(receipt_dict)
    tampered_envelope2 = dict(tampered_dict2["envelope"])
    tampered_envelope2["signature"] = "00" * 64
    tampered_dict2["envelope"] = tampered_envelope2

    assert verify_decision_witness_receipt(tampered_dict2, public_key=pub_key) is False

    # 5. Direct tampering with receipt values is rejected
    tampered_dict3 = dict(receipt_dict)
    tampered_dict3["values"] = {"is_safe": True, "risk": 0.01}
    assert verify_decision_witness_receipt(tampered_dict3, public_key=pub_key) is False

    # 6. Direct tampering with confidences is rejected
    tampered_dict4 = dict(receipt_dict)
    tampered_dict4["confidences"] = {"is_safe": 0.10, "risk": 0.10}
    assert verify_decision_witness_receipt(tampered_dict4, public_key=pub_key) is False

    # 7. Direct tampering with prompt is rejected
    tampered_dict5 = dict(receipt_dict)
    tampered_dict5["prompt"] = "Harmless read operation"
    assert verify_decision_witness_receipt(tampered_dict5, public_key=pub_key) is False

    # 8. Direct tampering with receipt_digest is rejected
    tampered_dict6 = dict(receipt_dict)
    tampered_dict6["receipt_digest"] = "0" * 64
    assert verify_decision_witness_receipt(tampered_dict6, public_key=pub_key) is False

    # 9. Verification succeeds when public_key is passed as hex string or raw bytes
    pub_hex = public_key_bytes(pub_key).hex()
    pub_raw_bytes = public_key_bytes(pub_key)
    assert verify_decision_witness_receipt(receipt_dict, public_key=pub_hex) is True
    assert verify_decision_witness_receipt(receipt_dict, public_key=pub_raw_bytes) is True
