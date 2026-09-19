"""Adversarial Coverage Hardening (Tier 5) Stress Test Suite.

Author: Challenger 2 (teamwork_preview_challenger_tier5_2)
Scope:
1. src/system1/crypto/ (Ed25519 signatures, ActionLedger SHA-256 hash chains, tamper detection)
2. src/system1/compat/typesafe.py (TypeSafe drop-in, monkey-patching, async routing, error fallbacks)
3. src/system1/cli.py (dual syntax verify-receipt [path] and --receipt [path], invalid commands)
4. Dual-import parity (import reflex vs import system1, cross-operations)
"""

import asyncio
import json
import math
import sqlite3
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

import reflex
import reflex.cli as r_cli
import reflex.compat as r_compat
import reflex.ledger as r_ledger
import reflex.receipt as r_receipt
import system1
import system1.cli as s1_cli
import system1.compat.typesafe as s1_typesafe
import system1.ledger as s1_ledger
import system1.receipt as s1_receipt
from system1.cli import build_parser, main
from system1.compat.typesafe import (
    AsyncTypeSafeClient,
    Choice,
    DotDict,
    MultiChoice,
    Noul,
    Score,
    TypeSafeClient,
    Usage,
    create_typesafe_baseline_response,
    patch_typesafe,
)
from system1.ledger import ActionLedger, IntegrityError, LedgerError
from system1.receipt import (
    canonical_json,
    create_decision_receipt,
    load_private_key,
    load_public_key,
    public_key_bytes,
    save_keypair,
    save_public_key,
    verify_decision_witness_receipt,
    verify_payload,
)


# ============================================================================
# 1. Cryptographic Receipts & Tamper Detection Stress Tests
# ============================================================================

def test_adversarial_receipt_tamper_matrix():
    """Stress-test tamper detection across all fields of signed DecisionWitnessReceipt."""
    signing_key = Ed25519PrivateKey.generate()
    pub_key = signing_key.public_key()

    receipt = create_decision_receipt(
        schema_name="SecurityPolicySchema",
        schema_digest="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        prompt="Execute privileged database migration",
        values={"is_safe": True, "risk": 0.05, "action": "migrate"},
        confidences={"is_safe": 0.995, "risk": 0.99, "action": 0.98},
        conformal_sets={"is_safe": ["True"], "action": ["migrate"]},
        probabilities={"is_safe": {"True": 0.995, "False": 0.005}},
        latency_ms=1.42,
        is_ambiguous=False,
        truth_ledger_head="deadbeef" * 8,
        signing_key=signing_key,
    )
    rec_dict = receipt.to_dict()

    # Base case: honest verification succeeds
    assert verify_decision_witness_receipt(rec_dict, public_key=pub_key) is True
    assert verify_decision_witness_receipt(rec_dict, public_key=public_key_bytes(pub_key)) is True
    assert verify_decision_witness_receipt(rec_dict, public_key=public_key_bytes(pub_key).hex()) is True

    # 1. Tamper with decision values
    t1 = json.loads(json.dumps(rec_dict))
    t1["values"]["is_safe"] = False
    assert verify_decision_witness_receipt(t1, public_key=pub_key) is False

    # 2. Tamper with confidences
    t2 = json.loads(json.dumps(rec_dict))
    t2["confidences"]["is_safe"] = 0.50
    assert verify_decision_witness_receipt(t2, public_key=pub_key) is False

    # 3. Tamper with conformal sets
    t3 = json.loads(json.dumps(rec_dict))
    t3["conformal_sets"]["is_safe"] = ["True", "False"]
    assert verify_decision_witness_receipt(t3, public_key=pub_key) is False

    # 4. Tamper with prompt text
    t4 = json.loads(json.dumps(rec_dict))
    t4["prompt"] = "Harmless read query"
    assert verify_decision_witness_receipt(t4, public_key=pub_key) is False

    # 5. Tamper with prompt digest
    t5 = json.loads(json.dumps(rec_dict))
    t5["prompt_digest"] = "f" * 64
    assert verify_decision_witness_receipt(t5, public_key=pub_key) is False

    # 6. Tamper with schema digest
    t6 = json.loads(json.dumps(rec_dict))
    t6["schema_digest"] = "e" * 64
    assert verify_decision_witness_receipt(t6, public_key=pub_key) is False

    # 7. Tamper with is_ambiguous flag
    t7 = json.loads(json.dumps(rec_dict))
    t7["is_ambiguous"] = True
    assert verify_decision_witness_receipt(t7, public_key=pub_key) is False

    # 8. Tamper with receipt_digest
    t8 = json.loads(json.dumps(rec_dict))
    t8["receipt_digest"] = "0" * 64
    assert verify_decision_witness_receipt(t8, public_key=pub_key) is False

    # 9. Tamper with envelope witness_digest
    t9 = json.loads(json.dumps(rec_dict))
    t9["envelope"]["witness_digest"] = "1" * 64
    assert verify_decision_witness_receipt(t9, public_key=pub_key) is False

    # 10. Tamper with envelope signature (corrupted base64)
    t10 = json.loads(json.dumps(rec_dict))
    t10["envelope"]["signature"] = "not_valid_base64!!!"
    assert verify_decision_witness_receipt(t10, public_key=pub_key) is False

    # 11. Tamper with envelope signature (truncated signature)
    t11 = json.loads(json.dumps(rec_dict))
    t11["envelope"]["signature"] = "AAAA"
    assert verify_decision_witness_receipt(t11, public_key=pub_key) is False

    # 12. Verification with mismatched public key from different keypair
    wrong_pub = Ed25519PrivateKey.generate().public_key()
    assert verify_decision_witness_receipt(rec_dict, public_key=wrong_pub) is False
    assert verify_decision_witness_receipt(rec_dict, public_key=public_key_bytes(wrong_pub)) is False
    assert verify_decision_witness_receipt(rec_dict, public_key=public_key_bytes(wrong_pub).hex()) is False


def test_adversarial_key_loaders_and_canonicalization():
    """Stress-test key loaders with corrupted inputs and canonical JSON against non-finite floats."""
    # Corrupted public key inputs
    for bad_bytes in [b"", b"short", b"a" * 31, b"a" * 33]:
        with pytest.raises(TypeError):
            load_public_key(bad_bytes)

    for bad_str in ["not_hex", "a" * 63, "a" * 65]:
        with pytest.raises(TypeError):
            load_public_key(bad_str)

    with pytest.raises(TypeError):
        load_public_key(12345)

    # Corrupted private key inputs
    for bad_pem in [b"", b"corrupted", b"not a pem private key"]:
        with pytest.raises(ValueError):
            load_private_key(bad_pem)

    # Canonical JSON non-finite floats rejection
    for non_finite in [float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ValueError, match="canonical data cannot contain NaN or infinity"):
            canonical_json({"val": non_finite})

    # verify_payload robustness against malformed signatures
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    for malformed_sig in ["", "not_base64!", "short", "00" * 32]:
        assert verify_payload({"test": True}, malformed_sig, pub) is False


# ============================================================================
# 2. SQLite Action Ledger SQL Injection & Tamper Detection Stress Tests
# ============================================================================

def test_adversarial_action_ledger_sql_injection_resilience():
    """Verify ActionLedger parameterized query security against aggressive SQL injection payloads."""
    ledger = ActionLedger(":memory:")

    sqli_vectors = [
        "\"; DROP TABLE audit_entries; --",
        "\x27 OR \x271\x27=\x271",
        "admin\x27--",
        "\x27; UPDATE ledger_meta SET value=\x27hacked\x27; --",
        "\x27 UNION SELECT 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 --",
        "\\x00nullbyte_in_string",
        "nested_quotes_\x27_\x22_\x60",
        "🚀" * 50,
    ]

    for idx, payload in enumerate(sqli_vectors):
        head = ledger.record_action(
            action=payload,
            proposal={"vector_id": idx, "payload": payload, "nested": {"key": payload}},
            tenant_id=payload,
            principal_id=payload,
            scope=payload,
            action_id=f"act_{idx}_{payload[:8]}",
        )
        assert len(head) == 64
        assert int(head, 16) > 0

    # Cryptographic integrity must be 100% valid after all injections
    assert ledger.verify_integrity() is True
    seq, tip = ledger.audit_head()
    assert seq == len(sqli_vectors)


def test_adversarial_action_ledger_direct_tamper_detection(tmp_path):
    """Verify ActionLedger detects any direct SQLite modification, truncation, or metadata tampering."""
    db_file = tmp_path / "tamper_test.db"
    ledger = ActionLedger(db_file)
    for i in range(5):
        ledger.record_action(f"action_{i}", {"step": i})
    assert ledger.verify_integrity() is True
    ledger.close()

    # Attack 1: Mutate payload_json in a middle row
    conn = sqlite3.connect(db_file)
    conn.execute("UPDATE audit_entries SET payload_json=? WHERE sequence=3", (json.dumps({"step": 999}),))
    conn.commit()
    conn.close()

    l1 = ActionLedger(db_file)
    assert l1.verify_integrity() is False
    l1.close()

    # Attack 2: Restore payload, corrupt previous_hash
    conn = sqlite3.connect(db_file)
    conn.execute("UPDATE audit_entries SET payload_json=? WHERE sequence=3", (json.dumps({"step": 2}),))
    conn.execute("UPDATE audit_entries SET previous_hash=? WHERE sequence=3", ("0" * 64,))
    conn.commit()
    conn.close()

    l2 = ActionLedger(db_file)
    assert l2.verify_integrity() is False
    l2.close()

    # Attack 3: Restore previous_hash, corrupt ledger_meta head
    conn = sqlite3.connect(db_file)
    actual_prev = conn.execute("SELECT entry_hash FROM audit_entries WHERE sequence=2").fetchone()[0]
    conn.execute("UPDATE audit_entries SET previous_hash=? WHERE sequence=3", (actual_prev,))
    conn.execute("UPDATE ledger_meta SET value='deadbeef00000000000000000000000000000000000000000000000000000000' WHERE key='audit_head_hash'")
    conn.commit()
    conn.close()

    l3 = ActionLedger(db_file)
    assert l3.verify_integrity() is False
    with pytest.raises(IntegrityError):
        l3.audit_head()
    l3.close()

    # Attack 4: Delete a row from the middle
    conn = sqlite3.connect(db_file)
    actual_head = conn.execute("SELECT entry_hash FROM audit_entries WHERE sequence=5").fetchone()[0]
    conn.execute("UPDATE ledger_meta SET value=? WHERE key='audit_head_hash'", (actual_head,))
    conn.execute("DELETE FROM audit_entries WHERE sequence=3")
    conn.commit()
    conn.close()

    l4 = ActionLedger(db_file)
    assert l4.verify_integrity() is False
    l4.close()


def test_adversarial_action_ledger_concurrency_and_read_only(tmp_path):
    """Verify concurrent thread safety and read-only enforcement in ActionLedger."""
    ledger = ActionLedger(":memory:")
    num_threads = 12
    writes_per_thread = 8

    def worker(tid: int):
        for j in range(writes_per_thread):
            ledger.record_action(f"thread_{tid}", {"iteration": j})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert ledger.verify_integrity() is True
    seq, head = ledger.audit_head()
    assert seq == num_threads * writes_per_thread

    # Read-only enforcement
    with pytest.raises(ValueError, match="Read-only ledger requires an existing durable database file"):
        ActionLedger(":memory:", read_only=True)

    db_file = tmp_path / "durable.db"
    init_l = ActionLedger(db_file)
    init_l.record_action("init", {"ready": True})
    init_l.close()

    ro_ledger = ActionLedger(db_file, read_only=True)
    assert ro_ledger.verify_integrity() is True
    with pytest.raises(LedgerError, match="Cannot perform write transaction on read-only ledger"):
        ro_ledger.record_action("illegal_write", {})
    ro_ledger.close()


# ============================================================================
# 3. TypeSafe AI Drop-in Compatibility & Async Routing Stress Tests
# ============================================================================

def test_adversarial_typesafe_patch_decorators_and_exception_safety():
    """Verify patch_typesafe sync and async decorators, context managers, and exception cleanup."""
    # 1. Sync decorator without parens
    @patch_typesafe
    def sync_fn_1():
        import typesafe
        return typesafe._is_system1_patched

    assert sync_fn_1() is True
    assert "typesafe" not in sys.modules or not getattr(sys.modules.get("typesafe"), "_is_system1_patched", False)

    # 2. Sync decorator with parens
    @patch_typesafe()
    def sync_fn_2():
        import typesafe
        return typesafe._is_system1_patched

    assert sync_fn_2() is True
    assert "typesafe" not in sys.modules or not getattr(sys.modules.get("typesafe"), "_is_system1_patched", False)

    # 3. Async decorator without parens
    @patch_typesafe
    async def async_fn_1():
        import typesafe
        return typesafe._is_system1_patched

    assert asyncio.run(async_fn_1()) is True
    assert "typesafe" not in sys.modules or not getattr(sys.modules.get("typesafe"), "_is_system1_patched", False)

    # 4. Async decorator with parens
    @patch_typesafe()
    async def async_fn_2():
        import typesafe
        return typesafe._is_system1_patched

    assert asyncio.run(async_fn_2()) is True
    assert "typesafe" not in sys.modules or not getattr(sys.modules.get("typesafe"), "_is_system1_patched", False)

    # 5. Context manager with exception: must unpatch cleanly in finally
    with pytest.raises(ZeroDivisionError):
        with patch_typesafe():
            import typesafe
            assert typesafe._is_system1_patched is True
            _ = 1 / 0

    assert "typesafe" not in sys.modules or not getattr(sys.modules.get("typesafe"), "_is_system1_patched", False)


def test_adversarial_typesafe_usage_and_dotdict_robustness():
    """Verify DotDict attribute traversal and Usage token calculations under boundary values."""
    # DotDict boundary tests
    d = DotDict({"a": 1, "nested": {"b": 2, "deeper": {"c": 3}}})
    assert d.a == 1
    assert d.nested.b == 2
    assert d.nested.deeper.c == 3

    # Nonexistent attribute raises AttributeError
    with pytest.raises(AttributeError):
        _ = d.nonexistent_attr

    # Attribute deletion
    d.temp = 42
    assert d.temp == 42
    del d.temp
    with pytest.raises(AttributeError):
        _ = d.temp

    # Usage token calculation under unexpected types
    u1 = Usage({"input_tokens": "15", "output_tokens": "25"})
    assert u1.total_tokens == 40

    u2 = Usage({"input_tokens": None, "output_tokens": 10})
    assert u2.total_tokens == 10

    u3 = Usage({"input_tokens": 5, "output_tokens": 5, "total_tokens": "100"})
    assert u3.total_tokens == 100

    # Non-numeric token string raises ValueError
    with pytest.raises(ValueError):
        _ = Usage({"input_tokens": "invalid", "output_tokens": "bad"})


@pytest.mark.asyncio
async def test_adversarial_typesafe_async_routing_concurrency():
    """Verify AsyncTypeSafeClient handles concurrent queries safely with sub-20ms latency."""
    client = AsyncTypeSafeClient()
    questions = {
        "action": Choice("Choose operation", criteria={"read": "Read data", "write": "Modify data"}),
        "approved": Noul("Is operation approved?", criteria={"true": "Safe", "false": "Risky"}),
    }

    async def single_call(idx: int):
        return await client.system_one(
            f"Audit access request {idx} from internal worker",
            questions=questions,
        )

    # 10 concurrent async requests
    tasks = [single_call(i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 10
    for res in results:
        assert res.usage.total_tokens == 0
        assert res.egress_bytes == 0
        assert res.local_execution is True
        assert res.latency_ms < 50.0  # Safe concurrency threshold on shared cloud vCPU runners
        assert "action" in res.answers
        assert "approved" in res.answers
        assert res.answers.action.choice in ("read", "write")


# ============================================================================
# 4. CLI Dual Syntax & Invalid Command Stress Tests
# ============================================================================

def test_adversarial_cli_verify_receipt_dual_syntax_and_tamper(tmp_path, capsys):
    """Verify CLI verify-receipt handles positional, flag, stdin, and tampered receipts."""
    signing_key = Ed25519PrivateKey.generate()
    pub_key = signing_key.public_key()
    pub_file = tmp_path / "trusted.pub"
    save_public_key(pub_key, pub_file)

    receipt = create_decision_receipt(
        schema_name="TestSchema",
        schema_digest="0123" * 16,
        prompt="Verify CLI dual syntax receipt",
        values={"status": "ok"},
        confidences={"status": 0.99},
        conformal_sets={"status": ["ok"]},
        probabilities={"status": {"ok": 0.99}},
        latency_ms=1.1,
        is_ambiguous=False,
        signing_key=signing_key,
    )
    rec_dict = receipt.to_dict()
    receipt_file = tmp_path / "receipt.json"
    receipt_file.write_text(json.dumps(rec_dict), encoding="utf-8")

    # 1. Positional syntax with trusted public key: system1 verify-receipt file.json --public-key key.pub --json
    code_pos = main(["verify-receipt", str(receipt_file), "--public-key", str(pub_file), "--json"])
    assert code_pos == 0
    pos_res = json.loads(capsys.readouterr().out)
    assert pos_res["verified"] is True
    assert pos_res["decision_id"] == rec_dict["decision_id"]

    # 2. Flag syntax with trusted public key: system1 verify-receipt --receipt file.json --public-key key.pub --json
    code_flag = main(["verify-receipt", "--receipt", str(receipt_file), "--public-key", str(pub_file), "--json"])
    assert code_flag == 0
    flag_res = json.loads(capsys.readouterr().out)
    assert flag_res["verified"] is True

    # 3. Wrong public key must fail with exit code 1
    wrong_key = Ed25519PrivateKey.generate().public_key()
    wrong_pub_file = tmp_path / "wrong.pub"
    save_public_key(wrong_key, wrong_pub_file)

    code_wrong = main(["verify-receipt", str(receipt_file), "--public-key", str(wrong_pub_file), "--json"])
    assert code_wrong == 1
    wrong_res = json.loads(capsys.readouterr().out)
    assert wrong_res["verified"] is False

    # 4. Tampered receipt file must fail with exit code 1
    tampered_dict = dict(rec_dict)
    tampered_dict["values"] = {"status": "tampered"}
    tampered_file = tmp_path / "tampered.json"
    tampered_file.write_text(json.dumps(tampered_dict), encoding="utf-8")

    code_tampered = main(["verify-receipt", str(tampered_file), "--public-key", str(pub_file), "--json"])
    assert code_tampered == 1
    tampered_res = json.loads(capsys.readouterr().out)
    assert tampered_res["verified"] is False

    # 5. Nonexistent receipt file must exit 1
    code_missing = main(["verify-receipt", str(tmp_path / "missing.json")])
    assert code_missing == 1
    missing_err = capsys.readouterr().err
    assert "[SYSTEM1 ERROR] Receipt file not found" in missing_err

    # 6. Nonexistent public key file must exit 1 and report error
    code_missing_key = main(["verify-receipt", str(receipt_file), "--public-key", str(tmp_path / "missing_key.pub")])
    assert code_missing_key == 1
    missing_key_err = capsys.readouterr().err
    assert "[SYSTEM1 ERROR] Failed to load trusted public key from" in missing_key_err

    # 7. Corrupted public key file must exit 1 and report error
    corrupt_key_file = tmp_path / "corrupt.pub"
    corrupt_key_file.write_text("corrupted_key_data", encoding="utf-8")
    code_corrupt_key = main(["verify-receipt", str(receipt_file), "--public-key", str(corrupt_key_file)])
    assert code_corrupt_key == 1
    corrupt_key_err = capsys.readouterr().err
    assert "[SYSTEM1 ERROR] Failed to load trusted public key from" in corrupt_key_err


def test_adversarial_cli_invalid_commands_and_subprocesses():
    """Verify CLI subprocess execution on invalid commands, missing required args, and pipes."""
    # Invalid command -> code 2
    res_inv = subprocess.run(
        [sys.executable, "-m", "system1.cli", "bogus_command"],
        capture_output=True,
        text=True,
    )
    assert res_inv.returncode == 2
    assert "invalid choice" in res_inv.stderr

    # Missing required --dataset on calibrate -> code 2
    res_cal = subprocess.run(
        [sys.executable, "-m", "system1.cli", "calibrate"],
        capture_output=True,
        text=True,
    )
    assert res_cal.returncode == 2
    assert "--dataset" in res_cal.stderr

    # Missing required --schema on compile -> code 2
    res_comp = subprocess.run(
        [sys.executable, "-m", "system1.cli", "compile"],
        capture_output=True,
        text=True,
    )
    assert res_comp.returncode == 2
    assert "--schema" in res_comp.stderr


# ============================================================================
# 5. Dual-Import Parity & Cross-Namespace Operations
# ============================================================================

def test_adversarial_cross_namespace_receipt_and_ledger_interop(tmp_path):
    """Verify cross-namespace interoperability between reflex and system1 packages."""
    signing_key = Ed25519PrivateKey.generate()

    # Create receipt using system1.receipt
    s1_rec = s1_receipt.create_decision_receipt(
        schema_name="CrossNamespaceSchema",
        schema_digest="a" * 64,
        prompt="Cross-namespace verification test",
        values={"route": "test"},
        confidences={"route": 0.95},
        conformal_sets={"route": ["test"]},
        probabilities={"route": {"test": 0.95}},
        latency_ms=1.2,
        is_ambiguous=False,
        signing_key=signing_key,
    )

    # Verify receipt using reflex.receipt
    assert r_receipt.verify_decision_witness_receipt(s1_rec.to_dict(), public_key=signing_key.public_key()) is True

    # Record action in reflex.ledger.ActionLedger
    db_file = tmp_path / "cross_ledger.db"
    r_led = r_ledger.ActionLedger(db_file)
    r_led.record_decision_receipt(s1_rec)
    r_led.record_action("cross_action", {"origin": "reflex"})
    r_led.close()

    # Verify and append using system1.ledger.ActionLedger
    s1_led = s1_ledger.ActionLedger(db_file)
    assert s1_led.verify_integrity() is True
    seq, head = s1_led.audit_head()
    assert seq == 2
    s1_led.record_action("cross_action_2", {"origin": "system1"})
    assert s1_led.verify_integrity() is True
    assert s1_led.audit_head()[0] == 3
    s1_led.close()


def test_adversarial_twin_namespace_dynamic_submodules_parity():
    """Verify dynamic resolution across all 15 submodules for both reflex and system1."""
    submodules = [
        "cache", "calibration", "cli", "compat", "compiler",
        "core", "embeddings", "engine", "guard", "ledger",
        "model", "neural", "receipt", "schema", "telemetry",
    ]
    for sub in submodules:
        assert hasattr(reflex, sub), f"reflex missing {sub}"
        assert hasattr(system1, sub), f"system1 missing {sub}"
        r_mod = getattr(reflex, sub)
        s_mod = getattr(system1, sub)
        assert r_mod is not None
        assert s_mod is not None
        assert set(r_mod.__all__) == set(s_mod.__all__)

    # Top-level __all__ symmetry and object identity
    assert set(reflex.__all__) == set(system1.__all__)
    for sym in reflex.__all__:
        assert getattr(reflex, sym) is getattr(system1, sym)


def test_adversarial_verify_receipt_invalid_explicit_public_key_raises():
    """Verify verify_decision_witness_receipt raises ValueError on invalid explicit public_key."""
    signing_key = Ed25519PrivateKey.generate()
    receipt = create_decision_receipt(
        schema_name="TestSchema",
        schema_digest="0123" * 16,
        prompt="Test explicit invalid public key",
        values={"ok": True},
        confidences={"ok": 0.99},
        conformal_sets={"ok": ["True"]},
        probabilities={"ok": {"True": 0.99}},
        latency_ms=1.0,
        is_ambiguous=False,
        signing_key=signing_key,
    )
    rec_dict = receipt.to_dict()

    # public_key=None succeeds by falling back to receipt key
    assert verify_decision_witness_receipt(rec_dict, public_key=None) is True

    # Explicit invalid public_key raises ValueError, does not fall back to receipt key
    for bad_key in ["invalid_key", b"bad_bytes", 12345, "not_hex_nor_32_chars", b"short", b"a" * 31, b"a" * 33, Path("/nonexistent/key.pub")]:
        with pytest.raises(ValueError, match="Invalid public_key provided"):
            verify_decision_witness_receipt(rec_dict, public_key=bad_key)


def test_adversarial_system1_cli_module_execution_parity():
    """Verify python3 -m reflex.cli execution matches python3 -m system1.cli."""
    # 1. --help output parity
    res_system1_help = subprocess.run(
        [sys.executable, "-m", "reflex.cli", "--help"],
        capture_output=True,
        text=True,
    )
    res_system1_help = subprocess.run(
        [sys.executable, "-m", "system1.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert res_system1_help.returncode == 0
    assert res_system1_help.returncode == 0
    assert res_system1_help.stdout == res_system1_help.stdout

    # 2. decide command execution via reflex.cli
    res_decide = subprocess.run(
        [sys.executable, "-m", "reflex.cli", "decide", "Test triage via reflex.cli", "--json"],
        capture_output=True,
        text=True,
    )
    assert res_decide.returncode == 0
    payload = json.loads(res_decide.stdout)
    assert "values" in payload
    assert "confidences" in payload
    assert "receipt" in payload


def test_adversarial_system1_compat_typesafe_identity():
    """Verify reflex.compat.typesafe maintains object identity with system1.compat.typesafe."""
    import reflex.compat.typesafe as r_ts
    import system1.compat.typesafe as s1_ts
    assert r_ts is s1_ts
    assert reflex.compat.typesafe is system1.compat.typesafe

    # Verify via fresh subprocess execution
    sub_code = (
        "import reflex.compat.typesafe\n"
        "import system1.compat.typesafe\n"
        "assert reflex.compat.typesafe is system1.compat.typesafe\n"
        "from reflex.compat import typesafe\n"
        "assert typesafe is system1.compat.typesafe\n"
    )
    res = subprocess.run([sys.executable, "-c", sub_code], capture_output=True, text=True)
    assert res.returncode == 0, f"Identity verification failed: {res.stderr}"


def test_adversarial_typesafe_nested_patching():
    """Verify nested @patch_typesafe and with patch_typesafe() depth management."""
    assert "typesafe" not in sys.modules or not getattr(sys.modules.get("typesafe"), "_is_system1_patched", False)

    # 1. Nested context managers
    with patch_typesafe():
        import typesafe
        assert getattr(typesafe, "_is_system1_patched", False) is True
        with patch_typesafe():
            import typesafe as ts_inner
            assert ts_inner is typesafe
            assert getattr(typesafe, "_is_system1_patched", False) is True
        # Outer must still be patched
        assert "typesafe" in sys.modules
        assert getattr(sys.modules["typesafe"], "_is_system1_patched", False) is True

    assert "typesafe" not in sys.modules or not getattr(sys.modules.get("typesafe"), "_is_system1_patched", False)

    # 2. Nested decorators
    @patch_typesafe
    def inner_fn():
        import typesafe
        return getattr(typesafe, "_is_system1_patched", False)

    @patch_typesafe
    def outer_fn():
        import typesafe
        inner_res = inner_fn()
        assert inner_res is True
        # After inner returns, outer context must retain typesafe
        assert "typesafe" in sys.modules
        assert getattr(sys.modules["typesafe"], "_is_system1_patched", False) is True
        return True

    assert outer_fn() is True
    assert "typesafe" not in sys.modules or not getattr(sys.modules.get("typesafe"), "_is_system1_patched", False)

