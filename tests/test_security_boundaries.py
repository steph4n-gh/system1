"""Regression coverage for the September 2026 receipt, ledger and loader audit."""

import io
import json
import struct
import zipfile
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from system1 import compiler
from system1.compiler import CompiledHeadWeights, CompiledSystemOneModel
from system1.core import ChoiceField, DecisionSchema
from system1.engine import SystemOneEngine
from system1.grpc_server import SystemOneServiceServicer
from system1.ledger import ActionLedger, IntegrityError
from system1.receipt import (
    check_decision_receipt_integrity, create_decision_receipt,
    verify_decision_witness_receipt,
)


def receipt(key=None):
    return create_decision_receipt(
        schema_name="test", schema_digest="a" * 64, prompt="hello",
        values={"route": "a"}, confidences={"route": 1.0},
        conformal_sets={"route": ["a"]}, probabilities={"route": {"a": 1.0}},
        latency_ms=1, is_ambiguous=False, signing_key=key,
    )


@pytest.mark.parametrize("signed", [False, True])
def test_authentication_requires_independent_key(signed):
    key = Ed25519PrivateKey.generate()
    data = receipt(key if signed else None).to_dict()
    assert not verify_decision_witness_receipt(data)
    assert check_decision_receipt_integrity(data)
    assert verify_decision_witness_receipt(data, key.public_key()) is signed
    assert not verify_decision_witness_receipt(data, Ed25519PrivateKey.generate().public_key())
    data["values"]["route"] = "tampered"
    assert not check_decision_receipt_integrity(data)


@pytest.mark.parametrize("record_id", [None, "", "0" * 64, "f" * 64])
def test_signed_ledger_claim_cannot_be_changed_or_removed(record_id):
    key = Ed25519PrivateKey.generate()
    with ActionLedger() as ledger:
        original = receipt(key)
        bound = original.with_ledger_record(ledger.append(original), key)
        assert bound.digest == original.digest
        assert verify_decision_witness_receipt(bound.to_dict(), key.public_key())
        assert ledger.verify_receipt_record(bound, key.public_key())
        forged = bound.to_dict()
        forged["ledger_record_id"] = record_id
        assert not verify_decision_witness_receipt(forged, key.public_key())
        assert not ledger.verify_receipt_record(forged, key.public_key())


def test_ledger_lookup_rejects_borrowed_signed_record_and_preserves_outcomes():
    key = Ed25519PrivateKey.generate()
    with ActionLedger() as ledger:
        first, second = receipt(key), receipt(key)
        first_id = ledger.append(first)
        second_id = ledger.append(second)
        assert not ledger.verify_receipt_record(second.with_ledger_record(first_id, key), key.public_key())
        bound = second.with_ledger_record(second_id, key)
        assert ledger.verify_receipt_record(bound, key.public_key())
        ledger.record_execution_outcome(action_id=second.decision_id, receipt_digest=bound.digest, status="SUCCEEDED")
        assert ledger.verify_integrity(key.public_key())


def test_engine_binds_both_cold_and_cached_receipts():
    key = Ed25519PrivateKey.generate()
    schema = DecisionSchema(fields={"route": ChoiceField(["a", "b"])})
    with ActionLedger() as ledger:
        model = CompiledSystemOneModel(schema, {"route": CompiledHeadWeights(
            "route", "choice", np.zeros((2, 16), dtype=np.float32), np.array([10, -10], dtype=np.float32),
            options=("a", "b"), calibration_scores=(0.1,) * 100, score_method="lac",
        )}, dimension=16)
        engine = SystemOneEngine(schema, model=model, signing_key=key, ledger=ledger, strict_mode=True)
        for i in range(2):
            result = engine.decide("same request")
            assert result.is_cache_hit is bool(i)
            assert verify_decision_witness_receipt(result.receipt.to_dict(), key.public_key())
            assert ledger.verify_receipt_record(result.receipt, key.public_key())


def test_guard_rejects_borrowed_ledger_record_even_with_a_valid_signature(monkeypatch):
    from system1.guard import ActionProposal, DecisionOutcome, SystemOneGuardHook
    key = Ed25519PrivateKey.generate()
    with ActionLedger() as ledger:
        hook = SystemOneGuardHook(signing_key=key, ledger=ledger)
        first = hook.engine.decide("first")
        second = hook.engine.decide("second")
        forged = second.receipt.with_ledger_record(first.receipt.ledger_record_id, key)
        assert verify_decision_witness_receipt(forged.to_dict(), key.public_key())
        monkeypatch.setattr(hook.engine, "decide", lambda *a, **k: replace(second, receipt=forged))
        proposal = ActionProposal.create(tenant_id="test", principal_id="test", scope="test", tool="read", arguments={}, purpose="test")
        result = hook.evaluate_proposal(proposal)
        assert result.outcome == DecisionOutcome.DENY
        assert "durably record" in result.reason


@pytest.mark.parametrize("mutation", ["UPDATE audit_entries SET payload_json='{}' WHERE sequence=1",
                                      "DELETE FROM audit_entries WHERE sequence=1"])
@pytest.mark.parametrize("operation", ["append", "action", "outcome"])
def test_every_write_rechecks_previously_verified_history(mutation, operation):
    with ActionLedger() as ledger:
        original = receipt()
        ledger.append(original)
        ledger.record_action("benign", {})  # populates the former prefix cache
        ledger._connection.execute(mutation)
        with pytest.raises(IntegrityError):
            if operation == "append":
                ledger.append(receipt())
            elif operation == "action":
                ledger.record_action("later", {})
            else:
                ledger.record_execution_outcome(action_id=original.decision_id, receipt_digest=original.digest, status="SUCCEEDED")


def test_changed_trust_key_revalidates_history():
    first, second = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    with ActionLedger() as ledger:
        ledger.append(receipt(first), trusted_public_key=first.public_key())
        ledger.record_action("benign", {}, trusted_public_key=first.public_key())
        with pytest.raises(IntegrityError):
            ledger.append(receipt(second), trusted_public_key=second.public_key())


class Context:
    code = None
    def set_code(self, code): self.code = code
    def set_details(self, details): self.details = details


def test_cli_and_grpc_require_a_trusted_key(tmp_path, capsys):
    from system1.cli import main
    key = Ed25519PrivateKey.generate()
    data = receipt(key).to_dict()
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(data))
    public_key = key.public_key().public_bytes_raw().hex()
    for supplied, valid in (("", False), (public_key, True)):
        args = ["verify-receipt", str(path), "--json"]
        if supplied:
            args += ["--public-key", supplied]
        assert main(args) == (0 if valid else 1)
        assert json.loads(capsys.readouterr().out)["verified"] is valid
        response = SystemOneServiceServicer().VerifyReceipt(
            {"receipt_json": path.read_bytes(), "public_key_hex": supplied}, Context(),
        )
        assert response.verified is valid


def test_grpc_registry_blocks_imports_files_json_and_cache_growth(tmp_path):
    schema = DecisionSchema(fields={"route": ChoiceField(["a", "b"])})
    supplied = {"custom": schema}
    server = SystemOneServiceServicer(schemas=supplied)
    supplied["late"] = schema  # mutation of the caller's registry is not registration
    file = tmp_path / "schema.json"
    file.write_text(json.dumps(schema.to_dict()))
    rejected = ["os.path", str(file), json.dumps(schema.to_dict()), "late"]
    rejected += [f"unknown_{i}" for i in range(100)]
    with patch("system1.cli._load_schema", side_effect=AssertionError("privileged loader called")):
        for name in rejected:
            for method in (server.Decide, server.Guard):
                ctx = Context()
                method({"prompt": "hello", "schema_name": name}, ctx)
                assert ctx.code is not None
        assert list(server._engines) == ["custom"]
        assert not server._guard_hooks
        for name in ("custom", "triage", "TRIAGE", "default", "GUARDRAIL", "guard"):
            ctx = Context()
            server.Decide({"prompt": "hello", "schema_name": name}, ctx)
            assert ctx.code is None
        assert set(server._engines) == {"custom", "triage", "guard"}


@pytest.fixture
def skill_bytes():
    schema = DecisionSchema(fields={"route": ChoiceField(["a", "b"])})
    model = CompiledSystemOneModel(schema, {"route": CompiledHeadWeights(
        "route", "choice", np.zeros((2, 16), dtype=np.float32), np.zeros(2, dtype=np.float32), options=("a", "b"),
    )}, dimension=16)
    return model.to_bytes(include_covariance=True)


def unpack(data):
    end = 8 + struct.unpack(">I", data[4:8])[0]
    return json.loads(data[8:end]), data[end:]


def repack(meta, payload):
    header = json.dumps(meta).encode()
    return compiler.MAGIC_HEADER + struct.pack(">I", len(header)) + header + payload


@pytest.mark.parametrize("change", ["dimension", "fields", "options", "projector", "hybrid", "hybrid_override", "runtime", "heads"])
def test_skill_metadata_rejected_before_array_or_projector_allocation(skill_bytes, change):
    meta, payload = unpack(skill_bytes)
    if change == "dimension": meta["dimension"] = 1 << 40
    elif change == "fields": meta["schema_dict"]["fields"] = {str(i): {} for i in range(65)}
    elif change == "options": meta["schema_dict"]["fields"]["route"]["options"] = [str(i) for i in range(1025)]
    elif change == "projector": meta["projector"]["dimension"] = 1 << 40
    elif change == "hybrid": meta["projector"] = {"type": "hybrid", "sparse_dim": 1, "dense_dim": 1 << 40}
    elif change == "hybrid_override": meta["projector"] = {"type": "hybrid", "sparse_dim": 12, "dense_dim": 4, "dimension": 1 << 40}
    elif change == "runtime":
        meta["dimension"] = 4096
        meta["projector"]["dimension"] = 4096
    else: meta["heads"] = {}
    with patch.object(np, "load", side_effect=AssertionError("allocated before validation")):
        with pytest.raises(ValueError):
            CompiledSystemOneModel.from_bytes(repack(meta, payload))


@pytest.mark.parametrize("change", ["huge_shape", "object", "covariance", "calibration", "duplicate", "expansion", "members"])
def test_npz_preflight_rejects_malicious_members_before_numpy_load(skill_bytes, change, monkeypatch):
    meta, payload = unpack(skill_bytes)
    with zipfile.ZipFile(io.BytesIO(payload)) as old:
        members = {name: old.read(name) for name in old.namelist()}
    if change in ("huge_shape", "object"):
        fake = io.BytesIO()
        np.lib.format.write_array_header_1_0(fake, {"descr": "|O" if change == "object" else "<f4",
                                                   "fortran_order": False, "shape": (2, 1 << 40)})
        members["route_w.npy"] = fake.getvalue()
    elif change in ("covariance", "calibration"):
        fake = io.BytesIO()
        np.save(fake, np.zeros((3, 4), dtype=np.float32))
        members["route_P.npy" if change == "covariance" else "route_calib_scores.npy"] = fake.getvalue()
    elif change == "expansion": monkeypatch.setattr(compiler, "MAX_SKILL_ARRAY_BYTES", 32)
    elif change == "members": members.update({f"extra{i}.npy": b"" for i in range(321)})
    new = io.BytesIO()
    with zipfile.ZipFile(new, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in members.items(): archive.writestr(name, value)
        if change == "duplicate":
            with pytest.warns(UserWarning): archive.writestr("route_w.npy", members["route_w.npy"])
    with patch.object(np, "load", side_effect=AssertionError("allocated before validation")):
        with pytest.raises(ValueError):
            CompiledSystemOneModel.from_bytes(repack(meta, new.getvalue()))


def test_file_header_ratio_budgets_and_legitimate_reload(skill_bytes, tmp_path, monkeypatch):
    loaded = CompiledSystemOneModel.from_bytes(skill_bytes)
    assert loaded.forward_single("hello").fields["route"].selected_value in ("a", "b")
    assert CompiledSystemOneModel.from_bytes(loaded.to_bytes()).dimension == 16
    path = tmp_path / "skill.s1m"
    path.write_bytes(skill_bytes)
    with monkeypatch.context() as m:
        m.setattr(compiler, "MAX_SKILL_BYTES", 32)
        with pytest.raises(ValueError, match="size limit"): CompiledSystemOneModel.load(path)
    with monkeypatch.context() as m:
        m.setattr(compiler, "MAX_SKILL_HEADER_BYTES", 16)
        with pytest.raises(ValueError, match="header"): CompiledSystemOneModel.from_bytes(skill_bytes)
    with monkeypatch.context() as m:
        m.setattr(compiler, "MAX_SKILL_COMPRESSION_RATIO", 1)
        with pytest.raises(ValueError, match="compression"): CompiledSystemOneModel.from_bytes(skill_bytes)
