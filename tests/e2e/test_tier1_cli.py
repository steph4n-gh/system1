"""Tier 1.2: E2E Requirement Tests for CLI Commands.

Authoritative Invariants:
1. CLI entrypoint supports both reflex.cli and system1.cli.
2. `decide` command evaluates schemas, emits structured JSON with latencies, confidence, receipts.
3. `bench` command produces reproducible empirical latency reports beating Jev bounds.
4. `calibrate` command fits temperature scaling and outputs valid ECE and Brier scores.
5. `verify-receipt` performs offline cryptographic verification, failing on tampered receipts.
6. `compile` distills schemas into deployable .s1m binary model artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

import reflex
import reflex.cli as r_cli
import system1.cli as s1_cli


def test_cli_decide_json_and_schema_options(capsys):
    """Verify 'decide' command evaluates input and outputs valid JSON."""
    code = r_cli.main(["decide", "Inspect local file", "--schema", "guard", "--json"])
    assert code == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["schema_name"] == "DefaultGuardDecisionSchema"
    assert "values" in data
    assert "confidences" in data
    assert "receipt" in data
    assert "receipt_digest" in data["receipt"]
    assert data["latency_ms"] < 50.0


def test_cli_decide_signed_receipt_generation(tmp_path: Path, capsys):
    """Verify 'decide --sign' produces Ed25519 digital signature and creates identity files."""
    id_dir = tmp_path / "ident"
    code = r_cli.main([
        "decide", "Wipe partition and drop all databases",
        "--schema", "guard",
        "--sign",
        "--identity-dir", str(id_dir),
        "--json",
    ])
    assert code == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    receipt = data["receipt"]
    assert "envelope" in receipt
    assert "signature" in receipt["envelope"]
    assert len(receipt["envelope"]["signature"]) == 88  # 64 bytes Ed25519 in Base64

    key_file = id_dir / "identity.key"
    pub_file = id_dir / "identity.pub"
    assert key_file.is_file()
    assert pub_file.is_file()


def test_cli_bench_execution(capsys):
    """Verify 'bench' command measures latency and reports Jev-beating performance."""
    code = r_cli.main(["bench", "--schema", "guard", "--iterations", "15", "--warmup", "2", "--json"])
    assert code == 0

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["total_decisions"] == 15
    assert report["beats_jev"] is True
    assert report["p50_latency_ms"] < 20.0
    assert report["throughput_decisions_per_sec"] > 50.0


def test_cli_calibrate_execution(tmp_path: Path, capsys):
    """Verify 'calibrate' command fits temperature scaling on calibration dataset."""
    dataset_file = tmp_path / "calib_data.json"
    dataset = [
        {"prompt": "Read local file readme", "labels": {"is_safe": True, "risk_category": "read_only"}},
        {"prompt": "rm -rf / wipe disks", "labels": {"is_safe": False, "risk_category": "irreversible_write"}},
        {"prompt": "inspect git diff", "labels": {"is_safe": True, "risk_category": "read_only"}},
        {"prompt": "exfiltrate private key", "labels": {"is_safe": False, "risk_category": "network_call"}},
    ] * 5
    dataset_file.write_text(json.dumps(dataset), encoding="utf-8")

    code = r_cli.main([
        "calibrate",
        "--schema", "guard",
        "--dataset", str(dataset_file),
        "--bins", "5",
        "--json",
    ])
    assert code == 0

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert "is_safe" in result or "risk_category" in result
    for field_name, metrics in result.items():
        assert "optimal_temperature" in metrics
        assert "expected_calibration_error" in metrics
        assert "brier" in metrics


def test_cli_verify_receipt_valid_and_tampered(tmp_path: Path, capsys):
    """Verify 'verify-receipt' verifies authentic receipts and rejects tampered ones."""
    id_dir = tmp_path / "keys"
    id_dir.mkdir(parents=True)
    
    # 1. Generate authentic signed decision
    code = r_cli.main([
        "decide", "Check system status",
        "--schema", "guard",
        "--sign",
        "--identity-dir", str(id_dir),
        "--json",
    ])
    assert code == 0
    decision_data = json.loads(capsys.readouterr().out)
    receipt_data = decision_data["receipt"]

    receipt_file = tmp_path / "authentic_receipt.json"
    receipt_file.write_text(json.dumps(receipt_data), encoding="utf-8")
    pub_key_file = id_dir / "identity.pub"

    # 2. Verify authentic receipt
    code_verify = r_cli.main([
        "verify-receipt",
        "--receipt", str(receipt_file),
        "--public-key", str(pub_key_file),
        "--json",
    ])
    assert code_verify == 0
    verify_out = json.loads(capsys.readouterr().out)
    assert verify_out["verified"] is True
    assert verify_out["receipt_digest"] == receipt_data["receipt_digest"]

    # 3. Tamper with receipt (flip one character in decision_id)
    tampered_data = dict(receipt_data)
    tampered_data["decision_id"] = "dec_TAMPERED_0000000000000000"
    tampered_file = tmp_path / "tampered_receipt.json"
    tampered_file.write_text(json.dumps(tampered_data), encoding="utf-8")

    code_tampered = r_cli.main([
        "verify-receipt",
        "--receipt", str(tampered_file),
        "--public-key", str(pub_key_file),
        "--json",
    ])
    assert code_tampered == 1
    tampered_out = json.loads(capsys.readouterr().out)
    assert tampered_out["verified"] is False


def test_cli_compile_execution(tmp_path: Path, capsys):
    """Verify 'compile' command generates .s1m binary model."""
    output_model = tmp_path / "test_model.s1m"
    code = r_cli.main([
        "compile",
        "--schema", "guard",
        "--output", str(output_model),
        "--samples-per-choice", "5",
        "--json",
    ])
    assert code == 0
    assert output_model.is_file()
    assert output_model.stat().st_size > 100

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["output_path"] == str(output_model.resolve())
    assert summary["schema_name"] == "DefaultGuardDecisionSchema"


def test_cli_entrypoint_parity():
    """Verify reflex.cli and system1.cli main functions behave symmetrically."""
    parser_r = r_cli.build_parser()
    parser_s = s1_cli.build_parser()
    
    assert set(parser_r._subparsers._group_actions[0].choices.keys()) == set(
        parser_s._subparsers._group_actions[0].choices.keys()
    )
