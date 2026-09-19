"""Tests for System 1 CLI commands."""

import json
from pathlib import Path
import pytest

from system1.cli import (
    build_parser,
    handle_bench_command,
    handle_calibrate_command,
    handle_decide_command,
    handle_verify_receipt_command,
    main,
)


def test_cli_decide_command(capsys):
    parser = build_parser()
    args = parser.parse_args(["decide", "Read code file", "--json"])

    code = args.func(args)
    assert code == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert data["schema_name"] == "DefaultTriageSchema"
    assert "route" in data["values"]
    assert "is_safe" in data["values"]
    assert "risk_level" in data["values"]
    assert "receipt" in data
    assert data["latency_ms"] < 20.0


def test_cli_decide_human_output(capsys):
    parser = build_parser()
    args = parser.parse_args(["decide", "Execute bash rm -rf /"])

    code = args.func(args)
    assert code == 0

    captured = capsys.readouterr()
    assert "[SYSTEM1 DECISION]" in captured.out
    assert "Calibrated Confidence:" in captured.out
    assert "Receipt Digest:" in captured.out


def test_cli_bench_command(capsys):
    parser = build_parser()
    args = parser.parse_args(["bench", "--iterations", "15", "--warmup", "2", "--json"])

    code = args.func(args)
    assert code == 0

    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert report["total_decisions"] == 15
    assert report["p50_latency_ms"] < 15.0
    assert report["baseline_label"] == "Cloud SaaS API"
    assert report["speedup_factor"] > 5.0


def test_cli_calibrate_command(tmp_path, capsys):
    dataset_file = tmp_path / "calib.json"
    dataset_file.write_text(json.dumps([
        {"prompt": "Inspect repo files", "labels": {"route": "filesystem", "is_safe": True}},
        {"prompt": "Query customer database", "labels": {"route": "database", "is_safe": True}},
        {"prompt": "Wipe entire server disk", "labels": {"route": "system_shell", "is_safe": False}},
    ]))

    parser = build_parser()
    args = parser.parse_args(["calibrate", "--dataset", str(dataset_file), "--bins", "5", "--json"])

    code = args.func(args)
    assert code == 0

    captured = capsys.readouterr()
    metrics = json.loads(captured.out)

    assert "route" in metrics
    assert "is_safe" in metrics
    assert metrics["route"]["expected_calibration_error"] >= 0.0


def test_cli_verify_receipt_command(tmp_path, capsys):
    parser = build_parser()
    decide_args = parser.parse_args(["decide", "Audit access logs", "--sign", "--json"])
    decide_args.func(decide_args)
    captured = capsys.readouterr()
    receipt_data = json.loads(captured.out)["receipt"]

    receipt_file = tmp_path / "test_receipt.json"
    receipt_file.write_text(json.dumps(receipt_data))

    verify_args = parser.parse_args(["verify-receipt", "--receipt", str(receipt_file), "--json"])
    code = verify_args.func(verify_args)
    assert code == 0

    verify_out = capsys.readouterr()
    verify_res = json.loads(verify_out.out)
    assert verify_res["verified"] is True

    # Tampered receipt causes verification failure
    tampered_data = dict(receipt_data)
    tampered_data["values"] = {"route": "system_shell", "is_safe": True, "risk_level": 0.0}
    tampered_file = tmp_path / "tampered_receipt.json"
    tampered_file.write_text(json.dumps(tampered_data))

    tampered_args = parser.parse_args(["verify-receipt", "--receipt", str(tampered_file), "--json"])
    tampered_code = tampered_args.func(tampered_args)
    assert tampered_code == 1

    tampered_out = capsys.readouterr()
    tampered_res = json.loads(tampered_out.out)
    assert tampered_res["verified"] is False


def test_cli_verify_receipt_positional_and_syntax_alignment(tmp_path, capsys):
    """Verifies that verify-receipt accepts positional argument as documented in README.md."""
    parser = build_parser()
    decide_args = parser.parse_args(["decide", "Audit access logs", "--sign", "--json"])
    decide_args.func(decide_args)
    captured = capsys.readouterr()
    receipt_data = json.loads(captured.out)["receipt"]

    receipt_file = tmp_path / "test_receipt.json"
    receipt_file.write_text(json.dumps(receipt_data))

    # 1. Positional argument syntax matching README.md (system1 verify-receipt path/to/receipt.json)
    pos_args = parser.parse_args(["verify-receipt", str(receipt_file), "--json"])
    assert pos_args.func(pos_args) == 0
    pos_out = capsys.readouterr()
    pos_res = json.loads(pos_out.out)
    assert pos_res["verified"] is True
    assert pos_res["decision_id"] == receipt_data["decision_id"]

    # 2. Positional argument with human-readable output
    pos_human_args = parser.parse_args(["verify-receipt", str(receipt_file)])
    assert pos_human_args.func(pos_human_args) == 0
    human_out = capsys.readouterr().out
    assert "[SYSTEM1 RECEIPT VERIFICATION] SUCCESS" in human_out
    assert "VALID (Cryptographically Authentic & Tamper-Evident)" in human_out

    # 3. Positional argument with tampered receipt fails with exit code 1
    tampered_data = dict(receipt_data)
    tampered_data["decision_id"] = "dec_TAMPERED_12345"
    tampered_file = tmp_path / "tampered.json"
    tampered_file.write_text(json.dumps(tampered_data))

    tampered_args = parser.parse_args(["verify-receipt", str(tampered_file), "--json"])
    assert tampered_args.func(tampered_args) == 1
    tampered_res = json.loads(capsys.readouterr().out)
    assert tampered_res["verified"] is False

    # 4. Nonexistent receipt path returns error code 1 with clear stderr message
    missing_args = parser.parse_args(["verify-receipt", str(tmp_path / "nonexistent.json")])
    assert missing_args.func(missing_args) == 1
    missing_err = capsys.readouterr().err
    assert "[SYSTEM1 ERROR] Receipt file not found" in missing_err


def test_cli_verify_receipt_entrypoints_system1_and_reflex(tmp_path, capsys):
    """Verifies verify-receipt across both system1 and reflex entrypoints with positional and flag syntax."""
    import reflex.cli
    import system1.cli

    # Create signed receipt
    ret = system1.cli.main(["decide", "Entrypoint verification prompt", "--sign", "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    receipt_data = json.loads(captured.out)["receipt"]

    receipt_file = tmp_path / "entrypoint_receipt.json"
    receipt_file.write_text(json.dumps(receipt_data))

    # Test both entrypoints
    for entrypoint in [system1.cli.main, reflex.cli.main]:
        # Positional syntax: <cli> verify-receipt <file> --json
        assert entrypoint(["verify-receipt", str(receipt_file), "--json"]) == 0
        res = json.loads(capsys.readouterr().out)
        assert res["verified"] is True
        assert res["decision_id"] == receipt_data["decision_id"]

        # Positional syntax human readable: <cli> verify-receipt <file>
        assert entrypoint(["verify-receipt", str(receipt_file)]) == 0
        human_out = capsys.readouterr().out
        assert "[SYSTEM1 RECEIPT VERIFICATION] SUCCESS" in human_out

        # Flag syntax: <cli> verify-receipt --receipt <file> --json
        assert entrypoint(["verify-receipt", "--receipt", str(receipt_file), "--json"]) == 0
        res_flag = json.loads(capsys.readouterr().out)
        assert res_flag["verified"] is True

        # Flag syntax human readable: <cli> verify-receipt --receipt <file>
        assert entrypoint(["verify-receipt", "--receipt", str(receipt_file)]) == 0
        human_flag_out = capsys.readouterr().out
        assert "[SYSTEM1 RECEIPT VERIFICATION] SUCCESS" in human_flag_out


def test_cli_main_entrypoint(capsys):
    ret = main(["decide", "Status check ping", "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["schema_name"] == "DefaultTriageSchema"
