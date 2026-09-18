"""Tests for system1 compile CLI subcommand."""

import json
from pathlib import Path
import pytest

import system1.cli as s1_cli
from system1.compiler import CompiledSystemOneModel


def test_cli_compile_triage_preset_json(tmp_path: Path, capsys):
    """Verify 'system1 compile --schema triage --json' compiles and outputs JSON."""
    out_file = tmp_path / "triage.s1m"
    ret = s1_cli.main([
        "compile",
        "--schema", "triage",
        "--output", str(out_file),
        "--samples-per-choice", "10",
        "--json",
    ])
    assert ret == 0
    assert out_file.is_file()

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["status"] == "compiled"
    assert summary["schema_name"] == "DefaultTriageSchema"
    assert "route" in summary["fields_compiled"]
    assert summary["file_size_kb"] > 0

    # Verify loaded model executes correctly
    model = CompiledSystemOneModel.load(out_file)
    assert model.schema.schema_name == "DefaultTriageSchema"
    res = model.forward_single("Read /etc/resolv.conf")
    assert "route" in res.fields
    assert res.fields["route"].selected_value in ["filesystem", "database", "network", "system_shell", "human_approval"]


def test_cli_compile_guard_preset_text(tmp_path: Path, capsys):
    """Verify 'system1 compile --schema guard' prints human-readable summary."""
    out_file = tmp_path / "guard.s1m"
    ret = s1_cli.main([
        "compile",
        "--schema", "guard",
        "--output", str(out_file),
        "--samples-per-choice", "5",
    ])
    assert ret == 0
    assert out_file.is_file()

    captured = capsys.readouterr()
    assert "[SYSTEM 1 COMPILER] Distillation and Compilation Complete" in captured.out
    assert "guard.s1m" in captured.out


def test_cli_compile_dotted_import_path(tmp_path: Path, capsys):
    """Verify 'system1 compile --schema <module.Class>' imports and compiles custom schema class."""
    out_file = tmp_path / "custom.s1m"
    ret = s1_cli.main([
        "compile",
        "--schema", "tests.test_compiler.SupportTriageSchema",
        "--output", str(out_file),
        "--samples-per-choice", "5",
        "--json",
    ])
    assert ret == 0
    assert out_file.is_file()

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["schema_name"] == "SupportTriageSchema"
    assert "category" in summary["fields_compiled"]


def test_cli_compile_invalid_schema():
    """Verify error on nonexistent or invalid schema argument."""
    with pytest.raises(ValueError, match="Unable to parse schema"):
        s1_cli.main([
            "compile",
            "--schema", "nonexistent_schema_name_12345",
        ])
