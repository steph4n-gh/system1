"""Tests for Enterprise Auto-Cutover Showcase (examples/auto_cutover_showcase.py)."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import auto_cutover_showcase
from auto_cutover_showcase import (
    get_clinical_triage_use_case,
    get_ecommerce_support_use_case,
    get_fintech_wire_fraud_use_case,
    run_auto_cutover_showcase,
    run_burst_benchmark,
)
from system1.compat.typesafe import TypeSafeClient
from system1.ledger import ActionLedger


def test_use_case_definitions():
    """Verify all 3 killer enterprise use cases return valid schemas and query streams."""
    cases = [
        get_fintech_wire_fraud_use_case(),
        get_clinical_triage_use_case(),
        get_ecommerce_support_use_case(),
    ]
    for title, questions, queries in cases:
        assert isinstance(title, str) and len(title) > 0
        assert isinstance(questions, dict) and len(questions) >= 3
        assert isinstance(queries, list) and len(queries) >= 8
        for q in queries:
            assert "prompt" in q
            assert len(q["prompt"]) > 0


def test_fintech_auto_cutover_showcase_run(tmp_path: Path):
    """Verify complete FinTech auto-cutover lifecycle execution with durable ledger."""
    db_file = str(tmp_path / "fintech_audit.db")

    # Run showcase with threshold=3, total=6
    run_auto_cutover_showcase(
        use_case="fintech",
        cutover_threshold=3,
        total_queries=6,
        db_path=db_file,
        run_benchmark=False,
        run_monkey_patch=False,
        throttle_ms=0.0,
    )

    # Verify durable SQLite ledger file exists and is valid
    ledger = ActionLedger(db_file, read_only=True)
    assert ledger.verify_integrity() is True
    head_seq, _ = ledger.audit_head()
    # At least 3 passthrough queries + 1 cutover event + 3 local receipts = 7 entries
    assert head_seq >= 7


def test_clinical_and_ecommerce_showcase_runs():
    """Verify clinical and e-commerce use cases execute cleanly in in-memory mode."""
    # Clinical
    run_auto_cutover_showcase(
        use_case="clinical",
        cutover_threshold=2,
        total_queries=4,
        db_path=":memory:",
        run_benchmark=False,
        run_monkey_patch=True,
        throttle_ms=0.0,
    )

    # E-Commerce
    run_auto_cutover_showcase(
        use_case="ecommerce",
        cutover_threshold=2,
        total_queries=4,
        db_path=":memory:",
        run_benchmark=False,
        run_monkey_patch=True,
        throttle_ms=0.0,
    )


def test_burst_benchmark_execution():
    """Verify burst benchmark runs on post-cutover client."""
    title, questions, queries = get_fintech_wire_fraud_use_case()
    client = TypeSafeClient(mode="local")

    # Run burst benchmark for 10 iterations
    run_burst_benchmark(client, questions, queries, burst_count=10)


def test_showcase_non_cutover_edge_case(tmp_path: Path):
    """Verify showcase completes gracefully without cutover when total < threshold."""
    db_file = str(tmp_path / "non_cutover.db")

    # threshold=6, total=3 -> cutover does not trigger
    run_auto_cutover_showcase(
        use_case="fintech",
        cutover_threshold=6,
        total_queries=3,
        db_path=db_file,
        run_benchmark=True,  # Should skip gracefully with notice
        run_monkey_patch=True,
        throttle_ms=0.0,
    )

    ledger = ActionLedger(db_file, read_only=True)
    assert ledger.verify_integrity() is True
    head_seq, _ = ledger.audit_head()
    assert head_seq == 3


def test_showcase_with_model_export(tmp_path: Path):
    """Verify model export option serializes valid .s1m binary loaded by CompiledSystemOneModel."""
    from system1.compiler import CompiledSystemOneModel

    export_path = str(tmp_path / "exported_showcase_model.s1m")

    run_auto_cutover_showcase(
        use_case="fintech",
        cutover_threshold=3,
        total_queries=5,
        db_path=":memory:",
        run_benchmark=False,
        run_monkey_patch=False,
        throttle_ms=0.0,
        export_model_path=export_path,
    )

    assert Path(export_path).exists()
    assert Path(export_path).stat().st_size > 0

    # Load exported binary model directly
    loaded_model = CompiledSystemOneModel.load(export_path)
    res = loaded_model.forward_single("Payroll ACH domestic wire transfer $1,200.00")
    assert "decision" in res.fields
    assert res.fields["decision"].selected_value is not None
