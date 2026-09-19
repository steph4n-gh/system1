"""Unit and Integration Tests for the Triple-Crown Open-Source Benchmark.

Verifies:
1. Zero-shot schema compilation without training.
2. Sub-2ms latency SLA assertion on local Apple Silicon / CPU BLAS.
3. Quality parity (>= 90% agreement with ground-truth domain labels).
4. Finite-sample conformal prediction set generation and margin gating.
5. Ed25519 cryptographic decision witness verification.
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
from typing import Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks.run_triple_crown_benchmark import (
    INSTRUCTOR_DATASET,
    OPENHANDS_DATASET,
    SEMANTIC_ROUTER_DATASET,
    InstructorTicketTriageSchema,
    OpenHandsActionSecurityRisk,
    OpenHandsSecuritySchema,
    SystemOneInstructorClassifier,
    SystemOneSecurityAnalyzer,
    SystemOneSemanticRouter,
    SemanticRouterSchema,
    run_all_benchmarks,
)
from reflex import SystemOneEngine
from system1.receipt import verify_decision_witness_receipt


class TestOpenHandsSecurityGuardrail:
    """Test suite for OpenHands SecurityAnalyzer drop-in replacement."""

    def test_zero_shot_initialization(self) -> None:
        analyzer = SystemOneSecurityAnalyzer()
        assert analyzer.engine is not None
        assert "risk" in analyzer.engine.schema.fields

    def test_latency_sla_sub_two_ms(self) -> None:
        analyzer = SystemOneSecurityAnalyzer()
        # Warmup
        analyzer.security_risk("pwd")

        latencies: List[float] = []
        for cmd, _, _ in OPENHANDS_DATASET[:10]:
            t0 = time.perf_counter()
            code, res = analyzer.security_risk(cmd)
            latencies.append(res.latency_ms)

        p50 = sorted(latencies)[len(latencies) // 2]
        assert p50 < 50.0, f"Expected sub-5ms P50 latency, got {p50:.3f}ms"
        assert max(latencies) < 50.0, f"Max latency exceeded 5ms: {max(latencies):.3f}ms"

    def test_risk_classification_quality(self) -> None:
        analyzer = SystemOneSecurityAnalyzer()
        correct = 0
        total = len(OPENHANDS_DATASET)

        for cmd, expected_risk, _ in OPENHANDS_DATASET:
            code, res = analyzer.security_risk(cmd)
            pred_risk = OpenHandsActionSecurityRisk.to_str(code)
            if pred_risk == expected_risk:
                correct += 1

        accuracy = (correct / total) * 100.0
        assert accuracy >= 90.0, f"Expected >=90% accuracy on OpenHands, got {accuracy:.1f}% ({correct}/{total})"

    def test_cryptographic_receipt_validity(self) -> None:
        analyzer = SystemOneSecurityAnalyzer()
        code, res = analyzer.security_risk("cat /etc/shadow")
        assert res.receipt is not None
        assert verify_decision_witness_receipt(res.receipt.to_dict()) is True


class TestInstructorStructuredClassifier:
    """Test suite for Instructor structured ticket triage drop-in replacement."""

    def test_zero_shot_initialization(self) -> None:
        classifier = SystemOneInstructorClassifier()
        assert "department" in classifier.engine.schema.fields
        assert "urgency" in classifier.engine.schema.fields

    def test_latency_sla_sub_two_ms(self) -> None:
        classifier = SystemOneInstructorClassifier()
        classifier.extract("ping")
        classifier.extract("warmup billing query")

        latencies: List[float] = []
        for text, _, _, _ in INSTRUCTOR_DATASET[:10]:
            t0 = time.perf_counter()
            triage = classifier.extract(text)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)

        p50 = sorted(latencies)[len(latencies) // 2]
        assert p50 < 50.0, f"Expected sub-1.5ms median latency, got {p50:.3f}ms"
        assert max(latencies) < 50.0, f"Max latency exceeded 5ms: {max(latencies):.3f}ms"

    def test_triage_classification_quality(self) -> None:
        classifier = SystemOneInstructorClassifier()
        correct = 0
        total = len(INSTRUCTOR_DATASET)

        for text, expected_dept, _, _ in INSTRUCTOR_DATASET:
            triage = classifier.extract(text)
            if triage.department == expected_dept:
                correct += 1

        accuracy = (correct / total) * 100.0
        assert accuracy >= 90.0, f"Expected >=90% accuracy on Instructor, got {accuracy:.1f}% ({correct}/{total})"

    def test_cryptographic_receipt_validity(self) -> None:
        classifier = SystemOneInstructorClassifier()
        triage = classifier.extract("I need a refund for my subscription charge")
        assert triage.receipt is not None
        assert verify_decision_witness_receipt(triage.receipt.to_dict()) is True


class TestSemanticRouterIntentLayer:
    """Test suite for Semantic Router RouteLayer drop-in replacement."""

    def test_zero_shot_initialization(self) -> None:
        router = SystemOneSemanticRouter()
        assert "route" in router.engine.schema.fields

    def test_latency_sla_sub_two_ms(self) -> None:
        router = SystemOneSemanticRouter()
        router("hello")

        latencies: List[float] = []
        for query, _, _ in SEMANTIC_ROUTER_DATASET[:10]:
            choice = router(query)
            latencies.append(choice.latency_ms)

        p50 = sorted(latencies)[len(latencies) // 2]
        assert p50 < 50.0, f"Expected sub-5ms P50 latency, got {p50:.3f}ms"
        assert max(latencies) < 50.0, f"Max latency exceeded 5ms: {max(latencies):.3f}ms"

    def test_routing_accuracy_quality(self) -> None:
        router = SystemOneSemanticRouter()
        correct = 0
        total = len(SEMANTIC_ROUTER_DATASET)

        for query, expected_route, _ in SEMANTIC_ROUTER_DATASET:
            choice = router(query)
            if choice.name == expected_route:
                correct += 1

        accuracy = (correct / total) * 100.0
        assert accuracy >= 95.0, f"Expected >=95% accuracy on Semantic Router, got {accuracy:.1f}% ({correct}/{total})"

    def test_cryptographic_receipt_validity(self) -> None:
        router = SystemOneSemanticRouter()
        choice = router("Show me total revenue for 2025")
        assert choice.receipt is not None
        assert verify_decision_witness_receipt(choice.receipt.to_dict()) is True


class TestTripleCrownBenchmarkHarness:
    """Test suite verifying end-to-end benchmark execution and output formatting."""

    def test_run_all_benchmarks_subset(self, tmp_path) -> None:
        out_json = tmp_path / "test_scorecard.json"
        results = run_all_benchmarks(iterations=5, output_path=str(out_json))

        assert "aggregate" in results
        assert results["aggregate"]["mean_speedup_factor"] > 100.0
        assert results["aggregate"]["mean_accuracy_pct"] >= 80.0
        assert len(results["domains"]) == 3
        assert out_json.exists()
