#!/usr/bin/env python3
"""Reflex Decision Engine — Quality Benchmark Suite.

Measures decision accuracy, precision, recall, F1, confusion matrices,
regression error (MAE/RMSE/correlation), and latency (P50/P99) for three
benchmark scenarios:

1. Security Triage  — 3-class classification (ALLOW/QUARANTINE/BLOCK)
2. Intent Routing   — 4-class classification (technical_support/billing/sales/escalate)
3. Threat Scoring   — Continuous regression on a 0–10 threat scale

Usage:
    python3 benchmarks/quality/run_quality_benchmarks.py          # run all
    python3 benchmarks/quality/run_quality_benchmarks.py --bench security
    python3 benchmarks/quality/run_quality_benchmarks.py --bench intent
    python3 benchmarks/quality/run_quality_benchmarks.py --bench scoring
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Path setup — mirrors the pattern used by examples/agent_guard.py
# ---------------------------------------------------------------------------
BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATASETS_DIR = BENCH_DIR / "datasets"
RESULTS_DIR = BENCH_DIR / "results"

from system1 import (  # noqa: E402
    BooleanField,
    ChoiceField,
    DecisionSchema,
    ReflexEngine,
    ScoreField,
)


# ============================================================================
# Metric helpers
# ============================================================================

def _precision_recall_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, Any]:
    """Compute per-class precision / recall / F1 and macro / micro averages."""
    tp: Dict[str, int] = defaultdict(int)
    fp: Dict[str, int] = defaultdict(int)
    fn: Dict[str, int] = defaultdict(int)

    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            tp[yt] += 1
        else:
            fp[yp] += 1
            fn[yt] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    for lbl in labels:
        p = tp[lbl] / (tp[lbl] + fp[lbl]) if (tp[lbl] + fp[lbl]) > 0 else 0.0
        r = tp[lbl] / (tp[lbl] + fn[lbl]) if (tp[lbl] + fn[lbl]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        support = tp[lbl] + fn[lbl]
        per_class[lbl] = {"precision": p, "recall": r, "f1": f1, "support": support}

    macro_p = statistics.mean(c["precision"] for c in per_class.values()) if per_class else 0.0
    macro_r = statistics.mean(c["recall"] for c in per_class.values()) if per_class else 0.0
    macro_f1 = statistics.mean(c["f1"] for c in per_class.values()) if per_class else 0.0

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return {
        "per_class": per_class,
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
    }


def _confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    """Row = actual, column = predicted."""
    matrix: Dict[str, Dict[str, int]] = {lbl: {l2: 0 for l2 in labels} for lbl in labels}
    for yt, yp in zip(y_true, y_pred):
        if yt in matrix and yp in matrix[yt]:
            matrix[yt][yp] += 1
    return matrix


def _latency_stats(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """Compute P50, P90, P95, P99, mean, min, max from a list of latencies."""
    if not latencies_ms:
        return {}
    s = sorted(latencies_ms)
    n = len(s)

    def _percentile(pct: float) -> float:
        idx = int(math.ceil(pct / 100.0 * n)) - 1
        return s[max(0, min(idx, n - 1))]

    return {
        "count": n,
        "mean_ms": statistics.mean(s),
        "min_ms": s[0],
        "max_ms": s[-1],
        "p50_ms": _percentile(50),
        "p90_ms": _percentile(90),
        "p95_ms": _percentile(95),
        "p99_ms": _percentile(99),
    }


def _regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> Dict[str, float]:
    """Compute MAE, RMSE, and Pearson correlation for regression."""
    n = len(y_true)
    if n == 0:
        return {"mae": 0.0, "rmse": 0.0, "pearson_r": 0.0}

    errors = [abs(t - p) for t, p in zip(y_true, y_pred)]
    mae = statistics.mean(errors)
    mse = statistics.mean((t - p) ** 2 for t, p in zip(y_true, y_pred))
    rmse = math.sqrt(mse)

    # Pearson correlation
    if n < 2:
        pearson_r = 0.0
    else:
        mean_t = statistics.mean(y_true)
        mean_p = statistics.mean(y_pred)
        num = sum((t - mean_t) * (p - mean_p) for t, p in zip(y_true, y_pred))
        den_t = math.sqrt(sum((t - mean_t) ** 2 for t in y_true))
        den_p = math.sqrt(sum((p - mean_p) ** 2 for p in y_pred))
        pearson_r = num / (den_t * den_p) if (den_t * den_p) > 0 else 0.0

    return {"mae": mae, "rmse": rmse, "pearson_r": pearson_r}


# ============================================================================
# Pretty-printing helpers
# ============================================================================

def _print_classification_report(
    title: str,
    labels: Sequence[str],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    latencies: Sequence[float],
) -> Dict[str, Any]:
    """Print and return a full classification quality report."""
    metrics = _precision_recall_f1(y_true, y_pred, labels)
    cm = _confusion_matrix(y_true, y_pred, labels)
    lat = _latency_stats(latencies)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0

    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"\n  Total examples:  {len(y_true)}")
    print(f"  Accuracy:        {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print()

    # Per-class table
    header = f"  {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    print(header)
    print(f"  {'-' * 65}")
    for lbl in labels:
        c = metrics["per_class"].get(lbl, {})
        print(
            f"  {lbl:<25} {c.get('precision', 0):.4f}     {c.get('recall', 0):.4f}     "
            f"{c.get('f1', 0):.4f}     {c.get('support', 0):>5}"
        )
    print(f"  {'-' * 65}")
    print(
        f"  {'macro avg':<25} {metrics['macro']['precision']:.4f}     "
        f"{metrics['macro']['recall']:.4f}     {metrics['macro']['f1']:.4f}"
    )
    print(
        f"  {'micro avg':<25} {metrics['micro']['precision']:.4f}     "
        f"{metrics['micro']['recall']:.4f}     {metrics['micro']['f1']:.4f}"
    )

    # Confusion matrix
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    col_w = max(len(l) for l in labels) + 2
    row_header = "  " + " " * col_w + "".join(f"{l:>{col_w}}" for l in labels)
    print(row_header)
    for actual in labels:
        row_vals = "".join(f"{cm[actual][pred]:>{col_w}}" for pred in labels)
        print(f"  {actual:<{col_w}}{row_vals}")

    # Latency
    print(f"\n  Latency Statistics:")
    print(f"    Mean:  {lat.get('mean_ms', 0):.3f} ms")
    print(f"    P50:   {lat.get('p50_ms', 0):.3f} ms")
    print(f"    P90:   {lat.get('p90_ms', 0):.3f} ms")
    print(f"    P95:   {lat.get('p95_ms', 0):.3f} ms")
    print(f"    P99:   {lat.get('p99_ms', 0):.3f} ms")
    print(f"    Min:   {lat.get('min_ms', 0):.3f} ms")
    print(f"    Max:   {lat.get('max_ms', 0):.3f} ms")

    return {
        "accuracy": accuracy,
        "metrics": metrics,
        "confusion_matrix": cm,
        "latency": lat,
        "total_examples": len(y_true),
    }


def _print_regression_report(
    title: str,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    latencies: Sequence[float],
) -> Dict[str, Any]:
    """Print and return a full regression quality report."""
    reg = _regression_metrics(y_true, y_pred)
    lat = _latency_stats(latencies)

    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"\n  Total examples:  {len(y_true)}")
    print(f"  MAE:             {reg['mae']:.4f}")
    print(f"  RMSE:            {reg['rmse']:.4f}")
    print(f"  Pearson r:       {reg['pearson_r']:.4f}")

    # Score distribution table
    print(f"\n  Prediction Error Distribution:")
    buckets = {"[0, 1)": 0, "[1, 2)": 0, "[2, 3)": 0, "[3, 5)": 0, "[5, 10]": 0}
    for t, p in zip(y_true, y_pred):
        err = abs(t - p)
        if err < 1:
            buckets["[0, 1)"] += 1
        elif err < 2:
            buckets["[1, 2)"] += 1
        elif err < 3:
            buckets["[2, 3)"] += 1
        elif err < 5:
            buckets["[3, 5)"] += 1
        else:
            buckets["[5, 10]"] += 1
    for bucket, count in buckets.items():
        pct = count / len(y_true) * 100 if y_true else 0
        bar = "█" * int(pct / 2)
        print(f"    {bucket:<10} {count:>4} ({pct:5.1f}%) {bar}")

    # Sample predictions
    print(f"\n  Sample Predictions (first 10):")
    print(f"    {'Expected':>10} {'Predicted':>10} {'Error':>10}")
    print(f"    {'-' * 32}")
    for t, p in list(zip(y_true, y_pred))[:10]:
        print(f"    {t:>10.2f} {p:>10.2f} {abs(t - p):>10.2f}")

    # Latency
    print(f"\n  Latency Statistics:")
    print(f"    Mean:  {lat.get('mean_ms', 0):.3f} ms")
    print(f"    P50:   {lat.get('p50_ms', 0):.3f} ms")
    print(f"    P90:   {lat.get('p90_ms', 0):.3f} ms")
    print(f"    P95:   {lat.get('p95_ms', 0):.3f} ms")
    print(f"    P99:   {lat.get('p99_ms', 0):.3f} ms")
    print(f"    Min:   {lat.get('min_ms', 0):.3f} ms")
    print(f"    Max:   {lat.get('max_ms', 0):.3f} ms")

    return {
        "regression": reg,
        "latency": lat,
        "total_examples": len(y_true),
        "error_distribution": buckets,
    }


# ============================================================================
# Schema definitions (following patterns from support_triage.py / agent_guard.py)
# ============================================================================

class SecurityTriageSchema(DecisionSchema):
    """3-class security triage for agent tool calls."""
    action = ChoiceField(
        options=["ALLOW", "QUARANTINE", "BLOCK"],
        descriptions={
            "ALLOW": [
                "Safe read-only or diagnostic operation",
                "Viewing files, listing directories, reading logs",
                "Running tests, linting, or type checking",
                "Displaying system information or git status",
            ],
            "QUARANTINE": [
                "Needs human review before execution",
                "Modifying configuration files or installing packages",
                "Creating or modifying files in the project",
                "Network requests to known internal endpoints",
                "Running development servers or Docker containers",
                "Database writes on development environments",
            ],
            "BLOCK": [
                "Destructive, malicious, or dangerous operation",
                "Deleting system files, wiping disks, rm -rf",
                "Exfiltrating secrets, SSH keys, API keys, credentials",
                "Reverse shells, exploits, malware, rootkits",
                "Prompt injection and jailbreak attempts",
                "Unauthorized privilege escalation",
                "Ransomware or data destruction",
                "SQL injection, unauthorized database manipulation",
            ],
        },
        description="Security authorization decision for an agent tool call",
    )


class IntentRoutingSchema(DecisionSchema):
    """4-class customer query intent routing."""
    department = ChoiceField(
        options=["technical_support", "billing", "sales", "escalate"],
        descriptions={
            "technical_support": [
                "System bugs, API errors, 502/500 errors, crashes",
                "Integration issues, SDK questions, webhook problems",
                "Performance problems, slow loading, timeouts",
                "Feature questions, configuration help, documentation",
                "Login issues, authentication, access problems",
            ],
            "billing": [
                "Invoice disputes, duplicate charges, refund requests",
                "Payment method updates, billing address changes",
                "Pricing questions, plan comparisons",
                "Subscription cancellation and prorated refunds",
                "Payment failures, credit card issues",
            ],
            "sales": [
                "Enterprise seat purchases, volume licensing",
                "Product demos, feature comparisons with competitors",
                "Custom pricing, non-profit discounts",
                "Multi-year contracts, partner programs",
                "Upgrade inquiries, plan expansion",
            ],
            "escalate": [
                "Angry customer demanding manager or supervisor",
                "Repeated unresolved issues, formal complaints",
                "SLA violations, legal threats, regulatory issues",
                "Data breach concerns, security incidents",
                "Executive-level engagement requests",
                "High financial impact or revenue loss",
            ],
        },
        description="Route customer query to the appropriate department",
    )


class ThreatScoringSchema(DecisionSchema):
    """Continuous 0–10 threat score for security events."""
    threat_score = ScoreField(
        min_value=0.0,
        max_value=10.0,
        low_description=(
            "Benign routine operation: read-only file access, directory listing, "
            "log inspection, git status, documentation reading, safe diagnostics"
        ),
        high_description=(
            "Critical security threat: remote code execution, credential theft, "
            "data exfiltration, ransomware, rootkit installation, system destruction, "
            "reverse shells, prompt injection jailbreaks"
        ),
        description="Continuous threat risk score from 0.0 (benign) to 10.0 (critical)",
    )


# ============================================================================
# Benchmark runners
# ============================================================================

def _load_dataset(name: str) -> List[Dict[str, Any]]:
    """Load a JSON dataset from the datasets/ directory."""
    path = DATASETS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_security_triage_benchmark() -> Dict[str, Any]:
    """Benchmark: Security Triage (3-class classification)."""
    dataset = _load_dataset("security_triage")
    engine = ReflexEngine(SecurityTriageSchema, backend="auto")

    labels = ["ALLOW", "QUARANTINE", "BLOCK"]
    y_true: List[str] = []
    y_pred: List[str] = []
    latencies: List[float] = []
    raw_results: List[Dict[str, Any]] = []

    for item in dataset:
        prompt = item["prompt"]
        expected = item["expected_action"]

        t0 = time.perf_counter()
        result = engine.decide(prompt, alpha=0.05)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        predicted = result.values.get("action", result.action)
        y_true.append(expected)
        y_pred.append(predicted)
        latencies.append(elapsed_ms)
        raw_results.append({
            "prompt": prompt,
            "expected": expected,
            "predicted": predicted,
            "correct": expected == predicted,
            "confidence": result.confidences.get("action", 0.0),
            "latency_ms": elapsed_ms,
            "category": item.get("category", ""),
        })

    report = _print_classification_report(
        "BENCHMARK 1: Security Triage (ALLOW / QUARANTINE / BLOCK)",
        labels, y_true, y_pred, latencies,
    )
    report["raw_results"] = raw_results
    return report


def run_intent_routing_benchmark() -> Dict[str, Any]:
    """Benchmark: Intent Routing (4-class classification)."""
    dataset = _load_dataset("intent_routing")
    engine = ReflexEngine(IntentRoutingSchema, backend="auto")

    labels = ["technical_support", "billing", "sales", "escalate"]
    y_true: List[str] = []
    y_pred: List[str] = []
    latencies: List[float] = []
    raw_results: List[Dict[str, Any]] = []

    for item in dataset:
        prompt = item["prompt"]
        expected = item["expected_department"]

        t0 = time.perf_counter()
        result = engine.decide(prompt, alpha=0.05)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        predicted = result.values.get("department", result.department)
        y_true.append(expected)
        y_pred.append(predicted)
        latencies.append(elapsed_ms)
        raw_results.append({
            "prompt": prompt,
            "expected": expected,
            "predicted": predicted,
            "correct": expected == predicted,
            "confidence": result.confidences.get("department", 0.0),
            "latency_ms": elapsed_ms,
            "category": item.get("category", ""),
        })

    report = _print_classification_report(
        "BENCHMARK 2: Intent Routing (technical_support / billing / sales / escalate)",
        labels, y_true, y_pred, latencies,
    )
    report["raw_results"] = raw_results
    return report


def run_threat_scoring_benchmark() -> Dict[str, Any]:
    """Benchmark: Threat Scoring (continuous 0–10 regression)."""
    dataset = _load_dataset("threat_scoring")
    engine = ReflexEngine(ThreatScoringSchema, backend="auto")

    y_true: List[float] = []
    y_pred: List[float] = []
    latencies: List[float] = []
    raw_results: List[Dict[str, Any]] = []

    for item in dataset:
        prompt = item["prompt"]
        expected = float(item["expected_score"])

        t0 = time.perf_counter()
        result = engine.decide(prompt, alpha=0.05)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        predicted = float(result.values.get("threat_score", result.threat_score))
        y_true.append(expected)
        y_pred.append(predicted)
        latencies.append(elapsed_ms)
        raw_results.append({
            "prompt": prompt,
            "expected": expected,
            "predicted": predicted,
            "error": abs(expected - predicted),
            "latency_ms": elapsed_ms,
            "category": item.get("category", ""),
        })

    report = _print_regression_report(
        "BENCHMARK 3: Threat Scoring (0.0 – 10.0 scale)",
        y_true, y_pred, latencies,
    )
    report["raw_results"] = raw_results
    return report


# ============================================================================
# Main entry point
# ============================================================================

BENCHMARKS = {
    "security": ("Security Triage", run_security_triage_benchmark),
    "intent": ("Intent Routing", run_intent_routing_benchmark),
    "scoring": ("Threat Scoring", run_threat_scoring_benchmark),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reflex Decision Engine — Quality Benchmark Suite",
    )
    parser.add_argument(
        "--bench",
        choices=list(BENCHMARKS.keys()) + ["all"],
        default="all",
        help="Which benchmark to run (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output JSON path (default: benchmarks/quality/results/quality_results_<timestamp>.json)",
    )
    args = parser.parse_args()

    selected = list(BENCHMARKS.keys()) if args.bench == "all" else [args.bench]

    print("#" * 80)
    print("  REFLEX DECISION ENGINE — QUALITY BENCHMARK SUITE")
    print(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("#" * 80)

    all_results: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmarks": {},
    }

    for key in selected:
        name, runner = BENCHMARKS[key]
        print(f"\n>>> Running: {name} ...")
        try:
            report = runner()
            # Strip raw_results from summary display but keep in JSON
            all_results["benchmarks"][key] = report
        except Exception as exc:
            print(f"\n  ERROR running {name}: {exc}")
            import traceback
            traceback.print_exc()
            all_results["benchmarks"][key] = {"error": str(exc)}

    # Write JSON results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"quality_results_{ts}.json"

    # Make raw_results JSON-serialisable (strip numpy/dataclass artefacts)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 80}")
    print(f"  Results saved to: {out_path}")
    print(f"{'=' * 80}")

    # Print compact summary
    print(f"\n{'─' * 80}")
    print("  SUMMARY")
    print(f"{'─' * 80}")
    for key in selected:
        br = all_results["benchmarks"].get(key, {})
        if "error" in br:
            print(f"  {key:<15} ERROR: {br['error']}")
        elif "accuracy" in br:
            lat = br.get("latency", {})
            print(
                f"  {key:<15} Accuracy: {br['accuracy']:.4f}  "
                f"Macro-F1: {br['metrics']['macro']['f1']:.4f}  "
                f"P50: {lat.get('p50_ms', 0):.3f}ms  "
                f"P99: {lat.get('p99_ms', 0):.3f}ms"
            )
        elif "regression" in br:
            reg = br["regression"]
            lat = br.get("latency", {})
            print(
                f"  {key:<15} MAE: {reg['mae']:.4f}  "
                f"RMSE: {reg['rmse']:.4f}  "
                f"Pearson r: {reg['pearson_r']:.4f}  "
                f"P50: {lat.get('p50_ms', 0):.3f}ms  "
                f"P99: {lat.get('p99_ms', 0):.3f}ms"
            )
    print(f"{'─' * 80}\n")


if __name__ == "__main__":
    main()
