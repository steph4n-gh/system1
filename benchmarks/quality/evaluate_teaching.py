"""Compare starter classifiers with skills taught from the existing labeled data.

Five fixed, label-stratified folds; each example is evaluated exactly once.
Identical normalized prompts stay together. Related paraphrases and workflows
are not annotated, so results may be optimistic. These are development datasets,
not an independent release test. No hyperparameter search is performed.
"""

from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.quality.run_quality_benchmarks import (
    IntentRoutingSchema, SecurityTriageSchema, _confusion_matrix,
    _latency_stats, _precision_recall_f1,
)
from system1 import System1Engine
from system1.compiler import SystemOneCompiler
from system1.core.model import SystemOneModel


def make_folds(rows, label_key):
    groups = {}
    for index, row in enumerate(rows):
        key = " ".join(row["prompt"].casefold().split())
        groups.setdefault(key, []).append(index)
    strata = {}
    for indices in groups.values():
        strata.setdefault(rows[indices[0]][label_key], []).append(indices)
    folds = np.zeros(len(rows), dtype=int)
    rng = np.random.RandomState(42)
    for groups_in_class in strata.values():
        order = rng.permutation(len(groups_in_class))
        for offset, group_index in enumerate(order):
            folds[groups_in_class[group_index]] = offset % 5
    return folds


def evaluate(task, schema, field, label_key):
    path = ROOT / f"benchmarks/quality/datasets/{task}.json"
    rows = json.loads(path.read_text())
    folds = make_folds(rows, label_key)
    labels = list(schema().fields[field].options)
    report = {"dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "variants": {}}
    for variant, dimension in [("seed_384", 384), ("taught_384", 384), ("taught_2048", 2048)]:
        predictions, fold_details = [], []
        for fold in range(5):
            fit_indices = np.flatnonzero(folds != fold)
            test_indices = np.flatnonzero(folds == fold)
            if variant.startswith("taught"):
                model = SystemOneCompiler(schema, dimension=dimension).compile(
                    {field: [(rows[i]["prompt"], rows[i][label_key]) for i in fit_indices]},
                    augment=False, calibration_split=0.25,
                )
                model.use_cache = False
                fold_details.append({
                    "fold": fold, "sample_counts": model.metadata["sample_counts"][field],
                    "skill_bytes": len(model.to_bytes()),
                })
            else:
                model = SystemOneModel(schema, dimension=dimension, backend="numpy")
            engine = System1Engine(schema, model=model, use_cache=False, strict_mode=True)
            for i in test_indices:
                start = time.perf_counter()
                result = engine.decide(rows[i]["prompt"], record_receipt=False)
                elapsed = (time.perf_counter() - start) * 1000
                predictions.append({
                    "index": int(i), "fold": fold, "expected": rows[i][label_key],
                    "predicted": result.values[field], "needs_review": result.is_ambiguous,
                    "prediction_set": list(result.conformal_sets[field]), "latency_ms": elapsed,
                })
        predictions.sort(key=lambda p: p["index"])
        assert [p["index"] for p in predictions] == list(range(len(rows)))
        expected = [p["expected"] for p in predictions]
        predicted = [p["predicted"] for p in predictions]
        accepted = [p for p in predictions if not p["needs_review"]]
        metrics = {
            "accuracy": sum(a == b for a, b in zip(expected, predicted)) / len(rows),
            "classification": _precision_recall_f1(expected, predicted, labels),
            "confusion_matrix": _confusion_matrix(expected, predicted, labels),
            "accepted_count": len(accepted),
            "accepted_accuracy": (sum(p["expected"] == p["predicted"] for p in accepted) / len(accepted)) if accepted else None,
            "review_rate": 1 - len(accepted) / len(rows),
            "set_coverage": sum(p["expected"] in p["prediction_set"] for p in predictions) / len(rows),
            "latency": _latency_stats([p["latency_ms"] for p in predictions]),
            "folds": fold_details, "predictions": predictions,
        }
        if task == "security_triage":
            metrics["block_as_allow"] = sum(p["expected"] == "BLOCK" and p["predicted"] == "ALLOW" for p in predictions)
            metrics["accepted_block_as_allow"] = sum(p["expected"] == "BLOCK" and p["predicted"] == "ALLOW" for p in accepted)
        report["variants"][variant] = metrics
        print(f"{task:18} {variant:12} accuracy={metrics['accuracy']:.0%} review={metrics['review_rate']:.0%}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/quality/results/teaching_review.json")
    args = parser.parse_args()
    report = {
        "protocol": __doc__, "folds": 5, "seed": 42, "regularization": 1.0,
        "calibration_split": 0.25, "alpha": 0.05,
        "cache": False, "receipts": False,
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tasks": {
            "security_triage": evaluate("security_triage", SecurityTriageSchema, "action", "expected_action"),
            "intent_routing": evaluate("intent_routing", IntentRoutingSchema, "department", "expected_department"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
