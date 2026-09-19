"""Shared measurement code for the three single-skill examples, not a runtime API."""

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np

from system1 import System1Engine, __version__
from system1.compiler import CompiledSystemOneModel, SystemOneCompiler


def load_cases(path):
    data = json.loads(path.read_text())
    seen_groups, seen_prompts = set(), set()
    for split in ("teach", "calibration", "evaluate"):
        if not data[split]:
            raise ValueError(f"{split} must contain labeled examples")
        groups, prompts = set(), set()
        for row in data[split]:
            if not row["group"] or not row["prompt"].strip():
                raise ValueError(f"Empty case in {split}")
            groups.add(row["group"])
            prompt = " ".join(row["prompt"].casefold().split())
            if prompt in prompts:
                raise ValueError(f"Repeated prompt in {split}")
            prompts.add(prompt)
        if groups & seen_groups or prompts & seen_prompts:
            raise ValueError(f"Teaching, calibration, and evaluation cases must be disjoint: {split}")
        seen_groups.update(groups)
        seen_prompts.update(prompts)
    return data


def run_example(schema, name, argv=None):
    parser = argparse.ArgumentParser(description=f"Teach, save, and check the {name} skill locally.")
    parser.add_argument("--output-dir", type=Path, default=Path(".system1/examples"))
    args = parser.parse_args(argv)
    output = args.output_dir / name
    output.mkdir(parents=True, exist_ok=True)
    data_path = Path(__file__).parent / "teaching" / f"{name}.json"
    data = load_cases(data_path)
    definition = schema()
    field = next(iter(definition.fields))
    for split in ("teach", "calibration", "evaluate"):
        for row in data[split]:
            definition.fields[field].validate_value(row["label"])

    # The teaching loop: examples in, one compact skill out. No synthetic expansion.
    examples = {field: [(row["prompt"], row["label"]) for row in data["teach"]]}
    calibration = {field: [(row["prompt"], row["label"]) for row in data["calibration"]]}
    start = time.perf_counter()
    skill = SystemOneCompiler(schema, dimension=2048).compile(
        examples, augment=False, calibration_exemplars=calibration,
    )
    teaching_ms = (time.perf_counter() - start) * 1000
    data_digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    skill.metadata["example_dataset_sha256"] = data_digest
    start = time.perf_counter()
    path = output / "skill.s1m"
    skill.save(path)
    save_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    restored = CompiledSystemOneModel.load(path)
    load_ms = (time.perf_counter() - start) * 1000
    restored.use_cache = False
    engine = System1Engine(restored.schema, model=restored, strict_mode=True, use_cache=False)
    starter = System1Engine(schema, dimension=2048, strict_mode=True, use_cache=False)

    # These cases never enter compile(). Measure decisions from the saved skill.
    rows = []
    for case in data["evaluate"]:
        seed = starter.decide(case["prompt"], record_receipt=False)
        start = time.perf_counter()
        decision = engine.decide(case["prompt"], alpha=0.05, record_receipt=False)
        latency_ms = (time.perf_counter() - start) * 1000
        rows.append({
            **case, "starter_prediction": seed.values[field],
            "prediction": decision.values[field], "needs_review": decision.is_ambiguous,
            "prediction_set": list(decision.conformal_sets[field]), "latency_ms": latency_ms,
        })
    accepted = [row for row in rows if not row["needs_review"]]
    labels = list(definition.fields[field].options)
    counts = skill.metadata["sample_counts"][field]
    report = {
        "name": name, "provenance": data["provenance"], "policy": data["policy"],
        "dataset_sha256": data_digest, "skill_path": str(path),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "system1": __version__},
        "settings": {"dimension": 2048, "regularization": 1.0, "alpha": 0.05,
                     "strict": True, "cache": False, "receipts": False},
        "counts": {split: dict(Counter(row["label"] for row in data[split]))
                   for split in ("teach", "calibration", "evaluate")},
        "compiler_counts": counts,
        "teaching_ms": teaching_ms, "save_ms": save_ms, "load_ms": load_ms,
        "skill_bytes": path.stat().st_size,
        "starter_accuracy": sum(row["starter_prediction"] == row["label"] for row in rows) / len(rows),
        "accuracy": sum(row["prediction"] == row["label"] for row in rows) / len(rows),
        "accepted_count": len(accepted), "review_count": len(rows) - len(accepted),
        "accepted_accuracy": (sum(row["prediction"] == row["label"] for row in accepted) / len(accepted)) if accepted else None,
        "set_coverage": sum(row["label"] in row["prediction_set"] for row in rows) / len(rows),
        "confusion_matrix": {label: {pred: sum(row["label"] == label and row["prediction"] == pred for row in rows)
                                     for pred in labels} for label in labels},
        "decision_p50_ms": float(np.median([row["latency_ms"] for row in rows])),
        "decision_p95_ms": float(np.percentile([row["latency_ms"] for row in rows], 95)),
        "predictions": rows,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n{name}: {len(data['teach'])} teaching + {len(data['calibration'])} calibration + {len(rows)} unseen cases")
    print(f"Teach + calibrate: {teaching_ms:.1f} ms | Saved skill: {report['skill_bytes'] / 1024:.1f} KiB")
    print(f"Save: {save_ms:.1f} ms | Load: {load_ms:.1f} ms")
    print(f"Unseen accuracy: starter {report['starter_accuracy']:.1%} -> taught {report['accuracy']:.1%}")
    accepted_text = f"{report['accepted_accuracy']:.1%}" if accepted else "n/a"
    print(f"Answered without review: {len(accepted)}/{len(rows)} | Accuracy on those: {accepted_text}")
    print(f"Decision latency, uncached, no receipts: median {report['decision_p50_ms']:.3f} ms; p95 {report['decision_p95_ms']:.3f} ms")
    for row in rows:
        status = "REVIEW" if row["needs_review"] else "ANSWER"
        match = "correct" if row["prediction"] == row["label"] else f"expected {row['label']}"
        print(f"  {status:6} {row['prediction']:20} ({match}) {row['prompt']}")
    print(f"Skill: {path}\nFull report: {output / 'report.json'}")
    print("Authored demonstration cases; accuracy and timings are not production guarantees.")
    return report
