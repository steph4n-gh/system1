"""Recompute retained regression evidence without loading or running a model."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(correct, total):
    if not total:
        return dict(numerator=correct, denominator=0, fraction=None, wilson95=None)
    p, z = correct / total, 1.959963984540054
    d = 1 + z * z / total
    center = (p + z * z / (2 * total)) / d
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return dict(numerator=correct, denominator=total, fraction=p,
                wilson95=[max(0., center - radius), min(1., center + radius)])


def metrics(rows):
    supported = [r for r in rows if r["truth"] != "oos"]
    accepted = [r for r in supported if not r["needsReview"]]
    oos = [r for r in rows if r["truth"] == "oos"]
    return dict(supported_coverage=ratio(len(accepted), len(supported)),
        accepted_accuracy=ratio(sum(r["category"] == r["truth"] for r in accepted), len(accepted)),
        raw_supported_accuracy=ratio(sum(r.get("suggestion") == r["truth"] for r in supported), len(supported)),
        oos_false_acceptance=ratio(sum(not r["needsReview"] for r in oos), len(oos)))


def same(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            same(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            same(a, b)
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    else:
        assert actual == expected


def main():
    path = HERE / "results/observed-regression.json"
    report = json.loads(path.read_text())
    freeze_path = HERE / "observed-regression-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    original = json.loads((HERE / "freeze.json").read_text())
    manifest = json.loads((HERE / "manifest.json").read_text())
    assert report["qualified"] is False and report["independent_confirmation"] is False
    assert report["freeze_sha256"] == digest(freeze_path)
    assert freeze["original_freeze_sha256"] == digest(HERE / "freeze.json")
    assert freeze["criteria"] == original["criteria"] == report["criteria"]
    for name, expected in freeze["files"].items():
        assert digest(ROOT / name) == expected, name
    checks = []
    for result in report["results"]:
        dataset, kind = result["dataset"], result["kind"]
        source = ROOT / f".system1/n8n-gauntlet/{dataset}-test.json"
        expected = manifest["datasets"][dataset]["splits"]["test"]
        assert digest(source) == expected["sha256"]
        pinned = json.loads(source.read_text())
        rows = result["outcomes"]
        assert len(pinned) == len(rows) == expected["rows"]
        for index, (row, original_row) in enumerate(zip(rows, pinned, strict=True)):
            assert row["index"] == index
            assert row["text"] == original_row["prompt"] and row["truth"] == original_row["label"]
            assert row["group"] == original_row["group"]
            assert row["candidate"] == result["candidate_sha256"]
            assert row["teacherCalls"] == 0 and math.isfinite(row["latency_ms"]) and row["latency_ms"] >= 0
        same(result["metrics"], metrics(rows))
        per_intent = {label: metrics([r for r in rows if r["truth"] == label])
                      for label in sorted({r["truth"] for r in rows})}
        same(result["per_intent"], per_intent)
        assert result["accepted_errors"] == [r for r in rows if not r["needsReview"] and r["category"] != r["truth"]]
        times = [r["latency_ms"] for r in rows]
        same(result["latency"], dict(requests=len(rows), warmups=100, p50_ms=float(np.median(times)),
             p95_ms=float(np.percentile(times, 95)), max_ms=max(times)))
        m = result["metrics"]
        errors = sum("error" in r for r in rows)
        gates = dict(coverage_at_least_80=m["supported_coverage"]["fraction"] >= .8,
            accepted_accuracy_at_least_99=m["accepted_accuracy"]["fraction"] >= .99,
            oos_false_acceptance_at_most_1=m["oos_false_acceptance"]["fraction"] is None or m["oos_false_acceptance"]["fraction"] <= .01,
            adapter_p95_below_5_ms=result["latency"]["p95_ms"] < 5, no_runtime_errors=errors == 0)
        assert gates == result["point_gates"] and all(gates.values()) == result["all_point_targets_pass"]
        assert result["qualified"] is False and result["errors"] == errors and result["teacher_calls"] == 0
        assert result["response_cache"] is False and result["receipts"] is False
        folder = HERE / "artifacts" / freeze["candidates"][dataset][kind]
        assert result["candidate_sha256"] == digest(folder / "manifest.json")
        assert result["manifest"] == json.loads((folder / "manifest.json").read_text())
        folders = [folder]
        if kind == "baseline":
            folders.append(folder / result["manifest"]["base_artifact"])
        assert result["artifact_bytes"] == sum(f.stat().st_size for p in folders for f in p.iterdir() if f.is_file())
        accepted = [r for r in rows if not r["needsReview"]]
        checks.append(dict(dataset=dataset, kind=kind, original_rows_preserved=len(rows),
            all_labels_and_texts_identical_to_pinned_test=True, all_counts_intervals_gates_and_timings_recomputed=True,
            all_accepted_accuracy_including_unfamiliar=ratio(sum(r["category"] == r["truth"] for r in accepted), len(accepted)),
            teacher_calls=0, errors=errors))
    assert [(r["dataset"], r["kind"]) for r in checks] == [(d, k) for d in ("clinc150", "banking77") for k in ("system1", "baseline")]
    assert report["all_system1_point_targets_pass"] == all(r["all_point_targets_pass"] for r in report["results"] if r["kind"] == "system1")
    output = dict(scope="arithmetic and source preservation audit; no model evaluation", qualified=False,
        report_sha256=digest(path), freeze_commit=report["freeze_commit"],
        frozen_files_unchanged=len(freeze["files"]), original_frozen_files_unchanged=len(original["files"]),
        original_failed_report_sha256=digest(HERE / "results/official-test.json"), checks=checks)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
