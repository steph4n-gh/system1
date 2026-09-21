"""Verify full recorded regression counts and original labels without models."""
import json
import math
from pathlib import Path

import numpy as np

from audit_observed_regression import digest, metrics, ratio, same

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    path = HERE / "results/latest-regression.json"
    report = json.loads(path.read_text())
    freeze_path = HERE / "latest-regression-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    original = json.loads((HERE / "freeze.json").read_text())
    manifest = json.loads((HERE / "manifest.json").read_text())
    assert report["qualified"] is report["independent_confirmation"] is False
    assert report["freeze_sha256"] == digest(freeze_path)
    assert report["criteria"] == freeze["criteria"] == original["criteria"]
    for name, expected in freeze["files"].items():
        assert digest(ROOT / name) == expected, name
    checks = []
    for result in report["results"]:
        dataset, kind = result["dataset"], result["kind"]
        source = ROOT / f".system1/n8n-gauntlet/{dataset}-test.json"
        expected = manifest["datasets"][dataset]["splits"]["test"]
        assert digest(source) == expected["sha256"]
        rows, pinned = result["outcomes"], json.loads(source.read_text())
        assert len(rows) == len(pinned) == expected["rows"]
        for i, (row, raw) in enumerate(zip(rows, pinned, strict=True)):
            assert row["index"] == i and row["group"] == raw["group"]
            assert row["text"] == raw["prompt"] and row["truth"] == raw["label"]
            assert row["candidate"] == result["candidate_sha256"] and row["teacherCalls"] == 0
            assert math.isfinite(row["latency_ms"]) and row["latency_ms"] >= 0
        same(result["metrics"], metrics(rows))
        same(result["per_intent"], {label: metrics([r for r in rows if r["truth"] == label]) for label in sorted({r["truth"] for r in rows})})
        assert result["accepted_errors"] == [r for r in rows if not r["needsReview"] and r["category"] != r["truth"]]
        times = [r["latency_ms"] for r in rows]
        same(result["latency"], dict(requests=len(rows), warmups=100, p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)), max_ms=max(times)))
        errors, m = sum("error" in r for r in rows), result["metrics"]
        gates = dict(coverage_at_least_80=m["supported_coverage"]["fraction"] >= .8,
            accepted_accuracy_at_least_99=m["accepted_accuracy"]["fraction"] >= .99,
            oos_false_acceptance_at_most_1=m["oos_false_acceptance"]["fraction"] is None or m["oos_false_acceptance"]["fraction"] <= .01,
            adapter_p95_below_5_ms=result["latency"]["p95_ms"] < 5, no_runtime_errors=errors == 0)
        assert result["point_gates"] == gates and result["all_point_targets_pass"] == all(gates.values())
        assert result["errors"] == errors and result["teacher_calls"] == 0 and result["qualified"] is False
        assert result["response_cache"] is result["receipts"] is False
        folder = HERE / "artifacts" / freeze["candidates"][dataset][kind]
        assert folder.name == result["folder"] and digest(folder / "manifest.json") == result["candidate_sha256"]
        assert json.loads((folder / "manifest.json").read_text()) == result["manifest"]
        required = [folder]
        if "parent_manifest_sha256" in result["manifest"]:
            parent = ROOT / result["manifest"]["base_artifact"]
            assert digest(parent / "manifest.json") == result["manifest"]["parent_manifest_sha256"]
            required.append(parent)
        assert result["artifact_bytes"] == sum(p.stat().st_size for directory in required for p in directory.iterdir() if p.is_file())
        assert result["external_encoder_bytes"] == sum(result["manifest"].get("encoder", {}).get(key, 0) for key in ("model_bytes", "tokenizer_bytes"))
        accepted = [r for r in rows if not r["needsReview"]]
        checks.append(dict(dataset=dataset, kind=kind, original_rows_preserved=len(rows),
            original_texts_groups_labels_preserved=True, all_counts_intervals_and_timing_percentiles_recomputed=True,
            public_artifacts_and_sizes_verified=True, all_accepted_accuracy_including_unfamiliar=ratio(sum(r["category"] == r["truth"] for r in accepted), len(accepted)),
            teacher_calls=0, runtime_errors=errors))
    assert [(r["dataset"], r["kind"]) for r in checks] == [(d, k) for d in ("clinc150", "banking77") for k in ("system1", "baseline")]
    assert report["all_system1_point_targets_pass"] == all(r["all_point_targets_pass"] for r in report["results"] if r["kind"] == "system1")
    assert report["teacher_calls"] == report["new_api_cost"] == 0 and report["os_network_denial_errno"] == 1
    print(json.dumps(dict(scope="arithmetic, original-source and saved-artifact audit; no model evaluation", qualified=False,
        report_sha256=digest(path), freeze_sha256=digest(freeze_path), freeze_commit=report["freeze_commit"],
        frozen_files_unchanged=len(freeze["files"]), original_failed_report_sha256=digest(HERE / "results/official-test.json"),
        prior_observed_failure_sha256=digest(HERE / "results/observed-regression.json"), checks=checks), indent=2))


if __name__ == "__main__":
    main()
