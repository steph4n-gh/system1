"""Recompute experiment 2g evidence without running a model or opening tests."""
import json
import math
from pathlib import Path

import numpy as np

from audit_observed_regression import digest, metrics, same

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    source_path, runtime_path = (HERE / "results" / name for name in ("context-development.json", "context-runtime.json"))
    source, runtime = (json.loads(path.read_text()) for path in (source_path, runtime_path))
    assert source["qualified"] is runtime["qualified"] is False
    assert source["teacher_calls"] == runtime["teacher_calls"] == source["new_api_cost"] == 0
    assert source["os_network_denial_errno"] == runtime["os_network_denial_errno"] == 1
    assert runtime["source_report_sha256"] == digest(source_path)
    for key in ("source_revision", "protocol_sha256", "comparison_report_sha256"):
        assert runtime[key] == source[key]
    assert source["protocol_sha256"] == digest(HERE / "CONTEXT_REVIEW_PROTOCOL.md")
    assert source["comparison_report_sha256"] == digest(HERE / "results/polynomial-runtime.json")
    freeze = json.loads((HERE / "observed-regression-freeze.json").read_text())
    for name, expected in dict(freeze["files"], **source["source_files"]).items():
        assert digest(ROOT / name) == expected, name
    assert len(source["control_checks"]) == 4
    for check in source["control_checks"]:
        assert check["original_lineage_identical"] and check["predictions_match"] and check["max_score_difference"] <= 1e-10
    expected_order = [(d, k) for d in ("clinc150", "banking77") for k in ("system1", "baseline")]
    assert [(r["dataset"], r["kind"]) for r in runtime["results"]] == expected_order
    manifest = json.loads((HERE / "manifest.json").read_text())
    checks = []
    for fit, run in zip(source["results"], runtime["results"], strict=True):
        for key in ("dataset", "kind", "condition", "folder", "manifest"):
            assert fit[key] == run[key]
        path = ROOT / f'.system1/n8n-gauntlet/{run["dataset"]}-development.json'
        assert digest(path) == manifest["datasets"][run["dataset"]]["splits"]["development"]["sha256"]
        rows, pinned = run["outcomes"], json.loads(path.read_text())
        differences = []
        for row, selected, original in zip(rows, fit["outcomes"], pinned, strict=True):
            assert row["group"] == selected["group"] == original["group"] and row["truth"] == selected["truth"] == original["label"]
            assert row["candidate"] == run["candidate_sha256"] == fit["manifest_sha256"]
            assert row["teacherCalls"] == 0 and "error" not in row and math.isfinite(row["latency_ms"]) and row["latency_ms"] >= 0
            assert all(row[k] == selected[k] for k in ("suggestion", "category", "needsReview"))
            differences.append(abs(row["reliability" if run["kind"] == "system1" else "acceptance_score"] - selected["score"]))
        same(run["metrics"], metrics(rows))
        same(fit["metrics"], metrics(fit["outcomes"]))
        same(run["per_intent"], {label: metrics([r for r in rows if r["truth"] == label]) for label in sorted({r["truth"] for r in rows})})
        assert run["accepted_errors"] == [r for r in rows if not r["needsReview"] and r["category"] != r["truth"]]
        times = [r["latency_ms"] for r in rows]
        same(run["latency"], dict(requests=len(rows), warmups=100, p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95))))
        m = run["metrics"]
        assert run["development_gates"] == dict(coverage_at_least_80=m["supported_coverage"]["fraction"] >= .8,
            accepted_accuracy_at_least_99=m["accepted_accuracy"]["fraction"] >= .99,
            oos_false_acceptance_at_most_1=m["oos_false_acceptance"]["fraction"] is None or m["oos_false_acceptance"]["fraction"] <= .01,
            adapter_p95_below_5_ms=run["latency"]["p95_ms"] < 5, no_runtime_errors=True)
        assert run["errors"] == 0 and run["selection_runtime_mismatches"] == [] and run["qualified"] is False
        child = HERE / "artifacts" / run["folder"] / "manifest.json"
        assert digest(child) == run["candidate_sha256"] and json.loads(child.read_text()) == run["manifest"]
        parent = ROOT / run["manifest"]["base_artifact"]
        assert digest(parent / "manifest.json") == run["manifest"]["parent_manifest_sha256"]
        for name, expected in run["manifest"]["files"].items():
            assert digest(parent / name) == expected
        assert run["additional_review_bytes"] == child.stat().st_size
        assert run["artifact_bytes"] == child.stat().st_size + sum(p.stat().st_size for p in parent.iterdir() if p.is_file())
        assert run["external_encoder_bytes"] == sum(run["manifest"].get("encoder", {}).get(k, 0) for k in ("model_bytes", "tokenizer_bytes"))
        assert fit["evidence"]["context_rows"] == fit["evidence"]["review_rows"]
        assert fit["evidence"]["context_dimension"] == run["manifest"]["context_dimension"] == len(run["manifest"]["context_weights"])
        checks.append(dict(dataset=run["dataset"], kind=run["kind"], rows=len(rows),
            source_labels_groups_routing_artifacts_and_metrics_verified=True, max_selection_runtime_score_difference=max(differences)))
    prior = json.loads((HERE / "results/polynomial-runtime.json").read_text())
    for current, previous in zip(runtime["comparisons"], prior["comparisons"], strict=True):
        assert current["incumbent"] == previous["selected_development_candidate"]
        winner = current["incumbent"]
        candidate = next(r for r in runtime["results"] if (r["dataset"], r["kind"]) == (current["dataset"], current["kind"]))
        eligible = all(v for k, v in candidate["development_gates"].items() if k != "coverage_at_least_80")
        if eligible and candidate["metrics"]["supported_coverage"]["numerator"] > winner["metrics"]["supported_coverage"]["numerator"]:
            winner = {key: candidate[key] for key in ("folder", "candidate_sha256", "metrics", "latency")}
            winner["source"] = candidate["condition"]
        assert current["selected_development_candidate"] == winner and current["qualified"] is False
    assert runtime["response_cache"] is runtime["receipts"] is False
    print(json.dumps(dict(scope="development arithmetic, lineage and saved-artifact audit; no model evaluation", qualified=False,
        source_sha256=digest(source_path), runtime_sha256=digest(runtime_path), frozen_files_unchanged=len(freeze["files"]),
        recorded_source_files_unchanged=len(source["source_files"]), stronger_incumbents_preserved=True, checks=checks), indent=2))


if __name__ == "__main__":
    main()
