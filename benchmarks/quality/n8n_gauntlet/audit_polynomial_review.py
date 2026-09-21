"""Audit recorded experiment 2f development outcomes without executing a model."""
import ast
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from audit_observed_regression import digest, metrics, same

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    source_path = HERE / "results/polynomial-development.json"
    runtime_path = HERE / "results/polynomial-runtime.json"
    source = json.loads(source_path.read_text())
    runtime = json.loads(runtime_path.read_text())
    prior = json.loads((HERE / "results/boundary-runtime.json").read_text())
    assert source["qualified"] is runtime["qualified"] is False
    assert source["teacher_calls"] == runtime["teacher_calls"] == 0
    assert runtime["response_cache"] is runtime["receipts"] is False
    assert source["os_network_denial_errno"] == runtime["os_network_denial_errno"] == 1
    assert source["protocol_sha256"] == digest(HERE / "POLYNOMIAL_REVIEW_PROTOCOL.md")
    assert source["comparison_report_sha256"] == digest(HERE / "results/boundary-runtime.json")
    revision = source["source_revision"]
    old_source = subprocess.check_output(["git", "show", f"{revision}:benchmarks/quality/n8n_gauntlet/polynomial_runtime.py"], cwd=ROOT, text=True)
    # The post-run CI repair moved these functions without changing their bodies.
    old_functions = [ast.dump(node) for node in ast.parse(old_source).body if isinstance(node, ast.FunctionDef)]
    new_functions = [ast.dump(node) for node in ast.parse((HERE / "polynomial_scores.py").read_text()).body if isinstance(node, ast.FunctionDef)]
    assert old_functions == new_functions
    freeze = json.loads((HERE / "observed-regression-freeze.json").read_text())
    for name, expected in freeze["files"].items():
        assert digest(ROOT / name) == expected, name
    manifest = json.loads((HERE / "manifest.json").read_text())
    identities = [(d, k, c) for d in ("clinc150", "banking77") for k in ("system1", "baseline") for c in ("linear-control", "quadratic")]
    assert [(r["dataset"], r["kind"], r["condition"]) for r in runtime["results"]] == identities
    checks = []
    for measured, fitted in zip(runtime["results"], source["results"], strict=True):
        dataset, kind, condition = (measured[key] for key in ("dataset", "kind", "condition"))
        for key in ("dataset", "kind", "condition", "folder", "manifest"):
            assert measured[key] == fitted[key]
        rows = measured["outcomes"]
        path = ROOT / f".system1/n8n-gauntlet/{dataset}-development.json"
        assert digest(path) == manifest["datasets"][dataset]["splits"]["development"]["sha256"]
        pinned = json.loads(path.read_text())
        differences = []
        for row, selection, original in zip(rows, fitted["outcomes"], pinned, strict=True):
            assert row["group"] == selection["group"] == original["group"]
            assert row["truth"] == selection["truth"] == original["label"]
            assert row["candidate"] == measured["candidate_sha256"] == fitted["manifest_sha256"]
            assert row["teacherCalls"] == 0 and "error" not in row
            assert math.isfinite(row["latency_ms"]) and row["latency_ms"] >= 0
            assert all(row[key] == selection[key] for key in ("suggestion", "needsReview", "category"))
            differences.append(abs(row["reliability" if kind == "system1" else "acceptance_score"] - selection["score"]))
        assert measured["selection_runtime_mismatches"] == [] and measured["errors"] == 0
        same(measured["metrics"], metrics(rows))
        same(fitted["metrics"], metrics(fitted["outcomes"]))
        same(measured["per_intent"], {label: metrics([r for r in rows if r["truth"] == label]) for label in sorted({r["truth"] for r in rows})})
        assert measured["accepted_errors"] == [r for r in rows if not r["needsReview"] and r["category"] != r["truth"]]
        times = [r["latency_ms"] for r in rows]
        same(measured["latency"], dict(requests=len(rows), warmups=100, p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95))))
        m = measured["metrics"]
        gates = dict(coverage_at_least_80=m["supported_coverage"]["fraction"] >= .8,
            accepted_accuracy_at_least_99=m["accepted_accuracy"]["fraction"] >= .99,
            oos_false_acceptance_at_most_1=m["oos_false_acceptance"]["fraction"] is None or m["oos_false_acceptance"]["fraction"] <= .01,
            adapter_p95_below_5_ms=measured["latency"]["p95_ms"] < 5, no_runtime_errors=True)
        assert gates == measured["development_gates"] and measured["qualified"] is False
        child = HERE / "artifacts" / ("polynomial-" + measured["folder"]) / "manifest.json"
        assert digest(child) == measured["candidate_sha256"]
        assert json.loads(child.read_text()) == measured["manifest"]
        parent = ROOT / measured["manifest"]["base_artifact"]
        assert digest(parent / "manifest.json") == measured["manifest"]["parent_manifest_sha256"]
        for name, expected in measured["manifest"]["files"].items():
            assert digest(parent / name) == expected
        assert child.stat().st_size == measured["additional_review_bytes"]
        assert measured["artifact_bytes"] == child.stat().st_size + sum(p.stat().st_size for p in parent.iterdir() if p.is_file())
        assert measured["external_encoder_bytes"] == sum(measured["manifest"].get("encoder", {}).get(k, 0) for k in ("model_bytes", "tokenizer_bytes"))
        assert fitted["evidence"]["generated_examples_in_fold_heads"] == 0
        checks.append(dict(dataset=dataset, kind=kind, condition=condition, original_development_rows=len(rows),
            all_groups_labels_routing_and_counts_preserved=True, intervals_and_latencies_recomputed=True,
            public_child_and_parent_hashes_verified=True, max_selection_runtime_score_difference=max(differences),
            teacher_calls=0, errors=0))
    for comparison, previous in zip(runtime["comparisons"], prior["comparisons"], strict=True):
        assert comparison["incumbent"] == previous["selected_development_candidate"]
        winner = comparison["incumbent"]
        for measured in runtime["results"]:
            if (measured["dataset"], measured["kind"]) != (comparison["dataset"], comparison["kind"]):
                continue
            valid = all(value for key, value in measured["development_gates"].items() if key != "coverage_at_least_80")
            if valid and measured["metrics"]["supported_coverage"]["numerator"] > winner["metrics"]["supported_coverage"]["numerator"]:
                winner = {key: measured[key] for key in ("folder", "candidate_sha256", "metrics", "latency")}
                winner["source"] = measured["condition"]
        assert comparison["selected_development_candidate"] == winner
        assert comparison["qualified"] is False and comparison["incumbent_timings_from_prior_run"] is True
    print(json.dumps(dict(scope="record arithmetic, source and artifact audit; no model evaluation", qualified=False,
        original_run_source_revision=revision, source_sha256=digest(source_path), runtime_sha256=digest(runtime_path),
        frozen_files_unchanged=len(freeze["files"]), numerical_functions_unchanged_after_ci_import_repair=True,
        stronger_prior_baseline_comparisons_preserved=True, checks=checks), indent=2))


if __name__ == "__main__":
    main()
