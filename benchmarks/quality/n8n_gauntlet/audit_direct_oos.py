"""Audit recorded direct-OOS evidence without running models or opening tests."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from audit_observed_regression import digest, metrics, same

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    source_path = HERE / "results/direct-oos-development.json"
    runtime_path = HERE / "results/direct-oos-runtime.json"
    source, runtime = (json.loads(p.read_text()) for p in (source_path, runtime_path))
    assert source["qualified"] is runtime["qualified"] is False
    assert source["new_teacher_calls"] == runtime["teacher_calls"] == source["new_api_cost"] == runtime["new_api_cost"] == 0
    assert source["os_network_denial_errno"] == runtime["os_network_denial_errno"] == 1
    assert runtime["source_report_sha256"] == digest(source_path)
    for key in ("source_revision", "protocol_sha256", "comparison_report_sha256"):
        assert source[key] == runtime[key]
    assert source["protocol_sha256"] == digest(HERE / "DIRECT_OOS_PROTOCOL.md")
    assert source["comparison_report_sha256"] == digest(HERE / "results/context-runtime.json")
    freeze = json.loads((HERE / "latest-regression-freeze.json").read_text())
    historical = []
    for name, expected in dict(freeze["files"], **source["source_files"]).items():
        if name.endswith("/direct_oos_teaching.py"):
            original = subprocess.check_output(["git", "show", f'{source["source_revision"]}:{name}'], cwd=ROOT)
            assert hashlib.sha256(original).hexdigest() == expected
            historical.append(name)
        else:
            assert digest(ROOT / name) == expected, name
    recovery = runtime["recovery"]
    assert recovery["original_exit_code"] == 1 and recovery["rerun_requests"] == 0
    assert recovery["fitting_functions_unchanged"]
    assert recovery["original_log_sha256"] == digest(HERE / "results/direct-oos-runtime-failure.log")
    assert recovery["original_journal_sha256"] == digest(ROOT / ".system1/n8n-gauntlet/direct-oos-development/runtime.jsonl")
    assert runtime["process_import_preflight_ms"] is None
    order = [(kind, condition) for kind in ("system1", "baseline") for condition in ("positive-control", "direct-oos")]
    assert [(r["kind"], r["condition"]) for r in source["results"]] == order
    assert [(r["kind"], r["condition"]) for r in runtime["results"]] == order
    data_path = ROOT / ".system1/n8n-gauntlet/clinc150-development.json"
    manifest = json.loads((HERE / "manifest.json").read_text())
    assert digest(data_path) == manifest["datasets"]["clinc150"]["splits"]["development"]["sha256"]
    pinned = json.loads(data_path.read_text())
    original_folds = json.loads((HERE / "results/review-teaching-development.json").read_text())["datasets"]["clinc150"]["fold_groups"]
    checks = []
    for fit, run in zip(source["results"], runtime["results"], strict=True):
        for key in ("dataset", "kind", "condition", "folder", "manifest"):
            assert fit[key] == run[key]
        assert fit["dataset"] == "clinc150"
        assert fit["original_fit_rows"] == 12019 and fit["new_supported"] == 1696
        assert fit["new_unsupported_intent"] == (674 if fit["condition"] == "direct-oos" else 0)
        assert fit["earlier_generated_supported"] == fit["synthetic_unsupported_review"] == fit["generated_rows_in_fold_heads"] == 0
        assert fit["wikipedia_review_rows"] == 2000 and fit["fold_groups"] == original_folds
        assert run["manifest"]["fit_rows"] == 13715 + fit["new_unsupported_intent"]
        assert run["manifest"]["review_rows"] == 17039
        rows, differences = run["outcomes"], []
        for row, selected, original in zip(rows, fit["outcomes"], pinned, strict=True):
            assert row["group"] == selected["group"] == original["group"]
            assert row["truth"] == selected["truth"] == original["label"]
            assert row["candidate"] == run["candidate_sha256"] == fit["manifest_sha256"]
            assert row["teacherCalls"] == 0 and "error" not in row
            assert math.isfinite(row["latency_ms"]) and row["latency_ms"] >= 0
            assert all(row[k] == selected[k] for k in ("suggestion", "category", "needsReview"))
            differences.append(abs(row["reliability" if run["kind"] == "system1" else "acceptance_score"] - selected["score"]))
        same(run["metrics"], metrics(rows))
        same(fit["metrics"], metrics(fit["outcomes"]))
        same(run["per_intent"], {label: metrics([r for r in rows if r["truth"] == label]) for label in sorted({r["truth"] for r in rows})})
        assert run["accepted_errors"] == [r for r in rows if not r["needsReview"] and r["category"] != r["truth"]]
        times = [r["latency_ms"] for r in rows]
        same(run["latency"], dict(requests=3095, warmups=100, p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95))))
        m = run["metrics"]
        assert run["development_gates"] == dict(coverage_at_least_80=m["supported_coverage"]["fraction"] >= .8,
            accepted_accuracy_at_least_99=m["accepted_accuracy"]["fraction"] >= .99,
            oos_false_acceptance_at_most_1=m["oos_false_acceptance"]["fraction"] <= .01,
            adapter_p95_below_5_ms=run["latency"]["p95_ms"] < 5, no_runtime_errors=True)
        assert run["errors"] == 0 and run["selection_runtime_mismatches"] == [] and run["qualified"] is False
        assert run["load_ms"] is None
        local = ROOT / ".system1/n8n-gauntlet/direct-oos-development" / run["folder"]
        public = HERE / "artifacts" / run["folder"]
        assert digest(local / "manifest.json") == digest(public / "manifest.json") == run["candidate_sha256"]
        assert json.loads((local / "manifest.json").read_text()) == run["manifest"]
        for name, expected in run["manifest"]["files"].items():
            assert digest(local / name) == expected
            if (public / name).exists():
                assert digest(public / name) == expected
        assert run["artifact_bytes"] == sum(p.stat().st_size for p in local.iterdir() if p.is_file())
        assert run["external_encoder_bytes"] == sum(run["manifest"].get("encoder", {}).get(k, 0) for k in ("model_bytes", "tokenizer_bytes"))
        checks.append(dict(kind=run["kind"], condition=run["condition"], rows=len(rows),
            source_labels_groups_routing_metrics_and_local_artifacts_verified=True,
            complete_artifact_published=all((public / name).exists() for name in run["manifest"]["files"]),
            max_selection_runtime_score_difference=max(differences)))
    control = source["results"][0]["control_reconstruction"]
    assert control["original_routing_matches"] and control["max_score_difference"] < 1e-10
    assert control["reference_sha256"] == digest(HERE / "results/boundary-development.json")
    prior = json.loads((HERE / "results/context-runtime.json").read_text())
    for current, previous in zip(runtime["comparisons"], prior["comparisons"], strict=True):
        assert (current["dataset"], current["kind"]) == (previous["dataset"], previous["kind"])
        assert current["incumbent"] == previous["selected_development_candidate"]
        winner = current["incumbent"]
        for candidate in runtime["results"]:
            if (candidate["dataset"], candidate["kind"]) != (current["dataset"], current["kind"]):
                continue
            eligible = all(v for k, v in candidate["development_gates"].items() if k != "coverage_at_least_80")
            if eligible and candidate["metrics"]["supported_coverage"]["numerator"] > winner["metrics"]["supported_coverage"]["numerator"]:
                winner = {key: candidate[key] for key in ("folder", "candidate_sha256", "metrics", "latency")}
                winner["source"] = candidate["condition"]
        assert current["selected_development_candidate"] == winner and current["qualified"] is False
    assert runtime["response_cache"] is runtime["receipts"] is False
    print(json.dumps(dict(scope="recorded development audit, not qualification or inference", qualified=False,
        source_sha256=digest(source_path), runtime_sha256=digest(runtime_path), frozen_files_unchanged=len(freeze["files"]),
        declared_source_files_verified=len(source["source_files"]), historical_sources_verified_from_git=historical,
        recovered_requests=12380, rerun_timing_requests=0, checks=checks), indent=2))


if __name__ == "__main__":
    main()
