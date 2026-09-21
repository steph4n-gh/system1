"""Audit the banking ablation's recorded data, artifacts and complete outcomes."""
import json
import math
from pathlib import Path

import numpy as np

from audit_observed_regression import digest, metrics, same

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    source_path, runtime_path = (HERE / "results" / name for name in ("human-only-development.json", "human-only-runtime.json"))
    source, runtime = (json.loads(p.read_text()) for p in (source_path, runtime_path))
    assert source["qualified"] is runtime["qualified"] is False and "failure" not in source
    assert source["teacher_calls"] == runtime["teacher_calls"] == source["new_api_cost"] == runtime["new_api_cost"] == 0
    assert source["os_network_denial_errno"] == runtime["os_network_denial_errno"] == 1
    assert runtime["source_report_sha256"] == digest(source_path)
    for key in ("source_revision", "protocol_sha256", "comparison_report_sha256"):
        assert source[key] == runtime[key]
    freeze = json.loads((HERE / "latest-regression-freeze.json").read_text())
    for name, expected in dict(freeze["files"], **source["source_files"]).items():
        assert digest(ROOT / name) == expected, name
    data = {}
    manifest = json.loads((HERE / "manifest.json").read_text())["datasets"]["banking77"]["splits"]
    for part in ("fit", "calibration", "development"):
        path = ROOT / ".system1/n8n-gauntlet" / f"banking77-{part}.json"
        assert digest(path) == manifest[part]["sha256"]
        data[part] = json.loads(path.read_text())
    folds = json.loads((HERE / "results/review-teaching-development.json").read_text())["datasets"]["banking77"]["fold_groups"]
    controls = json.loads((HERE / "results/polynomial-runtime.json").read_text())["results"]
    assert len(source["results"]) == len(runtime["results"]) == 2
    assert [r["kind"] for r in runtime["results"]] == ["system1", "baseline"]
    differences, total, paired_changes = [], 0, []
    for fitted, measured in zip(source["results"], runtime["results"], strict=True):
        assert fitted["dataset"] == measured["dataset"] == "banking77" and fitted["kind"] == measured["kind"]
        assert fitted["fit_groups"] == [r["group"] for r in data["fit"]]
        assert fitted["calibration_groups"] == [r["group"] for r in data["calibration"]]
        assert fitted["original_fit_rows"] == 6026 and fitted["removed_generated_rows"] == 924
        assert fitted["manifest"]["review_rows"] == 7986 and fitted["manifest"]["generated_fit_rows"] == 0
        assert fitted["evidence"]["fold_groups"] == folds and fitted["evidence"]["generated_examples_in_fold_heads"] == 0
        control = next(r for r in controls if r["dataset"] == "banking77" and r["kind"] == fitted["kind"] and r["condition"] == "quadratic")
        reproduction = fitted["evidence"]["control_reproduced"]
        assert reproduction["candidate_sha256"] == control["candidate_sha256"] and reproduction["matching_outcomes"] == 1960
        assert reproduction["max_review_difference"] < 1e-4 and reproduction["retimed"] is False
        same(reproduction["metrics"], control["metrics"])
        pairs = list(zip(measured["outcomes"], control["outcomes"], strict=True))
        assert all(a["group"] == b["group"] and a["truth"] == b["truth"] for a, b in pairs)
        paired_changes.append(dict(kind=measured["kind"],
            raw_fixed=sum(a["suggestion"] == a["truth"] and b["suggestion"] != b["truth"] for a, b in pairs),
            raw_broken=sum(a["suggestion"] != a["truth"] and b["suggestion"] == b["truth"] for a, b in pairs),
            shared_accepted_mistakes=len({r["group"] for r in measured["accepted_errors"]} & {r["group"] for r in control["accepted_errors"]})))
        assert fitted["scalar_review_preflight_max_difference"] < 1e-4
        assert len(fitted["outcomes"]) == len(measured["outcomes"]) == 1960
        for row, selected, original in zip(measured["outcomes"], fitted["outcomes"], data["development"], strict=True):
            assert row["group"] == selected["group"] == original["group"] and row["truth"] == selected["truth"] == original["label"]
            assert row["candidate"] == measured["candidate_sha256"] == fitted["manifest_sha256"]
            assert row["teacherCalls"] == 0 and "error" not in row and math.isfinite(row["latency_ms"]) and row["latency_ms"] >= 0
            assert all(row[k] == selected[k] for k in ("suggestion", "category", "needsReview"))
            differences.append(abs(row["reliability" if measured["kind"] == "system1" else "acceptance_score"] - selected["score"]))
        same(fitted["metrics"], metrics(fitted["outcomes"]))
        same(measured["metrics"], metrics(measured["outcomes"]))
        rows = measured["outcomes"]
        total += len(rows)
        same(measured["per_intent"], {label: metrics([r for r in rows if r["truth"] == label]) for label in sorted({r["truth"] for r in rows})})
        assert len(measured["per_intent"]) == 77
        assert measured["accepted_errors"] == [r for r in rows if not r["needsReview"] and r["category"] != r["truth"]]
        times = [r["latency_ms"] for r in rows]
        same(measured["latency"], dict(requests=1960, warmups=100, p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95))))
        m = measured["metrics"]
        assert m["oos_false_acceptance"]["denominator"] == 0 and m["oos_false_acceptance"]["fraction"] is None
        assert measured["development_gates"] == dict(coverage_at_least_80=m["supported_coverage"]["fraction"] >= .8,
            accepted_accuracy_at_least_99=m["accepted_accuracy"]["fraction"] >= .99,
            oos_false_acceptance_at_most_1=True, adapter_p95_below_5_ms=measured["latency"]["p95_ms"] < 5, no_runtime_errors=True)
        assert measured["selection_runtime_mismatches"] == [] and measured["errors"] == 0
        folder = HERE / "artifacts" / measured["folder"]
        assert digest(folder / "manifest.json") == measured["candidate_sha256"]
        assert json.loads((folder / "manifest.json").read_text()) == fitted["manifest"] == measured["manifest"]
        for name, expected in measured["manifest"]["files"].items():
            assert digest(folder / name) == expected
        assert measured["artifact_bytes"] == sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
        assert measured["external_encoder_bytes"] == (133804886 if measured["kind"] == "system1" else 0)
        with np.load(folder / ("scope.npz" if measured["kind"] == "system1" else "weights.npz"), allow_pickle=False) as arrays:
            assert arrays["mean"].shape == arrays["scale"].shape == (35,)
            if measured["kind"] == "system1":
                assert arrays["prototypes"].shape == (6026, 384) and arrays["class_offsets"][-1] == 6026
                assert arrays["weights"].shape == (112,)
            else:
                labels = list(arrays["labels"])
                assert len(labels) == 77 and list(arrays["fit_labels"]) == [labels.index(r["label"]) for r in data["fit"]]
                assert arrays["gate_weights"].shape == (1, 112)
    previous = json.loads((HERE / "results/fused-review-runtime.json").read_text())["comparisons"]
    assert len(previous) == len(runtime["comparisons"]) == 4
    for actual, prior in zip(runtime["comparisons"], previous, strict=True):
        assert (actual["dataset"], actual["kind"]) == (prior["dataset"], prior["kind"])
        assert actual["incumbent"] == prior["selected_development_candidate"]
        winner = actual["incumbent"]
        if actual["dataset"] == "banking77":
            measured = next(r for r in runtime["results"] if r["kind"] == actual["kind"])
            valid = all(v for k, v in measured["development_gates"].items() if k != "coverage_at_least_80")
            if valid and not measured["selection_runtime_mismatches"] and measured["metrics"]["supported_coverage"]["numerator"] > winner["metrics"]["supported_coverage"]["numerator"]:
                winner = {k: measured[k] for k in ("folder", "candidate_sha256", "metrics", "latency")}
                winner["source"] = measured["condition"]
        assert actual["selected_development_candidate"] == winner and actual["qualified"] is False
    assert runtime["response_cache"] is runtime["receipts"] is False
    assert max(differences) < 1e-4
    print(json.dumps(dict(scope="recorded original-only ablation audit; no inference or test access", qualified=False,
        source_sha256=digest(source_path), runtime_sha256=digest(runtime_path), frozen_files_unchanged=len(freeze["files"]),
        declared_sources_verified=len(source["source_files"]), outcomes_verified=total,
        original_fit_rows_per_method=6026, generated_fit_rows=0, control_outcomes_reproduced=3920,
        paired_changes=paired_changes, max_selection_runtime_review_difference=max(differences)), indent=2))


if __name__ == "__main__":
    main()
