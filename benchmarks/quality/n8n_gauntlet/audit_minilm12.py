"""Recompute MiniLM12 development records without fitting or model inference."""
import json
import math
from pathlib import Path

import numpy as np

from audit_observed_regression import digest, metrics, same
from develop import frontier, load_splits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    path = HERE / "results/minilm12-development.json"
    report = json.loads(path.read_text())
    assert report["qualified"] is False and "failure" not in report
    assert report["teacher_calls"] == report["new_api_cost"] == 0
    assert report["os_network_denial_errno"] == 1 and report["response_cache"] is report["receipts"] is False
    frozen = json.loads((HERE / "latest-regression-freeze.json").read_text())
    for name, expected in dict(frozen["files"], **report["source_files"]).items():
        assert digest(ROOT / name) == expected, name
    assets = json.loads((HERE / "minilm12-assets.json").read_text())
    controls = {r["dataset"]: r for r in json.loads((HERE / "results/fused-head-development.json").read_text())["results"]
                if r["condition"] == "semantic-control"}
    baselines = json.loads((HERE / "results/baseline-review-runtime.json").read_text())
    incumbents = json.loads((HERE / "results/fused-review-runtime.json").read_text())["comparisons"]
    assert len(report["results"]) == len(report["comparisons"]) == 2
    checks = []
    for result, comparison in zip(report["results"], report["comparisons"], strict=True):
        dataset, manifest = result["dataset"], result["manifest"]
        data, control = load_splits(dataset), controls[dataset]
        assert dataset == comparison["dataset"] == manifest["dataset"]
        assert manifest["control_manifest_sha256"] == comparison["control_manifest_sha256"] == control["candidate_sha256"]
        assert manifest["fit_rows"] == control["manifest"]["fit_rows"]
        assert manifest["original_fit_rows"] == len(data["fit"])
        assert manifest["earlier_generated_supported"] == (924 if dataset == "banking77" else 0)
        assert manifest["dimension"] == 384 and manifest["encoder"]["name"] == "minilm12"
        assert manifest["encoder"]["threads"] == 2 and manifest["encoder"]["pooling"] == "mean"
        assert manifest["assets_manifest_sha256"] == digest(HERE / "minilm12-assets.json")
        for name, expected in assets["files"].items():
            raw = ROOT / ".system1/n8n-gauntlet/minilm12" / name
            assert raw.stat().st_size == expected["bytes"] and digest(raw) == expected["sha256"]
        rows = result["outcomes"]
        for row, original in zip(rows, data["development"], strict=True):
            assert row["group"] == original["group"] and row["truth"] == original["label"] and row["teacherCalls"] == 0
            assert all(math.isfinite(row[k]) and 0 <= row[k] <= 1 for k in ("confidence", "margin"))
            assert math.isfinite(row["latency_ms"]) and row["latency_ms"] >= 0
            review = not row["strict_eligible"] or row["suggestion"] == "oos"
            assert row["needsReview"] == review and row["category"] == (None if review else row["suggestion"])
        same(result["strict_only_metrics"], metrics(rows))
        assert result["raw_oos_predictions"] == sum(r["truth"] == r["suggestion"] == "oos" for r in rows)
        assert result["supported_predicted_oos"] == sum(r["truth"] != "oos" and r["suggestion"] == "oos" for r in rows)
        truth, pred = [r["truth"] for r in rows], [r["suggestion"] for r in rows]
        for score in ("confidence", "margin"):
            same(result["probability_margin_frontiers"][score], frontier(truth, pred, [r[score] for r in rows]))
            same(result["strict_probability_margin_frontiers"][score], frontier(truth, pred, [r[score] for r in rows],
                eligible=[r["strict_eligible"] for r in rows]))
        times = [r["latency_ms"] for r in rows]
        same(result["saved_head_latency"], dict(requests=len(rows), warmups=100,
            p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95))))
        best = lambda r: max(r["probability_margin_frontiers"][k]["best_development_coverage_at_quality_targets"]["coverage"] for k in ("confidence", "margin"))
        gain = result["strict_only_metrics"]["raw_supported_accuracy"]["numerator"] - control["strict_only_metrics"]["raw_supported_accuracy"]["numerator"]
        assert comparison["raw_gain"] == gain and comparison["control_best_frontier"] == best(control) and comparison["new_best_frontier"] == best(result)
        assert comparison["saved_head_p95_below_5_ms"] == (result["saved_head_latency"]["p95_ms"] < 5)
        assert result["errors"] == 0 and result["saved_replay"]["requests"] == 16 and result["saved_replay"]["max_confidence_difference"] < 1e-5
        assert comparison["advance_to_review_comparison"] == (gain >= 10 and best(result) > best(control) and result["saved_head_latency"]["p95_ms"] < 5)
        folder = HERE / "artifacts" / f"minilm12-{dataset}"
        assert digest(folder / "manifest.json") == result["candidate_sha256"]
        assert json.loads((folder / "manifest.json").read_text()) == manifest
        for name, expected in manifest["files"].items():
            assert digest(folder / name) == expected
        assert result["artifact_bytes"] == sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
        assert result["external_encoder_bytes"] == 34584885
        baseline = next(r for r in baselines["results"] if r["dataset"] == dataset)
        assert [(r["group"], r["truth"]) for r in baseline["outcomes"]] == [(r["group"], r["truth"]) for r in rows]
        checks.append(dict(dataset=dataset, outcomes=len(rows), comparison=comparison,
            matching_labels_baseline_raw=baseline["metrics"]["raw_supported_accuracy"],
            retained_incumbents=[r["selected_development_candidate"] for r in incumbents if r["dataset"] == dataset],
            baseline_extra_supported_teaching=1696 if dataset == "clinc150" else 0))
    print(json.dumps(dict(scope="record audit only; no model inference or complete adapter qualification", qualified=False,
        source_sha256=digest(path), frozen_files_unchanged=len(frozen["files"]), declared_sources_verified=len(report["source_files"]),
        saved_artifacts_verified=2, outcomes_verified=sum(c["outcomes"] for c in checks), checks=checks), indent=2))


if __name__ == "__main__":
    main()
