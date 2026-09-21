"""Recompute the fixed head-comparison record; no model or original-test access."""
import json
import math
from pathlib import Path

from audit_observed_regression import digest, metrics, same
from develop import frontier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    path = HERE / "results/fused-head-development.json"
    report = json.loads(path.read_text())
    assert report["qualified"] is False and "failure" not in report
    assert report["teacher_calls"] == report["new_api_cost"] == 0 and report["os_network_denial_errno"] == 1
    freeze = json.loads((HERE / "latest-regression-freeze.json").read_text())
    for name, expected in dict(freeze["files"], **report["source_files"]).items():
        assert digest(ROOT / name) == expected, name
    expected_order = [(d, c) for d in ("clinc150", "banking77") for c in ("semantic-control", "semantic-plus-words")]
    assert [(r["dataset"], r["condition"]) for r in report["results"]] == expected_order
    data_manifest = json.loads((HERE / "manifest.json").read_text())
    matching_baseline = json.loads((HERE / "results/baseline-review-runtime.json").read_text())
    selected = json.loads((HERE / "results/direct-oos-runtime.json").read_text())
    checks = []
    for dataset in ("clinc150", "banking77"):
        source_path = ROOT / ".system1/n8n-gauntlet" / f"{dataset}-development.json"
        assert digest(source_path) == data_manifest["datasets"][dataset]["splits"]["development"]["sha256"]
        source = json.loads(source_path.read_text())
        control, fused = [r for r in report["results"] if r["dataset"] == dataset]
        for result in (control, fused):
            rows = result["outcomes"]
            assert len(rows) == len(source)
            for row, original in zip(rows, source, strict=True):
                assert row["group"] == original["group"] and row["truth"] == original["label"]
                assert row["teacherCalls"] == 0 and 0 <= row["confidence"] <= 1 and 0 <= row["margin"] <= 1
                assert math.isfinite(row["confidence"]) and math.isfinite(row["margin"])
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
            folder = HERE / "artifacts" / ("fused-" + result["folder"])
            assert digest(folder / "manifest.json") == result["candidate_sha256"]
            assert json.loads((folder / "manifest.json").read_text()) == result["manifest"]
            for name, expected in result["manifest"]["files"].items():
                assert digest(folder / name) == expected
            assert result["artifact_bytes"] == sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
            assert result["manifest"]["dimension"] == (384 if result["condition"] == "semantic-control" else 2432)
            assert result["saved_projection_replay"]["requests"] == 16 and result["saved_projection_replay"]["max_confidence_difference"] < 1e-5
        baseline = next(r for r in matching_baseline["results"] if r["dataset"] == dataset)
        assert [(r["group"], r["truth"]) for r in baseline["outcomes"]] == [(r["group"], r["truth"]) for r in control["outcomes"]]
        baseline_incumbent = next(r["selected_development_candidate"] for r in selected["comparisons"]
                                  if r["dataset"] == dataset and r["kind"] == "baseline")
        a, b = control["strict_only_metrics"]["raw_supported_accuracy"], fused["strict_only_metrics"]["raw_supported_accuracy"]
        assert report["advance_to_review_comparison"][dataset] == (b["numerator"] > a["numerator"])
        checks.append(dict(dataset=dataset, original_development_rows=len(source), raw_control=a, raw_fused=b,
            matching_labels_baseline_raw=baseline["metrics"]["raw_supported_accuracy"],
            matching_labels_baseline_source_sha256=digest(HERE / "results/baseline-review-runtime.json"),
            stronger_retained_baseline=baseline_incumbent,
            selected_baseline_extra_positive_lessons=1696 if dataset == "clinc150" else 0,
            raw_gain=b["numerator"] - a["numerator"], proceed_to_review=report["advance_to_review_comparison"][dataset]))
    print(json.dumps(dict(scope="recorded head comparison audit; not qualification or adapter measurement", qualified=False,
        report_sha256=digest(path), frozen_files_unchanged=len(freeze["files"]), declared_sources_verified=len(report["source_files"]),
        saved_artifacts_verified=4, development_outcomes_verified=sum(len(r["outcomes"]) for r in report["results"]), checks=checks), indent=2))


if __name__ == "__main__":
    main()
