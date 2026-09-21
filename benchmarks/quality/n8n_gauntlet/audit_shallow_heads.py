"""Audit experiment-2l records and fitted assets without model inference."""
import json
import math
from pathlib import Path

import numpy as np

from audit_observed_regression import digest, metrics, same
from develop import frontier, load_splits
from shallow_features import ShallowFeatures

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    path = HERE / "results/shallow-head-development.json"
    report = json.loads(path.read_text())
    assert report["qualified"] is False and "failure" not in report
    assert report["teacher_calls"] == report["new_api_cost"] == 0 and report["os_network_denial_errno"] == 1
    freeze = json.loads((HERE / "latest-regression-freeze.json").read_text())
    for name, expected in dict(freeze["files"], **report["source_files"]).items():
        assert digest(ROOT / name) == expected, name
    controls = {r["dataset"]: r for r in json.loads((HERE / "results/fused-head-development.json").read_text())["results"]
                if r["condition"] == "semantic-control"}
    baselines = json.loads((HERE / "results/baseline-review-runtime.json").read_text())
    incumbents = json.loads((HERE / "results/fused-review-runtime.json").read_text())["comparisons"]
    assert len(report["results"]) == len(report["teaching"]) == len(report["comparisons"]) == 2
    checks = []
    for result, teaching, comparison in zip(report["results"], report["teaching"], report["comparisons"], strict=True):
        dataset = result["dataset"]
        assert dataset == teaching["dataset"] == comparison["dataset"]
        data = load_splits(dataset)
        earlier = json.loads((HERE / "results/contrast-lessons.json").read_text())["lessons"] if dataset == "banking77" else []
        augmented = data["fit"] + earlier
        assert teaching["fit_groups"] == [r["group"] for r in augmented]
        assert teaching["labels"] == sorted({r["label"] for r in augmented})
        assert teaching["epochs"] == len(teaching["loss_curve"]) == 100
        assert all(math.isfinite(v) and v >= 0 for v in teaching["loss_curve"])
        assert teaching["loss"] == teaching["loss_curve"][-1]
        for name, expected in dict(hidden_layer_sizes=[128], activation="relu", solver="adam", alpha=.01,
                batch_size=128, learning_rate_init=.001, max_iter=100, shuffle=True, random_state=20260921,
                tol=0., early_stopping=False, n_iter_no_change=101).items():
            assert teaching["parameters"][name] == expected
        control = controls[dataset]
        assert result["manifest"]["control_manifest_sha256"] == comparison["control_manifest_sha256"] == control["candidate_sha256"]
        assert result["manifest"]["encoder"] == control["manifest"]["encoder"]
        assert result["manifest"]["fit_rows"] == len(augmented) and result["manifest"]["dimension"] == 512
        rows = result["outcomes"]
        for row, source, prior in zip(rows, data["development"], control["outcomes"], strict=True):
            assert row["group"] == source["group"] == prior["group"] and row["truth"] == source["label"] == prior["truth"]
            assert row["teacherCalls"] == 0
            assert all(math.isfinite(row[k]) and 0 <= row[k] <= 1 for k in ("confidence", "margin"))
            review = not row["strict_eligible"] or row["suggestion"] == "oos"
            assert row["needsReview"] == review and row["category"] == (None if review else row["suggestion"])
        same(result["strict_only_metrics"], metrics(rows))
        truth, pred = [r["truth"] for r in rows], [r["suggestion"] for r in rows]
        for score in ("confidence", "margin"):
            same(result["probability_margin_frontiers"][score], frontier(truth, pred, [r[score] for r in rows]))
            same(result["strict_probability_margin_frontiers"][score], frontier(truth, pred, [r[score] for r in rows],
                eligible=[r["strict_eligible"] for r in rows]))
        assert result["raw_oos_predictions"] == sum(r["truth"] == r["suggestion"] == "oos" for r in rows)
        assert result["supported_predicted_oos"] == sum(r["truth"] != "oos" and r["suggestion"] == "oos" for r in rows)
        best = lambda r: max(r["probability_margin_frontiers"][k]["best_development_coverage_at_quality_targets"]["coverage"] for k in ("confidence", "margin"))
        gain = result["strict_only_metrics"]["raw_supported_accuracy"]["numerator"] - control["strict_only_metrics"]["raw_supported_accuracy"]["numerator"]
        assert comparison["raw_gain"] == gain and comparison["control_best_frontier"] == best(control) and comparison["new_best_frontier"] == best(result)
        assert comparison["advance_to_review_comparison"] == (gain >= 10 and best(result) > best(control))
        folder = HERE / "artifacts" / f"shallow-{dataset}"
        assert digest(folder / "manifest.json") == result["candidate_sha256"]
        assert json.loads((folder / "manifest.json").read_text()) == result["manifest"]
        for name, expected in result["manifest"]["files"].items():
            assert digest(folder / name) == expected
        assert teaching["transform_file_sha256"] == digest(folder / "transform.npz")
        with np.load(folder / "transform.npz", allow_pickle=False) as params:
            assert params["weights"].shape == (384, 128) and params["bias"].shape == (128,)
            assert params["teaching_output_weights"].shape == (128, len(teaching["labels"]))
            assert params["teaching_output_bias"].shape == (len(teaching["labels"]),)
            assert all(np.isfinite(params[k]).all() for k in params.files)
        class IdentityOnlyEncoder:
            dimension = 384
            def projector_digest(self):
                import hashlib
                return hashlib.sha256(json.dumps(result["manifest"]["encoder"], sort_keys=True).encode()).hexdigest()
        transform = ShallowFeatures.load(IdentityOnlyEncoder(), folder / "transform.npz")
        assert transform.projector_digest() == result["manifest"]["projector_digest"]
        assert result["artifact_bytes"] == sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
        assert result["hidden_export_preflight"]["requests"] == 32 and result["hidden_export_preflight"]["max_difference"] < 1e-5
        assert result["saved_projection_replay"]["requests"] == 16 and result["saved_projection_replay"]["max_confidence_difference"] < 1e-5
        baseline = next(r for r in baselines["results"] if r["dataset"] == dataset)
        assert [(r["group"], r["truth"]) for r in baseline["outcomes"]] == [(r["group"], r["truth"]) for r in rows]
        checks.append(dict(dataset=dataset, outcomes=len(rows), comparison=comparison,
            matching_labels_baseline_raw=baseline["metrics"]["raw_supported_accuracy"],
            retained_incumbents=[r["selected_development_candidate"] for r in incumbents if r["dataset"] == dataset],
            baseline_extra_supported_teaching=1696 if dataset == "clinc150" else 0))
    print(json.dumps(dict(scope="record audit; no model inference, latency measurement or test access", qualified=False,
        source_sha256=digest(path), frozen_files_unchanged=len(freeze["files"]), declared_sources_verified=len(report["source_files"]),
        saved_artifacts_verified=2, new_development_outcomes_verified=sum(c["outcomes"] for c in checks), checks=checks), indent=2))


if __name__ == "__main__":
    main()
