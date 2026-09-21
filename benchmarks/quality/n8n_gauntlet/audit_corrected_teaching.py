"""Audit correction membership, original controls and complete saved responses."""
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from audit_observed_regression import digest, metrics, same

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    source_path, runtime_path = (HERE / "results" / name for name in ("corrected-teaching-development.json", "corrected-teaching-runtime.json"))
    source, runtime = (json.loads(p.read_text()) for p in (source_path, runtime_path))
    assert source["qualified"] is runtime["qualified"] is source["independent_annotations"] is False and "failure" not in source
    assert source["teacher_calls"] == runtime["teacher_calls"] == source["new_api_cost"] == runtime["new_api_cost"] == 0
    assert source["os_network_denial_errno"] == runtime["os_network_denial_errno"] == 1
    assert runtime["source_report_sha256"] == digest(source_path)
    for key in ("source_revision", "protocol_sha256", "proposals_sha256", "comparison_report_sha256"):
        assert source[key] == runtime[key]
    freeze = json.loads((HERE / "latest-regression-freeze.json").read_text())
    for name, expected in dict(freeze["files"], **source["source_files"]).items():
        assert digest(ROOT / name) == expected, name
    prior_path = HERE / "results/corrected-teaching-attempt1.json"
    previous = json.loads(prior_path.read_text())
    assert previous["failure"]["error"] == "AssertionError" and len(previous["results"]) == 5
    assert source["resumed_from"]["report_sha256"] == digest(prior_path)
    assert source["resumed_from"]["source_revision"] == previous["source_revision"]
    assert source["resumed_from"]["reused_results"] == 4
    assert source["results"][:4] == previous["results"][:4]
    for name, expected in previous["source_files"].items():
        raw = subprocess.check_output(["git", "show", f"{previous['source_revision']}:{name}"], cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest() == expected, name
    diagnostic = json.loads((HERE / "results/corrected-teaching-control-diagnostic.json").read_text())
    assert diagnostic["prototypes_exact"] and all(diagnostic[k] == 0 for k in ("max_weight_difference", "max_bias_difference", "temperature_difference"))
    for entry in diagnostic["caches"].values():
        assert digest(ROOT / entry["path"]) == entry["sha256"]
    proposals_path = HERE / "results/teaching-correction-proposals.json"
    assert digest(proposals_path) == source["proposals_sha256"]
    proposals = json.loads(proposals_path.read_text())["proposals"]
    references = {r["candidate_sha256"]: r for name in ("context-runtime.json", "polynomial-runtime.json", "boundary-runtime.json")
        for r in json.loads((HERE / "results" / name).read_text())["results"]}
    assert len(source["results"]) == 8 and len(runtime["results"]) == 4
    data, total, differences, controls = {}, 0, [], []
    for dataset in ("clinc150", "banking77"):
        manifest = json.loads((HERE / "manifest.json").read_text())["datasets"][dataset]["splits"]
        data[dataset] = {}
        for part in ("fit", "calibration", "development"):
            p = ROOT / ".system1/n8n-gauntlet" / f"{dataset}-{part}.json"
            assert digest(p) == manifest[part]["sha256"]
            data[dataset][part] = json.loads(p.read_text())
    for fitted in source["results"]:
        dataset, kind = fitted["dataset"], fitted["kind"]
        original, rows = data[dataset]["fit"], data[dataset]["development"]
        corrected = fitted["condition"] == "corrected"
        expected_changes = [r for r in proposals if r["dataset"] == dataset] if corrected else []
        assert fitted["corrections_applied"] == [r for row in original for r in expected_changes if r["group"] == row["group"]]
        replacements = {r["group"]: r for r in expected_changes}
        effective = [dict(row, label=replacements[row["group"]]["proposed_label"]) if row["group"] in replacements else dict(row) for row in original]
        assert fitted["effective_fit_sha256"] == hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()
        assert fitted["fit_groups"] == [r["group"] for r in original]
        folds = json.loads((HERE / "results/review-teaching-development.json").read_text())["datasets"][dataset]["fold_groups"]
        assert fitted["fold_groups"] == folds and fitted["generated_rows_in_fold_heads"] == 0
        bank, neural = dataset == "banking77", kind == "system1"
        extra = json.loads((HERE / "results/contrast-lessons.json").read_text())["lessons"] if bank else []
        if not bank and not neural:
            extra = [r for r in json.loads((HERE / "results/boundary-teaching/lessons.json").read_text())["lessons"] if r["dataset"] == dataset and r["label"] != "oos"]
        negative_count = 0 if bank else 2000 if neural else 674
        assert fitted["original_fit_rows"] == len(original) and fitted["generated_fit_rows"] == len(extra)
        assert fitted["negative_review_rows"] == negative_count and fitted["review_rows"] == len(original) + len(data[dataset]["calibration"]) + negative_count
        assert len(fitted["outcomes"]) == len(rows)
        for row, expected in zip(fitted["outcomes"], rows, strict=True):
            assert row["group"] == expected["group"] and row["truth"] == expected["label"]
        same(fitted["metrics"], metrics(fitted["outcomes"]))
        old = references[fitted["reference_sha256"]]
        if not corrected:
            score_diff, confidence_diff = [], []
            for row, expected in zip(fitted["outcomes"], old["outcomes"], strict=True):
                assert all(row[k] == expected[k] for k in ("group", "truth", "suggestion", "category", "needsReview"))
                if neural:
                    assert row["predictionSet"] == expected["predictionSet"]
                score_diff.append(abs(row["score"] - expected["reliability" if neural else "acceptance_score"]))
                confidence_diff.append(abs(row["confidence"] - expected["confidence"]))
            expected = dict(route_mismatches=[], max_score_difference=max(score_diff), max_confidence_difference=max(confidence_diff), retimed=False)
            same(fitted["control_reproduction"], expected)
            assert max(score_diff) <= 1e-4 and max(confidence_diff) <= 1e-4
            controls.append(dict(dataset=dataset, kind=kind, outcomes=len(rows), **expected))
            continue
        measured = next(r for r in runtime["results"] if (r["dataset"], r["kind"]) == (dataset, kind))
        assert fitted["manifest_sha256"] == measured["candidate_sha256"]
        assert measured["selection_runtime_mismatches"] == [] and measured["errors"] == 0
        assert len(measured["outcomes"]) == len(rows)
        for row, selected in zip(measured["outcomes"], fitted["outcomes"], strict=True):
            assert row["candidate"] == measured["candidate_sha256"] and row["teacherCalls"] == 0 and "error" not in row
            assert math.isfinite(row["latency_ms"]) and row["latency_ms"] >= 0
            assert all(row[k] == selected[k] for k in ("group", "truth", "suggestion", "category", "needsReview"))
            if neural:
                assert row["predictionSet"] == selected["predictionSet"]
            differences.append(abs(row["reliability" if neural else "acceptance_score"] - selected["score"]))
        total += len(rows)
        outcomes = measured["outcomes"]
        same(measured["metrics"], metrics(outcomes))
        same(measured["per_intent"], {label: metrics([r for r in outcomes if r["truth"] == label]) for label in sorted({r["truth"] for r in outcomes})})
        assert measured["accepted_errors"] == [r for r in outcomes if not r["needsReview"] and r["category"] != r["truth"]]
        times = [r["latency_ms"] for r in outcomes]
        same(measured["latency"], dict(requests=len(rows), warmups=100, p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95))))
        m = measured["metrics"]
        assert measured["development_gates"] == dict(coverage_at_least_80=m["supported_coverage"]["fraction"] >= .8,
            accepted_accuracy_at_least_99=m["accepted_accuracy"]["fraction"] >= .99,
            oos_false_acceptance_at_most_1=bank or m["oos_false_acceptance"]["fraction"] <= .01,
            adapter_p95_below_5_ms=measured["latency"]["p95_ms"] < 5, no_runtime_errors=True)
        folder = HERE / "artifacts" / measured["folder"]
        assert digest(folder / "manifest.json") == measured["candidate_sha256"]
        assert json.loads((folder / "manifest.json").read_text()) == measured["manifest"] == fitted["manifest"]
        for name, expected in measured["manifest"]["files"].items():
            assert digest(folder / name) == expected
        assert measured["manifest"]["corrected_fit_groups"] == [r["group"] for r in fitted["corrections_applied"]]
        assert measured["manifest"]["fit_rows"] == len(effective) + len(extra)
        assert measured["artifact_bytes"] == sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
        assert measured["external_encoder_bytes"] == ((133804886 if bank else 23492300) if neural else 0)
        labels = sorted({r["label"] for r in original})
        ya = np.asarray([labels.index(r["label"]) for r in effective + extra])
        with np.load(folder / ("scope.npz" if neural else "weights.npz"), allow_pickle=False) as arrays:
            if neural:
                np.testing.assert_array_equal(arrays["class_offsets"], np.concatenate([[0], np.cumsum(np.bincount(ya, minlength=len(labels)))]))
                if bank:
                    vectors = np.load(ROOT / diagnostic["caches"]["augmented"]["path"], allow_pickle=False)
                    assert fitted["manifest"]["teaching_projection"] == "final-head-and-prototypes-batch32; review-folds-and-serving-single"
                else:
                    identity = fitted["manifest"]["encoder"]
                    encoder_digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                    key = hashlib.sha256(("single-request-v1:" + encoder_digest + json.dumps(original + extra, sort_keys=True)).encode()).hexdigest()
                    vectors = np.load(ROOT / ".system1/n8n-gauntlet" / f"features-{key}.npy", allow_pickle=False)
                np.testing.assert_array_equal(arrays["prototypes"], vectors[np.argsort(ya, kind="stable")])
            else:
                np.testing.assert_array_equal(arrays["fit_labels"], ya)
                assert list(arrays["labels"]) == labels
        assert fitted["scalar_preflight_max_difference"] < 1e-4
    prior = json.loads((HERE / "results/human-only-runtime.json").read_text())["comparisons"]
    assert len(runtime["comparisons"]) == len(prior) == 4
    for actual, old in zip(runtime["comparisons"], prior, strict=True):
        assert (actual["dataset"], actual["kind"]) == (old["dataset"], old["kind"])
        assert actual["incumbent"] == old["selected_development_candidate"] and actual["qualified"] is False
        winner = actual["incumbent"]
        measured = next(r for r in runtime["results"] if (r["dataset"], r["kind"]) == (actual["dataset"], actual["kind"]))
        valid = all(v for k, v in measured["development_gates"].items() if k != "coverage_at_least_80")
        if valid and measured["metrics"]["supported_coverage"]["numerator"] > winner["metrics"]["supported_coverage"]["numerator"]:
            winner = {k: measured[k] for k in ("folder", "candidate_sha256", "metrics", "latency")}
            winner["source"] = "corrected-teaching"
        assert actual["selected_development_candidate"] == winner
    assert runtime["response_cache"] is runtime["receipts"] is False
    assert max(differences) < 1e-4
    print(json.dumps(dict(scope="correction overlay, control and complete-runtime audit; no fitting or inference", qualified=False,
        source_sha256=digest(source_path), runtime_sha256=digest(runtime_path), frozen_files_unchanged=len(freeze["files"]),
        declared_sources_verified=len(source["source_files"]), complete_outcomes_verified=total, controls=controls,
        overlay_changes_per_method=dict(clinc150=6, banking77=1), evaluation_label_changes=0,
        retained_failed_attempt_sha256=digest(prior_path), reused_clinc_fits=4,
        historical_source_files_verified=len(previous["source_files"]), original_banking_teaching_caches_verified=2,
        max_selection_runtime_review_difference=max(differences)), indent=2))


if __name__ == "__main__":
    main()
