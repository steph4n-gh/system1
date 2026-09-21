"""Audit saved review evidence, folds and full adapter records without inference."""
import json
import math
from pathlib import Path

import numpy as np

from audit_observed_regression import digest, metrics, same
from system1.core.text import TfidfProjector

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    source_path, runtime_path = (HERE / "results" / name for name in ("fused-review-development.json", "fused-review-runtime.json"))
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
    assert len(source["results"]) == len(runtime["results"]) == 1
    fitted, measured = source["results"][0], runtime["results"][0]
    data = {}
    source_manifest = json.loads((HERE / "manifest.json").read_text())["datasets"]["clinc150"]["splits"]
    for part in ("fit", "calibration", "development"):
        p = ROOT / ".system1/n8n-gauntlet" / f"clinc150-{part}.json"
        assert digest(p) == source_manifest[part]["sha256"]
        data[part] = json.loads(p.read_text())
    assert fitted["original_fit_rows"] == 12019 and fitted["calibration_rows"] == 3020 and fitted["negative_rows"] == 2000
    assert fitted["generated_examples_in_fold_heads"] == 0 and fitted["manifest"]["review_rows"] == 17039
    old_folds = json.loads((HERE / "results/review-teaching-development.json").read_text())["datasets"]["clinc150"]["fold_groups"]
    assert fitted["fold_groups"] == old_folds
    for fold in fitted["folds"]:
        held_out = set(old_folds[fold["fold"]])
        training = [r for r in data["fit"] if r["group"] not in held_out]
        assert fold["lexical_fit_groups"] == [r["group"] for r in training]
        assert fold["fit_rows"] == len(training) and fold["held_out_rows"] == len(held_out)
        # Reconstruct vocabulary/IDF from the declared text only; no model fit.
        lexical = TfidfProjector.fit([r["prompt"] for r in training], max_features=2048)
        assert lexical.projector_digest() == fold["lexical_digest"]
    differences = []
    for row, expected, original in zip(measured["outcomes"], fitted["outcomes"], data["development"], strict=True):
        assert row["group"] == expected["group"] == original["group"] and row["truth"] == expected["truth"] == original["label"]
        assert row["candidate"] == measured["candidate_sha256"] == fitted["manifest_sha256"]
        assert row["teacherCalls"] == 0 and "error" not in row and math.isfinite(row["latency_ms"]) and row["latency_ms"] >= 0
        assert all(row[key] == expected[key] for key in ("suggestion", "category", "needsReview"))
        differences.append(abs(row["reliability"] - expected["score"]))
    same(fitted["metrics"], metrics(fitted["outcomes"]))
    same(measured["metrics"], metrics(measured["outcomes"]))
    rows = measured["outcomes"]
    same(measured["per_intent"], {label: metrics([r for r in rows if r["truth"] == label]) for label in sorted({r["truth"] for r in rows})})
    assert measured["accepted_errors"] == [r for r in rows if not r["needsReview"] and r["category"] != r["truth"]]
    times = [r["latency_ms"] for r in rows]
    same(measured["latency"], dict(requests=3095, warmups=100, p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95))))
    m = measured["metrics"]
    assert measured["development_gates"] == dict(coverage_at_least_80=m["supported_coverage"]["fraction"] >= .8,
        accepted_accuracy_at_least_99=m["accepted_accuracy"]["fraction"] >= .99,
        oos_false_acceptance_at_most_1=m["oos_false_acceptance"]["fraction"] <= .01,
        adapter_p95_below_5_ms=measured["latency"]["p95_ms"] < 5, no_runtime_errors=True)
    assert measured["selection_runtime_mismatches"] == [] and measured["errors"] == 0
    folder = HERE / "artifacts" / measured["folder"]
    assert digest(folder / "manifest.json") == measured["candidate_sha256"]
    assert json.loads((folder / "manifest.json").read_text()) == fitted["manifest"] == measured["manifest"]
    for name, expected in measured["manifest"]["files"].items():
        assert digest(folder / name) == expected
    head = HERE / "artifacts/fused-clinc150-semantic-plus-words"
    assert digest(head / "manifest.json") == measured["manifest"]["head_source_manifest_sha256"]
    assert all((head / name).read_bytes() == (folder / name).read_bytes() for name in ("intent.s1m", "lexical.json"))
    assert measured["artifact_bytes"] == sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
    assert measured["external_encoder_bytes"] == 23492300
    assert fitted["fixed_head_matches_2j"] and fitted["fixed_head_max_confidence_difference"] < 1e-5
    assert fitted["scalar_review_preflight_max_difference"] < 1e-4
    for actual, prior in zip(runtime["comparisons"], json.loads((HERE / "results/direct-oos-runtime.json").read_text())["comparisons"], strict=True):
        assert (actual["dataset"], actual["kind"]) == (prior["dataset"], prior["kind"])
        assert actual["incumbent"] == prior["selected_development_candidate"]
        winner = actual["incumbent"]
        eligible = all(v for k, v in measured["development_gates"].items() if k != "coverage_at_least_80")
        if actual["dataset"] == "clinc150" and actual["kind"] == "system1" and eligible and m["supported_coverage"]["numerator"] > winner["metrics"]["supported_coverage"]["numerator"]:
            winner = {key: measured[key] for key in ("folder", "candidate_sha256", "metrics", "latency")}
            winner["source"] = measured["condition"]
        assert actual["selected_development_candidate"] == winner and actual["qualified"] is False
    assert runtime["response_cache"] is runtime["receipts"] is False
    print(json.dumps(dict(scope="recorded fused-review lineage and adapter audit; no model inference or test access", qualified=False,
        source_sha256=digest(source_path), runtime_sha256=digest(runtime_path), frozen_files_unchanged=len(freeze["files"]),
        declared_sources_verified=len(source["source_files"]), fold_lexical_vocabularies_reconstructed=3, outcomes_verified=len(rows),
        intent_head_and_lexical_config_unchanged=True, max_selection_runtime_review_difference=max(differences)), indent=2))


if __name__ == "__main__":
    main()
