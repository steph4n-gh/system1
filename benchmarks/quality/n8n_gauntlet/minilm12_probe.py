"""Experiment 2m: one fixed encoder comparison; saved-head timings only."""
import time
STARTED = time.perf_counter()

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from threadpoolctl import threadpool_limits

from boundary_development import compile_head
from develop import frontier, load_splits
from encoder_probe import PreparedFeatures
from evaluate import deny_network_control, quality
from latest_regression import frozen
from minilm12_encoder import MiniLM12Encoder
from review_teaching import features
from system1 import ChoiceField, CompiledSystemOneModel, DecisionSchema, System1Engine, SystemOneCompiler
from teach_boundaries import HERE, ROOT, committed, digest

PROTOCOL = HERE / "MINILM12_PROTOCOL.md"


def outcome(decision, row):
    suggestion = decision.values["intent"]
    ordered = sorted(decision.probabilities["intent"].values())
    review = decision.is_ambiguous or suggestion == "oos"
    return dict(group=row["group"], truth=row["label"], suggestion=suggestion,
        confidence=ordered[-1], margin=ordered[-1] - ordered[-2],
        strict_eligible=not decision.is_ambiguous, predictionSet=decision.conformal_sets["intent"],
        needsReview=bool(review), category=None if review else suggestion, teacherCalls=0)


def run_one(dataset, folder, report, control):
    bank = dataset == "banking77"
    data = load_splits(dataset)
    earlier = json.loads((HERE / "results/contrast-lessons.json").read_text())["lessons"] if bank else []
    augmented = data["fit"] + earlier
    labels = sorted({r["label"] for r in augmented})
    assert control["manifest"]["fit_rows"] == len(augmented)
    assert [(r["group"], r["label"]) for r in data["development"]] == [(r["group"], r["truth"]) for r in control["outcomes"]]
    schema = type("Intent", (DecisionSchema,), {"intent": ChoiceField(options=labels)})
    start = time.perf_counter()
    encoder = MiniLM12Encoder()
    encoder_load_ms = (time.perf_counter() - start) * 1000
    preparation = []
    print(f"{dataset}: preparing single-request fitting/calibration features", flush=True)
    x = {part: features(rows, encoder, single=True, evidence=preparation)
         for part, rows in dict(fit=augmented, calibration=data["calibration"]).items()}
    prepared = PreparedFeatures(augmented + data["calibration"], np.concatenate([x["fit"], x["calibration"]]), encoder)
    compiler = SystemOneCompiler(schema, projector=prepared, choice_solver="logistic", regularization=.01 if bank else .1)
    start = time.perf_counter()
    taught = compile_head(compiler, augmented, data["calibration"])
    model = CompiledSystemOneModel(taught.schema, taught.heads, dimension=384,
        projector=encoder, metadata=taught.metadata, use_cache=False)
    fit_ms = (time.perf_counter() - start) * 1000
    bundle = folder / f"{dataset}-minilm12"
    bundle.mkdir()
    model.save(bundle / "intent.s1m")
    manifest = dict(status="unqualified-minilm12-head-comparison", dataset=dataset, condition="minilm12",
        dimension=384, encoder=encoder.identity, projector_digest=encoder.projector_digest(), labels=labels,
        protocol_sha256=digest(PROTOCOL), assets_manifest_sha256=digest(HERE / "minilm12-assets.json"),
        regularization=.01 if bank else .1, alpha=.075 if bank else .05, fit_rows=len(augmented),
        original_fit_rows=len(data["fit"]), earlier_generated_supported=len(earlier),
        control_manifest_sha256=control["candidate_sha256"], files={"intent.s1m": digest(bundle / "intent.s1m")})
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    start = time.perf_counter()
    restored = CompiledSystemOneModel.load(bundle / "intent.s1m", projector=encoder)
    restored.use_cache = False
    engine = System1Engine(restored.schema, model=restored, strict_mode=True, use_cache=False)
    head_load_ms = (time.perf_counter() - start) * 1000
    result = dict(dataset=dataset, condition="minilm12", folder=bundle.name, manifest=manifest,
        candidate_sha256=digest(bundle / "manifest.json"), encoder_load_ms=encoder_load_ms,
        head_load_ms=head_load_ms, head_fit_ms=fit_ms, semantic_preparation=preparation,
        artifact_bytes=sum(p.stat().st_size for p in bundle.iterdir() if p.is_file()),
        external_encoder_bytes=encoder.identity["model_bytes"] + encoder.identity["tokenizer_bytes"],
        errors=0, outcomes=[])
    report["results"].append(result)
    path = folder / "head-development.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for row in data["development"][:100]:
        engine.decide(row["prompt"], alpha=manifest["alpha"], record_receipt=False)
    print(f"{dataset}: measuring every saved-head development decision", flush=True)
    with (bundle / "outcomes.jsonl").open("x") as journal:
        for row in data["development"]:
            start = time.perf_counter()
            decision = engine.decide(row["prompt"], alpha=manifest["alpha"], record_receipt=False)
            elapsed = (time.perf_counter() - start) * 1000
            measured = outcome(decision, row)
            measured["latency_ms"] = elapsed
            result["outcomes"].append(measured)
            journal.write(json.dumps(measured) + "\n")
            journal.flush()
    path.write_text(json.dumps(report, indent=2) + "\n")
    rows = result["outcomes"]
    truth, pred = [r["truth"] for r in rows], [r["suggestion"] for r in rows]
    times = [r["latency_ms"] for r in rows]
    result.update(strict_only_metrics=quality(rows),
        probability_margin_frontiers={score: frontier(truth, pred, [r[score] for r in rows]) for score in ("confidence", "margin")},
        strict_probability_margin_frontiers={score: frontier(truth, pred, [r[score] for r in rows],
            eligible=[r["strict_eligible"] for r in rows]) for score in ("confidence", "margin")},
        raw_oos_predictions=sum(r["truth"] == r["suggestion"] == "oos" for r in rows),
        supported_predicted_oos=sum(r["truth"] != "oos" and r["suggestion"] == "oos" for r in rows),
        saved_head_latency=dict(requests=len(rows), warmups=100, p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95))))
    path.write_text(json.dumps(report, indent=2) + "\n")
    differences = []
    for row, expected in zip(data["development"][:16], rows[:16], strict=True):
        actual = outcome(engine.decide(row["prompt"], alpha=manifest["alpha"], record_receipt=False), row)
        assert all(actual[k] == expected[k] for k in ("suggestion", "predictionSet", "strict_eligible", "needsReview", "category"))
        delta = abs(actual["confidence"] - expected["confidence"])
        assert delta < 1e-5
        differences.append(delta)
    result["saved_replay"] = dict(requests=16, max_confidence_difference=max(differences))
    best = lambda r: max(r["probability_margin_frontiers"][k]["best_development_coverage_at_quality_targets"]["coverage"] for k in ("confidence", "margin"))
    comparison = dict(dataset=dataset, control_manifest_sha256=control["candidate_sha256"],
        raw_gain=result["strict_only_metrics"]["raw_supported_accuracy"]["numerator"] - control["strict_only_metrics"]["raw_supported_accuracy"]["numerator"],
        control_best_frontier=best(control), new_best_frontier=best(result),
        saved_head_p95_below_5_ms=result["saved_head_latency"]["p95_ms"] < 5)
    comparison["advance_to_review_comparison"] = comparison["raw_gain"] >= 10 and best(result) > best(control) and comparison["saved_head_p95_below_5_ms"]
    report.setdefault("comparisons", []).append(comparison)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(comparison=comparison, saved_head_latency=result["saved_head_latency"])), flush=True)
    del restored, engine, model, taught, prepared, compiler, encoder
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    folder = parser.parse_args().folder
    paths = [PROTOCOL, Path(__file__), HERE / "minilm12_encoder.py", HERE / "minilm12-assets.json",
             HERE / "results/contrast-lessons.json", HERE / "results/fused-head-development.json",
             HERE / "results/baseline-review-runtime.json", HERE / "results/fused-review-runtime.json"]
    prior = json.loads((HERE / "results/fused-head-development.json").read_text())
    controls = {r["dataset"]: r for r in prior["results"] if r["condition"] == "semantic-control"}
    for control in controls.values():
        bundle = HERE / "artifacts" / ("fused-" + control["folder"])
        assert digest(bundle / "manifest.json") == control["candidate_sha256"]
        paths.append(bundle / "manifest.json")
        for name, expected in control["manifest"]["files"].items():
            assert digest(bundle / name) == expected
            paths.append(bundle / name)
    revision = committed(paths)
    frozen(False)
    denied = deny_network_control()
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(scope="experiment 2m development saved-head comparison; no learned review or complete adapter qualification",
        qualified=False, source_revision=revision, protocol_sha256=digest(PROTOCOL),
        source_files={str(p.relative_to(ROOT)): digest(p) for p in paths},
        process_import_preflight_ms=(time.perf_counter() - STARTED) * 1000,
        environment=dict(platform=platform.platform(), python=platform.python_version()),
        packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "onnxruntime", "tokenizers", "threadpoolctl")},
        os_network_denial_errno=denied, teacher_calls=0, new_api_cost=0, response_cache=False, receipts=False, results=[])
    with threadpool_limits(limits=1):
        for dataset in ("clinc150", "banking77"):
            try:
                run_one(dataset, folder, report, controls[dataset])
            except Exception as exc:
                report["failure"] = dict(dataset=dataset, error=type(exc).__name__, message=str(exc))
                (folder / "head-development.json").write_text(json.dumps(report, indent=2) + "\n")
                raise


if __name__ == "__main__":
    main()
