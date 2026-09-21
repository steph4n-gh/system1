"""Fixed experiment-2l: teach a small feature transform, with no encoder update."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import warnings

import numpy as np
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from boundary_development import compile_head
from develop import frontier, load_splits
from encoder_probe import Encoder, PreparedFeatures
from evaluate import deny_network_control, quality
from latest_regression import frozen
from review_teaching import features
from shallow_features import ShallowFeatures
from system1 import ChoiceField, CompiledSystemOneModel, DecisionSchema, System1Engine, SystemOneCompiler
from teach_boundaries import HERE, ROOT, committed, digest

PROTOCOL = HERE / "SHALLOW_FEATURES_PROTOCOL.md"


def fit_one(dataset, folder, report, control):
    bank = dataset == "banking77"
    data = load_splits(dataset)
    original = data["fit"]
    earlier = json.loads((HERE / "results/contrast-lessons.json").read_text())["lessons"] if bank else []
    augmented = original + earlier
    labels = sorted({r["label"] for r in original})
    schema = type("Intent", (DecisionSchema,), {"intent": ChoiceField(options=labels)})
    assert control["manifest"]["fit_rows"] == len(augmented)
    assert [(r["group"], r["label"]) for r in data["development"]] == [(r["group"], r["truth"]) for r in control["outcomes"]]
    encoder = Encoder("bge-small" if bank else "minilm", threads=4 if bank else 1)
    assert encoder.identity == control["manifest"]["encoder"]
    parts = dict(fit=augmented, calibration=data["calibration"], development=data["development"])
    extraction = []
    semantic = {key: features(rows, encoder, single=True, evidence=extraction) for key, rows in parts.items()}
    learner = MLPClassifier(hidden_layer_sizes=(128,), activation="relu", solver="adam", alpha=.01,
        batch_size=128, learning_rate_init=.001, max_iter=100, shuffle=True, random_state=20260921,
        tol=0., early_stopping=False, n_iter_no_change=101)
    started = time.perf_counter()
    print(f"{dataset}: starting fixed 100-epoch feature teaching", flush=True)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        learner.fit(semantic["fit"], [r["label"] for r in augmented])
    fit_ms = (time.perf_counter() - started) * 1000
    assert learner.n_iter_ == 100 and list(learner.classes_) == labels
    transform = ShallowFeatures(encoder, learner.coefs_[0], learner.intercepts_[0])
    candidate_folder = folder / f"{dataset}-shallow"
    candidate_folder.mkdir()
    np.savez_compressed(candidate_folder / "transform.npz", weights=transform.weights, bias=transform.bias,
        teaching_output_weights=learner.coefs_[1], teaching_output_bias=learner.intercepts_[1])
    # Keep completed teaching evidence before any head fit, replay or comparison.
    teaching = dict(dataset=dataset, parameters=learner.get_params(), epochs=int(learner.n_iter_),
        loss_curve=[float(v) for v in learner.loss_curve_], loss=float(learner.loss_), fit_ms=fit_ms,
        warnings=[dict(category=n.category.__name__, message=str(n.message)) for n in notices],
        fit_groups=[r["group"] for r in augmented], labels=labels, semantic_preparation=extraction,
        transform_file_sha256=digest(candidate_folder / "transform.npz"))
    report.setdefault("teaching", []).append(teaching)
    (folder / "head-development.json").write_text(json.dumps(report, indent=2) + "\n")
    preflight = []
    for vector in semantic["fit"][:32]:
        expected = np.maximum(vector @ learner.coefs_[0] + learner.intercepts_[0], 0).astype(np.float32)
        expected /= max(float(np.linalg.norm(expected)), 1e-12)
        actual = transform.transform(vector)[384:]
        # Each nonzero component has norm 1/sqrt(2) after concatenation.
        expected /= np.sqrt(2)
        preflight.append(float(np.max(np.abs(actual - expected))))
    assert max(preflight) < 1e-5
    started = time.perf_counter()
    x = {part: np.stack([transform.transform(v) for v in values]) for part, values in semantic.items()}
    preparation_ms = (time.perf_counter() - started) * 1000
    prepared = PreparedFeatures(augmented + data["calibration"], np.concatenate([x["fit"], x["calibration"]]), transform)
    compiler = SystemOneCompiler(schema, projector=prepared, choice_solver="logistic", regularization=.01 if bank else .1)
    started = time.perf_counter()
    taught = compile_head(compiler, augmented, data["calibration"])
    model = CompiledSystemOneModel(taught.schema, taught.heads, dimension=512, projector=transform,
                                  metadata=taught.metadata, use_cache=False)
    head_fit_ms = (time.perf_counter() - started) * 1000
    model.save(candidate_folder / "intent.s1m")
    manifest = dict(status="unqualified-shallow-feature-head-comparison", dataset=dataset, condition="shallow",
        dimension=512, encoder=encoder.identity, projector_digest=transform.projector_digest(), labels=labels,
        protocol_sha256=digest(PROTOCOL), regularization=.01 if bank else .1, alpha=.075 if bank else .05,
        fit_rows=len(augmented), original_fit_rows=len(original), earlier_generated_supported=len(earlier),
        control_manifest_sha256=control["candidate_sha256"],
        files={name: digest(candidate_folder / name) for name in ("intent.s1m", "transform.npz")})
    (candidate_folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    restored_projector = ShallowFeatures.load(encoder, candidate_folder / "transform.npz")
    restored = CompiledSystemOneModel.load(candidate_folder / "intent.s1m", projector=restored_projector)
    restored.use_cache = False
    engine = System1Engine(restored.schema, model=restored, strict_mode=True, use_cache=False)
    outcomes = []
    with (candidate_folder / "outcomes.jsonl").open("x") as journal:
        for row, vector in zip(data["development"], x["development"], strict=True):
            decision = engine.decide(row["prompt"], embedding=vector, alpha=manifest["alpha"], record_receipt=False)
            suggestion = decision.values["intent"]
            ordered = sorted(decision.probabilities["intent"].values())
            review = decision.is_ambiguous or suggestion == "oos"
            outcome = dict(group=row["group"], truth=row["label"], suggestion=suggestion, confidence=ordered[-1],
                margin=ordered[-1] - ordered[-2], strict_eligible=not decision.is_ambiguous,
                predictionSet=decision.conformal_sets["intent"], needsReview=bool(review),
                category=None if review else suggestion, teacherCalls=0)
            outcomes.append(outcome)
            journal.write(json.dumps(outcome) + "\n")
            journal.flush()
    truth, pred = [r["truth"] for r in outcomes], [r["suggestion"] for r in outcomes]
    curves = {score: frontier(truth, pred, [r[score] for r in outcomes]) for score in ("confidence", "margin")}
    strict = {score: frontier(truth, pred, [r[score] for r in outcomes],
              eligible=[r["strict_eligible"] for r in outcomes]) for score in ("confidence", "margin")}
    result = dict(dataset=dataset, condition="shallow", manifest=manifest, folder=candidate_folder.name,
        candidate_sha256=digest(candidate_folder / "manifest.json"), semantic_preparation=extraction,
        transform_fit_ms=fit_ms, additional_feature_preparation_ms=preparation_ms, head_fit_ms=head_fit_ms,
        artifact_bytes=sum((candidate_folder / name).stat().st_size for name in ("intent.s1m", "transform.npz", "manifest.json")),
        external_encoder_bytes=encoder.identity["model_bytes"] + encoder.identity["tokenizer_bytes"],
        strict_only_metrics=quality(outcomes), probability_margin_frontiers=curves, strict_probability_margin_frontiers=strict,
        raw_oos_predictions=sum(r["truth"] == r["suggestion"] == "oos" for r in outcomes),
        supported_predicted_oos=sum(r["truth"] != "oos" and r["suggestion"] == "oos" for r in outcomes),
        hidden_export_preflight=dict(requests=32, max_difference=max(preflight)), outcomes=outcomes)
    report["results"].append(result)
    (folder / "head-development.json").write_text(json.dumps(report, indent=2) + "\n")
    replay_differences = []
    for row, expected in zip(data["development"][:16], outcomes[:16], strict=True):
        actual = engine.decide(row["prompt"], alpha=manifest["alpha"], record_receipt=False)
        assert actual.values["intent"] == expected["suggestion"]
        assert actual.conformal_sets["intent"] == expected["predictionSet"]
        assert (not actual.is_ambiguous) == expected["strict_eligible"]
        delta = abs(actual.probabilities["intent"][expected["suggestion"]] - expected["confidence"])
        assert delta < 1e-5
        replay_differences.append(delta)
    result["saved_projection_replay"] = dict(requests=16, max_confidence_difference=max(replay_differences))
    a = control["strict_only_metrics"]["raw_supported_accuracy"]["numerator"]
    b = result["strict_only_metrics"]["raw_supported_accuracy"]["numerator"]
    best = lambda values: max(values[k]["best_development_coverage_at_quality_targets"]["coverage"] for k in ("confidence", "margin"))
    comparison = dict(dataset=dataset, control_manifest_sha256=control["candidate_sha256"], raw_gain=b-a,
        control_best_frontier=best(control["probability_margin_frontiers"]), new_best_frontier=best(curves))
    comparison["advance_to_review_comparison"] = b-a >= 10 and comparison["new_best_frontier"] > comparison["control_best_frontier"]
    report.setdefault("comparisons", []).append(comparison)
    (folder / "head-development.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(comparison), flush=True)
    del learner, compiler, prepared, taught, model, restored, engine, restored_projector, transform, encoder
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    folder = parser.parse_args().folder
    paths = [PROTOCOL, Path(__file__), HERE / "shallow_features.py", HERE / "fused_features.py",
             ROOT / "tests/test_shallow_features.py", HERE / "results/contrast-lessons.json",
             HERE / "results/fused-head-development.json", HERE / "results/baseline-review-runtime.json",
             HERE / "results/fused-review-runtime.json"]
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
    report = dict(scope="experiment 2l development head comparison; no new review, adapter latency or qualification",
        qualified=False, source_revision=revision, protocol_sha256=digest(PROTOCOL),
        source_files={str(p.relative_to(ROOT)): digest(p) for p in paths},
        environment=dict(platform=platform.platform(), python=platform.python_version()),
        packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "onnxruntime", "tokenizers", "threadpoolctl")},
        os_network_denial_errno=denied, teacher_calls=0, new_api_cost=0, results=[])
    with threadpool_limits(limits=1):
        for dataset in ("clinc150", "banking77"):
            try:
                fit_one(dataset, folder, report, controls[dataset])
            except Exception as exc:
                report["failure"] = dict(dataset=dataset, error=type(exc).__name__, message=str(exc))
                (folder / "head-development.json").write_text(json.dumps(report, indent=2) + "\n")
                raise


if __name__ == "__main__":
    main()
