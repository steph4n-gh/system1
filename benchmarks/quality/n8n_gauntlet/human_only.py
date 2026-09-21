"""Experiment 2n: remove generated banking lessons from both complete adapters."""
import time
STARTED = time.perf_counter()

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from baseline import Baseline
from baseline_review import meta_features as lexical_meta, midpoint
from boundary_development import compile_head, logistic_fit
from crossfit_probe import probabilities
from develop import frontier, load_splits
from encoder_probe import PreparedFeatures
from evaluate import deny_network_control, gates, quality
from human_only_runtime import HumanOnlyCandidate
from latest_regression import frozen
from polynomial_review import PolynomialBaseline, prepare
from polynomial_scores import expand, load_review
from review_teaching import features, meta_features
from system1 import CompiledSystemOneModel, System1Engine, SystemOneCompiler
from teach_boundaries import HERE, ROOT, committed, digest

PROTOCOL = HERE / "HUMAN_ONLY_PROTOCOL.md"
COMPARISON = HERE / "results/fused-review-runtime.json"
CONTROL = HERE / "results/polynomial-runtime.json"


class HumanOnlyBaseline(PolynomialBaseline):
    def __init__(self, folder):
        Baseline.__init__(self, folder)
        self.review_parameters = load_review(dict(self.manifest, review_parameters=dict(
            mean=self.mean, scale=self.scale, weights=self.gate_weights[0],
            bias=float(self.gate_bias[0]))), len(self.labels))


def review_fit(teaching, predicted, target, development, guesses, labels, rows, eligible):
    scaler = StandardScaler().fit(expand(teaching, True))
    train = np.column_stack([scaler.transform(expand(teaching, True)), np.eye(len(labels))[predicted]])
    dev = np.column_stack([scaler.transform(expand(development, True)), np.eye(len(labels))[guesses]])
    started = time.perf_counter()
    model = LogisticRegression(C=.1, max_iter=1000).fit(train, target)
    elapsed = (time.perf_counter() - started) * 1000
    assert model.n_iter_.max() < 1000, "Review did not converge"
    scores = model.predict_proba(dev)[:, 1]
    predictions = np.asarray(labels)[guesses]
    selection = frontier([r["label"] for r in rows], predictions, scores, eligible=eligible)
    threshold = midpoint(scores, selection["best_development_coverage_at_quality_targets"]["threshold"])
    outcomes = []
    for i, row in enumerate(rows):
        review = bool(scores[i] < threshold or (eligible is not None and not eligible[i]))
        outcomes.append(dict(group=row["group"], truth=row["label"], suggestion=str(predictions[i]),
            score=float(scores[i]), needsReview=review, category=None if review else str(predictions[i])))
    return scaler, model, elapsed, selection, threshold, outcomes


def fit_one(kind, folder, report):
    begin = time.perf_counter()
    parent, labels, rows, teaching, predicted, targets, meta, guesses, eligible, evidence = prepare("banking77", kind)
    # Reproduce the existing control using the exact same fold/calibration lessons.
    _, _, _, _, _, control = review_fit(teaching, predicted, targets, meta, guesses, labels, rows, eligible)
    previous = next(r for r in json.loads(CONTROL.read_text())["results"]
        if r["dataset"] == "banking77" and r["kind"] == kind and r["condition"] == "quadratic")
    assert len(control) == len(previous["outcomes"]) == 1960
    differences = []
    for actual, old in zip(control, previous["outcomes"], strict=True):
        assert all(actual[k] == old[k] for k in ("group", "truth", "suggestion", "category", "needsReview"))
        differences.append(abs(actual["score"] - old["reliability" if kind == "system1" else "acceptance_score"]))
    assert max(differences) < 1e-4
    evidence["control_reproduced"] = dict(candidate_sha256=previous["candidate_sha256"], matching_outcomes=len(control),
        max_review_difference=max(differences), metrics=quality(control), retimed=False)
    data = load_splits("banking77")
    original, calibration = data["fit"], data["calibration"]
    assert len(original) == 6026 and len(calibration) == 1960 and len(labels) == 77 and "oos" not in labels
    yi = np.asarray([labels.index(r["label"]) for r in original])
    yc = np.asarray([labels.index(r["label"]) for r in calibration])
    extraction = []
    started = time.perf_counter()
    if kind == "system1":
        x = {key: features(data[key], parent.encoder, single=True, evidence=extraction) for key in data}
        prepared = PreparedFeatures(original + calibration, np.concatenate([x["fit"], x["calibration"]]), parent.encoder)
        compiler = SystemOneCompiler(parent.model.schema, projector=prepared, choice_solver="logistic", regularization=.01)
        fit_started = time.perf_counter()
        taught = compile_head(compiler, original, calibration)
        model = CompiledSystemOneModel(taught.schema, taught.heads, dimension=384,
            projector=parent.encoder, metadata=taught.metadata, use_cache=False)
        engine = System1Engine(model.schema, model=model, strict_mode=True, use_cache=False)
        values = {key: probabilities(model.heads["intent"], x[key]) for key in ("calibration", "development")}
        intent_fit_ms = (time.perf_counter() - fit_started) * 1000
        review_meta = meta_features
        eligible = [not engine.decide(row["prompt"], embedding=vector, alpha=.075, record_receipt=False).is_ambiguous
            for row, vector in zip(rows, x["development"], strict=True)]
    else:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        x = {"fit": vectorizer.fit_transform([r["prompt"] for r in original])}
        x.update({key: vectorizer.transform([r["prompt"] for r in data[key]]) for key in ("calibration", "development")})
        fit_started = time.perf_counter()
        model = logistic_fit(x["fit"], yi)
        intent_fit_ms = (time.perf_counter() - fit_started) * 1000
        assert np.array_equal(model.classes_, np.arange(len(labels)))
        values = {key: model.predict_proba(x[key]) for key in ("calibration", "development")}
        review_meta, eligible = lexical_meta, None
    intent_preparation_and_fit_ms = (time.perf_counter() - started) * 1000
    meta = {key: review_meta(p, x[key], x["fit"], yi) for key, p in values.items()}
    guesses = {key: p.argmax(axis=1) for key, p in values.items()}
    # Only the calibration portion changes; the original out-of-fold targets stay fixed.
    teaching = np.concatenate([teaching[:len(original)], meta["calibration"]])
    predicted = np.concatenate([predicted[:len(original)], guesses["calibration"]])
    targets = np.concatenate([targets[:len(original)], guesses["calibration"] == yc])
    scaler, review, review_ms, selection, threshold, outcomes = review_fit(
        teaching, predicted, targets, meta["development"], guesses["development"], labels, rows, eligible)
    saved = folder / f"banking77-{kind}-human-only"
    saved.mkdir()
    manifest = dict(status="unqualified-human-only-development", dataset="banking77", kind=kind, condition="human-only",
        categories=parent.manifest["categories"], instructions=parent.manifest["instructions"],
        protocol_sha256=digest(PROTOCOL), fit_rows=len(original), calibration_rows=len(calibration),
        generated_fit_rows=0, review_rows=len(targets), quadratic=True, review_C=.1, predicted_class_feature=True)
    if kind == "system1":
        model.save(saved / "intent.s1m")
        order = np.argsort(yi, kind="stable")
        np.savez_compressed(saved / "scope.npz", prototypes=x["fit"][order],
            class_offsets=np.concatenate([[0], np.cumsum(np.bincount(yi, minlength=len(labels)))]),
            mean=scaler.mean_, scale=scaler.scale_, weights=review.coef_[0], bias=review.intercept_[0])
        manifest.update(encoder=parent.encoder.identity, guard="category-reliability", alpha=.075,
            reliability_threshold=threshold, teaching_projection="single-request", regularization=.01,
            files={name: digest(saved / name) for name in ("intent.s1m", "scope.npz")})
    else:
        (saved / "vocabulary.json").write_text(json.dumps(vectorizer.vocabulary_, sort_keys=True) + "\n")
        np.savez_compressed(saved / "weights.npz", weights=model.coef_, bias=model.intercept_, idf=vectorizer.idf_,
            labels=np.asarray(labels), mean=scaler.mean_, scale=scaler.scale_, gate_weights=review.coef_,
            gate_bias=review.intercept_, fit_labels=yi)
        sparse.save_npz(saved / "prototypes.npz", x["fit"])
        manifest.update(C=10, policy="reliability", threshold=threshold, ngram_range=[1, 2], sublinear_tf=True,
            files={name: digest(saved / name) for name in ("weights.npz", "prototypes.npz", "vocabulary.json")})
    (saved / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    candidate = (HumanOnlyCandidate if kind == "system1" else HumanOnlyBaseline)(saved)
    scalar = [candidate.reliability(values["development"][i], x["development"][i]) if kind == "system1"
        else candidate.review_score(values["development"][i], x["development"][i]) for i in range(32)]
    scalar_difference = max(abs(value - outcomes[i]["score"]) for i, value in enumerate(scalar))
    assert scalar_difference < 1e-4
    result = dict(dataset="banking77", kind=kind, condition="human-only", folder=saved.name,
        manifest=manifest, manifest_sha256=candidate.identity, evidence=evidence,
        fit_groups=[r["group"] for r in original], calibration_groups=[r["group"] for r in calibration],
        original_fit_rows=len(original), removed_generated_rows=924, review_negative_targets=int((~targets).sum()),
        intent_fit_ms=intent_fit_ms, intent_preparation_and_fit_ms=intent_preparation_and_fit_ms,
        review_fit_ms=review_ms, total_preparation_and_fit_ms=(time.perf_counter() - begin) * 1000,
        feature_extraction=extraction, scalar_review_preflight_max_difference=scalar_difference,
        quality=selection, metrics=quality(outcomes), outcomes=outcomes)
    report["results"].append(result)
    (folder / "development.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("kind", "metrics", "intent_fit_ms", "review_fit_ms")}), flush=True)


def fit(folder):
    paths = [PROTOCOL, Path(__file__).resolve(), HERE / "human_only_runtime.py", COMPARISON, CONTROL,
        HERE / "results/contrast-lessons.json", HERE / "manifest.json", HERE / "polynomial_review.py",
        HERE / "polynomial_scores.py", HERE / "runtime_probe.py", HERE / "baseline.py"]
    revision = committed(paths)
    frozen(False)
    denied = deny_network_control()
    assert len(json.loads((HERE / "results/contrast-lessons.json").read_text())["lessons"]) == 924
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(scope="experiment 2n complete banking ablation development; full CLINC failure unchanged", qualified=False,
        source_revision=revision, source_files={str(p.relative_to(ROOT)): digest(p) for p in paths},
        protocol_sha256=digest(PROTOCOL), comparison_report_sha256=digest(COMPARISON),
        os_network_denial_errno=denied, teacher_calls=0, new_api_cost=0, results=[])
    with threadpool_limits(limits=1):
        try:
            for kind in ("system1", "baseline"):
                fit_one(kind, folder, report)
                gc.collect()
        except Exception as exc:
            report["failure"] = dict(error=type(exc).__name__, message=str(exc))
            (folder / "development.json").write_text(json.dumps(report, indent=2) + "\n")
            raise


def measure(folder):
    frozen(False)
    source = json.loads((folder / "development.json").read_text())
    assert len(source["results"]) == 2 and "failure" not in source
    for name, expected in source["source_files"].items():
        assert digest(ROOT / name) == expected, name
    report = dict(scope="experiment 2n complete banking development adapters; full scope not qualified", qualified=False,
        source_report_sha256=digest(folder / "development.json"), source_revision=source["source_revision"],
        protocol_sha256=digest(PROTOCOL), comparison_report_sha256=digest(COMPARISON), os_network_denial_errno=deny_network_control(),
        teacher_calls=0, new_api_cost=0, response_cache=False, receipts=False,
        process_import_and_preflight_ms=(time.perf_counter() - STARTED) * 1000,
        environment=dict(platform=platform.platform(), python=platform.python_version()),
        packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "onnxruntime", "tokenizers", "threadpoolctl")}, results=[])
    with (folder / "runtime.json").open("x") as stream, (folder / "runtime.jsonl").open("x") as journal, threadpool_limits(limits=1):
        for selected in source["results"]:
            saved = folder / selected["folder"]
            begin = time.perf_counter()
            candidate = (HumanOnlyCandidate if selected["kind"] == "system1" else HumanOnlyBaseline)(saved)
            load_ms = (time.perf_counter() - begin) * 1000
            assert candidate.identity == selected["manifest_sha256"]
            rows = load_splits("banking77")["development"]
            payload = {key: candidate.manifest[key] for key in ("categories", "instructions")}
            for row in rows[:100]:
                candidate.classify(dict(payload, text=row["prompt"]))
            outcomes, mismatches = [], []
            for row, expected in zip(rows, selected["outcomes"], strict=True):
                begin = time.perf_counter()
                try:
                    response = candidate.classify(dict(payload, text=row["prompt"]))
                except Exception as exc:
                    response = dict(candidate=candidate.identity, needsReview=True, category=None, suggestion=None, teacherCalls=0, error=type(exc).__name__)
                elapsed = (time.perf_counter() - begin) * 1000
                if any(response[k] != expected[k] for k in ("suggestion", "category", "needsReview")):
                    mismatches.append(row["group"])
                outcome = dict(group=row["group"], truth=row["label"], latency_ms=elapsed, **response)
                outcomes.append(outcome)
                journal.write(json.dumps(dict(folder=saved.name, **outcome)) + "\n")
                journal.flush()
            times = [r["latency_ms"] for r in outcomes]
            metrics = quality(outcomes)
            errors = sum("error" in r for r in outcomes)
            result = dict(dataset="banking77", kind=selected["kind"], condition="human-only", folder=saved.name,
                manifest=candidate.manifest, candidate_sha256=candidate.identity, metrics=metrics, qualified=False,
                errors=errors, selection_runtime_mismatches=mismatches, load_ms=load_ms,
                artifact_bytes=sum(p.stat().st_size for p in saved.iterdir() if p.is_file()),
                external_encoder_bytes=sum(candidate.manifest["encoder"][k] for k in ("model_bytes", "tokenizer_bytes")) if selected["kind"] == "system1" else 0,
                latency=dict(requests=len(rows), warmups=100, p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95))),
                development_gates=gates(metrics, float(np.percentile(times, 95)), errors),
                accepted_errors=[r for r in outcomes if not r["needsReview"] and r["category"] != r["truth"]],
                per_intent={label: quality([r for r in outcomes if r["truth"] == label]) for label in sorted({r["truth"] for r in outcomes})}, outcomes=outcomes)
            report["results"].append(result)
            stream.seek(0)
            stream.write(json.dumps(report, indent=2) + "\n")
            stream.truncate()
            stream.flush()
            print(json.dumps({k: result[k] for k in ("kind", "metrics", "latency", "development_gates", "selection_runtime_mismatches")}), flush=True)
            del candidate
            gc.collect()
        report["comparisons"] = []
        for prior in json.loads(COMPARISON.read_text())["comparisons"]:
            incumbent = prior["selected_development_candidate"]
            winner = incumbent
            if prior["dataset"] == "banking77":
                result = next(r for r in report["results"] if r["kind"] == prior["kind"])
                valid = all(v for k, v in result["development_gates"].items() if k != "coverage_at_least_80")
                if valid and not result["selection_runtime_mismatches"] and result["metrics"]["supported_coverage"]["numerator"] > incumbent["metrics"]["supported_coverage"]["numerator"]:
                    winner = {k: result[k] for k in ("folder", "candidate_sha256", "metrics", "latency")}
                    winner["source"] = result["condition"]
            report["comparisons"].append(dict(dataset=prior["dataset"], kind=prior["kind"], incumbent=incumbent,
                selected_development_candidate=winner, qualified=False, incumbent_timings_from_prior_run=True))
        stream.seek(0)
        stream.write(json.dumps(report, indent=2) + "\n")
        stream.truncate()
    assert all(not r["selection_runtime_mismatches"] for r in report["results"]), "Routing differs; failure retained"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "measure"))
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    (fit if args.action == "fit" else measure)(args.folder)
