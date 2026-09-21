"""Experiment 2f: fixed intent heads, matched linear/quadratic review teaching."""
import time
STARTED = time.perf_counter()

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from baseline import Baseline, probabilities as softmax
from baseline_review import meta_features as lexical_meta, midpoint
from crossfit_probe import probabilities
from develop import ROOT, OUTPUT, frontier, load_splits
from encoder_probe import PreparedFeatures
from evaluate import deny_network_control, gates, quality
from observed_regression import frozen
from polynomial_runtime import PolynomialCandidate, expand, load_review, numerical_signals, score
from review_runtime import ReviewCandidate
from review_teaching import features, meta_features
from runtime_probe import digest
from system1 import ChoiceField, DecisionSchema, SystemOneCompiler
from teach_boundaries import committed

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "POLYNOMIAL_REVIEW_PROTOCOL.md"


class PolynomialBaseline(Baseline):
    def __init__(self, folder):
        folder = Path(folder)
        manifest = json.loads((folder / "manifest.json").read_text())
        self.parent_folder = ROOT / manifest["base_artifact"]
        super().__init__(self.parent_folder)
        if self.identity != manifest["parent_manifest_sha256"]:
            raise ValueError("Parent intent artifact changed")
        if any(manifest[key] != self.manifest[key] for key in ("categories", "instructions", "files")):
            raise ValueError("Review changed the parent decision contract")
        self.review_parameters = load_review(manifest, len(self.labels))
        self.folder, self.manifest, self.identity = folder, manifest, digest(folder / "manifest.json")

    def classify(self, payload):
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise ValueError("Expected text")
        if not payload["text"].strip() or len(payload["text"]) > 4000:
            raise ValueError("Text must contain 1 to 4000 characters")
        if any(payload.get(key) != self.manifest[key] for key in ("categories", "instructions")):
            return dict(candidate=self.identity, needsReview=True, category=None, reason="changed_contract", teacherCalls=0)
        x = self.vectorizer.transform([payload["text"]])
        values = softmax(x @ self.transposed_weights + self.bias)[0]
        signals, chosen = numerical_signals(values, (x @ self.prototypes).toarray()[0], self.offsets)
        confidence = score(signals, chosen, self.manifest["quadratic"], self.review_parameters)
        suggestion = str(self.labels[chosen])
        review = suggestion == "oos" or confidence < self.manifest["threshold"]
        return dict(candidate=self.identity, needsReview=bool(review), category=None if review else suggestion,
            suggestion=suggestion, confidence=float(values[chosen]), acceptance_score=confidence, teacherCalls=0)


def prepare(dataset, kind):
    """Rebuild original-only correctness lessons; never refit the final head."""
    started = time.perf_counter()
    data = load_splits(dataset)
    original = data["fit"]
    neural, bank = kind == "system1", dataset == "banking77"
    name = ("banking-review" if bank else "clinc-review-consistent") if neural else f"{dataset}-baseline"
    parent = (ReviewCandidate if neural else Baseline)(HERE / "artifacts" / name)
    labels = list(parent.model.heads["intent"].options) if neural else list(parent.labels)
    yi = np.asarray([labels.index(row["label"]) for row in original])
    yc = np.asarray([labels.index(row["label"]) for row in data["calibration"]])
    folds = np.zeros(len(original), dtype=int)
    for label in labels:
        ids = sorted([i for i, row in enumerate(original) if row["label"] == label], key=lambda i: original[i]["group"])
        for position, index in enumerate(ids):
            folds[index] = position % 3
    fold_groups = [[original[i]["group"] for i in np.flatnonzero(folds == fold)] for fold in range(3)]
    previous = json.loads((HERE / "results/review-teaching-development.json").read_text())["datasets"][dataset]
    assert fold_groups == previous["fold_groups"]
    parts = dict(calibration=data["calibration"], development=data["development"])
    if not bank:
        path = OUTPUT / "review-negative-fit.json"
        expected = json.loads((HERE / "review-data-manifest.json").read_text())["files"]["negative-fit"]
        if digest(path) != expected["sha256"]:
            raise ValueError("Review-negative source changed")
        parts["negative"] = json.loads(path.read_text())
    extraction = []
    if neural:
        x = {key: features(rows, parent.encoder, single=True, evidence=extraction) for key, rows in dict(fit=original, **parts).items()}
        projector = PreparedFeatures(original + data["calibration"], np.concatenate([x["fit"], x["calibration"]]), parent.encoder)
        schema = type("Intent", (DecisionSchema,), {"intent": ChoiceField(options=labels)})
        compiler = SystemOneCompiler(schema, projector=projector, choice_solver="logistic", regularization=.01 if bank else .1)
    cf, cp = np.zeros((len(original), 7)), np.zeros(len(original), dtype=int)
    for fold in range(3):
        train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if neural:
            model = compiler.compile({"intent": [(original[i]["prompt"], original[i]["label"]) for i in train]}, augment=False,
                calibration_exemplars={"intent": [(r["prompt"], r["label"]) for r in data["calibration"]]})
            values = probabilities(model.heads["intent"], x["fit"][valid])
            cf[valid] = meta_features(values, x["fit"][valid], x["fit"][train], yi[train])
        else:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            fit = vectorizer.fit_transform([original[i]["prompt"] for i in train])
            valid_x = vectorizer.transform([original[i]["prompt"] for i in valid])
            model = LogisticRegression(C=10, max_iter=1000).fit(fit, yi[train])
            if model.n_iter_.max() >= 1000 or not np.array_equal(model.classes_, np.arange(len(labels))):
                raise RuntimeError("Review-fold head failed to converge or lost a class")
            values = model.predict_proba(valid_x)
            cf[valid] = lexical_meta(values, valid_x, fit, yi[train])
        cp[valid] = values.argmax(axis=1)
        print(f"{dataset}/{kind}: original-only fold {fold + 1}/3", flush=True)
    meta, guesses = {}, {}
    prototype_labels = np.repeat(np.arange(len(labels)), np.diff(parent.class_offsets if neural else parent.offsets))
    for split, rows in parts.items():
        if neural:
            values = probabilities(parent.model.heads["intent"], x[split])
            meta[split] = meta_features(values, x[split], parent.prototypes, prototype_labels)
        else:
            projected = parent.vectorizer.transform([row["prompt"] for row in rows])
            values = softmax(projected @ parent.transposed_weights + parent.bias)
            meta[split] = lexical_meta(values, projected, parent.prototypes.T, prototype_labels)
        guesses[split] = values.argmax(axis=1)
    eligible = ([not parent.engine.decide(row["prompt"], embedding=vector, alpha=parent.manifest["alpha"], record_receipt=False).is_ambiguous
                 for row, vector in zip(data["development"], x["development"], strict=True)] if neural else None)
    extras = ["negative"] if not bank else []
    teaching = np.concatenate([cf, meta["calibration"]] + [meta[key] for key in extras])
    predicted = np.concatenate([cp, guesses["calibration"]] + [guesses[key] for key in extras])
    target = np.concatenate([(cp == yi) & np.asarray([r["label"] != "oos" for r in original]),
        (guesses["calibration"] == yc) & np.asarray([r["label"] != "oos" for r in data["calibration"]])] +
        [np.zeros(len(parts[key]), dtype=bool) for key in extras])
    evidence = dict(parent_manifest_sha256=parent.identity, parent_folder=str(parent.folder.relative_to(ROOT)),
        review_rows=len(target), negative_targets=int((~target).sum()), fold_groups=fold_groups,
        generated_examples_in_fold_heads=0, numerical_signals=7, feature_extraction=extraction,
        original_fit_rows=len(original), review_preparation_ms=(time.perf_counter() - started) * 1000)
    return parent, labels, data["development"], teaching, predicted, target, meta["development"], guesses["development"], eligible, evidence


def fit(folder):
    revision = committed([PROTOCOL, Path(__file__).resolve(), HERE / "polynomial_runtime.py", HERE / "polynomial_scores.py", HERE / "results/boundary-runtime.json"])
    frozen(False)
    denied = deny_network_control()
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(scope="experiment 2f development only", qualified=False, source_revision=revision,
        protocol_sha256=digest(PROTOCOL), comparison_report_sha256=digest(HERE / "results/boundary-runtime.json"),
        os_network_denial_errno=denied, teacher_calls=0, results=[])
    with threadpool_limits(limits=1):
        for dataset in ("clinc150", "banking77"):
            for kind in ("system1", "baseline"):
                parent, labels, rows, teaching, predicted, target, development, guesses, eligible, evidence = prepare(dataset, kind)
                for quadratic in (False, True):
                    scaler = StandardScaler().fit(expand(teaching, quadratic))
                    train = np.column_stack([scaler.transform(expand(teaching, quadratic)), np.eye(len(labels))[predicted]])
                    dev = np.column_stack([scaler.transform(expand(development, quadratic)), np.eye(len(labels))[guesses]])
                    begin = time.perf_counter()
                    model = LogisticRegression(C=.1, max_iter=1000).fit(train, target)
                    fit_ms = (time.perf_counter() - begin) * 1000
                    if model.n_iter_.max() >= 1000:
                        raise RuntimeError("Review fit did not converge")
                    scores = model.predict_proba(dev)[:, 1]
                    predictions = np.asarray(labels)[guesses]
                    frontier_result = frontier([r["label"] for r in rows], predictions, scores, eligible=eligible)
                    threshold = midpoint(scores, frontier_result["best_development_coverage_at_quality_targets"]["threshold"])
                    condition = "quadratic" if quadratic else "linear-control"
                    name = f"{dataset}-{kind}-{condition}"
                    saved = folder / name
                    saved.mkdir()
                    manifest = dict(parent.manifest, status="unqualified-polynomial-review-development", dataset=dataset, kind=kind,
                        base_artifact=evidence["parent_folder"], parent_manifest_sha256=parent.identity,
                        protocol_sha256=digest(PROTOCOL), quadratic=quadratic, review_C=.1,
                        review_parameters=dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                            weights=model.coef_[0].tolist(), bias=float(model.intercept_[0])))
                    manifest["reliability_threshold" if kind == "system1" else "threshold"] = threshold
                    # `files` describes the hash-bound parent files, not copies in this child.
                    (saved / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                    outcomes = []
                    for i, row in enumerate(rows):
                        review = bool(predictions[i] == "oos" or scores[i] < threshold or (eligible is not None and not eligible[i]))
                        outcomes.append(dict(group=row["group"], truth=row["label"], suggestion=str(predictions[i]),
                            category=None if review else str(predictions[i]), needsReview=review, score=float(scores[i])))
                    result = dict(dataset=dataset, kind=kind, condition=condition, folder=name, manifest=manifest,
                        manifest_sha256=digest(saved / "manifest.json"), review_fit_ms=fit_ms, evidence=evidence,
                        metrics=quality(outcomes), quality=frontier_result, outcomes=outcomes)
                    report["results"].append(result)
                    (folder / "development.json").write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps({key: result[key] for key in ("dataset", "kind", "condition", "metrics", "review_fit_ms")}), flush=True)
                del parent
                gc.collect()


def measure(folder):
    frozen(False)
    source = json.loads((folder / "development.json").read_text())
    assert source["protocol_sha256"] == digest(PROTOCOL)
    assert source["comparison_report_sha256"] == digest(HERE / "results/boundary-runtime.json")
    report = dict(scope="experiment 2f complete development adapters; not qualification", qualified=False,
        os_network_denial_errno=deny_network_control(), teacher_calls=0, response_cache=False, receipts=False,
        process_import_preflight_ms=(time.perf_counter() - STARTED) * 1000,
        environment=dict(platform=platform.platform(), python=platform.python_version(), machine=platform.machine()),
        packages={name: importlib.metadata.version(name) for name in
                  ("numpy", "scipy", "scikit-learn", "onnxruntime", "tokenizers", "threadpoolctl")}, results=[])
    with (folder / "runtime.json").open("x") as stream, (folder / "runtime.jsonl").open("x") as journal, threadpool_limits(limits=1):
        for selected in source["results"]:
            begin = time.perf_counter()
            candidate = (PolynomialCandidate if selected["kind"] == "system1" else PolynomialBaseline)(folder / selected["folder"])
            load_ms = (time.perf_counter() - begin) * 1000
            assert candidate.identity == selected["manifest_sha256"]
            rows = load_splits(selected["dataset"])["development"]
            payload = {key: candidate.manifest[key] for key in ("categories", "instructions")}
            for row in rows[:100]:
                candidate.classify(dict(payload, text=row["prompt"]))
            outcomes, times, mismatches = [], [], []
            for row, expected in zip(rows, selected["outcomes"], strict=True):
                begin = time.perf_counter()
                try:
                    value = candidate.classify(dict(payload, text=row["prompt"]))
                except Exception as exc:
                    value = dict(candidate=candidate.identity, needsReview=True, category=None, suggestion=None,
                                 teacherCalls=0, error=type(exc).__name__)
                elapsed = (time.perf_counter() - begin) * 1000
                if any(value[key] != expected[key] for key in ("suggestion", "needsReview", "category")):
                    mismatches.append(row["group"])
                times.append(elapsed)
                outcome = dict(group=row["group"], truth=row["label"], latency_ms=elapsed, **value)
                outcomes.append(outcome)
                journal.write(json.dumps(dict(folder=selected["folder"], **outcome)) + "\n")
                journal.flush()
            metrics = quality(outcomes)
            errors = sum("error" in row for row in outcomes)
            parent_bytes = sum(p.stat().st_size for p in candidate.parent_folder.iterdir() if p.is_file())
            new_bytes = (folder / selected["folder"] / "manifest.json").stat().st_size
            result = dict(dataset=selected["dataset"], kind=selected["kind"], condition=selected["condition"],
                folder=selected["folder"], manifest=candidate.manifest, candidate_sha256=candidate.identity,
                metrics=metrics, errors=errors, qualified=False,
                latency=dict(requests=len(rows), warmups=100, p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95))),
                development_gates=gates(metrics, float(np.percentile(times, 95)), errors),
                load_ms=load_ms, artifact_bytes=parent_bytes + new_bytes, additional_review_bytes=new_bytes,
                external_encoder_bytes=sum(candidate.manifest.get("encoder", {}).get(k, 0) for k in ("model_bytes", "tokenizer_bytes")),
                selection_runtime_mismatches=mismatches,
                accepted_errors=[r for r in outcomes if not r["needsReview"] and r["category"] != r["truth"]],
                per_intent={label: quality([r for r in outcomes if r["truth"] == label]) for label in sorted({r["truth"] for r in outcomes})},
                outcomes=outcomes)
            report["results"].append(result)
            print(json.dumps({key: result[key] for key in ("dataset", "kind", "condition", "metrics", "latency", "selection_runtime_mismatches")}), flush=True)
            del candidate
            gc.collect()
        report["comparisons"] = []
        previous = json.loads((HERE / "results/boundary-runtime.json").read_text())
        for prior in previous["comparisons"]:
            winner = prior["selected_development_candidate"]
            incumbent = winner.copy()
            for result in report["results"]:
                if (result["dataset"], result["kind"]) != (prior["dataset"], prior["kind"]):
                    continue
                valid = all(value for key, value in result["development_gates"].items() if key != "coverage_at_least_80")
                if valid and not result["selection_runtime_mismatches"] and result["metrics"]["supported_coverage"]["numerator"] > winner["metrics"]["supported_coverage"]["numerator"]:
                    winner = {key: result[key] for key in ("folder", "candidate_sha256", "metrics", "latency")}
                    winner["source"] = result["condition"]
            report["comparisons"].append(dict(dataset=prior["dataset"], kind=prior["kind"], incumbent=incumbent,
                selected_development_candidate=winner, qualified=False, incumbent_timings_from_prior_run=True))
        stream.write(json.dumps(report, indent=2) + "\n")
    if any(row["selection_runtime_mismatches"] for row in report["results"]):
        raise RuntimeError("Selection/runtime mismatch; failed result retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "measure"))
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    (fit if args.action == "fit" else measure)(args.folder)
