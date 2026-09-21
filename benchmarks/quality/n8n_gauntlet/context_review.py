"""Experiment 2g: teach review using the classifier's existing input features."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from baseline_review import midpoint
from context_runtime import ContextCandidate
from context_scores import load_context, score
from develop import OUTPUT, ROOT, frontier, load_splits
from evaluate import deny_network_control, quality
from observed_regression import frozen
from polynomial_review import PolynomialBaseline, measure, prepare
from polynomial_scores import load_review, numerical_signals
from review_teaching import features
from runtime_probe import digest
from teach_boundaries import committed

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "CONTEXT_REVIEW_PROTOCOL.md"
COMPARISON = HERE / "results/polynomial-runtime.json"


class ContextBaseline(PolynomialBaseline):
    def __init__(self, folder):
        super().__init__(folder)
        if self.manifest["quadratic"] is not False:
            raise ValueError("Context review requires linear numerical signals")
        self.context_weights = load_context(self.manifest, self.weights.shape[1])

    def review_score(self, values, x):
        signals, chosen = numerical_signals(values, (x @ self.prototypes).toarray()[0], self.offsets)
        return score(signals, chosen, float((x @ self.context_weights)[0]), self.review_parameters)


def input_features(dataset, kind, parent):
    started = time.perf_counter()
    data = load_splits(dataset)
    rows = data["fit"] + data["calibration"]
    if dataset == "clinc150":
        path = OUTPUT / "review-negative-fit.json"
        expected = json.loads((HERE / "review-data-manifest.json").read_text())["files"]["negative-fit"]["sha256"]
        if digest(path) != expected:
            raise ValueError("Changed negative teaching source")
        rows += json.loads(path.read_text())
    extraction = []
    if kind == "system1":
        # Use the existing per-split cache keys, all with single-request projection.
        parts = [data["fit"], data["calibration"]]
        if dataset == "clinc150":
            parts.append(rows[len(data["fit"]) + len(data["calibration"]):])
        teaching = np.concatenate([features(part, parent.encoder, single=True, evidence=extraction) for part in parts])
        development = features(data["development"], parent.encoder, single=True, evidence=extraction)
        norms = np.linalg.norm(teaching, axis=1)
    else:
        teaching = parent.vectorizer.transform([row["prompt"] for row in rows])
        development = parent.vectorizer.transform([row["prompt"] for row in data["development"]])
        norms = np.sqrt(teaching.multiply(teaching).sum(axis=1)).A1
    return teaching, development, dict(context_dimension=teaching.shape[1], context_rows=len(rows),
        context_preparation_ms=(time.perf_counter() - started) * 1000, context_feature_extraction=extraction,
        context_norm_range=[float(norms.min()), float(norms.max())],
        context_nonzero_values=int(teaching.nnz) if sparse.issparse(teaching) else int(np.count_nonzero(teaching)))


def fit(folder):
    files = [PROTOCOL, Path(__file__).resolve(), HERE / "context_runtime.py", HERE / "context_scores.py",
             HERE / "polynomial_review.py", HERE / "polynomial_runtime.py", HERE / "polynomial_scores.py",
             HERE / "results/polynomial-development.json", COMPARISON]
    revision = committed(files)
    frozen(False)
    denied = deny_network_control()
    folder.mkdir(parents=True, exist_ok=False)
    controls = json.loads((HERE / "results/polynomial-development.json").read_text())
    report = dict(scope="experiment 2g development only", qualified=False, source_revision=revision,
        protocol_sha256=digest(PROTOCOL), comparison_report_sha256=digest(COMPARISON),
        source_files={str(path.relative_to(ROOT)): digest(path) for path in files},
        os_network_denial_errno=denied, teacher_calls=0, new_api_cost=0, control_checks=[], results=[])
    with threadpool_limits(limits=1):
        for dataset in ("clinc150", "banking77"):
            for kind in ("system1", "baseline"):
                parent, labels, rows, teaching, predicted, target, dev, guesses, eligible, evidence = prepare(dataset, kind)
                control = next(r for r in controls["results"] if (r["dataset"], r["kind"], r["condition"]) == (dataset, kind, "linear-control"))
                for key in ("parent_manifest_sha256", "fold_groups", "review_rows", "negative_targets", "original_fit_rows"):
                    assert evidence[key] == control["evidence"][key], key
                mean, scale, weights, bias = load_review(control["manifest"], len(labels))
                control_scores = 1 / (1 + np.exp(-np.clip(((dev - mean) / scale) @ weights[:7] + weights[7 + guesses] + bias, -700, 700)))
                expected = np.asarray([row["score"] for row in control["outcomes"]])
                difference = float(np.max(np.abs(control_scores - expected)))
                check = dict(dataset=dataset, kind=kind, manifest_sha256=control["manifest_sha256"],
                    original_lineage_identical=True, max_score_difference=difference,
                    predictions_match=all(labels[g] == row["suggestion"] for g, row in zip(guesses, control["outcomes"], strict=True)))
                report["control_checks"].append(check)
                (folder / "development.json").write_text(json.dumps(report, indent=2) + "\n")
                if difference > 1e-10 or not check["predictions_match"]:
                    raise RuntimeError("Prior control was not reproduced; check retained before new fit")
                contexts, dev_contexts, context_evidence = input_features(dataset, kind, parent)
                assert contexts.shape[0] == len(target) and dev_contexts.shape[0] == len(rows)
                scaler = StandardScaler().fit(teaching)
                core = np.column_stack([scaler.transform(teaching), np.eye(len(labels))[predicted]])
                dev_core = np.column_stack([scaler.transform(dev), np.eye(len(labels))[guesses]])
                train = sparse.hstack([sparse.csr_matrix(core), contexts], format="csr") if sparse.issparse(contexts) else np.column_stack([core, contexts])
                development = sparse.hstack([sparse.csr_matrix(dev_core), dev_contexts], format="csr") if sparse.issparse(dev_contexts) else np.column_stack([dev_core, dev_contexts])
                started = time.perf_counter()
                model = LogisticRegression(C=.1, max_iter=1000).fit(train, target)
                fit_ms = (time.perf_counter() - started) * 1000
                if model.n_iter_.max() >= 1000:
                    raise RuntimeError("Context review did not converge")
                scores = model.predict_proba(development)[:, 1]
                predictions = np.asarray(labels)[guesses]
                selected = frontier([r["label"] for r in rows], predictions, scores, eligible=eligible)
                threshold = midpoint(scores, selected["best_development_coverage_at_quality_targets"]["threshold"])
                name = f"{dataset}-{kind}-context"
                saved = folder / name
                saved.mkdir()
                manifest = dict(parent.manifest, status="unqualified-input-context-review-development", dataset=dataset, kind=kind,
                    base_artifact=evidence["parent_folder"], parent_manifest_sha256=parent.identity,
                    protocol_sha256=digest(PROTOCOL), quadratic=False, review_C=.1,
                    review_parameters=dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                        weights=model.coef_[0, :7 + len(labels)].tolist(), bias=float(model.intercept_[0])),
                    context_dimension=int(contexts.shape[1]), context_weights=model.coef_[0, 7 + len(labels):].tolist())
                manifest["reliability_threshold" if kind == "system1" else "threshold"] = threshold
                (saved / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                outcomes = []
                for i, row in enumerate(rows):
                    review = bool(predictions[i] == "oos" or scores[i] < threshold or (eligible is not None and not eligible[i]))
                    outcomes.append(dict(group=row["group"], truth=row["label"], suggestion=str(predictions[i]),
                        category=None if review else str(predictions[i]), needsReview=review, score=float(scores[i])))
                result = dict(dataset=dataset, kind=kind, condition="input-context", folder=name, manifest=manifest,
                    manifest_sha256=digest(saved / "manifest.json"), review_fit_ms=fit_ms,
                    evidence=dict(evidence, **context_evidence), metrics=quality(outcomes), quality=selected, outcomes=outcomes)
                report["results"].append(result)
                (folder / "development.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({key: result[key] for key in ("dataset", "kind", "metrics", "review_fit_ms")}), flush=True)
                del parent
                gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "measure"))
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "fit":
        fit(args.folder)
    else:
        measure(args.folder, protocol=PROTOCOL, scope="experiment 2g complete development adapters; not qualification",
                candidate_classes=(ContextCandidate, ContextBaseline), comparison_path=COMPARISON)
