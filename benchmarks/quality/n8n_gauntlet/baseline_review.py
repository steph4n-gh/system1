"""Matched conventional review teaching; development only, never official tests.

Run under OS network denial. Each fitting run requires a new output folder.
The original frozen baseline implementation and artifacts remain untouched.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

IMPORTS_STARTED = time.perf_counter()
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from baseline import Baseline, probabilities
from develop import OUTPUT, ROOT, frontier, load_splits
from evaluate import deny_network_control, frozen_state, gates, quality
from reliability_probe import reliability_features
NUMERICAL_IMPORT_MS = (time.perf_counter() - IMPORTS_STARTED) * 1000

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "BASELINE_REVIEW_PROTOCOL.md"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReviewBaseline(Baseline):
    """Original saved TF-IDF head, with a numerical/category review head."""

    def __init__(self, folder):
        folder = Path(folder)
        manifest = json.loads((folder / "manifest.json").read_text())
        self.base_folder = folder / manifest["base_artifact"]
        super().__init__(self.base_folder)
        if self.identity != manifest["parent_manifest_sha256"]:
            raise ValueError("Original baseline changed")
        if any(manifest[key] != self.manifest[key] for key in ("categories", "instructions")):
            raise ValueError("Review contract differs from original intent contract")
        parameters = manifest["review_parameters"]
        if set(parameters) != {"mean", "scale", "gate_weights", "gate_bias"}:
            raise ValueError("Unexpected review parameter names")
        for key, value in parameters.items():
            setattr(self, key, np.asarray(value, dtype=np.float64))
        expected = 7 + (len(self.labels) if manifest["predicted_class_feature"] else 0)
        if (self.mean.shape != (7,) or self.scale.shape != (7,) or self.gate_weights.shape != (1, expected)
                or self.gate_bias.shape != (1,) or np.any(self.scale <= 0)
                or not all(np.isfinite(value).all() for value in (self.mean, self.scale, self.gate_weights, self.gate_bias))):
            raise ValueError("Invalid review parameters")
        self.folder, self.manifest, self.identity = folder, manifest, digest(folder / "manifest.json")

    def classify(self, payload):
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise ValueError("Expected text")
        if not payload["text"].strip() or len(payload["text"]) > 4000:
            raise ValueError("Text must contain 1 to 4000 characters")
        if any(payload.get(key) != self.manifest[key] for key in ("categories", "instructions")):
            return dict(candidate=self.identity, needsReview=True, category=None, reason="changed_contract", teacherCalls=0)
        x = self.vectorizer.transform([payload["text"]])
        values = probabilities(x @ self.transposed_weights + self.bias)[0]
        chosen = int(values.argmax())
        ordered = np.sort(values)
        features = [np.log(max(ordered[-1], 1e-12) / max(1 - ordered[-1], 1e-12)),
                    ordered[-1] - ordered[-2], -(values * np.log(np.maximum(values, 1e-12))).sum()]
        cosines = (x @ self.prototypes).toarray()[0]
        classes = [np.sort(cosines[start:end]) for start, end in zip(self.offsets[:-1], self.offsets[1:])]
        for count in (1, 5):
            similarity = np.asarray([part[-count:].mean() for part in classes])
            own = similarity[chosen].copy()
            similarity[chosen] = -np.inf
            features.extend([own, own - similarity.max()])
        logit = float(((np.asarray(features) - self.mean) / self.scale) @ self.gate_weights[0, :7] + self.gate_bias[0])
        if self.manifest["predicted_class_feature"]:
            logit += float(self.gate_weights[0, 7 + chosen])
        score = float(1 / (1 + np.exp(-np.clip(logit, -700, 700))))
        prediction = str(self.labels[chosen])
        review = prediction == "oos" or score < self.manifest["threshold"]
        return dict(candidate=self.identity, needsReview=bool(review), category=None if review else prediction,
                    suggestion=prediction, confidence=float(ordered[-1]), acceptance_score=score, teacherCalls=0)


def meta_features(values, x, prototypes, labels):
    return np.concatenate([reliability_features(values[i:i + 256], (x[i:i + 256] @ prototypes.T).toarray(), labels)
                           for i in range(0, len(values), 256)])


def midpoint(scores, threshold):
    if threshold is None:
        return 2.0  # Explicitly reject everything if no positive acceptance qualifies.
    lower = scores[scores < threshold]
    return float((threshold + lower.max()) / 2) if len(lower) else float(threshold)


def fit_one(dataset, folder, report):
    started = time.perf_counter()
    data = load_splits(dataset)
    original = data["fit"]
    augmented = original + (json.loads((HERE / "results/contrast-lessons.json").read_text())["lessons"]
                            if dataset == "banking77" else [])
    parent = HERE / "artifacts" / f"{dataset}-baseline"
    baseline = Baseline(parent)
    labels = list(baseline.labels)
    yi = np.asarray([labels.index(row["label"]) for row in original])
    ya = np.asarray([labels.index(row["label"]) for row in augmented])
    folds = np.zeros(len(original), dtype=int)
    for label in labels:
        indices = sorted([i for i, row in enumerate(original) if row["label"] == label], key=lambda i: original[i]["group"])
        for position, index in enumerate(indices):
            folds[index] = position % 3
    fold_groups = [[original[i]["group"] for i in np.flatnonzero(folds == fold)] for fold in range(3)]
    system1 = json.loads((HERE / "results/review-teaching-development.json").read_text())["datasets"][dataset]
    assert fold_groups == system1["fold_groups"]
    cf, cp = np.zeros((len(original), 7)), np.zeros(len(original), dtype=int)
    fold_times = []
    for fold in range(3):
        begin = time.perf_counter()
        train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        fit = vectorizer.fit_transform([original[i]["prompt"] for i in train])
        valid_x = vectorizer.transform([original[i]["prompt"] for i in valid])
        model = LogisticRegression(C=10, max_iter=1000).fit(fit, yi[train])
        if model.n_iter_.max() >= 1000 or not np.array_equal(model.classes_, np.arange(len(labels))):
            raise RuntimeError("Fold head unconverged or missing a category")
        p = model.predict_proba(valid_x)
        cf[valid] = meta_features(p, valid_x, fit, yi[train])
        cp[valid] = p.argmax(axis=1)
        fold_times.append((time.perf_counter() - begin) * 1000)
        print(f"{dataset}: matched review fold {fold + 1}/3", flush=True)
    del vectorizer, fit, valid_x, model
    negative = []
    if dataset == "clinc150":
        expected = json.loads((HERE / "review-data-manifest.json").read_text())["files"]["negative-fit"]
        path = OUTPUT / "review-negative-fit.json"
        if digest(path) != expected["sha256"]:
            raise ValueError("Negative teaching data changed")
        negative = json.loads(path.read_text())
        assert len(negative) == expected["rows"] == 2000
    prototypes = baseline.vectorizer.transform([row["prompt"] for row in augmented])
    parts = {key: data[key] for key in ("calibration", "development")}
    if negative:
        parts["negative"] = negative
    meta, guesses, predictions = {}, {}, {}
    for split, rows in parts.items():
        x = baseline.vectorizer.transform([row["prompt"] for row in rows])
        p = probabilities(x @ baseline.transposed_weights + baseline.bias)
        meta[split] = meta_features(p, x, prototypes, ya)
        guesses[split] = p.argmax(axis=1)
        predictions[split] = baseline.labels[guesses[split]]
    cf_target = (cp == yi) & np.asarray([row["label"] != "oos" for row in original])
    cal_target = ((predictions["calibration"] == np.asarray([row["label"] for row in data["calibration"]]))
                  & np.asarray([row["label"] != "oos" for row in data["calibration"]]))
    payload = {key: baseline.manifest[key] for key in ("categories", "instructions")}
    incumbent = [baseline.classify(dict(payload, text=row["prompt"])) for row in data["development"]]
    dev_truth = [row["label"] for row in data["development"]]
    scores = np.asarray([row["acceptance_score"] for row in incumbent])
    accepted = np.asarray([not row["needsReview"] for row in incumbent])
    supported = np.asarray(dev_truth) != "oos"
    incumbent_metrics = quality([dict(truth=truth, **row) for truth, row in zip(dev_truth, incumbent, strict=True)])
    assert incumbent_metrics["accepted_accuracy"]["fraction"] >= .99
    assert (incumbent_metrics["oos_false_acceptance"]["fraction"] or 0) <= .01
    best = dict(config=dict(kind="original-incumbent", extra_negatives=0, predicted_class_feature=False,
                            C=baseline.manifest["gate_C"]), accepted=int((accepted & supported).sum()),
                threshold=baseline.manifest["threshold"], scores=scores, arrays=None)
    report["configurations"].append(dict(dataset=dataset, config=best["config"],
        threshold=best["threshold"], metrics=incumbent_metrics,
        quality=frontier(dev_truth, predictions["development"], scores)))
    for count in ((0, 2000) if negative else (0,)):
        teaching = np.concatenate([cf, meta["calibration"]] + ([meta["negative"]] if count else []))
        chosen = np.concatenate([cp, guesses["calibration"]] + ([guesses["negative"]] if count else []))
        target = np.concatenate([cf_target, cal_target] + ([np.zeros(count, dtype=bool)] if count else []))
        scaler = StandardScaler().fit(teaching)
        scaled, dev_scaled = scaler.transform(teaching), scaler.transform(meta["development"])
        for use_class in (False, True):
            train = np.concatenate([scaled, np.eye(len(labels))[chosen]], axis=1) if use_class else scaled
            dev = np.concatenate([dev_scaled, np.eye(len(labels))[guesses["development"]]], axis=1) if use_class else dev_scaled
            for c in (.01, .1, 1.0):
                begin = time.perf_counter()
                gate = LogisticRegression(C=c, max_iter=1000).fit(train, target)
                if gate.n_iter_.max() >= 1000:
                    raise RuntimeError("Review head did not converge")
                gate_ms = (time.perf_counter() - begin) * 1000
                scores = gate.predict_proba(dev)[:, 1]
                config = dict(kind="matched-review", extra_negatives=count, predicted_class_feature=use_class, C=c)
                result = dict(dataset=dataset, config=config, gate_fit_ms=gate_ms, review_teaching_rows=len(target),
                              negative_targets=int((~target).sum()), quality=frontier(dev_truth, predictions["development"], scores))
                report["configurations"].append(result)
                selected = result["quality"]["best_development_coverage_at_quality_targets"]
                if selected["accepted"] > best["accepted"]:
                    best = dict(config=config, accepted=selected["accepted"], threshold=midpoint(scores, selected["threshold"]), scores=scores.copy(),
                        arrays=dict(mean=scaler.mean_, scale=scaler.scale_, gate_weights=gate.coef_, gate_bias=gate.intercept_))
                print(json.dumps(dict(dataset=dataset, config=config, selected=selected)), flush=True)
                (folder / "development.json").write_text(json.dumps(report, indent=2) + "\n")
    saved_folder = folder / dataset
    saved_folder.mkdir()
    # Research run is standalone; publishing reuses the already-versioned base
    # directories, without duplicating ~40 MB of unchanged intent artifacts.
    shutil.copytree(parent, folder / parent.name)
    review = best["arrays"] or dict(mean=baseline.mean, scale=baseline.scale,
                                   gate_weights=baseline.gate_weights, gate_bias=baseline.gate_bias)
    manifest = dict(baseline.manifest, status="unqualified-development-matched-review-baseline",
        parent_manifest_sha256=baseline.identity, review_protocol_sha256=digest(PROTOCOL),
        base_artifact="../" + parent.name, review_parameters={key: value.tolist() for key, value in review.items()},
        predicted_class_feature=best["config"]["predicted_class_feature"], review_config=best["config"],
        gate_C=best["config"]["C"], threshold=best["threshold"])
    (saved_folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    saved = ReviewBaseline(saved_folder)
    outcomes = []
    mismatches = []
    for i, row in enumerate(data["development"]):
        value = saved.classify(dict(payload, text=row["prompt"]))
        expected_review = predictions["development"][i] == "oos" or best["scores"][i] < best["threshold"]
        if value["suggestion"] != predictions["development"][i] or value["needsReview"] != expected_review:
            mismatches.append(row["group"])
        outcomes.append(dict(group=row["group"], truth=row["label"], **value))
    report["selected"][dataset] = dict(manifest_sha256=saved.identity, manifest=manifest, review_config=best["config"],
        review_development_total_ms=(time.perf_counter() - started) * 1000, fold_fit_and_feature_ms=fold_times,
        original_fit_rows=len(original), final_intent_fit_rows=len(augmented), generated_examples_in_fold_heads=0,
        fold_groups=fold_groups, metrics=quality(outcomes), selection_runtime_mismatches=mismatches, outcomes=outcomes)
    (folder / "development.json").write_text(json.dumps(report, indent=2) + "\n")
    if mismatches:
        raise RuntimeError("Saved baseline routing differs from development selection; result retained")
    gc.collect()


def measure(folder):
    selection = json.loads((folder / "development.json").read_text())
    if selection["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Protocol changed")
    path = folder / "runtime.json"
    if path.exists():
        raise ValueError("Runtime report already exists")
    report = dict(scope="experiment 2c complete development adapters; not qualification", results=[], teacher_calls=0,
                  os_network_denial_errno=deny_network_control(), response_cache=False, receipts=False,
                  numerical_module_import_ms=NUMERICAL_IMPORT_MS)
    for dataset in ("clinc150", "banking77"):
        begin = time.perf_counter()
        candidate = ReviewBaseline(folder / dataset)
        load_ms = (time.perf_counter() - begin) * 1000
        assert candidate.identity == selection["selected"][dataset]["manifest_sha256"]
        data = load_splits(dataset)["development"]
        payload = {key: candidate.manifest[key] for key in ("categories", "instructions")}
        for row in data[:100]:
            candidate.classify(dict(payload, text=row["prompt"]))
        timings, outcomes = [], []
        for row in data:
            begin = time.perf_counter()
            value = candidate.classify(dict(payload, text=row["prompt"]))
            elapsed = (time.perf_counter() - begin) * 1000
            timings.append(elapsed)
            outcomes.append(dict(group=row["group"], truth=row["label"], latency_ms=elapsed, **value))
        same = all({k: v for k, v in row.items() if k != "latency_ms"} == expected
                   for row, expected in zip(outcomes, selection["selected"][dataset]["outcomes"], strict=True))
        metrics = quality(outcomes)
        p95 = float(np.percentile(timings, 95))
        report["results"].append(dict(dataset=dataset, manifest_sha256=candidate.identity, manifest=candidate.manifest,
            load_ms=load_ms, artifact_bytes=sum(p.stat().st_size for p in (folder / dataset).iterdir())
                + sum(p.stat().st_size for p in candidate.base_folder.iterdir()),
            additional_review_bytes=sum(p.stat().st_size for p in (folder / dataset).iterdir()), external_encoder_bytes=0,
            metrics=metrics, development_gates=gates(metrics, p95, 0), reload_identical=same,
            latency=dict(requests=len(data), warmups=100, p50_ms=float(np.median(timings)), p95_ms=p95), outcomes=outcomes))
        path.write_text(json.dumps(report, indent=2) + "\n")
        if not same:
            raise RuntimeError("Reload mismatch; evidence retained")
        print(json.dumps({k: v for k, v in report["results"][-1].items() if k not in ("outcomes", "manifest")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "measure"))
    parser.add_argument("--folder", type=Path, default=OUTPUT / "matched-baseline-review")
    args = parser.parse_args()
    frozen_state(False)
    denied = deny_network_control()
    with threadpool_limits(limits=1):
        if args.action == "measure":
            measure(args.folder)
            return
        relative = str(PROTOCOL.relative_to(ROOT))
        revision = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", relative], cwd=ROOT, text=True).strip()
        if subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=ROOT) != PROTOCOL.read_bytes():
            raise ValueError("Protocol must be committed before fitting")
        args.folder.mkdir(parents=True, exist_ok=False)
        report = dict(scope="experiment 2c development only; no official/reserve scoring", protocol_sha256=digest(PROTOCOL),
            protocol_revision=revision, os_network_denial_errno=denied, live_teacher_calls=0, configurations=[], selected={},
            environment=dict(platform=platform.platform(), python=platform.python_version(),
                versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")}))
        for dataset in ("clinc150", "banking77"):
            fit_one(dataset, args.folder, report)


if __name__ == "__main__":
    main()
