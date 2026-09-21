"""Development-only comparison. Never opens an official test split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from system1 import ChoiceField, DecisionSchema, SystemOneCompiler
from system1.core.text import TfidfProjector

OUTPUT = ROOT / ".system1/n8n-gauntlet"


def load_splits(name):
    manifest = json.loads((ROOT / "benchmarks/quality/n8n_gauntlet/manifest.json").read_text())
    result = {}
    for split in ("fit", "calibration", "development"):
        raw = (OUTPUT / f"{name}-{split}.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["datasets"][name]["splits"][split]["sha256"]:
            raise ValueError(f"Changed {name} {split}")
        result[split] = json.loads(raw)
    return result


def frontier(labels, predictions, scores, *, eligible=None):
    """Optimistic development frontier, not a separately qualified threshold."""
    labels, predictions, scores = np.asarray(labels), np.asarray(predictions), np.asarray(scores)
    supported = labels != "oos"
    eligible = (predictions != "oos") & (True if eligible is None else np.asarray(eligible, dtype=bool))
    correct = predictions == labels
    best = dict(accepted=0, coverage=0.0, threshold=None)
    at_coverage = None
    for threshold in sorted(set(float(s) for s in scores), reverse=True):
        accept = (scores >= threshold) & eligible
        n = int(np.sum(accept & supported))
        if n == 0:
            continue
        c = int(np.sum(accept & supported & correct))
        false_accept = int(np.sum(accept & ~supported))
        row = dict(accepted=n, accepted_correct=c, accepted_accuracy=c / n,
                   coverage=n / int(supported.sum()), oos_false_accept=false_accept,
                   threshold=threshold)
        if at_coverage is None and row["coverage"] >= .8:
            at_coverage = row
        if c / n >= .99 and false_accept <= .01 * int((~supported).sum()) and n > best["accepted"]:
            best = row
    return dict(cases=int(supported.sum()), oos_cases=int((~supported).sum()),
                raw_correct=int(np.sum(supported & correct)),
                best_development_coverage_at_quality_targets=best,
                quality_at_80_percent_coverage=at_coverage)


def summarize(name, config, rows, labels, logits, fit_ms):
    logits = np.asarray(logits)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    pred = np.asarray(labels)[np.argmax(probabilities, axis=1)]
    truth = [r["label"] for r in rows]
    ordered = np.sort(probabilities, axis=1)
    result = dict(dataset=name, config=config, fit_ms=fit_ms,
                  probability=frontier(truth, pred, ordered[:, -1]),
                  margin=frontier(truth, pred, ordered[:, -1] - ordered[:, -2]))
    print(json.dumps(result), flush=True)
    return result


def main():
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    report = dict(scope="development only; optimistic frontiers are not qualification",
                  environment=dict(python=platform.python_version(), platform=platform.platform(),
                                   numpy=np.__version__, sklearn=sklearn.__version__), results=[])
    path = OUTPUT / "lexical-development.json"
    with threadpool_limits(limits=1):
        for name in ("clinc150", "banking77"):
            data = load_splits(name)
            fit = [(r["prompt"], r["label"]) for r in data["fit"]]
            calibration = [(r["prompt"], r["label"]) for r in data["calibration"]]
            texts, truth = zip(*fit)
            labels = sorted(set(truth))
            schema = type("Intent", (DecisionSchema,), {"intent": ChoiceField(options=labels)})
            for features in (2048, 4096):
                for regularization in (.1, 1.0):
                    start = time.perf_counter()
                    projector = TfidfProjector.fit(texts, max_features=features)
                    model = SystemOneCompiler(schema, projector=projector, regularization=regularization).compile(
                        {"intent": fit}, augment=False, calibration_exemplars={"intent": calibration})
                    fit_ms = (time.perf_counter() - start) * 1000
                    x = projector.project_batch([r["prompt"] for r in data["development"]])
                    head = model.heads["intent"]
                    logits = (x @ head.weights.T + head.biases) / (.25 * head.temperature)
                    config = dict(kind="system1-tfidf-ridge", features=features, regularization=regularization,
                                  augmentation=False)
                    report["results"].append(summarize(name, config, data["development"], head.options, logits, fit_ms))
                    path.write_text(json.dumps(report, indent=2) + "\n")
            for c in (1.0, 10.0):
                start = time.perf_counter()
                vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
                x = vectorizer.fit_transform(texts)
                model = LogisticRegression(C=c, max_iter=1000).fit(x, truth)
                fit_ms = (time.perf_counter() - start) * 1000
                logits = model.decision_function(vectorizer.transform([r["prompt"] for r in data["development"]]))
                config = dict(kind="sklearn-tfidf-logistic", features=x.shape[1], C=c, max_iter=1000)
                report["results"].append(summarize(name, config, data["development"], model.classes_, logits, fit_ms))
                path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
