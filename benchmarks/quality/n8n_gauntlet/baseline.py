"""Freeze a conventional TF-IDF/logistic baseline using development data only.

Calibration may teach the same seven-feature reliability gate explored for
System1. Development chooses C, probability/margin/reliability and a threshold.
The saved baseline has no pickle, network dependency or response cache.
"""
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from develop import load_splits, frontier
from reliability_probe import reliability_features

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probabilities(logits):
    values = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


class Baseline:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / "manifest.json").read_text())
        self.identity = digest(self.folder / "manifest.json")
        for name, expected in self.manifest["files"].items():
            if digest(self.folder / name) != expected:
                raise ValueError("Baseline artifact changed")
        vocab = json.loads((self.folder / "vocabulary.json").read_text())
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, vocabulary=vocab)
        with np.load(self.folder / "weights.npz", allow_pickle=False) as arrays:
            self.vectorizer.idf_ = arrays["idf"]
            self.weights, self.bias = arrays["weights"], arrays["bias"]
            # Sparse single-row multiplication must not transpose/copy the full
            # dense classifier matrix for every request.
            self.transposed_weights = np.ascontiguousarray(self.weights.T)
            self.labels = arrays["labels"]
            if self.manifest["policy"] == "reliability":
                self.mean, self.scale = arrays["mean"], arrays["scale"]
                self.gate_weights, self.gate_bias = arrays["gate_weights"], arrays["gate_bias"]
                fit_labels = arrays["fit_labels"]
                order = np.argsort(fit_labels, kind="stable")
                self.offsets = np.concatenate([[0], np.cumsum(np.bincount(fit_labels, minlength=len(self.labels)))])
                self.prototypes = sparse.load_npz(self.folder / "prototypes.npz")[order].T.tocsr()

    def classify(self, payload):
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise ValueError("Expected text")
        if not payload["text"].strip() or len(payload["text"]) > 4000:
            raise ValueError("Text must contain 1 to 4000 characters")
        if payload.get("categories") != self.manifest["categories"] or payload.get("instructions") != self.manifest["instructions"]:
            return dict(candidate=self.identity, needsReview=True, category=None, reason="changed_contract", teacherCalls=0)
        x = self.vectorizer.transform([payload["text"]])
        p = probabilities(x @ self.transposed_weights + self.bias)
        prediction = str(self.labels[p[0].argmax()])
        ordered = np.sort(p[0])
        if self.manifest["policy"] == "reliability":
            values = p[0]
            chosen = values.argmax()
            features = [np.log(max(ordered[-1], 1e-12) / max(1 - ordered[-1], 1e-12)),
                        ordered[-1] - ordered[-2], -(values * np.log(np.maximum(values, 1e-12))).sum()]
            cosines = (x @ self.prototypes).toarray()[0]
            classes = [np.sort(cosines[start:end]) for start, end in zip(self.offsets[:-1], self.offsets[1:])]
            for count in (1, 5):
                similarity = np.asarray([values[-count:].mean() for values in classes])
                own = similarity[chosen].copy()
                similarity[chosen] = -np.inf
                features.extend([own, own - similarity.max()])
            z = ((np.asarray([features]) - self.mean) / self.scale) @ self.gate_weights.T + self.gate_bias
            score = float((1 / (1 + np.exp(-np.clip(z, -700, 700))))[0, 0])
        elif self.manifest["policy"] == "margin":
            score = float(ordered[-1] - ordered[-2])
        else:
            score = float(ordered[-1])
        review = prediction == "oos" or score < self.manifest["threshold"]
        return dict(candidate=self.identity, needsReview=bool(review), category=None if review else prediction,
                    suggestion=prediction, confidence=float(ordered[-1]), acceptance_score=score, teacherCalls=0)


def main():
    report = dict(scope="development selection only; official tests unopened", configurations=[], selected={})
    report_path = HERE / "results/baseline-development.json"
    with threadpool_limits(limits=1):
        for dataset, candidate in (("clinc150", "clinc-development"), ("banking77", "banking-development")):
            data = load_splits(dataset)
            rows = list(data["fit"])
            if dataset == "banking77":
                rows += json.loads((HERE / "results/contrast-lessons.json").read_text())["lessons"]
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            started = time.perf_counter()
            fit = vectorizer.fit_transform([r["prompt"] for r in rows])
            vectorizer_ms = (time.perf_counter() - started) * 1000
            x = {split: vectorizer.transform([r["prompt"] for r in data[split]])
                 for split in ("calibration", "development")}
            labels = np.asarray(sorted({r["label"] for r in rows}))
            fit_labels = np.asarray([int(np.where(labels == r["label"])[0][0]) for r in rows])
            cosines = {split: (values @ fit.T).toarray() for split, values in x.items()}
            best = None
            for c in (1.0, 10.0):
                started = time.perf_counter()
                model = LogisticRegression(C=c, max_iter=1000).fit(fit, [r["label"] for r in rows])
                if model.n_iter_.max() >= 1000:
                    raise RuntimeError("Baseline did not converge")
                fit_ms = vectorizer_ms + (time.perf_counter() - started) * 1000
                assert np.array_equal(labels, model.classes_)
                p = {split: model.predict_proba(values) for split, values in x.items()}
                predictions = {split: labels[values.argmax(axis=1)] for split, values in p.items()}
                ordered = np.sort(p["development"], axis=1)
                policies = [("probability", None, ordered[:, -1], {}),
                            ("margin", None, ordered[:, -1] - ordered[:, -2], {})]
                features = {split: reliability_features(p[split], cosines[split], fit_labels) for split in x}
                scaler = StandardScaler().fit(features["calibration"])
                scaled = {split: scaler.transform(values) for split, values in features.items()}
                correct = predictions["calibration"] == np.asarray([r["label"] for r in data["calibration"]])
                for gate_c in (.01, .1, 1.0):
                    gate = LogisticRegression(C=gate_c, max_iter=1000).fit(scaled["calibration"], correct)
                    if gate.n_iter_.max() >= 1000:
                        raise RuntimeError("Baseline reliability gate did not converge")
                    policies.append(("reliability", gate_c, gate.predict_proba(scaled["development"])[:, 1],
                                     dict(mean=scaler.mean_, scale=scaler.scale_, gate_weights=gate.coef_,
                                          gate_bias=gate.intercept_, fit_labels=fit_labels)))
                for policy, gate_c, scores, extra in policies:
                    quality = frontier([r["label"] for r in data["development"]], predictions["development"], scores)
                    chosen = quality["best_development_coverage_at_quality_targets"]
                    result = dict(dataset=dataset, C=c, policy=policy, gate_C=gate_c,
                                  fit_rows=len(rows), fit_ms=fit_ms, features=fit.shape[1], quality=quality)
                    report["configurations"].append(result)
                    print(json.dumps(result), flush=True)
                    if best is None or chosen["accepted"] > best["result"]["quality"]["best_development_coverage_at_quality_targets"]["accepted"]:
                        threshold = chosen["threshold"]
                        if threshold is None:
                            threshold = 2.0
                        else:
                            lower = scores[scores < threshold]
                            if len(lower):
                                threshold = float((threshold + lower.max()) / 2)
                        best = dict(result=result, threshold=threshold,
                                    arrays=dict(weights=model.coef_.copy(), bias=model.intercept_.copy(),
                                                idf=vectorizer.idf_, labels=labels, **extra))
            folder = HERE / "artifacts" / f"{dataset}-baseline"
            folder.mkdir(exist_ok=True)
            (folder / "vocabulary.json").write_text(json.dumps(vectorizer.vocabulary_, sort_keys=True) + "\n")
            np.savez_compressed(folder / "weights.npz", **best["arrays"])
            names = ["weights.npz", "vocabulary.json"]
            if best["result"]["policy"] == "reliability":
                sparse.save_npz(folder / "prototypes.npz", fit)
                names.append("prototypes.npz")
            contract = json.loads((HERE / "artifacts" / candidate / "manifest.json").read_text())
            manifest = dict(status="unqualified-development-baseline", dataset=dataset,
                kind="sklearn-tfidf-logistic", C=best["result"]["C"], policy=best["result"]["policy"],
                gate_C=best["result"]["gate_C"], threshold=best["threshold"], fit_rows=len(rows),
                calibration_rows=len(data["calibration"]), fit_ms=best["result"]["fit_ms"],
                ngram_range=[1, 2], sublinear_tf=True, max_iter=1000,
                categories=contract["categories"], instructions=contract["instructions"],
                files={name: digest(folder / name) for name in names})
            (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            saved = Baseline(folder)
            payload = {key: manifest[key] for key in ("categories", "instructions")}
            outcomes = [dict(group=row["group"], truth=row["label"], **saved.classify(dict(payload, text=row["prompt"])))
                        for row in data["development"]]
            counts = dict(supported=sum(r["truth"] != "oos" for r in outcomes),
                accepted=sum(r["truth"] != "oos" and not r["needsReview"] for r in outcomes),
                accepted_correct=sum(r["truth"] != "oos" and not r["needsReview"] and r["category"] == r["truth"] for r in outcomes),
                oos_false_accept=sum(r["truth"] == "oos" and not r["needsReview"] for r in outcomes))
            expected = best["result"]["quality"]["best_development_coverage_at_quality_targets"]
            assert counts["accepted"] == expected["accepted"]
            assert counts["accepted_correct"] == expected["accepted_correct"]
            assert counts["oos_false_accept"] == expected["oos_false_accept"]
            report["selected"][dataset] = dict(manifest_sha256=saved.identity, manifest=manifest, counts=counts, outcomes=outcomes)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            del saved, cosines, model, features, fit, x
            gc.collect()


if __name__ == "__main__":
    main()
