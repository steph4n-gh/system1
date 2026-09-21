"""Evaluate the prospectively declared generated lessons on development only."""
import argparse
import hashlib
import gc
import json
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from develop import OUTPUT, frontier, load_splits, summarize
from encoder_probe import Encoder, PreparedFeatures
from reliability_probe import reliability_features
from system1 import ChoiceField, DecisionSchema, SystemOneCompiler


def main(length_feature=False):
    source = OUTPUT / "contrast-lessons.json"
    generated = json.loads(source.read_text())
    if not generated["lessons"]:
        raise ValueError("No generated lessons to evaluate")
    data = load_splits("banking77")
    original_fit = data["fit"]
    data["fit"] = original_fit + generated["lessons"]
    labels = sorted({r["label"] for r in original_fit})
    fit_labels = np.asarray([labels.index(r["label"]) for r in data["fit"]])
    schema = type("Intent", (DecisionSchema,), {"intent": ChoiceField(options=labels)})
    truth = [r["label"] for r in data["development"]]
    report = dict(scope="development only; public fitting labels plus API-generated contrast lessons",
                  teaching_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  original_examples=len(original_fit), added_examples=len(generated["lessons"]), results=[])
    report["length_feature"] = length_feature
    output = OUTPUT / ("contrast-length-development.json" if length_feature else "contrast-development.json")
    with threadpool_limits(limits=1):
        for name in ("minilm", "bge-small"):
            encoder = Encoder(name)
            features = {}
            for split, rows in data.items():
                key = hashlib.sha256((encoder.projector_digest() + json.dumps(rows, sort_keys=True)).encode()).hexdigest()
                path = OUTPUT / f"features-{key}.npy"
                if path.exists():
                    features[split] = np.load(path, allow_pickle=False)
                else:
                    if split != "fit":
                        raise FileNotFoundError("Run encoder_probe.py first to prepare development features")
                    base_key = hashlib.sha256((encoder.projector_digest() + json.dumps(original_fit, sort_keys=True)).encode()).hexdigest()
                    base = np.load(OUTPUT / f"features-{base_key}.npy", allow_pickle=False)
                    texts = [r["prompt"] for r in generated["lessons"]]
                    extra = np.concatenate([encoder.project_batch(texts[i:i + 32]) for i in range(0, len(texts), 32)])
                    features[split] = np.concatenate([base, extra])
                    np.save(path, features[split], allow_pickle=False)
            projector = PreparedFeatures(data["fit"] + data["calibration"],
                np.concatenate([features["fit"], features["calibration"]]), encoder)
            for regularization in (.01, .1, 1):
                start = time.perf_counter()
                model = SystemOneCompiler(schema, projector=projector, choice_solver="logistic", regularization=regularization).compile(
                    {"intent": [(r["prompt"], r["label"]) for r in data["fit"]]}, augment=False,
                    calibration_exemplars={"intent": [(r["prompt"], r["label"]) for r in data["calibration"]]})
                fit_ms = (time.perf_counter() - start) * 1000
                head = model.heads["intent"]
                probabilities, predictions, reliability = {}, {}, {}
                for split in ("calibration", "development"):
                    logits = (features[split] @ head.weights.T + head.biases).astype(np.float64) / (.25 * head.temperature)
                    values = np.exp(logits - logits.max(axis=1, keepdims=True))
                    probabilities[split] = values / values.sum(axis=1, keepdims=True)
                    predictions[split] = np.asarray(labels)[values.argmax(axis=1)]
                    reliability[split] = reliability_features(probabilities[split], features[split] @ features["fit"].T, fit_labels)
                    if length_feature:
                        reliability[split] = np.column_stack([reliability[split],
                            np.log1p([len(r["prompt"].split()) for r in data[split]])])
                ordered = np.sort(probabilities["development"], axis=1)
                scaler = StandardScaler().fit(reliability["calibration"])
                gate = LogisticRegression(C=.01, max_iter=1000).fit(scaler.transform(reliability["calibration"]),
                    predictions["calibration"] == np.asarray([r["label"] for r in data["calibration"]]))
                scores = gate.predict_proba(scaler.transform(reliability["development"]))[:, 1]
                result = dict(dataset="banking77", encoder=encoder.identity, fit_ms=fit_ms,
                    config=dict(kind="system1-logistic-generated-contrasts", regularization=regularization,
                                reliability_C=.01, length_feature=length_feature),
                    probability=frontier(truth, predictions["development"], ordered[:, -1]),
                    margin=frontier(truth, predictions["development"], ordered[:, -1] - ordered[:, -2]),
                    reliability=frontier(truth, predictions["development"], scores))
                cal_correct = predictions["calibration"] == np.asarray([r["label"] for r in data["calibration"]])
                lengths = np.asarray([len(r["prompt"].split()) for r in data["calibration"]])
                result["calibration_length_diagnostic"] = [dict(min_words=low, max_words=high,
                    requests=int(((lengths >= low) & (lengths <= high)).sum()),
                    correct=int((cal_correct & (lengths >= low) & (lengths <= high)).sum()))
                    for low, high in ((1, 5), (6, 10), (11, 20), (21, 1000))]
                report["results"].append(result)
                output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(result), flush=True)
            # Release native encoder sessions while threadpool controls are alive.
            del model, projector, encoder
            gc.collect()
        if length_feature:
            return  # The same baseline is already retained by the original run.
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        x = vectorizer.fit_transform([r["prompt"] for r in data["fit"]])
        evaluation = vectorizer.transform([r["prompt"] for r in data["development"]])
        for c in (1, 10):
            start = time.perf_counter()
            model = LogisticRegression(C=c, max_iter=1000).fit(x, [r["label"] for r in data["fit"]])
            elapsed = (time.perf_counter() - start) * 1000
            config = dict(kind="sklearn-tfidf-logistic-generated-contrasts", C=c, features=x.shape[1])
            report["results"].append(summarize("banking77", config, data["development"], model.classes_,
                                               model.decision_function(evaluation), elapsed))
            output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length-feature", action="store_true",
                        help="Additional development probe: teach reliability using request word count")
    main(parser.parse_args().length_feature)
