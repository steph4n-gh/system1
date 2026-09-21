"""Development-only reliability teaching from the reserved calibration fold.

The intent head sees fitting rows only. A second, small logistic head learns
whether that frozen head was right on calibration examples. Development labels
only measure/select configurations, as in the other probes. No test file opens.
This is an exploratory rejection policy, not qualified or added to the product.
"""
import hashlib
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from develop import OUTPUT, frontier, load_splits
from encoder_probe import Encoder, PreparedFeatures
from system1 import ChoiceField, DecisionSchema, SystemOneCompiler


def reliability_features(probabilities, cosines, fit_labels):
    """Seven fixed confidence/agreement features; no evaluation labels."""
    chosen = probabilities.argmax(axis=1)
    ordered = np.sort(probabilities, axis=1)
    features = [np.log(np.maximum(ordered[:, -1], 1e-12) / np.maximum(1 - ordered[:, -1], 1e-12)),
                ordered[:, -1] - ordered[:, -2],
                -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1)]
    for count in (1, 5):
        similarity = np.stack([np.sort(cosines[:, fit_labels == i], axis=1)[:, -count:].mean(axis=1)
                               for i in range(probabilities.shape[1])], axis=1)
        own = similarity[np.arange(len(chosen)), chosen].copy()
        similarity[np.arange(len(chosen)), chosen] = -np.inf
        features.extend([own, own - similarity.max(axis=1)])
    return np.stack(features, axis=1)


def main():
    report = dict(scope="development only; calibration-trained reliability gate", results=[])
    path = OUTPUT / "reliability-development.json"
    with threadpool_limits(limits=1):
        data = load_splits("banking77")
        labels = sorted({r["label"] for r in data["fit"]})
        fit_labels = np.asarray([labels.index(r["label"]) for r in data["fit"]])
        schema = type("Intent", (DecisionSchema,), {"intent": ChoiceField(options=labels)})
        for name in ("minilm", "bge-small"):
            encoder = Encoder(name)
            x = {}
            for split, rows in data.items():
                key = hashlib.sha256((encoder.projector_digest() + json.dumps(rows, sort_keys=True)).encode()).hexdigest()
                x[split] = np.load(OUTPUT / f"features-{key}.npy", allow_pickle=False)
            projector = PreparedFeatures(data["fit"] + data["calibration"],
                np.concatenate([x["fit"], x["calibration"]]), encoder)
            model = SystemOneCompiler(schema, projector=projector, choice_solver="logistic", regularization=.1).compile(
                {"intent": [(r["prompt"], r["label"]) for r in data["fit"]]}, augment=False,
                calibration_exemplars={"intent": [(r["prompt"], r["label"]) for r in data["calibration"]]})
            head = model.heads["intent"]
            predictions, reliability = {}, {}
            for split in ("calibration", "development"):
                logits = (x[split] @ head.weights.T + head.biases).astype(np.float64) / (.25 * head.temperature)
                probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                chosen = probabilities.argmax(axis=1)
                predictions[split] = np.asarray(labels)[chosen]
                cosines = x[split] @ x["fit"].T
                reliability[split] = (reliability_features(probabilities, cosines, fit_labels), np.eye(len(labels))[chosen])
            correctness = predictions["calibration"] == np.asarray([r["label"] for r in data["calibration"]])
            scaler = StandardScaler().fit(reliability["calibration"][0])
            scaled = {split: scaler.transform(values[0]) for split, values in reliability.items()}
            for use_class in (False, True):
                inputs = {split: np.concatenate([values, reliability[split][1]], axis=1) if use_class else values
                          for split, values in scaled.items()}
                for c in (.01, .1, 1, 10):
                    gate = LogisticRegression(C=c, max_iter=1000).fit(inputs["calibration"], correctness)
                    scores = gate.predict_proba(inputs["development"])[:, 1]
                    result = dict(dataset="banking77", encoder=encoder.identity,
                        config=dict(kind="calibration-logistic-reliability", head_regularization=.1, C=c, class_features=use_class),
                        quality=frontier([r["label"] for r in data["development"]], predictions["development"], scores))
                    report["results"].append(result)
                    path.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
