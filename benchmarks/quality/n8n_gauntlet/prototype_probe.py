"""Development probe: combine a taught head with nearest-example class scores.

Both signals share one frozen MiniLM pass. Examples are classifier prototypes,
not a response cache. Calibration teaches review; development measures it.
"""
import gc
import hashlib
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from develop import OUTPUT, frontier, load_splits
from encoder_probe import Encoder, PreparedFeatures
from reliability_probe import reliability_features
from system1 import ChoiceField, DecisionSchema, SystemOneCompiler


def main():
    data = load_splits("banking77")
    data["fit"] += json.loads((OUTPUT / "contrast-lessons.json").read_text())["lessons"]
    labels = sorted({r["label"] for r in data["fit"]})
    fit_labels = np.asarray([labels.index(r["label"]) for r in data["fit"]])
    schema = type("Intent", (DecisionSchema,), {"intent": ChoiceField(options=labels)})
    report = dict(scope="development only; logit combination, not qualified", results=[])
    path = OUTPUT / "prototype-development.json"
    with threadpool_limits(limits=1):
        encoder = Encoder("minilm")
        x = {}
        for split, rows in data.items():
            key = hashlib.sha256((encoder.projector_digest() + json.dumps(rows, sort_keys=True)).encode()).hexdigest()
            x[split] = np.load(OUTPUT / f"features-{key}.npy", allow_pickle=False)
        projector = PreparedFeatures(data["fit"] + data["calibration"], np.concatenate([x["fit"], x["calibration"]]), encoder)
        model = SystemOneCompiler(schema, projector=projector, choice_solver="logistic", regularization=.1).compile(
            {"intent": [(r["prompt"], r["label"]) for r in data["fit"]]}, augment=False,
            calibration_exemplars={"intent": [(r["prompt"], r["label"]) for r in data["calibration"]]})
        head = model.heads["intent"]
        cosines = {split: x[split] @ x["fit"].T for split in ("calibration", "development")}
        class_scores = {split: np.stack([values[:, fit_labels == i].max(axis=1) for i in range(len(labels))], axis=1)
                        for split, values in cosines.items()}
        logits = {split: (x[split] @ head.weights.T + head.biases).astype(np.float64) / (.25 * head.temperature)
                  for split in cosines}
        for weight in (.25, .5, .75):
            for scale in (10, 20):
                probabilities, predictions, meta = {}, {}, {}
                for split in logits:
                    mixed = (1 - weight) * logits[split] + weight * scale * class_scores[split]
                    values = np.exp(mixed - mixed.max(axis=1, keepdims=True))
                    probabilities[split] = values / values.sum(axis=1, keepdims=True)
                    predictions[split] = np.asarray(labels)[values.argmax(axis=1)]
                    meta[split] = reliability_features(probabilities[split], cosines[split], fit_labels)
                correct = predictions["calibration"] == np.asarray([r["label"] for r in data["calibration"]])
                scaler = StandardScaler().fit(meta["calibration"])
                gate = LogisticRegression(C=.01, max_iter=1000).fit(scaler.transform(meta["calibration"]), correct)
                scores = gate.predict_proba(scaler.transform(meta["development"]))[:, 1]
                ordered = np.sort(probabilities["development"], axis=1)
                truth = [r["label"] for r in data["development"]]
                result = dict(dataset="banking77", encoder=encoder.identity,
                    config=dict(kind="linear-and-prototype-logits", prototype_weight=weight, prototype_scale=scale),
                    probability=frontier(truth, predictions["development"], ordered[:, -1]),
                    margin=frontier(truth, predictions["development"], ordered[:, -1] - ordered[:, -2]),
                    reliability=frontier(truth, predictions["development"], scores))
                report["results"].append(result)
                path.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(result), flush=True)
        del model, projector, encoder
        gc.collect()


if __name__ == "__main__":
    main()
