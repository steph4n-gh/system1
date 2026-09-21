"""Development-only check for conflicting local teaching evidence."""
import hashlib
import json

import numpy as np
from threadpoolctl import threadpool_limits

from develop import OUTPUT, frontier, load_splits
from encoder_probe import Encoder, PreparedFeatures
from system1 import ChoiceField, DecisionSchema, SystemOneCompiler


def main():
    report = dict(scope="development only; head predictions checked against nearby fitting labels", results=[])
    path = OUTPUT / "agreement-development.json"
    with threadpool_limits(limits=1):
        for name in ("minilm", "bge-small"):
            encoder = Encoder(name)
            dataset = "banking77"
            data = load_splits(dataset)
            features = {}
            for split, rows in data.items():
                key = hashlib.sha256((encoder.projector_digest() + json.dumps(rows, sort_keys=True)).encode()).hexdigest()
                features[split] = np.load(OUTPUT / f"features-{key}.npy", allow_pickle=False)
            labels = sorted({r["label"] for r in data["fit"]})
            fit_labels = np.asarray([labels.index(r["label"]) for r in data["fit"]])
            schema = type("Intent", (DecisionSchema,), {"intent": ChoiceField(options=labels)})
            projector = PreparedFeatures(data["fit"] + data["calibration"],
                np.concatenate([features["fit"], features["calibration"]]), encoder)
            model = SystemOneCompiler(schema, projector=projector, choice_solver="logistic", regularization=.1).compile(
                {"intent": [(r["prompt"], r["label"]) for r in data["fit"]]}, augment=False,
                calibration_exemplars={"intent": [(r["prompt"], r["label"]) for r in data["calibration"]]})
            head = model.heads["intent"]
            logits = (features["development"] @ head.weights.T + head.biases).astype(np.float64) / (.25 * head.temperature)
            probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            chosen = probabilities.argmax(axis=1)
            predictions = np.asarray(labels)[chosen]
            ordered = np.sort(probabilities, axis=1)
            cosines = features["development"] @ features["fit"].T
            for count in (1, 3, 5):
                # Compare the strongest independent fitting evidence for each label.
                class_similarity = np.stack([np.sort(cosines[:, fit_labels == i], axis=1)[:, -count:].mean(axis=1)
                                             for i in range(len(labels))], axis=1)
                own = class_similarity[np.arange(len(chosen)), chosen]
                competing = class_similarity.copy()
                competing[np.arange(len(chosen)), chosen] = -np.inf
                gap = own - competing.max(axis=1)
                for threshold in (0, .005, .01, .02, .04):
                    config = dict(kind="system1-logistic-local-agreement", encoder=name,
                                  neighbors_per_class=count, agreement_margin=threshold, regularization=.1)
                    result = dict(dataset=dataset, config=config)
                    for score, values in [("probability", ordered[:, -1]), ("margin", ordered[:, -1] - ordered[:, -2])]:
                        result[score] = frontier([r["label"] for r in data["development"]], predictions, values,
                                                eligible=gap >= threshold)
                    report["results"].append(result)
                    print(json.dumps(result), flush=True)
            path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
