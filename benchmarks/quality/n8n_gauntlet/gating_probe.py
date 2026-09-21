"""Development-only density rejection; no test inputs or test labels are read."""
import hashlib
import json

import numpy as np
from threadpoolctl import threadpool_limits

from develop import OUTPUT, frontier, load_splits
from encoder_probe import Encoder, PreparedFeatures
from system1 import ChoiceField, DecisionSchema, SystemOneCompiler


def main():
    report = dict(scope="development frontier, not qualification", results=[])
    path = OUTPUT / "gating-development.json"
    with threadpool_limits(limits=1):
        encoder = Encoder("minilm")
        for dataset in ("clinc150", "banking77"):
            data = load_splits(dataset)
            x = {}
            for split, rows in data.items():
                key = hashlib.sha256((encoder.projector_digest() + json.dumps(rows, sort_keys=True)).encode()).hexdigest()
                x[split] = np.load(OUTPUT / f"features-{key}.npy", allow_pickle=False)
            fit_labels = np.asarray([r["label"] for r in data["fit"]])
            labels = sorted(set(fit_labels))
            schema = type("Intent", (DecisionSchema,), {"intent": ChoiceField(options=labels)})
            projector = PreparedFeatures(data["fit"] + data["calibration"],
                                         np.concatenate([x["fit"], x["calibration"]]), encoder)
            model = SystemOneCompiler(schema, projector=projector, choice_solver="logistic", regularization=.1).compile(
                {"intent": [(r["prompt"], r["label"]) for r in data["fit"]]}, augment=False,
                calibration_exemplars={"intent": [(r["prompt"], r["label"]) for r in data["calibration"]]})
            head = model.heads["intent"]
            probabilities, predictions, confidence = {}, {}, {}
            for split in ("calibration", "development"):
                logits = (x[split] @ head.weights.T + head.biases).astype(np.float64) / (.25 * head.temperature)
                values = np.exp(logits - logits.max(axis=1, keepdims=True))
                probabilities[split] = values / values.sum(axis=1, keepdims=True)
                predictions[split] = np.asarray(labels)[values.argmax(axis=1)]
                confidence[split] = probabilities[split].max(axis=1)
            truth = [r["label"] for r in data["development"]]
            report["results"].append(dict(dataset=dataset, config=dict(kind="system1-logistic-minilm"),
                quality=frontier(truth, predictions["development"], confidence["development"])))
            supported = [label for label in labels if label != "oos"]
            centers = np.stack([x["fit"][fit_labels == label].mean(axis=0) for label in supported])
            residuals = np.concatenate([x["fit"][fit_labels == label] - center for label, center in zip(supported, centers)])
            covariance = residuals.T @ residuals / len(residuals)
            for shrinkage in (.01, .1, 1.0):
                shrunk = (1 - shrinkage) * covariance + shrinkage * np.eye(384) * np.trace(covariance) / 384
                whitening = np.linalg.inv(np.linalg.cholesky(shrunk)).T
                transformed_centers = centers @ whitening
                scores = {}
                for split in ("calibration", "development"):
                    vectors = x[split] @ whitening
                    distances = np.maximum(0, (vectors ** 2).sum(axis=1)[:, None]
                        + (transformed_centers ** 2).sum(axis=1)[None, :] - 2 * vectors @ transformed_centers.T)
                    scores[split] = -distances.min(axis=1)
                supported_calibration = np.asarray([r["label"] != "oos" for r in data["calibration"]])
                for quantile in (.01, .025, .05, .1):
                    threshold = float(np.quantile(scores["calibration"][supported_calibration], quantile))
                    # Rejected candidates cannot re-enter the acceptance frontier.
                    result = dict(dataset=dataset, config=dict(kind="system1-logistic-minilm-density",
                        shrinkage=shrinkage, calibration_quantile=quantile, density_threshold=threshold),
                        quality=frontier(truth, predictions["development"], confidence["development"],
                                         eligible=scores["development"] >= threshold))
                    report["results"].append(result)
                    print(json.dumps(result), flush=True)
            path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
