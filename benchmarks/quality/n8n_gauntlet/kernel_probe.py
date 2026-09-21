"""Development-only nonlinear features from a fixed subset of teaching vectors."""
import hashlib
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from develop import OUTPUT, load_splits, summarize
from encoder_probe import Encoder


def main():
    report = dict(scope="development only; learned radial-basis features, no response cache", results=[])
    path = OUTPUT / "kernel-development.json"
    with threadpool_limits(limits=1):
        for name in ("minilm", "bge-small"):
            encoder = Encoder(name)
            for dataset in ("clinc150", "banking77"):
                data = load_splits(dataset)
                x = {}
                for split in ("fit", "development"):
                    rows = data[split]
                    key = hashlib.sha256((encoder.projector_digest() + json.dumps(rows, sort_keys=True)).encode()).hexdigest()
                    x[split] = np.load(OUTPUT / f"features-{key}.npy", allow_pickle=False)
                labels = sorted({r["label"] for r in data["fit"]})
                anchors = [i for label in labels for i in [j for j, r in enumerate(data["fit"]) if r["label"] == label][:16]]
                landmarks = x["fit"][anchors]
                cosines = {split: x[split] @ landmarks.T for split in x}
                for gamma in (5, 10, 20):
                    features = {}
                    for split in cosines:
                        values = np.exp(gamma * (cosines[split] - 1))
                        features[split] = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-9)
                    for c in (10, 100):
                        start = time.perf_counter()
                        model = LogisticRegression(C=c, max_iter=1000).fit(features["fit"], [r["label"] for r in data["fit"]])
                        elapsed = (time.perf_counter() - start) * 1000
                        logits = model.decision_function(features["development"])
                        config = dict(kind="radial-basis-logistic", encoder=name, landmarks_per_label=16,
                                      dimension=len(landmarks), gamma=gamma, C=c)
                        report["results"].append(summarize(dataset, config, data["development"], model.classes_, logits, elapsed))
                        path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
