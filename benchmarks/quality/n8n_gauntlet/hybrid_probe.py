"""Development-only test of lexical features beside a small frozen encoder."""
import hashlib
import json
import time

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from develop import OUTPUT, load_splits, summarize
from encoder_probe import Encoder
from system1.core.text import TfidfProjector


def main():
    report = dict(scope="development only; lexical + frozen encoder, no test split opened", results=[])
    path = OUTPUT / "hybrid-development.json"
    with threadpool_limits(limits=1):
        encoder = Encoder("minilm")
        for dataset in ("clinc150", "banking77"):
            data = load_splits(dataset)
            lexical = TfidfProjector.fit([r["prompt"] for r in data["fit"]], max_features=4096)
            dense, sparse = {}, {}
            for split in ("fit", "development"):
                rows = data[split]
                key = hashlib.sha256((encoder.projector_digest() + json.dumps(rows, sort_keys=True)).encode()).hexdigest()
                dense[split] = csr_matrix(np.load(OUTPUT / f"features-{key}.npy", allow_pickle=False))
                sparse[split] = csr_matrix(lexical.project_batch([r["prompt"] for r in rows]))
            for lexical_weight in (.25, .5, 1.0):
                x = {split: hstack([dense[split], lexical_weight * sparse[split]], format="csr") / np.sqrt(1 + lexical_weight ** 2)
                     for split in dense}
                for c in (10.0, 100.0):
                    start = time.perf_counter()
                    model = LogisticRegression(C=c, max_iter=1000).fit(x["fit"], [r["label"] for r in data["fit"]])
                    elapsed = (time.perf_counter() - start) * 1000
                    logits = model.decision_function(x["development"])
                    config = dict(kind="sklearn-hybrid-logistic", encoder="minilm", lexical_features=4096,
                                  lexical_weight=lexical_weight, C=c)
                    report["results"].append(summarize(dataset, config, data["development"], model.classes_, logits, elapsed))
                    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
