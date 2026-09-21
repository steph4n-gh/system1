"""Optional frozen-encoder development probe, outside the product dependency set."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from develop import OUTPUT, load_splits, summarize, frontier
from system1 import ChoiceField, DecisionSchema, SystemOneCompiler


class Encoder:
    SETTINGS = {
        "minilm": (384, "mean", 256, 0, "model_qint8_arm64.onnx"),
        "bge-small": (384, "cls", 512, 0, "model.onnx"),
        "mpnet-base": (768, "mean", 384, 1, "model_qint8_arm64.onnx"),
    }

    def __init__(self, name, *, threads=1):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        folder = OUTPUT / name
        self.dimension, self.pooling, max_tokens, pad_id, filename = self.SETTINGS[name]
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_tokens)
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>" if pad_id == 1 else "[PAD]")
        path = folder / "onnx" / filename
        options = ort.SessionOptions()
        if type(threads) is not int or not 1 <= threads <= 4:
            raise ValueError("Encoder threads must be an integer from one to four")
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.identity = dict(name=name, model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             tokenizer_sha256=hashlib.sha256((folder / "tokenizer.json").read_bytes()).hexdigest(),
                             model_bytes=path.stat().st_size, pooling=self.pooling,
                             tokenizer_bytes=(folder / "tokenizer.json").stat().st_size,
                             runtime=ort.__version__, threads=threads, provider="CPUExecutionProvider")

    def project_batch(self, texts):
        tokens = self.tokenizer.encode_batch(texts)
        feed = {"input_ids": np.asarray([t.ids for t in tokens], dtype=np.int64),
                "attention_mask": np.asarray([t.attention_mask for t in tokens], dtype=np.int64),
                "token_type_ids": np.asarray([t.type_ids for t in tokens], dtype=np.int64)}
        output = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
        if self.pooling == "mean":
            mask = feed["attention_mask"][..., None]
            vectors = (output * mask).sum(axis=1) / mask.sum(axis=1)
        else:
            vectors = output[:, 0]
        vectors = vectors.astype(np.float32)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)

    def project(self, text, recency_weighted=False):
        if recency_weighted:
            raise ValueError("Encoder does not support recency weights")
        return self.project_batch([text])[0]

    def projector_digest(self):
        return hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()


class PreparedFeatures:
    """Only for development compilation; measured inference uses the real encoder."""
    def __init__(self, rows, vectors, encoder):
        self.dimension = encoder.dimension
        self.features = {r["prompt"]: x for r, x in zip(rows, vectors)}
        self.encoder = encoder

    def project(self, text):
        if text not in self.features:
            self.features[text] = self.encoder.project(text)
        return self.features[text]


def main(name, extended=False):
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    start = time.perf_counter()
    encoder = Encoder(name)
    report = dict(scope="development only; optional pretrained encoder, no final test opened",
                  encoder=encoder.identity, load_ms=(time.perf_counter() - start) * 1000, results=[])
    path = OUTPUT / f"{name}{'-extended' if extended else ''}-development.json"
    with threadpool_limits(limits=1):
        for dataset in ("clinc150", "banking77"):
            data = load_splits(dataset)
            features = {}
            for split, rows in data.items():
                key = hashlib.sha256((encoder.projector_digest() + json.dumps(rows, sort_keys=True)).encode()).hexdigest()
                cache = OUTPUT / f"features-{key}.npy"
                if cache.exists():
                    features[split] = np.load(cache, allow_pickle=False)
                else:
                    vectors = []
                    started = time.perf_counter()
                    for i in range(0, len(rows), 32):
                        vectors.extend(encoder.project_batch([r["prompt"] for r in rows[i:i + 32]]))
                    features[split] = np.asarray(vectors)
                    np.save(cache, features[split], allow_pickle=False)
                    print(f"{dataset} {split}: encoded {len(rows)} in {time.perf_counter() - started:.1f}s", flush=True)
            prepared = PreparedFeatures(data["fit"] + data["calibration"],
                                        np.concatenate([features["fit"], features["calibration"]]), encoder)
            labels = sorted({r["label"] for r in data["fit"]})
            schema = type("Intent", (DecisionSchema,), {"intent": ChoiceField(options=labels)})
            for regularization in (() if extended else (.01, .1, 1.0)):
                start = time.perf_counter()
                model = SystemOneCompiler(schema, projector=prepared, regularization=regularization).compile(
                    {"intent": [(r["prompt"], r["label"]) for r in data["fit"]]}, augment=False,
                    calibration_exemplars={"intent": [(r["prompt"], r["label"]) for r in data["calibration"]]})
                ms = (time.perf_counter() - start) * 1000
                head = model.heads["intent"]
                logits = (features["development"] @ head.weights.T + head.biases) / (.25 * head.temperature)
                config = dict(kind="system1-frozen-encoder-ridge", encoder=name, regularization=regularization)
                report["results"].append(summarize(dataset, config, data["development"], head.options, logits, ms))
            for c in ((100.0, 1000.0) if extended else (1.0, 10.0)):
                start = time.perf_counter()
                model = LogisticRegression(C=c, max_iter=1000).fit(features["fit"], [r["label"] for r in data["fit"]])
                ms = (time.perf_counter() - start) * 1000
                logits = model.decision_function(features["development"])
                config = dict(kind="sklearn-frozen-encoder-logistic", encoder=name, C=c)
                report["results"].append(summarize(dataset, config, data["development"], model.classes_, logits, ms))
            # Similarity probe diagnoses whether the feature representation is limiting.
            cosines = features["development"] @ features["fit"].T
            class_scores = np.stack([cosines[:, [i for i, r in enumerate(data["fit"]) if r["label"] == label]].max(axis=1)
                                     for label in labels], axis=1)
            prediction = np.asarray(labels)[class_scores.argmax(axis=1)]
            ordered = np.sort(class_scores, axis=1)
            truth = [r["label"] for r in data["development"]]
            report["results"].append(dict(dataset=dataset, config=dict(kind="nearest-teaching-embedding", encoder=name),
                similarity=frontier(truth, prediction, ordered[:, -1]),
                margin=frontier(truth, prediction, ordered[:, -1] - ordered[:, -2])))
            # Actual individual encoding; never time cached development vectors.
            timings = []
            for i, row in enumerate(data["development"][:600]):
                start = time.perf_counter()
                encoder.project(row["prompt"])
                if i >= 100:
                    timings.append((time.perf_counter() - start) * 1000)
            report.setdefault("encoder_only_latency", {})[dataset] = {
                "requests": len(timings), "p50_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95))}
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report["results"][-1]), flush=True)
            print(json.dumps(report["encoder_only_latency"][dataset]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("encoder", choices=list(Encoder.SETTINGS))
    parser.add_argument("--extended", action="store_true", help="Additional C=100/1000 head fits; retain original report")
    args = parser.parse_args()
    main(args.encoder, args.extended)
