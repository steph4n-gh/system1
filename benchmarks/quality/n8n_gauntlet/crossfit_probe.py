"""Development-only reliability teaching from out-of-fold intent mistakes.

Three deterministic folds partition original fitting rows by label/hash. Each
row's reliability lesson uses a head and nearest examples that exclude its fold.
Generated examples are excluded from ALL fold heads because their prompt lineage
includes original fitting examples. The final head may use the declared extra
lessons. Calibration and development remain separate; no test file is opened.
"""
import argparse
import gc
import hashlib
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from develop import OUTPUT, frontier, load_splits
from encoder_probe import Encoder, PreparedFeatures
from reliability_probe import reliability_features
from system1 import ChoiceField, DecisionSchema, SystemOneCompiler


def probabilities(head, features):
    logits = (features @ head.weights.T + head.biases).astype(np.float64) / (.25 * head.temperature)
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def main(refine=False):
    data = load_splits("banking77")
    original = data["fit"]
    generated = json.loads((OUTPUT / "contrast-lessons.json").read_text())["lessons"]
    labels = sorted({r["label"] for r in original})
    fit_labels = np.asarray([labels.index(r["label"]) for r in original])
    augmented = original + generated
    augmented_labels = np.asarray([labels.index(r["label"]) for r in augmented])
    schema = type("Intent", (DecisionSchema,), {"intent": ChoiceField(options=labels)})
    folds = np.zeros(len(original), dtype=int)
    for label in labels:
        ordered = sorted([i for i, r in enumerate(original) if r["label"] == label], key=lambda i: original[i]["group"])
        for j, i in enumerate(ordered):
            folds[i] = j % 3
    report = dict(scope="development only; out-of-fold reliability, no final test", results=[],
                  folds=3, generated_examples_used_in_fold_heads=0,
                  fold_groups=[[original[i]["group"] for i in np.flatnonzero(folds == fold)] for fold in range(3)])
    output = OUTPUT / ("crossfit-refined-development.json" if refine else "crossfit-development.json")
    with threadpool_limits(limits=1):
        for name in ("minilm", "bge-small"):
            encoder = Encoder(name)
            features = {}
            for split, rows in {**data, "augmented": augmented}.items():
                key = hashlib.sha256((encoder.projector_digest() + json.dumps(rows, sort_keys=True)).encode()).hexdigest()
                features[split] = np.load(OUTPUT / f"features-{key}.npy", allow_pickle=False)
            prepared = PreparedFeatures(augmented + data["calibration"],
                np.concatenate([features["augmented"], features["calibration"]]), encoder)
            calibration = {"intent": [(r["prompt"], r["label"]) for r in data["calibration"]]}
            for regularization in ((.01, .03, .1) if refine else (.01, .1)):
                start = time.perf_counter()
                compiler = SystemOneCompiler(schema, projector=prepared, choice_solver="logistic", regularization=regularization)
                cross_features = np.zeros((len(original), 7))
                cross_correct = np.zeros(len(original), dtype=bool)
                for fold in range(3):
                    train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                    model = compiler.compile({"intent": [(original[i]["prompt"], original[i]["label"]) for i in train]},
                                             augment=False, calibration_exemplars=calibration)
                    values = probabilities(model.heads["intent"], features["fit"][valid])
                    cross_features[valid] = reliability_features(values,
                        features["fit"][valid] @ features["fit"][train].T, fit_labels[train])
                    cross_correct[valid] = values.argmax(axis=1) == fit_labels[valid]
                model = compiler.compile({"intent": [(r["prompt"], r["label"]) for r in augmented]},
                                         augment=False, calibration_exemplars=calibration)
                values, meta = {}, {}
                for split in ("calibration", "development"):
                    values[split] = probabilities(model.heads["intent"], features[split])
                    meta[split] = reliability_features(values[split],
                        features[split] @ features["augmented"].T, augmented_labels)
                cal_correct = values["calibration"].argmax(axis=1) == np.asarray([labels.index(r["label"]) for r in data["calibration"]])
                pred = np.asarray(labels)[values["development"].argmax(axis=1)]
                for include_calibration in ((True,) if refine else (False, True)):
                    gate_features = np.concatenate([cross_features, meta["calibration"]]) if include_calibration else cross_features
                    gate_truth = np.concatenate([cross_correct, cal_correct]) if include_calibration else cross_correct
                    scaler = StandardScaler().fit(gate_features)
                    for c in ((.01, .1, 1) if refine else (.01,)):
                        gate = LogisticRegression(C=c, max_iter=1000).fit(scaler.transform(gate_features), gate_truth)
                        scores = gate.predict_proba(scaler.transform(meta["development"]))[:, 1]
                        result = dict(dataset="banking77", encoder=encoder.identity,
                            config=dict(kind="crossfit-reliability", regularization=regularization, C=c,
                                        include_calibration=include_calibration),
                            head_and_crossfit_ms=(time.perf_counter() - start) * 1000,
                            reliability_lessons=len(gate_truth), observed_errors=int((~gate_truth).sum()),
                            quality=frontier([r["label"] for r in data["development"]], pred, scores))
                        report["results"].append(result)
                        output.write_text(json.dumps(report, indent=2) + "\n")
                        print(json.dumps(result), flush=True)
            del model, compiler, prepared, encoder
            gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refine", action="store_true", help="Additional regularization fits; retains the original report")
    main(parser.parse_args().refine)
