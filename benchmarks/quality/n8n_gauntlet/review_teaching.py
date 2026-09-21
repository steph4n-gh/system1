"""Run the 18 preregistered development review-head comparisons.

Never opens official tests or the human OOS reserve. Frozen intent heads and
encoders are unchanged. See REVIEW_TEACHING_PROTOCOL.md for scope and limitations.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from crossfit_probe import probabilities
from develop import OUTPUT, frontier, load_splits
from encoder_probe import Encoder, PreparedFeatures
from reliability_probe import reliability_features
from runtime_probe import LocalCandidate
from system1 import ChoiceField, CompiledSystemOneModel, DecisionSchema, System1Engine, SystemOneCompiler

HERE = Path(__file__).resolve().parent


def features(rows, encoder, *, single=False, evidence=None):
    started = time.perf_counter()
    key = hashlib.sha256((("single-request-v1:" if single else "") + encoder.projector_digest() + json.dumps(rows, sort_keys=True)).encode()).hexdigest()
    path = OUTPUT / f"features-{key}.npy"
    cached = path.exists()
    if not cached:
        vectors = (np.stack([encoder.project(r["prompt"]) for r in rows]) if single else
                   np.concatenate([encoder.project_batch([r["prompt"] for r in rows[i:i + 32]])
                                   for i in range(0, len(rows), 32)]))
        np.save(path, vectors, allow_pickle=False)
        print(f"Encoded {len(rows)} teaching rows in {time.perf_counter() - started:.2f}s", flush=True)
    if evidence is not None:
        evidence.append(dict(rows=len(rows), cache_hit=cached, mode="single" if single else "batch32",
                             preparation_ms=(time.perf_counter() - started) * 1000))
    return np.load(path, allow_pickle=False)


def meta_features(values, x, prototypes, prototype_labels):
    # Bound temporary cosine matrices; never treat this cached-vector research
    # path as end-to-end inference latency.
    return np.concatenate([reliability_features(values[i:i + 256], x[i:i + 256] @ prototypes.T, prototype_labels)
                           for i in range(0, len(values), 256)])


def run(dataset, report, *, single=False):
    is_bank = dataset == "banking77"
    data = load_splits(dataset)
    original = data["fit"]
    augmented = original + (json.loads((HERE / "results/contrast-lessons.json").read_text())["lessons"] if is_bank else [])
    negative = []
    if not is_bank:
        expected = json.loads((HERE / "review-data-manifest.json").read_text())["files"]["negative-fit"]
        raw = (OUTPUT / "review-negative-fit.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError("Changed negative teaching data")
        negative = json.loads(raw)
        assert len(negative) == expected["rows"] == 2000
    folder = HERE / "artifacts" / ("banking-development" if is_bank else "clinc-development")
    candidate = LocalCandidate(folder)
    encoder = Encoder("bge-small" if is_bank else "minilm")
    labels = candidate.model.heads["intent"].options
    yi = np.asarray([labels.index(r["label"]) for r in original])
    ya = np.asarray([labels.index(r["label"]) for r in augmented])
    all_rows = dict(data, augmented=augmented)
    if negative:
        all_rows["negative"] = negative
    preparation = []
    x = {split: features(rows, encoder, single=single, evidence=preparation) for split, rows in all_rows.items()}
    schema = type("Intent", (DecisionSchema,), {"intent": ChoiceField(options=labels)})
    prepared = PreparedFeatures(augmented + data["calibration"], np.concatenate([x["augmented"], x["calibration"]]), encoder)
    compiler = SystemOneCompiler(schema, projector=prepared, choice_solver="logistic", regularization=.01 if is_bank else .1)
    calibration = {"intent": [(r["prompt"], r["label"]) for r in data["calibration"]]}
    folds = np.zeros(len(original), dtype=int)
    for label in labels:
        indices = sorted([i for i, r in enumerate(original) if r["label"] == label], key=lambda i: original[i]["group"])
        for position, index in enumerate(indices):
            folds[index] = position % 3
    started = time.perf_counter()
    cf = np.zeros((len(original), 7))
    cp = np.zeros(len(original), dtype=int)
    for fold in range(3):
        train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        model = compiler.compile({"intent": [(original[i]["prompt"], original[i]["label"]) for i in train]},
                                 augment=False, calibration_exemplars=calibration)
        values = probabilities(model.heads["intent"], x["fit"][valid])
        cf[valid] = meta_features(values, x["fit"][valid], x["fit"][train], yi[train])
        cp[valid] = values.argmax(axis=1)
        print(f"{dataset}: completed review-teaching fold {fold + 1}/3", flush=True)
    cy = (cp == yi) & np.asarray([r["label"] != "oos" for r in original])
    replacement_sha = None
    if single:
        model = compiler.compile({"intent": [(r["prompt"], r["label"]) for r in augmented]},
                                 augment=False, calibration_exemplars=calibration)
        candidate.model = CompiledSystemOneModel(model.schema, model.heads, dimension=384,
            projector=candidate.encoder, metadata=model.metadata, use_cache=False)
        candidate.engine = System1Engine(candidate.model.schema, model=candidate.model, strict_mode=True, use_cache=False)
        path = OUTPUT / f"{report['file_prefix']}-{dataset}-intent.s1m"
        candidate.model.save(path)
        replacement_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    values, meta, guesses = {}, {}, {}
    for split in ("calibration", "development") + (("negative",) if negative else ()):
        values[split] = probabilities(candidate.model.heads["intent"], x[split])
        meta[split] = meta_features(values[split], x[split], x["augmented"], ya)
        guesses[split] = values[split].argmax(axis=1)
    calibration_labels = np.asarray([labels.index(r["label"]) for r in data["calibration"]])
    cal_correct = (guesses["calibration"] == calibration_labels) & np.asarray([r["label"] != "oos" for r in data["calibration"]])
    eligible = [not candidate.engine.decide(row["prompt"], embedding=vector,
                alpha=candidate.manifest["alpha"], record_receipt=False).is_ambiguous
                for row, vector in zip(data["development"], x["development"], strict=True)]
    pred = np.asarray(labels)[guesses["development"]]
    report["datasets"][dataset] = dict(frozen_manifest_sha256=candidate.identity, head_and_crossfit_ms=(time.perf_counter() - started) * 1000,
        feature_preparation=preparation, replacement_intent_sha256=replacement_sha,
        original_fit_rows=len(original), final_intent_fit_rows=len(augmented), folds=3,
        generated_examples_in_fold_heads=0, alpha=candidate.manifest["alpha"],
        fold_groups=[[original[i]["group"] for i in np.flatnonzero(folds == fold)] for fold in range(3)])
    best = None
    for negatives in ((2000,) if single else ((0,) if is_bank else (0, 2000))):
        teaching = np.concatenate([cf, meta["calibration"]] + ([meta["negative"]] if negatives else []))
        teaching_guesses = np.concatenate([cp, guesses["calibration"]] + ([guesses["negative"]] if negatives else []))
        target = np.concatenate([cy, cal_correct] + ([np.zeros(negatives, dtype=bool)] if negatives else []))
        scaler = StandardScaler().fit(teaching)
        scaled = scaler.transform(teaching)
        dev_scaled = scaler.transform(meta["development"])
        for use_class in ((True,) if single else (False, True)):
            train = np.concatenate([scaled, np.eye(len(labels))[teaching_guesses]], axis=1) if use_class else scaled
            dev = np.concatenate([dev_scaled, np.eye(len(labels))[guesses["development"]]], axis=1) if use_class else dev_scaled
            for c in ((.1,) if single else (.01, .1, 1.0)):
                fit_started = time.perf_counter()
                gate = LogisticRegression(C=c, max_iter=1000).fit(train, target)
                gate_fit_ms = (time.perf_counter() - fit_started) * 1000
                if gate.n_iter_.max() >= 1000:
                    raise RuntimeError("Review head failed to converge")
                scores = gate.predict_proba(dev)[:, 1]
                result = dict(dataset=dataset, config=dict(extra_negatives=negatives, predicted_class_feature=use_class, C=c),
                    gate_fit_ms=gate_fit_ms,
                    review_teaching_rows=len(target), negative_targets=int((~target).sum()),
                    quality=frontier([r["label"] for r in data["development"]], pred, scores, eligible=eligible))
                report["results"].append(result)
                print(json.dumps(result), flush=True)
                chosen = result["quality"]["best_development_coverage_at_quality_targets"]
                if best is None or chosen["accepted"] > best["quality"]["best_development_coverage_at_quality_targets"]["accepted"]:
                    best = result
                    threshold = chosen["threshold"]
                    if threshold is not None:
                        lower = scores[scores < threshold]
                        threshold = float((threshold + lower.max()) / 2) if len(lower) else float(threshold)
                        guard = OUTPUT / f"{report['file_prefix']}-{dataset}-guard.npz"
                        np.savez_compressed(guard, mean=scaler.mean_, scale=scaler.scale_, weights=gate.coef_[0], bias=gate.intercept_[0])
                        report["datasets"][dataset]["selected"] = dict(config=result["config"], threshold=threshold,
                            guard_sha256=hashlib.sha256(guard.read_bytes()).hexdigest(),
                            outcomes=[dict(group=row["group"], truth=row["label"], prediction=str(pred[i]), score=float(scores[i]),
                                           needs_review=bool(not eligible[i] or pred[i] == "oos" or scores[i] < threshold))
                                      for i, row in enumerate(data["development"])])
                (HERE / "results" / f"{report['file_prefix']}-teaching-development.json").write_text(json.dumps(report, indent=2) + "\n")
    del candidate, model, compiler, prepared, encoder
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--consistent-clinc", action="store_true", help="Fixed experiment 2b; keeps 2a results intact")
    single = parser.parse_args().consistent_clinc
    protocol = "CONSISTENT_FEATURES_PROTOCOL.md" if single else "REVIEW_TEACHING_PROTOCOL.md"
    report = dict(scope=f"experiment {'2b' if single else '2a'}: development only; no official/reserve scoring",
                  experiment="2b" if single else "2a", file_prefix="consistent-review" if single else "review",
                  protocol_sha256=hashlib.sha256((HERE / protocol).read_bytes()).hexdigest(),
                  datasets={}, results=[], live_teacher_calls=0)
    with threadpool_limits(limits=1):
        for dataset in (("clinc150",) if single else ("clinc150", "banking77")):
            run(dataset, report, single=single)
