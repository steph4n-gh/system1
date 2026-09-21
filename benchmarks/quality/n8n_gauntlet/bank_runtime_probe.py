"""Build/reload the banking development candidate and measure its full runtime."""
import argparse
import gc
import hashlib
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from crossfit_probe import probabilities
from develop import OUTPUT, frontier, load_splits
from encoder_probe import Encoder, PreparedFeatures
from reliability_probe import reliability_features
from runtime_probe import LocalCandidate, digest
from system1 import ChoiceField, CompiledSystemOneModel, DecisionSchema, System1Engine, SystemOneCompiler


def main(threads=1):
    data = load_splits("banking77")
    original = data["fit"]
    augmented = original + json.loads((OUTPUT / "contrast-lessons.json").read_text())["lessons"]
    labels = sorted({r["label"] for r in original})
    yi = np.asarray([labels.index(r["label"]) for r in original])
    ya = np.asarray([labels.index(r["label"]) for r in augmented])
    schema = type("Intent", (DecisionSchema,), {"intent": ChoiceField(options=labels)})
    folder = OUTPUT / f"bank-runtime-candidate-{threads}"
    folder.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        encoder = Encoder("bge-small")
        x = {}
        for split, rows in {**data, "augmented": augmented}.items():
            key = hashlib.sha256((encoder.projector_digest() + json.dumps(rows, sort_keys=True)).encode()).hexdigest()
            x[split] = np.load(OUTPUT / f"features-{key}.npy", allow_pickle=False)
        prepared = PreparedFeatures(augmented + data["calibration"], np.concatenate([x["augmented"], x["calibration"]]), encoder)
        compiler = SystemOneCompiler(schema, projector=prepared, choice_solver="logistic", regularization=.01)
        calibration = {"intent": [(r["prompt"], r["label"]) for r in data["calibration"]]}
        folds = np.zeros(len(original), dtype=int)
        for label in labels:
            ordered = sorted([i for i, r in enumerate(original) if r["label"] == label], key=lambda i: original[i]["group"])
            for j, i in enumerate(ordered):
                folds[i] = j % 3
        started = time.perf_counter()
        cf, cy = np.zeros((len(original), 7)), np.zeros(len(original), dtype=bool)
        for fold in range(3):
            train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            model = compiler.compile({"intent": [(original[i]["prompt"], original[i]["label"]) for i in train]},
                                     augment=False, calibration_exemplars=calibration)
            values = probabilities(model.heads["intent"], x["fit"][valid])
            cf[valid] = reliability_features(values, x["fit"][valid] @ x["fit"][train].T, yi[train])
            cy[valid] = values.argmax(axis=1) == yi[valid]
        model = compiler.compile({"intent": [(r["prompt"], r["label"]) for r in augmented]},
                                 augment=False, calibration_exemplars=calibration)
        values = probabilities(model.heads["intent"], x["calibration"])
        meta = reliability_features(values, x["calibration"] @ x["augmented"].T, ya)
        targets = values.argmax(axis=1) == np.asarray([labels.index(r["label"]) for r in data["calibration"]])
        scaler = StandardScaler().fit(np.concatenate([cf, meta]))
        gate = LogisticRegression(C=.1, max_iter=1000).fit(scaler.transform(np.concatenate([cf, meta])), np.concatenate([cy, targets]))
        teaching_ms = (time.perf_counter() - started) * 1000
        # Use the chosen serving thread count in the saved external-projector identity.
        serving_encoder = Encoder("bge-small", threads=threads)
        saved = CompiledSystemOneModel(model.schema, model.heads, dimension=384,
            projector=serving_encoder, metadata=model.metadata, use_cache=False)
        saved.save(folder / "intent.s1m")
        order = np.argsort(ya, kind="stable")
        offsets = np.concatenate([[0], np.cumsum(np.bincount(ya, minlength=len(labels)))])
        np.savez_compressed(folder / "scope.npz", prototypes=x["augmented"][order], class_offsets=offsets,
            mean=scaler.mean_, scale=scaler.scale_, weights=gate.coef_[0], bias=gate.intercept_[0])
        manifest = dict(status="unqualified-development-candidate", dataset="banking77", guard="reliability",
            encoder=serving_encoder.identity, files={name: digest(folder / name) for name in ("intent.s1m", "scope.npz")},
            categories={label: label.replace("_", " ") for label in labels},
            instructions="Choose one supported banking intent. Uncertain requests require review.",
            alpha=.05, reliability_threshold=.9169134348334932)
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        candidate = LocalCandidate(folder)
        payload = dict(categories=manifest["categories"], instructions=manifest["instructions"])
        # Select uncertainty settings using development only; full timed runs follow.
        scores, guesses, eligible = [], [], {alpha: [] for alpha in (.05, .075, .1, .125, .15)}
        for row, vector in zip(data["development"], x["development"]):
            for alpha in eligible:
                decision = candidate.engine.decide(row["prompt"], embedding=vector, alpha=alpha, record_receipt=False)
                eligible[alpha].append(not decision.is_ambiguous)
                if alpha == .05:
                    scores.append(candidate.reliability([decision.probabilities["intent"][label] for label in labels], vector))
                    guesses.append(decision.values["intent"])
        policies = [dict(alpha=alpha, quality=frontier([r["label"] for r in data["development"]], guesses, scores, eligible=mask))
                    for alpha, mask in eligible.items()]
        chosen = max(policies, key=lambda p: p["quality"]["best_development_coverage_at_quality_targets"]["coverage"])
        threshold = chosen["quality"]["best_development_coverage_at_quality_targets"]["threshold"]
        if threshold is None:
            raise RuntimeError("No usable development policy")
        lower = max((value for value in scores if value < threshold), default=threshold)
        manifest.update(alpha=chosen["alpha"], reliability_threshold=(threshold + lower) / 2)
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        del candidate
        candidate = LocalCandidate(folder)
        for row in data["development"][:100]:
            candidate.classify(dict(payload, text=row["prompt"]))
        timings, outcomes = [], []
        for row in data["development"]:
            started = time.perf_counter()
            result = candidate.classify(dict(payload, text=row["prompt"]))
            timings.append((time.perf_counter() - started) * 1000)
            outcomes.append(dict(group=row["group"], truth=row["label"], **result))
        counts = dict(supported=len(outcomes), accepted=sum(not r["needsReview"] for r in outcomes),
            accepted_correct=sum(not r["needsReview"] and r["category"] == r["truth"] for r in outcomes))
        report = dict(scope="banking complete adapter on development, not qualification", candidate=manifest,
            manifest_sha256=candidate.identity, policies=policies, counts=counts,
            teaching_ms_excluding_features=teaching_ms, latency=dict(requests=len(timings), warmups=100,
                p50_ms=float(np.median(timings)), p95_ms=float(np.percentile(timings, 95))),
            receipts=False, response_cache=False, outcomes=outcomes)
        (OUTPUT / f"bank-runtime-development-{threads}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k not in ("candidate", "outcomes")}, indent=2), flush=True)
        del candidate, saved, model, compiler, prepared, encoder, serving_encoder
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1, choices=(1, 2, 4))
    main(parser.parse_args().threads)
