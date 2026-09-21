"""Save and exercise selected experiment-2a review heads on development only.

The original intent-head bytes and encoder identities are retained unchanged.
Neither official tests nor the unused human OOS reserve are opened or scored.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from threadpoolctl import threadpool_limits

from develop import OUTPUT, load_splits
from runtime_probe import LocalCandidate, digest

HERE = Path(__file__).resolve().parent


class ReviewCandidate(LocalCandidate):
    def reliability(self, probabilities, embedding):
        values = np.asarray(probabilities, dtype=np.float64)
        chosen = int(values.argmax())
        ordered = np.sort(values)
        features = [np.log(max(ordered[-1], 1e-12) / max(1 - ordered[-1], 1e-12)),
                    ordered[-1] - ordered[-2], -(values * np.log(np.maximum(values, 1e-12))).sum()]
        cosines = self.prototypes @ embedding
        classes = [np.sort(cosines[start:end]) for start, end in zip(self.class_offsets[:-1], self.class_offsets[1:])]
        for count in (1, 5):
            similarity = np.asarray([values[-count:].mean() for values in classes])
            own = similarity[chosen].copy()
            similarity[chosen] = -np.inf
            features.extend([own, own - similarity.max()])
        logit = ((np.asarray(features) - self.risk_mean) / self.risk_scale) @ self.risk_weights[:7] + self.risk_bias
        if self.manifest["predicted_class_feature"]:
            logit += self.risk_weights[7 + chosen]
        return float(1 / (1 + np.exp(-np.clip(logit, -700, 700))))


def prepare(dataset, source):
    consistent = source.get("experiment") == "2b"
    prefix = "consistent-review" if consistent else "review"
    is_bank = dataset == "banking77"
    original = HERE / "artifacts" / ("banking-development" if is_bank else "clinc-development")
    data = load_splits(dataset)
    selection = source["datasets"][dataset]["selected"]
    folder = OUTPUT / f"{prefix}-{dataset}-candidate"
    folder.mkdir(exist_ok=True)
    manifest = json.loads((original / "manifest.json").read_text())
    if digest(original / "manifest.json") != source["datasets"][dataset]["frozen_manifest_sha256"]:
        raise ValueError("Original intent candidate changed")
    intent = OUTPUT / f"{prefix}-{dataset}-intent.s1m" if consistent else original / "intent.s1m"
    expected_intent = source["datasets"][dataset]["replacement_intent_sha256"] if consistent else manifest["files"]["intent.s1m"]
    if digest(intent) != expected_intent:
        raise ValueError("Intent head changed")
    shutil.copyfile(intent, folder / "intent.s1m")
    guard = OUTPUT / f"{prefix}-{dataset}-guard.npz"
    if digest(guard) != selection["guard_sha256"]:
        raise ValueError("Selected review head changed")
    with np.load(guard, allow_pickle=False) as arrays:
        saved = {key: arrays[key] for key in arrays.files}
    if is_bank:
        with np.load(original / "scope.npz", allow_pickle=False) as arrays:
            saved.update(prototypes=arrays["prototypes"], class_offsets=arrays["class_offsets"])
    else:
        # These are the same fitting features used to teach review. Runtime
        # requests always execute the saved encoder again.
        key = hashlib.sha256((("single-request-v1:" if consistent else "") + hashlib.sha256(json.dumps(manifest["encoder"], sort_keys=True).encode()).hexdigest()
                              + json.dumps(data["fit"], sort_keys=True)).encode()).hexdigest()
        features = np.load(OUTPUT / f"features-{key}.npy", allow_pickle=False)
        labels = sorted(manifest["categories"])
        yi = np.asarray([labels.index(row["label"]) for row in data["fit"]])
        order = np.argsort(yi, kind="stable")
        saved.update(prototypes=features[order], class_offsets=np.concatenate([[0], np.cumsum(np.bincount(yi, minlength=len(labels)))]))
    np.savez_compressed(folder / "scope.npz", **saved)
    for key in ("probability_threshold", "density_threshold"):
        manifest.pop(key, None)
    manifest.update(status="unqualified-development-review-candidate", guard="category-reliability",
        predicted_class_feature=selection["config"]["predicted_class_feature"],
        reliability_threshold=selection["threshold"],
        base_intent_manifest_sha256=digest(original / "manifest.json"),
        review_protocol_sha256=source["protocol_sha256"], review_config=selection["config"],
        teaching_projection="single-request" if consistent else "batch32",
        files={name: digest(folder / name) for name in ("intent.s1m", "scope.npz")})
    assert manifest["files"]["intent.s1m"] == expected_intent
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--consistent-clinc", action="store_true")
    consistent = parser.parse_args().consistent_clinc
    prefix = "consistent-review" if consistent else "review"
    # Verify OS isolation without importing or calling a teacher.
    from evaluate import deny_network_control, quality, gates
    denied = deny_network_control()
    source = json.loads((HERE / "results" / f"{prefix}-teaching-development.json").read_text())
    protocol = "CONSISTENT_FEATURES_PROTOCOL.md" if consistent else "REVIEW_TEACHING_PROTOCOL.md"
    if digest(HERE / protocol) != source["protocol_sha256"]:
        raise ValueError("Review protocol changed")
    report = dict(scope=f"experiment {'2b' if consistent else '2a'} complete development adapters; not qualification", results=[],
                  os_network_denial_errno=denied, teacher_calls=0, response_cache=False, receipts=False)
    with threadpool_limits(limits=1):
        for dataset in (("clinc150",) if consistent else ("clinc150", "banking77")):
            folder = prepare(dataset, source)
            started = time.perf_counter()
            candidate = ReviewCandidate(folder)
            load_ms = (time.perf_counter() - started) * 1000
            payload = {key: candidate.manifest[key] for key in ("categories", "instructions")}
            development = load_splits(dataset)["development"]
            for row in development[:100]:
                candidate.classify(dict(payload, text=row["prompt"]))
            timings, outcomes = [], []
            for row in development:
                started = time.perf_counter()
                result = candidate.classify(dict(payload, text=row["prompt"]))
                elapsed = (time.perf_counter() - started) * 1000
                timings.append(elapsed)
                outcomes.append(dict(group=row["group"], truth=row["label"], latency_ms=elapsed, **result))
            metrics = quality(outcomes)
            p95 = float(np.percentile(timings, 95))
            result = dict(dataset=dataset, manifest_sha256=candidate.identity, manifest=candidate.manifest,
                metrics=metrics, development_gates=gates(metrics, p95, 0), load_ms=load_ms,
                artifact_bytes=sum(path.stat().st_size for path in folder.iterdir() if path.is_file()),
                latency=dict(requests=len(outcomes), warmups=100, p50_ms=float(np.median(timings)), p95_ms=p95),
                outcomes=outcomes)
            selected = source["datasets"][dataset]["selected"]["outcomes"]
            result["selection_runtime_mismatches"] = [row["group"] for row, expected in zip(outcomes, selected, strict=True)
                if row["group"] != expected["group"] or row["needsReview"] != expected["needs_review"]
                or row["suggestion"] != expected["prediction"]]
            report["results"].append(result)
            (HERE / "results" / f"{prefix}-runtime-development.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: value for key, value in result.items() if key not in ("manifest", "outcomes")}), flush=True)
            del candidate
            gc.collect()


if __name__ == "__main__":
    main()
