"""Exercise the CLINC development candidate through saved System1 decisions.

No final test is opened. This is a development check, not qualification. It
measures actual encoder, engine, density check and response construction together.
"""
import gc
import hashlib
import json
import socket
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from develop import OUTPUT, load_splits
from encoder_probe import Encoder, PreparedFeatures
from system1 import ChoiceField, CompiledSystemOneModel, DecisionSchema, System1Engine, SystemOneCompiler


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LocalCandidate:
    """Actual saved candidate path; caller must honor review and qualification."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / "manifest.json").read_text())
        self.identity = digest(self.folder / "manifest.json")
        for name in ("intent.s1m", "scope.npz"):
            if digest(self.folder / name) != self.manifest["files"][name]:
                raise ValueError("Candidate artifact changed")
        self.encoder = Encoder(self.manifest["encoder"]["name"], threads=self.manifest["encoder"]["threads"])
        if self.encoder.identity != self.manifest["encoder"]:
            raise ValueError("Candidate encoder identity changed")
        self.model = CompiledSystemOneModel.load(self.folder / "intent.s1m", projector=self.encoder)
        self.model.use_cache = False
        self.engine = System1Engine(self.model.schema, model=self.model, strict_mode=True, use_cache=False)
        with np.load(self.folder / "scope.npz", allow_pickle=False) as arrays:
            if self.manifest.get("guard", "density") == "density":
                self.whitening = arrays["whitening"]
                self.centers = arrays["centers"]
                self.center_norms = (self.centers ** 2).sum(axis=1)
            else:
                self.prototypes = arrays["prototypes"]
                self.class_offsets = arrays["class_offsets"]
                self.risk_mean, self.risk_scale = arrays["mean"], arrays["scale"]
                self.risk_weights, self.risk_bias = arrays["weights"], float(arrays["bias"])

    def reliability(self, probabilities, embedding):
        """The same seven learned-review features used by the development probe."""
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
        logit = ((np.asarray(features) - self.risk_mean) / self.risk_scale) @ self.risk_weights + self.risk_bias
        return float(1 / (1 + np.exp(-np.clip(logit, -700, 700))))

    def classify(self, payload):
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise ValueError("Expected text")
        text = payload["text"]
        if not text.strip() or len(text) > 4000:
            raise ValueError("Text must contain 1 to 4000 characters")
        if payload.get("categories") != self.manifest["categories"] or payload.get("instructions") != self.manifest["instructions"]:
            return dict(candidate=self.identity, needsReview=True, reason="changed_contract", category=None, teacherCalls=0)
        decision = self.engine.decide(text, alpha=self.manifest["alpha"], record_receipt=False)
        confidence = decision.probabilities["intent"][decision.values["intent"]]
        if self.manifest.get("guard", "density") == "density":
            transformed = decision.embedding @ self.whitening
            distances = np.maximum(0, (transformed ** 2).sum() + self.center_norms - 2 * self.centers @ transformed)
            density = -float(distances.min())
            guarded = confidence < self.manifest["probability_threshold"] or density < self.manifest["density_threshold"]
            evidence = dict(density=density)
        else:
            reliability = self.reliability([decision.probabilities["intent"][label] for label in self.model.heads["intent"].options], decision.embedding)
            guarded = reliability < self.manifest["reliability_threshold"]
            evidence = dict(reliability=reliability)
        review = bool(decision.is_ambiguous or decision.values["intent"] == "oos" or guarded)
        return dict(candidate=self.identity, needsReview=review,
                    category=None if review else decision.values["intent"], suggestion=decision.values["intent"],
                    confidence=confidence, predictionSet=decision.conformal_sets["intent"],
                    **evidence, teacherCalls=0)


def main():
    folder = OUTPUT / "clinc-runtime-candidate"
    folder.mkdir(exist_ok=True)
    data = load_splits("clinc150")
    with threadpool_limits(limits=1):
        encoder = Encoder("minilm")
        features = {}
        for split, rows in data.items():
            key = hashlib.sha256((encoder.projector_digest() + json.dumps(rows, sort_keys=True)).encode()).hexdigest()
            features[split] = np.load(OUTPUT / f"features-{key}.npy", allow_pickle=False)
        labels = sorted({r["label"] for r in data["fit"]})
        schema = type("Intent", (DecisionSchema,), {"intent": ChoiceField(options=labels)})
        prepared = PreparedFeatures(data["fit"] + data["calibration"],
            np.concatenate([features["fit"], features["calibration"]]), encoder)
        started = time.perf_counter()
        taught = SystemOneCompiler(schema, projector=prepared, choice_solver="logistic", regularization=.1).compile(
            {"intent": [(r["prompt"], r["label"]) for r in data["fit"]]}, augment=False,
            calibration_exemplars={"intent": [(r["prompt"], r["label"]) for r in data["calibration"]]})
        model = CompiledSystemOneModel(taught.schema, taught.heads, dimension=encoder.dimension,
                                      projector=encoder, metadata=taught.metadata, use_cache=False)
        model.save(folder / "intent.s1m")
        teaching_ms = (time.perf_counter() - started) * 1000
        fit_labels = np.asarray([r["label"] for r in data["fit"]])
        supported = [label for label in labels if label != "oos"]
        centers = np.stack([features["fit"][fit_labels == label].mean(axis=0) for label in supported])
        residuals = np.concatenate([features["fit"][fit_labels == label] - center for label, center in zip(supported, centers)])
        covariance = residuals.T @ residuals / len(residuals)
        shrunk = .9 * covariance + .1 * np.eye(384) * np.trace(covariance) / 384
        whitening = np.linalg.inv(np.linalg.cholesky(shrunk)).T
        centers = centers @ whitening
        np.savez_compressed(folder / "scope.npz", whitening=whitening, centers=centers)
        manifest = dict(status="unqualified-development-candidate", dataset="clinc150", encoder=encoder.identity,
                        files={name: digest(folder / name) for name in ("intent.s1m", "scope.npz")},
                        categories={label: label.replace("_", " ") for label in labels},
                        instructions="Choose one supported intent. Unfamiliar requests require review.",
                        alpha=.05, probability_threshold=.8961238765292784, density_threshold=-658.9875995494192)
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Start isolation before loading. Neither load nor fresh decisions may call a teacher.
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Network connection during local inference")):
            started = time.perf_counter()
            candidate = LocalCandidate(folder)
            load_ms = (time.perf_counter() - started) * 1000
            original = System1Engine(model.schema, model=model, strict_mode=True, use_cache=False)
            for row in data["development"][:100]:
                expected = original.decide(row["prompt"], record_receipt=False)
                actual = candidate.engine.decide(row["prompt"], record_receipt=False)
                assert expected.values == actual.values
                assert expected.probabilities == actual.probabilities
                assert expected.conformal_sets == actual.conformal_sets
                assert expected.is_ambiguous == actual.is_ambiguous
            payload = dict(categories=manifest["categories"], instructions=manifest["instructions"])
            for row in data["development"][:100]:
                candidate.classify(dict(payload, text=row["prompt"]))
            outcomes, timings = [], []
            for row in data["development"]:
                started = time.perf_counter()
                result = candidate.classify(dict(payload, text=row["prompt"]))
                timings.append((time.perf_counter() - started) * 1000)
                outcomes.append(dict(group=row["group"], truth=row["label"], **result))
            counts = dict(supported=sum(r["truth"] != "oos" for r in outcomes),
                accepted=sum(not r["needsReview"] and r["truth"] != "oos" for r in outcomes),
                accepted_correct=sum(not r["needsReview"] and r["truth"] != "oos" and r["category"] == r["truth"] for r in outcomes),
                oos=sum(r["truth"] == "oos" for r in outcomes),
                oos_false_accept=sum(not r["needsReview"] and r["truth"] == "oos" for r in outcomes))
            report = dict(scope="complete adapter on development; not final qualification", candidate=manifest,
                manifest_sha256=candidate.identity, teaching_ms_excluding_features=teaching_ms, load_ms=load_ms,
                latency=dict(requests=len(timings), warmups=100, p50_ms=float(np.median(timings)), p95_ms=float(np.percentile(timings, 95))),
                save_reload_equivalent_cases=100, socket_connections=0, receipts=False, response_cache=False,
                counts=counts, outcomes=outcomes)
            (OUTPUT / "runtime-development.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: v for k, v in report.items() if k not in ("candidate", "outcomes")}, indent=2), flush=True)
        del original, candidate, model, taught, prepared, encoder
        gc.collect()


if __name__ == "__main__":
    main()
