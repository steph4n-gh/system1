"""Numerical review interactions; no additional inference dependency for System1."""
import json
from pathlib import Path

import numpy as np

from develop import ROOT
from review_runtime import ReviewCandidate
from runtime_probe import digest


def expand(values, quadratic):
    values = np.asarray(values, dtype=np.float64)
    if not quadratic:
        return values
    return np.concatenate([values, np.stack([values[..., i] * values[..., j]
        for i in range(7) for j in range(i, 7)], axis=-1)], axis=-1)


def numerical_signals(probabilities, cosines, offsets):
    values = np.asarray(probabilities, dtype=np.float64)
    chosen = int(values.argmax())
    ordered = np.sort(values)
    signals = [np.log(max(ordered[-1], 1e-12) / max(1 - ordered[-1], 1e-12)),
        ordered[-1] - ordered[-2], -(values * np.log(np.maximum(values, 1e-12))).sum()]
    classes = [np.sort(cosines[start:end]) for start, end in zip(offsets[:-1], offsets[1:])]
    for count in (1, 5):
        similarity = np.asarray([part[-count:].mean() for part in classes])
        own = similarity[chosen].copy()
        similarity[chosen] = -np.inf
        signals.extend([own, own - similarity.max()])
    return np.asarray(signals), chosen


def load_review(manifest, labels):
    if type(manifest["quadratic"]) is not bool:
        raise ValueError("Invalid feature expansion flag")
    parameters = manifest["review_parameters"]
    if set(parameters) != {"mean", "scale", "weights", "bias"}:
        raise ValueError("Unexpected review parameters")
    mean, scale, weights, bias = (np.asarray(parameters[key], dtype=np.float64)
                                 for key in ("mean", "scale", "weights", "bias"))
    width = 35 if manifest["quadratic"] else 7
    if (mean.shape != (width,) or scale.shape != (width,) or weights.shape != (width + labels,)
            or bias.shape != () or np.any(scale <= 0)
            or not all(np.isfinite(v).all() for v in (mean, scale, weights, bias))):
        raise ValueError("Invalid numerical review parameters")
    return mean, scale, weights, float(bias)


def score(signals, chosen, quadratic, parameters):
    values = expand(signals, quadratic)
    mean, scale, weights, bias = parameters
    logit = ((values - mean) / scale) @ weights[:len(values)] + weights[len(values) + chosen] + bias
    return float(1 / (1 + np.exp(-np.clip(logit, -700, 700))))


class PolynomialCandidate(ReviewCandidate):
    def __init__(self, folder):
        folder = Path(folder)
        manifest = json.loads((folder / "manifest.json").read_text())
        self.parent_folder = ROOT / manifest["base_artifact"]
        super().__init__(self.parent_folder)
        if self.identity != manifest["parent_manifest_sha256"]:
            raise ValueError("Parent intent artifact changed")
        if any(manifest[key] != self.manifest[key] for key in ("categories", "instructions", "alpha", "encoder", "files")):
            raise ValueError("Review changed the parent decision contract")
        self.review_parameters = load_review(manifest, len(self.model.heads["intent"].options))
        self.folder, self.manifest, self.identity = folder, manifest, digest(folder / "manifest.json")

    def reliability(self, probabilities, embedding):
        signals, chosen = numerical_signals(probabilities, self.prototypes @ embedding, self.class_offsets)
        return score(signals, chosen, self.manifest["quadratic"], self.review_parameters)
