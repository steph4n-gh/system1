"""Small taught feature transform; deployment needs only NumPy and one encoder."""
import hashlib
import json

import numpy as np

from fused_features import join_features


class ShallowFeatures:
    def __init__(self, encoder, weights, bias):
        self.encoder = encoder
        self.weights = np.array(weights, dtype=np.float32, copy=True)
        self.bias = np.array(bias, dtype=np.float32, copy=True)
        if (self.weights.ndim != 2 or self.weights.shape[0] != encoder.dimension
                or self.bias.shape != (self.weights.shape[1],) or not 1 <= self.weights.shape[1] <= 512
                or not np.isfinite(self.weights).all() or not np.isfinite(self.bias).all()):
            raise ValueError("Invalid taught transform shape or values")
        self.weights.setflags(write=False)
        self.bias.setflags(write=False)
        self.dimension = encoder.dimension + len(self.bias)

    def transform(self, semantic):
        semantic = np.asarray(semantic, dtype=np.float32)
        if semantic.shape != (self.encoder.dimension,) or not np.isfinite(semantic).all():
            raise ValueError("Invalid semantic input")
        hidden = np.maximum(semantic @ self.weights + self.bias, 0)
        hidden /= max(float(np.linalg.norm(hidden)), 1e-12)
        return join_features(semantic, hidden)

    def project(self, text, recency_weighted=False):
        if recency_weighted:
            raise ValueError("The frozen encoder does not support recency weighting")
        return self.transform(self.encoder.project(text))

    def projector_digest(self):
        description = dict(version=1, encoder=self.encoder.projector_digest(),
                           shape=list(self.weights.shape), mixing="equal-unit-semantic-relu")
        h = hashlib.sha256(json.dumps(description, sort_keys=True).encode())
        h.update(self.weights.astype("<f4").tobytes())
        h.update(self.bias.astype("<f4").tobytes())
        return h.hexdigest()

    @classmethod
    def load(cls, encoder, path):
        with np.load(path, allow_pickle=False) as arrays:
            return cls(encoder, arrays["weights"], arrays["bias"])
