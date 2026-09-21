"""Complete fused-feature candidate using one encoder and the existing review."""
import json
from pathlib import Path

import numpy as np

from encoder_probe import Encoder
from fused_features import FusedFeatures
from review_runtime import ReviewCandidate
from runtime_probe import digest
from system1 import CompiledSystemOneModel, System1Engine
from system1.core.text import TfidfProjector


def semantic_component(embedding):
    vector = np.asarray(embedding, dtype=np.float32)[:384]
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


class FusedReviewCandidate(ReviewCandidate):
    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / "manifest.json").read_text())
        self.identity = digest(self.folder / "manifest.json")
        for name in ("intent.s1m", "lexical.json", "scope.npz"):
            if digest(self.folder / name) != self.manifest["files"][name]:
                raise ValueError("Fused review artifact changed")
        self.encoder = Encoder(self.manifest["encoder"]["name"], threads=self.manifest["encoder"]["threads"])
        if self.encoder.identity != self.manifest["encoder"]:
            raise ValueError("Fused review encoder changed")
        lexical = TfidfProjector.from_config(json.loads((self.folder / "lexical.json").read_text()))
        projector = FusedFeatures(self.encoder, lexical)
        if projector.projector_digest() != self.manifest["projector_digest"]:
            raise ValueError("Fused review feature identity changed")
        self.model = CompiledSystemOneModel.load(self.folder / "intent.s1m", projector=projector)
        self.model.use_cache = False
        self.engine = System1Engine(self.model.schema, model=self.model, strict_mode=True, use_cache=False)
        with np.load(self.folder / "scope.npz", allow_pickle=False) as arrays:
            self.prototypes, self.class_offsets = arrays["prototypes"], arrays["class_offsets"]
            self.risk_mean, self.risk_scale = arrays["mean"], arrays["scale"]
            self.risk_weights, self.risk_bias = arrays["weights"], float(arrays["bias"])

    def reliability(self, probabilities, embedding):
        return super().reliability(probabilities, semantic_component(embedding))
