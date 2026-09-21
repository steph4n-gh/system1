"""Numerical review interactions; no additional inference dependency for System1."""
import json
from pathlib import Path

from develop import ROOT
from review_runtime import ReviewCandidate
from runtime_probe import digest


from polynomial_scores import expand, load_review, numerical_signals, score


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
