"""Fixed semantic features plus System1's existing portable word features."""
import hashlib
import json

import numpy as np


def join_features(semantic, lexical):
    """Equal-weight unit components; normalize safely if lexical input is zero."""
    vector = np.concatenate([semantic, lexical]).astype(np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


class FusedFeatures:
    def __init__(self, encoder, lexical):
        self.encoder, self.lexical = encoder, lexical
        self.dimension = encoder.dimension + lexical.dimension

    def project(self, text, recency_weighted=False):
        if recency_weighted:
            raise ValueError("The frozen encoder does not support recency weighting")
        return join_features(self.encoder.project(text), self.lexical.project(text))

    def projector_digest(self):
        specification = dict(version=1, encoder_digest=self.encoder.projector_digest(),
                             lexical=self.lexical.to_config(), mixing="equal-unit-components")
        return hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
