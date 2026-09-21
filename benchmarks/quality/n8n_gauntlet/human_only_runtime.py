"""Standalone original-only banking artifact; reuse the existing saved adapter."""
from polynomial_scores import load_review, numerical_signals, score
from runtime_probe import LocalCandidate


class HumanOnlyCandidate(LocalCandidate):
    def __init__(self, folder):
        super().__init__(folder)
        self.review_parameters = load_review(dict(self.manifest, review_parameters=dict(
            mean=self.risk_mean, scale=self.risk_scale, weights=self.risk_weights,
            bias=self.risk_bias)), len(self.model.heads["intent"].options))

    def reliability(self, probabilities, embedding):
        signals, chosen = numerical_signals(probabilities, self.prototypes @ embedding, self.class_offsets)
        return score(signals, chosen, self.manifest["quadratic"], self.review_parameters)
