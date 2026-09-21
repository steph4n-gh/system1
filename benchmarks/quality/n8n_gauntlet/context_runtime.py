"""Reuse the saved System1 encoder's input vector for review."""
from context_scores import load_context, score
from polynomial_runtime import PolynomialCandidate
from polynomial_scores import numerical_signals


class ContextCandidate(PolynomialCandidate):
    def __init__(self, folder):
        super().__init__(folder)
        if self.manifest["quadratic"] is not False:
            raise ValueError("Context review requires linear numerical signals")
        self.context_weights = load_context(self.manifest, self.model.heads["intent"].weights.shape[1])

    def reliability(self, probabilities, embedding):
        signals, chosen = numerical_signals(probabilities, self.prototypes @ embedding, self.class_offsets)
        return score(signals, chosen, float(embedding @ self.context_weights), self.review_parameters)
