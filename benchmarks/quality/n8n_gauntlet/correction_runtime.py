"""Standalone correction experiment; reuse the existing numerical adapters."""
from context_scores import load_context, score
from human_only_runtime import HumanOnlyCandidate
from polynomial_scores import numerical_signals


class CorrectionCandidate(HumanOnlyCandidate):
    def __init__(self, folder):
        super().__init__(folder)
        self.context_weights = (load_context(self.manifest, self.model.heads["intent"].weights.shape[1])
            if "context_dimension" in self.manifest else None)

    def reliability(self, probabilities, embedding):
        if self.context_weights is None:
            return super().reliability(probabilities, embedding)
        signals, chosen = numerical_signals(probabilities, self.prototypes @ embedding, self.class_offsets)
        return score(signals, chosen, float(embedding @ self.context_weights), self.review_parameters)
