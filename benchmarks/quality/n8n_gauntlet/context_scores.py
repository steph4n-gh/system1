"""Add existing input features to a numerical review logit; NumPy only."""
import numpy as np


def load_context(manifest, dimension):
    if type(manifest.get("context_dimension")) is not int or manifest["context_dimension"] != dimension or dimension < 1:
        raise ValueError("Context dimension changed")
    weights = np.asarray(manifest.get("context_weights", []), dtype=np.float64)
    if weights.shape != (dimension,) or not np.isfinite(weights).all():
        raise ValueError("Invalid context weights")
    return weights


def score(signals, chosen, context_logit, parameters):
    mean, scale, weights, bias = parameters
    logit = ((np.asarray(signals) - mean) / scale) @ weights[:7] + weights[7 + chosen] + bias + context_logit
    return float(1 / (1 + np.exp(-np.clip(logit, -700, 700))))
