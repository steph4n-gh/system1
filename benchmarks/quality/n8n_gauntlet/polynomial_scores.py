"""Portable numerical review scores; NumPy is the only dependency."""
import numpy as np


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


