"""Tests for Reflex Split Conformal Prediction Engine."""

import numpy as np
import pytest

from reflex.calibration import ConformalPredictor


def test_conformal_predictor_uncalibrated_fallback():
    options = ["cat", "dog", "bird"]
    predictor = ConformalPredictor("animal", options)

    probs = np.array([0.9, 0.08, 0.02])
    cset = predictor.predict_set(probs, alpha=0.10)

    assert not predictor.is_calibrated
    assert "cat" in cset.prediction_set
    assert cset.target_coverage == 0.90


def test_conformal_finite_sample_coverage_guarantee():
    rng = np.random.default_rng(2026)
    options = ["auth_error", "rate_limit", "server_error", "not_found"]
    k = len(options)
    n_calib = 400
    n_test = 200
    alpha = 0.10

    def generate_data(num_samples: int):
        labels_idx = rng.integers(0, k, size=num_samples)
        logits = rng.standard_normal((num_samples, k))
        for i in range(num_samples):
            logits[i, labels_idx[i]] += 2.0
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        return probs, [options[idx] for idx in labels_idx]

    calib_probs, calib_labels = generate_data(n_calib)
    test_probs, test_labels = generate_data(n_test)

    predictor = ConformalPredictor("error_kind", options)
    predictor.calibrate(calib_probs, calib_labels)

    assert predictor.is_calibrated
    assert len(predictor.calibration_scores) == n_calib

    covered_count = 0
    for i in range(n_test):
        cset = predictor.predict_set(test_probs[i], alpha=alpha)
        true_label = test_labels[i]
        if true_label in cset.prediction_set:
            covered_count += 1

    empirical_coverage = covered_count / n_test
    assert empirical_coverage >= (1.0 - alpha) - 0.05, (
        f"Empirical coverage {empirical_coverage:.3f} failed mathematical guarantee for alpha={alpha}"
    )


def test_conformal_ambiguity_detection():
    options = ["read_disk", "read_db", "write_disk"]
    predictor = ConformalPredictor("task", options)

    base_probs = np.array([
        [0.85, 0.10, 0.05],
        [0.10, 0.85, 0.05],
        [0.05, 0.05, 0.90],
        [0.70, 0.25, 0.05],
    ])
    calib_probs = np.tile(base_probs, (5, 1))
    calib_labels = ["read_disk", "read_db", "write_disk", "read_disk"] * 5
    predictor.calibrate(calib_probs, calib_labels)

    # 1. Clear, confident query
    confident_p = np.array([0.95, 0.03, 0.02])
    cset_clear = predictor.predict_set(confident_p, alpha=0.05)
    assert not cset_clear.is_ambiguous
    assert cset_clear.prediction_set == ("read_disk",)

    # 2. Conflicted / borderline query
    ambiguous_p = np.array([0.50, 0.48, 0.02])
    cset_ambig = predictor.predict_set(ambiguous_p, alpha=0.05)
    assert cset_ambig.is_ambiguous
    assert len(cset_ambig.prediction_set) >= 2
    assert "read_disk" in cset_ambig.prediction_set
    assert "read_db" in cset_ambig.prediction_set

    # 3. Out-of-distribution input yields empty prediction set
    ood_p = np.array([0.05, 0.05, 0.05])
    cset_ood = predictor.predict_set(ood_p, alpha=0.05)
    assert cset_ood.is_empty is True
    assert len(cset_ood.prediction_set) == 0
    assert not cset_ood.is_ambiguous
