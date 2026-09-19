"""Tests for System 1 Calibration and Proper Scoring Rules."""

import numpy as np
import pytest

from system1.calibration import (
    BrierDecomposition,
    CalibrationMetrics,
    DecisionCalibrator,
    RegressionConformalInterval,
    RegressionConformalPredictor,
    compute_brier_decomposition,
    compute_ece_and_bins,
    compute_nll,
)


def test_nll_computation():
    probs = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ])
    labels = np.array([0, 1, 0])
    nll = compute_nll(probs, labels)
    assert nll < 1e-4

    uncertain_probs = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ])
    uncertain_labels = np.array([0, 1])
    uncertain_nll = compute_nll(uncertain_probs, uncertain_labels)
    assert abs(uncertain_nll - float(np.log(2.0))) < 1e-4


def test_brier_score_decomposition_identity():
    # Sanders-Murphy Identity: total_brier = reliability - resolution + uncertainty (exact identity)
    rng = np.random.default_rng(42)
    n, k = 100, 3
    raw = rng.standard_normal((n, k))
    exp = np.exp(raw - np.max(raw, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    labels = rng.integers(0, k, size=n)

    brier = compute_brier_decomposition(probs, labels, n_bins=10)

    reconstructed = brier.reliability - brier.resolution + brier.uncertainty
    assert brier.total_brier >= 0.0
    assert brier.reliability >= 0.0
    assert brier.resolution >= 0.0
    assert brier.uncertainty >= 0.0
    assert brier.empirical_brier >= 0.0
    assert abs(brier.total_brier - reconstructed) < 1e-12


def test_ece_and_mce_computation():
    probs = np.array([
        [0.9, 0.1],
        [0.85, 0.15],
        [0.8, 0.2],
        [0.95, 0.05],
    ])
    labels = np.array([1, 1, 1, 1])

    ece, mce, accs, confs, counts = compute_ece_and_bins(probs, labels, n_bins=5)

    assert ece > 0.7
    assert mce > 0.7
    assert sum(counts) == 4


def test_temperature_scaling_optimization():
    rng = np.random.default_rng(101)
    n, k = 200, 4

    true_labels = rng.integers(0, k, size=n)
    logits = rng.standard_normal((n, k)) * 4.0
    for i in range(n):
        logits[i, true_labels[i]] += 2.5

    calibrator = DecisionCalibrator(temperature=1.0)
    uncalibrated_probs = calibrator.calibrate_logits(logits)
    initial_nll = compute_nll(uncalibrated_probs, true_labels)

    metrics = calibrator.fit(logits, true_labels, n_bins=10)

    assert metrics.optimal_temperature > 0.1
    assert metrics.negative_log_likelihood <= initial_nll
    calibrated_probs = calibrator.calibrate_logits(logits)
    np.testing.assert_allclose(np.sum(calibrated_probs, axis=1), np.ones(n), atol=1e-5)


def test_temperature_scaling_1d_logits():
    logits = np.array([2.5, -1.8, 3.2, -2.1, 1.4, -0.9, 2.0, -1.5])
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])

    calibrator = DecisionCalibrator(temperature=1.0)
    metrics = calibrator.fit(logits, labels, n_bins=4)

    assert metrics.optimal_temperature > 0.0
    assert metrics.expected_calibration_error >= 0.0
    assert metrics.brier.total_brier >= 0.0

    probs = calibrator.calibrate_logits(logits)
    assert probs.shape == (8, 2)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(8), atol=1e-5)


def test_regression_conformal_predictor_coverage():
    rng = np.random.default_rng(2026)
    n_calib = 300
    n_test = 150
    alpha = 0.10

    x_calib = rng.uniform(0, 2 * np.pi, size=n_calib)
    targets_calib = np.clip(0.5 + 0.3 * np.sin(x_calib) + rng.normal(0, 0.05, size=n_calib), 0.0, 1.0)
    preds_calib = np.clip(0.5 + 0.3 * np.sin(x_calib), 0.0, 1.0)

    predictor = RegressionConformalPredictor("score", min_value=0.0, max_value=1.0)
    predictor.calibrate(preds_calib, targets_calib)

    assert predictor.is_calibrated
    assert len(predictor.residuals) == n_calib

    x_test = rng.uniform(0, 2 * np.pi, size=n_test)
    targets_test = np.clip(0.5 + 0.3 * np.sin(x_test) + rng.normal(0, 0.05, size=n_test), 0.0, 1.0)
    preds_test = np.clip(0.5 + 0.3 * np.sin(x_test), 0.0, 1.0)

    covered = 0
    for i in range(n_test):
        interval = predictor.predict_interval(preds_test[i], alpha=alpha)
        if interval.lower_bound <= targets_test[i] <= interval.upper_bound:
            covered += 1

    empirical_coverage = covered / n_test
    assert empirical_coverage >= (1.0 - alpha) - 0.05


def test_regression_conformal_margin_scales_with_coverage():
    predictor = RegressionConformalPredictor("risk", min_value=0.0, max_value=1.0)

    int_99 = predictor.predict_interval(0.5, alpha=0.01)
    int_80 = predictor.predict_interval(0.5, alpha=0.20)

    assert int_99.margin > int_80.margin
    assert (int_99.upper_bound - int_99.lower_bound) > (int_80.upper_bound - int_80.lower_bound)
    assert int_99.target_coverage == 0.99
    assert int_80.target_coverage == 0.80
