"""Reflex Calibration & Split Conformal Prediction Engine.

Implements:
1. Temperature Scaling & Platt Scaling with numerical stability.
2. Proper scoring rules: Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
   Negative Log-Likelihood (NLL), and Sanders-Murphy Brier Score Decomposition
   (Reliability, Resolution, Uncertainty).
3. Split Conformal Prediction with finite-sample, distribution-free mathematical
   coverage guarantees (1 - alpha).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class BrierDecomposition:
    """Sanders-Murphy decomposition of the Brier Score.

    Identity: total_brier = reliability - resolution + uncertainty.
    - reliability >= 0: calibration error (lower is better, 0 is perfectly calibrated).
    - resolution >= 0: discriminative sorting ability (higher is better).
    - uncertainty >= 0: inherent variance of base rates (fixed for a given dataset).
    - empirical_brier: direct unbinned sample mean squared error.
    """

    total_brier: float
    reliability: float
    resolution: float
    uncertainty: float
    empirical_brier: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationMetrics:
    """Comprehensive calibration diagnostic metrics."""

    expected_calibration_error: float
    maximum_calibration_error: float
    negative_log_likelihood: float
    brier: BrierDecomposition
    optimal_temperature: float
    bin_accuracies: Tuple[float, ...]
    bin_confidences: Tuple[float, ...]
    bin_counts: Tuple[int, ...]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["bin_accuracies"] = list(self.bin_accuracies)
        d["bin_confidences"] = list(self.bin_confidences)
        d["bin_counts"] = list(self.bin_counts)
        return d


@dataclass(frozen=True)
class ConformalPredictionSet:
    """Result of evaluating a test input through a split conformal predictor."""

    field_name: str
    target_coverage: float  # 1 - alpha (e.g. 0.95)
    alpha: float
    prediction_set: Tuple[str, ...]
    conformal_threshold: float
    is_ambiguous: bool
    is_empty: bool
    p_values: Dict[str, float]
    margin: float = 0.0
    margin_threshold: float = 0.0
    margin_gate_active: bool = False
    raw_is_ambiguous: bool = False
    odds_ratio: float = 1.0
    relative_odds_ratio_threshold: float = 0.0
    confidence_floor: float = 0.0
    needs_escalation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "target_coverage": self.target_coverage,
            "alpha": self.alpha,
            "prediction_set": list(self.prediction_set),
            "conformal_threshold": self.conformal_threshold,
            "is_ambiguous": self.is_ambiguous,
            "is_empty": self.is_empty,
            "p_values": self.p_values,
            "margin": self.margin,
            "margin_threshold": self.margin_threshold,
            "margin_gate_active": self.margin_gate_active,
            "raw_is_ambiguous": self.raw_is_ambiguous,
            "needs_escalation": self.needs_escalation,
            "odds_ratio": self.odds_ratio,
            "relative_odds_ratio_threshold": self.relative_odds_ratio_threshold,
            "confidence_floor": self.confidence_floor,
        }


def _stable_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature."""
    temp = max(1e-4, float(temperature))
    scaled = np.asarray(logits, dtype=np.float64) / temp
    max_val = np.max(scaled, axis=-1, keepdims=True)
    exp = np.exp(scaled - max_val)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Computes Negative Log-Likelihood under categorical distribution."""
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    if n == 0:
        return 0.0
    correct_probs = probs[np.arange(n), labels]
    return float(-np.mean(np.log(correct_probs)))


def compute_ece_and_bins(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, Tuple[float, ...], Tuple[float, ...], Tuple[int, ...]]:
    """Calculates ECE, MCE, and binned calibration stats across [0, 1]."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    if n == 0:
        return 0.0, 0.0, (), (), ()

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_accs: List[float] = []
    bin_confs: List[float] = []
    bin_counts: List[int] = []

    for b in range(n_bins):
        low, high = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (confidences > low) & (confidences <= high) if b > 0 else (confidences >= low) & (confidences <= high)
        count = int(np.sum(mask))
        bin_counts.append(count)
        if count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            gap = abs(bin_acc - bin_conf)
            ece += (count / n) * gap
            mce = max(mce, gap)
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    return float(ece), float(mce), tuple(bin_accs), tuple(bin_confs), tuple(bin_counts)


def compute_brier_decomposition(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> BrierDecomposition:
    """Computes Sanders-Murphy decomposition of multi-class Brier Score.

    Formula:
      BS = Reliability - Resolution + Uncertainty
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n, k = probs.shape
    if n == 0:
        return BrierDecomposition(0.0, 0.0, 0.0, 0.0)

    # One-hot true outcomes Y
    y_one_hot = np.zeros((n, k), dtype=np.float64)
    y_one_hot[np.arange(n), labels] = 1.0

    # Empirical sample mean squared error
    empirical_brier = float(np.mean(np.sum((probs - y_one_hot) ** 2, axis=1)))

    # Overall base rate \bar{y}
    base_rate = np.mean(y_one_hot, axis=0)
    uncertainty = float(np.sum(base_rate * (1.0 - base_rate)))

    # Binning predictions by max confidence
    confidences = np.max(probs, axis=1)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    reliability = 0.0
    resolution = 0.0

    for b in range(n_bins):
        low, high = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (confidences > low) & (confidences <= high) if b > 0 else (confidences >= low) & (confidences <= high)
        n_m = int(np.sum(mask))
        if n_m > 0:
            weight = n_m / n
            p_m = np.mean(probs[mask], axis=0)
            y_m = np.mean(y_one_hot[mask], axis=0)

            reliability += weight * float(np.sum((p_m - y_m) ** 2))
            resolution += weight * float(np.sum((y_m - base_rate) ** 2))

    rel = float(max(0.0, reliability))
    res = float(max(0.0, resolution))
    unc = float(uncertainty)
    binned_brier = float(max(0.0, rel - res + unc))

    return BrierDecomposition(
        total_brier=binned_brier,
        reliability=rel,
        resolution=res,
        uncertainty=unc,
        empirical_brier=empirical_brier,
    )


class DecisionCalibrator:
    """Calibrator implementing temperature scaling and Platt scaling."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature: float = max(1e-3, float(temperature))
        self.metrics: Optional[CalibrationMetrics] = None
        self.is_binary: bool = False

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        *,
        n_bins: int = 10,
        temp_bounds: Tuple[float, float] = (0.15, 10.0),
    ) -> CalibrationMetrics:
        """Finds optimal temperature T > 0 minimizing NLL on calibration logits."""
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        if len(logits) == 0:
            raise ValueError("Cannot fit calibrator on empty data")

        if logits.ndim == 1:
            self.is_binary = True
            logits = np.column_stack([-logits / 2.0, logits / 2.0])

        a, b = temp_bounds
        inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
        inv_phi2 = (3.0 - math.sqrt(5.0)) / 2.0

        c = a + inv_phi2 * (b - a)
        d = a + inv_phi * (b - a)

        def eval_temp(t: float) -> float:
            scaled = logits / t
            max_val = np.max(scaled, axis=-1, keepdims=True)
            exp = np.exp(scaled - max_val)
            probs = exp / np.sum(exp, axis=-1, keepdims=True)
            return compute_nll(probs, labels)

        fc = eval_temp(c)
        fd = eval_temp(d)

        for _ in range(60):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = a + inv_phi2 * (b - a)
                fc = eval_temp(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + inv_phi * (b - a)
                fd = eval_temp(d)

        best_temp = float((a + b) / 2.0)
        self.temperature = best_temp

        calibrated_probs = self.calibrate_logits(logits)
        nll = compute_nll(calibrated_probs, labels)
        ece, mce, bin_accs, bin_confs, bin_counts = compute_ece_and_bins(
            calibrated_probs, labels, n_bins=n_bins
        )
        brier = compute_brier_decomposition(calibrated_probs, labels, n_bins=n_bins)

        self.metrics = CalibrationMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            negative_log_likelihood=nll,
            brier=brier,
            optimal_temperature=best_temp,
            bin_accuracies=bin_accs,
            bin_confidences=bin_confs,
            bin_counts=bin_counts,
        )
        return self.metrics

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        """Applies learned temperature to raw logits to yield calibrated probabilities."""
        logits = np.asarray(logits, dtype=np.float64)
        if self.is_binary:
            if logits.ndim == 0:
                logits = np.array([[-float(logits) / 2.0, float(logits) / 2.0]], dtype=np.float64)
                scaled = logits / self.temperature
                return _stable_softmax(scaled)[0]
            elif logits.ndim == 1:
                logits = np.column_stack([-logits / 2.0, logits / 2.0])
                scaled = logits / self.temperature
                return _stable_softmax(scaled)

        if logits.ndim == 1 and len(logits) == 1:
            z = logits[0]
            logits = np.array([-z / 2.0, z / 2.0], dtype=np.float64)
        scaled = logits / self.temperature
        if scaled.ndim == 1:
            scaled = scaled.reshape(1, -1)
            single = True
        else:
            single = False
        max_val = np.max(scaled, axis=-1, keepdims=True)
        exp = np.exp(scaled - max_val)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)
        return probs[0] if single else probs


class ConformalPredictor:
    """Split Conformal Prediction Engine with distribution-free coverage guarantees.

    Guarantees:
      P(Y in C(X)) >= 1 - alpha

    Enhanced with Margin-Based Conformal Gating (Lever 3):
    Calibrates on margin of dominance M(x) = s_(1)(x) - s_(2)(x) in addition to set size |C(x)|.
    When the top-ranked candidate dominates the runner-up by greater than the calibrated margin
    threshold, false-positive ambiguity escalations are suppressed.
    """

    def __init__(
        self,
        field_name: str,
        options: Sequence[str],
        margin_threshold: Optional[float] = None,
        *,
        relative_odds_ratio: Optional[float] = 1.5,
        confidence_floor_tau0: Optional[float] = 0.15,
    ) -> None:
        self.field_name = field_name
        self.options = tuple(options)
        self.option_to_idx = {opt: idx for idx, opt in enumerate(self.options)}
        self.calibration_scores: np.ndarray = np.array([], dtype=np.float64)
        self.is_calibrated: bool = False
        self.quantile: Optional[float] = None
        self.margin_threshold: Optional[float] = (
            float(margin_threshold) if margin_threshold is not None else None
        )
        self.calibrated_margin_threshold: float = 0.0
        self.relative_odds_ratio: float = (
            float(relative_odds_ratio) if relative_odds_ratio is not None else 1.5
        )
        self.confidence_floor_tau0: float = (
            float(confidence_floor_tau0) if confidence_floor_tau0 is not None else 0.15
        )

    def calibrate(
        self,
        calibrated_probs: np.ndarray,
        ground_truth_labels: Sequence[str],
        margin_threshold: Optional[float] = None,
        *,
        relative_odds_ratio: Optional[float] = None,
        confidence_floor_tau0: Optional[float] = None,
    ) -> None:
        """Fits Adaptive Prediction Set (APS) non-conformity scores and margin threshold on calibration samples."""
        probs = np.asarray(calibrated_probs, dtype=np.float64)
        if len(probs) != len(ground_truth_labels):
            raise ValueError("Mismatched probabilities and labels length")
        if len(probs) == 0:
            raise ValueError("Cannot calibrate conformal predictor with 0 samples")

        if relative_odds_ratio is not None:
            self.relative_odds_ratio = float(relative_odds_ratio)
        if confidence_floor_tau0 is not None:
            self.confidence_floor_tau0 = float(confidence_floor_tau0)

        scores: List[float] = []
        error_margins: List[float] = []

        for i, label in enumerate(ground_truth_labels):
            if label not in self.option_to_idx:
                raise ValueError(f"Unknown label {label!r} not in options {self.options}")
            true_idx = self.option_to_idx[label]
            row = probs[i]

            sorted_indices = np.argsort(-row)
            top_idx = int(sorted_indices[0])
            runner_up_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else top_idx
            margin_i = float(row[top_idx] - row[runner_up_idx])

            if top_idx != true_idx:
                # Top candidate was an error: record its margin
                error_margins.append(margin_i)

            cum_sum = 0.0
            score_val = 1.0
            for idx in sorted_indices:
                cum_sum += float(row[idx])
                if idx == true_idx:
                    score_val = cum_sum
                    break
            scores.append(score_val)

        self.calibration_scores = np.sort(np.asarray(scores, dtype=np.float64))
        self.is_calibrated = True

        if margin_threshold is not None:
            self.margin_threshold = float(margin_threshold)
            self.calibrated_margin_threshold = self.margin_threshold
        elif self.margin_threshold is not None:
            self.calibrated_margin_threshold = self.margin_threshold
        else:
            # Calibrate threshold above observed error margins
            if error_margins:
                # Safe upper margin guaranteeing top prediction dominance
                self.calibrated_margin_threshold = float(
                    np.clip(np.quantile(error_margins, 0.95) + 0.05, 0.15, 0.85)
                )
            else:
                self.calibrated_margin_threshold = 0.35

    def predict_set(
        self,
        calibrated_probs: np.ndarray,
        *,
        alpha: float = 0.05,
        margin_threshold: Optional[float] = None,
        relative_odds_ratio: Optional[float] = None,
        confidence_floor_tau0: Optional[float] = None,
        strict: bool = False,
    ) -> ConformalPredictionSet:
        """Constructs conformal prediction set C(x) satisfying P(Y in C(x)) >= 1 - alpha via APS.

        Applies Cardinality-Scaled Margin Gate & Relative Odds Ratio Dominance:
        When raw_is_ambiguous (set size > 1), false-positive ambiguity escalation is suppressed
        ONLY IF strict=False AND:
          1. Absolute margin difference: margin >= eff_margin_thresh
          2. Absolute confidence floor scaled by cardinality: top_prob >= (1/K) + eff_tau0
          3. Relative odds ratio dominance: (top_prob / max(1e-6, runner_up_prob)) >= eff_gamma
        This prevents pseudo-random hash dispersion at high cardinality (e.g. K=77) from
        falsely suppressing conformal ambiguity escalation while maintaining precision.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")

        probs = np.asarray(calibrated_probs, dtype=np.float64).flatten()
        if len(probs) != len(self.options):
            raise ValueError(
                f"Probabilities length ({len(probs)}) does not match options ({len(self.options)})"
            )

        n = len(self.calibration_scores)
        if not self.is_calibrated or n == 0:
            q_hat = 1.0 - alpha
        else:
            k = int(math.ceil((n + 1) * (1.0 - alpha)))
            if k > n:
                q_hat = 1.0
            else:
                q_hat = float(self.calibration_scores[k - 1])

        sorted_indices = np.argsort(-probs)
        prediction_set: List[str] = []
        cum_mass = 0.0

        top_prob = float(probs[sorted_indices[0]]) if len(sorted_indices) > 0 else 0.0
        runner_up_prob = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
        margin = float(top_prob - runner_up_prob) if len(sorted_indices) > 1 else 1.0

        total_mass = float(np.sum(probs))
        # True OOD: total probability mass cannot reach q_hat or is degenerate (e.g. unnormalized / near-zero vectors)
        if total_mass >= q_hat - 1e-7 and total_mass >= 0.5:
            for idx in sorted_indices:
                opt = self.options[idx]
                p = float(probs[idx])
                prediction_set.append(opt)
                cum_mass += p
                if cum_mass >= q_hat - 1e-7:
                    break

        p_values: Dict[str, float] = {}
        cum_masses: Dict[int, float] = {}
        running_mass = 0.0
        for idx in sorted_indices:
            running_mass += float(probs[idx])
            cum_masses[idx] = running_mass

        for idx, opt in enumerate(self.options):
            c_mass = cum_masses[idx]
            if n > 0:
                rank = int(np.sum(self.calibration_scores >= c_mass))
                p_val = float((1.0 + rank) / (n + 1))
            else:
                p_val = float(probs[idx])
            p_values[opt] = p_val

        # Cardinality-Scaled Margin Gate & Relative Odds Ratio
        eff_margin_thresh = (
            float(margin_threshold)
            if margin_threshold is not None
            else (self.margin_threshold if self.margin_threshold is not None else self.calibrated_margin_threshold)
        )
        eff_gamma = (
            float(relative_odds_ratio)
            if relative_odds_ratio is not None
            else self.relative_odds_ratio
        )
        eff_tau0 = (
            float(confidence_floor_tau0)
            if confidence_floor_tau0 is not None
            else self.confidence_floor_tau0
        )

        raw_is_ambiguous = len(prediction_set) > 1
        is_empty = len(prediction_set) == 0

        # Cardinality-scaled confidence floor: p_(1) >= 1/K + tau_0
        k_card = max(1, len(self.options))
        conf_floor = float((1.0 / float(k_card)) + eff_tau0)

        # Relative odds ratio: p_(1) / max(1e-6, p_(2))
        odds_ratio = float(top_prob / max(1e-6, runner_up_prob))

        # Margin-Based Conformal Dominance Gating (suppressed in strict mode)
        margin_gate_active = bool(
            not strict
            and raw_is_ambiguous
            and (eff_margin_thresh > 0.0)
            and (margin >= eff_margin_thresh)
            and (top_prob >= conf_floor)
            and (odds_ratio >= eff_gamma)
        )
        is_ambiguous = raw_is_ambiguous and not margin_gate_active
        needs_escalation = is_empty or is_ambiguous

        return ConformalPredictionSet(
            field_name=self.field_name,
            target_coverage=1.0 - alpha,
            alpha=alpha,
            prediction_set=tuple(prediction_set),
            conformal_threshold=q_hat,
            is_ambiguous=is_ambiguous,
            is_empty=is_empty,
            p_values=p_values,
            margin=margin,
            margin_threshold=eff_margin_thresh,
            margin_gate_active=margin_gate_active,
            raw_is_ambiguous=raw_is_ambiguous,
            odds_ratio=odds_ratio,
            relative_odds_ratio_threshold=eff_gamma,
            confidence_floor=conf_floor,
            needs_escalation=needs_escalation,
        )


@dataclass(frozen=True)
class RegressionConformalInterval:
    """Conformal prediction interval for continuous score fields with 1 - alpha coverage."""

    field_name: str
    target_coverage: float
    alpha: float
    point_prediction: float
    lower_bound: float
    upper_bound: float
    margin: float
    is_calibrated: bool

    def to_interval_string(self) -> str:
        return f"[{self.lower_bound:.4f}, {self.upper_bound:.4f}]"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RegressionConformalPredictor:
    """Split conformal predictor for continuous bounded regression scores."""

    def __init__(self, field_name: str, min_value: float = 0.0, max_value: float = 1.0) -> None:
        self.field_name = field_name
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.residuals: np.ndarray = np.array([], dtype=np.float64)
        self.is_calibrated: bool = False

    def calibrate(self, predictions: Sequence[float], ground_truth: Sequence[float]) -> None:
        preds = np.asarray(predictions, dtype=np.float64)
        targets = np.asarray(ground_truth, dtype=np.float64)
        if len(preds) != len(targets):
            raise ValueError("Predictions and targets length mismatch")
        if len(preds) == 0:
            raise ValueError("Cannot calibrate regression conformal predictor with 0 samples")
        res = np.abs(preds - targets)
        self.residuals = np.sort(res)
        self.is_calibrated = True

    def predict_interval(self, point_prediction: float, *, alpha: float = 0.05) -> RegressionConformalInterval:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")
        y_hat = float(point_prediction)
        val_range = self.max_value - self.min_value
        n = len(self.residuals)

        if not self.is_calibrated or n == 0:
            margin = val_range * ((1.0 - alpha) / 2.0)
            low = max(self.min_value, y_hat - margin)
            high = min(self.max_value, y_hat + margin)
        else:
            k = int(math.ceil((n + 1) * (1.0 - alpha)))
            if k > n:
                margin = val_range
                low = self.min_value
                high = self.max_value
            else:
                margin = float(self.residuals[k - 1])
                low = max(self.min_value, y_hat - margin)
                high = min(self.max_value, y_hat + margin)
        return RegressionConformalInterval(
            field_name=self.field_name,
            target_coverage=1.0 - alpha,
            alpha=alpha,
            point_prediction=y_hat,
            lower_bound=low,
            upper_bound=high,
            margin=margin,
            is_calibrated=self.is_calibrated,
        )


__all__ = [
    "BrierDecomposition",
    "CalibrationMetrics",
    "ConformalPredictionSet",
    "ConformalPredictor",
    "DecisionCalibrator",
    "RegressionConformalInterval",
    "RegressionConformalPredictor",
    "compute_brier_decomposition",
    "compute_ece_and_bins",
    "compute_nll",
]

