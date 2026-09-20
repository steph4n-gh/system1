"""TypeSafe-style questions and responses with local execution and observed teaching.

Supports the SDK's basic sync/async decision workflow. Transport extensions and
model-management APIs are outside this adapter's contract; see docs/typesafe.md.
"""

from __future__ import annotations

import asyncio
import collections
import functools
import hashlib
import inspect
import importlib
import importlib.abc
import importlib.machinery
import json
import math
import os
import sys
import threading
import time
import types
import urllib.error
import urllib.parse
import urllib.request
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from system1.engine import DecisionResult, SystemOneEngine
from system1.ledger import ActionLedger
from system1.receipt import DecisionWitnessReceipt
from system1.core import (
    BooleanField,
    ChoiceField,
    DecisionField,
    DecisionSchema,
    MultiChoiceField,
    ScoreField,
)

logger = logging.getLogger("system1.compat.typesafe")


class ZeroEgressViolationError(RuntimeError):
    """Raised when network egress is attempted in zero-egress mode."""


_EGRESS_AUDIT_LOG: List[Dict[str, Any]] = []
_EGRESS_AUDIT_LOCK = threading.Lock()


def get_egress_audit_log() -> List[Dict[str, Any]]:
    """Returns a copy of all recorded network egress audit events."""
    with _EGRESS_AUDIT_LOCK:
        return [dict(rec) for rec in _EGRESS_AUDIT_LOG]


def clear_egress_audit_log() -> None:
    """Clears recorded network egress audit events."""
    with _EGRESS_AUDIT_LOCK:
        _EGRESS_AUDIT_LOG.clear()


def compute_wilson_score_lower(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> float:
    """Computes the lower bound of the Wilson score confidence interval.

    Parameters
    ----------
    successes : int
        Number of successful / matching evaluations.
    total : int
        Total number of validation evaluations.
    confidence : float
        Statistical confidence level (default 0.95).

    Returns
    -------
    float
        Lower bound of the confidence interval in [0.0, 1.0].
    """
    if total <= 0:
        return 0.0
    if successes <= 0:
        return 0.0

    # Normal critical value approximation (e.g. 1.95996 for 95%)
    if abs(confidence - 0.95) < 0.01:
        z = 1.959963984540054
    elif abs(confidence - 0.99) < 0.01:
        z = 2.5758293035489004
    elif abs(confidence - 0.90) < 0.01:
        z = 1.6448536269514722
    else:
        p = 1.0 - (1.0 - confidence) / 2.0
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308
        z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)

    p_hat = float(successes) / float(total)
    denominator = 1.0 + (z * z) / float(total)
    center = p_hat + (z * z) / (2.0 * float(total))
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) / float(total)) + ((z * z) / (4.0 * float(total) * float(total))))

    lower = (center - margin) / denominator
    return max(0.0, min(1.0, lower))


def _evidence_keys(item: Mapping[str, Any]) -> Set[str]:
    keys = {f"{key}:{item[key]}" for key in ("group_id", "lineage_id", "request_id", "id", "nonce")
            if item.get(key) is not None}
    state = item.get("state", "")
    if state:
        text = _content_text(state)
        keys.add("state:" + " ".join(text.split()).casefold())
    return keys


def _binomial_lower_bound(successes: int, total: int, failure_probability: float) -> float:
    """One-sided exact binomial lower bound, without a statistics dependency."""
    if successes == 0 or total == 0:
        return 0.0
    if successes == total:
        return failure_probability ** (1.0 / total)
    coefficients = [math.lgamma(total + 1) - math.lgamma(k + 1) - math.lgamma(total - k + 1)
                    for k in range(successes, total + 1)]
    low, high = 0.0, 1.0
    for _ in range(50):
        p = (low + high) / 2
        tail = sum(math.exp(c + k * math.log(p) + (total - k) * math.log1p(-p))
                   for k, c in zip(range(successes, total + 1), coefficients))
        if tail > failure_probability:
            high = p
        else:
            low = p
    return low


def _evidence_groups(history: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Keep every connected request/lineage/prompt family in one evidence unit."""
    parent = list(range(len(history)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, item in enumerate(history):
        for key in _evidence_keys(item):
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = collections.defaultdict(list)
    for i, item in enumerate(history):
        groups[find(i)].append(item)
    return list(groups.values())


@dataclass
class CutoverPartition:
    """Disjoint data partitions for cutover distillation, calibration, and validation."""

    train_history: List[Dict[str, Any]]
    calib_history: List[Dict[str, Any]]
    val_history: List[Dict[str, Any]]
    total_samples: int
    train_ratio: float = 0.60
    calib_ratio: float = 0.20
    val_ratio: float = 0.20

    def assert_disjoint(self) -> None:
        """Verify that training, calibration, and validation partitions do not share identical records or lineages."""
        train_keys = set().union(*(_evidence_keys(x) for x in self.train_history))
        calib_keys = set().union(*(_evidence_keys(x) for x in self.calib_history))
        val_keys = set().union(*(_evidence_keys(x) for x in self.val_history))

        train_ids = {id(x) for x in self.train_history}
        calib_ids = {id(x) for x in self.calib_history}
        val_ids = {id(x) for x in self.val_history}

        if (train_ids & val_ids) or (train_keys & val_keys):
            raise AssertionError("Invariant 10 Violation: Training and validation folds share overlapping instances!")
        if (calib_ids & val_ids) or (calib_keys & val_keys):
            raise AssertionError("Invariant 10 Violation: Calibration and validation folds share overlapping instances!")
        if (train_ids & calib_ids) or (train_keys & calib_keys):
            raise AssertionError("Invariant 10 Violation: Training and calibration folds share overlapping instances!")


def partition_cutover_history(
    history: Sequence[Dict[str, Any]],
    train_ratio: float = 0.60,
    calib_ratio: float = 0.20,
    val_ratio: float = 0.20,
) -> CutoverPartition:
    """Decouples query history into 3 strictly disjoint, held-out partitions:
    training (e.g. 60%), calibration (e.g. 20%), and held-out validation (e.g. 20%).

    Ensures zero in-sample contamination between model parameter fitting,
    conformal calibration, and promotion gating using stable request/lineage grouping.
    """
    total = len(history)
    if total == 0:
        return CutoverPartition([], [], [], total_samples=0, train_ratio=train_ratio, calib_ratio=calib_ratio, val_ratio=val_ratio)

    if total < 3:
        return CutoverPartition(
            train_history=list(history),
            calib_history=[],
            val_history=[],
            total_samples=total,
            train_ratio=train_ratio,
            calib_ratio=calib_ratio,
            val_ratio=val_ratio,
        )

    groups = _evidence_groups(history)
    if len(groups) < 3:
        raise ValueError("Insufficient evidence: distinct groups cannot supply the 3 required folds")
    n_val = max(1, int(math.floor(len(groups) * val_ratio)))
    n_calib = max(1, int(math.floor(len(groups) * calib_ratio)))
    n_train = len(groups) - n_val - n_calib
    if n_train < 1:
        raise ValueError("Partition ratios leave no fitting groups")
    train_part = [row for group in groups[:n_train] for row in group]
    calib_part = [row for group in groups[n_train:n_train + n_calib] for row in group]
    val_part = [row for group in groups[n_train + n_calib:] for row in group]

    return CutoverPartition(
        train_history=train_part,
        calib_history=calib_part,
        val_history=val_part,
        total_samples=total,
        train_ratio=train_ratio,
        calib_ratio=calib_ratio,
        val_ratio=val_ratio,
    )


@dataclass
class PromotionPolicy:
    """Per-schema statistical promotion criteria and critical safety guardrails."""

    min_validation_samples: int = 1
    min_agreement_threshold: float = 0.80
    statistical_confidence: float = 0.95
    critical_classes: Set[str] = field(default_factory=lambda: {
        "deny", "unsafe", "fraud", "block", "high_risk", "malicious", "critical", "require_approval"
    })
    false_allow_ceiling: float = 0.0  # Exactly 0.0% tolerance: zero false-allows permitted
    require_statistical_bound: bool = True
    min_local_acceptance: float = 0.8
    # None preserves the original policy. Set, for example, .95 to qualify
    # agreement specifically among the decisions that will run unattended.
    min_accepted_agreement: Optional[float] = None

    def __post_init__(self) -> None:
        if self.min_accepted_agreement is not None and not 0 < self.min_accepted_agreement <= 1:
            raise ValueError("min_accepted_agreement must be between 0 (exclusive) and 1")


@dataclass
class PromotionReport:
    """Detailed audit report evaluating generalization eligibility for local cutover."""

    is_eligible: bool
    total_validation_checks: int
    matching_checks: int
    agreement_rate: float
    wilson_lower_bound: float
    false_allow_count: int
    critical_class_count: int
    false_allow_rate: float
    rejection_reasons: List[str]
    schema_digest: str = ""
    metrics_per_field: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    artifact_digest: str = ""
    manifest: Dict[str, Any] = field(default_factory=dict)
    validation_requests: int = 0
    accepted_requests: int = 0
    local_acceptance_rate: float = 0.0
    accepted_agreement_rate: Optional[float] = None
    independent_validation_groups: int = 0
    exact_lower_bound: float = 0.0
    acceptance_lower_bound: float = 0.0
    validation_attempt: Optional[int] = None
    accepted_agreement_lower_bound: Optional[float] = None


def _is_critical_class(value: Any, critical_classes: Set[str]) -> bool:
    """Check whether a target value represents a safety-critical class."""
    if value is False:
        return True
    s_val = str(value).strip().lower()
    return s_val in critical_classes or s_val in ("false", "deny", "unsafe", "block")


def _is_allow_class(value: Any) -> bool:
    """Check whether a predicted value represents an allow/permissive action."""
    if value is True:
        return True
    s_val = str(value).strip().lower()
    return s_val in (
        "true", "allow", "safe", "permit", "pass",
        "approve", "proceed", "grant", "yes", "enable", "continue",
        "execute", "accept", "confirm",
    )


def evaluate_promotion_eligibility(
    engine: Any,
    val_history: Sequence[Dict[str, Any]],
    schema: Any,
    policy: Optional[PromotionPolicy] = None,
    *,
    validation_attempt: Optional[int] = None,
) -> PromotionReport:
    """Evaluates cutover promotion eligibility strictly on held-out validation data.

    Enforces:
    1. Minimum validation sample thresholds.
    2. Minimum empirical agreement on held-out data.
    3. Strict zero-tolerance false-allow ceiling on critical security classes.
    4. Optional Wilson score statistical lower bound.
    """
    if policy is None:
        policy = PromotionPolicy()

    rejection_reasons: List[str] = []
    total_checks = 0
    matching = 0
    false_allows = 0
    critical_targets = 0
    per_field_stats: Dict[str, Dict[str, Any]] = collections.defaultdict(lambda: {
        "total": 0, "matching": 0, "false_allows": 0, "critical_targets": 0
    })

    groups = _evidence_groups(val_history)

    scored_units = 0
    sum_unit_rates = 0.0
    validation_requests = accepted_requests = accepted_checks = accepted_matches = 0
    accepted_units = accepted_matching_units = 0

    for g_items in groups:
        unit_has_scored_evidence = False
        unit_matching = 0
        unit_total = 0
        unit_accepted = True

        for item in g_items:
            state = item.get("state", "")
            answers = item.get("answers", {})
            if not answers:
                rejection_reasons.append("Validation response is missing required answers")
                unit_accepted = False
                continue

            res = engine.decide(state, record_receipt=False)
            validation_requests += 1
            accepted = not getattr(res, "is_ambiguous", True)
            accepted_requests += int(accepted)
            unit_accepted = unit_accepted and accepted

            for f_name, f_def in schema.fields.items():
                if f_name not in answers or answers[f_name] is None:
                    rejection_reasons.append(f"Validation response is missing field {f_name!r}")
                    unit_accepted = False
                    continue

                target_v = answers[f_name]
                pred_v = res.values.get(f_name)
                if f_def.metadata.get("typesafe_score_levels"):
                    pred_v = sum(float(k) * p for k, p in res.probabilities[f_name].items())

                total_checks += 1
                unit_total += 1
                per_field_stats[f_name]["total"] += 1
                unit_has_scored_evidence = True

                is_target_critical = _is_critical_class(target_v, policy.critical_classes)
                if is_target_critical:
                    critical_targets += 1
                    per_field_stats[f_name]["critical_targets"] += 1

                # Check matching correctness
                is_match = False
                if (
                    isinstance(pred_v, (int, float))
                    and isinstance(target_v, (int, float))
                    and not isinstance(pred_v, bool)
                    and not isinstance(target_v, bool)
                ):
                    margin = max(0.5, 0.20 * (getattr(f_def, "max_value", 1.0) - getattr(f_def, "min_value", 0.0))) if f_def else 0.5
                    if abs(float(pred_v) - float(target_v)) <= margin:
                        is_match = True
                elif isinstance(f_def, MultiChoiceField):
                    s_pred = set(pred_v) if isinstance(pred_v, (list, tuple, set)) else ({pred_v} if pred_v else set())
                    s_target = set(target_v) if isinstance(target_v, (list, tuple, set)) else ({target_v} if target_v else set())
                    if s_pred == s_target:
                        is_match = True
                elif pred_v == target_v:
                    is_match = True

                if is_match:
                    matching += 1
                    unit_matching += 1
                    per_field_stats[f_name]["matching"] += 1
                else:
                    if is_target_critical and _is_allow_class(pred_v):
                        false_allows += 1
                        per_field_stats[f_name]["false_allows"] += 1
                if accepted:
                    accepted_checks += 1
                    accepted_matches += int(is_match)

        if unit_has_scored_evidence:
            scored_units += 1
            if unit_total > 0:
                # One independent family is one Bernoulli observation. Every
                # field and every member must agree; extra easy fields or retries
                # cannot dilute an incorrect decision.
                matched = unit_matching == unit_total
                sum_unit_rates += int(matched)
                accepted_units += int(unit_accepted)
                accepted_matching_units += int(unit_accepted and matched)

    if scored_units == 0:
        agreement_rate = 0.0
        rejection_reasons.append("Zero scored validation checks; cannot evaluate promotion")
    else:
        agreement_rate = sum_unit_rates / scored_units

    effective_n = scored_units
    effective_matching = agreement_rate * effective_n
    wilson_lower = compute_wilson_score_lower(effective_matching, effective_n, confidence=policy.statistical_confidence) if effective_n > 0 else 0.0
    # Share the error budget across every exact bound used for qualification.
    bound_count = 3 if policy.min_accepted_agreement is not None else 2
    failure_probability = (1.0 - policy.statistical_confidence) / bound_count
    if not 0 < policy.statistical_confidence < 1:
        raise ValueError("statistical_confidence must be between 0 and 1")
    if validation_attempt is not None:
        if validation_attempt < 1:
            raise ValueError("validation_attempt must be positive")
        failure_probability /= validation_attempt * (validation_attempt + 1)
    exact_lower = _binomial_lower_bound(int(sum_unit_rates), effective_n, failure_probability)
    acceptance_lower = _binomial_lower_bound(accepted_units, effective_n, failure_probability)
    false_allow_rate = (false_allows / critical_targets) if critical_targets > 0 else 0.0
    acceptance_rate = accepted_units / effective_n if effective_n else 0.0
    accepted_agreement = accepted_matching_units / accepted_units if accepted_units else None
    accepted_lower = _binomial_lower_bound(accepted_matching_units, accepted_units, failure_probability)

    effective_thresh = policy.min_agreement_threshold

    if scored_units < policy.min_validation_samples:
        rejection_reasons.append(
            f"Validation sample count ({scored_units}) below minimum threshold ({policy.min_validation_samples})"
        )

    if agreement_rate < effective_thresh:
        rejection_reasons.append(
            f"Held-out agreement rate ({agreement_rate:.4f}) below minimum threshold ({policy.min_agreement_threshold:.4f})"
        )

    if policy.require_statistical_bound and wilson_lower < policy.min_agreement_threshold:
        rejection_reasons.append(
            f"Wilson statistical lower bound ({wilson_lower:.4f}) below required threshold ({policy.min_agreement_threshold:.4f})"
        )

    if false_allow_rate > policy.false_allow_ceiling:
        rejection_reasons.append(
            f"Critical false-allow rate ({false_allow_rate:.4f}, count={false_allows}) exceeds strict ceiling ({policy.false_allow_ceiling:.4f})"
        )

    if policy.require_statistical_bound and exact_lower < policy.min_agreement_threshold:
        rejection_reasons.append(f"Exact statistical lower bound ({exact_lower:.4f}) below required threshold ({policy.min_agreement_threshold:.4f})")

    if acceptance_rate < policy.min_local_acceptance:
        rejection_reasons.append(
            f"Local acceptance rate ({acceptance_rate:.4f}) below required threshold ({policy.min_local_acceptance:.4f})"
        )
    if policy.require_statistical_bound and acceptance_lower < policy.min_local_acceptance:
        rejection_reasons.append(f"Local acceptance lower bound ({acceptance_lower:.4f}) below required threshold ({policy.min_local_acceptance:.4f})")
    if policy.min_local_acceptance > 0 and accepted_agreement is not None and accepted_agreement < policy.min_agreement_threshold:
        rejection_reasons.append("Agreement on accepted decisions below the required threshold")
    if policy.min_accepted_agreement is not None:
        if accepted_agreement is None:
            rejection_reasons.append("No accepted validation groups; cannot qualify accepted agreement")
        elif accepted_agreement < policy.min_accepted_agreement:
            rejection_reasons.append(
                f"Accepted agreement ({accepted_agreement:.4f}) below required threshold ({policy.min_accepted_agreement:.4f})"
            )
        if policy.require_statistical_bound and accepted_lower < policy.min_accepted_agreement:
            rejection_reasons.append(
                f"Accepted agreement lower bound ({accepted_lower:.4f}) below required threshold ({policy.min_accepted_agreement:.4f})"
            )

    is_eligible = len(rejection_reasons) == 0

    return PromotionReport(
        is_eligible=is_eligible,
        total_validation_checks=total_checks,
        matching_checks=matching,
        agreement_rate=round(agreement_rate, 4),
        wilson_lower_bound=round(wilson_lower, 4),
        false_allow_count=false_allows,
        critical_class_count=critical_targets,
        false_allow_rate=round(false_allow_rate, 4),
        rejection_reasons=rejection_reasons,
        schema_digest=getattr(schema, "schema_digest", lambda: "")() if hasattr(schema, "schema_digest") else "",
        metrics_per_field=dict(per_field_stats),
        artifact_digest="",
        manifest={},
        validation_requests=validation_requests,
        accepted_requests=accepted_requests,
        independent_validation_groups=effective_n,
        exact_lower_bound=round(exact_lower, 4),
        acceptance_lower_bound=round(acceptance_lower, 4),
        validation_attempt=validation_attempt,
        local_acceptance_rate=round(acceptance_rate, 4),
        accepted_agreement_rate=round(accepted_agreement, 4) if accepted_agreement is not None else None,
        accepted_agreement_lower_bound=round(accepted_lower, 4) if accepted_units else None,
    )


class DriftDetector:
    """Sliding-window concept drift and out-of-distribution monitor for post-cutover execution."""

    def __init__(
        self,
        window_size: int = 100,
        ambiguity_threshold: float = 0.20,
        ood_threshold: float = 0.05,
    ) -> None:
        self.window_size = int(window_size)
        self.ambiguity_threshold = float(ambiguity_threshold)
        self.ood_threshold = float(ood_threshold)
        self._window: collections.deque = collections.deque(maxlen=self.window_size)
        self._drift_detected: bool = False
        self._last_drift_reason: str = ""

    def record(self, is_ambiguous: bool, is_ood: bool, confidence: float = 1.0) -> None:
        """Record a single inference evaluation telemetry observation."""
        self._window.append({
            "is_ambiguous": bool(is_ambiguous),
            "is_ood": bool(is_ood),
            "confidence": float(confidence),
        })
        self._evaluate()

    def _evaluate(self) -> None:
        if len(self._window) < min(10, self.window_size):
            return

        amb_count = sum(1 for w in self._window if w["is_ambiguous"])
        ood_count = sum(1 for w in self._window if w["is_ood"])
        n = len(self._window)

        amb_rate = amb_count / n
        ood_rate = ood_count / n

        if amb_rate >= self.ambiguity_threshold:
            self._drift_detected = True
            self._last_drift_reason = f"Concept drift detected: ambiguity rate {amb_rate:.1%} exceeds threshold {self.ambiguity_threshold:.1%}"
        elif ood_rate >= self.ood_threshold:
            self._drift_detected = True
            self._last_drift_reason = f"Distributional drift detected: OOD rate {ood_rate:.1%} exceeds threshold {self.ood_threshold:.1%}"
        else:
            self._drift_detected = False
            self._last_drift_reason = ""

    @property
    def is_drifted(self) -> bool:
        return self._drift_detected

    @property
    def drift_reason(self) -> str:
        return self._last_drift_reason

    @property
    def sample_count(self) -> int:
        return len(self._window)


def _content_text(value: Any) -> str:
    """Canonical local representation; HTTP requests keep the original JSON value."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _teaching_target(field: DecisionField, value: Any) -> Any:
    if field.metadata.get("typesafe_score_levels"):
        number = float(value)
        if not math.isfinite(number) or not 0 <= number <= len(field.options) - 1:
            raise ValueError("Teacher ordinal score is outside the question's range")
        return str(int(math.floor(number + 0.5)))
    return value


def _ordinal_answer(field: DecisionField, probabilities: Mapping[str, float]) -> Dict[str, Any]:
    levels = field.metadata["typesafe_score_levels"]
    probs = {int(k): float(v) for k, v in probabilities.items()}
    score = sum(k * p for k, p in probs.items())
    return {"score": score, "value": score, "probabilities": probs,
            "legend": dict(enumerate(levels))}


# The official SDK exposes this constructor as a TypedDict.
NoulCriteria = dict


class DotDict(dict):
    """Dict supporting dot notation attribute access for nested response fields."""

    def __getattr__(self, name: str) -> Any:
        if name in self:
            val = self[name]
            if isinstance(val, dict) and not isinstance(val, DotDict):
                val = DotDict(val)
                self[name] = val
            return val
        raise AttributeError(f"'DotDict' object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        if name in self:
            del self[name]
        else:
            raise AttributeError(f"'DotDict' object has no attribute {name!r}")


class Usage(DotDict):
    """Token consumption metadata matching TypeSafe AI API schema."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        inp = self.get("input_tokens")
        out = self.get("output_tokens")
        tot = self.get("total_tokens")

        inp_int = int(inp) if inp is not None else 0
        out_int = int(out) if out is not None else 0
        if tot is None:
            self["total_tokens"] = inp_int + out_int
        else:
            try:
                self["total_tokens"] = int(tot)
            except (ValueError, TypeError):
                self["total_tokens"] = inp_int + out_int

    @property
    def total_tokens(self) -> int:
        val = self.get("total_tokens")
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
        inp = int(self.get("input_tokens", 0) or 0)
        out = int(self.get("output_tokens", 0) or 0)
        return inp + out

    @total_tokens.setter
    def total_tokens(self, value: Any) -> None:
        self["total_tokens"] = int(value) if value is not None else 0


class TypeSafeResponse(DotDict):
    """Response object matching TypeSafe AI's HTTP JSON response schema."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if "answers" in self and isinstance(self["answers"], dict):
            enriched_answers = {}
            for k, v in self["answers"].items():
                if isinstance(v, dict):
                    v_dict = DotDict(v)
                    ans_type = v_dict.get("type")
                    if ans_type == "choice":
                        if "value" not in v_dict and "choice" in v_dict:
                            v_dict["value"] = v_dict["choice"]
                        if "choice" not in v_dict and "value" in v_dict:
                            v_dict["choice"] = v_dict["value"]
                        v_dict.setdefault("conformal_set", None)
                    elif ans_type == "noul":
                        if "noul" in v_dict:
                            p = float(v_dict["noul"])
                            if not math.isfinite(p) or not 0 <= p <= 1:
                                raise ValueError("Noul probability must be finite and between 0 and 1")
                        if "value" not in v_dict and "noul" in v_dict:
                            v_dict["value"] = p >= 0.5
                        if "noul" not in v_dict and "value" in v_dict:
                            v_dict["noul"] = 1.0 if v_dict["value"] else 0.0
                        if "confidence" not in v_dict and "noul" in v_dict:
                            try:
                                p = float(v_dict["noul"])
                                v_dict["confidence"] = max(p, 1.0 - p)
                            except (TypeError, ValueError):
                                v_dict["confidence"] = 0.5
                        v_dict.setdefault("conformal_set", None)
                    elif ans_type == "score":
                        if "value" not in v_dict and "score" in v_dict:
                            v_dict["value"] = v_dict["score"]
                        if "score" not in v_dict and "value" in v_dict:
                            v_dict["score"] = v_dict["value"]
                        v_dict.setdefault("conformal_set", None)
                        for key in ("probabilities", "legend"):
                            if key in v_dict:
                                v_dict[key] = {int(level): value for level, value in v_dict[key].items()}
                    elif ans_type in ("multi_choice", "multichoice"):
                        if "value" not in v_dict and "choices" in v_dict:
                            v_dict["value"] = v_dict["choices"]
                        if "choices" not in v_dict and "value" in v_dict:
                            v_dict["choices"] = v_dict["value"]
                        v_dict.setdefault("conformal_set", None)
                    else:
                        v_dict.setdefault("conformal_set", None)
                    enriched_answers[k] = v_dict
                else:
                    enriched_answers[k] = v
            self["answers"] = DotDict(enriched_answers)
        if "usage" in self and isinstance(self["usage"], dict):
            self["usage"] = Usage(self["usage"])
        self.setdefault("receipt", None)
        self.setdefault("baseline_fallback", False)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)

    @property
    def choices(self) -> DotDict:
        return DotDict({k: v for k, v in self.answers.items() if v.get("type") == "choice"})

    @property
    def nouls(self) -> DotDict:
        return DotDict({k: v for k, v in self.answers.items() if v.get("type") == "noul"})

    @property
    def scores(self) -> DotDict:
        return DotDict({k: v for k, v in self.answers.items() if v.get("type") == "score"})


# Answer type aliases for drop-in TypeSafe AI type annotation parity
ChoiceAnswer = DotDict
NoulAnswer = DotDict
ScoreAnswer = DotDict
MultiChoiceAnswer = DotDict
SystemOneResponse = TypeSafeResponse


class Choice:
    """Categorical choice question definition in TypeSafe AI format."""

    def __init__(
        self,
        instructions: str = "",
        criteria: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
        *,
        options: Optional[Sequence[str]] = None,
        descriptions: Optional[Mapping[str, str]] = None,
        default: Optional[str] = None,
        description: str = "",
        required: bool = True,
        **kwargs: Any,
    ) -> None:
        self.type = "choice"
        self.instructions = instructions or description
        self.description = description or instructions
        self.default = default
        self.required = required

        if criteria is not None:
            if isinstance(criteria, str):
                raise TypeError("Invalid criteria type for Choice: str; expected Sequence[str] or Mapping[str, str]")
            elif isinstance(criteria, Mapping):
                self.criteria = dict(criteria)
                self.options = tuple(str(k) for k in criteria.keys())
                self.descriptions = {str(k): _content_text(v) if v is not None else str(k) for k, v in criteria.items()}
            elif isinstance(criteria, (Sequence, set)):
                self.criteria = list(criteria)
                self.options = tuple(str(x) for x in criteria)
                self.descriptions = {str(x): str(x) for x in criteria}
            else:
                raise TypeError(f"Invalid criteria type for Choice: {type(criteria).__name__}")
        elif options is not None:
            self.options = tuple(str(x) for x in options)
            self.descriptions = dict(descriptions or {})
            self.criteria = self.descriptions if self.descriptions else list(self.options)
        else:
            raise ValueError("Choice question requires 'criteria' or 'options'")

    def to_dict(self) -> Dict[str, Any]:
        if isinstance(self.criteria, Mapping):
            crit = dict(self.criteria)
        elif self.descriptions:
            crit = dict(self.descriptions)
        else:
            crit = {str(opt): str(opt) for opt in self.options}
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": crit,
        }

    def to_field(self, name: Optional[str] = None) -> ChoiceField:
        field = ChoiceField(
            options=list(self.options),
            descriptions=self.descriptions,
            default=self.default,
            description=_content_text(self.instructions),
            required=self.required,
        )
        if name:
            field.bind_name(name)
        return field


class Noul:
    """Binary boolean probability question definition in TypeSafe AI format.

    In TypeSafe AI nomenclature, 'Noul' evaluates boolean probability distributions.
    """

    def __init__(
        self,
        instructions: str = "",
        criteria: Optional[Mapping[str, str]] = None,
        *,
        threshold: float = 0.5,
        default: Optional[bool] = None,
        description: str = "",
        required: bool = True,
        **kwargs: Any,
    ) -> None:
        self.type = "noul"
        self.instructions = instructions or description
        self.description = description or instructions
        self.threshold = threshold
        self.default = default
        self.required = required

        if criteria is not None:
            self.criteria = dict(criteria)
        else:
            self.criteria = {
                "true": "Condition is met or statement is true",
                "false": "Condition is not met or statement is false",
            }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "noul",
            "instructions": self.instructions,
            "criteria": self.criteria,
        }

    def to_field(self, name: Optional[str] = None) -> BooleanField:
        true_desc = _content_text(self.criteria.get("true")) or "True / Positive"
        false_desc = _content_text(self.criteria.get("false")) or "False / Negative"
        instructions = _content_text(self.instructions)
        if instructions:
            true_desc = f"{instructions} Yes. {true_desc}"
            false_desc = f"{instructions} No. {false_desc}"
        field = BooleanField(
            threshold=self.threshold,
            true_description=true_desc,
            false_description=false_desc,
            default=self.default,
            description=instructions,
            required=self.required,
        )
        if name:
            field.bind_name(name)
        return field


class Score:
    """Continuous or ordinal rating question definition in TypeSafe AI format."""

    def __init__(
        self,
        instructions: str = "",
        criteria: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
        *,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        default: Optional[float] = None,
        description: str = "",
        required: bool = True,
        **kwargs: Any,
    ) -> None:
        self.type = "score"
        self.instructions = instructions or description
        self.description = description or instructions
        self.default = default
        self.required = required
        self.ordinal = isinstance(criteria, (list, tuple)) and min_value is None and max_value is None
        if self.ordinal and not 2 <= len(criteria) <= 10:
            raise ValueError("Ordinal Score requires between 2 and 10 levels")
        self.min_value = float(min_value if min_value is not None else 0.0)

        # In TypeSafe, if criteria is a list of rating labels like ['Trivial', 'Easy', 'Moderate', 'Hard'],
        # the score range spans from 0 to len(criteria)-1 (or max_value if explicitly provided).
        if criteria is not None and isinstance(criteria, (list, tuple)) and len(criteria) > 1 and max_value is None:
            self.max_value = float(len(criteria) - 1)
        else:
            self.max_value = float(max_value if max_value is not None else 1.0)

        self.criteria = criteria

    def _get_criteria_list(self) -> List[str]:
        if self.criteria is not None:
            if isinstance(self.criteria, (list, tuple, set)):
                return list(self.criteria)
            elif isinstance(self.criteria, Mapping):
                try:
                    sorted_keys = sorted(self.criteria.keys(), key=lambda k: float(k))
                except (ValueError, TypeError):
                    sorted_keys = sorted(self.criteria.keys(), key=str)
                return [str(self.criteria[k]) for k in sorted_keys]
        int_min = int(self.min_value)
        int_max = int(self.max_value)
        if 0 < (int_max - int_min) <= 10:
            return [str(i) for i in range(int_min, int_max + 1)]
        return ["Min", "Max"]

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": "score",
            "instructions": self.instructions,
            "criteria": self._get_criteria_list(),
        }
        if not self.ordinal:
            result.update(min_value=self.min_value, max_value=self.max_value)
        return result

    def to_field(self, name: Optional[str] = None) -> DecisionField:
        if self.ordinal:
            field = ChoiceField(
                options=[str(i) for i in range(len(self.criteria))],
                descriptions={str(i): _content_text(level) for i, level in enumerate(self.criteria)},
                description=_content_text(self.instructions), required=self.required,
                metadata={"typesafe_score_levels": list(self.criteria)},
            )
            if name:
                field.bind_name(name)
            return field
        low_desc = "Minimum boundary"
        high_desc = "Maximum boundary"
        if isinstance(self.criteria, (list, tuple)) and self.criteria:
            low_desc = str(self.criteria[0])
            high_desc = str(self.criteria[-1])
        elif isinstance(self.criteria, Mapping) and self.criteria:
            try:
                keys = sorted(self.criteria.keys(), key=lambda k: float(k))
            except (ValueError, TypeError):
                keys = sorted(self.criteria.keys(), key=str)
            low_desc = str(self.criteria[keys[0]])
            high_desc = str(self.criteria[keys[-1]])

        field = ScoreField(
            min_value=self.min_value,
            max_value=self.max_value,
            low_description=low_desc,
            high_description=high_desc,
            default=self.default,
            description=_content_text(self.instructions),
            required=self.required,
        )
        if name:
            field.bind_name(name)
        return field


class MultiChoice:
    """Multi-label choice question definition in TypeSafe AI format.

    Evaluates whether each independent option or policy tag applies simultaneously.
    """

    def __init__(
        self,
        instructions: str = "",
        criteria: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
        *,
        options: Optional[Sequence[str]] = None,
        descriptions: Optional[Mapping[str, str]] = None,
        threshold: float = 0.5,
        default: Optional[Sequence[str]] = None,
        description: str = "",
        required: bool = False,
        **kwargs: Any,
    ) -> None:
        self.type = "multi_choice"
        self.instructions = instructions or description
        self.description = description or instructions
        self.threshold = float(threshold)
        self.default = list(default) if default is not None else None
        self.required = required

        if criteria is not None:
            if isinstance(criteria, str):
                raise TypeError("Invalid criteria type for MultiChoice: str; expected Sequence[str] or Mapping[str, str]")
            elif isinstance(criteria, Mapping):
                self.criteria = dict(criteria)
                self.options = tuple(str(k) for k in criteria.keys())
                self.descriptions = {str(k): str(v) for k, v in criteria.items()}
            elif isinstance(criteria, (Sequence, set)):
                self.criteria = list(criteria)
                self.options = tuple(str(x) for x in criteria)
                self.descriptions = {str(x): str(x) for x in criteria}
            else:
                raise TypeError(f"Invalid criteria type for MultiChoice: {type(criteria).__name__}")
        elif options is not None:
            self.options = tuple(str(x) for x in options)
            self.descriptions = dict(descriptions or {})
            self.criteria = self.descriptions if self.descriptions else list(self.options)
        else:
            raise ValueError("MultiChoice question requires 'criteria' or 'options'")

    def to_dict(self) -> Dict[str, Any]:
        if isinstance(self.criteria, Mapping):
            crit = dict(self.criteria)
        elif self.descriptions:
            crit = dict(self.descriptions)
        else:
            crit = {str(opt): str(opt) for opt in self.options}
        return {
            "type": "multi_choice",
            "instructions": self.instructions,
            "criteria": crit,
            "threshold": self.threshold,
        }

    def to_field(self, name: Optional[str] = None) -> MultiChoiceField:
        field = MultiChoiceField(
            options=list(self.options),
            threshold=self.threshold,
            descriptions=self.descriptions if self.descriptions else None,
            default=self.default,
            description=self.instructions,
            required=self.required,
        )
        if name:
            field.bind_name(name)
        return field


def _build_field_from_question(name: str, q_spec: Any) -> DecisionField:
    """Converts a TypeSafe question definition or dictionary into a DecisionField."""
    if hasattr(q_spec, "to_field"):
        return q_spec.to_field(name)

    if isinstance(q_spec, DecisionField):
        q_spec.bind_name(name)
        return q_spec

    if hasattr(q_spec, "model_dump"):
        q_spec = q_spec.model_dump(exclude_none=True)

    if isinstance(q_spec, Mapping):
        q_type = str(q_spec.get("type", "")).lower().strip()
        instructions = q_spec.get("instructions", q_spec.get("description", ""))
        criteria = q_spec.get("criteria")

        if q_type == "choice":
            return Choice(instructions=instructions, criteria=criteria).to_field(name)
        elif q_type in ("noul", "boolean"):
            thresh = float(q_spec.get("threshold", 0.5))
            return Noul(instructions=instructions, criteria=criteria, threshold=thresh).to_field(name)
        elif q_type == "score":
            min_v = q_spec.get("min_value")
            max_v = q_spec.get("max_value")
            return Score(instructions=instructions, criteria=criteria, min_value=min_v, max_value=max_v).to_field(name)
        elif q_type in ("multi_choice", "multichoice"):
            opts = list(criteria.keys()) if isinstance(criteria, Mapping) else list(criteria or [])
            thresh = float(q_spec.get("threshold", 0.5))
            descs = dict(criteria) if isinstance(criteria, Mapping) else None
            f = MultiChoiceField(options=opts, descriptions=descs, threshold=thresh, description=instructions)
            f.bind_name(name)
            return f
        else:
            raise ValueError(f"Unsupported question type {q_type!r} for question {name!r}")

    raise TypeError(f"Cannot construct question field from {type(q_spec).__name__} for {name!r}")


def _build_dynamic_schema(questions: Mapping[str, Any], schema_name: str = "TypeSafeDynamicSchema") -> DecisionSchema:
    """Constructs a DecisionSchema instance dynamically from a TypeSafe questions dictionary."""
    fields: Dict[str, DecisionField] = {}
    for name, spec in questions.items():
        fields[name] = _build_field_from_question(name, spec)
    return DecisionSchema(fields=fields, schema_name=schema_name)


def create_typesafe_baseline_response(
    state: str,
    questions: Mapping[str, Any],
    model: str = "jev-latest",
    measured_latency_ms: Optional[float] = None,
    egress_bytes: Optional[int] = None,
) -> TypeSafeResponse:
    """Generates a realistic TypeSafe AI (Jev) baseline response profile.

    Simulates cloud SaaS WAN roundtrips, token billing, and point probabilities
    when running offline or without an active API key.
    """
    if measured_latency_ms is None or measured_latency_ms < 5.0:
        latency_ms = 220.0
    else:
        latency_ms = float(measured_latency_ms)

    serialized_questions: Dict[str, Any] = {}
    for k, v in questions.items():
        if hasattr(v, "to_dict"):
            serialized_questions[k] = v.to_dict()
        elif isinstance(v, Mapping):
            serialized_questions[k] = dict(v)
        else:
            serialized_questions[k] = str(v)

    if egress_bytes is None:
        payload = {"model": model, "state": state, "questions": serialized_questions}
        egress_bytes = len(json.dumps(payload).encode("utf-8"))

    # Estimate realistic token consumption
    prompt_tokens = max(10, len(_content_text(state).split()))
    question_tokens = max(20, len(json.dumps(serialized_questions)) // 4)
    input_tokens = prompt_tokens + question_tokens
    output_tokens = max(15, 20 * len(questions))

    # Run zero-shot semantic inference to produce realistic, context-aware baseline predictions
    zero_shot_fields: Dict[str, Any] = {}
    try:
        from system1.core.model import SystemOneModel
        schema_obj = _build_dynamic_schema(questions)
        zs_model = SystemOneModel(schema_obj)
        zs_res = zs_model.forward_single(_content_text(state))
        zero_shot_fields = zs_res.fields
    except Exception:
        zero_shot_fields = {}

    answers: Dict[str, Any] = {}
    for q_name, q_spec in questions.items():
        q_type = getattr(q_spec, "type", None)
        if q_type is None and isinstance(q_spec, Mapping):
            q_type = q_spec.get("type")
        if q_type is None and hasattr(q_spec, "field_type"):
            q_type = getattr(q_spec, "field_type")
        if isinstance(q_type, str):
            q_type = q_type.lower().strip()

        zs_field = zero_shot_fields.get(q_name)

        if q_type == "choice":
            options: List[str] = []
            if hasattr(q_spec, "options") and q_spec.options:
                options = list(q_spec.options)
            elif isinstance(q_spec, Mapping):
                crit = q_spec.get("criteria")
                if isinstance(crit, Mapping):
                    options = list(crit.keys())
                elif isinstance(crit, (list, tuple)):
                    options = list(crit)
                elif "options" in q_spec:
                    options = list(q_spec["options"])

            if zs_field is not None and getattr(zs_field, "selected_value", None) in options:
                chosen = zs_field.selected_value
                conf = float(getattr(zs_field, "confidence", 0.88))
                if hasattr(zs_field, "raw_probabilities") and hasattr(zs_field, "options"):
                    probs = {
                        opt: round(float(p), 4)
                        for opt, p in zip(zs_field.options, zs_field.raw_probabilities)
                    }
                else:
                    probs = {
                        opt: (conf if opt == chosen else round((1.0 - conf) / max(1, len(options) - 1), 3))
                        for opt in options
                    }
            else:
                chosen = options[0] if options else "default"
                conf = 0.88
                probs = {
                    opt: (conf if opt == chosen else round(0.12 / max(1, len(options) - 1), 3))
                    for opt in options
                } if options else {}

            answers[q_name] = ChoiceAnswer({
                "type": "choice",
                "choice": chosen,
                "value": chosen,
                "confidence": conf,
                "conformal_set": None,
                "probabilities": probs,
            })
        elif q_type in ("noul", "boolean"):
            if zs_field is not None and hasattr(zs_field, "selected_value"):
                bool_val = bool(zs_field.selected_value)
                conf = float(getattr(zs_field, "confidence", 0.85))
                if hasattr(zs_field, "raw_probabilities") and len(zs_field.raw_probabilities) > 1:
                    prob_val = round(float(zs_field.raw_probabilities[1]), 4)
                else:
                    prob_val = conf if bool_val else round(1.0 - conf, 4)
            else:
                bool_val = True
                conf = 0.85
                prob_val = 0.85

            answers[q_name] = NoulAnswer({
                "type": "noul",
                "noul": prob_val,
                "value": bool_val,
                "confidence": conf,
                "conformal_set": None,
            })
        elif q_type == "score":
            ordinal_field = _build_field_from_question(q_name, q_spec)
            if ordinal_field.metadata.get("typesafe_score_levels") and zs_field is not None:
                answers[q_name] = ScoreAnswer({
                    "type": "score", "confidence": float(zs_field.confidence), "conformal_set": None,
                    **_ordinal_answer(ordinal_field, dict(zip(zs_field.options, zs_field.raw_probabilities))),
                })
                continue
            min_v = 0.0
            max_v = 1.0
            if hasattr(q_spec, "min_value"):
                min_v = float(q_spec.min_value)
                max_v = float(q_spec.max_value)
            elif isinstance(q_spec, Mapping):
                min_v = float(q_spec.get("min_value", 0.0))
                max_v = float(q_spec.get("max_value", 1.0))

            if zs_field is not None and hasattr(zs_field, "selected_value"):
                score_val = round(float(zs_field.selected_value), 2)
                conf = float(getattr(zs_field, "confidence", 0.80))
            else:
                score_val = round((min_v + max_v) / 2.0, 2)
                conf = 0.80

            answers[q_name] = ScoreAnswer({
                "type": "score",
                "score": score_val,
                "value": score_val,
                "confidence": conf,
                "conformal_set": None,
            })
        elif q_type in ("multi_choice", "multichoice"):
            options = []
            if hasattr(q_spec, "options"):
                options = list(q_spec.options)
            elif isinstance(q_spec, Mapping):
                crit = q_spec.get("criteria")
                if isinstance(crit, Mapping):
                    options = list(crit.keys())
                elif isinstance(crit, (list, tuple)):
                    options = list(crit)
            if zs_field is not None and hasattr(zs_field, "selected_value") and isinstance(zs_field.selected_value, (list, tuple)):
                chosen_list = list(zs_field.selected_value)
                conf = float(getattr(zs_field, "confidence", 0.82))
            else:
                default_val = getattr(q_spec, "default", None)
                chosen_list = list(default_val) if default_val is not None else []
                conf = 0.82

            answers[q_name] = MultiChoiceAnswer({
                "type": "multi_choice",
                "choices": chosen_list,
                "value": chosen_list,
                "confidence": conf,
                "conformal_set": None,
            })
        else:
            answers[q_name] = DotDict({
                "value": "baseline_prediction",
                "confidence": 0.80,
                "conformal_set": None,
            })

    return TypeSafeResponse({
        "model": model,
        "state": state,
        "answers": answers,
        "usage": Usage({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }),
        "latency_ms": round(latency_ms, 3),
        "local_execution": False,
        "egress_bytes": egress_bytes,
        "baseline_fallback": True,
        "conformal_alpha": None,
        "receipt": None,
        "is_ambiguous": False,
    })


def call_real_typesafe_api(
    state: str,
    questions: Mapping[str, Any],
    api_key: Optional[str] = None,
    base_url: str = "https://api.typesafe.ai/v1",
    model: str = "jev-latest",
    timeout: float = 15.0,
    fallback_baseline: bool = False,
    zero_egress: bool = True,
) -> Tuple[Optional[TypeSafeResponse], float, int]:
    """Calls the real TypeSafe AI (Jev) API over HTTP WAN, capturing latency and egress.

    If zero_egress is True, immediately raises ZeroEgressViolationError before touching sockets.
    If an API key is provided and the server returns 200 OK, returns the live parsed
    response. In production (fallback_baseline=False), network/auth exceptions raise explicit
    transport errors rather than silently fabricating baseline data.

    Returns:
        Tuple of (response, latency_ms, egress_bytes)
    """
    if zero_egress:
        raise ZeroEgressViolationError(
            "Outbound network egress blocked at provider transport boundary (zero_egress=True)."
        )

    serialized_questions: Dict[str, Any] = {}
    for k, v in questions.items():
        if hasattr(v, "model_dump"):
            v = v.model_dump(exclude_none=True)
        if hasattr(v, "to_dict"):
            serialized_questions[k] = v.to_dict()
        elif isinstance(v, Mapping):
            q_dict = dict(v)
            q_type = str(q_dict.get("type", "")).lower()
            if q_type == "choice" and isinstance(q_dict.get("criteria"), (list, tuple, set)):
                q_dict["criteria"] = {str(opt): str(opt) for opt in q_dict["criteria"]}
            elif q_type == "score":
                if "criteria" not in q_dict or q_dict["criteria"] is None:
                    try:
                        min_v = int(float(q_dict.get("min_value", 0.0)))
                        max_v = int(float(q_dict.get("max_value", 1.0)))
                        if 0 < (max_v - min_v) <= 10:
                            q_dict["criteria"] = [str(i) for i in range(min_v, max_v + 1)]
                        else:
                            q_dict["criteria"] = ["Min", "Max"]
                    except (ValueError, TypeError):
                        q_dict["criteria"] = ["Min", "Max"]
                elif isinstance(q_dict.get("criteria"), Mapping):
                    try:
                        sorted_keys = sorted(q_dict["criteria"].keys(), key=lambda x: float(x))
                    except (ValueError, TypeError):
                        sorted_keys = sorted(q_dict["criteria"].keys(), key=str)
                    q_dict["criteria"] = [str(q_dict["criteria"][x]) for x in sorted_keys]
            serialized_questions[k] = q_dict
        else:
            serialized_questions[k] = str(v)

    for question in serialized_questions.values():
        if isinstance(question, dict) and question.get("type") == "score":
            question.pop("min_value", None)
            question.pop("max_value", None)

    payload = {
        "model": model,
        "state": state,
        "questions": serialized_questions,
    }
    raw_bytes = json.dumps(payload).encode("utf-8")
    egress_bytes = len(raw_bytes)

    if api_key is None:
        api_key = os.environ.get("TYPESAFE_API_KEY", "") or os.environ.get("JEV_API_KEY", "")

    endpoint = f"{base_url.rstrip('/')}/systemone"
    parsed = urllib.parse.urlparse(endpoint)
    host = parsed.hostname or "unknown"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    audit_record = {
        "timestamp": time.time(),
        "destination_host": host,
        "destination_port": port,
        "payload_bytes": egress_bytes,
        "operation_type": "typesafe_wan_call",
        "host": host,
        "port": port,
        "bytes_out": egress_bytes,
        "operation": "typesafe_wan_call",
        "endpoint": endpoint,
    }
    logger.info("Egress event audit: %s", audit_record)
    with _EGRESS_AUDIT_LOCK:
        _EGRESS_AUDIT_LOG.append(audit_record)

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "System 1-TypeSafe-Compat/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(endpoint, data=raw_bytes, headers=headers)
    t0 = time.perf_counter()
    measured_latency_ms: Optional[float] = None
    last_error: Optional[Exception] = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            measured_latency_ms = (time.perf_counter() - t0) * 1000.0
            if isinstance(data, dict):
                data.setdefault("egress_bytes", egress_bytes)
                data.setdefault("local_execution", False)
                data.setdefault("latency_ms", measured_latency_ms)
            return TypeSafeResponse(data), measured_latency_ms, egress_bytes
    except urllib.error.HTTPError as err:
        last_error = err
        measured_latency_ms = (time.perf_counter() - t0) * 1000.0
        try:
            err.close()
        except Exception:
            pass
    except Exception as err:
        last_error = err
        measured_latency_ms = (time.perf_counter() - t0) * 1000.0

    if fallback_baseline:
        baseline = create_typesafe_baseline_response(
            state=state,
            questions=questions,
            model=model,
            measured_latency_ms=measured_latency_ms,
            egress_bytes=egress_bytes,
        )
        if last_error is not None:
            baseline["error"] = str(last_error)
        return baseline, measured_latency_ms, egress_bytes

    if last_error is not None:
        raise last_error

    return None, measured_latency_ms, egress_bytes



class TypeSafeClient:
    """TypeSafe-style decisions using a local skill or an observed teacher.

    Local execution uses small decision heads, not a language model. Teacher
    calls require a callback or explicitly enabled network access. Latency and
    quality depend on the workload; uncertain local answers can require review.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.typesafe.ai/v1",
        *,
        mode: str = "local",
        cutover_threshold: int = 50,
        min_agreement_threshold: float = 0.8,
        signing_key: Optional[Ed25519PrivateKey] = None,
        ledger: Optional[ActionLedger] = None,
        backend: str = "auto",
        dimension: int = 384,
        regularization: float = 1.0,
        projector: Optional[Any] = None,
        timeout: float = 15.0,
        zero_egress: bool = True,
        allow_cloud_fallback: bool = False,
        fallback_baseline: bool = False,
        promotion_policy: Optional[PromotionPolicy] = None,
        drift_detector: Optional[DriftDetector] = None,
        **kwargs: Any,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("TYPESAFE_API_KEY", "") or os.environ.get("JEV_API_KEY", "")
        self.api_key = str(api_key)
        self.base_url = str(base_url)

        valid_modes = ("local", "passthrough", "auto_cutover")
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode {mode!r}. Expected one of: {', '.join(valid_modes)}")
        self.mode = mode
        self.cutover_threshold = int(cutover_threshold)
        agreement_alias = kwargs.get("agreement_threshold", min_agreement_threshold)
        self.min_agreement_threshold = float(agreement_alias)

        self.zero_egress = bool(kwargs.get("zero_egress", zero_egress))
        self.allow_cloud_fallback = bool(kwargs.get("allow_cloud_fallback", allow_cloud_fallback))
        self.fallback_baseline = bool(kwargs.get("fallback_baseline", fallback_baseline))
        self.promotion_policy = promotion_policy or kwargs.get("promotion_policy", None)
        self._drift_detector = drift_detector or kwargs.get("drift_detector", None) or DriftDetector()
        self._drift_detectors: Dict[str, DriftDetector] = {}
        self._last_promotion_report: Optional[PromotionReport] = None

        self.signing_key = signing_key
        self.ledger = ledger
        self.backend = backend
        self.dimension = dimension
        self.regularization = float(regularization)
        if not math.isfinite(self.regularization) or self.regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        self.projector = projector
        self.timeout = timeout
        self.baseline_handler = kwargs.get("baseline_handler", None)
        self.extra_kwargs = kwargs
        self.strict_mode = bool(kwargs.get("strict_mode", True))
        self.augment = bool(kwargs.get("augment", False))
        self.default_model = kwargs.get("model") or "jev-latest"
        self._closed = False
        for name in ("retry", "headers", "transport", "http_client"):
            if kwargs.get(name) is not None:
                raise NotImplementedError(f"The local adapter does not implement the SDK's {name!r} option")

        self._engine_cache: Dict[str, SystemOneEngine] = {}
        self._engine_lock = threading.Lock()
        self._call_count: Dict[str, int] = collections.defaultdict(int)
        self._teacher_sample_count: Dict[str, int] = collections.defaultdict(int)
        self._history: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        self._validation_seen_keys: Dict[str, Set[str]] = collections.defaultdict(set)
        self._validation_attempts: Dict[str, int] = collections.defaultdict(int)
        self._has_cutover: bool = False
        self._has_cutover_schemas: Set[str] = set()
        self._compiled_models: Dict[str, Any] = {}
        self._compiled_model: Optional[Any] = kwargs.get("compiled_model", None)
        if self._compiled_model is None and kwargs.get("model_path") is not None:
            from system1.compiler import CompiledSystemOneModel
            self._compiled_model = CompiledSystemOneModel.load(
                kwargs["model_path"],
                projector=self.projector,
                backend=self.backend,
            )
        if self._compiled_model is not None:
            self._has_cutover = True
            saved_strict = self._compiled_model.metadata.get("typesafe_strict_mode", True)
            if "strict_mode" in kwargs and self.strict_mode != saved_strict:
                raise ValueError("strict_mode must match the saved skill's evaluated setting")
            self.strict_mode = saved_strict
        self._cutover_audit_log: List[Dict[str, Any]] = []

        if self.zero_egress:
            if self.mode == "passthrough":
                raise ValueError(
                    "Incompatible configuration: mode='passthrough' requires network egress, but zero_egress=True."
                )
            if self.mode == "auto_cutover" and self.baseline_handler is None and not self._has_cutover:
                raise ValueError(
                    "Incompatible configuration: mode='auto_cutover' with zero_egress=True "
                    "requires a local baseline_handler to provide teacher exemplars without egress."
                )
            if self.allow_cloud_fallback:
                raise ValueError(
                    "Incompatible configuration: allow_cloud_fallback=True cannot be combined with zero_egress=True."
                )

    def __enter__(self) -> TypeSafeClient:
        if self._closed:
            raise RuntimeError("Client is closed")
        return self

    def close(self) -> None:
        self._closed = True

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @property
    def is_cutover(self) -> bool:
        """Whether the client has automatically cut over to 100% local execution."""
        return self._has_cutover or bool(self._has_cutover_schemas)

    @property
    def call_count(self) -> int:
        """Total number of queries evaluated by this client."""
        return sum(self._call_count.values())

    @property
    def total_requests(self) -> int:
        """Total number of queries evaluated by this client."""
        return sum(self._call_count.values())

    @property
    def teacher_sample_count(self) -> int:
        """Total number of teacher exemplar samples collected."""
        return sum(self._teacher_sample_count.values())

    @property
    def cutover_audit_log(self) -> List[Dict[str, Any]]:
        """Log of cutover audit events."""
        return list(self._cutover_audit_log)

    @property
    def compiled_model(self) -> Optional[Any]:
        """Returns the most recently distilled CompiledSystemOneModel, or None."""
        with self._engine_lock:
            if self._compiled_models:
                return list(self._compiled_models.values())[-1]
            return self._compiled_model

    @property
    def drift_detector(self) -> DriftDetector:
        """Detector for the most recently evaluated schema (each schema is isolated)."""
        return self._drift_detector

    def _record_drift(self, digest: str, ambiguous: bool, ood: bool, confidence: float) -> bool:
        with self._engine_lock:
            if digest not in self._drift_detectors:
                template = self._drift_detector
                self._drift_detectors[digest] = template if not self._drift_detectors else DriftDetector(
                    window_size=template.window_size, ambiguity_threshold=template.ambiguity_threshold,
                    ood_threshold=template.ood_threshold,
                )
            detector = self._drift_detectors[digest]
            detector.record(is_ambiguous=ambiguous, is_ood=ood, confidence=confidence)
            self._drift_detector = detector
            return detector.is_drifted

    @property
    def last_promotion_report(self) -> Optional[PromotionReport]:
        """Audit report from the most recent cutover promotion evaluation."""
        return self._last_promotion_report

    def export_model(self, path: Union[str, Path]) -> bool:
        """Compiles and exports the validated System 1 model to disk."""
        cm = self.compiled_model
        if cm is None:
            return False
            
        cm.save(path)
        return True

    def _new_engine(self, schema: DecisionSchema, compiled_model: Any = None) -> SystemOneEngine:
        if compiled_model is not None:
            if compiled_model.schema.schema_digest() != schema.schema_digest():
                raise ValueError("Saved skill does not match these questions; teach a separate skill for this schema")
            compiled_model.use_cache = False
        return SystemOneEngine(
            schema, model=compiled_model, signing_key=self.signing_key, ledger=self.ledger,
            dimension=self.dimension, backend=self.backend, projector=self.projector,
            strict_mode=self.strict_mode, use_cache=False,
        )

    def _get_engine(self, questions: Mapping[str, Any]) -> SystemOneEngine:
        schema = _build_dynamic_schema(questions)
        digest = schema.schema_digest()
        with self._engine_lock:
            if digest not in self._engine_cache:
                model = self._compiled_models.get(digest) or self._compiled_model
                self._engine_cache[digest] = self._new_engine(schema, model)
            return self._engine_cache[digest]

    def _execute_local(
        self,
        state: str,
        questions: Mapping[str, Any],
        model: str = "jev-latest",
        *,
        alpha: float = 0.05,
        record_receipt: bool = True,
        **kwargs: Any,
    ) -> TypeSafeResponse:
        """Executes single-pass System 1 evaluation locally on the prompt state."""
        engine = self._get_engine(questions)
        t0 = time.perf_counter()
        result = engine.decide(_content_text(state), alpha=alpha, record_receipt=record_receipt)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        answers: Dict[str, Any] = {}
        for q_name, q_spec in questions.items():
            q_type = getattr(q_spec, "type", None)
            if q_type is None:
                q_type = getattr(q_spec, "field_type", None)
            if q_type is None and isinstance(q_spec, Mapping):
                q_type = q_spec.get("type")
            if isinstance(q_type, str):
                q_type = q_type.lower().strip()

            val = result.values.get(q_name)
            conf = float(result.confidences.get(q_name, 1.0))
            cset = result.conformal_sets.get(q_name, [])

            if q_type == "choice":
                probs = result.probabilities.get(q_name, {})
                answers[q_name] = ChoiceAnswer({
                    "type": "choice",
                    "choice": val,
                    "value": val,
                    "confidence": conf,
                    "conformal_set": cset,
                    "probabilities": probs,
                })
            elif q_type in ("noul", "boolean"):
                prob = conf if val else (1.0 - conf)
                answers[q_name] = NoulAnswer({
                    "type": "noul",
                    "noul": prob,
                    "value": bool(val),
                    "confidence": conf,
                    "conformal_set": cset,
                })
            elif q_type == "score":
                score_val = float(val) if val is not None else 0.0
                answer = {
                    "type": "score",
                    "score": score_val,
                    "value": score_val,
                    "confidence": conf,
                    "conformal_set": cset,
                }
                field = engine.schema.fields[q_name]
                if field.metadata.get("typesafe_score_levels"):
                    answer.update(_ordinal_answer(field, result.probabilities[q_name]))
                answers[q_name] = ScoreAnswer(answer)
            elif q_type in ("multi_choice", "multichoice"):
                choices_list = list(val) if val is not None else []
                answers[q_name] = MultiChoiceAnswer({
                    "type": "multi_choice",
                    "choices": choices_list,
                    "value": choices_list,
                    "confidence": conf,
                    "conformal_set": cset,
                })
            else:
                answers[q_name] = DotDict({
                    "value": val,
                    "confidence": conf,
                    "conformal_set": cset,
                })

        receipt_dict = None
        if result.receipt:
            receipt_dict = DotDict(result.receipt.to_dict())
            receipt_dict["receipt_id"] = result.receipt.decision_id
            if result.receipt.envelope and result.receipt.envelope.signature:
                receipt_dict["signature"] = result.receipt.envelope.signature

        return SystemOneResponse({
            "model": model,
            "state": state,
            "answers": answers,
            "usage": Usage({
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }),
            "latency_ms": round(latency_ms, 3),
            "local_execution": True,
            "egress_bytes": 0,
            "conformal_alpha": alpha,
            "receipt": receipt_dict,
            "is_ambiguous": result.is_ambiguous,
        })

    def _distill_and_cutover_locked(self, questions: Mapping[str, Any]) -> None:
        """Fits closed-form local weights from collected query history and flips to local."""
        from system1.compiler import SystemOneCompiler

        schema = _build_dynamic_schema(questions)
        digest = schema.schema_digest()

        # Decouple history into train (60%), calib (20%), and val (20%) partitions
        # Invariant 10: Training, calibration, and promotion validation must be strictly disjoint!
        schema_history = self._history[digest]
        def examples_from(history):
            return {
                name: [(item["state"], _teaching_target(field, item["answers"][name]))
                       for item in history if item["answers"].get(name) is not None]
                for name, field in schema.fields.items()
            }

        compiler = SystemOneCompiler(
            schema=schema, dimension=self.dimension, projector=self.projector, backend=self.backend,
            regularization=self.regularization,
        )
        try:
            partition = partition_cutover_history(schema_history)
            partition.assert_disjoint()
            validation_keys = set().union(*(_evidence_keys(row) for row in partition.val_history))
            if validation_keys & self._validation_seen_keys[digest]:
                # Wait for an entirely fresh validation block. Replaying the same
                # holdout after every answer cannot establish promotion evidence.
                return
            compiled_model = compiler.compile(
                exemplars=examples_from(partition.train_history), augment=self.augment,
                calibration_exemplars=examples_from([group[0] for group in _evidence_groups(partition.calib_history)]), calibration_split=0,
            )
        except (ValueError, AssertionError) as exc:
            # Missing classes or disjoint evidence defer promotion, not the teacher's answer.
            self._last_promotion_report = PromotionReport(
                False, 0, 0, 0.0, 0.0, 0, 0, 0.0, [str(exc)], schema_digest=digest,
            )
            self._cutover_audit_log.append({
                "event": "trojan_horse_cutover_deferred", "schema_digest": digest,
                "rejection_reasons": [str(exc)], "status": "deferred_insufficient_evidence",
                "report": self._last_promotion_report,
            })
            return
        compiled_model.metadata["typesafe_strict_mode"] = self.strict_mode
        engine = self._new_engine(schema, compiled_model)

        # Evaluate promotion eligibility strictly on held-out validation fold
        policy = self.promotion_policy or PromotionPolicy(
            min_agreement_threshold=self.min_agreement_threshold,
            false_allow_ceiling=0.0,
            require_statistical_bound=True,
            min_local_acceptance=0.8,
        )

        report = evaluate_promotion_eligibility(
            engine=engine,
            val_history=partition.val_history,
            schema=schema,
            policy=policy,
            validation_attempt=self._validation_attempts[digest] + 1,
        )
        self._validation_attempts[digest] += 1
        self._validation_seen_keys[digest].update(validation_keys)
        self._last_promotion_report = report

        if report.is_eligible:
            # Release Invariant 3: Freeze, hash, validate, and promote the EXACT evaluated candidate artifact.
            # Never recompile or recalibrate on all history post-validation.
            candidate_bytes = compiled_model.to_bytes() if hasattr(compiled_model, "to_bytes") else b""
            artifact_digest = (
                hashlib.sha256(candidate_bytes).hexdigest()
                if candidate_bytes
                else hashlib.sha256(str(id(compiled_model)).encode()).hexdigest()
            )

            def _serialize_samples(samples: Sequence[Dict[str, Any]]) -> str:
                clean = []
                for x in samples:
                    clean.append({
                        "state": str(x.get("state", "")),
                        "answers": str(x.get("answers", "")),
                    })
                return hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()

            # The exact calibration state, schema digest, and evaluated artifacts are verified
            report.artifact_digest = artifact_digest
            report.schema_digest = digest
            report.manifest = {
                "qualification_policy": {
                    "min_agreement_threshold": policy.min_agreement_threshold,
                    "min_local_acceptance": policy.min_local_acceptance,
                    "min_accepted_agreement": policy.min_accepted_agreement,
                    "statistical_confidence": policy.statistical_confidence,
                    "require_statistical_bound": policy.require_statistical_bound,
                },
                "training_set_digest": _serialize_samples(partition.train_history),
                "calibration_set_digest": _serialize_samples(partition.calib_history),
                "validation_set_digest": _serialize_samples(partition.val_history),
                "dataset_manifest": {
                    "train_digest": _serialize_samples(partition.train_history),
                    "calib_digest": _serialize_samples(partition.calib_history),
                    "val_digest": _serialize_samples(partition.val_history),
                    "train_samples": len(partition.train_history),
                    "calib_samples": len(partition.calib_history),
                    "val_samples": len(partition.val_history),
                },
                "teacher_provenance": {
                    "provider_model": schema_history[-1].get("teacher", "unknown"),
                    "total_samples": len(schema_history),
                    "timestamp": time.time(),
                },
            }

            if self.ledger is not None:
                self.ledger.record_action(
                    action="auto_cutover_promotion",
                    proposal={
                        "schema_digest": digest,
                        "artifact_digest": artifact_digest,
                        "metrics": {
                            "total_validation_checks": report.total_validation_checks,
                            "agreement_rate": report.agreement_rate,
                            "wilson_lower_bound": report.wilson_lower_bound,
                            "critical_class_count": report.critical_class_count,
                            "false_allow_rate": report.false_allow_rate,
                            "accepted_agreement_rate": report.accepted_agreement_rate,
                            "accepted_agreement_lower_bound": report.accepted_agreement_lower_bound,
                        },
                        "manifest": report.manifest,
                    },
                )

            # Promote exact evaluated candidate artifact
            self._engine_cache[digest] = engine
            self._compiled_models[digest] = compiled_model
            self._has_cutover_schemas.add(digest)
            event = {
                "event": "trojan_horse_cutover",
                "timestamp": time.time(),
                "call_count": self._call_count[digest],
                "samples_collected": len(schema_history),
                "agreement_rate": report.agreement_rate,
                "min_agreement_threshold": policy.min_agreement_threshold,
                "wilson_lower_bound": report.wilson_lower_bound,
                "false_allow_count": report.false_allow_count,
                "artifact_digest": artifact_digest,
                "candidate_artifact_digest": artifact_digest,
                "manifest": report.manifest,
                "status": "active_100_percent_local",
                "message": (
                    f"Autonomous Trojan Horse cutover completed with {report.agreement_rate*100:.1f}% agreement "
                    f"across {len(partition.val_history)} held-out validation samples (threshold: {policy.min_agreement_threshold*100:.1f}%). "
                    f"Local execution active; uncertain responses may still require review."
                ),
                "report": report,
            }
            if self.ledger is not None:
                try:
                    self.ledger.record_action(
                        action="trojan_horse_cutover",
                        proposal=event,
                    )
                except Exception:
                    pass
        else:
            self._has_cutover = False
            event = {
                "event": "trojan_horse_cutover_deferred",
                "timestamp": time.time(),
                "call_count": self._call_count[digest],
                "samples_collected": len(schema_history),
                "agreement_rate": report.agreement_rate,
                "min_agreement_threshold": policy.min_agreement_threshold,
                "wilson_lower_bound": report.wilson_lower_bound,
                "false_allow_count": report.false_allow_count,
                "rejection_reasons": report.rejection_reasons,
                "status": "deferred_insufficient_agreement",
                "message": (
                    f"Autonomous cutover deferred: {'; '.join(report.rejection_reasons)}. "
                    f"Continuing passthrough mode to gather additional exemplars."
                ),
                "report": report,
            }
        self._cutover_audit_log.append(event)

    def distill_and_cutover(self, questions: Optional[Mapping[str, Any]] = None) -> bool:
        """Manually triggers distillation and flips client to 100% local execution."""
        with self._engine_lock:
            if questions is None and self._history:
                first_digest = next(iter(self._history))
                if self._history[first_digest]:
                    questions = self._history[first_digest][0].get("questions", {})
            if not questions:
                return False
            self._distill_and_cutover_locked(questions)
            schema = _build_dynamic_schema(questions)
            return self._has_cutover or (schema.schema_digest() in self._has_cutover_schemas)

    def systemone(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Mapping[str, Any],
        model: Optional[str] = None,
        *,
        alpha: float = 0.05,
        record_receipt: bool = True,
        **kwargs: Any,
    ) -> TypeSafeResponse:
        """Evaluates prompt state matching the client's configured execution mode."""
        if self._closed:
            raise RuntimeError("Client is closed")
        if not isinstance(state, (str, dict, list)):
            raise TypeError("State must be text, a JSON object, or an array")
        encoded_state = _content_text(state)
        if not questions:
            raise ValueError("At least one question is required")
        for option in ("retry", "extra_headers", "extra_body", "response_model"):
            if kwargs.get(option) is not None:
                raise NotImplementedError(f"The local adapter does not implement {option!r}")
        model = model or self.default_model

        # Passthrough Mode: Always forward to TypeSafe AI
        if self.mode == "passthrough":
            if self.zero_egress:
                raise ZeroEgressViolationError(
                    "Incompatible configuration: mode='passthrough' requires network egress, but zero_egress=True."
                )
            fb = kwargs.get("fallback_baseline", self.fallback_baseline)
            resp, _, _ = self.call_real_api(
                state=state,
                questions=questions,
                model=model,
                timeout=self.timeout,
                fallback_baseline=fb,
                zero_egress=False,
            )
            return resp

        # Auto-Cutover Mode (The Trojan Horse)
        if self.mode == "auto_cutover":
            schema = _build_dynamic_schema(questions)
            digest = schema.schema_digest()

            with self._engine_lock:
                already_cutover = self._has_cutover or (digest in self._has_cutover_schemas)

            if not already_cutover:
                if self.baseline_handler is not None:
                    cloud_resp = TypeSafeResponse(self.baseline_handler(state, questions))
                    egress_bytes = int(cloud_resp.get("egress_bytes", 0 if self.zero_egress else 800))
                else:
                    if self.zero_egress:
                        raise ZeroEgressViolationError(
                            "Incompatible configuration: mode='auto_cutover' with zero_egress=True "
                            "requires a local baseline_handler to provide teacher exemplars without egress."
                        )
                    fb = kwargs.get("fallback_baseline", self.fallback_baseline)
                    cloud_resp, _, egress_bytes = self.call_real_api(
                        state=state,
                        questions=questions,
                        model=model,
                        timeout=self.timeout,
                        fallback_baseline=fb,
                        zero_egress=False,
                    )

                answers = {}
                for name, definition in schema.fields.items():
                    answer = cloud_resp.get("answers", {}).get(name)
                    value = answer.get("value") if isinstance(answer, Mapping) else answer
                    if value is None:
                        raise ValueError(f"Teacher response is missing field {name!r}")
                    validated = definition.validate_value(_teaching_target(definition, value))
                    answers[name] = float(value) if definition.metadata.get("typesafe_score_levels") else validated

                with self._engine_lock:
                    self._call_count[digest] += 1
                    self._teacher_sample_count[digest] += 1
                    current_count = self._call_count[digest]
                    sample_record = {
                        "state": encoded_state,
                        "teacher": ("simulated_baseline" if cloud_resp.get("baseline_fallback") else
                                    "callback" if self.baseline_handler is not None else cloud_resp.get("model", model)),
                        "questions": dict(questions),
                        **{k: kwargs[k] for k in ("group_id", "lineage_id", "request_id") if k in kwargs},
                        "answers": answers,
                        "timestamp": time.time(),
                    }
                    self._history[digest].append(sample_record)

                    if self.ledger is not None:
                        try:
                            self.ledger.record_action(
                                action="typesafe_passthrough_query",
                                proposal={
                                    "state": state,
                                    "call_count": current_count,
                                    "answers": sample_record["answers"],
                                },
                            )
                        except Exception:
                            pass

                    if current_count >= self.cutover_threshold and digest not in self._has_cutover_schemas:
                        self._distill_and_cutover_locked(questions)

                    current_cutover = self._has_cutover or (digest in self._has_cutover_schemas)

                cloud_resp["auto_cutover_active"] = True
                cloud_resp["is_cutover"] = current_cutover
                cloud_resp["cutover_progress"] = f"{current_count}/{self.cutover_threshold}"
                cloud_resp["egress_bytes"] = egress_bytes
                return cloud_resp

            # Already cutover: execute locally
            resp = self._execute_local(
                state=state,
                questions=questions,
                model=model,
                alpha=alpha,
                record_receipt=record_receipt,
                **kwargs,
            )
            with self._engine_lock:
                self._call_count[digest] += 1
            resp["auto_cutover_active"] = True
            resp["is_cutover"] = True

            # Drift & Conformal Ambiguity Monitoring
            is_ambiguous = bool(resp.get("is_ambiguous", False))
            min_conf = 1.0
            for ans in getattr(resp, "answers", {}).values():
                c = getattr(ans, "confidence", None)
                if c is not None and isinstance(c, (int, float)):
                    min_conf = min(min_conf, float(c))
            is_ood = bool(min_conf < 0.20)
            drifted = self._record_drift(digest, is_ambiguous, is_ood, min_conf)

            if is_ambiguous or is_ood or drifted:
                if self.allow_cloud_fallback and not self.zero_egress:
                    fallback_resp, _, fallback_egress = self.call_real_api(
                        state=state,
                        questions=questions,
                        model=model,
                        timeout=self.timeout,
                        fallback_baseline=self.fallback_baseline,
                    )
                    fallback_resp["auto_cutover_active"] = True
                    fallback_resp["is_cutover"] = True
                    fallback_resp["fallback_routed"] = True
                    fallback_resp["drift_detected"] = drifted
                    fallback_resp["egress_bytes"] = fallback_egress
                    return fallback_resp
                else:
                    # Zero-egress offline abstention
                    resp["abstain"] = True
                    resp["verdict"] = "REQUIRE_APPROVAL"
                    resp["status"] = "ABSTAINED_AMBIGUOUS" if is_ambiguous else "ABSTAINED_DRIFT"
                    resp["drift_detected"] = drifted
                    resp["egress_bytes"] = 0
                    return resp

            resp["egress_bytes"] = 0
            return resp

        # Default Local Mode
        resp = self._execute_local(
            state=state,
            questions=questions,
            model=model,
            alpha=alpha,
            record_receipt=record_receipt,
            **kwargs,
        )
        schema = _build_dynamic_schema(questions)
        digest = schema.schema_digest()
        with self._engine_lock:
            self._call_count[digest] += 1
        is_ambiguous = bool(resp.get("is_ambiguous", False))
        min_conf = 1.0
        for ans in getattr(resp, "answers", {}).values():
            c = getattr(ans, "confidence", None)
            if c is not None and isinstance(c, (int, float)):
                min_conf = min(min_conf, float(c))
        is_ood = bool(min_conf < 0.20)
        drifted = self._record_drift(digest, is_ambiguous, is_ood, min_conf)

        if drifted:
            if self.allow_cloud_fallback and not self.zero_egress:
                fallback_resp, _, fallback_egress = self.call_real_api(
                    state=state,
                    questions=questions,
                    model=model,
                    timeout=self.timeout,
                    fallback_baseline=self.fallback_baseline,
                )
                fallback_resp["fallback_routed"] = True
                fallback_resp["drift_detected"] = True
                fallback_resp["egress_bytes"] = fallback_egress
                return fallback_resp
            else:
                resp["abstain"] = True
                resp["verdict"] = "REQUIRE_APPROVAL"
                resp["status"] = "ABSTAINED_DRIFT"
                resp["drift_detected"] = True
                resp["egress_bytes"] = 0
                return resp
        return resp

    # Backward-compatible and convenience aliases
    system_one = systemone
    decide = systemone
    evaluate = systemone

    def batch_systemone(
        self,
        states: Sequence[str],
        questions: Mapping[str, Any],
        model: Optional[str] = None,
        *,
        alpha: float = 0.05,
        record_receipt: bool = True,
        **kwargs: Any,
    ) -> List[TypeSafeResponse]:
        """Batch evaluation over multiple prompt states."""
        return [
            self.systemone(
                s,
                questions,
                model=model,
                alpha=alpha,
                record_receipt=record_receipt,
                **kwargs,
            )
            for s in states
        ]

    batch_system_one = batch_systemone

    def call_real_api(
        self,
        state: str,
        questions: Mapping[str, Any],
        model: str = "jev-latest",
        timeout: float = 15.0,
        fallback_baseline: Optional[bool] = None,
        zero_egress: Optional[bool] = None,
    ) -> Tuple[Optional[TypeSafeResponse], float, int]:
        """Calls the real TypeSafe AI API using this client's api_key and base_url."""
        fb = self.fallback_baseline if fallback_baseline is None else fallback_baseline
        ze = self.zero_egress if zero_egress is None else zero_egress
        return call_real_typesafe_api(
            state=state,
            questions=questions,
            api_key=self.api_key,
            base_url=self.base_url,
            model=model,
            timeout=timeout,
            fallback_baseline=fb,
            zero_egress=ze,
        )

    def compare(
        self,
        state: str,
        questions: Mapping[str, Any],
        model: str = "jev-latest",
        *,
        alpha: float = 0.05,
        timeout: float = 15.0,
        fallback_baseline: Optional[bool] = None,
        zero_egress: Optional[bool] = None,
    ) -> DotDict:
        """Runs a side-by-side head-to-head comparison: Local System 1 vs TypeSafe Cloud.

        Measures wall-clock latency, egress bytes, token consumption, conformal bounds,
        and Ed25519 cryptographic receipts.
        """
        effective_ze = self.zero_egress if zero_egress is None else zero_egress
        if effective_ze:
            raise ZeroEgressViolationError(
                "Cannot perform cloud comparison when zero_egress=True. "
                "Instantiate TypeSafeClient(zero_egress=False) to enable WAN comparison."
            )
        fb = self.fallback_baseline if fallback_baseline is None else fallback_baseline
        local_resp = self._execute_local(state, questions, model=model, alpha=alpha)
        cloud_resp, cloud_lat, cloud_egress = self.call_real_api(
            state, questions, model=model, timeout=timeout,
            fallback_baseline=fb, zero_egress=False,
        )
        is_live = bool(cloud_resp and not cloud_resp.get("baseline_fallback", False))
        speedup = cloud_lat / local_resp.latency_ms if is_live and local_resp.latency_ms > 0 else None
        cloud_tokens = cloud_resp.usage.total_tokens if cloud_resp and cloud_resp.get("usage") else 0
        from system1.receipt import verify_decision_witness_receipt
        receipt_verified = bool(local_resp.receipt) and verify_decision_witness_receipt(
            local_resp.receipt, public_key=self.signing_key.public_key() if self.signing_key else None,
        )

        return DotDict({
            "state": state,
            "local_response": local_resp,
            "cloud_response": cloud_resp,
            "local_latency_ms": local_resp.latency_ms,
            "cloud_latency_ms": cloud_lat,
            "speedup_factor": round(speedup, 1) if speedup is not None else None,
            "local_egress_bytes": 0,
            "cloud_egress_bytes": cloud_egress,
            "local_tokens": 0,
            "cloud_tokens": cloud_tokens,
            "local_cost_usd": 0.0,
            "cloud_cost_usd": None,  # Provider billing is not available in this response.
            "local_receipt_verified": receipt_verified,
            "cloud_receipt_verified": False,
            "has_conformal_guarantee": False,  # Coverage requires exchangeable, independent evidence.
            "is_live": is_live,
            "baseline_fallback": not is_live,
        })


class AsyncTypeSafeClient:
    """Asynchronous drop-in replacement client for TypeSafe AI (Jev)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.typesafe.ai/v1",
        *,
        mode: str = "local",
        cutover_threshold: int = 50,
        min_agreement_threshold: float = 0.8,
        signing_key: Optional[Ed25519PrivateKey] = None,
        ledger: Optional[ActionLedger] = None,
        backend: str = "auto",
        dimension: int = 384,
        regularization: float = 1.0,
        projector: Optional[Any] = None,
        timeout: float = 15.0,
        zero_egress: bool = True,
        allow_cloud_fallback: bool = False,
        promotion_policy: Optional[PromotionPolicy] = None,
        drift_detector: Optional[DriftDetector] = None,
        **kwargs: Any,
    ) -> None:
        self._sync_client = TypeSafeClient(
            api_key=api_key,
            base_url=base_url,
            mode=mode,
            cutover_threshold=cutover_threshold,
            min_agreement_threshold=min_agreement_threshold,
            signing_key=signing_key,
            ledger=ledger,
            backend=backend,
            dimension=dimension,
            regularization=regularization,
            projector=projector,
            timeout=timeout,
            zero_egress=zero_egress,
            allow_cloud_fallback=allow_cloud_fallback,
            promotion_policy=promotion_policy,
            drift_detector=drift_detector,
            **kwargs,
        )

    async def __aenter__(self) -> AsyncTypeSafeClient:
        self._sync_client.__enter__()
        return self

    async def close(self) -> None:
        self._sync_client.close()

    aclose = close

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    @property
    def api_key(self) -> str:
        return self._sync_client.api_key

    @property
    def base_url(self) -> str:
        return self._sync_client.base_url

    @property
    def mode(self) -> str:
        return self._sync_client.mode

    @property
    def zero_egress(self) -> bool:
        return self._sync_client.zero_egress

    @property
    def allow_cloud_fallback(self) -> bool:
        return self._sync_client.allow_cloud_fallback

    @property
    def promotion_policy(self) -> Optional[PromotionPolicy]:
        return self._sync_client.promotion_policy

    @property
    def drift_detector(self) -> DriftDetector:
        return self._sync_client.drift_detector

    @property
    def last_promotion_report(self) -> Optional[PromotionReport]:
        return self._sync_client.last_promotion_report

    @property
    def is_cutover(self) -> bool:
        return self._sync_client.is_cutover

    @property
    def call_count(self) -> int:
        return self._sync_client.call_count

    @property
    def total_requests(self) -> int:
        return self._sync_client.total_requests

    @property
    def teacher_sample_count(self) -> int:
        return self._sync_client.teacher_sample_count

    @property
    def cutover_audit_log(self) -> List[Dict[str, Any]]:
        return self._sync_client.cutover_audit_log

    @property
    def compiled_model(self) -> Optional[Any]:
        """Returns the most recently distilled CompiledSystemOneModel, or None."""
        return self._sync_client.compiled_model

    async def export_model(self, path: Union[str, Path]) -> bool:
        """Asynchronously exports the distilled closed-form model as a static .s1m binary."""
        return await asyncio.to_thread(self._sync_client.export_model, path)

    async def distill_and_cutover(self, questions: Optional[Mapping[str, Any]] = None) -> bool:
        """Asynchronously triggers distillation and flips client to local execution."""
        return await asyncio.to_thread(self._sync_client.distill_and_cutover, questions)

    async def systemone(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Mapping[str, Any],
        model: Optional[str] = None,
        *,
        alpha: float = 0.05,
        record_receipt: bool = True,
        **kwargs: Any,
    ) -> TypeSafeResponse:
        """Asynchronously evaluates System 1 schema in background thread pool."""
        return await asyncio.to_thread(
            self._sync_client.systemone,
            state,
            questions,
            model=model,
            alpha=alpha,
            record_receipt=record_receipt,
            **kwargs,
        )

    system_one = systemone
    decide = systemone
    evaluate = systemone

    async def batch_systemone(
        self,
        states: Sequence[str],
        questions: Mapping[str, Any],
        model: Optional[str] = None,
        *,
        alpha: float = 0.05,
        record_receipt: bool = True,
        **kwargs: Any,
    ) -> List[TypeSafeResponse]:
        return await asyncio.to_thread(
            self._sync_client.batch_systemone,
            states,
            questions,
            model=model,
            alpha=alpha,
            record_receipt=record_receipt,
            **kwargs,
        )

    batch_system_one = batch_systemone

    async def call_real_api(
        self,
        state: str,
        questions: Mapping[str, Any],
        model: str = "jev-latest",
        timeout: float = 15.0,
        fallback_baseline: Optional[bool] = None,
        zero_egress: Optional[bool] = None,
    ) -> Tuple[Optional[TypeSafeResponse], float, int]:
        return await asyncio.to_thread(
            self._sync_client.call_real_api,
            state,
            questions,
            model=model,
            timeout=timeout,
            fallback_baseline=fallback_baseline,
            zero_egress=zero_egress,
        )

    async def compare(
        self,
        state: str,
        questions: Mapping[str, Any],
        model: str = "jev-latest",
        *,
        alpha: float = 0.05,
        timeout: float = 15.0,
        fallback_baseline: Optional[bool] = None,
        zero_egress: Optional[bool] = None,
    ) -> DotDict:
        effective_ze = self.zero_egress if zero_egress is None else zero_egress
        if effective_ze:
            raise ZeroEgressViolationError(
                "Cannot perform cloud comparison when zero_egress=True. "
                "Instantiate TypeSafeClient(zero_egress=False) to enable WAN comparison."
            )
        return await asyncio.to_thread(
            self._sync_client.compare,
            state,
            questions,
            model=model,
            alpha=alpha,
            timeout=timeout,
            fallback_baseline=fallback_baseline,
            zero_egress=zero_egress,
        )


# Module-level aliases
Client = TypeSafeClient
AsyncClient = AsyncTypeSafeClient


def system_one(
    state: str,
    questions: Mapping[str, Any],
    model: str = "jev-latest",
    *,
    alpha: float = 0.05,
    record_receipt: bool = True,
    **kwargs: Any,
) -> TypeSafeResponse:
    """One-shot functional API for evaluating questions against prompt state using TypeSafeClient."""
    client = TypeSafeClient(**kwargs)
    return client.system_one(
        state=state,
        questions=questions,
        model=model,
        alpha=alpha,
        record_receipt=record_receipt,
    )


systemone = system_one
evaluate = system_one


def batch_system_one(
    states: Sequence[str],
    questions: Mapping[str, Any],
    model: str = "jev-latest",
    *,
    alpha: float = 0.05,
    record_receipt: bool = True,
    **kwargs: Any,
) -> List[TypeSafeResponse]:
    """One-shot functional API for evaluating batched prompt states using TypeSafeClient."""
    client = TypeSafeClient(**kwargs)
    return client.batch_system_one(
        states=states,
        questions=questions,
        model=model,
        alpha=alpha,
        record_receipt=record_receipt,
    )


batch_systemone = batch_system_one


def compare(
    state: str,
    questions: Mapping[str, Any],
    model: str = "jev-latest",
    *,
    alpha: float = 0.05,
    api_key: Optional[str] = None,
    timeout: float = 15.0,
    fallback_baseline: bool = False,
    zero_egress: bool = True,
    **kwargs: Any,
) -> DotDict:
    """One-shot functional API to run a side-by-side local vs cloud comparison."""
    client = TypeSafeClient(
        api_key=api_key,
        timeout=timeout,
        zero_egress=zero_egress,
        fallback_baseline=fallback_baseline,
        **kwargs,
    )
    return client.compare(
        state,
        questions,
        model=model,
        alpha=alpha,
        timeout=timeout,
        fallback_baseline=fallback_baseline,
        zero_egress=zero_egress,
    )


class _SyntheticLoader(importlib.abc.Loader):
    """Loader for in-memory patched modules supporting importlib.reload."""

    def __init__(self, module: types.ModuleType) -> None:
        self.module = module

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> Optional[types.ModuleType]:
        return self.module

    def exec_module(self, module: types.ModuleType) -> None:
        # Module attributes are already populated and preserved
        pass


class _SyntheticFinder(importlib.abc.MetaPathFinder):
    """MetaPathFinder providing ModuleSpecs for in-memory patched modules."""

    def __init__(self, modules: Mapping[str, types.ModuleType]) -> None:
        self.modules = dict(modules)

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]] = None,
        target: Optional[types.ModuleType] = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        if fullname in self.modules:
            mod = self.modules[fullname]
            loader = _SyntheticLoader(mod)
            return importlib.machinery.ModuleSpec(
                fullname,
                loader,
                is_package=hasattr(mod, "__path__"),
            )
        return None


_PATCH_LOCK = threading.RLock()
_PATCHED_MODULE_NAMES = (
    "typesafe",
    "typesafe.compat",
    "typesafe.client",
    "typesafe_sdk",
    "typesafe_sdk.compat",
    "typesafe_sdk.client",
)
_PATCH_NESTING_DEPTH = 0
_SAVED_ORIGINAL_MODULES: Dict[str, Any] = {}
_ACTIVE_FINDER: Optional[_SyntheticFinder] = None


class _Unpatcher:
    """Context manager and callable to unpatch typesafe and typesafe_sdk modules."""

    def __init__(
        self,
        original_modules: Optional[Dict[str, Any]] = None,
        finder: Optional[_SyntheticFinder] = None,
        mode: str = "local",
    ) -> None:
        self.original_modules = original_modules if original_modules is not None else _SAVED_ORIGINAL_MODULES
        self.finder = finder if finder is not None else _ACTIVE_FINDER
        self.mode = mode
        self._active = True

    def __enter__(self) -> _Unpatcher:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.unpatch()

    def __call__(self, func: Optional[Callable[..., Any]] = None) -> Any:
        if func is None:
            self.unpatch()
            return None
        self.unpatch()
        return _wrap_with_patch(func, mode=self.mode)

    def unpatch(self) -> None:
        global _PATCH_NESTING_DEPTH, _SAVED_ORIGINAL_MODULES, _ACTIVE_FINDER
        with _PATCH_LOCK:
            if not self._active:
                return
            self._active = False
            if _PATCH_NESTING_DEPTH > 0:
                _PATCH_NESTING_DEPTH -= 1
            if _PATCH_NESTING_DEPTH == 0:
                for name, mod in _SAVED_ORIGINAL_MODULES.items():
                    if mod is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = mod
                _SAVED_ORIGINAL_MODULES = {}
                if _ACTIVE_FINDER is not None and _ACTIVE_FINDER in sys.meta_path:
                    try:
                        sys.meta_path.remove(_ACTIVE_FINDER)
                    except ValueError:
                        pass
                _ACTIVE_FINDER = None


def _apply_patch(mode: str = "local") -> _Unpatcher:
    global _PATCH_NESTING_DEPTH, _SAVED_ORIGINAL_MODULES, _ACTIVE_FINDER
    with _PATCH_LOCK:
        if mode == "passthrough":
            class PassthroughTypeSafeClient(TypeSafeClient):
                def __init__(self, *args: Any, mode: str = "passthrough", **kwargs: Any) -> None:
                    kwargs.setdefault("zero_egress", False)
                    kwargs.setdefault("fallback_baseline", False)
                    super().__init__(*args, mode=mode, **kwargs)

            class AsyncPassthroughTypeSafeClient(AsyncTypeSafeClient):
                def __init__(self, *args: Any, mode: str = "passthrough", **kwargs: Any) -> None:
                    kwargs.setdefault("zero_egress", False)
                    kwargs.setdefault("fallback_baseline", False)
                    super().__init__(*args, mode=mode, **kwargs)

            client_cls: Any = PassthroughTypeSafeClient
            async_client_cls: Any = AsyncPassthroughTypeSafeClient
        else:
            client_cls = TypeSafeClient
            async_client_cls = AsyncTypeSafeClient

        if _PATCH_NESTING_DEPTH > 0:
            _PATCH_NESTING_DEPTH += 1
            if mode == "passthrough":
                for m_name in ("typesafe", "typesafe_sdk", "typesafe.client", "typesafe_sdk.client"):
                    m = sys.modules.get(m_name)
                    if m is not None and getattr(m, "_is_system1_patched", False):
                        setattr(m, "Client", client_cls)
                        setattr(m, "TypeSafeClient", client_cls)
                        setattr(m, "AsyncClient", async_client_cls)
                        setattr(m, "AsyncTypeSafeClient", async_client_cls)
            return _Unpatcher(_SAVED_ORIGINAL_MODULES, _ACTIVE_FINDER, mode=mode)

        orig_modules = {}
        for name in _PATCHED_MODULE_NAMES:
            existing = sys.modules.get(name)
            if existing is not None and getattr(existing, "_is_system1_patched", False):
                orig_modules[name] = None
            else:
                orig_modules[name] = existing
        _SAVED_ORIGINAL_MODULES = orig_modules

        mod = types.ModuleType("typesafe")
        mod.__path__ = []
        mod.__package__ = "typesafe"

        compat_mod = types.ModuleType("typesafe.compat")
        compat_mod.__package__ = "typesafe"

        client_mod = types.ModuleType("typesafe.client")
        client_mod.__package__ = "typesafe"

        sdk_mod = types.ModuleType("typesafe_sdk")
        sdk_mod.__path__ = []
        sdk_mod.__package__ = "typesafe_sdk"

        sdk_compat_mod = types.ModuleType("typesafe_sdk.compat")
        sdk_compat_mod.__package__ = "typesafe_sdk"

        sdk_client_mod = types.ModuleType("typesafe_sdk.client")
        sdk_client_mod.__package__ = "typesafe_sdk"

        exports = {
            "TypeSafeClient": client_cls,
            "Client": client_cls,
            "AsyncTypeSafeClient": async_client_cls,
            "AsyncClient": async_client_cls,
            "Choice": Choice,
            "MultiChoice": MultiChoice,
            "Noul": Noul,
            "NoulCriteria": NoulCriteria,
            "Score": Score,
            "TypeSafeResponse": TypeSafeResponse,
            "SystemOneResponse": SystemOneResponse,
            "ChoiceAnswer": ChoiceAnswer,
            "NoulAnswer": NoulAnswer,
            "ScoreAnswer": ScoreAnswer,
            "MultiChoiceAnswer": MultiChoiceAnswer,
            "Usage": Usage,
            "system_one": system_one,
            "systemone": systemone,
            "evaluate": evaluate,
            "batch_system_one": batch_system_one,
            "batch_systemone": batch_systemone,
            "patch_typesafe": patch_typesafe,
            "DotDict": DotDict,
            "call_real_typesafe_api": call_real_typesafe_api,
            "create_typesafe_baseline_response": create_typesafe_baseline_response,
            "compare": compare,
            "PromotionPolicy": PromotionPolicy,
            "PromotionReport": PromotionReport,
            "CutoverPartition": CutoverPartition,
            "evaluate_promotion_eligibility": evaluate_promotion_eligibility,
            "partition_cutover_history": partition_cutover_history,
            "compute_wilson_score_lower": compute_wilson_score_lower,
        }

        all_exports = [
            "TypeSafeClient",
            "Client",
            "AsyncTypeSafeClient",
            "AsyncClient",
            "Choice",
            "MultiChoice",
            "Noul",
            "NoulCriteria",
            "Score",
            "TypeSafeResponse",
            "SystemOneResponse",
            "ChoiceAnswer",
            "NoulAnswer",
            "ScoreAnswer",
            "MultiChoiceAnswer",
            "Usage",
            "system_one",
            "systemone",
            "evaluate",
            "batch_system_one",
            "batch_systemone",
            "patch_typesafe",
            "DotDict",
            "call_real_typesafe_api",
            "create_typesafe_baseline_response",
            "compare",
            "PromotionPolicy",
            "PromotionReport",
            "CutoverPartition",
            "evaluate_promotion_eligibility",
            "partition_cutover_history",
            "compute_wilson_score_lower",
        ]

        for target_mod in (mod, sdk_mod):
            for k, v in exports.items():
                setattr(target_mod, k, v)
            target_mod._is_system1_patched = True
            target_mod.__all__ = list(all_exports)

        compat_all = [x for x in all_exports if x != "patch_typesafe"]
        for target_compat in (compat_mod, sdk_compat_mod):
            for k, v in exports.items():
                if k != "patch_typesafe":
                    setattr(target_compat, k, v)
            target_compat._is_system1_patched = True
            target_compat.__all__ = list(compat_all)

        client_exports = {
            "TypeSafeClient": client_cls,
            "Client": client_cls,
            "AsyncTypeSafeClient": async_client_cls,
            "AsyncClient": async_client_cls,
        }
        for target_client in (client_mod, sdk_client_mod):
            for k, v in client_exports.items():
                setattr(target_client, k, v)
            target_client._is_system1_patched = True
            target_client.__all__ = list(client_exports.keys())

        mod.compat = compat_mod
        mod.client = client_mod
        sdk_mod.compat = sdk_compat_mod
        sdk_mod.client = sdk_client_mod

        patched_dict = {
            "typesafe": mod,
            "typesafe.compat": compat_mod,
            "typesafe.client": client_mod,
            "typesafe_sdk": sdk_mod,
            "typesafe_sdk.compat": sdk_compat_mod,
            "typesafe_sdk.client": sdk_client_mod,
        }

        finder = _SyntheticFinder(patched_dict)
        for target_mod in patched_dict.values():
            target_mod.__spec__ = finder.find_spec(target_mod.__name__)
            target_mod.__loader__ = target_mod.__spec__.loader if target_mod.__spec__ else None

        sys.meta_path.insert(0, finder)
        sys.modules.update(patched_dict)
        _ACTIVE_FINDER = finder
        _PATCH_NESTING_DEPTH = 1

        return _Unpatcher(orig_modules, finder, mode=mode)


def _wrap_with_patch(func: Callable[..., Any], mode: str = "local") -> Callable[..., Any]:
    """Wraps a sync or async function to apply patch_typesafe on each execution."""
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            unpatcher = _apply_patch(mode=mode)
            try:
                return await func(*args, **kwargs)
            finally:
                unpatcher.unpatch()
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            unpatcher = _apply_patch(mode=mode)
            try:
                return func(*args, **kwargs)
            finally:
                unpatcher.unpatch()
        return sync_wrapper


def patch_typesafe(func: Optional[Callable[..., Any]] = None, *, mode: str = "local") -> Any:
    """Monkey-patches the typesafe and typesafe_sdk namespaces in sys.modules.

    Allows legacy projects running:
        import typesafe
        import typesafe_sdk
        from typesafe import TypeSafeClient, Choice, Noul, Score
        from typesafe_sdk import TypeSafeClient, Choice, Noul, Score, SystemOneResponse
    to run locally on-device with zero code edits, zero egress, and sub-2ms latency.

    Supports mode='local' (default) or mode='passthrough' to route through
    the TypeSafe API client / WAN baseline fallback.

    Can be used as:
    1. Context manager:
       with patch_typesafe():
           import typesafe_sdk
       with patch_typesafe(mode="passthrough"):
           import typesafe
    2. Decorator (with or without parentheses, sync or async):
       @patch_typesafe
       def test_something(): ...

       @patch_typesafe()
       async def test_async(): ...

       @patch_typesafe(mode="passthrough")
       def test_passthrough(): ...
    3. Explicit unpatcher callable:
       unpatcher = patch_typesafe()
       ...
       unpatcher.unpatch()
    """
    if isinstance(func, str):
        mode = func
        func = None
    if func is not None and callable(func):
        return _wrap_with_patch(func, mode=mode)
    return _apply_patch(mode=mode)


__all__ = [
    "Choice",
    "MultiChoice",
    "Noul",
    "NoulCriteria",
    "Score",
    "TypeSafeResponse",
    "SystemOneResponse",
    "ChoiceAnswer",
    "NoulAnswer",
    "ScoreAnswer",
    "MultiChoiceAnswer",
    "Usage",
    "TypeSafeClient",
    "Client",
    "AsyncTypeSafeClient",
    "AsyncClient",
    "system_one",
    "systemone",
    "evaluate",
    "batch_system_one",
    "batch_systemone",
    "patch_typesafe",
    "DotDict",
    "call_real_typesafe_api",
    "create_typesafe_baseline_response",
    "compare",
    "compute_wilson_score_lower",
    "CutoverPartition",
    "DriftDetector",
    "PromotionPolicy",
    "PromotionReport",
    "evaluate_promotion_eligibility",
    "partition_cutover_history",
    "ZeroEgressViolationError",
    "get_egress_audit_log",
    "clear_egress_audit_log",
]
