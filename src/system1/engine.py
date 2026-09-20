"""System 1 Decision Engine Runtime.

Orchestrates non-autoregressive single-pass schema evaluation, calibrated
confidence scoring, conformal prediction sets, cryptographic receipt generation,
and ActionLedger audit integration.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import secrets
import statistics
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from system1.calibration import (
    CalibrationMetrics,
    ConformalPredictionSet,
    ConformalPredictor,
    DecisionCalibrator,
    RegressionConformalInterval,
    RegressionConformalPredictor,
)
from system1.core import (
    BooleanField,
    ChoiceField,
    DecisionField,
    DecisionSchema,
    MultiChoiceField,
    RawFieldEvaluation,
    ScoreField,
    SystemOneModel,
)
from system1.core.model import _stable_sigmoid
from system1.cache import SemanticSystemOneCache, _validate_cache_inputs
from system1.core.telemetry import TelemetryProjector
from system1.ledger import ActionLedger, LedgerWriteError
from system1.receipt import (
    DecisionWitnessReceipt,
    create_decision_receipt,
    public_key_fingerprint,
)


@dataclass(frozen=True)
class BenchmarkReport:
    """Performance and latency benchmark report."""

    schema_name: str
    total_decisions: int
    mean_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_decisions_per_sec: float
    target_latency_ms: float = 20.0
    baseline_label: str = "Cloud SaaS API"
    baseline_latency_ms: float = 0.0
    speedup_factor: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionResult:
    """Strongly-typed outcome of a System 1 decision evaluation."""

    schema_name: str
    schema_digest: str
    prompt: str
    values: Dict[str, Any]
    confidences: Dict[str, float]
    conformal_sets: Dict[str, List[str]]
    probabilities: Dict[str, Dict[str, float]]
    is_ambiguous: bool
    latency_ms: float
    alpha: float
    receipt: DecisionWitnessReceipt
    margins: Dict[str, float] = field(default_factory=dict)
    margin_thresholds: Dict[str, float] = field(default_factory=dict)
    margin_gate_active: Dict[str, bool] = field(default_factory=dict)
    odds_ratios: Dict[str, float] = field(default_factory=dict)
    relative_odds_ratio_thresholds: Dict[str, float] = field(default_factory=dict)
    confidence_floors: Dict[str, float] = field(default_factory=dict)
    ambiguous_fields: List[str] = field(default_factory=list)
    escalated_fields: List[str] = field(default_factory=list)
    telemetry: Optional[Any] = None
    is_cache_hit: bool = False
    embedding: Optional[np.ndarray] = None

    def __getattr__(self, item: str) -> Any:
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        values = self.__dict__.get("values")
        if values is not None and item in values:
            return values[item]
        raise AttributeError(f"'DecisionResult' object has no attribute {item!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_digest": self.schema_digest,
            "prompt": self.prompt,
            "values": self.values,
            "confidences": self.confidences,
            "conformal_sets": self.conformal_sets,
            "probabilities": self.probabilities,
            "is_ambiguous": self.is_ambiguous,
            "latency_ms": self.latency_ms,
            "alpha": self.alpha,
            "receipt": self.receipt.to_dict(),
            "margins": self.margins,
            "margin_thresholds": self.margin_thresholds,
            "margin_gate_active": self.margin_gate_active,
            "odds_ratios": self.odds_ratios,
            "relative_odds_ratio_thresholds": self.relative_odds_ratio_thresholds,
            "confidence_floors": self.confidence_floors,
            "ambiguous_fields": self.ambiguous_fields,
            "escalated_fields": self.escalated_fields,
            "telemetry": self.telemetry,
            "is_cache_hit": self.is_cache_hit,
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class SystemOneEngine:
    """Local structured decisions with calibration, caching and optional audit evidence.

    Use strict_mode=True for taught skills and honor the returned review flags.
    Prediction-set coverage needs exchangeable held-out evidence; it is not
    accepted-answer accuracy or tool authorization. Performance is workload-dependent."""

    def __init__(
        self,
        schema: Union[DecisionSchema, Type[DecisionSchema]],
        *,
        model: Optional[Any] = None,
        dimension: int = 384,
        backend: str = "auto",
        signing_key: Optional[Ed25519PrivateKey] = None,
        ledger: Optional[ActionLedger] = None,
        projector: Optional[Any] = None,
        contrastive_whitening: bool = True,
        use_cache: bool = True,
        cache_threshold: float = 0.98,
        cache_capacity: int = 2048,
        enable_margin_gating: bool = True,
        margin_threshold: Optional[float] = None,
        forgetting_factor: float = 0.995,
        relative_odds_ratio: Optional[float] = 1.5,
        confidence_floor_tau0: Optional[float] = 0.15,
        recency_weighted: bool = False,
        fail_closed_ledger: bool = False,
        strict_mode: bool = False,
        policy_scope: str = "",
    ) -> None:
        if isinstance(schema, type) and issubclass(schema, DecisionSchema):
            self.schema: DecisionSchema = schema()
        elif isinstance(schema, DecisionSchema):
            self.schema = schema
        else:
            raise TypeError(f"Expected DecisionSchema subclass or instance, got {type(schema).__name__}")

        self.dimension = dimension
        self.backend = backend
        self.signing_key: Optional[Ed25519PrivateKey] = signing_key
        self.ledger: Optional[ActionLedger] = ledger
        self.fail_closed_ledger = bool(fail_closed_ledger)
        self.projector = projector
        self.contrastive_whitening = contrastive_whitening
        self.use_cache = bool(use_cache)
        self.cache = SemanticSystemOneCache(
            capacity=cache_capacity,
            similarity_threshold=cache_threshold,
        )
        self.enable_margin_gating = bool(enable_margin_gating)
        self.margin_threshold = margin_threshold
        self.forgetting_factor = float(forgetting_factor)
        self.relative_odds_ratio = (
            float(relative_odds_ratio) if relative_odds_ratio is not None else None
        )
        self.confidence_floor_tau0 = (
            float(confidence_floor_tau0) if confidence_floor_tau0 is not None else None
        )
        self.recency_weighted = bool(recency_weighted)
        self.strict_mode = bool(strict_mode)
        self.policy_scope = str(policy_scope)
        self.policy_epoch: int = 0
        self.model_version: int = 1
        self._lock = threading.RLock()

        # Non-autoregressive decision model
        if model is not None:
            self.model = model
            self.dimension = getattr(model, "dimension", dimension)
            self.backend = getattr(model, "backend", backend)
            self.projector = getattr(model, "projector", projector)
            self.recency_weighted = getattr(model, "recency_weighted", self.recency_weighted)
        else:
            self.model = SystemOneModel(
                self.schema,
                dimension=self.dimension,
                backend=self.backend,
                projector=self.projector,
                contrastive_whitening=self.contrastive_whitening,
                forgetting_factor=self.forgetting_factor,
                recency_weighted=self.recency_weighted,
            )
            self.dimension = self.model.dimension
            self.backend = self.model.backend
            self.projector = self.model.projector

        # Calibrators and Conformal Predictors per field
        self.calibrators: Dict[str, DecisionCalibrator] = {
            name: DecisionCalibrator(temperature=1.0)
            for name in self.schema.fields.keys()
        }
        self.conformal_predictors: Dict[str, ConformalPredictor] = {}
        self.regression_conformal_predictors: Dict[str, RegressionConformalPredictor] = {}
        for name, f in self.schema.fields.items():
            if isinstance(f, ChoiceField):
                self.conformal_predictors[name] = ConformalPredictor(
                    name,
                    f.options,
                    margin_threshold=self.margin_threshold,
                    relative_odds_ratio=self.relative_odds_ratio,
                    confidence_floor_tau0=self.confidence_floor_tau0,
                )
            elif isinstance(f, BooleanField):
                self.conformal_predictors[name] = ConformalPredictor(
                    name,
                    ("False", "True"),
                    margin_threshold=self.margin_threshold,
                    relative_odds_ratio=self.relative_odds_ratio,
                    confidence_floor_tau0=self.confidence_floor_tau0,
                )
            elif isinstance(f, MultiChoiceField):
                self.conformal_predictors[name] = ConformalPredictor(
                    name,
                    f.options,
                    margin_threshold=self.margin_threshold,
                    relative_odds_ratio=self.relative_odds_ratio,
                    confidence_floor_tau0=self.confidence_floor_tau0,
                )
            elif isinstance(f, ScoreField):
                self.regression_conformal_predictors[name] = RegressionConformalPredictor(
                    name, min_value=f.min_value, max_value=f.max_value
                )

        # Synchronize calibrated temperatures and quantiles from compiled model if provided
        if model is not None and hasattr(model, "heads"):
            for name, ch in model.heads.items():
                if name in self.calibrators:
                    self.calibrators[name].temperature = getattr(ch, "temperature", 1.0)
                if name in self.conformal_predictors:
                    self.conformal_predictors[name].score_method = getattr(ch, "score_method", "aps")
                    calib_scores = getattr(ch, "calibration_scores", ())
                    if len(calib_scores) > 0:
                        self.conformal_predictors[name].calibration_scores = np.sort(
                            np.asarray(calib_scores, dtype=np.float64)
                        )
                        self.conformal_predictors[name].is_calibrated = True
                    q = getattr(ch, "conformal_quantile", 0.0)
                    if q > 0:
                        self.conformal_predictors[name].quantile = q
                        self.conformal_predictors[name].is_calibrated = True
                    m = getattr(ch, "margin_threshold", None)
                    if m is not None and self.margin_threshold is None:
                        self.conformal_predictors[name].calibrated_margin_threshold = float(m)
                    ror = getattr(ch, "relative_odds_ratio", None)
                    if ror is not None and self.relative_odds_ratio is None:
                        self.conformal_predictors[name].relative_odds_ratio = float(ror)
                    cft = getattr(ch, "confidence_floor_tau0", None)
                    if cft is not None and self.confidence_floor_tau0 is None:
                        self.conformal_predictors[name].confidence_floor_tau0 = float(cft)
                if name in self.regression_conformal_predictors:
                    calib_scores = getattr(ch, "calibration_scores", ())
                    if len(calib_scores) > 0:
                        self.regression_conformal_predictors[name].residuals = np.sort(
                            np.asarray(calib_scores, dtype=np.float64)
                        )
                        self.regression_conformal_predictors[name].is_calibrated = True

    def _model_digest(self) -> str:
        """Computes deterministic digest of active model weights."""
        if hasattr(self.model, "model_digest") and callable(self.model.model_digest):
            return self.model.model_digest()
        heads = getattr(self.model, "heads", getattr(self.model, "_field_heads", {}))
        if not heads:
            return f"v{self.model_version}"
        h = hashlib.sha256(f"v:{self.model_version}".encode("utf-8"))
        for fname in sorted(heads.keys()):
            hd = heads[fname]
            w = getattr(hd, "weights", None)
            b = getattr(hd, "biases", None)
            if w is not None:
                h.update(np.asarray(w, dtype=np.float32).tobytes())
            if b is not None:
                h.update(np.asarray(b, dtype=np.float32).tobytes())
        return h.hexdigest()

    def _projector_digest(self) -> str:
        """Computes deterministic digest of neural projector."""
        if hasattr(self.projector, "projector_digest") and callable(self.projector.projector_digest):
            return self.projector.projector_digest()
        from system1.compiler import _projector_config
        p_str = json.dumps(_projector_config(self.projector), sort_keys=True)
        return hashlib.sha256(p_str.encode("utf-8")).hexdigest()

    def _calibration_digest(self) -> str:
        """Computes deterministic digest of calibration parameters and conformal nonconformity scores."""
        h = hashlib.sha256()
        for fname in sorted(self.schema.fields.keys()):
            calib = self.calibrators.get(fname)
            if calib is not None:
                h.update(f"{fname}:temp:{calib.temperature:.6f}".encode("utf-8"))
            cp = self.conformal_predictors.get(fname)
            if cp is not None and getattr(cp, "is_calibrated", False):
                scores = getattr(cp, "calibration_scores", np.array([]))
                h.update(f"{fname}:{cp.score_method}:cp_scores:{scores.tobytes()}".encode("utf-8"))
            rcp = self.regression_conformal_predictors.get(fname)
            if rcp is not None and getattr(rcp, "is_calibrated", False):
                res = getattr(rcp, "residuals", np.array([]))
                h.update(f"{fname}:rcp_res:{res.tobytes()}".encode("utf-8"))
        return h.hexdigest()

    def encode(
        self,
        prompt: str,
        telemetry: Optional[Any] = None,
        recency_weighted: Optional[bool] = None,
    ) -> np.ndarray:
        """Encodes prompt into semantic embedding vector."""
        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
        if hasattr(self.model, "encode"):
            try:
                return self.model.encode(prompt, telemetry=telemetry, recency_weighted=use_recency)
            except TypeError:
                return self.model.encode(prompt, telemetry=telemetry)
        if self.projector is not None:
            try:
                emb = self.projector.project(prompt, recency_weighted=use_recency)
            except TypeError:
                emb = self.projector.project(prompt)
            norm = float(np.linalg.norm(emb))
            return (emb / norm).astype(np.float32) if norm > 1e-12 else emb
        v = np.zeros(self.dimension, dtype=np.float32)
        v[0] = 1.0
        return v

    def calibrate(
        self,
        dataset: Sequence[Tuple[str, Mapping[str, Any]]],
        *,
        n_bins: int = 10,
    ) -> Dict[str, CalibrationMetrics]:
        """Calibrate on held-out examples and preserve the evidence in compiled skills.

        Repeated normalized prompts are one observation. Temperature fitting and
        conformal calibration use disjoint prompts, including for multi-label and
        continuous outputs. Callers must keep these examples out of teaching.
        """
        if not dataset:
            raise ValueError("Calibration dataset must contain at least one example")
        grouped = {}
        for prompt, labels in dataset:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Calibration prompts must be non-empty strings")
            key = " ".join(prompt.casefold().split())
            validated = {}
            for name, value in labels.items():
                if name not in self.schema.fields:
                    raise ValueError(f"Unknown calibration field: {name!r}")
                validated[name] = self.schema.fields[name].validate_value(value)
            if key in grouped and grouped[key][1] != validated:
                raise ValueError("Conflicting labels for repeated calibration prompt")
            grouped[key] = (prompt, validated)

        with self._lock, getattr(self.model, "_lock", nullcontext()):
            rows = list(grouped.values())
            results = self.model.forward_batch([row[0] for row in rows])
            metrics = {}
            for name, definition in self.schema.fields.items():
                indices = [i for i, row in enumerate(rows) if name in row[1]]
                if not indices:
                    continue
                rng = np.random.RandomState(42)
                rng.shuffle(indices)
                count = len(indices) // 2
                logits = np.asarray([results[i].fields[name].logits for i in indices], dtype=np.float64)
                labels = [rows[i][1][name] for i in indices]
                calibrator = DecisionCalibrator()
                if isinstance(definition, ChoiceField):
                    targets = np.asarray([definition.options.index(label) for label in labels])
                elif isinstance(definition, BooleanField):
                    targets = np.asarray(labels, dtype=np.int64)
                elif isinstance(definition, MultiChoiceField):
                    targets = np.asarray([[int(opt in label) for opt in definition.options] for label in labels])
                else:
                    midpoint = (definition.min_value + definition.max_value) / 2
                    targets = np.asarray([int(label >= midpoint) for label in labels])

                temp_logits = logits[:count]
                temp_targets = targets[:count]
                if isinstance(definition, MultiChoiceField):
                    temp_logits, temp_targets = temp_logits.flatten(), temp_targets.flatten()
                elif isinstance(definition, ScoreField) or (isinstance(definition, BooleanField) and logits.shape[1] == 1):
                    temp_logits = temp_logits.flatten()
                metrics[name] = calibrator.fit(temp_logits, temp_targets, n_bins=n_bins)
                # Runtime always passes one head's vector, rather than a batch of
                # binary scalars. Temperature is the complete inference parameter.
                calibrator.is_binary = False
                self.calibrators[name] = calibrator
                conf_logits = logits[count:]
                conf_labels = labels[count:]

                if isinstance(definition, (ChoiceField, BooleanField)):
                    if isinstance(definition, BooleanField) and conf_logits.shape[1] == 1:
                        z = conf_logits.reshape(-1)
                        conf_logits = np.column_stack([-z / 2, z / 2])
                    probs = calibrator.calibrate_logits(conf_logits)
                    predictor = self.conformal_predictors[name]
                    predictor.calibrate(probs, [str(label) for label in conf_labels])
                    scores = predictor.calibration_scores
                elif isinstance(definition, MultiChoiceField):
                    probs = _stable_sigmoid(conf_logits / calibrator.temperature, temperature=1.0)
                    errors = np.where(targets[count:] == 1, 1.0 - probs, probs)
                    predictor = self.conformal_predictors[name]
                    predictor.calibration_scores = np.sort(np.max(errors, axis=1))
                    predictor.is_calibrated = True
                    scores = predictor.calibration_scores
                else:
                    probs = _stable_sigmoid(conf_logits.reshape(-1) / calibrator.temperature, temperature=1.0)
                    predictions = definition.min_value + (definition.max_value - definition.min_value) * probs
                    predictor = self.regression_conformal_predictors[name]
                    predictor.calibrate(predictions, conf_labels)
                    scores = predictor.residuals

                compiled = getattr(self.model, "heads", {}).get(name)
                if compiled is not None and hasattr(compiled, "calibration_scores"):
                    compiled.temperature = calibrator.temperature
                    compiled.calibration_scores = tuple(float(score) for score in scores)
                    compiled.score_method = getattr(predictor, "score_method", "aps")
                    k = int(math.ceil((len(scores) + 1) * 0.95))
                    upper = definition.max_value - definition.min_value if isinstance(definition, ScoreField) else 1.0
                    compiled.conformal_quantile = float(scores[k - 1]) if k <= len(scores) else upper
            if self.cache is not None:
                self.cache.clear()
            return metrics

    def decide(
        self,
        prompt: str,
        *,
        telemetry: Optional[Any] = None,
        embedding: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        record_receipt: bool = True,
        ledger: Optional[ActionLedger] = None,
        margin_threshold: Optional[float] = None,
        relative_odds_ratio: Optional[float] = None,
        confidence_floor_tau0: Optional[float] = None,
        recency_weighted: Optional[bool] = None,
        fail_closed_ledger: Optional[bool] = None,
        strict: Optional[bool] = None,
        policy_scope: Optional[str] = None,
    ) -> DecisionResult:
        """Evaluates prompt against schema in a single non-autoregressive pass."""
        # A decision, its uncertainty, and its receipt use one model snapshot.
        with self._lock, getattr(self.model, "_lock", nullcontext()):
            version = getattr(self.model, "model_version", self.model_version)
            if version != self.model_version:
                self.model_version = version
                for name in self.schema.fields:
                    head = getattr(self.model, "heads", {}).get(name)
                    if not getattr(head, "calibration_scores", ()):
                        self.calibrators[name] = DecisionCalibrator()
                        if name in self.conformal_predictors:
                            self.conformal_predictors[name].is_calibrated = False
                        if name in self.regression_conformal_predictors:
                            self.regression_conformal_predictors[name].is_calibrated = False
            if not isinstance(prompt, str):
                raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")
            if not (0.0 < alpha < 1.0) or math.isnan(alpha) or math.isinf(alpha):
                raise ValueError(f"alpha must be in (0, 1), got {alpha}")
            if margin_threshold is not None and (margin_threshold < 0.0 or math.isnan(margin_threshold) or math.isinf(margin_threshold)):
                raise ValueError(f"margin_threshold must be non-negative, got {margin_threshold}")
            if relative_odds_ratio is not None and (relative_odds_ratio <= 0.0 or math.isnan(relative_odds_ratio) or math.isinf(relative_odds_ratio)):
                raise ValueError(f"relative_odds_ratio must be strictly positive, got {relative_odds_ratio}")
            if confidence_floor_tau0 is not None and (confidence_floor_tau0 < 0.0 or math.isnan(confidence_floor_tau0) or math.isinf(confidence_floor_tau0)):
                raise ValueError(f"confidence_floor_tau0 must be non-negative, got {confidence_floor_tau0}")
            if embedding is not None:
                if not isinstance(embedding, (np.ndarray, list, tuple)):
                    raise TypeError(f"embedding must be array-like, got {type(embedding).__name__}")
                arr_emb = np.asarray(embedding, dtype=np.float32).flatten()
                if not np.all(np.isfinite(arr_emb)):
                    raise ValueError("embedding must contain only finite numbers (no NaN or Inf)")
                if arr_emb.shape[0] != self.dimension:
                    raise ValueError(f"embedding dimension mismatch: expected {self.dimension}, got {arr_emb.shape[0]}")
                query_emb = arr_emb
            else:
                query_emb = None
            if telemetry is not None:
                if not isinstance(telemetry, (dict, list, tuple, np.ndarray, float, int)):
                    raise TypeError(f"telemetry must be a dict, sequence, ndarray, or numeric, got {type(telemetry).__name__}")

            t_start = time.perf_counter()

            is_fail_closed = self.fail_closed_ledger if fail_closed_ledger is None else bool(fail_closed_ledger)
            is_strict = self.strict_mode if strict is None else bool(strict)
            eff_scope = self.policy_scope if policy_scope is None else str(policy_scope)
            eff_m_thresh = margin_threshold if margin_threshold is not None else self.margin_threshold
            eff_gamma = relative_odds_ratio if relative_odds_ratio is not None else self.relative_odds_ratio
            eff_tau0 = confidence_floor_tau0 if confidence_floor_tau0 is not None else self.confidence_floor_tau0

            active_ledger = ledger or self.ledger
            truth_ledger_head = ""
            ledger_record_id: Optional[str] = None

            if active_ledger is not None:
                try:
                    truth_ledger_head = active_ledger.head_hash()
                except Exception as ex:
                    if is_fail_closed:
                        raise LedgerWriteError(f"Fail-closed ledger inspection failed: {ex}") from ex
                    truth_ledger_head = ""

            # Precompute snapshot digests for context binding
            m_dig = self._model_digest()
            p_dig = self._projector_digest()
            c_dig = self._calibration_digest()

            # 1. Cache lookup with request and model context
            if self.use_cache:
                cached = self.cache.get(
                    prompt,
                    embedding=embedding,
                    telemetry=telemetry,
                    schema_digest=self.schema.schema_digest(),
                    model_version=self.model_version,
                    policy_scope=eff_scope,
                    alpha=alpha,
                    margin_threshold=eff_m_thresh if self.enable_margin_gating else 0.0,
                    strict=is_strict,
                    relative_odds_ratio=eff_gamma,
                    confidence_floor_tau0=eff_tau0,
                    recency_weighted=recency_weighted,
                    model_digest=m_dig,
                    projector_digest=p_dig,
                    calibration_digest=c_dig,
                    policy_epoch=self.policy_epoch,
                    enforce_durability=is_fail_closed,
                )
                if cached is None:
                    if query_emb is None:
                        query_emb = self.encode(prompt, telemetry=telemetry, recency_weighted=recency_weighted)
                    cached = self.cache.get(
                        prompt,
                        embedding=query_emb,
                        telemetry=telemetry,
                        schema_digest=self.schema.schema_digest(),
                        model_version=self.model_version,
                        policy_scope=eff_scope,
                        alpha=alpha,
                        margin_threshold=eff_m_thresh if self.enable_margin_gating else 0.0,
                        strict=is_strict,
                        relative_odds_ratio=eff_gamma,
                        confidence_floor_tau0=eff_tau0,
                        recency_weighted=recency_weighted,
                        model_digest=m_dig,
                        projector_digest=p_dig,
                        calibration_digest=c_dig,
                        policy_epoch=self.policy_epoch,
                        enforce_durability=is_fail_closed,
                    )

                if cached is not None:
                    entry, sim = cached
                    if isinstance(entry.result, DecisionResult):
                        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                        hit_is_ambiguous = bool(getattr(entry.result, "is_ambiguous", False))
                        hit_ambiguous_fields = copy.deepcopy(getattr(entry.result, "ambiguous_fields", []))
                        hit_escalated_fields = copy.deepcopy(getattr(entry.result, "escalated_fields", []))

                        hit_receipt = None
                        if record_receipt:
                            hit_receipt = create_decision_receipt(
                                schema_name=self.schema.schema_name,
                                schema_digest=self.schema.schema_digest(),
                                prompt=prompt,
                                values=copy.deepcopy(entry.result.values),
                                confidences=copy.deepcopy(entry.result.confidences),
                                conformal_sets=copy.deepcopy(entry.result.conformal_sets),
                                probabilities=copy.deepcopy(entry.result.probabilities),
                                latency_ms=elapsed_ms,
                                is_ambiguous=hit_is_ambiguous,
                                truth_ledger_head=truth_ledger_head,
                                ledger_record_id=None,
                                signing_key=self.signing_key,
                            )
                            if active_ledger is not None:
                                try:
                                    ledger_record_id = active_ledger.record_decision_receipt(hit_receipt)
                                    hit_receipt = hit_receipt.with_ledger_record(ledger_record_id, self.signing_key)
                                except Exception as ex:
                                    if is_fail_closed:
                                        raise LedgerWriteError(f"Fail-closed ledger recording failed on cache hit: {ex}") from ex
                                    pass

                        return DecisionResult(
                            schema_name=self.schema.schema_name,
                            schema_digest=self.schema.schema_digest(),
                            prompt=prompt,
                            values=copy.deepcopy(entry.result.values),
                            confidences=copy.deepcopy(entry.result.confidences),
                            conformal_sets=copy.deepcopy(entry.result.conformal_sets),
                            probabilities=copy.deepcopy(entry.result.probabilities),
                            is_ambiguous=hit_is_ambiguous,
                            latency_ms=elapsed_ms,
                            alpha=alpha,
                            receipt=hit_receipt,
                            margins=copy.deepcopy(entry.result.margins),
                            margin_thresholds=copy.deepcopy(entry.result.margin_thresholds),
                            margin_gate_active=copy.deepcopy(entry.result.margin_gate_active),
                            odds_ratios=copy.deepcopy(getattr(entry.result, "odds_ratios", {})),
                            relative_odds_ratio_thresholds=copy.deepcopy(getattr(entry.result, "relative_odds_ratio_thresholds", {})),
                            confidence_floors=copy.deepcopy(getattr(entry.result, "confidence_floors", {})),
                            ambiguous_fields=hit_ambiguous_fields,
                            escalated_fields=hit_escalated_fields,
                            telemetry=copy.deepcopy(telemetry) if telemetry is not None else None,
                            is_cache_hit=True,
                            embedding=entry.embedding if entry.embedding is not None else query_emb,
                        )

            with self._lock:
                try:
                    raw_result = self.model.forward_single(
                        prompt,
                        telemetry=telemetry,
                        embedding=query_emb,
                        recency_weighted=recency_weighted,
                    )
                except TypeError:
                    raw_result = self.model.forward_single(
                        prompt,
                        telemetry=telemetry,
                        embedding=query_emb,
                    )

            values: Dict[str, Any] = {}
            confidences: Dict[str, float] = {}
            conformal_sets: Dict[str, List[str]] = {}
            probabilities: Dict[str, Dict[str, float]] = {}
            margins: Dict[str, float] = {}
            margin_thresholds: Dict[str, float] = {}
            margin_gate_active: Dict[str, bool] = {}
            odds_ratios: Dict[str, float] = {}
            relative_odds_ratio_thresholds: Dict[str, float] = {}
            confidence_floors: Dict[str, float] = {}
            ambiguous_fields: List[str] = []
            escalated_fields: List[str] = []
            is_ambiguous = False

            for name, f_def in self.schema.fields.items():
                raw_eval = raw_result.fields[name]
                calibrator = self.calibrators[name]

                if isinstance(f_def, ChoiceField):
                    calibrated_p = calibrator.calibrate_logits(raw_eval.logits)
                    best_idx = int(np.argmax(calibrated_p))
                    selected_choice = f_def.options[best_idx]
                    conf = float(calibrated_p[best_idx])

                    prob_dict = {opt: float(calibrated_p[i]) for i, opt in enumerate(f_def.options)}
                    values[name] = selected_choice
                    confidences[name] = conf
                    probabilities[name] = prob_dict

                    conformal = self.conformal_predictors.get(name)
                    if conformal is not None:
                        try:
                            cset = conformal.predict_set(
                                calibrated_p,
                                alpha=alpha,
                                margin_threshold=eff_m_thresh if (self.enable_margin_gating and not is_strict) else 0.0,
                                relative_odds_ratio=eff_gamma if not is_strict else None,
                                confidence_floor_tau0=eff_tau0 if not is_strict else None,
                                strict=is_strict,
                            )
                        except TypeError:
                            cset = conformal.predict_set(calibrated_p, alpha=alpha)
                        conformal_sets[name] = list(cset.prediction_set)
                        margins[name] = getattr(cset, "margin", 1.0)
                        margin_thresholds[name] = getattr(cset, "margin_threshold", 0.0)
                        margin_gate_active[name] = getattr(cset, "margin_gate_active", False)
                        odds_ratios[name] = getattr(cset, "odds_ratio", 1.0)
                        relative_odds_ratio_thresholds[name] = getattr(cset, "relative_odds_ratio_threshold", 0.0)
                        confidence_floors[name] = getattr(cset, "confidence_floor", 0.0)

                        is_empty_set = getattr(cset, "is_empty", len(cset.prediction_set) == 0)
                        cardinality = len(cset.prediction_set)
                        if is_strict:
                            field_ambiguous = (cardinality != 1) or cset.is_ambiguous or is_empty_set
                        else:
                            field_ambiguous = is_empty_set or cset.is_ambiguous

                        if field_ambiguous:
                            ambiguous_fields.append(name)
                            if getattr(f_def, "escalate_on_ambiguity", True):
                                is_ambiguous = True
                                escalated_fields.append(name)
                    else:
                        conformal_sets[name] = [selected_choice]
                        margins[name] = 1.0

                elif isinstance(f_def, BooleanField):
                    calibrated_p = calibrator.calibrate_logits(raw_eval.logits)
                    prob_true = float(calibrated_p[1])
                    val = bool(prob_true >= f_def.threshold)
                    conf = prob_true if val else (1.0 - prob_true)

                    values[name] = val
                    confidences[name] = conf
                    probabilities[name] = {"False": float(calibrated_p[0]), "True": prob_true}

                    conformal = self.conformal_predictors.get(name)
                    if conformal is not None:
                        try:
                            cset = conformal.predict_set(
                                calibrated_p,
                                alpha=alpha,
                                margin_threshold=eff_m_thresh if (self.enable_margin_gating and not is_strict) else 0.0,
                                relative_odds_ratio=eff_gamma if not is_strict else None,
                                confidence_floor_tau0=eff_tau0 if not is_strict else None,
                                strict=is_strict,
                            )
                        except TypeError:
                            cset = conformal.predict_set(calibrated_p, alpha=alpha)
                        conformal_sets[name] = list(cset.prediction_set)
                        margins[name] = getattr(cset, "margin", 1.0)
                        margin_thresholds[name] = getattr(cset, "margin_threshold", 0.0)
                        margin_gate_active[name] = getattr(cset, "margin_gate_active", False)
                        odds_ratios[name] = getattr(cset, "odds_ratio", 1.0)
                        relative_odds_ratio_thresholds[name] = getattr(cset, "relative_odds_ratio_threshold", 0.0)
                        confidence_floors[name] = getattr(cset, "confidence_floor", 0.0)

                        is_empty_set = getattr(cset, "is_empty", len(cset.prediction_set) == 0)
                        cardinality = len(cset.prediction_set)
                        if is_strict:
                            field_ambiguous = (cardinality != 1) or cset.is_ambiguous or is_empty_set
                        else:
                            field_ambiguous = is_empty_set or cset.is_ambiguous

                        if field_ambiguous:
                            ambiguous_fields.append(name)
                            if getattr(f_def, "escalate_on_ambiguity", True):
                                is_ambiguous = True
                                escalated_fields.append(name)
                    else:
                        conformal_sets[name] = ["True" if val else "False"]
                        margins[name] = 1.0

                elif isinstance(f_def, MultiChoiceField):
                    calibrated_temp = calibrator.temperature
                    scaled_logits = raw_eval.logits / max(1e-4, calibrated_temp)
                    probs = _stable_sigmoid(scaled_logits, temperature=1.0)
                    selected = tuple(
                        f_def.options[i] for i, p in enumerate(probs) if p >= f_def.threshold
                    )
                    conf = float(np.mean([max(p, 1.0 - p) for p in probs])) if len(probs) > 0 else 1.0
                    values[name] = selected
                    confidences[name] = conf
                    probabilities[name] = {opt: float(probs[i]) for i, opt in enumerate(f_def.options)}
                    conformal_sets[name] = list(selected)

                    eff_boundary_margin = eff_m_thresh if (self.enable_margin_gating and eff_m_thresh is not None) else 0.05
                    is_boundary_uncertain = any(
                        abs(p - f_def.threshold) < eff_boundary_margin
                        for p in probs
                    )
                    mc_ambiguous = False
                    if is_strict:
                        conformal = self.conformal_predictors[name]
                        scores = conformal.calibration_scores
                        k = int(math.ceil((len(scores) + 1) * (1.0 - alpha)))
                        if not conformal.is_calibrated or k > len(scores):
                            conformal_sets[name] = list(f_def.options)
                            mc_ambiguous = True
                        else:
                            # Compiler scores bound the largest error over all labels.
                            # Each label has a set of possible binary assignments. Only
                            # a unique complete assignment can be accepted, including {}.
                            q = float(scores[k - 1])
                            can_include = (1.0 - probs) <= q + 1e-12
                            can_exclude = probs <= q + 1e-12
                            conformal_sets[name] = [opt for opt, possible in zip(f_def.options, can_include) if possible]
                            unique = np.logical_xor(can_include, can_exclude)
                            consistent = can_include == np.array([opt in selected for opt in f_def.options])
                            mc_ambiguous = not bool(np.all(unique & consistent))

                    if mc_ambiguous:
                        ambiguous_fields.append(name)
                        if getattr(f_def, "escalate_on_ambiguity", True):
                            is_ambiguous = True
                            escalated_fields.append(name)

                elif isinstance(f_def, ScoreField):
                    calibrator = self.calibrators.get(name)
                    calibrated_temp = calibrator.temperature if calibrator is not None else 1.0
                    scaled_logit = float(raw_eval.logits[0] / max(1e-4, calibrated_temp))
                    prob = float(_stable_sigmoid(np.array([scaled_logit]), temperature=1.0)[0])
                    val = f_def.min_value + (f_def.max_value - f_def.min_value) * prob
                    conf = float(getattr(raw_eval, "confidence", max(prob, 1.0 - prob)))
                    values[name] = val
                    confidences[name] = conf
                    probabilities[name] = {"score_ratio": prob}
                    reg_conformal = self.regression_conformal_predictors.get(name)
                    is_uncalibrated = reg_conformal is None or not reg_conformal.is_calibrated
                    if not is_uncalibrated:
                        interval = reg_conformal.predict_interval(val, alpha=alpha, strict=is_strict)
                        conformal_sets[name] = [interval.to_interval_string()]
                        interval_margin = interval.margin
                    else:
                        if is_strict:
                            interval_margin = f_def.max_value - f_def.min_value
                            low_b = f_def.min_value
                            high_b = f_def.max_value
                        else:
                            interval_margin = (f_def.max_value - f_def.min_value) * ((1.0 - alpha) / 2.0)
                            low_b = max(f_def.min_value, val - interval_margin)
                            high_b = min(f_def.max_value, val + interval_margin)
                        conformal_sets[name] = [f"[{low_b:.4f}, {high_b:.4f}]"]

                    score_range = max(1e-6, f_def.max_value - f_def.min_value)
                    normalized_margin = interval_margin / score_range
                    score_ambiguous = False
                    if is_strict:
                        eff_score_thresh = eff_m_thresh if (self.enable_margin_gating and eff_m_thresh is not None) else 0.5
                        score_ambiguous = is_uncalibrated or (normalized_margin > eff_score_thresh)

                    if score_ambiguous:
                        ambiguous_fields.append(name)
                        if getattr(f_def, "escalate_on_ambiguity", True):
                            is_ambiguous = True
                            escalated_fields.append(name)

            # Unknown vocabulary carries no evidence. Do not accept a class prior
            # as a taught answer when a fixed-vocabulary skill sees no features.
            from system1.core.text import TfidfProjector
            if isinstance(self.projector, TfidfProjector) and not np.any(raw_result.embedding):
                for name, definition in self.schema.fields.items():
                    conformal_sets[name] = []
                    if name not in ambiguous_fields:
                        ambiguous_fields.append(name)
                    if getattr(definition, "escalate_on_ambiguity", True):
                        is_ambiguous = True
                        if name not in escalated_fields:
                            escalated_fields.append(name)

            validated_values = self.schema.validate_decision(values)
            total_latency_ms = (time.perf_counter() - t_start) * 1000.0

            safe_values = copy.deepcopy(validated_values)
            safe_confidences = copy.deepcopy(confidences)
            safe_conformal_sets = copy.deepcopy(conformal_sets)
            safe_probabilities = copy.deepcopy(probabilities)
            safe_margins = copy.deepcopy(margins)
            safe_margin_thresholds = copy.deepcopy(margin_thresholds)
            safe_margin_gate_active = copy.deepcopy(margin_gate_active)
            safe_odds_ratios = copy.deepcopy(odds_ratios)
            safe_relative_odds_ratio_thresholds = copy.deepcopy(relative_odds_ratio_thresholds)
            safe_confidence_floors = copy.deepcopy(confidence_floors)
            safe_ambiguous_fields = copy.deepcopy(ambiguous_fields)
            safe_escalated_fields = copy.deepcopy(escalated_fields)

            # ActionLedger integration
            active_ledger = ledger or self.ledger
            truth_ledger_head = ""
            ledger_record_id: Optional[str] = None

            if active_ledger is not None:
                try:
                    truth_ledger_head = active_ledger.head_hash()
                except Exception as ex:
                    if is_fail_closed:
                        raise LedgerWriteError(f"Fail-closed ledger inspection failed: {ex}") from ex
                    truth_ledger_head = ""

            # Emit proof-carrying cryptographic receipt
            receipt = create_decision_receipt(
                schema_name=self.schema.schema_name,
                schema_digest=self.schema.schema_digest(),
                prompt=prompt,
                values=safe_values,
                confidences=safe_confidences,
                conformal_sets=safe_conformal_sets,
                probabilities=safe_probabilities,
                latency_ms=total_latency_ms,
                is_ambiguous=is_ambiguous,
                truth_ledger_head=truth_ledger_head,
                ledger_record_id=None,
                signing_key=self.signing_key,
            )

            # Commit receipt into ActionLedger audit log
            if active_ledger is not None and record_receipt:
                try:
                    ledger_record_id = active_ledger.record_decision_receipt(receipt)
                    receipt = receipt.with_ledger_record(ledger_record_id, self.signing_key)
                except Exception as ex:
                    if is_fail_closed:
                        raise LedgerWriteError(f"Fail-closed ledger recording failed: {ex}") from ex
                    pass

            result = DecisionResult(
                schema_name=self.schema.schema_name,
                schema_digest=self.schema.schema_digest(),
                prompt=prompt,
                values=safe_values,
                confidences=safe_confidences,
                conformal_sets=safe_conformal_sets,
                probabilities=safe_probabilities,
                is_ambiguous=is_ambiguous,
                latency_ms=total_latency_ms,
                alpha=alpha,
                receipt=receipt,
                margins=safe_margins,
                margin_thresholds=safe_margin_thresholds,
                margin_gate_active=safe_margin_gate_active,
                odds_ratios=safe_odds_ratios,
                relative_odds_ratio_thresholds=safe_relative_odds_ratio_thresholds,
                confidence_floors=safe_confidence_floors,
                ambiguous_fields=safe_ambiguous_fields,
                escalated_fields=safe_escalated_fields,
                telemetry=telemetry,
                is_cache_hit=False,
                embedding=raw_result.embedding,
            )

            if self.use_cache and not is_ambiguous:
                self.cache.put(
                    prompt,
                    result,
                    embedding=raw_result.embedding,
                    telemetry=telemetry,
                    schema_digest=self.schema.schema_digest(),
                    model_version=self.model_version,
                    policy_scope=eff_scope,
                    alpha=alpha,
                    margin_threshold=eff_m_thresh if self.enable_margin_gating else 0.0,
                    strict=is_strict,
                    relative_odds_ratio=eff_gamma,
                    confidence_floor_tau0=eff_tau0,
                    recency_weighted=recency_weighted,
                    model_digest=m_dig,
                    projector_digest=p_dig,
                    calibration_digest=c_dig,
                    policy_epoch=self.policy_epoch,
                    explicit_embedding=(embedding is not None),
                    source="evaluation",
                )

            return result

    def evaluate(
        self,
        prompt: str,
        telemetry: Optional[Any] = None,
        embedding: Optional[np.ndarray] = None,
        *,
        alpha: float = 0.05,
        record_receipt: bool = True,
        ledger: Optional[ActionLedger] = None,
        margin_threshold: Optional[float] = None,
        relative_odds_ratio: Optional[float] = None,
        confidence_floor_tau0: Optional[float] = None,
        recency_weighted: Optional[bool] = None,
        strict: Optional[bool] = None,
        policy_scope: Optional[str] = None,
    ) -> DecisionResult:
        """Evaluates prompt against schema with optional continuous numeric telemetry vector and precomputed embedding."""
        return self.decide(
            prompt,
            telemetry=telemetry,
            embedding=embedding,
            alpha=alpha,
            record_receipt=record_receipt,
            ledger=ledger,
            margin_threshold=margin_threshold,
            relative_odds_ratio=relative_odds_ratio,
            confidence_floor_tau0=confidence_floor_tau0,
            recency_weighted=recency_weighted,
            strict=strict,
            policy_scope=policy_scope,
        )

    def learn_from_tier2(
        self,
        prompt: str,
        target: Union[Mapping[str, Any], Any],
        telemetry: Optional[Any] = None,
        embedding: Optional[np.ndarray] = None,
        forgetting_factor: Optional[float] = None,
        recency_weighted: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Apply an online correction using the model's supported update path.

        Adapts the decision hyperplanes of SystemOneEngine for resolved Tier 2 edge cases,
        and updates the local cache. Changed calibration must be refreshed.
        """
        with self._lock, getattr(self.model, "_lock", nullcontext()):
            eff_forgetting = forgetting_factor if forgetting_factor is not None else self.forgetting_factor
            t0 = time.perf_counter()

            if embedding is not None:
                emb = np.asarray(embedding, dtype=np.float32).flatten()
            else:
                emb = self.encode(prompt, telemetry=telemetry, recency_weighted=recency_weighted)
            x_aug = np.append(emb, 1.0).astype(np.float32)

            if hasattr(self.model, "learn_from_tier2"):
                try:
                    res_dict = self.model.learn_from_tier2(
                        prompt,
                        target,
                        telemetry=telemetry,
                        embedding=emb,
                        forgetting_factor=eff_forgetting,
                        recency_weighted=recency_weighted,
                    )
                except TypeError:
                    try:
                        res_dict = self.model.learn_from_tier2(
                            prompt, target, telemetry=telemetry, embedding=emb, forgetting_factor=eff_forgetting
                        )
                    except TypeError:
                        res_dict = self.model.learn_from_tier2(
                            prompt, target, telemetry=telemetry, embedding=emb
                        )
                updated_fields = res_dict.get("updated_fields", [])
                rank1_ms = res_dict.get("update_latency_ms", 0.0)
            else:
                target_dict: Dict[str, Any]
                if isinstance(target, Mapping):
                    target_dict = dict(target)
                else:
                    first_field = next(iter(self.schema.fields.keys()))
                    target_dict = {first_field: target}

                updated_fields = []
                update_durations_ms: List[float] = []
                heads_dict = getattr(self.model, "_field_heads", getattr(self.model, "heads", {}))
                for f_name, target_val in target_dict.items():
                    if f_name in heads_dict:
                        head = heads_dict[f_name]
                        if hasattr(head, "format_target_vector") and hasattr(head, "online_update"):
                            y_target = head.format_target_vector(target_val)
                            try:
                                dt = head.online_update(x_aug, y_target, forgetting_factor=eff_forgetting)
                            except TypeError:
                                dt = head.online_update(x_aug, y_target)
                            update_durations_ms.append(dt)
                            updated_fields.append(f_name)

                rank1_ms = float(sum(update_durations_ms)) if update_durations_ms else 0.0

            # Atomic version increment and cache invalidation AFTER weights are updated
            self.model_version += 1
            if hasattr(self.model, "model_version"):
                self.model.model_version = self.model_version

            for f_name in updated_fields:
                if f_name in self.calibrators:
                    self.calibrators[f_name] = DecisionCalibrator()
                if f_name in self.conformal_predictors:
                    self.conformal_predictors[f_name].is_calibrated = False
                if f_name in self.regression_conformal_predictors:
                    self.regression_conformal_predictors[f_name].is_calibrated = False

            if self.use_cache and self.cache is not None:
                self.cache.evict_prompt(prompt)
                self.cache.invalidate_prior_versions(self.model_version)

            if hasattr(self.model, "cache") and self.model.cache is not None:
                if hasattr(self.model.cache, "evict_prompt"):
                    self.model.cache.evict_prompt(prompt)
                if hasattr(self.model.cache, "invalidate_prior_versions"):
                    self.model.cache.invalidate_prior_versions(self.model_version)

            # Re-evaluate with updated weights
            res = self.decide(
                prompt,
                telemetry=telemetry,
                embedding=emb,
                record_receipt=False,
                recency_weighted=recency_weighted,
                strict=self.strict_mode,
                policy_scope=self.policy_scope,
            )

            # Store the corrected result in the cache
            if self.use_cache:
                eff_m_thresh = self.margin_threshold
                m_dig = self._model_digest()
                p_dig = self._projector_digest()
                c_dig = self._calibration_digest()
                self.cache.put(
                    prompt,
                    res,
                    embedding=emb,
                    telemetry=telemetry,
                    schema_digest=self.schema.schema_digest(),
                    model_version=self.model_version,
                    policy_scope=self.policy_scope,
                    alpha=0.05,
                    margin_threshold=eff_m_thresh if self.enable_margin_gating else 0.0,
                    strict=self.strict_mode,
                    relative_odds_ratio=self.relative_odds_ratio,
                    confidence_floor_tau0=self.confidence_floor_tau0,
                    recency_weighted=recency_weighted,
                    source="tier2",
                    model_digest=m_dig,
                    projector_digest=p_dig,
                    calibration_digest=c_dig,
                    policy_epoch=self.policy_epoch,
                    explicit_embedding=(emb is not None),
                )

            total_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "status": "updated",
                "update_latency_ms": round(rank1_ms, 4),
                "total_latency_ms": round(total_ms, 4),
                "updated_fields": updated_fields,
            }

    # Cognitive dual-process and execution pipeline aliases
    learn_from_system2 = learn_from_tier2
    learn_from_tier3 = learn_from_tier2

    def decide_batch(
        self,
        prompts: Sequence[str],
        *,
        telemetry: Optional[Sequence[Any]] = None,
        alpha: float = 0.05,
        record_receipt: bool = False,
        margin_threshold: Optional[float] = None,
        relative_odds_ratio: Optional[float] = None,
        confidence_floor_tau0: Optional[float] = None,
        recency_weighted: Optional[bool] = None,
        strict: Optional[bool] = None,
        policy_scope: Optional[str] = None,
    ) -> List[DecisionResult]:
        """Processes multiple inputs in batch with optional continuous telemetry."""
        if telemetry is not None and len(telemetry) == len(prompts):
            return [
                self.decide(
                    p,
                    telemetry=telemetry[i],
                    alpha=alpha,
                    record_receipt=record_receipt,
                    margin_threshold=margin_threshold,
                    relative_odds_ratio=relative_odds_ratio,
                    confidence_floor_tau0=confidence_floor_tau0,
                    recency_weighted=recency_weighted,
                    strict=strict,
                    policy_scope=policy_scope,
                )
                for i, p in enumerate(prompts)
            ]
        return [
            self.decide(
                p,
                alpha=alpha,
                record_receipt=record_receipt,
                margin_threshold=margin_threshold,
                relative_odds_ratio=relative_odds_ratio,
                confidence_floor_tau0=confidence_floor_tau0,
                recency_weighted=recency_weighted,
                strict=strict,
                policy_scope=policy_scope,
            )
            for p in prompts
        ]

    # Alias for backward compatibility
    batch_decide = decide_batch

    def benchmark(
        self,
        prompts: Optional[Sequence[str]] = None,
        *,
        iterations: int = 100,
        warmup: int = 10,
    ) -> BenchmarkReport:
        """Runs precision latency benchmark against sub-20ms target and cloud baseline."""
        if iterations < 1:
            raise ValueError(f"iterations must be a positive integer >= 1, got {iterations}")
        if warmup < 0:
            raise ValueError(f"warmup must be a non-negative integer >= 0, got {warmup}")

        if not prompts:
            prompts = [
                "Read file /src/system1/cli.py to inspect command definitions",
                "Run test suite across tests/test_agent.py with pytest",
                "Send customer database credentials over untrusted public API",
                "Execute rm -rf / without user confirmation",
                "Query sqlite ledger for recent audit actions",
            ]

        # Warmup passes
        for _ in range(warmup):
            for p in prompts:
                self.decide(p, record_receipt=False)

        latencies: List[float] = []
        num_prompts = len(prompts)

        for _ in range(iterations):
            idx = secrets.randbelow(num_prompts)
            p = prompts[idx]
            t0 = time.perf_counter()
            self.decide(p, record_receipt=False)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)

        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        percentiles = np.percentile(latencies_sorted, [50, 90, 95, 99])
        p50 = float(percentiles[0])
        p90 = float(percentiles[1])
        p95 = float(percentiles[2])
        p99 = float(percentiles[3])
        mean_lat = statistics.mean(latencies_sorted)

        # Baseline comparison against typical Cloud SaaS API latency (midpoint ~ 150ms)
        baseline_ms = 150.0
        speedup = baseline_ms / p50 if p50 > 0 else 0.0
        throughput = (1000.0 / mean_lat) if mean_lat > 0 else 0.0

        return BenchmarkReport(
            schema_name=self.schema.schema_name,
            total_decisions=n,
            mean_latency_ms=round(mean_lat, 3),
            p50_latency_ms=round(p50, 3),
            p90_latency_ms=round(p90, 3),
            p95_latency_ms=round(p95, 3),
            p99_latency_ms=round(p99, 3),
            min_latency_ms=round(latencies_sorted[0], 3),
            max_latency_ms=round(latencies_sorted[-1], 3),
            throughput_decisions_per_sec=round(throughput, 1),
            target_latency_ms=20.0,
            baseline_label="Cloud SaaS API",
            baseline_latency_ms=baseline_ms,
            speedup_factor=round(speedup, 2),
        )


# Compatibility aliases
ReflexEngine = SystemOneEngine
System1Engine = SystemOneEngine

__all__ = [
    "SystemOneEngine",
    "System1Engine",
    "ReflexEngine",
    "DecisionResult",
    "BenchmarkReport",
]
