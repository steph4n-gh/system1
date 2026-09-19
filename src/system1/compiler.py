"""Teach compact local decision skills from labeled examples using NumPy ridge
regression, held-out uncertainty calibration, and portable .s1m serialization.

Schema-derived synthetic examples remain available for bootstrapping demos.
"""

from __future__ import annotations

import io
import json
import math
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np

from system1.core.model import (
    DecisionFieldHead,
    DeterministicSemanticProjector,
    ModelInferenceResult,
    RawFieldEvaluation,
    SystemOneModel,
    _stable_sigmoid,
    _stable_softmax,
)
from system1.core.schema import (
    BooleanField,
    ChoiceField,
    DecisionField,
    DecisionSchema,
    MultiChoiceField,
    ScoreField,
)
from system1.calibration import ConformalPredictor, DecisionCalibrator
from system1.core.telemetry import TelemetryProjector
from system1.cache import SemanticSystemOneCache

MAGIC_HEADER = b"S1M\x02"


def _projector_config(projector: Any) -> Dict[str, Any]:
    """Persist the settings that define the feature space, not just its width."""
    from system1.core.embeddings import HybridProjector

    if type(projector) is DeterministicSemanticProjector:
        return {"type": "deterministic", "dimension": projector.dimension,
                "recency_weighted": projector.recency_weighted}
    if type(projector) is HybridProjector:
        return {"type": "hybrid", "sparse_dim": projector.sparse_dim,
                "dense_dim": projector.dense_dim, "alpha": projector.alpha,
                "seed": projector.seed}
    config = {"type": "external", "class": f"{type(projector).__module__}.{type(projector).__qualname__}",
              "dimension": getattr(projector, "dimension", None)}
    if callable(getattr(projector, "projector_digest", None)):
        config["digest"] = projector.projector_digest()
    return config


@dataclass
class CompiledHeadWeights:
    """Calibrated weight and bias matrices for a single schema field."""

    field_name: str
    field_type: str
    weights: np.ndarray  # shape: (K, D) or (1, D)
    biases: np.ndarray   # shape: (K,) or (1,)
    temperature: float = 1.0
    conformal_quantile: float = 0.0
    options: Tuple[str, ...] = ()
    P: Optional[np.ndarray] = None  # Inverted regularized covariance matrix (D+1, D+1)
    B: Optional[np.ndarray] = None  # Cross-covariance matrix (D+1, K)
    margin_threshold: float = 0.20
    forgetting_factor: float = 0.995
    relative_odds_ratio: float = 1.5
    confidence_floor_tau0: float = 0.15
    escalate_on_ambiguity: bool = True
    calibration_scores: Tuple[float, ...] = ()
    score_method: str = "aps"


class CompiledSystemOneModel:
    """Static compiled System 1 decision model loaded from a .s1m binary.

    Executes single-pass non-autoregressive forward evaluation using pre-distilled
    optimal decision hyperplanes and calibrated conformal non-conformity bounds.
    """

    def __init__(
        self,
        schema: DecisionSchema,
        heads: Dict[str, CompiledHeadWeights],
        dimension: int = 384,
        projector: Optional[Any] = None,
        backend: str = "numpy",
        metadata: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
        cache_threshold: float = 0.98,
        cache_capacity: int = 2048,
        forgetting_factor: float = 0.995,
        recency_weighted: bool = False,
    ) -> None:
        self.schema = schema
        self.heads = heads
        self.dimension = dimension
        self.backend = backend
        self.metadata = metadata or {}
        self.created_at = float(self.metadata.get("created_at", 0.0))
        self.forgetting_factor = float(forgetting_factor)
        self.recency_weighted = bool(recency_weighted)
        self.projector = (
            projector
            if projector is not None
            else DeterministicSemanticProjector(
                dimension=self.dimension,
                recency_weighted=self.recency_weighted,
            )
        )
        self.use_cache = bool(use_cache)
        self.cache = SemanticSystemOneCache(
            capacity=cache_capacity,
            similarity_threshold=cache_threshold,
        )
        self.model_version: int = 1
        self.telemetry_projector = TelemetryProjector(embedding_dim=self.dimension)
        self._lock = threading.RLock()

        # Build live heads for evaluation
        self._field_heads: Dict[str, DecisionFieldHead] = {}
        for name, f_def in self.schema.fields.items():
            head = DecisionFieldHead(
                f_def,
                dimension=self.dimension,
                backend=self.backend,
                projector=self.projector,
                forgetting_factor=self.forgetting_factor,
            )
            if name in self.heads:
                ch = self.heads[name]
                head.set_weights(ch.weights, ch.biases)
                head.init_covariance(P=ch.P, B=ch.B)
            else:
                head.init_covariance()
            self._field_heads[name] = head

    def encode(
        self,
        prompt: str,
        telemetry: Optional[Any] = None,
        recency_weighted: Optional[bool] = None,
    ) -> np.ndarray:
        """Projects input prompt into normalized dense vector, fusing continuous telemetry if present."""
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")
        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
        if hasattr(self.projector, "project"):
            try:
                emb = self.projector.project(prompt, recency_weighted=use_recency)
            except TypeError:
                emb = self.projector.project(prompt)
        else:
            emb = self.projector(prompt)
        norm = float(np.linalg.norm(emb))
        if norm > 1e-12:
            emb = emb / norm
        else:
            emb = np.zeros(self.dimension, dtype=np.float32)
            emb[0] = 1.0
        if telemetry is not None:
            emb = self.telemetry_projector.fuse(emb, telemetry)
        return emb

    def forward_single(
        self,
        prompt: str,
        telemetry: Optional[Any] = None,
        embedding: Optional[np.ndarray] = None,
        recency_weighted: Optional[bool] = None,
    ) -> ModelInferenceResult:
        """Executes a forward evaluation over the schema."""
        t0 = time.perf_counter()
        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
        if embedding is not None:
            emb = np.asarray(embedding, dtype=np.float32).flatten()
        else:
            emb = self.encode(prompt, telemetry=telemetry, recency_weighted=use_recency)

        # Check Tier 0 Semantic System 1 Cache
        cur_version = getattr(self, "model_version", 1)
        schema_dig = self.schema.schema_digest()
        if self.use_cache:
            hit = self.cache.get(
                prompt,
                embedding=emb,
                telemetry=telemetry,
                schema_digest=schema_dig,
                model_version=cur_version, strict=True,
            )
            if hit is not None:
                entry, sim = hit
                if isinstance(entry.result, ModelInferenceResult):
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    return ModelInferenceResult(
                        prompt=prompt,
                        embedding=emb,
                        fields=entry.result.fields,
                        inference_latency_ms=elapsed_ms,
                    )

        evals: Dict[str, RawFieldEvaluation] = {}
        for name, head in self._field_heads.items():
            evals[name] = head.forward(emb)
        latency = (time.perf_counter() - t0) * 1000.0
        result = ModelInferenceResult(
            prompt=prompt,
            embedding=emb,
            fields=evals,
            inference_latency_ms=latency,
        )

        if self.use_cache:
            self.cache.put(
                prompt,
                result,
                embedding=emb,
                telemetry=telemetry,
                schema_digest=schema_dig,
                model_version=cur_version, strict=True,
                source="evaluation",
            )

        return result

    def forward_batch(
        self,
        prompts: Sequence[str],
        telemetry: Optional[Sequence[Any]] = None,
        recency_weighted: Optional[bool] = None,
    ) -> List[ModelInferenceResult]:
        """Batched forward evaluation over multiple inputs."""
        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
        if telemetry is not None and len(telemetry) == len(prompts):
            return [self.forward_single(p, telemetry[i], recency_weighted=use_recency) for i, p in enumerate(prompts)]
        return [self.forward_single(p, recency_weighted=use_recency) for p in prompts]

    def evaluate(
        self,
        prompt: str,
        telemetry: Optional[Any] = None,
        embedding: Optional[np.ndarray] = None,
        recency_weighted: Optional[bool] = None,
    ) -> ModelInferenceResult:
        """Alias for forward_single with optional continuous telemetry vector and precomputed embedding."""
        return self.forward_single(
            prompt,
            telemetry=telemetry,
            embedding=embedding,
            recency_weighted=recency_weighted,
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
        """Closed-form rank-1 Sherman-Morrison online update on the metal (<0.1ms).

        Permanently adapts decision hyperplanes for resolved edge cases and updates the
        Tier 0 Semantic System 1 Cache for sub-0.05ms certified execution on repeat edge cases.
        """
        with self._lock:
            t0 = time.perf_counter()

            # Bump model_version and evict prompt before certified evaluation
            self.model_version = getattr(self, "model_version", 1) + 1
            if self.use_cache and self.cache is not None:
                if hasattr(self.cache, "evict_prompt"):
                    self.cache.evict_prompt(prompt)
                if hasattr(self.cache, "invalidate_prior_versions"):
                    self.cache.invalidate_prior_versions(self.model_version)

            if embedding is not None:
                emb = np.asarray(embedding, dtype=np.float32).flatten()
            else:
                emb = self.encode(prompt, telemetry=telemetry, recency_weighted=recency_weighted)
            x_aug = np.append(emb, 1.0).astype(np.float32)

            target_dict: Dict[str, Any]
            if isinstance(target, Mapping):
                target_dict = dict(target)
            else:
                first_field = next(iter(self.schema.fields.keys()))
                target_dict = {first_field: target}

            updated_fields: List[str] = []
            update_durations_ms: List[float] = []

            for f_name, target_val in target_dict.items():
                if f_name in self._field_heads:
                    head = self._field_heads[f_name]
                    y_target = head.format_target_vector(target_val)
                    dt = head.online_update(x_aug, y_target, forgetting_factor=forgetting_factor)
                    update_durations_ms.append(dt)

                    # Sync back to compiled head weights
                    if f_name in self.heads:
                        self.heads[f_name].weights = head.weights
                        self.heads[f_name].biases = head.biases
                        self.heads[f_name].P = head.covariance_inv
                        self.heads[f_name].B = head.cross_covariance
                        # Old evidence describes the previous weights, including on disk.
                        self.heads[f_name].calibration_scores = ()
                        self.heads[f_name].conformal_quantile = 0.0
                        self.heads[f_name].temperature = 1.0
                    updated_fields.append(f_name)

            rank1_ms = float(sum(update_durations_ms)) if update_durations_ms else 0.0
            total_ms = (time.perf_counter() - t0) * 1000.0

            # Evaluate model to get fresh result and store in Tier 0 cache
            evals: Dict[str, RawFieldEvaluation] = {}
            for name, head in self._field_heads.items():
                evals[name] = head.forward(emb)
            inference_res = ModelInferenceResult(
                prompt=prompt,
                embedding=emb,
                fields=evals,
                inference_latency_ms=rank1_ms,
            )
            if self.use_cache:
                self.cache.put(
                    prompt,
                    inference_res,
                    embedding=emb,
                    telemetry=telemetry,
                    schema_digest=self.schema.schema_digest(),
                    model_version=self.model_version, strict=True,
                    source="tier2",
                )

            return {
                "status": "updated",
                "update_latency_ms": round(rank1_ms, 4),
                "total_latency_ms": round(total_ms, 4),
                "updated_fields": updated_fields,
            }

    # Cognitive dual-process and execution pipeline aliases
    learn_from_system2 = learn_from_tier2
    learn_from_tier3 = learn_from_tier2

    def to_bytes(self, include_covariance: bool = False) -> bytes:
        """Serializes the compiled model into compact .s1m binary bytes.

        By default, excludes covariance matrices to keep the binary ultra-compact (<20 KB).
        Covariance is automatically initialized on-demand if online learning is invoked.
        """
        with self._lock:
            header_meta = {
                "version": 2,
                "projector": _projector_config(self.projector),
                "schema_name": self.schema.schema_name,
                "schema_dict": self.schema.to_dict(),
                "schema_digest": self.schema.schema_digest(),
                "dimension": self.dimension,
                "created_at": getattr(self, "created_at", 0.0),
                "metadata": {
                    **self.metadata,
                    "forgetting_factor": self.forgetting_factor,
                    "recency_weighted": self.recency_weighted,
                },
                "heads": {
                    name: {
                        "field_type": ch.field_type,
                        "temperature": float(ch.temperature),
                        "conformal_quantile": float(ch.conformal_quantile),
                        "score_method": ch.score_method,
                        "margin_threshold": float(getattr(ch, "margin_threshold", 0.20)),
                        "forgetting_factor": float(getattr(ch, "forgetting_factor", 0.995)),
                        "relative_odds_ratio": float(getattr(ch, "relative_odds_ratio", 1.5)),
                        "confidence_floor_tau0": float(getattr(ch, "confidence_floor_tau0", 0.15)),
                        "escalate_on_ambiguity": bool(getattr(ch, "escalate_on_ambiguity", True)),
                        "options": list(ch.options),
                        "weights_shape": list(ch.weights.shape),
                        "biases_shape": list(ch.biases.shape),
                    }
                    for name, ch in sorted(self.heads.items())
                },
            }
            header_bytes = json.dumps(header_meta, sort_keys=True).encode("utf-8")
            header_len = len(header_bytes)

            # Pack numpy weights into compressed zip buffer
            npz_buf = io.BytesIO()
            arrays_to_save: Dict[str, np.ndarray] = {}
            for name, ch in sorted(self.heads.items()):
                arrays_to_save[f"{name}_w"] = ch.weights.astype(np.float32)
                arrays_to_save[f"{name}_b"] = ch.biases.astype(np.float32)
                if hasattr(ch, "calibration_scores") and len(ch.calibration_scores) > 0:
                    arrays_to_save[f"{name}_calib_scores"] = np.asarray(ch.calibration_scores, dtype=np.float64)
                if include_covariance:
                    if ch.P is not None:
                        arrays_to_save[f"{name}_P"] = ch.P.astype(np.float32)
                    if ch.B is not None:
                        arrays_to_save[f"{name}_B"] = ch.B.astype(np.float32)
            np.savez_compressed(npz_buf, **arrays_to_save)
            npz_bytes = npz_buf.getvalue()

            # Wire format: MAGIC (4 bytes) + uint32(header_len) + header_bytes + npz_bytes
            out = bytearray()
            out.extend(MAGIC_HEADER)
            out.extend(struct.pack(">I", header_len))
            out.extend(header_bytes)
            out.extend(npz_bytes)
            return bytes(out)

    def save(self, path: Union[str, Path], include_covariance: bool = False) -> None:
        """Saves compiled model to a .s1m binary file on disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.to_bytes(include_covariance=include_covariance))

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        projector: Optional[Any] = None,
        backend: str = "numpy",
    ) -> CompiledSystemOneModel:
        """Deserializes a CompiledSystemOneModel from binary bytes."""
        if len(data) < 8 or data[:4] not in (b"S1M\x01", MAGIC_HEADER):
            raise ValueError("Invalid System 1 Model binary: missing or invalid magic header")

        header_len = struct.unpack(">I", data[4:8])[0]
        header_end = 8 + header_len
        if len(data) < header_end:
            raise ValueError("Corrupted System 1 Model binary: truncated header")

        header_json = data[8:header_end].decode("utf-8")
        meta = json.loads(header_json)
        if meta.get("version") not in (1, 2) or meta["version"] != data[3]:
            raise ValueError("Unsupported or inconsistent saved skill format version")

        # Unpack npz payload
        npz_bytes = data[header_end:]
        npz_file = np.load(io.BytesIO(npz_bytes))

        # Reconstruct schema
        schema = DecisionSchema.from_dict(meta["schema_dict"])
        if meta.get("schema_digest") != schema.schema_digest():
            raise ValueError("Saved skill schema digest does not match its definition")
        dimension = int(meta["dimension"])
        if dimension <= 0:
            raise ValueError("Saved skill dimension must be positive")
        projector_config = meta.get("projector")
        if projector_config is not None:
            if projector is not None:
                if _projector_config(projector) != projector_config:
                    raise ValueError("Supplied projector does not match the saved skill's feature space")
            elif projector_config["type"] == "deterministic":
                projector = DeterministicSemanticProjector(
                    dimension=projector_config["dimension"],
                    recency_weighted=projector_config["recency_weighted"],
                )
            elif projector_config["type"] == "hybrid":
                from system1.core.embeddings import HybridProjector
                projector = HybridProjector(**{k: v for k, v in projector_config.items() if k != "type"}, backend=backend)
            else:
                raise ValueError("This skill requires its original external projector; pass projector= when loading")

        # Reconstruct head weights
        heads: Dict[str, CompiledHeadWeights] = {}
        for name, h_info in meta["heads"].items():
            if name not in schema.fields or h_info["field_type"] != schema.fields[name].field_type:
                raise ValueError(f"Saved head {name!r} does not match the schema")
            definition = schema.fields[name]
            w = npz_file[f"{name}_w"]
            b = npz_file[f"{name}_b"]
            widths = (1, 2) if isinstance(definition, BooleanField) else (len(definition.options),) if hasattr(definition, "options") else (1,)
            if w.ndim != 2 or w.shape[0] not in widths or w.shape[1] != dimension or b.shape != (w.shape[0],):
                raise ValueError(f"Saved head {name!r} has incompatible weight dimensions")
            if not np.all(np.isfinite(w)) or not np.all(np.isfinite(b)):
                raise ValueError(f"Saved head {name!r} contains non-finite weights")
            temperature = float(h_info.get("temperature", 1.0))
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError(f"Saved head {name!r} has an invalid temperature")
            if h_info.get("score_method", "aps") not in ("aps", "lac"):
                raise ValueError(f"Saved head {name!r} uses an unsupported conformal score")
            if isinstance(definition, (ChoiceField, MultiChoiceField)) and tuple(h_info.get("options", ())) != definition.options:
                raise ValueError(f"Saved head {name!r} options do not match the schema")
            p_mat = npz_file[f"{name}_P"] if f"{name}_P" in npz_file else None
            b_mat = npz_file[f"{name}_B"] if f"{name}_B" in npz_file else None
            calib_scores_arr = npz_file[f"{name}_calib_scores"] if f"{name}_calib_scores" in npz_file else None
            calib_scores = tuple(float(s) for s in calib_scores_arr) if calib_scores_arr is not None else ()
            if any(not math.isfinite(score) or score < 0 for score in calib_scores):
                raise ValueError(f"Saved head {name!r} contains invalid calibration scores")
            heads[name] = CompiledHeadWeights(
                field_name=name,
                field_type=h_info["field_type"],
                weights=w,
                biases=b,
                temperature=float(h_info.get("temperature", 1.0)),
                conformal_quantile=float(h_info.get("conformal_quantile", 0.0)),
                score_method=h_info.get("score_method", "aps"),
                options=tuple(h_info.get("options", [])),
                P=p_mat,
                B=b_mat,
                margin_threshold=float(h_info.get("margin_threshold", 0.20)),
                forgetting_factor=float(h_info.get("forgetting_factor", 0.995)),
                relative_odds_ratio=float(h_info.get("relative_odds_ratio", 1.5)),
                confidence_floor_tau0=float(h_info.get("confidence_floor_tau0", 0.15)),
                escalate_on_ambiguity=bool(h_info.get("escalate_on_ambiguity", True)),
                calibration_scores=calib_scores,
            )

        saved_meta = meta.get("metadata", {})
        return cls(
            schema=schema,
            heads=heads,
            dimension=dimension,
            projector=projector,
            backend=backend,
            metadata=saved_meta,
            forgetting_factor=float(saved_meta.get("forgetting_factor", 0.995)),
            recency_weighted=bool(saved_meta.get("recency_weighted", False)),
        )

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        projector: Optional[Any] = None,
        backend: str = "numpy",
    ) -> CompiledSystemOneModel:
        """Loads a CompiledSystemOneModel from a .s1m file."""
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Model file not found: {target}")
        data = target.read_bytes()
        return cls.from_bytes(data, projector=projector, backend=backend)


class SystemOneCompiler:
    """The System 1 Closed-Form Distillation & Compilation Engine.

    Takes a DecisionSchema and training exemplars (or generates synthetic exemplars),
    fits closed-form Ridge Regression hyperplanes:
        W* = (X^T X + lambda I)^(-1) X^T Y
    calibrates temperature scaling and conformal prediction bounds, and serializes
    the resulting model to a compact .s1m binary.
    """

    def __init__(
        self,
        schema: Union[DecisionSchema, Type[DecisionSchema], Mapping[str, Any]],
        *,
        projector: Optional[Any] = None,
        dimension: int = 384,
        regularization: float = 1.0,
        backend: str = "numpy",
        forgetting_factor: float = 0.995,
        relative_odds_ratio: float = 1.5,
        confidence_floor_tau0: float = 0.15,
        recency_weighted: bool = False,
    ) -> None:
        if isinstance(schema, type) and issubclass(schema, DecisionSchema):
            self.schema = schema()
        elif isinstance(schema, DecisionSchema):
            self.schema = schema
        elif isinstance(schema, Mapping):
            from system1.compat.typesafe import _build_dynamic_schema
            self.schema = _build_dynamic_schema(schema)
        else:
            raise TypeError(f"Unsupported schema type: {type(schema).__name__}")

        self.dimension = dimension
        self.regularization = float(regularization)
        if not math.isfinite(self.regularization) or self.regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        self.backend = backend
        self.forgetting_factor = float(forgetting_factor)
        self.relative_odds_ratio = float(relative_odds_ratio)
        self.confidence_floor_tau0 = float(confidence_floor_tau0)
        self.recency_weighted = bool(recency_weighted)
        if projector is not None:
            self.projector = projector
            if hasattr(projector, "dimension"):
                self.dimension = projector.dimension
        else:
            self.projector = DeterministicSemanticProjector(
                dimension=self.dimension,
                recency_weighted=self.recency_weighted,
            )
        self.telemetry_projector = TelemetryProjector(embedding_dim=self.dimension)

    def generate_synthetic_exemplars(
        self,
        samples_per_choice: int = 20,
        teacher: str = "synthetic",
    ) -> Dict[str, List[Tuple[str, Any]]]:
        """Generates rich synthetic training exemplars for all schema fields."""
        exemplars: Dict[str, List[Tuple[str, Any]]] = {}

        for field_name, f_def in self.schema.fields.items():
            field_samples: List[Tuple[str, Any]] = []

            if isinstance(f_def, ChoiceField):
                templates = [
                    "User request regarding {opt}: {desc}",
                    "Please process {clean_opt} operation immediately",
                    "Handle inquiry: {desc}",
                    "Action required: {clean_opt}. Details: {desc}",
                    "Status update for {clean_opt}",
                    "Incoming event classified under {clean_opt}: {desc}",
                    "Requesting {clean_opt} assistance",
                    "Dispatch to {clean_opt} handler: {desc}",
                    "Log entry indicating {clean_opt} condition",
                    "Notification: {desc}",
                ]
                for opt in f_def.options:
                    desc_raw = f_def.descriptions.get(opt, "")
                    if isinstance(desc_raw, (list, tuple)):
                        desc = " ".join(desc_raw)
                        extra_exs = list(desc_raw)
                    else:
                        desc = str(desc_raw)
                        extra_exs = [desc]

                    clean_opt = opt.replace("_", " ").replace("-", " ")
                    # Add base variations
                    for t in templates:
                        prompt = t.format(opt=opt, clean_opt=clean_opt, desc=desc)
                        field_samples.append((prompt, opt))

                    for ex in extra_exs:
                        if ex.strip():
                            field_samples.append((ex.strip(), opt))
                            field_samples.append((f"{opt}: {ex.strip()}", opt))
                            # Add sub-phrase splits
                            for phrase in ex.replace(" and ", ", ").replace(";", ",").split(","):
                                p = phrase.strip()
                                if p and len(p) > 2:
                                    field_samples.append((p, opt))
                                    field_samples.append((f"User reporting {clean_opt}: {p}", opt))

                    # Common domain expansions for triage & support
                    domain_synonyms = {
                        "technical": [
                            "server crashed with error trace", "software bug in backend service",
                            "NullPointerException thrown by api service", "unhandled exception in runtime",
                            "application crash and downtime", "database connection failure",
                            "system crash dump", "stack trace error log",
                        ],
                        "billing": [
                            "why did you charge my credit card", "unauthorized charge on card for renewal",
                            "refund request for invoice", "dispute subscription fee",
                            "payment failed for monthly plan", "billing invoice question",
                        ],
                        "account": [
                            "forgot my password reset link", "cannot pass MFA token verification",
                            "locked out of user account login", "change profile email address",
                            "login trouble authentication failed", "two factor auth code not working",
                        ],
                    }
                    if opt in domain_synonyms:
                        for s in domain_synonyms[opt]:
                            field_samples.append((s, opt))

                    # Expand to reach requested sample count
                    idx = 0
                    while len([s for s in field_samples if s[1] == opt]) < samples_per_choice:
                        field_samples.append((
                            f"Automated test case {idx} for {clean_opt}: {desc} ({opt})",
                            opt,
                        ))
                        idx += 1

            elif isinstance(f_def, BooleanField):
                true_templates = [
                    f"Confirmed state true: {f_def.true_description}",
                    f"Affirmative condition: {f_def.true_description}",
                    f"Verified true: {f_def.description or f_def.true_description}",
                    f"Status true: {f_def.true_description}",
                    f"Criteria satisfied true: {f_def.true_description}",
                ]
                false_templates = [
                    f"Confirmed state false: {f_def.false_description}",
                    f"Negative condition: {f_def.false_description}",
                    f"Verified false: {f_def.false_description}",
                    f"Status false: {f_def.false_description}",
                    f"Criteria unsatisfied false: {f_def.false_description}",
                ]
                for p in true_templates:
                    field_samples.append((p, True))
                for p in false_templates:
                    field_samples.append((p, False))

                idx = 0
                while len(field_samples) < max(samples_per_choice * 2, 20):
                    field_samples.append((f"Synthetic positive {idx}: {f_def.true_description}", True))
                    field_samples.append((f"Synthetic negative {idx}: {f_def.false_description}", False))
                    idx += 1

            elif isinstance(f_def, MultiChoiceField):
                templates = [
                    "Apply tag {clean_opt}: {desc}",
                    "{clean_opt} condition active: {desc}",
                    "Tagged with {clean_opt}",
                    "Feature detected: {clean_opt}",
                    "Operational signal for {clean_opt}: {desc}",
                    "Notification concerning {clean_opt} status",
                    "Event flagged with {clean_opt}",
                    "Trigger for {clean_opt}: {desc}",
                ]
                for opt in f_def.options:
                    desc_raw = f_def.descriptions.get(opt, "")
                    desc = " ".join(desc_raw) if isinstance(desc_raw, (list, tuple)) else str(desc_raw)
                    clean_opt = opt.replace("_", " ").replace("-", " ")
                    for t in templates[:max(3, samples_per_choice // 2)]:
                        field_samples.append((t.format(clean_opt=clean_opt, desc=desc), (opt,)))

                # Add pairwise tag combinations if multiple options
                if len(f_def.options) >= 2:
                    for idx_a in range(len(f_def.options) - 1):
                        opt_a = f_def.options[idx_a]
                        opt_b = f_def.options[idx_a + 1]
                        field_samples.append((
                            f"Combined indicators for {opt_a} and {opt_b}",
                            (opt_a, opt_b),
                        ))

                # Add neutral / unflagged negative samples
                neutral_negatives = [
                    "Normal standard operation with no active tags",
                    "Benign routine background event",
                    "General routine request with no tags",
                    "Good morning, standard greeting and conversation",
                    "Routine heartbeat check passed",
                    "Standard documentation page lookup",
                    "Regular ping status report",
                    "System idling normally with zero alerts",
                ]
                for neg_text in neutral_negatives:
                    field_samples.append((neg_text, ()))

            elif isinstance(f_def, ScoreField):
                val_range = max(1e-6, f_def.max_value - f_def.min_value)
                low_d = f_def.low_description
                high_d = f_def.high_description
                desc_mid = f_def.description or "Intermediate balanced condition"
                score_levels = [
                    (0.0, f"Low boundary value: {low_d}"),
                    (0.0, f"Minimum rating condition: {low_d}"),
                    (0.01, f"Safe operational state: {low_d}"),
                    (0.01, f"Benign routine task: {low_d}"),
                    (0.01, f"Clean verification passed: {low_d}"),
                    (0.02, f"Read-only inspection and analysis: {low_d}"),
                    (0.02, f"Directory listing and file hierarchy inspection: {low_d}"),
                    (0.05, f"Harmless standard check: {low_d}"),
                    (0.10, f"Non-destructive operation: {low_d}"),
                    (0.45, f"Moderate rating indicator: {desc_mid}"),
                    (0.50, f"Intermediate moderate value: {desc_mid}"),
                    (0.50, f"Median balanced score: {desc_mid}"),
                    (0.55, f"Borderline operational action: {desc_mid}"),
                    (0.75, f"Elevated high indicator: {high_d}"),
                    (0.90, f"Critical security violation: {high_d}"),
                    (0.95, f"Severe vulnerability exploit: {high_d}"),
                    (0.98, f"High boundary value: {high_d}"),
                    (1.0, f"Maximum critical rating: {high_d}"),
                ]
                for fraction, prompt in score_levels:
                    target_val = f_def.min_value + fraction * val_range
                    field_samples.append((prompt, float(target_val)))

            exemplars[field_name] = field_samples

        return exemplars

    def _solve_ridge(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        regularization: float,
        sample_weights: Optional[np.ndarray] = None,
        regularize_bias: bool = True,
        return_covariance: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Solves closed-form Ridge Regression with optional sample weights and regularized bias.

        tilde{X} = [X, 1]
        A = tilde{X}^T W tilde{X} + lambda I'
        W_tilde = solve(A, tilde{X}^T W Y)
        Returns: (weights: shape (K, D), biases: shape (K,))
        Or if return_covariance=True: (weights, biases, P, B)
        """
        N, D = X.shape
        K = Y.shape[1] if Y.ndim > 1 else 1
        Y_mat = Y.reshape(N, K).astype(np.float32)

        X_aug = np.column_stack([X, np.ones((N, 1), dtype=np.float32)])
        reg_matrix = float(regularization) * np.eye(D + 1, dtype=np.float32)
        if not regularize_bias:
            reg_matrix[D, D] = 0.0
        else:
            reg_matrix[D, D] = max(1.0, float(regularization))

        if sample_weights is not None:
            sw = np.asarray(sample_weights, dtype=np.float32).flatten()
            sw_sqrt = np.sqrt(np.maximum(1e-8, sw))[:, None]
            X_aug = X_aug * sw_sqrt
            Y_mat = Y_mat * sw_sqrt

        A = X_aug.T @ X_aug + reg_matrix
        B = X_aug.T @ Y_mat

        try:
            P = np.linalg.inv(A)
            sol = P @ B
        except np.linalg.LinAlgError:
            P = np.linalg.pinv(A)
            sol, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

        weights = sol[:D, :].T  # shape: (K, D)
        biases = sol[D, :]      # shape: (K,)
        if return_covariance:
            return (
                weights.astype(np.float32),
                biases.astype(np.float32),
                P.astype(np.float32),
                B.astype(np.float32),
            )
        return weights.astype(np.float32), biases.astype(np.float32)

    @staticmethod
    def _prompt_key(prompt: str) -> str:
        return " ".join(prompt.casefold().split())

    def _validate_exemplars(self, dataset: Mapping[str, Sequence]) -> Dict[str, list]:
        if not isinstance(dataset, Mapping):
            raise ValueError("Examples must map schema field names to labeled samples")
        unknown = set(dataset) - set(self.schema.fields)
        if unknown:
            raise ValueError(f"Unknown example fields: {sorted(unknown)}")
        validated = {}
        for name, rows in dataset.items():
            field_def = self.schema.fields[name]
            validated[name] = []
            try:
                rows = iter(rows)
            except TypeError as exc:
                raise ValueError(f"Examples for {name!r} must be a sequence of labeled samples") from exc
            for index, row in enumerate(rows):
                if not isinstance(row, (list, tuple)) or len(row) not in (2, 3):
                    raise ValueError(f"{name}[{index}] must contain prompt, label, and optional telemetry")
                prompt, label = row[:2]
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"{name}[{index}] requires a nonempty text prompt")
                if label is None:
                    raise ValueError(f"{name}[{index}] requires an explicit label")
                try:
                    label = field_def.validate_value(label)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid label for {name}[{index}]: {exc}") from exc
                validated[name].append((prompt, label, *row[2:]))
        return validated

    def _split_samples(self, samples: list, field_def: Optional[DecisionField], fraction: float) -> Tuple[list, list]:
        groups: Dict[str, list] = {}
        for row in samples:
            groups.setdefault(self._prompt_key(row[0]), []).append(row)
        # Stratify discrete fields, retaining at least one training group per class.
        strata: Dict[Any, list] = {}
        for key, rows in groups.items():
            label = rows[0][1] if isinstance(field_def, (ChoiceField, BooleanField)) else None
            strata.setdefault(label, []).append(key)
        rng = np.random.RandomState(42)
        held_out = set()
        for keys in strata.values():
            keys = sorted(keys)
            rng.shuffle(keys)
            count = min(len(keys) - 1, max(1, int(len(keys) * fraction))) if fraction else 0
            held_out.update(keys[:count])
        training = [row for row in samples if self._prompt_key(row[0]) not in held_out]
        calibration = [row for row in samples if self._prompt_key(row[0]) in held_out]
        return training, calibration

    def compile(
        self,
        exemplars: Optional[Dict[str, Sequence[Tuple[str, Any]]]] = None,
        *,
        samples_per_choice: int = 20,
        teacher: str = "synthetic",
        calibration_split: float = 0.25,
        augment: bool = True,
        calibration_exemplars: Optional[Dict[str, Sequence[Tuple[str, Any]]]] = None,
    ) -> CompiledSystemOneModel:
        """Compile a local model, optionally using supplied examples only.

        Set ``augment=False`` to teach only from the examples you supply.
        Supply ``calibration_exemplars`` to control the held-out calibration set,
        or reserve a fraction of unique prompts with ``calibration_split``.
        A zero split produces an uncalibrated model; it never calibrates on training data.
        Related paraphrases/workflows should be split by the caller before compilation.
        """
        if not math.isfinite(calibration_split) or not 0.0 <= calibration_split < 1.0:
            raise ValueError("calibration_split must be finite and in [0, 1)")
        if not augment and exemplars is None:
            raise ValueError("Teaching without augmentation requires explicit exemplars")
        if calibration_exemplars is not None and exemplars is None:
            raise ValueError("Explicit calibration requires explicit training exemplars")
        if exemplars is None:
            dataset = self.generate_synthetic_exemplars(
                samples_per_choice=samples_per_choice,
                teacher=teacher,
            )
        else:
            dataset = exemplars

        dataset = self._validate_exemplars(dataset)
        calibration_data = (
            self._validate_exemplars(calibration_exemplars)
            if calibration_exemplars is not None else None
        )
        if calibration_data is not None:
            training_prompts = {self._prompt_key(row[0]) for rows in dataset.values() for row in rows}
            calibration_prompts = {self._prompt_key(row[0]) for rows in calibration_data.values() for row in rows}
            if training_prompts & calibration_prompts:
                raise ValueError("Training and calibration prompts must be disjoint")

        compiled_heads: Dict[str, CompiledHeadWeights] = {}
        sample_counts: Dict[str, Dict[str, int]] = {}

        for field_name, f_def in self.schema.fields.items():
            samples = list(dataset.get(field_name, []))
            if not samples:
                if not augment:
                    raise ValueError(f"Teaching requires examples for field {field_name!r}")
                # Fall back to default initialized head weights
                head = DecisionFieldHead(f_def, dimension=self.dimension, projector=self.projector)
                compiled_heads[field_name] = CompiledHeadWeights(
                    field_name=field_name,
                    field_type=f_def.field_type,
                    weights=head.weights,
                    biases=head.biases,
                    temperature=1.0,
                    conformal_quantile=0.0,
                    options=getattr(f_def, "options", ()),
                    forgetting_factor=self.forgetting_factor,
                    relative_odds_ratio=self.relative_odds_ratio,
                    confidence_floor_tau0=self.confidence_floor_tau0,
                    escalate_on_ambiguity=getattr(f_def, "escalate_on_ambiguity", True),
                )
                continue

            # Teaching from examples keeps repeated prompts in one partition.
            # Preserve the existing synthetic-demo recipe for compatibility.
            if calibration_data is not None:
                calibration_samples = list(calibration_data.get(field_name, []))
            elif not augment:
                samples, calibration_samples = self._split_samples(samples, f_def, calibration_split)
            else:
                calibration_samples = []
            if not augment:
                observed = {row[1] for row in samples} if isinstance(f_def, (ChoiceField, BooleanField)) else None
                required = set(f_def.options) if isinstance(f_def, ChoiceField) else {False, True}
                if observed is not None and not required.issubset(observed):
                    raise ValueError(f"Training examples for {field_name!r} must cover every class")
            real_training_count = len(samples)
            if augment:
                # Class balance and synthetic augmentation across all field types
                if isinstance(f_def, ChoiceField):
                    counts = {opt: sum(1 for s in samples if s[1] == opt) for opt in f_def.options}
                    max_c = max(counts.values()) if counts else 0
                    target_c = max(max_c, samples_per_choice, 15)
                    synth_all = self.generate_synthetic_exemplars(samples_per_choice=target_c)
                    synth_field = synth_all.get(field_name, [])

                    for opt in f_def.options:
                        curr_samples = [s for s in samples if s[1] == opt]
                        deficit = target_c - len(curr_samples)
                        if deficit > 0:
                            synth_opt = [s for s in synth_field if s[1] == opt]
                            if not synth_opt:
                                synth_opt = [(f"Operation {opt}: {f_def.descriptions.get(opt, opt)}", opt)]
                            for idx in range(deficit):
                                samples.append(synth_opt[idx % len(synth_opt)])

                elif isinstance(f_def, BooleanField):
                    true_count = sum(1 for s in samples if bool(s[1]))
                    false_count = sum(1 for s in samples if not bool(s[1]))
                    target_c = max(true_count, false_count, samples_per_choice, 15)
                    synth_all = self.generate_synthetic_exemplars(samples_per_choice=target_c)
                    synth_field = synth_all.get(field_name, [])

                    true_deficit = target_c - true_count
                    false_deficit = target_c - false_count
                    if true_deficit > 0:
                        synth_true = [s for s in synth_field if bool(s[1])]
                        for idx in range(true_deficit):
                            samples.append(synth_true[idx % len(synth_true)])
                    if false_deficit > 0:
                        synth_false = [s for s in synth_field if not bool(s[1])]
                        for idx in range(false_deficit):
                            samples.append(synth_false[idx % len(synth_false)])

                elif isinstance(f_def, MultiChoiceField):
                    synth_all = self.generate_synthetic_exemplars(samples_per_choice=15)
                    synth_field = synth_all.get(field_name, [])
                    for p, lab in synth_field:
                        samples.append((p, lab))

                elif isinstance(f_def, ScoreField):
                    val_range = max(1e-6, f_def.max_value - f_def.min_value)
                    synth_all = self.generate_synthetic_exemplars(samples_per_choice=samples_per_choice)
                    synth_field = synth_all.get(field_name, [])

                    low_synth = [s for s in synth_field if float(s[1]) <= f_def.min_value + 0.33 * val_range]
                    mid_synth = [s for s in synth_field if f_def.min_value + 0.33 * val_range < float(s[1]) <= f_def.min_value + 0.67 * val_range]
                    high_synth = [s for s in synth_field if float(s[1]) > f_def.min_value + 0.67 * val_range]

                    low_real = [s for s in samples if float(s[1]) <= f_def.min_value + 0.33 * val_range]
                    mid_real = [s for s in samples if f_def.min_value + 0.33 * val_range < float(s[1]) <= f_def.min_value + 0.67 * val_range]
                    high_real = [s for s in samples if float(s[1]) > f_def.min_value + 0.67 * val_range]

                    target_c = max(len(low_real), len(mid_real), len(high_real), samples_per_choice, 15)

                    if low_synth:
                        for idx in range(target_c - len(low_real)):
                            samples.append(low_synth[idx % len(low_synth)])
                    if mid_synth:
                        for idx in range(target_c - len(mid_real)):
                            samples.append(mid_synth[idx % len(mid_synth)])
                    if high_synth:
                        for idx in range(target_c - len(high_real)):
                            samples.append(high_synth[idx % len(high_synth)])


                # Synthetic templates must not duplicate held-out prompts either.
                held_out = {self._prompt_key(row[0]) for row in calibration_samples}
                samples = [row for row in samples if self._prompt_key(row[0]) not in held_out]
            if not samples:
                raise ValueError(f"No training examples remain for field {field_name!r}")
            generated_count = len(samples) - real_training_count
            if augment and calibration_data is None:
                rng = np.random.RandomState(42)
                samples = [samples[i] for i in rng.permutation(len(samples))]
                count = max(1, int(len(samples) * calibration_split)) if calibration_split and len(samples) >= 4 else 0
                samples, calibration_samples = samples[:len(samples) - count], samples[len(samples) - count:]
            # Temperature and conformal calibration also need disjoint prompts.
            num_temperature = 0
            if isinstance(f_def, ChoiceField) and not augment:
                temperature_samples, conformal_samples = self._split_samples(calibration_samples, None, 0.5)
                if conformal_samples:
                    num_temperature = len(temperature_samples)
                    calibration_samples = temperature_samples + conformal_samples
            elif isinstance(f_def, ChoiceField) and len(calibration_samples) >= 2:
                num_temperature = len(calibration_samples) // 2
            num_train, num_calib = len(samples), len(calibration_samples)
            sample_counts[field_name] = {
                "fit": num_train, "calibration": num_calib,
                "temperature": num_temperature,
                "generated": generated_count,
            }
            samples = samples + calibration_samples

            # Prompts and labels
            prompts = [s[0] for s in samples]
            raw_labels = [s[1] for s in samples]

            # Encode all prompts (and optional continuous telemetry) to embedding matrix X
            X_list = []
            for s in samples:
                p = s[0]
                t = s[2] if len(s) > 2 else None
                emb = self.projector.project(p)
                norm = float(np.linalg.norm(emb))
                emb = (emb / norm).astype(np.float32) if norm > 1e-12 else emb
                if t is not None:
                    emb = self.telemetry_projector.fuse(emb, t)
                X_list.append(emb)
            X_all = np.stack(X_list, axis=0).astype(np.float32)

            N = len(prompts)
            X_train = X_all[:num_train]
            X_calib = X_all[num_train:]

            # Construct target Y matrix
            if isinstance(f_def, ChoiceField):
                opt_to_idx = {opt: idx for idx, opt in enumerate(f_def.options)}
                K = len(f_def.options)
                Y_all = np.zeros((N, K), dtype=np.float32)
                for i, lab in enumerate(raw_labels):
                    Y_all[i, opt_to_idx[lab]] = 1.0

                Y_train = Y_all[:num_train]
                Y_calib = Y_all[num_train:]

                # Solve closed-form Ridge Regression
                weights, biases, P_mat, B_mat = self._solve_ridge(
                    X_train, Y_train, self.regularization, regularize_bias=augment, return_covariance=True
                )

                # Calibrate temperature and conformal bounds on independent held-out folds
                logits_calib = (X_calib @ weights.T + biases) / 0.25
                calib_labels = np.argmax(Y_calib, axis=1)

                logits_temp, labels_temp = logits_calib[:num_temperature], calib_labels[:num_temperature]
                logits_conf, labels_conf = logits_calib[num_temperature:], calib_labels[num_temperature:]

                calibrator = DecisionCalibrator()
                learned_temp = 1.0
                if len(labels_temp):
                    calibrator.fit(logits_temp, labels_temp)
                    learned_temp = float(calibrator.temperature)
                probs_calib = calibrator.calibrate_logits(logits_conf)

                # Adaptive Prediction Sets (APS) scoring matching calibration.py on held-out conformal fold
                nonconf_scores = []
                error_margins = []
                for i in range(len(labels_conf)):
                    p_row = probs_calib[i]
                    s_idx = np.argsort(-p_row)
                    top_i = int(s_idx[0])
                    runner_i = int(s_idx[1]) if len(s_idx) > 1 else top_i
                    true_c = int(labels_conf[i])
                    if top_i != true_c:
                        error_margins.append(float(p_row[top_i] - p_row[runner_i]))

                    cum_sum = 0.0
                    score_val = 1.0
                    for c_i in s_idx:
                        cum_sum += float(p_row[c_i])
                        if c_i == true_c:
                            score_val = cum_sum
                            break
                    nonconf_scores.append(score_val if augment else 1.0 - float(p_row[true_c]))

                m_thresh = float(np.clip(np.quantile(error_margins, 0.95) + 0.05, 0.15, 0.85)) if error_margins else 0.20

                sorted_calib_scores = np.sort(np.asarray(nonconf_scores, dtype=np.float64))
                alpha = 0.05
                n_scores = len(sorted_calib_scores)
                k = int(math.ceil((n_scores + 1) * (1.0 - alpha)))
                conformal_q = (1.0 if k > n_scores else float(sorted_calib_scores[k - 1])) if n_scores else 0.0

                compiled_heads[field_name] = CompiledHeadWeights(
                    field_name=field_name,
                    field_type=f_def.field_type,
                    weights=weights,
                    biases=biases,
                    temperature=learned_temp,
                    score_method="aps" if augment else "lac",
                    conformal_quantile=conformal_q,
                    options=f_def.options,
                    P=P_mat,
                    B=B_mat,
                    margin_threshold=m_thresh,
                    forgetting_factor=self.forgetting_factor,
                    relative_odds_ratio=self.relative_odds_ratio,
                    confidence_floor_tau0=self.confidence_floor_tau0,
                    escalate_on_ambiguity=getattr(f_def, "escalate_on_ambiguity", True),
                    calibration_scores=tuple(float(s) for s in sorted_calib_scores),
                )

            elif isinstance(f_def, BooleanField):
                # Logit-space targets: +1.0 for True, -1.0 for False
                Y_all = np.array([1.0 if lab else -1.0 for lab in raw_labels], dtype=np.float32).reshape(-1, 1)
                Y_train = Y_all[:num_train]
                pos_mask = (Y_train.flatten() == 1.0)
                neg_mask = (Y_train.flatten() == -1.0)
                n_pos = int(np.sum(pos_mask))
                n_neg = int(np.sum(neg_mask))
                if n_pos > 0 and n_neg > 0:
                    sw = np.zeros(num_train, dtype=np.float32)
                    sw[pos_mask] = (0.5 * num_train) / n_pos
                    sw[neg_mask] = (0.5 * num_train) / n_neg
                else:
                    sw = None
                weights, biases, P_mat, B_mat = self._solve_ridge(
                    X_train, Y_train, self.regularization, sample_weights=sw, regularize_bias=True, return_covariance=True
                )

                # Calibrate conformal bounds on X_calib
                logits_calib = (X_calib @ weights.T + biases).flatten()
                probs_calib = _stable_sigmoid(logits_calib, temperature=0.25)
                calib_bools = [bool(lab) for lab in (raw_labels[num_train:])]
                nonconf_scores = []
                error_margins_bool = []
                for i in range(len(calib_bools)):
                    p_true = float(probs_calib[i])
                    p_false = 1.0 - p_true
                    p_row = [p_false, p_true]
                    true_idx = 1 if calib_bools[i] else 0
                    top_b = p_true >= 0.5
                    if top_b != calib_bools[i]:
                        error_margins_bool.append(abs(p_true - p_false))

                    # APS score
                    s_idx = np.argsort(-np.array(p_row))
                    cum_s = 0.0
                    score_val = 1.0
                    for c_i in s_idx:
                        cum_s += p_row[c_i]
                        if c_i == true_idx:
                            score_val = cum_s
                            break
                    nonconf_scores.append(score_val if augment else 1.0 - float(p_row[true_idx]))

                m_thresh_bool = float(np.clip(np.quantile(error_margins_bool, 0.95) + 0.05, 0.15, 0.85)) if error_margins_bool else 0.20

                sorted_calib_scores = np.sort(np.asarray(nonconf_scores, dtype=np.float64))
                alpha = 0.05
                n_scores = len(sorted_calib_scores)
                k = int(math.ceil((n_scores + 1) * (1.0 - alpha)))
                conformal_q = (1.0 if k > n_scores else float(sorted_calib_scores[k - 1])) if n_scores else 0.0

                compiled_heads[field_name] = CompiledHeadWeights(
                    field_name=field_name,
                    field_type=f_def.field_type,
                    weights=weights,
                    biases=biases,
                    temperature=1.0,
                    conformal_quantile=conformal_q,
                    options=("False", "True"),
                    P=P_mat,
                    B=B_mat,
                    margin_threshold=m_thresh_bool,
                    score_method="aps" if augment else "lac",
                    forgetting_factor=self.forgetting_factor,
                    relative_odds_ratio=self.relative_odds_ratio,
                    confidence_floor_tau0=self.confidence_floor_tau0,
                    escalate_on_ambiguity=getattr(f_def, "escalate_on_ambiguity", True),
                    calibration_scores=tuple(float(s) for s in sorted_calib_scores),
                )

            elif isinstance(f_def, MultiChoiceField):
                opt_to_idx = {opt: idx for idx, opt in enumerate(f_def.options)}
                K = len(f_def.options)
                N_train = num_train
                weights = np.zeros((K, self.dimension), dtype=np.float32)
                biases = np.zeros(K, dtype=np.float32)

                for k, opt in enumerate(f_def.options):
                    y_train_k = np.array([
                        1.0 if (isinstance(raw_labels[i], (list, tuple, set)) and opt in raw_labels[i]) or raw_labels[i] == opt else -1.0
                        for i in range(N_train)
                    ], dtype=np.float32)
                    pos_mask = (y_train_k == 1.0)
                    neg_mask = (y_train_k == -1.0)
                    n_pos = int(np.sum(pos_mask))
                    n_neg = int(np.sum(neg_mask))
                    sw = np.zeros(N_train, dtype=np.float32)
                    sw[pos_mask] = (0.5 * N_train) / max(1, n_pos)
                    sw[neg_mask] = (0.5 * N_train) / max(1, n_neg)

                    w_k, b_k = self._solve_ridge(
                        X_train, y_train_k, self.regularization, sample_weights=sw, regularize_bias=True
                    )
                    weights[k] = w_k.flatten()
                    biases[k] = float(b_k[0])

                P_mat = (1.0 / max(float(self.regularization), 1e-4)) * np.eye(self.dimension + 1, dtype=np.float32)
                w_aug = np.hstack([weights, biases.reshape(-1, 1)])
                B_mat = (float(self.regularization) * w_aug.T).astype(np.float32)

                # Calibrate conformal bounds on X_calib
                logits_calib = X_calib @ weights.T + biases
                probs_calib = _stable_sigmoid(logits_calib, temperature=0.25)
                calib_labels = raw_labels[num_train:]
                nonconf_scores = []
                for i in range(len(probs_calib)):
                    active_set = set(calib_labels[i] if isinstance(calib_labels[i], (list, tuple, set)) else [calib_labels[i]])
                    scores_i = [
                        float(1.0 - probs_calib[i, k] if opt in active_set else probs_calib[i, k])
                        for k, opt in enumerate(f_def.options)
                    ]
                    nonconf_scores.append(float(np.max(scores_i)) if scores_i else 0.05)
                sorted_calib_scores = np.sort(np.asarray(nonconf_scores, dtype=np.float64))
                alpha = 0.05
                n_scores = len(sorted_calib_scores)
                k = int(math.ceil((n_scores + 1) * (1.0 - alpha)))
                conformal_q = (1.0 if k > n_scores else float(sorted_calib_scores[k - 1])) if n_scores else 0.0

                compiled_heads[field_name] = CompiledHeadWeights(
                    field_name=field_name,
                    field_type=f_def.field_type,
                    weights=weights,
                    biases=biases,
                    temperature=1.0,
                    conformal_quantile=conformal_q,
                    options=f_def.options,
                    P=P_mat,
                    B=B_mat,
                    forgetting_factor=self.forgetting_factor,
                    relative_odds_ratio=self.relative_odds_ratio,
                    confidence_floor_tau0=self.confidence_floor_tau0,
                    escalate_on_ambiguity=getattr(f_def, "escalate_on_ambiguity", True),
                    calibration_scores=tuple(float(s) for s in sorted_calib_scores),
                )

            elif isinstance(f_def, ScoreField):
                val_range = max(1e-6, f_def.max_value - f_def.min_value)
                raw_floats = np.array([float(lab) for lab in raw_labels], dtype=np.float32).reshape(-1, 1)
                # Map target into logit space matching sigmoid evaluation: prob = sigmoid(z / 0.25) => z = 0.25 * ln(r / (1 - r))
                r = np.clip((raw_floats - f_def.min_value) / val_range, 0.01, 0.99)
                Z_all = (0.25 * np.log(r / (1.0 - r))).astype(np.float32)
                Z_train = Z_all[:num_train]
                weights, biases, P_mat, B_mat = self._solve_ridge(
                    X_train, Z_train, self.regularization, regularize_bias=augment, return_covariance=True
                )

                # Calibrate residual bounds on X_calib
                logits_calib = (X_calib @ weights.T + biases).flatten()
                probs_calib = _stable_sigmoid(logits_calib, temperature=0.25)
                pred_vals = f_def.min_value + val_range * probs_calib
                true_calib_floats = raw_floats[num_train:].flatten()
                residuals = [
                    float(abs(pred_vals[i] - true_calib_floats[i]))
                    for i in range(len(pred_vals))
                ]
                sorted_residuals = np.sort(np.asarray(residuals, dtype=np.float64))
                alpha = 0.05
                n_res = len(sorted_residuals)
                k = int(math.ceil((n_res + 1) * (1.0 - alpha)))
                conformal_q = (float(val_range) if k > n_res else float(sorted_residuals[k - 1])) if n_res else 0.0

                compiled_heads[field_name] = CompiledHeadWeights(
                    field_name=field_name,
                    field_type=f_def.field_type,
                    weights=weights,
                    biases=biases,
                    temperature=1.0,
                    conformal_quantile=conformal_q,
                    options=(),
                    P=P_mat,
                    B=B_mat,
                    forgetting_factor=self.forgetting_factor,
                    relative_odds_ratio=self.relative_odds_ratio,
                    confidence_floor_tau0=self.confidence_floor_tau0,
                    escalate_on_ambiguity=getattr(f_def, "escalate_on_ambiguity", True),
                    calibration_scores=tuple(float(s) for s in sorted_residuals),
                )

        return CompiledSystemOneModel(
            schema=self.schema,
            heads=compiled_heads,
            dimension=self.dimension,
            projector=self.projector,
            backend=self.backend,
            forgetting_factor=self.forgetting_factor,
            recency_weighted=self.recency_weighted,
            metadata={
                "regularization": self.regularization,
                "teacher": teacher if augment else None,
                "teaching_mode": "schema_augmented" if augment else "examples",
                "sample_counts": sample_counts,
                "compiled_at": time.time(),
            },
        )

    def compile_and_save(
        self,
        output_path: Union[str, Path],
        exemplars: Optional[Dict[str, Sequence[Tuple[str, Any]]]] = None,
        samples_per_choice: int = 20,
        teacher: str = "synthetic",
        *,
        augment: bool = True,
        calibration_split: float = 0.25,
        calibration_exemplars: Optional[Dict[str, Sequence[Tuple[str, Any]]]] = None,
    ) -> CompiledSystemOneModel:
        """Compiles model and writes directly to target .s1m binary file."""
        model = self.compile(
            exemplars=exemplars,
            samples_per_choice=samples_per_choice,
            teacher=teacher,
            augment=augment,
            calibration_split=calibration_split,
            calibration_exemplars=calibration_exemplars,
        )
        model.save(output_path)
        return model

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        projector: Optional[Any] = None,
        backend: str = "numpy",
    ) -> CompiledSystemOneModel:
        """Loads a compiled model from a .s1m file."""
        return CompiledSystemOneModel.load(path, projector=projector, backend=backend)


__all__ = [
    "SystemOneCompiler",
    "CompiledSystemOneModel",
    "CompiledHeadWeights",
    "MAGIC_HEADER",
]
