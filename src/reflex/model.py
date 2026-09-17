"""Reflex Non-Autoregressive Decision Model.

Executes single-pass schema evaluation directly mapping input prompts to
typed decision fields without token-by-token autoregressive decoding.
Supports Apple Silicon Metal (MLX), NumPy BLAS, and deterministic offline projection.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union
import numpy as np

from reflex.schema import (
    BooleanField,
    ChoiceField,
    DecisionField,
    DecisionSchema,
    MultiChoiceField,
    ScoreField,
)

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    mx = None
    HAS_MLX = False


class DeterministicSemanticProjector:
    """High-dispersion deterministic semantic projector for offline/air-gapped environments.

    Generates consistent dense vectors (default dimension: 384) using character and subword n-grams,
    positional weighting, and pseudo-orthogonal projections seeded by SHA-256.
    Ensures identical strings produce identical vectors, semantic overlap yields positive
    cosine similarity, and orthogonal concepts yield near-zero similarity.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _hash_to_sparse_vec(self, token: str, weight: float = 1.0) -> np.ndarray:
        """Map a token or n-gram deterministically to a sparse pseudo-random vector."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        h = hashlib.sha256(token.encode("utf-8")).digest()
        for i in range(6):
            chunk = h[i * 4 : (i + 1) * 4]
            val = int.from_bytes(chunk, byteorder="big", signed=False)
            idx = val % self.dimension
            sign = 1.0 if ((val >> 16) & 1) == 0 else -1.0
            vec[idx] += sign * weight
        return vec

    def project(self, text: str) -> np.ndarray:
        """Project input text into a deterministic dense vector."""
        if not text or not text.strip():
            v = np.zeros(self.dimension, dtype=np.float32)
            v[0] = 1.0
            return v

        clean_text = text.lower().strip()
        if len(clean_text) > 8192:
            clean_text = clean_text[:4096] + " " + clean_text[-4096:]
        words = re.findall(r"\b\w+\b", clean_text)

        vec = np.zeros(self.dimension, dtype=np.float32)

        # 1. Whole word tokens with position decay and length weighting
        for pos, word in enumerate(words):
            word_weight = math.log1p(len(word)) / math.sqrt(1.0 + pos * 0.05)
            vec += self._hash_to_sparse_vec(f"word:{word}", weight=word_weight * 2.0)

            # Subword 3-grams and 4-grams for morphological/synonym tolerance
            if len(word) >= 3:
                for n in (3, 4):
                    for j in range(len(word) - n + 1):
                        ngram = word[j : j + n]
                        vec += self._hash_to_sparse_vec(f"ng:{ngram}", weight=0.5)

        # 2. Word pairs / bigrams for local contextual semantics
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            vec += self._hash_to_sparse_vec(f"bi:{bigram}", weight=1.5)

        # 3. Global character 4-grams across word boundaries
        boundary_text = f" {clean_text} "
        for j in range(max(0, len(boundary_text) - 4)):
            c_ngram = boundary_text[j : j + 4]
            vec += self._hash_to_sparse_vec(f"char4:{c_ngram}", weight=0.3)

        # L2-normalization
        max_abs = float(np.max(np.abs(vec))) if len(vec) > 0 else 0.0
        if max_abs < 1e-12 or not np.isfinite(max_abs):
            v = np.zeros(self.dimension, dtype=np.float32)
            v[0] = 1.0
            return v
        scaled = vec / max_abs
        norm = float(np.linalg.norm(scaled))
        if norm < 1e-12:
            vec_zero = np.zeros(self.dimension, dtype=np.float32)
            vec_zero[0] = 1.0
            return vec_zero
        return (scaled / norm).astype(np.float32)


def _stable_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature."""
    temp = max(1e-4, float(temperature))
    scaled = np.asarray(logits, dtype=np.float64) / temp
    max_val = np.max(scaled, axis=-1, keepdims=True)
    exp = np.exp(scaled - max_val)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _stable_sigmoid(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable sigmoid with temperature."""
    temp = max(1e-4, float(temperature))
    scaled = np.asarray(logits, dtype=np.float64) / temp
    return np.where(
        scaled >= 0,
        1.0 / (1.0 + np.exp(-scaled)),
        np.exp(scaled) / (1.0 + np.exp(scaled)),
    )


@dataclass(frozen=True)
class RawFieldEvaluation:
    """Intermediate raw evaluation for a single decision field before calibration."""

    field_name: str
    field_type: str
    logits: np.ndarray
    raw_probabilities: np.ndarray
    options: Tuple[str, ...] = ()
    selected_value: Any = None
    confidence: float = 0.0


@dataclass(frozen=True)
class ModelInferenceResult:
    """Raw model output across all fields for an evaluated prompt."""

    prompt: str
    embedding: np.ndarray
    fields: Dict[str, RawFieldEvaluation]
    inference_latency_ms: float


class DecisionFieldHead:
    """Head projecting dense embeddings to logits for a specific schema field."""

    def __init__(
        self,
        field_def: DecisionField,
        dimension: int = 384,
        backend: str = "numpy",
    ) -> None:
        self.field_def = field_def
        self.dimension = dimension
        self.backend = backend
        self.weights: np.ndarray
        self.biases: np.ndarray
        self.mx_weights: Any = None
        self.mx_biases: Any = None
        self._init_head()

    def _init_head(self) -> None:
        projector = DeterministicSemanticProjector(dimension=self.dimension)

        if isinstance(self.field_def, ChoiceField):
            num_options = len(self.field_def.options)
            W = np.zeros((num_options, self.dimension), dtype=np.float32)
            for idx, opt in enumerate(self.field_def.options):
                desc = self.field_def.descriptions.get(opt, "")
                clean_opt = opt.replace("_", " ").replace("-", " ")
                semantic_text = f"{self.field_def.name} option {opt} {clean_opt}: {desc}".strip()
                W[idx] = projector.project(semantic_text)
            norms = np.linalg.norm(W, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self.weights = (W / norms).astype(np.float32)
            self.biases = np.zeros(num_options, dtype=np.float32)

        elif isinstance(self.field_def, BooleanField):
            t_text = f"{self.field_def.name}: {self.field_def.true_description}"
            f_text = f"{self.field_def.name}: {self.field_def.false_description}"
            t_vec = projector.project(t_text)
            f_vec = projector.project(f_text)
            t_norm = t_vec / (np.linalg.norm(t_vec) or 1.0)
            f_norm = f_vec / (np.linalg.norm(f_vec) or 1.0)
            diff = t_norm - f_norm
            norm = np.linalg.norm(diff)
            w = (diff / norm if norm > 0 else t_norm).astype(np.float32)
            self.weights = w.reshape(1, self.dimension)
            thresh = self.field_def.threshold
            b0 = math.log(thresh / (1.0 - thresh)) if 0.0 < thresh < 1.0 else 0.0
            self.biases = np.array([-b0], dtype=np.float32)

        elif isinstance(self.field_def, MultiChoiceField):
            num_options = len(self.field_def.options)
            W = np.zeros((num_options, self.dimension), dtype=np.float32)
            for idx, opt in enumerate(self.field_def.options):
                desc = self.field_def.descriptions.get(opt, "")
                clean_opt = opt.replace("_", " ").replace("-", " ")
                semantic_text = f"{self.field_def.name} tag {opt} {clean_opt}: {desc}".strip()
                W[idx] = projector.project(semantic_text)
            norms = np.linalg.norm(W, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self.weights = (W / norms).astype(np.float32)
            thresh = self.field_def.threshold
            b0 = math.log(thresh / (1.0 - thresh)) if 0.0 < thresh < 1.0 else 0.0
            self.biases = np.full(num_options, -b0, dtype=np.float32)

        elif isinstance(self.field_def, ScoreField):
            l_text = f"{self.field_def.name} low: {self.field_def.low_description}"
            h_text = f"{self.field_def.name} high: {self.field_def.high_description}"
            l_vec = projector.project(l_text)
            h_vec = projector.project(h_text)
            l_norm = l_vec / (np.linalg.norm(l_vec) or 1.0)
            h_norm = h_vec / (np.linalg.norm(h_vec) or 1.0)
            diff = h_norm - l_norm
            norm = np.linalg.norm(diff)
            w = (diff / norm if norm > 0 else h_norm).astype(np.float32)
            self.weights = w.reshape(1, self.dimension)
            self.biases = np.zeros(1, dtype=np.float32)
        else:
            raise TypeError(f"Unsupported field type: {type(self.field_def).__name__}")

        if self.backend == "mlx" and HAS_MLX and mx is not None:
            self.mx_weights = mx.array(self.weights)
            self.mx_biases = mx.array(self.biases)

    def evaluate_logits(self, logits: np.ndarray) -> RawFieldEvaluation:
        """Constructs a RawFieldEvaluation from precomputed logits."""
        logits = np.asarray(logits, dtype=np.float64).flatten()
        if isinstance(self.field_def, ChoiceField):
            scaled_logits = logits / 0.25
            probs = _stable_softmax(scaled_logits, temperature=1.0)
            best_idx = int(np.argmax(probs))
            selected = self.field_def.options[best_idx]
            conf = float(probs[best_idx])
            return RawFieldEvaluation(
                field_name=self.field_def.name,
                field_type=self.field_def.field_type,
                logits=scaled_logits,
                raw_probabilities=probs,
                options=self.field_def.options,
                selected_value=selected,
                confidence=conf,
            )

        elif isinstance(self.field_def, BooleanField):
            prob = float(_stable_sigmoid(logits, temperature=0.25)[0])
            val = bool(prob >= self.field_def.threshold)
            conf = prob if val else (1.0 - prob)
            scaled_z = float(logits[0] / 0.25)
            two_class_logits = np.array([-scaled_z / 2.0, scaled_z / 2.0], dtype=np.float64)
            return RawFieldEvaluation(
                field_name=self.field_def.name,
                field_type=self.field_def.field_type,
                logits=two_class_logits,
                raw_probabilities=np.array([1.0 - prob, prob], dtype=np.float64),
                options=("False", "True"),
                selected_value=val,
                confidence=conf,
            )

        elif isinstance(self.field_def, MultiChoiceField):
            probs = _stable_sigmoid(logits, temperature=0.25)
            selected = tuple(
                self.field_def.options[i]
                for i, p in enumerate(probs)
                if p >= self.field_def.threshold
            )
            conf = float(np.mean([max(p, 1.0 - p) for p in probs])) if len(probs) > 0 else 1.0
            return RawFieldEvaluation(
                field_name=self.field_def.name,
                field_type=self.field_def.field_type,
                logits=logits,
                raw_probabilities=probs,
                options=self.field_def.options,
                selected_value=selected,
                confidence=conf,
            )

        elif isinstance(self.field_def, ScoreField):
            prob = float(_stable_sigmoid(logits, temperature=0.25)[0])
            val = self.field_def.min_value + (self.field_def.max_value - self.field_def.min_value) * prob
            return RawFieldEvaluation(
                field_name=self.field_def.name,
                field_type=self.field_def.field_type,
                logits=logits,
                raw_probabilities=np.array([prob], dtype=np.float64),
                options=(),
                selected_value=float(val),
                confidence=float(max(prob, 1.0 - prob)),
            )

        raise TypeError(f"Unknown field type: {self.field_def}")

    def forward(self, embedding: np.ndarray) -> RawFieldEvaluation:
        """Projects dense embedding into logits and raw probabilities."""
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        if self.backend == "mlx" and HAS_MLX and mx is not None and self.mx_weights is not None:
            mx_emb = mx.array(emb)
            mx_logits = mx.matmul(mx_emb, mx.transpose(self.mx_weights)) + self.mx_biases
            mx.eval(mx_logits)
            logits = np.array(mx_logits).flatten()
        else:
            logits = (emb @ self.weights.T + self.biases).flatten()

        return self.evaluate_logits(logits)


class SystemOneModel:
    """Complete non-autoregressive decision model executing single-pass schema evaluation.

    Encodes input text into a high-dimensional dense representation and routes it
    through calibrated multi-task decision heads.
    Supports Apple Silicon Metal (MLX) GPU execution and optimized NumPy BLAS.
    """

    def __init__(
        self,
        schema: Union[DecisionSchema, Type[DecisionSchema]],
        *,
        dimension: int = 384,
        backend: str = "auto",
    ) -> None:
        if isinstance(schema, type) and issubclass(schema, DecisionSchema):
            self.schema: DecisionSchema = schema()
        elif isinstance(schema, DecisionSchema):
            self.schema = schema
        else:
            raise TypeError(f"Expected DecisionSchema subclass or instance, got {type(schema).__name__}")

        self.dimension = dimension
        self.backend = self._resolve_backend(backend)
        self.projector = DeterministicSemanticProjector(dimension=self.dimension)
        self.heads: Dict[str, DecisionFieldHead] = {
            name: DecisionFieldHead(f, dimension=self.dimension, backend=self.backend)
            for name, f in self.schema.fields.items()
        }

        # Pre-warm backend kernels (e.g. Metal GPU JIT shader pipeline compilation)
        if self.backend == "mlx" and HAS_MLX and mx is not None:
            dummy_emb = np.zeros((1, self.dimension), dtype=np.float32)
            for head in self.heads.values():
                head.forward(dummy_emb)

    def _resolve_backend(self, backend: str) -> str:
        if backend == "auto":
            return "mlx" if HAS_MLX else "numpy"
        if backend == "mlx":
            if not HAS_MLX:
                raise RuntimeError("MLX requested but mlx is not installed on this system")
            return "mlx"
        return "numpy"

    def encode(self, prompt: str) -> np.ndarray:
        """Encodes prompt into a normalized dense embedding vector."""
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")
        emb = self.projector.project(prompt)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def forward_single(self, prompt: str) -> ModelInferenceResult:
        """Executes a single non-autoregressive forward pass over the schema."""
        start_t = time.perf_counter()
        embedding = self.encode(prompt)

        field_results: Dict[str, RawFieldEvaluation] = {}
        for name, head in self.heads.items():
            field_results[name] = head.forward(embedding)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return ModelInferenceResult(
            prompt=prompt,
            embedding=embedding,
            fields=field_results,
            inference_latency_ms=elapsed_ms,
        )

    def forward_batch(self, prompts: Sequence[str]) -> List[ModelInferenceResult]:
        """Batched forward pass over multiple inputs with Metal GPU and NumPy BLAS optimization."""
        if not prompts:
            return []

        start_t = time.perf_counter()
        embs = np.stack([self.encode(p) for p in prompts], axis=0).astype(np.float32)

        head_logits: Dict[str, np.ndarray] = {}
        if self.backend == "mlx" and HAS_MLX and mx is not None:
            mx_embs = mx.array(embs)
            for name, head in self.heads.items():
                if head.mx_weights is not None:
                    mx_out = mx.matmul(mx_embs, mx.transpose(head.mx_weights)) + head.mx_biases
                    mx.eval(mx_out)
                    head_logits[name] = np.array(mx_out)
                else:
                    head_logits[name] = embs @ head.weights.T + head.biases
        else:
            for name, head in self.heads.items():
                head_logits[name] = embs @ head.weights.T + head.biases

        total_elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        per_sample_latency = total_elapsed_ms / len(prompts)

        results: List[ModelInferenceResult] = []
        for i, p in enumerate(prompts):
            f_res: Dict[str, RawFieldEvaluation] = {}
            for name, head in self.heads.items():
                f_res[name] = head.evaluate_logits(head_logits[name][i])
            results.append(
                ModelInferenceResult(
                    prompt=p,
                    embedding=embs[i],
                    fields=f_res,
                    inference_latency_ms=per_sample_latency,
                )
            )
        return results
