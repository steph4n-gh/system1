"""System 1 Core Non-Autoregressive Decision Model.

Executes single-pass schema evaluation directly mapping input prompts to
typed decision fields without token-by-token autoregressive decoding.
Supports Apple Silicon Metal (MLX), NumPy BLAS, and deterministic offline projection.

Zero external dependencies beyond NumPy (and optional Apple Silicon MLX).
Zero cryptography, SQLite, or disk I/O dependencies.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union
import numpy as np

from system1.core.schema import (
    BooleanField,
    ChoiceField,
    DecisionField,
    DecisionSchema,
    MultiChoiceField,
    ScoreField,
)
from system1.core.telemetry import TelemetryProjector

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    mx = None
    HAS_MLX = False


class DeterministicSemanticProjector:
    """Deterministic normalized lexical features for local decisions.

    SHA-256-derived word and character features use 384 dimensions by default.
    Shared features can support generalization, but similarity does not guarantee
    semantic equivalence or distinguish all unrelated concepts."""

    def __init__(self, dimension: int = 384, recency_weighted: bool = False):
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self.dimension = dimension
        self.recency_weighted = bool(recency_weighted)

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

    def project(self, text: str, recency_weighted: Optional[bool] = None) -> np.ndarray:
        """Project input text into a deterministic dense vector.

        Supports recency weighting for multi-turn agent traces / command histories.
        When recency_weighted=True, trailing tokens receive full weight 1.0 and historical
        context decays backwards from the end:
            weight(i) = log(1 + len(w_i)) / sqrt(1.0 + 0.05 * (N - 1 - i))
        When recency_weighted=False (default), standard forward position decay applies:
            weight(i) = log(1 + len(w_i)) / sqrt(1.0 + 0.05 * i)
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be a string, got {type(text).__name__}")
        if not text or not text.strip():
            v = np.zeros(self.dimension, dtype=np.float32)
            v[0] = 1.0
            return v

        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)

        clean_text = text.lower().strip()
        if len(clean_text) > 8192:
            clean_text = clean_text[:4096] + " " + clean_text[-4096:]
        words = re.findall(r"\b\w+\b", clean_text)
        n_words = len(words)

        vec = np.zeros(self.dimension, dtype=np.float32)

        # 1. Whole word tokens with position decay and length weighting
        for pos, word in enumerate(words):
            dist = (n_words - 1 - pos) if use_recency else pos
            word_weight = math.log1p(len(word)) / math.sqrt(1.0 + dist * 0.05)
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

    encode = project

    def project_batch(self, texts: Sequence[str], recency_weighted: Optional[bool] = None) -> np.ndarray:
        """Batched projection of multiple text inputs."""
        if texts is None:
            raise TypeError("texts must be a sequence of strings, got None")
        if not hasattr(texts, "__iter__"):
            raise TypeError(f"texts must be a sequence of strings, got {type(texts).__name__}")
        if len(texts) == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        for idx, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(f"All elements in texts must be strings, got {type(t).__name__} at index {idx}")
        return np.stack([self.project(t, recency_weighted=recency_weighted) for t in texts], axis=0).astype(np.float32)

    def similarity(self, text_a: str, text_b: str, recency_weighted: Optional[bool] = None) -> float:
        """Computes cosine similarity between two text strings."""
        v_a = self.project(text_a, recency_weighted=recency_weighted)
        v_b = self.project(text_b, recency_weighted=recency_weighted)
        return float(np.dot(v_a, v_b))


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
    pos_mask = scaled >= 0
    out = np.empty_like(scaled)
    out[pos_mask] = 1.0 / (1.0 + np.exp(-scaled[pos_mask]))
    exp_neg = np.exp(scaled[~pos_mask])
    out[~pos_mask] = exp_neg / (1.0 + exp_neg)
    return out


# Public aliases for stable softmax and sigmoid
stable_softmax = _stable_softmax
stable_sigmoid = _stable_sigmoid


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
        projector: Optional[Any] = None,
        contrastive_whitening: bool = True,
        forgetting_factor: float = 0.995,
    ) -> None:
        self.field_def = field_def
        self.dimension = dimension
        self.backend = backend
        self.projector = projector
        self.contrastive_whitening = contrastive_whitening
        self.forgetting_factor = float(forgetting_factor)
        self.weights: np.ndarray
        self.biases: np.ndarray
        self.mx_weights: Any = None
        self.mx_biases: Any = None
        self.covariance_inv: Optional[np.ndarray] = None
        self.cross_covariance: Optional[np.ndarray] = None
        self.regularization: float = 1.0
        self._init_head()

    def _init_head(self) -> None:
        if self.projector is None:
            self.projector = DeterministicSemanticProjector(dimension=self.dimension)
        projector = self.projector

        if isinstance(self.field_def, ChoiceField):
            num_options = len(self.field_def.options)
            W = np.zeros((num_options, self.dimension), dtype=np.float32)
            for idx, opt in enumerate(self.field_def.options):
                desc = self.field_def.descriptions.get(opt, "")
                clean_opt = opt.replace("_", " ").replace("-", " ")
                if isinstance(desc, (list, tuple)) and len(desc) > 0:
                    ex_vecs = []
                    for ex in desc:
                        ex_text = str(ex).strip()
                        if ex_text:
                            v = projector.project(ex_text)
                            norm_v = float(np.linalg.norm(v))
                            ex_vecs.append(v / (norm_v if norm_v > 1e-12 else 1.0))
                    if ex_vecs:
                        centroid = np.mean(ex_vecs, axis=0)
                        norm_c = float(np.linalg.norm(centroid))
                        W[idx] = centroid / (norm_c if norm_c > 1e-12 else 1.0)
                    else:
                        semantic_text = f"{self.field_def.name} option {opt} {clean_opt}".strip()
                        v = projector.project(semantic_text)
                        norm_v = float(np.linalg.norm(v))
                        W[idx] = v / (norm_v if norm_v > 1e-12 else 1.0)
                else:
                    semantic_text = f"{self.field_def.name} option {opt} {clean_opt}: {desc}".strip()
                    v = projector.project(semantic_text)
                    norm_v = float(np.linalg.norm(v))
                    W[idx] = v / (norm_v if norm_v > 1e-12 else 1.0)

            # Contrastive centering and whitening across choices
            if self.contrastive_whitening and num_options > 1:
                mu = np.mean(W, axis=0, keepdims=True)
                W_centered = W - mu
                norms = np.linalg.norm(W_centered, axis=1, keepdims=True)
                norms = np.where(norms < 1e-12, 1.0, norms)
                self.weights = (W_centered / norms).astype(np.float32)
            else:
                norms = np.linalg.norm(W, axis=1, keepdims=True)
                norms = np.where(norms < 1e-12, 1.0, norms)
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
                if isinstance(desc, (list, tuple)) and len(desc) > 0:
                    ex_vecs = []
                    for ex in desc:
                        ex_text = str(ex).strip()
                        if ex_text:
                            v = projector.project(ex_text)
                            norm_v = float(np.linalg.norm(v))
                            ex_vecs.append(v / (norm_v if norm_v > 1e-12 else 1.0))
                    if ex_vecs:
                        centroid = np.mean(ex_vecs, axis=0)
                        norm_c = float(np.linalg.norm(centroid))
                        W[idx] = centroid / (norm_c if norm_c > 1e-12 else 1.0)
                    else:
                        semantic_text = f"{self.field_def.name} tag {opt} {clean_opt}".strip()
                        v = projector.project(semantic_text)
                        norm_v = float(np.linalg.norm(v))
                        W[idx] = v / (norm_v if norm_v > 1e-12 else 1.0)
                else:
                    semantic_text = f"{self.field_def.name} tag {opt} {clean_opt}: {desc}".strip()
                    v = projector.project(semantic_text)
                    norm_v = float(np.linalg.norm(v))
                    W[idx] = v / (norm_v if norm_v > 1e-12 else 1.0)

            norms = np.linalg.norm(W, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            self.weights = (W / norms).astype(np.float32)

            thresh = self.field_def.threshold
            b0 = (0.25 * math.log(thresh / (1.0 - thresh))) if 0.0 < thresh < 1.0 else 0.0
            # Apply baseline margin to prevent unflagged background text from spuriously triggering tags
            self.biases = np.full(num_options, b0 - 0.15, dtype=np.float32)

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

    def set_weights(self, weights: np.ndarray, biases: Optional[np.ndarray] = None) -> None:
        """Updates head weights and biases, synchronizing MLX buffers if active."""
        self.weights = np.asarray(weights, dtype=np.float32)
        if biases is not None:
            self.biases = np.asarray(biases, dtype=np.float32)
        else:
            self.biases = np.zeros(self.weights.shape[0], dtype=np.float32)
        if self.backend == "mlx" and HAS_MLX and mx is not None:
            self.mx_weights = mx.array(self.weights)
            self.mx_biases = mx.array(self.biases)

    def init_covariance(
        self,
        P: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        regularization: float = 1.0,
    ) -> None:
        """Initializes covariance inverse P and cross-covariance B for Sherman-Morrison online distillation.

        If P is None, initializes P = (1/lambda) * I_(D+1) and B = lambda * [W, b]^T,
        so (P B)^T == [W, b] identically reproducing existing weights.
        """
        self.regularization = max(1e-4, float(regularization))
        D = self.dimension
        if P is not None:
            self.covariance_inv = np.asarray(P, dtype=np.float32)
        else:
            self.covariance_inv = (1.0 / self.regularization) * np.eye(D + 1, dtype=np.float32)

        if B is not None:
            self.cross_covariance = np.asarray(B, dtype=np.float32)
        else:
            w_mat = self.weights  # shape (K, D)
            b_vec = self.biases.reshape(-1, 1)  # shape (K, 1)
            w_aug = np.hstack([w_mat, b_vec])  # shape (K, D+1)
            self.cross_covariance = (self.regularization * w_aug.T).astype(np.float32)

    def format_target_vector(self, target_value: Any) -> np.ndarray:
        """Converts high-level field target value into numeric regression target vector y*."""
        if isinstance(self.field_def, ChoiceField):
            opts = self.field_def.options
            y = np.zeros(len(opts), dtype=np.float32)
            if target_value in opts:
                idx = opts.index(target_value)
                y[idx] = 1.0
            elif isinstance(target_value, int) and 0 <= target_value < len(opts):
                y[target_value] = 1.0
            return y
        elif isinstance(self.field_def, BooleanField):
            val = bool(target_value)
            return np.array([1.0 if val else -1.0], dtype=np.float32)
        elif isinstance(self.field_def, ScoreField):
            val = float(target_value)
            val_range = max(1e-6, self.field_def.max_value - self.field_def.min_value)
            # Inverse sigmoid logit transformation matching _solve_ridge:
            # prob = sigmoid(z / 0.25) => z = 0.25 * ln(r / (1 - r))
            r = float(np.clip((val - self.field_def.min_value) / val_range, 1e-4, 1.0 - 1e-4))
            logit_target = 0.25 * math.log(r / (1.0 - r))
            return np.array([logit_target], dtype=np.float32)
        elif isinstance(self.field_def, MultiChoiceField):
            opts = self.field_def.options
            y = np.full(len(opts), -1.0, dtype=np.float32)
            if isinstance(target_value, (list, tuple, set)):
                for idx, opt in enumerate(opts):
                    if opt in target_value:
                        y[idx] = 1.0
            elif target_value in opts:
                y[opts.index(target_value)] = 1.0
            return y
        return np.asarray(target_value, dtype=np.float32)

    def online_update(
        self,
        x_aug: np.ndarray,
        y_target: np.ndarray,
        forgetting_factor: Optional[float] = None,
    ) -> float:
        r"""Closed-form rank-1 Sherman-Morrison update with exponential forgetting factor.

        P_{t+1} = (1 / \lambda_f) [ P_t - (P_t x x^T P_t) / (\lambda_f + x^T P_t x) ]
        B_{t+1} = \lambda_f B_t + x y^T
        W_{t+1} = (P_{t+1} B_{t+1})^T
        """
        t0 = time.perf_counter()
        if self.covariance_inv is None or self.cross_covariance is None:
            self.init_covariance(regularization=self.regularization)

        lam_f = float(
            forgetting_factor if forgetting_factor is not None else self.forgetting_factor
        )
        if not (0.0 < lam_f <= 1.0):
            raise ValueError(f"Forgetting factor lambda_f must be in (0, 1.0], got {lam_f}")

        P = self.covariance_inv
        B = self.cross_covariance
        D = self.dimension
        K = self.weights.shape[0]

        x = np.asarray(x_aug, dtype=np.float32).flatten()
        if len(x) == D:
            # If prompt embedding without bias 1, augment
            x = np.append(x, 1.0).astype(np.float32)
        elif len(x) != D + 1:
            raise ValueError(f"Expected x_aug of length {D} or {D+1}, got {len(x)}")

        if not np.all(np.isfinite(x)):
            raise ValueError("x_aug contains NaN or Inf non-finite values")

        y = np.asarray(y_target, dtype=np.float32).flatten()
        if not np.all(np.isfinite(y)):
            raise ValueError("y_target contains NaN or Inf non-finite values")
        if len(y) != K:
            raise ValueError(f"Expected target vector y of length {K}, got {len(y)}")

        # v = P @ x_aug (shape: D+1,)
        v = P @ x
        denom = float(lam_f + x @ v)
        if denom <= 1e-12 or not math.isfinite(denom):
            # Degenerate input or numerical singularity: preserve internal state
            return (time.perf_counter() - t0) * 1000.0

        # Rank-1 update with exponential forgetting factor:
        # P_{t+1} = (1 / lam_f) * [ P_t - (v v^T) / denom ]
        P_next = (1.0 / lam_f) * (P - (np.outer(v, v) / denom))
        P_next = 0.5 * (P_next + P_next.T)

        # Cross-covariance update: B_{t+1} = lam_f * B_t + x y^T
        B_next = (lam_f * B) + np.outer(x, y)

        # Covariance bounding: prevent covariance windup along unexcited subspace directions
        p_max = float(50.0 / max(1e-4, self.regularization))
        max_d = float(np.max(np.diag(P_next)))
        if max_d > p_max and max_d > 0:
            scale = p_max / max_d
            P_next *= scale
            B_next *= (1.0 / scale)

        # Enforce strict positive-definiteness on diagonal against floating-point drift
        np.fill_diagonal(P_next, np.maximum(np.diag(P_next), 1e-6))

        # Weight update: W_aug = (P_{t+1} @ B_{t+1})^T
        W_aug = (P_next @ B_next).T
        new_W = W_aug[:, :D].astype(np.float32)
        new_b = W_aug[:, D].astype(np.float32)

        # Atomic commit
        self.covariance_inv = P_next.astype(np.float32)
        self.cross_covariance = B_next.astype(np.float32)
        self.set_weights(new_W, new_b)
        return (time.perf_counter() - t0) * 1000.0

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
            scaled_logits = logits / 0.25
            probs = _stable_sigmoid(scaled_logits, temperature=1.0)
            selected = tuple(
                self.field_def.options[i]
                for i, p in enumerate(probs)
                if p >= self.field_def.threshold
            )
            conf = float(np.mean([max(p, 1.0 - p) for p in probs])) if len(probs) > 0 else 1.0
            return RawFieldEvaluation(
                field_name=self.field_def.name,
                field_type=self.field_def.field_type,
                logits=scaled_logits,
                raw_probabilities=probs,
                options=self.field_def.options,
                selected_value=selected,
                confidence=conf,
            )

        elif isinstance(self.field_def, ScoreField):
            scaled_logits = logits / 0.25
            prob = float(_stable_sigmoid(scaled_logits, temperature=1.0)[0])
            val = self.field_def.min_value + (self.field_def.max_value - self.field_def.min_value) * prob
            return RawFieldEvaluation(
                field_name=self.field_def.name,
                field_type=self.field_def.field_type,
                logits=scaled_logits,
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
        projector: Optional[Any] = None,
        contrastive_whitening: bool = True,
        forgetting_factor: float = 0.995,
        recency_weighted: bool = False,
    ) -> None:
        if isinstance(schema, type) and issubclass(schema, DecisionSchema):
            self.schema: DecisionSchema = schema()
        elif isinstance(schema, DecisionSchema):
            self.schema = schema
        else:
            raise TypeError(f"Expected DecisionSchema subclass or instance, got {type(schema).__name__}")

        self.dimension = dimension
        self.backend = self._resolve_backend(backend)
        self.contrastive_whitening = contrastive_whitening
        self.forgetting_factor = float(forgetting_factor)
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

        self.heads: Dict[str, DecisionFieldHead] = {
            name: DecisionFieldHead(
                f,
                dimension=self.dimension,
                backend=self.backend,
                projector=self.projector,
                contrastive_whitening=self.contrastive_whitening,
                forgetting_factor=self.forgetting_factor,
            )
            for name, f in self.schema.fields.items()
        }

        self._lock = threading.RLock()

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

    def encode(
        self,
        prompt: str,
        telemetry: Optional[Any] = None,
        recency_weighted: Optional[bool] = None,
    ) -> np.ndarray:
        """Encodes prompt into a normalized dense embedding vector, fusing continuous telemetry if present."""
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
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        if telemetry is not None:
            emb = self.telemetry_projector.fuse(emb, telemetry)
        return emb

    def encode_batch(
        self,
        prompts: Sequence[str],
        telemetry: Optional[Sequence[Any]] = None,
        recency_weighted: Optional[bool] = None,
    ) -> np.ndarray:
        """Encodes a sequence of prompts (and optional telemetry) into normalized dense embedding vectors."""
        if prompts is None:
            raise TypeError("prompts must be a sequence of strings, got None")
        if not hasattr(prompts, "__iter__"):
            raise TypeError(f"prompts must be a sequence of strings, got {type(prompts).__name__}")
        if len(prompts) == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        for idx, p in enumerate(prompts):
            if not isinstance(p, str):
                raise TypeError(f"All elements in prompts must be strings, got {type(p).__name__} at index {idx}")
        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
        if telemetry is not None and len(telemetry) == len(prompts):
            return np.stack([self.encode(p, telemetry[i], recency_weighted=use_recency) for i, p in enumerate(prompts)], axis=0).astype(np.float32)
        if hasattr(self.projector, "project_batch"):
            try:
                embs = self.projector.project_batch(prompts, recency_weighted=use_recency)
            except TypeError:
                embs = self.projector.project_batch(prompts)
        else:
            embs = np.stack([self.encode(p, recency_weighted=use_recency) for p in prompts], axis=0).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return (embs / norms).astype(np.float32)

    def forward_single(
        self,
        prompt: str,
        telemetry: Optional[Any] = None,
        embedding: Optional[np.ndarray] = None,
        recency_weighted: Optional[bool] = None,
    ) -> ModelInferenceResult:
        """Executes a single non-autoregressive forward pass over the schema."""
        start_t = time.perf_counter()
        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
        if embedding is not None:
            emb = np.asarray(embedding, dtype=np.float32).flatten()
        else:
            emb = self.encode(prompt, telemetry=telemetry, recency_weighted=use_recency)

        field_results: Dict[str, RawFieldEvaluation] = {}
        for name, head in self.heads.items():
            field_results[name] = head.forward(emb)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return ModelInferenceResult(
            prompt=prompt,
            embedding=emb,
            fields=field_results,
            inference_latency_ms=elapsed_ms,
        )

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
        """Closed-form rank-1 Sherman-Morrison online update on the metal (<0.1ms)."""
        with self._lock:
            t0 = time.perf_counter()
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
                if f_name in self.heads:
                    head = self.heads[f_name]
                    y_target = head.format_target_vector(target_val)
                    dt = head.online_update(x_aug, y_target, forgetting_factor=forgetting_factor)
                    update_durations_ms.append(dt)
                    updated_fields.append(f_name)

            rank1_ms = float(sum(update_durations_ms)) if update_durations_ms else 0.0
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

    def forward_batch(
        self,
        prompts: Sequence[str],
        telemetry: Optional[Sequence[Any]] = None,
        recency_weighted: Optional[bool] = None,
    ) -> List[ModelInferenceResult]:
        """Batched forward pass over multiple inputs with Metal GPU and NumPy BLAS optimization."""
        if prompts is None:
            raise TypeError("prompts must be a sequence of strings, got None")
        if not hasattr(prompts, "__iter__"):
            raise TypeError(f"prompts must be a sequence of strings, got {type(prompts).__name__}")
        if len(prompts) == 0:
            return []

        start_t = time.perf_counter()
        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
        embs = self.encode_batch(prompts, telemetry=telemetry, recency_weighted=use_recency)

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


__all__ = [
    "DeterministicSemanticProjector",
    "DecisionFieldHead",
    "RawFieldEvaluation",
    "ModelInferenceResult",
    "SystemOneModel",
    "HAS_MLX",
    "_stable_softmax",
    "_stable_sigmoid",
    "stable_softmax",
    "stable_sigmoid",
]
