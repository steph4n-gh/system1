"""System 1 Core Local Neural Projector.

Executes local neural semantic projection mapping text prompts directly into
calibrated dense representations for single-pass non-autoregressive decision modeling.
Supports Apple Silicon Metal (MLX) GPU execution and optimized NumPy BLAS execution.

Zero external dependencies beyond NumPy (and optional Apple Silicon MLX).
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    mx = None
    nn = None
    HAS_MLX = False


def _gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (GELU) numerical approximation."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization across the last axis."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


class LocalNeuralProjector:
    """On-device neural semantic projector for high-throughput decision modeling.

    Transforms raw text into normalized continuous dense representations using:
    1. Multi-scale subword and token feature hashing (feature_dim=1024).
    2. Dense feedforward projection layer with non-linear activation (GELU).
    3. Layer normalization and dimensional bottleneck reduction.
    4. Unit L2-sphere normalization for cosine distance stability.
    5. Apple Silicon Metal (MLX) hardware acceleration when available.
    """

    def __init__(
        self,
        dimension: int = 384,
        hidden_dim: int = 512,
        feature_dim: int = 1024,
        seed: int = 42,
        backend: str = "auto",
        activation: str = "gelu",
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")

        valid_activations = ("gelu", "relu", "tanh")
        if activation not in valid_activations:
            raise ValueError(
                f"Unsupported activation {activation!r}. Expected one of: {', '.join(valid_activations)}"
            )

        self.dimension = dimension
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.seed = seed
        self.backend = self._resolve_backend(backend)
        self.activation = activation

        # Deterministic weight initialization
        rng = np.random.RandomState(seed)
        limit1 = float(np.sqrt(2.0 / (feature_dim + hidden_dim)))
        self.w1 = rng.normal(0, limit1, size=(feature_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        limit2 = float(np.sqrt(2.0 / (hidden_dim + dimension)))
        self.w2 = rng.normal(0, limit2, size=(hidden_dim, dimension)).astype(np.float32)
        self.b2 = np.zeros(dimension, dtype=np.float32)

        # MLX tensors if enabled
        self.mx_w1 = None
        self.mx_b1 = None
        self.mx_w2 = None
        self.mx_b2 = None
        if self.backend == "mlx" and HAS_MLX and mx is not None:
            self.mx_w1 = mx.array(self.w1)
            self.mx_b1 = mx.array(self.b1)
            self.mx_w2 = mx.array(self.w2)
            self.mx_b2 = mx.array(self.b2)

    def _resolve_backend(self, backend: str) -> str:
        if backend == "auto":
            return "mlx" if HAS_MLX else "numpy"
        if backend == "mlx":
            if not HAS_MLX:
                raise RuntimeError("MLX requested but mlx is not installed on this system")
            return "mlx"
        return "numpy"

    def _hash_text_features(self, text: str) -> np.ndarray:
        """Hashes words, bigrams, and character n-grams into a fixed feature vector."""
        feats = np.zeros(self.feature_dim, dtype=np.float32)
        if not text or not text.strip():
            feats[0] = 1.0
            return feats

        clean_text = text.lower().strip()
        if len(clean_text) > 8192:
            clean_text = clean_text[:4096] + " " + clean_text[-4096:]

        words = re.findall(r"\b\w+\b", clean_text)
        for pos, word in enumerate(words):
            weight = math.log1p(len(word)) / math.sqrt(1.0 + pos * 0.05)
            h = hashlib.sha256(f"w:{word}".encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "big") % self.feature_dim
            sign = 1.0 if (h[4] & 1) == 0 else -1.0
            feats[idx] += sign * weight

            if len(word) >= 3:
                for n in (3, 4):
                    for j in range(len(word) - n + 1):
                        sub = word[j : j + n]
                        h_sub = hashlib.sha256(f"sub:{sub}".encode("utf-8")).digest()
                        s_idx = int.from_bytes(h_sub[:4], "big") % self.feature_dim
                        s_sign = 1.0 if (h_sub[4] & 1) == 0 else -1.0
                        feats[s_idx] += s_sign * 0.4

        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            h_bi = hashlib.sha256(f"bi:{bigram}".encode("utf-8")).digest()
            b_idx = int.from_bytes(h_bi[:4], "big") % self.feature_dim
            b_sign = 1.0 if (h_bi[4] & 1) == 0 else -1.0
            feats[b_idx] += b_sign * 1.2

        # L2-normalize input features
        norm = float(np.linalg.norm(feats))
        if norm > 1e-12:
            feats = feats / norm
        else:
            feats[0] = 1.0
        return feats

    def project(self, text: str) -> np.ndarray:
        """Projects a single text string into a normalized dense vector."""
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be a string, got {type(text).__name__}")
        feats = self._hash_text_features(text)
        return self._forward_vector(feats)

    encode = project

    def _forward_vector(self, feats: np.ndarray) -> np.ndarray:
        if self.backend == "mlx" and HAS_MLX and mx is not None and self.mx_w1 is not None:
            x = mx.array(feats.reshape(1, -1))
            h = mx.matmul(x, self.mx_w1) + self.mx_b1
            if self.activation == "gelu":
                if nn is not None and hasattr(nn, "gelu_approx"):
                    h = nn.gelu_approx(h)
                elif nn is not None and hasattr(nn, "gelu"):
                    h = nn.gelu(h)
                else:
                    h = 0.5 * h * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (h + 0.044715 * mx.power(h, 3))))
            elif self.activation == "relu":
                h = mx.maximum(h, 0.0)
            elif self.activation == "tanh":
                h = mx.tanh(h)
            m_mean = mx.mean(h, axis=-1, keepdims=True)
            m_var = mx.var(h, axis=-1, keepdims=True)
            h = (h - m_mean) / mx.sqrt(m_var + 1e-5)
            out = mx.matmul(h, self.mx_w2) + self.mx_b2
            mx.eval(out)
            vec = np.array(out).flatten()
        else:
            x = feats.reshape(1, -1)
            h = x @ self.w1 + self.b1
            if self.activation == "gelu":
                h = _gelu(h)
            elif self.activation == "relu":
                h = np.maximum(h, 0.0)
            elif self.activation == "tanh":
                h = np.tanh(h)
            h = _layer_norm(h)
            out = h @ self.w2 + self.b2
            vec = out.flatten()

        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec = vec / norm
        else:
            vec = np.zeros(self.dimension, dtype=np.float32)
            vec[0] = 1.0
        return vec.astype(np.float32)

    def project_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Batched projection of multiple text inputs."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        for idx, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(f"All elements in texts must be strings, got {type(t).__name__} at index {idx}")
        feats_list = [self._hash_text_features(t) for t in texts]
        X = np.stack(feats_list, axis=0).astype(np.float32)

        if self.backend == "mlx" and HAS_MLX and mx is not None and self.mx_w1 is not None:
            mx_X = mx.array(X)
            h = mx.matmul(mx_X, self.mx_w1) + self.mx_b1
            if self.activation == "gelu":
                if nn is not None and hasattr(nn, "gelu_approx"):
                    h = nn.gelu_approx(h)
                elif nn is not None and hasattr(nn, "gelu"):
                    h = nn.gelu(h)
                else:
                    h = 0.5 * h * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (h + 0.044715 * mx.power(h, 3))))
            elif self.activation == "relu":
                h = mx.maximum(h, 0.0)
            elif self.activation == "tanh":
                h = mx.tanh(h)
            m_mean = mx.mean(h, axis=-1, keepdims=True)
            m_var = mx.var(h, axis=-1, keepdims=True)
            h = (h - m_mean) / mx.sqrt(m_var + 1e-5)
            out = mx.matmul(h, self.mx_w2) + self.mx_b2
            mx.eval(out)
            res = np.array(out)
        else:
            h = X @ self.w1 + self.b1
            if self.activation == "gelu":
                h = _gelu(h)
            elif self.activation == "relu":
                h = np.maximum(h, 0.0)
            elif self.activation == "tanh":
                h = np.tanh(h)
            h = _layer_norm(h)
            res = h @ self.w2 + self.b2

        norms = np.linalg.norm(res, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return (res / norms).astype(np.float32)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Computes cosine similarity between two text strings."""
        v_a = self.project(text_a)
        v_b = self.project(text_b)
        return float(np.dot(v_a, v_b))


__all__ = [
    "LocalNeuralProjector",
    "HAS_MLX",
    "_gelu",
    "_layer_norm",
]
