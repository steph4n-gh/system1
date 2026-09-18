"""Telemetry & Continuous State Vector Fusion Engine.

Fuses continuous numeric telemetry vectors (floats, e.g. HP ratio, risk velocity,
account age, balance, CPU burst) with dense semantic embeddings so linear hyperplanes
have razor-sharp numeric decision boundaries.

Supports:
- Normalized telemetry projection (projects continuous vector into embedding space via
  deterministic pseudo-orthogonal projection, preserving base dimension D).
- Telemetry concatenation ([semantic_emb, beta * telemetry_norm]).
- Robust numeric normalization (tanh, minmax, standard, or pass-through).
- Zero external dependencies beyond pure NumPy and standard library.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


class TelemetryProjector:
    """Projects and fuses continuous numeric telemetry with semantic embeddings."""

    def __init__(
        self,
        embedding_dim: int = 384,
        telemetry_dim: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = None,
        normalization: str = "robust",
        mode: str = "project",
        fusion_weight: float = 0.5,
        means: Optional[Sequence[float]] = None,
        stds: Optional[Sequence[float]] = None,
        mins: Optional[Sequence[float]] = None,
        maxs: Optional[Sequence[float]] = None,
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        valid_norms = ("robust", "minmax", "standard", "none")
        if normalization not in valid_norms:
            raise ValueError(f"Invalid normalization {normalization!r}. Expected one of {valid_norms}")
        valid_modes = ("project", "concat")
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode {mode!r}. Expected one of {valid_modes}")

        self.embedding_dim = int(embedding_dim)
        self.telemetry_dim = int(telemetry_dim) if telemetry_dim is not None else None
        self.feature_names = tuple(feature_names) if feature_names is not None else None
        self.normalization = normalization
        self.mode = mode
        self.fusion_weight = float(np.clip(fusion_weight, 0.0, 1.0))

        self.means = np.asarray(means, dtype=np.float32) if means is not None else None
        self.stds = np.asarray(stds, dtype=np.float32) if stds is not None else None
        self.mins = np.asarray(mins, dtype=np.float32) if mins is not None else None
        self.maxs = np.asarray(maxs, dtype=np.float32) if maxs is not None else None

        # Cached deterministic projection matrices keyed by telemetry input dimension
        self._projection_matrices: Dict[int, np.ndarray] = {}

    def extract_vector(
        self,
        telemetry: Union[Sequence[float], np.ndarray, Mapping[str, float]],
    ) -> Tuple[np.ndarray, Tuple[str, ...]]:
        """Extracts a 1D float32 vector and corresponding keys from input telemetry."""
        if isinstance(telemetry, Mapping):
            if self.feature_names is not None:
                keys = self.feature_names
                vals = [float(telemetry.get(k, 0.0)) for k in keys]
            else:
                keys = tuple(sorted(str(k) for k in telemetry.keys()))
                vals = [float(telemetry[k]) for k in keys]
            vec = np.asarray(vals, dtype=np.float32)
            return vec, keys

        if isinstance(telemetry, (list, tuple, np.ndarray)):
            arr = np.asarray(telemetry, dtype=np.float32).flatten()
            keys = tuple(f"f_{i}" for i in range(len(arr)))
            return arr, keys

        raise TypeError(
            f"Expected telemetry to be Mapping or Sequence of floats, got {type(telemetry).__name__}"
        )

    def normalize(
        self,
        telemetry: Union[Sequence[float], np.ndarray, Mapping[str, float]],
    ) -> np.ndarray:
        """Normalizes continuous telemetry vector according to configured strategy."""
        vec, _ = self.extract_vector(telemetry)
        if len(vec) == 0:
            return vec

        if self.normalization == "none":
            return vec

        if self.normalization == "robust":
            # Smoothly maps unbounded floats into (-1.0, 1.0)
            return np.tanh(vec).astype(np.float32)

        if self.normalization == "standard":
            mu = self.means if (self.means is not None and len(self.means) == len(vec)) else 0.0
            sigma = self.stds if (self.stds is not None and len(self.stds) == len(vec)) else 1.0
            sigma = np.where(np.abs(sigma) < 1e-6, 1.0, sigma)
            return ((vec - mu) / sigma).astype(np.float32)

        if self.normalization == "minmax":
            v_min = self.mins if (self.mins is not None and len(self.mins) == len(vec)) else 0.0
            v_max = self.maxs if (self.maxs is not None and len(self.maxs) == len(vec)) else 1.0
            denom = np.where(np.abs(v_max - v_min) < 1e-6, 1.0, v_max - v_min)
            return ((vec - v_min) / denom).astype(np.float32)

        return vec

    def _get_projection_matrix(self, in_dim: int) -> np.ndarray:
        """Deterministically constructs a pseudo-orthogonal projection matrix from in_dim to embedding_dim."""
        if in_dim in self._projection_matrices:
            return self._projection_matrices[in_dim]

        # Seed matrix deterministically per feature without relying on global PRNG state
        matrix = np.zeros((self.embedding_dim, in_dim), dtype=np.float32)
        for j in range(in_dim):
            seed_key = f"telemetry_proj_dim{self.embedding_dim}_feat{j}"
            seed_int = int.from_bytes(hashlib.sha256(seed_key.encode("utf-8")).digest()[:4], "big")
            rng = np.random.RandomState(seed_int)
            matrix[:, j] = rng.standard_normal(self.embedding_dim).astype(np.float32)

        # QR pseudo-orthonormalization along columns: guarantees full rank and orthonormal columns
        q, _ = np.linalg.qr(matrix)
        self._projection_matrices[in_dim] = q.astype(np.float32)
        return self._projection_matrices[in_dim]

    def project(
        self,
        telemetry: Union[Sequence[float], np.ndarray, Mapping[str, float]],
    ) -> np.ndarray:
        """Projects continuous telemetry into dense embedding dimension."""
        t_norm = self.normalize(telemetry)
        in_dim = len(t_norm)
        if in_dim == 0:
            v = np.zeros(self.embedding_dim, dtype=np.float32)
            v[0] = 1.0
            return v

        W = self._get_projection_matrix(in_dim)
        projected = W @ t_norm
        norm = float(np.linalg.norm(projected))
        if norm > 1e-12:
            return (projected / norm).astype(np.float32)
        v = np.zeros(self.embedding_dim, dtype=np.float32)
        v[0] = 1.0
        return v

    def fuse(
        self,
        semantic_emb: np.ndarray,
        telemetry: Optional[Union[Sequence[float], np.ndarray, Mapping[str, float]]] = None,
        mode: Optional[str] = None,
        preserve_dimension: bool = False,
    ) -> np.ndarray:
        """Fuses semantic embedding with continuous telemetry state."""
        sem = np.asarray(semantic_emb, dtype=np.float32).flatten()
        if telemetry is None:
            norm = float(np.linalg.norm(sem))
            return (sem / norm).astype(np.float32) if norm > 1e-12 else sem

        effective_mode = (mode or self.mode).lower().strip()
        if effective_mode in ("project", "project_add"):
            telem_vec = self.project(telemetry)
            alpha = self.fusion_weight
            fused = (1.0 - alpha) * sem + alpha * telem_vec
            norm = float(np.linalg.norm(fused))
            return (fused / norm).astype(np.float32) if norm > 1e-12 else fused

        if effective_mode in ("concat", "concatenate"):
            telem_norm = self.normalize(telemetry)
            scaled_telem = self.fusion_weight * telem_norm
            if preserve_dimension and len(scaled_telem) < len(sem):
                sem_part = sem[: len(sem) - len(scaled_telem)]
                fused = np.concatenate([sem_part, scaled_telem])
            else:
                fused = np.concatenate([sem, scaled_telem])
            if mode is not None and "concat" in mode and not preserve_dimension:
                # When unnormalized concatenation is requested, return fused array directly
                return fused.astype(np.float32)
            norm = float(np.linalg.norm(fused))
            return (fused / norm).astype(np.float32) if norm > 1e-12 else fused

        return sem


__all__ = [
    "TelemetryProjector",
]
