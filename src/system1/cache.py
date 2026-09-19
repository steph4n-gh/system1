"""Tier 0 Semantic Reflex Cache (L1 Vector & Exact Match Cache).

In-memory, sub-millisecond L1 cache for ReflexEngine and CompiledSystemOneModel:
1. Exact Match Index: O(1) hash table lookup for identical prompts/telemetry (<0.005ms).
2. Semantic Vector Index: Matrix-vector cosine similarity search over cached embeddings
   for near-identical edge cases (cosine similarity >= tau, default tau=0.98) (<0.03ms).
3. Certified Execution: Bypasses forward pass and conformal ambiguity halts for certified
   prior Tier 2 resolutions and high-confidence evaluations.
4. Pure NumPy & Standard Library: Zero external vector database or disk dependencies.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class CacheEntry:
    """An entry stored in the Tier 0 Semantic Reflex Cache."""

    prompt: str
    prompt_digest: str
    result: Any
    embedding: Optional[np.ndarray] = None
    telemetry: Optional[Any] = None
    telemetry_digest: str = ""
    source: str = "evaluation"  # "tier2", "evaluation", "manual"
    hit_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    schema_digest: str = ""
    model_version: int = 0
    policy_scope: str = ""
    alpha: float = 0.05
    margin_threshold: float = 0.0
    strict: bool = False
    context_digest: str = ""
    relative_odds_ratio: Optional[float] = None
    confidence_floor_tau0: Optional[float] = None
    recency_weighted: bool = False
    model_digest: str = ""
    projector_digest: str = ""
    calibration_digest: str = ""
    policy_epoch: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes cache entry metadata (without raw numpy embedding)."""
        res_dict = self.result
        if hasattr(self.result, "to_dict") and callable(self.result.to_dict):
            res_dict = self.result.to_dict()
        return {
            "prompt": self.prompt,
            "prompt_digest": self.prompt_digest,
            "source": self.source,
            "hit_count": self.hit_count,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "telemetry_digest": self.telemetry_digest,
            "has_embedding": self.embedding is not None,
            "schema_digest": self.schema_digest,
            "model_version": self.model_version,
            "policy_scope": self.policy_scope,
            "alpha": self.alpha,
            "margin_threshold": self.margin_threshold,
            "strict": self.strict,
            "context_digest": self.context_digest,
            "relative_odds_ratio": self.relative_odds_ratio,
            "confidence_floor_tau0": self.confidence_floor_tau0,
            "recency_weighted": self.recency_weighted,
            "model_digest": self.model_digest,
            "projector_digest": self.projector_digest,
            "calibration_digest": self.calibration_digest,
            "policy_epoch": self.policy_epoch,
        }


def _validate_cache_inputs(
    prompt: str,
    *,
    embedding: Optional[Any] = None,
    telemetry: Optional[Any] = None,
    alpha: Optional[float] = None,
    margin_threshold: Optional[float] = None,
    relative_odds_ratio: Optional[float] = None,
    confidence_floor_tau0: Optional[float] = None,
    odds_ratio: Optional[float] = None,
) -> None:
    """Validates decision and cache parameters prior to key formation or index traversal."""
    if not isinstance(prompt, str):
        raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")

    if alpha is not None:
        try:
            a_val = float(alpha)
        except (ValueError, TypeError) as ex:
            raise TypeError(f"alpha must be a real number, got {alpha}") from ex
        if math.isnan(a_val) or not (0.0 < a_val < 1.0):
            raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")

    if margin_threshold is not None:
        try:
            m_val = float(margin_threshold)
        except (ValueError, TypeError) as ex:
            raise TypeError(f"margin_threshold must be a real number, got {margin_threshold}") from ex
        if math.isnan(m_val) or m_val < 0.0:
            raise ValueError(f"margin_threshold must be non-negative, got {margin_threshold}")

    eff_odds = odds_ratio if odds_ratio is not None else relative_odds_ratio
    if eff_odds is not None:
        try:
            o_val = float(eff_odds)
        except (ValueError, TypeError) as ex:
            raise TypeError(f"relative_odds_ratio must be a real number, got {eff_odds}") from ex
        if math.isnan(o_val) or o_val <= 0.0:
            raise ValueError(f"relative_odds_ratio must be positive, got {eff_odds}")

    if confidence_floor_tau0 is not None:
        try:
            c_val = float(confidence_floor_tau0)
        except (ValueError, TypeError) as ex:
            raise TypeError(f"confidence_floor_tau0 must be a real number, got {confidence_floor_tau0}") from ex
        if math.isnan(c_val) or c_val < 0.0:
            raise ValueError(f"confidence_floor_tau0 must be non-negative, got {confidence_floor_tau0}")

    if telemetry is not None:
        if not isinstance(telemetry, (Mapping, Sequence, np.ndarray)):
            raise TypeError(f"telemetry must be a Mapping, Sequence, or ndarray, got {type(telemetry).__name__}")
        if isinstance(telemetry, np.ndarray) and not np.all(np.isfinite(telemetry)):
            raise ValueError("telemetry array contains NaN or infinite values")

    if embedding is not None:
        if not isinstance(embedding, (np.ndarray, Sequence)):
            raise TypeError(f"embedding must be a sequence or ndarray, got {type(embedding).__name__}")
        emb_arr = np.asarray(embedding, dtype=np.float32)
        if emb_arr.ndim > 2 or (emb_arr.ndim == 2 and emb_arr.shape[0] != 1 and emb_arr.shape[1] != 1):
            raise ValueError(f"embedding must be a 1D vector, got shape {emb_arr.shape}")
        if not np.all(np.isfinite(emb_arr)):
            raise ValueError("embedding contains NaN or infinite values")


def _digest_prompt(prompt: str) -> str:
    """Computes canonical SHA-256 hash of a prompt string."""
    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()


def _digest_telemetry(telemetry: Optional[Any]) -> str:
    """Computes deterministic 64-hex SHA-256 digest for telemetry vectors or dicts."""
    if telemetry is None:
        return ""
    if isinstance(telemetry, Mapping):
        def _canonical_val(val: Any) -> Any:
            try:
                return float(val)
            except (ValueError, TypeError):
                return str(val)
        sorted_pairs = sorted((str(k), _canonical_val(v)) for k, v in telemetry.items())
        raw = json.dumps(sorted_pairs, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if isinstance(telemetry, (list, tuple, np.ndarray)):
        arr = np.asarray(telemetry, dtype=np.float32).flatten()
        return hashlib.sha256(arr.tobytes()).hexdigest()
    return hashlib.sha256(str(telemetry).encode("utf-8")).hexdigest()


def _format_context(
    schema_digest: str = "",
    model_version: int = 0,
    policy_scope: str = "",
    alpha: Optional[float] = 0.05,
    margin_threshold: Optional[float] = 0.0,
    strict: bool = False,
    relative_odds_ratio: Optional[float] = None,
    odds_ratio: Optional[float] = None,
    confidence_floor_tau0: Optional[float] = None,
    recency_weighted: Optional[bool] = None,
    model_digest: str = "",
    projector_digest: str = "",
    calibration_digest: str = "",
    policy_epoch: int = 0,
) -> str:
    """Formats execution context string for collision-resistant cache key isolation."""
    a_val = float(alpha) if alpha is not None else 0.05
    m_val = float(margin_threshold) if margin_threshold is not None else 0.0
    eff_odds = odds_ratio if odds_ratio is not None else relative_odds_ratio
    o_val = float(eff_odds) if eff_odds is not None else 0.0
    tau0_val = float(confidence_floor_tau0) if confidence_floor_tau0 is not None else 0.0
    rec_val = 1 if recency_weighted else 0
    st_val = 1 if strict else 0
    return (
        f"s:{schema_digest}|v:{model_version}|md:{model_digest}|pd:{projector_digest}|"
        f"cd:{calibration_digest}|sc:{policy_scope}|pe:{policy_epoch}|a:{float.hex(a_val)}|"
        f"m:{float.hex(m_val)}|or:{float.hex(o_val)}|cf:{float.hex(tau0_val)}|rw:{rec_val}|st:{st_val}"
    )


class SemanticReflexCache:
    """Sub-0.05ms exact and semantic L1 cache for Reflex decision outputs.

    Provides exact SHA-256 hash lookup (<0.005ms) with fallback to
    cosine similarity search (<0.03ms) over dense semantic embeddings.
    Thread-safe with RLock and LRU eviction.
    """

    def __init__(
        self,
        capacity: int = 2048,
        similarity_threshold: float = 0.98,
        eviction_policy: str = "lru",
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if not (0.0 < similarity_threshold <= 1.0):
            raise ValueError(f"similarity_threshold must be in (0, 1], got {similarity_threshold}")

        self.capacity = int(capacity)
        self.similarity_threshold = float(similarity_threshold)
        self.eviction_policy = eviction_policy

        self._lock = threading.RLock()
        # Exact match index: full_key -> CacheEntry (OrderedDict for LRU)
        self._exact_index: OrderedDict[str, CacheEntry] = OrderedDict()
        self._base_to_key: Dict[str, str] = {}

        # Semantic index: parallel array of normalized embeddings and entries
        self._embeddings: Optional[np.ndarray] = None  # shape: (N, D)
        self._embedding_entries: List[CacheEntry] = []
        self._key_to_emb_idx: Dict[str, int] = {}

        # Counters
        self._total_queries: int = 0
        self._exact_hits: int = 0
        self._semantic_hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    def _make_key(
        self,
        prompt: str,
        telemetry: Optional[Any] = None,
        embedding: Optional[np.ndarray] = None,
        schema_digest: str = "",
        model_version: int = 0,
        policy_scope: str = "",
        alpha: Optional[float] = 0.05,
        margin_threshold: Optional[float] = 0.0,
        strict: bool = False,
        relative_odds_ratio: Optional[float] = None,
        odds_ratio: Optional[float] = None,
        confidence_floor_tau0: Optional[float] = None,
        recency_weighted: Optional[bool] = None,
        model_digest: str = "",
        projector_digest: str = "",
        calibration_digest: str = "",
        policy_epoch: int = 0,
    ) -> str:
        p_dig = _digest_prompt(prompt)
        t_dig = _digest_telemetry(telemetry)
        e_dig = hashlib.sha256(np.asarray(embedding, dtype=np.float32).tobytes()).hexdigest() if embedding is not None else ""
        ctx = _format_context(
            schema_digest=schema_digest,
            model_version=model_version,
            policy_scope=policy_scope,
            alpha=alpha,
            margin_threshold=margin_threshold,
            strict=strict,
            relative_odds_ratio=relative_odds_ratio,
            odds_ratio=odds_ratio,
            confidence_floor_tau0=confidence_floor_tau0,
            recency_weighted=recency_weighted,
            model_digest=model_digest,
            projector_digest=projector_digest,
            calibration_digest=calibration_digest,
            policy_epoch=policy_epoch,
        )
        base = f"{p_dig}::{t_dig}::{ctx}" if t_dig else f"{p_dig}::{ctx}"
        return f"{base}::e:{e_dig}" if e_dig else base

    def get(
        self,
        prompt: str,
        embedding: Optional[np.ndarray] = None,
        telemetry: Optional[Any] = None,
        schema_digest: str = "",
        model_version: int = 0,
        policy_scope: str = "",
        alpha: Optional[float] = 0.05,
        margin_threshold: Optional[float] = 0.0,
        strict: bool = False,
        relative_odds_ratio: Optional[float] = None,
        odds_ratio: Optional[float] = None,
        confidence_floor_tau0: Optional[float] = None,
        recency_weighted: Optional[bool] = None,
        model_digest: str = "",
        projector_digest: str = "",
        calibration_digest: str = "",
        policy_epoch: int = 0,
        enforce_durability: bool = False,
    ) -> Optional[Tuple[CacheEntry, float]]:
        """Queries the cache with exact match first, then cosine similarity search.

        Pre-validates all inputs before forming keys or querying indices.
        Disables semantic cosine search when strict or enforce_durability is True.

        Returns:
            Tuple of (CacheEntry, similarity_score) if hit (similarity == 1.0 for exact hit),
            or None if miss.
        """
        _validate_cache_inputs(
            prompt,
            embedding=embedding,
            telemetry=telemetry,
            alpha=alpha,
            margin_threshold=margin_threshold,
            relative_odds_ratio=relative_odds_ratio,
            confidence_floor_tau0=confidence_floor_tau0,
            odds_ratio=odds_ratio,
        )

        with self._lock:
            self._total_queries += 1
            key = self._make_key(
                prompt,
                telemetry=telemetry,
                embedding=embedding,
                schema_digest=schema_digest,
                model_version=model_version,
                policy_scope=policy_scope,
                alpha=alpha,
                margin_threshold=margin_threshold,
                strict=strict,
                relative_odds_ratio=relative_odds_ratio,
                odds_ratio=odds_ratio,
                confidence_floor_tau0=confidence_floor_tau0,
                recency_weighted=recency_weighted,
                model_digest=model_digest,
                projector_digest=projector_digest,
                calibration_digest=calibration_digest,
                policy_epoch=policy_epoch,
            )

            # 1. Exact Match Lookup (O(1), <0.005ms)
            if key in self._exact_index:
                entry = self._exact_index[key]
                entry.hit_count += 1
                entry.last_accessed_at = time.time()
                # Move to end for LRU
                self._exact_index.move_to_end(key)
                self._exact_hits += 1
                return entry, 1.0
            elif embedding is None and key in self._base_to_key:
                real_key = self._base_to_key[key]
                if real_key in self._exact_index:
                    entry = self._exact_index[real_key]
                    entry.hit_count += 1
                    entry.last_accessed_at = time.time()
                    self._exact_index.move_to_end(real_key)
                    self._exact_hits += 1
                    return entry, 1.0

            # 2. Semantic Cosine Similarity Search (O(N) BLAS dot product, <0.03ms)
            # Suppressed in strict or enforcement mode to ensure exact, collision-resistant deterministic retrieval
            if not (strict or enforce_durability) and embedding is not None and self._embeddings is not None and len(self._embedding_entries) > 0:
                q_emb = np.asarray(embedding, dtype=np.float32).flatten()
                norm = float(np.linalg.norm(q_emb))
                if norm > 1e-12:
                    q_norm = q_emb / norm
                    sims = self._embeddings @ q_norm

                    # Isolate telemetry presence and execution context
                    has_query_telem = (telemetry is not None)
                    ctx = _format_context(
                        schema_digest=schema_digest,
                        model_version=model_version,
                        policy_scope=policy_scope,
                        alpha=alpha,
                        margin_threshold=margin_threshold,
                        strict=strict,
                        relative_odds_ratio=relative_odds_ratio,
                        odds_ratio=odds_ratio,
                        confidence_floor_tau0=confidence_floor_tau0,
                        recency_weighted=recency_weighted,
                        model_digest=model_digest,
                        projector_digest=projector_digest,
                        calibration_digest=calibration_digest,
                        policy_epoch=policy_epoch,
                    )
                    mask = np.array([
                        ((e.telemetry is not None) == has_query_telem)
                        and (getattr(e, "context_digest", "") == ctx or not getattr(e, "context_digest", ""))
                        for e in self._embedding_entries
                    ], dtype=bool)
                    sims = np.where(mask, sims, -1.0)

                    best_idx = int(np.argmax(sims))
                    best_sim = float(sims[best_idx])

                    if best_sim >= self.similarity_threshold:
                        entry = self._embedding_entries[best_idx]
                        entry.hit_count += 1
                        entry.last_accessed_at = time.time()
                        entry_key = self._make_key(
                            entry.prompt,
                            telemetry=entry.telemetry,
                            embedding=entry.embedding,
                            schema_digest=entry.schema_digest,
                            model_version=entry.model_version,
                            policy_scope=entry.policy_scope,
                            alpha=entry.alpha,
                            margin_threshold=entry.margin_threshold,
                            strict=entry.strict,
                            relative_odds_ratio=entry.relative_odds_ratio,
                            confidence_floor_tau0=entry.confidence_floor_tau0,
                            recency_weighted=entry.recency_weighted,
                            model_digest=entry.model_digest,
                            projector_digest=entry.projector_digest,
                            calibration_digest=entry.calibration_digest,
                            policy_epoch=entry.policy_epoch,
                        )
                        if entry_key in self._exact_index:
                            self._exact_index.move_to_end(entry_key)
                        self._semantic_hits += 1
                        return entry, best_sim

            self._misses += 1
            return None

    def put(
        self,
        prompt: str,
        result: Any,
        embedding: Optional[np.ndarray] = None,
        telemetry: Optional[Any] = None,
        source: str = "evaluation",
        schema_digest: str = "",
        model_version: int = 0,
        policy_scope: str = "",
        alpha: Optional[float] = 0.05,
        margin_threshold: Optional[float] = 0.0,
        strict: bool = False,
        relative_odds_ratio: Optional[float] = None,
        odds_ratio: Optional[float] = None,
        confidence_floor_tau0: Optional[float] = None,
        recency_weighted: Optional[bool] = None,
        model_digest: str = "",
        projector_digest: str = "",
        calibration_digest: str = "",
        policy_epoch: int = 0,
        explicit_embedding: bool = False,
    ) -> CacheEntry:
        """Stores or updates a result in the L1 cache."""
        _validate_cache_inputs(
            prompt,
            embedding=embedding,
            telemetry=telemetry,
            alpha=alpha,
            margin_threshold=margin_threshold,
            relative_odds_ratio=relative_odds_ratio,
            confidence_floor_tau0=confidence_floor_tau0,
            odds_ratio=odds_ratio,
        )

        with self._lock:
            key = self._make_key(
                prompt,
                telemetry=telemetry,
                embedding=embedding,
                schema_digest=schema_digest,
                model_version=model_version,
                policy_scope=policy_scope,
                alpha=alpha,
                margin_threshold=margin_threshold,
                strict=strict,
                relative_odds_ratio=relative_odds_ratio,
                odds_ratio=odds_ratio,
                confidence_floor_tau0=confidence_floor_tau0,
                recency_weighted=recency_weighted,
                model_digest=model_digest,
                projector_digest=projector_digest,
                calibration_digest=calibration_digest,
                policy_epoch=policy_epoch,
            )
            p_dig = _digest_prompt(prompt)
            t_dig = _digest_telemetry(telemetry)
            ctx = _format_context(
                schema_digest=schema_digest,
                model_version=model_version,
                policy_scope=policy_scope,
                alpha=alpha,
                margin_threshold=margin_threshold,
                strict=strict,
                relative_odds_ratio=relative_odds_ratio,
                odds_ratio=odds_ratio,
                confidence_floor_tau0=confidence_floor_tau0,
                recency_weighted=recency_weighted,
                model_digest=model_digest,
                projector_digest=projector_digest,
                calibration_digest=calibration_digest,
                policy_epoch=policy_epoch,
            )
            a_val = float(alpha) if alpha is not None else 0.05
            m_val = float(margin_threshold) if margin_threshold is not None else 0.0
            eff_odds = odds_ratio if odds_ratio is not None else relative_odds_ratio
            o_val = float(eff_odds) if eff_odds is not None else None
            tau0_val = float(confidence_floor_tau0) if confidence_floor_tau0 is not None else None

            safe_result = copy.deepcopy(result)

            norm_emb: Optional[np.ndarray] = None
            if embedding is not None:
                e = np.asarray(embedding, dtype=np.float32).flatten()
                n = float(np.linalg.norm(e))
                norm_emb = (e / n) if n > 1e-12 else e

            # Update existing entry if present
            if key in self._exact_index:
                entry = self._exact_index[key]
                entry.result = safe_result
                entry.source = source
                entry.last_accessed_at = time.time()
                entry.schema_digest = schema_digest
                entry.model_version = model_version
                entry.policy_scope = policy_scope
                entry.alpha = a_val
                entry.margin_threshold = m_val
                entry.strict = bool(strict)
                entry.context_digest = ctx
                entry.relative_odds_ratio = o_val
                entry.confidence_floor_tau0 = tau0_val
                entry.recency_weighted = bool(recency_weighted)
                entry.model_digest = model_digest
                entry.projector_digest = projector_digest
                entry.calibration_digest = calibration_digest
                entry.policy_epoch = int(policy_epoch)
                if norm_emb is not None:
                    entry.embedding = norm_emb
                    if key in self._key_to_emb_idx and self._embeddings is not None:
                        idx = self._key_to_emb_idx[key]
                        self._embeddings[idx] = norm_emb
                    else:
                        if self._embeddings is None:
                            self._embeddings = norm_emb.reshape(1, -1)
                        else:
                            self._embeddings = np.vstack([self._embeddings, norm_emb.reshape(1, -1)])
                        new_idx = len(self._embedding_entries)
                        self._embedding_entries.append(entry)
                        self._key_to_emb_idx[key] = new_idx
                self._exact_index.move_to_end(key)
                return entry

            # Evict LRU entry if at capacity
            if len(self._exact_index) >= self.capacity:
                self._evict_one()

            entry = CacheEntry(
                prompt=prompt,
                prompt_digest=p_dig,
                result=safe_result,
                embedding=norm_emb,
                telemetry=copy.deepcopy(telemetry),
                telemetry_digest=t_dig,
                source=source,
                hit_count=0,
                created_at=time.time(),
                last_accessed_at=time.time(),
                schema_digest=schema_digest,
                model_version=model_version,
                policy_scope=policy_scope,
                alpha=a_val,
                margin_threshold=m_val,
                strict=bool(strict),
                context_digest=ctx,
                relative_odds_ratio=o_val,
                confidence_floor_tau0=tau0_val,
                recency_weighted=bool(recency_weighted),
                model_digest=model_digest,
                projector_digest=projector_digest,
                calibration_digest=calibration_digest,
                policy_epoch=int(policy_epoch),
            )
            self._exact_index[key] = entry
            if norm_emb is not None and not explicit_embedding:
                base_key = self._make_key(
                    prompt,
                    telemetry=telemetry,
                    embedding=None,
                    schema_digest=schema_digest,
                    model_version=model_version,
                    policy_scope=policy_scope,
                    alpha=alpha,
                    margin_threshold=margin_threshold,
                    strict=strict,
                    relative_odds_ratio=relative_odds_ratio,
                    odds_ratio=odds_ratio,
                    confidence_floor_tau0=confidence_floor_tau0,
                    recency_weighted=recency_weighted,
                    model_digest=model_digest,
                    projector_digest=projector_digest,
                    calibration_digest=calibration_digest,
                    policy_epoch=policy_epoch,
                )
                self._base_to_key[base_key] = key

            # Append to semantic vector index if embedding present
            if norm_emb is not None:
                if self._embeddings is None:
                    self._embeddings = norm_emb.reshape(1, -1)
                else:
                    self._embeddings = np.vstack([self._embeddings, norm_emb.reshape(1, -1)])
                new_idx = len(self._embedding_entries)
                self._embedding_entries.append(entry)
                self._key_to_emb_idx[key] = new_idx

            return entry

    def _remove_key(self, key: str) -> bool:
        """Removes a key from exact and vector indices."""
        if key not in self._exact_index:
            return False
        entry = self._exact_index.pop(key)
        self._evictions += 1
        if key in self._key_to_emb_idx:
            idx = self._key_to_emb_idx.pop(key)
            self._embedding_entries.pop(idx)
            if self._embeddings is not None:
                if len(self._embedding_entries) == 0:
                    self._embeddings = None
                else:
                    self._embeddings = np.delete(self._embeddings, idx, axis=0)
            self._key_to_emb_idx.clear()
            for i, e in enumerate(self._embedding_entries):
                k = self._make_key(
                    e.prompt,
                    telemetry=e.telemetry,
                    embedding=e.embedding,
                    schema_digest=e.schema_digest,
                    model_version=e.model_version,
                    policy_scope=e.policy_scope,
                    alpha=e.alpha,
                    margin_threshold=e.margin_threshold,
                    strict=e.strict,
                    relative_odds_ratio=e.relative_odds_ratio,
                    confidence_floor_tau0=e.confidence_floor_tau0,
                    recency_weighted=e.recency_weighted,
                    model_digest=e.model_digest,
                    projector_digest=e.projector_digest,
                    calibration_digest=e.calibration_digest,
                    policy_epoch=e.policy_epoch,
                )
                self._key_to_emb_idx[k] = i
        return True

    def evict_prompt(self, prompt: str) -> int:
        """Evicts all cache entries matching prompt regardless of execution context."""
        with self._lock:
            p_dig = _digest_prompt(prompt)
            keys_to_remove = [
                k for k, entry in self._exact_index.items()
                if entry.prompt_digest == p_dig or entry.prompt.strip() == prompt.strip()
            ]
            for k in keys_to_remove:
                self._remove_key(k)
            return len(keys_to_remove)

    def invalidate_prior_versions(self, min_version: int) -> int:
        """Invalidates all entries with model_version < min_version."""
        with self._lock:
            keys_to_remove = [
                k for k, entry in self._exact_index.items()
                if entry.model_version < min_version
            ]
            for k in keys_to_remove:
                self._remove_key(k)
            return len(keys_to_remove)

    def invalidate_version(self, version: int) -> int:
        """Invalidates all entries with model_version == version."""
        with self._lock:
            keys_to_remove = [
                k for k, entry in self._exact_index.items()
                if entry.model_version == version
            ]
            for k in keys_to_remove:
                self._remove_key(k)
            return len(keys_to_remove)

    def _evict_one(self) -> None:
        """Evicts the least recently used entry from exact and vector indices."""
        oldest_key, oldest_entry = self._exact_index.popitem(last=False)
        self._evictions += 1

        if oldest_key in self._key_to_emb_idx:
            idx = self._key_to_emb_idx.pop(oldest_key)
            # Remove from _embedding_entries and _embeddings
            self._embedding_entries.pop(idx)
            if self._embeddings is not None:
                if len(self._embedding_entries) == 0:
                    self._embeddings = None
                else:
                    self._embeddings = np.delete(self._embeddings, idx, axis=0)

            # Re-index remaining entries
            self._key_to_emb_idx.clear()
            for i, e in enumerate(self._embedding_entries):
                k = self._make_key(
                    e.prompt,
                    telemetry=e.telemetry,
                    embedding=e.embedding,
                    schema_digest=e.schema_digest,
                    model_version=e.model_version,
                    policy_scope=e.policy_scope,
                    alpha=e.alpha,
                    margin_threshold=e.margin_threshold,
                    strict=e.strict,
                )
                self._key_to_emb_idx[k] = i

    def clear(self) -> None:
        """Clears all entries and resets statistics."""
        with self._lock:
            self._exact_index.clear()
            self._embeddings = None
            self._embedding_entries.clear()
            self._key_to_emb_idx.clear()
            self._total_queries = 0
            self._exact_hits = 0
            self._semantic_hits = 0
            self._misses = 0
            self._evictions = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._exact_index)

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        return len(self)

    def stats(self) -> Dict[str, Any]:
        """Returns diagnostic cache performance statistics."""
        with self._lock:
            total_hits = self._exact_hits + self._semantic_hits
            hit_rate = (total_hits / self._total_queries) if self._total_queries > 0 else 0.0
            return {
                "size": len(self._exact_index),
                "capacity": self.capacity,
                "vector_entries": len(self._embedding_entries),
                "total_queries": self._total_queries,
                "exact_hits": self._exact_hits,
                "semantic_hits": self._semantic_hits,
                "total_hits": total_hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": round(hit_rate, 4),
                "similarity_threshold": self.similarity_threshold,
            }

    def contains_exact(
        self,
        prompt: str,
        embedding: Optional[np.ndarray] = None,
        telemetry: Optional[Any] = None,
        schema_digest: str = "",
        model_version: int = 0,
        policy_scope: str = "",
        alpha: Optional[float] = 0.05,
        margin_threshold: Optional[float] = 0.0,
        strict: bool = False,
        relative_odds_ratio: Optional[float] = None,
        odds_ratio: Optional[float] = None,
        confidence_floor_tau0: Optional[float] = None,
        recency_weighted: Optional[bool] = None,
        model_digest: str = "",
        projector_digest: str = "",
        calibration_digest: str = "",
        policy_epoch: int = 0,
    ) -> bool:
        """Checks whether the exact query exists in the cache."""
        with self._lock:
            key = self._make_key(
                prompt,
                telemetry=telemetry,
                embedding=embedding,
                schema_digest=schema_digest,
                model_version=model_version,
                policy_scope=policy_scope,
                alpha=alpha,
                margin_threshold=margin_threshold,
                strict=strict,
                relative_odds_ratio=relative_odds_ratio,
                odds_ratio=odds_ratio,
                confidence_floor_tau0=confidence_floor_tau0,
                recency_weighted=recency_weighted,
                model_digest=model_digest,
                projector_digest=projector_digest,
                calibration_digest=calibration_digest,
                policy_epoch=policy_epoch,
            )
            return key in self._exact_index


validate_cache_inputs = _validate_cache_inputs

__all__ = [
    "CacheEntry",
    "SemanticReflexCache",
    "_validate_cache_inputs",
    "validate_cache_inputs",
]
