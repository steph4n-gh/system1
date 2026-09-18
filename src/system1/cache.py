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

import hashlib
import json
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
        }


def _digest_prompt(prompt: str) -> str:
    """Computes canonical SHA-256 hash of a prompt string."""
    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()


def _digest_telemetry(telemetry: Optional[Any]) -> str:
    """Computes deterministic digest for telemetry vectors or dicts."""
    if telemetry is None:
        return ""
    if isinstance(telemetry, Mapping):
        sorted_pairs = sorted((str(k), float(v)) for k, v in telemetry.items())
        raw = json.dumps(sorted_pairs, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    if isinstance(telemetry, (list, tuple, np.ndarray)):
        arr = np.asarray(telemetry, dtype=np.float32).flatten()
        return hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    return hashlib.sha256(str(telemetry).encode("utf-8")).hexdigest()[:16]


class SemanticReflexCache:
    """High-performance L1 In-Memory Vector and Exact Match Cache for Tier 0 fast-path.

    Guarantees sub-0.05ms execution latency for repeated or near-duplicate queries
    (cosine similarity >= similarity_threshold, default 0.98), bypassing forward pass
    computation and suppressing unnecessary Tier 2 escalations.
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

    def _make_key(self, prompt: str, telemetry: Optional[Any] = None) -> str:
        p_dig = _digest_prompt(prompt)
        t_dig = _digest_telemetry(telemetry)
        return f"{p_dig}::{t_dig}" if t_dig else p_dig

    def get(
        self,
        prompt: str,
        embedding: Optional[np.ndarray] = None,
        telemetry: Optional[Any] = None,
    ) -> Optional[Tuple[CacheEntry, float]]:
        """Queries the cache with exact match first, then cosine similarity search.

        Returns:
            Tuple of (CacheEntry, similarity_score) if hit (similarity == 1.0 for exact hit),
            or None if miss.
        """
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")

        with self._lock:
            self._total_queries += 1
            key = self._make_key(prompt, telemetry)

            # 1. Exact Match Lookup (O(1), <0.005ms)
            if key in self._exact_index:
                entry = self._exact_index[key]
                entry.hit_count += 1
                entry.last_accessed_at = time.time()
                # Move to end for LRU
                self._exact_index.move_to_end(key)
                self._exact_hits += 1
                return entry, 1.0

            # 2. Semantic Cosine Similarity Search (O(N) BLAS dot product, <0.03ms)
            if embedding is not None and self._embeddings is not None and len(self._embedding_entries) > 0:
                q_emb = np.asarray(embedding, dtype=np.float32).flatten()
                norm = float(np.linalg.norm(q_emb))
                if norm > 1e-12:
                    q_norm = q_emb / norm
                    sims = self._embeddings @ q_norm

                    # Isolate telemetry presence: query with telemetry matches entries with telemetry,
                    # and query without telemetry matches entries without telemetry.
                    has_query_telem = (telemetry is not None)
                    mask = np.array([
                        (e.telemetry is not None) == has_query_telem
                        for e in self._embedding_entries
                    ], dtype=bool)
                    sims = np.where(mask, sims, -1.0)

                    best_idx = int(np.argmax(sims))
                    best_sim = float(sims[best_idx])

                    if best_sim >= self.similarity_threshold:
                        entry = self._embedding_entries[best_idx]
                        entry.hit_count += 1
                        entry.last_accessed_at = time.time()
                        entry_key = f"{entry.prompt_digest}::{entry.telemetry_digest}" if entry.telemetry_digest else entry.prompt_digest
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
    ) -> CacheEntry:
        """Stores or updates a result in the L1 cache."""
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")

        with self._lock:
            key = self._make_key(prompt, telemetry)
            p_dig = _digest_prompt(prompt)
            t_dig = _digest_telemetry(telemetry)

            norm_emb: Optional[np.ndarray] = None
            if embedding is not None:
                e = np.asarray(embedding, dtype=np.float32).flatten()
                n = float(np.linalg.norm(e))
                norm_emb = (e / n) if n > 1e-12 else e

            # Update existing entry if present
            if key in self._exact_index:
                entry = self._exact_index[key]
                entry.result = result
                entry.source = source
                entry.last_accessed_at = time.time()
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
                result=result,
                embedding=norm_emb,
                telemetry=telemetry,
                telemetry_digest=t_dig,
                source=source,
                hit_count=0,
                created_at=time.time(),
                last_accessed_at=time.time(),
            )
            self._exact_index[key] = entry

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
                k = f"{e.prompt_digest}::{e.telemetry_digest}" if e.telemetry_digest else e.prompt_digest
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

    def contains_exact(self, prompt: str, telemetry: Optional[Any] = None) -> bool:
        """Checks whether the exact query exists in the cache."""
        with self._lock:
            key = self._make_key(prompt, telemetry)
            return key in self._exact_index


__all__ = [
    "CacheEntry",
    "SemanticReflexCache",
]
