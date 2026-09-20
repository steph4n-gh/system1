"""Small, portable TF-IDF features learned from teaching text; NumPy only."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np

# Match word unigram/bigram TF-IDF with smoothed IDF and sublinear term frequency.
_TOKEN = re.compile(r"(?u)\b\w\w+\b")
MAX_FEATURES = 4096
MAX_TERM_LENGTH = 128
MAX_TEXT_LENGTH = 8192


def _terms(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("TF-IDF input must be text")
    words = _TOKEN.findall(text[:MAX_TEXT_LENGTH].lower())
    return [term for term in words + [f"{a} {b}" for a, b in zip(words, words[1:])]
            if len(term) <= MAX_TERM_LENGTH]


@dataclass(frozen=True, init=False, repr=False, eq=False)
class TfidfProjector:
    """Frozen word unigram/bigram features fitted only on teaching examples.

    Fit before compiling, then supply separate calibration examples. The vocabulary
    and IDF values travel with the saved skill. New vocabulary requires a new skill;
    ``project`` never updates the feature space or contacts a teacher.
    """

    _terms: tuple[str, ...]
    _idf: tuple[float, ...]
    _vocabulary: Mapping[str, int]
    _digest: str

    def __init__(self, terms: Sequence[str], idf: Sequence[float]) -> None:
        if not isinstance(terms, (list, tuple)) or not 0 < len(terms) <= MAX_FEATURES:
            raise ValueError("TF-IDF vocabulary must contain 1–4096 terms")
        if any(not isinstance(term, str) or not term or len(term) > MAX_TERM_LENGTH for term in terms):
            raise ValueError("TF-IDF terms must be bounded nonempty strings")
        if len(set(terms)) != len(terms):
            raise ValueError("TF-IDF terms must be unique")
        if not isinstance(idf, (list, tuple)) or len(idf) != len(terms):
            raise ValueError("TF-IDF weights must match the vocabulary")
        if any(type(value) not in (int, float) or not math.isfinite(value) or not 1 <= value <= 100 for value in idf):
            raise ValueError("TF-IDF weights must be finite values between 1 and 100")
        object.__setattr__(self, "_terms", tuple(terms))
        object.__setattr__(self, "_idf", tuple(float(value) for value in idf))
        object.__setattr__(self, "_vocabulary", MappingProxyType({term: i for i, term in enumerate(terms)}))
        object.__setattr__(self, "_digest", hashlib.sha256(json.dumps(self.to_config(), sort_keys=True).encode()).hexdigest())

    @property
    def dimension(self) -> int:
        return len(self._terms)

    @classmethod
    def fit(cls, texts: Iterable[str], *, max_features: int = 2048) -> TfidfProjector:
        """Choose frequent terms, retaining deterministic lexicographic tie breaks."""
        if type(max_features) is not int or not 1 <= max_features <= MAX_FEATURES:
            raise ValueError("max_features must be an integer between 1 and 4096")
        if isinstance(texts, str):
            raise TypeError("Pass a collection of teaching texts")
        frequency, document_frequency = Counter(), Counter()
        count = 0
        for text in texts:
            terms = _terms(text)
            frequency.update(terms)
            document_frequency.update(set(terms))
            count += 1
        if not frequency:
            raise ValueError("Teaching texts must contain word features")
        selected = sorted(sorted(frequency, key=lambda term: (-frequency[term], term))[:max_features])
        return cls(selected, [math.log((1 + count) / (1 + document_frequency[term])) + 1 for term in selected])

    def project(self, text: str, recency_weighted: bool | None = None) -> np.ndarray:
        if recency_weighted:
            raise ValueError("TF-IDF uses document frequencies, not recency weighting")
        vector = np.zeros(self.dimension, dtype=np.float32)
        for term, count in Counter(_terms(text)).items():
            index = self._vocabulary.get(term)
            if index is not None:
                vector[index] = (1 + math.log(count)) * self._idf[index]
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    def project_batch(self, texts: Sequence[str], recency_weighted: bool | None = None) -> np.ndarray:
        return np.stack([self.project(text, recency_weighted) for text in texts]) if len(texts) else np.empty((0, self.dimension), dtype=np.float32)

    def projector_digest(self) -> str:
        return self._digest

    def to_config(self) -> dict:
        return {"type": "tfidf", "terms": list(self._terms), "idf": list(self._idf)}

    @classmethod
    def from_config(cls, config: dict) -> TfidfProjector:
        if not isinstance(config, dict) or set(config) != {"type", "terms", "idf"} or config["type"] != "tfidf":
            raise ValueError("Unsupported TF-IDF settings")
        return cls(config["terms"], config["idf"])
