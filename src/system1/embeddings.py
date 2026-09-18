"""System 1 Dense Subword Semantic Engine and Hybrid Projector.

Re-exports embeddings and hybrid projection primitives from system1.core.embeddings.
"""

from __future__ import annotations

from system1.core.embeddings import (
    HybridProjector,
    HybridSemanticProjector,
    SubwordSemanticEmbeddings,
)
from system1.core.embeddings import *
from system1.core.embeddings import __all__ as _embeddings_all

__all__ = list(_embeddings_all)
