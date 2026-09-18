"""System 1 Non-Autoregressive Decision Model.

Re-exports model evaluation kernel from system1.core.model.
"""

from __future__ import annotations

from system1.core.model import (
    HAS_MLX,
    DecisionFieldHead,
    DeterministicSemanticProjector,
    ModelInferenceResult,
    RawFieldEvaluation,
    SystemOneModel,
    _stable_sigmoid,
    _stable_softmax,
    stable_sigmoid,
    stable_softmax,
)
from system1.core.model import *
from system1.core.model import __all__ as _model_all

__all__ = list(_model_all)
