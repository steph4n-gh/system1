"""System 1 Local Neural Projector.

Re-exports local neural semantic projector from system1.core.neural.
"""

from __future__ import annotations

from system1.core.neural import (
    HAS_MLX,
    LocalNeuralProjector,
    _gelu,
    _layer_norm,
)
from system1.core.neural import *
from system1.core.neural import __all__ as _neural_all

__all__ = list(_neural_all)
