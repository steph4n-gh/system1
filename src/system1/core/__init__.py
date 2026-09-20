"""System 1 Core: Isolated Non-Autoregressive Forward Evaluator Kernel.

Isolated forward evaluation components:
- Multi-Head Parallel Linear Evaluator (SystemOneModel)
- Deterministic Semantic Projector (DeterministicSemanticProjector)
- Local Neural Semantic Projector (LocalNeuralProjector)
- Strongly-Typed Schema Primitives (DecisionSchema, ChoiceField, BooleanField, MultiChoiceField, ScoreField)

Requires ONLY NumPy (with optional Apple Silicon MLX Metal acceleration).
Has ZERO dependencies on cryptography (Ed25519 signing / receipts),
SQLite (ActionLedger), or disk I/O.
"""

from __future__ import annotations

from system1.core.schema import (
    BooleanField,
    ChoiceField,
    DecisionField,
    DecisionSchema,
    FieldDefinition,
    MultiChoiceField,
    ScoreField,
    SchemaMeta,
    _validate_field_name,
)
from system1.core.neural import (
    HAS_MLX as HAS_MLX_NEURAL,
    LocalNeuralProjector,
    _gelu,
    _layer_norm,
)
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

from system1.core.embeddings import (
    HybridProjector,
    HybridSemanticProjector,
    SubwordSemanticEmbeddings,
)

# Core submodules
from system1.core import (
    embeddings,
    model,
    neural,
    schema,
)

__version__ = "0.2.2"

__all__ = [
    # Schema primitives
    "DecisionSchema",
    "DecisionField",
    "FieldDefinition",
    "ChoiceField",
    "BooleanField",
    "MultiChoiceField",
    "ScoreField",
    "SchemaMeta",
    # Evaluation Kernel & Projectors
    "DeterministicSemanticProjector",
    "LocalNeuralProjector",
    "HybridProjector",
    "HybridSemanticProjector",
    "SubwordSemanticEmbeddings",
    "SystemOneModel",
    "DecisionFieldHead",
    "RawFieldEvaluation",
    "ModelInferenceResult",
    # Mathematical & Backend utilities
    "HAS_MLX",
    "stable_softmax",
    "stable_sigmoid",
]
