"""System 1: Machine-Native Decision Runtime (Reflex).

High-performance, non-autoregressive decision engine implementing Daniel Kahneman's System 1
fast cognitive layer, designed to pair with deliberate System 2 governors (Astra, Fable, Gemini, or Grok).
Symmetrically available as both `import system1` and `import reflex`.
"""

from __future__ import annotations

import sys
from typing import Any, Mapping, Optional, Sequence, Type, Union

# Core mathematical kernel and schema primitives (Zero crypto, Zero SQLite, pure NumPy)
from system1.core import (
    BooleanField,
    ChoiceField,
    DecisionField,
    DecisionFieldHead,
    DecisionSchema,
    DeterministicSemanticProjector,
    HybridProjector,
    HybridSemanticProjector,
    LocalNeuralProjector,
    ModelInferenceResult,
    MultiChoiceField,
    RawFieldEvaluation,
    ScoreField,
    SubwordSemanticEmbeddings,
    SystemOneModel,
    SchemaMeta,
)

# Calibration & Conformal Prediction primitives (pure NumPy, zero crypto, zero SQLite)
from system1.calibration import (
    BrierDecomposition,
    CalibrationMetrics,
    ConformalPredictionSet,
    ConformalPredictor,
    DecisionCalibrator,
    RegressionConformalInterval,
    RegressionConformalPredictor,
)

# Eager lightweight submodules
from system1 import (
    calibration,
    core,
    embeddings,
    model,
    neural,
    schema,
)

__version__ = "0.1.0"


def decide(
    prompt: str,
    schema: Union[DecisionSchema, Type[DecisionSchema]],
    *,
    telemetry: Optional[Any] = None,
    alpha: float = 0.05,
    record_receipt: bool = True,
    dimension: int = 384,
    backend: str = "auto",
    projector: Optional[Any] = None,
    margin_threshold: Optional[float] = None,
) -> Any:
    """One-liner functional API for evaluating a System 1 decision against a typed schema.

    Example:
        import system1

        class RoutingSchema(system1.DecisionSchema):
            route = system1.ChoiceField(options=["sales", "support", "billing"])
            is_escalation = system1.BooleanField()

        result = system1.decide("Customer invoice payment dispute", schema=RoutingSchema)
        print(result.route)
        print(result.confidences["route"])
        print(result.conformal_sets["route"])
    """
    if not isinstance(prompt, str):
        raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")
    from system1.engine import ReflexEngine
    engine_instance = ReflexEngine(
        schema,
        dimension=dimension,
        backend=backend,
        projector=projector,
        margin_threshold=margin_threshold,
    )
    return engine_instance.decide(
        prompt,
        telemetry=telemetry,
        alpha=alpha,
        record_receipt=record_receipt,
        margin_threshold=margin_threshold,
    )


evaluate = decide


_SUBMODULE_NAMES = {
    "cache",
    "calibration",
    "cli",
    "compat",
    "compiler",
    "core",
    "embeddings",
    "engine",
    "guard",
    "integrations",
    "ledger",
    "model",
    "neural",
    "receipt",
    "schema",
    "telemetry",
}

# Lazy-loaded governance modules and symbols (cryptography, SQLite, reference monitor, compat)
_MODULE_MAP = {
    # 16 submodules
    "cache": "system1.cache",
    "calibration": "system1.calibration",
    "cli": "system1.cli",
    "compat": "system1.compat",
    "compiler": "system1.compiler",
    "core": "system1.core",
    "embeddings": "system1.embeddings",
    "engine": "system1.engine",
    "guard": "system1.guard",
    "integrations": "system1.integrations",
    "ledger": "system1.ledger",
    "model": "system1.model",
    "neural": "system1.neural",
    "receipt": "system1.receipt",
    "schema": "system1.schema",
    "telemetry": "system1.telemetry",
    # cache (Lever 1)
    "SemanticReflexCache": "system1.cache",
    "CacheEntry": "system1.cache",
    # telemetry (Lever 4)
    "TelemetryProjector": "system1.telemetry",
    # engine
    "ReflexEngine": "system1.engine",
    "SystemOneEngine": "system1.engine",
    "System1Engine": "system1.engine",
    "DecisionResult": "system1.engine",
    "BenchmarkReport": "system1.engine",
    # receipt
    "DecisionWitnessReceipt": "system1.receipt",
    "create_decision_receipt": "system1.receipt",
    "verify_decision_witness_receipt": "system1.receipt",
    "compute_receipt_digest": "system1.receipt",
    "RunWitnessEnvelope": "system1.receipt",
    "create_run_witness_envelope": "system1.receipt",
    "sign_run_witness_envelope": "system1.receipt",
    "verify_run_witness_envelope": "system1.receipt",
    "canonical_json": "system1.receipt",
    "canonical_bytes": "system1.receipt",
    "fingerprint": "system1.receipt",
    "public_key_bytes": "system1.receipt",
    "public_key_fingerprint": "system1.receipt",
    "load_private_key": "system1.receipt",
    "load_public_key": "system1.receipt",
    "save_keypair": "system1.receipt",
    "save_private_key": "system1.receipt",
    "save_public_key": "system1.receipt",
    "sign_payload": "system1.receipt",
    "verify_payload": "system1.receipt",
    "DECISION_WITNESS_PROFILE": "system1.receipt",
    # ledger
    "ActionLedger": "system1.ledger",
    "LedgerError": "system1.ledger",
    "IntegrityError": "system1.ledger",
    # guard
    "ReflexGuardHook": "system1.guard",
    "SystemOneGuardHook": "system1.guard",
    "DefaultGuardDecisionSchema": "system1.guard",
    "GuardInterceptionResult": "system1.guard",
    "ActionProposal": "system1.guard",
    "PolicyDecision": "system1.guard",
    "EvidenceRef": "system1.guard",
    "DecisionOutcome": "system1.guard",
    "RiskLevel": "system1.guard",
    "ActionState": "system1.guard",
    "ResultStatus": "system1.guard",
    # compat
    "TypeSafeClient": "system1.compat.typesafe",
    "Client": "system1.compat.typesafe",
    "AsyncTypeSafeClient": "system1.compat.typesafe",
    "AsyncClient": "system1.compat.typesafe",
    "Choice": "system1.compat.typesafe",
    "MultiChoice": "system1.compat.typesafe",
    "Noul": "system1.compat.typesafe",
    "Score": "system1.compat.typesafe",
    "TypeSafeResponse": "system1.compat.typesafe",
    "SystemOneResponse": "system1.compat.typesafe",
    "ChoiceAnswer": "system1.compat.typesafe",
    "NoulAnswer": "system1.compat.typesafe",
    "ScoreAnswer": "system1.compat.typesafe",
    "MultiChoiceAnswer": "system1.compat.typesafe",
    "Usage": "system1.compat.typesafe",
    "system_one": "system1.compat.typesafe",
    "systemone": "system1.compat.typesafe",
    "batch_system_one": "system1.compat.typesafe",
    "batch_systemone": "system1.compat.typesafe",
    "patch_typesafe": "system1.compat.typesafe",
    "DotDict": "system1.compat.typesafe",
    # compiler
    "ReflexCompiler": "system1.compiler",
    "CompiledSystemOneModel": "system1.compiler",
    # integrations
    "ReflexMCPProxy": "system1.integrations",
    "ReflexMCPBlockedError": "system1.integrations",
    "wrap_mcp_tool": "system1.integrations",
    "ReflexGatewayMiddleware": "system1.integrations",
    "add_reflex_gateway": "system1.integrations",
    "ReflexGuardCallbackHandler": "system1.integrations",
    "ReflexToolInterceptor": "system1.integrations",
    "ReflexGuardBlockedException": "system1.integrations",
    "wrap_langchain_tool": "system1.integrations",
}


def __getattr__(name: str) -> Any:
    mod_name = _MODULE_MAP.get(name)
    if mod_name is not None:
        import importlib
        mod = importlib.import_module(mod_name)
        if name in _SUBMODULE_NAMES:
            val = mod
        else:
            val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__) | set(_MODULE_MAP.keys()) | _SUBMODULE_NAMES)


__all__ = [
    # Version
    "__version__",
    # Engine & Core Results
    "ReflexEngine",
    "SystemOneEngine",
    "System1Engine",
    "DecisionResult",
    "BenchmarkReport",
    "decide",
    # Schemas & Fields
    "DecisionSchema",
    "DecisionField",
    "ChoiceField",
    "BooleanField",
    "MultiChoiceField",
    "ScoreField",
    "SchemaMeta",
    # Calibration & Conformal Prediction
    "DecisionCalibrator",
    "CalibrationMetrics",
    "BrierDecomposition",
    "ConformalPredictor",
    "ConformalPredictionSet",
    "RegressionConformalPredictor",
    "RegressionConformalInterval",
    # Proof-Carrying Evidence Receipts & Crypto
    "DecisionWitnessReceipt",
    "create_decision_receipt",
    "verify_decision_witness_receipt",
    "compute_receipt_digest",
    "RunWitnessEnvelope",
    "create_run_witness_envelope",
    "sign_run_witness_envelope",
    "verify_run_witness_envelope",
    "canonical_json",
    "canonical_bytes",
    "fingerprint",
    "public_key_bytes",
    "public_key_fingerprint",
    "load_private_key",
    "load_public_key",
    "save_keypair",
    "save_private_key",
    "save_public_key",
    "sign_payload",
    "verify_payload",
    "DECISION_WITNESS_PROFILE",
    # Action Ledger
    "ActionLedger",
    "LedgerError",
    "IntegrityError",
    # Reference Monitor & Guard
    "ReflexGuardHook",
    "SystemOneGuardHook",
    "DefaultGuardDecisionSchema",
    "GuardInterceptionResult",
    "ActionProposal",
    "PolicyDecision",
    "EvidenceRef",
    "DecisionOutcome",
    "RiskLevel",
    "ActionState",
    "ResultStatus",
    # Model & Neural Projection
    "SystemOneModel",
    "DecisionFieldHead",
    "DeterministicSemanticProjector",
    "LocalNeuralProjector",
    "HybridProjector",
    "HybridSemanticProjector",
    "SubwordSemanticEmbeddings",
    "ModelInferenceResult",
    "RawFieldEvaluation",
    # TypeSafe Compatibility
    "TypeSafeClient",
    "Client",
    "AsyncTypeSafeClient",
    "AsyncClient",
    "Choice",
    "MultiChoice",
    "Noul",
    "Score",
    "TypeSafeResponse",
    "SystemOneResponse",
    "ChoiceAnswer",
    "NoulAnswer",
    "ScoreAnswer",
    "MultiChoiceAnswer",
    "Usage",
    "system_one",
    "systemone",
    "batch_system_one",
    "batch_systemone",
    "patch_typesafe",
    "DotDict",
    # Compiler
    "ReflexCompiler",
    "CompiledSystemOneModel",
    # 4 Levers (Cache, Telemetry, Online Update, Margin Gating)
    "SemanticReflexCache",
    "CacheEntry",
    "TelemetryProjector",
    "evaluate",
    # Framework Integrations (MCP, FastAPI, LangChain)
    "ReflexMCPProxy",
    "ReflexMCPBlockedError",
    "wrap_mcp_tool",
    "ReflexGatewayMiddleware",
    "add_reflex_gateway",
    "ReflexGuardCallbackHandler",
    "ReflexToolInterceptor",
    "ReflexGuardBlockedException",
    "wrap_langchain_tool",
]
