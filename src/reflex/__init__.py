"""Reflex: Machine-Native System 1 Decision Runtime.

Re-exports all public symbols and submodules from system1 for seamless drop-in compatibility.
"""

from __future__ import annotations

from typing import Any

import system1
from system1 import *
from system1 import (
    ReflexEngine,
    SystemOneEngine,
    System1Engine,
    DecisionResult,
    BenchmarkReport,
    decide,
    DecisionSchema,
    DecisionField,
    ChoiceField,
    BooleanField,
    MultiChoiceField,
    ScoreField,
    SchemaMeta,
    DecisionCalibrator,
    CalibrationMetrics,
    BrierDecomposition,
    ConformalPredictor,
    ConformalPredictionSet,
    RegressionConformalPredictor,
    RegressionConformalInterval,
    DecisionWitnessReceipt,
    create_decision_receipt,
    verify_decision_witness_receipt,
    compute_receipt_digest,
    RunWitnessEnvelope,
    create_run_witness_envelope,
    sign_run_witness_envelope,
    verify_run_witness_envelope,
    canonical_json,
    canonical_bytes,
    fingerprint,
    public_key_bytes,
    public_key_fingerprint,
    load_private_key,
    load_public_key,
    save_keypair,
    save_private_key,
    save_public_key,
    sign_payload,
    verify_payload,
    DECISION_WITNESS_PROFILE,
    ActionLedger,
    LedgerError,
    IntegrityError,
    ReflexGuardHook,
    SystemOneGuardHook,
    DefaultGuardDecisionSchema,
    GuardInterceptionResult,
    ActionProposal,
    PolicyDecision,
    EvidenceRef,
    DecisionOutcome,
    RiskLevel,
    ActionState,
    ResultStatus,
    SystemOneModel,
    DecisionFieldHead,
    DeterministicSemanticProjector,
    LocalNeuralProjector,
    HybridProjector,
    HybridSemanticProjector,
    SubwordSemanticEmbeddings,
    ModelInferenceResult,
    RawFieldEvaluation,
    TypeSafeClient,
    Client,
    AsyncTypeSafeClient,
    AsyncClient,
    Choice,
    MultiChoice,
    Noul,
    Score,
    TypeSafeResponse,
    SystemOneResponse,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    MultiChoiceAnswer,
    Usage,
    system_one,
    systemone,
    batch_system_one,
    batch_systemone,
    patch_typesafe,
    DotDict,
    ReflexCompiler,
    CompiledSystemOneModel,
    SemanticReflexCache,
    CacheEntry,
    TelemetryProjector,
    evaluate,
)
from system1 import __all__ as _system1_all

__version__ = system1.__version__
__all__ = list(_system1_all)

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
    "ledger",
    "model",
    "neural",
    "receipt",
    "schema",
    "telemetry",
}

# Dynamic submodule and lazy symbol dispatch map
_MODULE_MAP = {
    # 15 submodules
    "cache": "reflex.cache",
    "calibration": "reflex.calibration",
    "cli": "reflex.cli",
    "compat": "reflex.compat",
    "compiler": "reflex.compiler",
    "core": "reflex.core",
    "embeddings": "reflex.embeddings",
    "engine": "reflex.engine",
    "guard": "reflex.guard",
    "ledger": "reflex.ledger",
    "model": "reflex.model",
    "neural": "reflex.neural",
    "receipt": "reflex.receipt",
    "schema": "reflex.schema",
    "telemetry": "reflex.telemetry",
    # cache (Lever 1)
    "SemanticReflexCache": "reflex.cache",
    "CacheEntry": "reflex.cache",
    # telemetry (Lever 4)
    "TelemetryProjector": "reflex.telemetry",
    # engine
    "ReflexEngine": "reflex.engine",
    "SystemOneEngine": "reflex.engine",
    "System1Engine": "reflex.engine",
    "DecisionResult": "reflex.engine",
    "BenchmarkReport": "reflex.engine",
    # receipt
    "DecisionWitnessReceipt": "reflex.receipt",
    "create_decision_receipt": "reflex.receipt",
    "verify_decision_witness_receipt": "reflex.receipt",
    "compute_receipt_digest": "reflex.receipt",
    "RunWitnessEnvelope": "reflex.receipt",
    "create_run_witness_envelope": "reflex.receipt",
    "sign_run_witness_envelope": "reflex.receipt",
    "verify_run_witness_envelope": "reflex.receipt",
    "canonical_json": "reflex.receipt",
    "canonical_bytes": "reflex.receipt",
    "fingerprint": "reflex.receipt",
    "public_key_bytes": "reflex.receipt",
    "public_key_fingerprint": "reflex.receipt",
    "load_private_key": "reflex.receipt",
    "load_public_key": "reflex.receipt",
    "save_keypair": "reflex.receipt",
    "save_private_key": "reflex.receipt",
    "save_public_key": "reflex.receipt",
    "sign_payload": "reflex.receipt",
    "verify_payload": "reflex.receipt",
    "DECISION_WITNESS_PROFILE": "reflex.receipt",
    # ledger
    "ActionLedger": "reflex.ledger",
    "LedgerError": "reflex.ledger",
    "IntegrityError": "reflex.ledger",
    # guard
    "ReflexGuardHook": "reflex.guard",
    "SystemOneGuardHook": "reflex.guard",
    "DefaultGuardDecisionSchema": "reflex.guard",
    "GuardInterceptionResult": "reflex.guard",
    "ActionProposal": "reflex.guard",
    "PolicyDecision": "reflex.guard",
    "EvidenceRef": "reflex.guard",
    "DecisionOutcome": "reflex.guard",
    "RiskLevel": "reflex.guard",
    "ActionState": "reflex.guard",
    "ResultStatus": "reflex.guard",
    # compat
    "TypeSafeClient": "reflex.compat.typesafe",
    "Client": "reflex.compat.typesafe",
    "AsyncTypeSafeClient": "reflex.compat.typesafe",
    "AsyncClient": "reflex.compat.typesafe",
    "Choice": "reflex.compat.typesafe",
    "MultiChoice": "reflex.compat.typesafe",
    "Noul": "reflex.compat.typesafe",
    "Score": "reflex.compat.typesafe",
    "TypeSafeResponse": "reflex.compat.typesafe",
    "SystemOneResponse": "reflex.compat.typesafe",
    "ChoiceAnswer": "reflex.compat.typesafe",
    "NoulAnswer": "reflex.compat.typesafe",
    "ScoreAnswer": "reflex.compat.typesafe",
    "MultiChoiceAnswer": "reflex.compat.typesafe",
    "Usage": "reflex.compat.typesafe",
    "system_one": "reflex.compat.typesafe",
    "systemone": "reflex.compat.typesafe",
    "batch_system_one": "reflex.compat.typesafe",
    "batch_systemone": "reflex.compat.typesafe",
    "patch_typesafe": "reflex.compat.typesafe",
    "DotDict": "reflex.compat.typesafe",
    # compiler
    "ReflexCompiler": "reflex.compiler",
    "CompiledSystemOneModel": "reflex.compiler",
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
    return sorted(set(globals().keys()) | set(__all__) | set(_MODULE_MAP.keys()))
