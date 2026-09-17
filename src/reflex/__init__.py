"""Reflex: Machine-Native System 1 Decision Runtime.

High-performance, non-autoregressive decision engine with sub-2ms local execution,
calibrated confidence scoring, split conformal prediction guarantees, and
proof-carrying Ed25519 RunWitnessEnvelope decision receipts.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Type, Union

from reflex.calibration import (
    BrierDecomposition,
    CalibrationMetrics,
    ConformalPredictionSet,
    ConformalPredictor,
    DecisionCalibrator,
    RegressionConformalInterval,
    RegressionConformalPredictor,
)
from reflex.engine import (
    BenchmarkReport,
    DecisionResult,
    ReflexEngine,
    SystemOneEngine,
)
from reflex.guard import (
    ActionProposal,
    ActionState,
    DecisionOutcome,
    DefaultGuardDecisionSchema,
    EvidenceRef,
    GuardInterceptionResult,
    PolicyDecision,
    ReflexGuardHook,
    ResultStatus,
    RiskLevel,
    SystemOneGuardHook,
)
from reflex.ledger import (
    ActionLedger,
    IntegrityError,
    LedgerError,
)
from reflex.model import (
    DeterministicSemanticProjector,
    ModelInferenceResult,
    RawFieldEvaluation,
    SystemOneModel,
)
from reflex.receipt import (
    DECISION_WITNESS_PROFILE,
    DecisionWitnessReceipt,
    RunWitnessEnvelope,
    canonical_bytes,
    canonical_json,
    compute_receipt_digest,
    create_decision_receipt,
    create_run_witness_envelope,
    fingerprint,
    load_private_key,
    load_public_key,
    public_key_bytes,
    public_key_fingerprint,
    sign_payload,
    sign_run_witness_envelope,
    verify_decision_witness_receipt,
    verify_payload,
    verify_run_witness_envelope,
)
from reflex.schema import (
    BooleanField,
    ChoiceField,
    DecisionField,
    DecisionSchema,
    MultiChoiceField,
    ScoreField,
)

__version__ = "0.1.0"


def decide(
    prompt: str,
    schema: Union[DecisionSchema, Type[DecisionSchema]],
    *,
    alpha: float = 0.05,
    record_receipt: bool = True,
    dimension: int = 384,
    backend: str = "auto",
) -> DecisionResult:
    """One-liner functional API for evaluating a Reflex decision against a typed schema.

    Example:
        import reflex

        class RoutingSchema(reflex.DecisionSchema):
            route = reflex.ChoiceField(options=["sales", "support", "billing"])
            is_escalation = reflex.BooleanField()

        result = reflex.decide("Customer invoice payment dispute", schema=RoutingSchema)
        print(result.route)
        print(result.confidences["route"])
        print(result.conformal_sets["route"])
    """
    if not isinstance(prompt, str):
        raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")
    engine = ReflexEngine(schema, dimension=dimension, backend=backend)
    return engine.decide(prompt, alpha=alpha, record_receipt=record_receipt)


__all__ = [
    # Version
    "__version__",
    # Engine & Core Results
    "ReflexEngine",
    "SystemOneEngine",
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
    # Model & Projection
    "SystemOneModel",
    "DeterministicSemanticProjector",
    "ModelInferenceResult",
    "RawFieldEvaluation",
]
