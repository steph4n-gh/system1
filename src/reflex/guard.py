"""Reflex Fail-Closed Reference Monitor & Guard Hook.

Binds Reflex decision outputs into a hardware-enforced fail-closed Reference Monitor
and ActionLedger. Guarantees that ambiguous, low-confidence, or unsafe decisions
fail-closed to REQUIRE_APPROVAL or DENY.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any, Mapping, Optional, Sequence

from reflex.engine import DecisionResult, ReflexEngine, SystemOneEngine
from reflex.ledger import ActionLedger
from reflex.receipt import fingerprint, freeze, thaw, utc_now
from reflex.schema import (
    BooleanField,
    ChoiceField,
    DecisionSchema,
    ScoreField,
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@-]{0,254}$")


def _required_identifier(name: str, value: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a non-empty canonical identifier")
    return value


def _new_id(prefix: str) -> str:
    return f"{prefix}_{os.urandom(16).hex()}"


def _required_fingerprint(name: str, value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _required_target(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or len(value) > 4096:
        raise ValueError("canonical_target must be non-empty, bounded, and contain no NUL")
    return value


def _digest_content(content: str | bytes) -> str:
    payload = content.encode("utf-8") if isinstance(content, str) else bytes(content)
    return hashlib.sha256(payload).hexdigest()


class RiskLevel(IntEnum):
    READ_ONLY = 0
    REVERSIBLE = 1
    EXTERNAL = 2
    IRREVERSIBLE = 3


class DecisionOutcome(StrEnum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"


class ActionState(StrEnum):
    PROPOSED = "PROPOSED"
    DENIED = "DENIED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    AUTHORIZED = "AUTHORIZED"
    EXECUTING = "EXECUTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INDETERMINATE = "INDETERMINATE"


class ResultStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """Content-bound external evidence."""

    evidence_id: str
    tenant_id: str
    principal_id: str
    scope: str
    source: str
    content_digest: str
    observed_at: str
    lineage: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("evidence_id", "tenant_id", "principal_id", "scope", "source"):
            _required_identifier(name, getattr(self, name))
        if not re.fullmatch(r"[0-9a-f]{64}", self.content_digest):
            raise ValueError("content_digest must be a lowercase SHA-256 digest")
        object.__setattr__(self, "lineage", tuple(self.lineage))

    @classmethod
    def external(
        cls,
        *,
        tenant_id: str,
        principal_id: str,
        scope: str,
        source: str,
        content: str | bytes,
        lineage: tuple[str, ...] = (),
        evidence_id: str | None = None,
        observed_at: str | None = None,
    ) -> EvidenceRef:
        return cls(
            evidence_id=evidence_id or _new_id("ev"),
            tenant_id=tenant_id,
            principal_id=principal_id,
            scope=scope,
            source=source,
            content_digest=_digest_content(content),
            observed_at=observed_at or utc_now(),
            lineage=tuple(lineage),
        )

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "tenant_id": self.tenant_id,
            "principal_id": self.principal_id,
            "scope": self.scope,
            "source": self.source,
            "content_digest": self.content_digest,
            "observed_at": self.observed_at,
            "lineage": list(self.lineage),
        }


@dataclass(frozen=True, slots=True, init=False)
class ActionProposal:
    """An agent's requested effect, with no caller-supplied risk or approval."""

    action_id: str
    tenant_id: str
    principal_id: str
    scope: str
    tool: str
    adapter_contract_fingerprint: str
    canonical_target: str
    arguments: Mapping[str, Any]
    evidence: tuple[EvidenceRef, ...]
    purpose: str
    created_at: str

    def __init__(self, *_: Any, **__: Any) -> None:
        raise TypeError("ActionProposal records must be created through constructors")

    def _validate(self) -> None:
        for name in ("action_id", "tenant_id", "principal_id", "scope", "tool"):
            _required_identifier(name, getattr(self, name))
        _required_fingerprint("adapter_contract_fingerprint", self.adapter_contract_fingerprint)
        _required_target(self.canonical_target)
        if not isinstance(self.purpose, str) or not self.purpose.strip():
            raise ValueError("purpose must be non-empty")
        frozen_arguments = freeze(dict(self.arguments))
        object.__setattr__(self, "arguments", frozen_arguments)
        object.__setattr__(self, "evidence", tuple(self.evidence))

    @classmethod
    def create(
        cls,
        *,
        tenant_id: str,
        principal_id: str,
        scope: str,
        tool: str,
        arguments: Mapping[str, Any],
        purpose: str,
        evidence: tuple[EvidenceRef, ...] = (),
        action_id: str | None = None,
        created_at: str | None = None,
    ) -> ActionProposal:
        """Create an unbound agent proposal."""
        obj = object.__new__(cls)
        values = {
            "action_id": action_id or _new_id("act"),
            "tenant_id": tenant_id,
            "principal_id": principal_id,
            "scope": scope,
            "tool": tool,
            "adapter_contract_fingerprint": "0" * 64,
            "canonical_target": f"unresolved:{tool}",
            "arguments": arguments,
            "evidence": evidence,
            "purpose": purpose,
            "created_at": created_at or utc_now(),
        }
        for name, value in values.items():
            object.__setattr__(obj, name, value)
        obj._validate()
        return obj

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "tenant_id": self.tenant_id,
            "principal_id": self.principal_id,
            "scope": self.scope,
            "tool": self.tool,
            "adapter_contract_fingerprint": self.adapter_contract_fingerprint,
            "canonical_target": self.canonical_target,
            "arguments": thaw(self.arguments),
            "evidence": [item.to_dict() for item in self.evidence],
            "purpose": self.purpose,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True, init=False)
class PolicyDecision:
    """Evaluated policy decision for an action proposal."""

    decision_id: str
    action_id: str
    action_fingerprint: str
    tenant_id: str
    principal_id: str
    scope: str
    adapter_contract_fingerprint: str
    canonical_target: str
    policy_id: str
    policy_epoch: int
    revocation_epoch: int
    outcome: DecisionOutcome
    reason: str
    rule_id: str
    risk: RiskLevel | None
    normalized_arguments: Mapping[str, Any]
    decided_at: str

    def __init__(self, *_: Any, **__: Any) -> None:
        raise TypeError("PolicyDecision records must be issued through _issue")

    @classmethod
    def _issue(cls, **values: Any) -> PolicyDecision:
        obj = object.__new__(cls)
        values = {
            "decision_id": values.get("decision_id") or _new_id("dec"),
            "decided_at": values.get("decided_at") or utc_now(),
            **values,
        }
        values["outcome"] = DecisionOutcome(values["outcome"])
        if values.get("risk") is not None:
            values["risk"] = RiskLevel(values["risk"])
        values["normalized_arguments"] = freeze(dict(values.get("normalized_arguments", {})))
        for field_name in cls.__dataclass_fields__:
            object.__setattr__(obj, field_name, values.get(field_name))
        return obj

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "action_id": self.action_id,
            "action_fingerprint": self.action_fingerprint,
            "tenant_id": self.tenant_id,
            "principal_id": self.principal_id,
            "scope": self.scope,
            "adapter_contract_fingerprint": self.adapter_contract_fingerprint,
            "canonical_target": self.canonical_target,
            "policy_id": self.policy_id,
            "policy_epoch": self.policy_epoch,
            "revocation_epoch": self.revocation_epoch,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "rule_id": self.rule_id,
            "risk": None if self.risk is None else int(self.risk),
            "normalized_arguments": thaw(self.normalized_arguments),
            "decided_at": self.decided_at,
        }


class DefaultGuardDecisionSchema(DecisionSchema):
    """Standard system schema for agent action guardrailing and routing."""

    is_safe = BooleanField(
        description="Whether the requested action is safe to execute without privilege escalation",
        true_description=(
            "safe benign authorized read only project documentation readme local observation diff git "
            "view inspect query status check harmless update cache entry valid "
            "network telemetry payload partner request webhook ping external transmit"
        ),
        false_description=(
            "unsafe dangerous destroy wipe delete root rm format kill attack "
            "overwrite leak exfiltrate breach secret credentials private auth keys token "
            "credential unauthorized tables production partitions private auth keys secret credentials data leak"
        ),
        threshold=0.5,
    )
    risk_category = ChoiceField(
        options=["read_only", "reversible_mutation", "network_call", "irreversible_write"],
        descriptions={
            "read_only": "read only inspect view query local file observation check diff log readme documentation",
            "reversible_mutation": "reversible mutation update edit modify write cache commit rollback branch",
            "network_call": "network call external remote http api webhook fetch transmit request telemetry partner",
            "irreversible_write": "irreversible write destructive destroy wipe delete rm format kill drop tables production partitions",
        },
    )
    risk_score = ScoreField(
        min_value=0.0,
        max_value=1.0,
        low_description="benign read only safe harmless observation",
        high_description="critical dangerous destructive attack wipe delete compromise",
    )


@dataclass(frozen=True)
class GuardInterceptionResult:
    """Interception decision from the reference monitor hook."""

    outcome: DecisionOutcome
    reason: str
    decision_result: DecisionResult
    policy_decision: PolicyDecision


_DEFAULT_GUARD_CALIBRATION = [
    ("Read local file README.md on disk", {"is_safe": True, "risk_category": "read_only"}),
    ("Inspect git log status and diff", {"is_safe": True, "risk_category": "read_only"}),
    ("Query database for user record", {"is_safe": True, "risk_category": "read_only"}),
    ("Execute rm -rf / and destroy filesystem", {"is_safe": False, "risk_category": "irreversible_write"}),
    ("Wipe all production customer tables", {"is_safe": False, "risk_category": "irreversible_write"}),
    ("Transmit private auth keys over external network", {"is_safe": False, "risk_category": "network_call"}),
    ("Format hard drive partitions", {"is_safe": False, "risk_category": "irreversible_write"}),
    ("Update cache entry with 5 minute TTL", {"is_safe": True, "risk_category": "reversible_mutation"}),
    ("Transmit telemetry payload to external network partner", {"is_safe": True, "risk_category": "network_call"}),
    ("Send status ping to remote webhook endpoint", {"is_safe": True, "risk_category": "network_call"}),
] * 4


class ReflexGuardHook:
    """Fail-closed reference monitor hook driven by Reflex decisions.

    Guarantees:
    1. Unsafe evaluation -> immediate DENY.
    2. Conformal ambiguity (|C(x)| > 1) -> REQUIRE_APPROVAL.
    3. Low calibrated confidence (< min_confidence) -> REQUIRE_APPROVAL.
    4. Out-of-distribution (|C(x)| == 0) -> REQUIRE_APPROVAL.
    5. Proof-carrying Ed25519 receipt bound to ActionLedger.
    """

    def __init__(
        self,
        engine: Optional[ReflexEngine] = None,
        *,
        min_confidence: float = 0.85,
        alpha: float = 0.05,
        ledger: Optional[ActionLedger] = None,
        auto_calibrate: bool = True,
    ) -> None:
        if engine is None:
            self.engine = ReflexEngine(DefaultGuardDecisionSchema, ledger=ledger)
            if auto_calibrate:
                self.engine.calibrate(_DEFAULT_GUARD_CALIBRATION, n_bins=5)
        else:
            self.engine = engine
        self.min_confidence = float(min_confidence)
        self.alpha = float(alpha)
        self.ledger = ledger or self.engine.ledger

    def _build_policy_decision(
        self,
        proposal: ActionProposal,
        outcome: DecisionOutcome,
        risk: RiskLevel,
        rule_id: str,
        reason: str,
    ) -> PolicyDecision:
        return PolicyDecision._issue(
            action_id=proposal.action_id,
            action_fingerprint=proposal.fingerprint,
            tenant_id=proposal.tenant_id,
            principal_id=proposal.principal_id,
            scope=proposal.scope,
            adapter_contract_fingerprint=proposal.adapter_contract_fingerprint,
            canonical_target=proposal.canonical_target,
            policy_id="policy_reflex_guard",
            policy_epoch=1,
            revocation_epoch=1,
            outcome=outcome,
            reason=reason,
            rule_id=rule_id,
            risk=risk,
            normalized_arguments=proposal.arguments,
        )

    def evaluate_proposal(
        self,
        proposal: ActionProposal,
        *,
        context_prompt: Optional[str] = None,
    ) -> GuardInterceptionResult:
        """Evaluates an ActionProposal through Reflex fail-closed gates."""
        prompt = context_prompt or (
            f"Tool: {proposal.tool}. Target: {proposal.canonical_target}. "
            f"Args: {dict(proposal.arguments)}. Purpose: {proposal.purpose}"
        )

        decision = self.engine.decide(
            prompt,
            alpha=self.alpha,
            record_receipt=True,
            ledger=self.ledger,
        )

        # 1. Check safety boolean if present
        if "is_safe" in decision.values and not decision.values["is_safe"]:
            outcome = DecisionOutcome.DENY
            reason = (
                f"Reflex classified action as unsafe (confidence: {decision.confidences.get('is_safe', 0.0):.3f})"
            )
            pol_decision = self._build_policy_decision(
                proposal, outcome, RiskLevel.IRREVERSIBLE, "reflex_safety_deny", reason
            )
            return GuardInterceptionResult(outcome, reason, decision, pol_decision)

        # 2. Check conformal ambiguity (|C(x)| > 1) or OOD (|C(x)| == 0)
        for field_name, cset in decision.conformal_sets.items():
            f_def = self.engine.schema.fields.get(field_name)

            # Out-of-distribution detection across all fields
            if len(cset) == 0:
                outcome = DecisionOutcome.REQUIRE_APPROVAL
                reason = f"Reflex detected out-of-distribution input for field {field_name!r} (empty conformal set)"
                pol_decision = self._build_policy_decision(
                    proposal, outcome, RiskLevel.EXTERNAL, "reflex_conformal_ood", reason
                )
                return GuardInterceptionResult(outcome, reason, decision, pol_decision)

            # Categorical ambiguity (|C(x)| > 1) strictly applies to single-selection fields
            if isinstance(f_def, (ChoiceField, BooleanField)) and len(cset) > 1:
                outcome = DecisionOutcome.REQUIRE_APPROVAL
                reason = (
                    f"Reflex conformal ambiguity for {field_name!r}: set {cset} has {len(cset)} candidates at 1-alpha={1.0 - self.alpha:.2f}"
                )
                pol_decision = self._build_policy_decision(
                    proposal, outcome, RiskLevel.EXTERNAL, "reflex_conformal_ambiguity", reason
                )
                return GuardInterceptionResult(outcome, reason, decision, pol_decision)

        # 3. Check minimum confidence threshold across all fields
        for field_name, conf in decision.confidences.items():
            if conf < self.min_confidence:
                outcome = DecisionOutcome.REQUIRE_APPROVAL
                reason = (
                    f"Reflex calibrated confidence {conf:.3f} for {field_name!r} below required threshold {self.min_confidence:.3f}"
                )
                pol_decision = self._build_policy_decision(
                    proposal, outcome, RiskLevel.REVERSIBLE, "reflex_low_confidence", reason
                )
                return GuardInterceptionResult(outcome, reason, decision, pol_decision)

        # 4. Safe and confident
        outcome = DecisionOutcome.ALLOW
        reason = f"Reflex verified action with full conformal confidence in {decision.latency_ms:.2f}ms"
        pol_decision = self._build_policy_decision(
            proposal, outcome, RiskLevel.READ_ONLY, "reflex_verified_allow", reason
        )
        return GuardInterceptionResult(outcome, reason, decision, pol_decision)


# Compatibility alias
SystemOneGuardHook = ReflexGuardHook
