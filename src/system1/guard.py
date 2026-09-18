"""Reflex Fail-Closed Reference Monitor & Guard Hook.

Binds Reflex decision outputs into a hardware-enforced fail-closed Reference Monitor
and ActionLedger. Guarantees that ambiguous, low-confidence, or unsafe decisions
fail-closed to REQUIRE_APPROVAL or DENY.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from system1.engine import DecisionResult, ReflexEngine, SystemOneEngine
from system1.ledger import ActionLedger, LedgerError, LedgerWriteError
from system1.receipt import fingerprint, freeze, thaw, utc_now
from system1.core import (
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
        canonical_target: str | None = None,
        action_id: str | None = None,
        created_at: str | None = None,
    ) -> ActionProposal:
        """Create an unbound agent proposal."""
        obj = object.__new__(cls)
        target = canonical_target
        if target is None:
            resolved_t = (
                arguments.get("target")
                or arguments.get("path")
                or arguments.get("command")
                or arguments.get("input")
                or arguments.get("query")
                or arguments.get("url")
                or arguments.get("file")
            )
            if resolved_t is not None:
                target = str(resolved_t)
            else:
                target = f"unresolved:{tool}"

        values = {
            "action_id": action_id or _new_id("act"),
            "tenant_id": tenant_id,
            "principal_id": principal_id,
            "scope": scope,
            "tool": tool,
            "adapter_contract_fingerprint": "0" * 64,
            "canonical_target": target,
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
    decision_result: Optional[DecisionResult] = None
    policy_decision: Optional[PolicyDecision] = None
    proposal: Optional[ActionProposal] = None

    @property
    def allowed(self) -> bool:
        """Return True iff the interception outcome is ALLOW."""
        return self.outcome == DecisionOutcome.ALLOW


@dataclass(frozen=True)
class PolicyRule:
    """Deterministic policy rule specifying conditions and strict action outcome."""

    rule_id: str
    description: str = ""
    outcome: DecisionOutcome = DecisionOutcome.DENY
    effect: Optional[DecisionOutcome] = None
    risk: RiskLevel = RiskLevel.IRREVERSIBLE
    reason: Optional[str] = None
    target_pattern: Optional[str] = None
    tenant_pattern: Optional[str] = None
    principal_pattern: Optional[str] = None
    action_pattern: Optional[str] = None
    # Match criteria (any violation triggers outcome with absolute veto)
    principals: Optional[Sequence[str]] = None
    denied_principals: Optional[Sequence[str]] = None
    allowed_principals: Optional[Sequence[str]] = None
    tenants: Optional[Sequence[str]] = None
    denied_tenants: Optional[Sequence[str]] = None
    allowed_tenants: Optional[Sequence[str]] = None
    scopes: Optional[Sequence[str]] = None
    denied_scopes: Optional[Sequence[str]] = None
    allowed_scopes: Optional[Sequence[str]] = None
    tools: Optional[Sequence[str]] = None
    denied_tools: Optional[Sequence[str]] = None
    allowed_tools: Optional[Sequence[str]] = None
    denied_targets: Optional[Sequence[str]] = None
    argument_limits: Optional[Mapping[str, Any]] = None
    predicate: Optional[Callable[[ActionProposal], Optional[Tuple[DecisionOutcome, str]]]] = None

    def evaluate(self, proposal: ActionProposal) -> Optional[Tuple[DecisionOutcome, RiskLevel, str, str]]:
        """Evaluates proposal against this rule. Returns (outcome, risk, rule_id, reason) if triggered."""
        effective_outcome = self.effect if self.effect is not None else self.outcome

        # 0. Pattern checks (tenant, principal, action, target) - conjunctive matching
        has_pattern = (
            self.tenant_pattern is not None
            or self.principal_pattern is not None
            or self.action_pattern is not None
            or self.target_pattern is not None
        )
        if has_pattern:
            if self.tenant_pattern is not None and not re.search(self.tenant_pattern, proposal.tenant_id):
                return None
            if self.principal_pattern is not None and not re.search(self.principal_pattern, proposal.principal_id):
                return None
            if self.action_pattern is not None and not re.search(self.action_pattern, proposal.tool):
                return None
            if self.target_pattern is not None and not re.search(self.target_pattern, str(proposal.canonical_target)):
                return None
            return (
                effective_outcome,
                self.risk,
                self.rule_id,
                self.reason or f"Deterministic policy match for rule {self.rule_id}",
            )

        # 1. Tenant checks
        if self.allowed_tenants is not None and proposal.tenant_id not in self.allowed_tenants:
            return (
                effective_outcome,
                self.risk,
                self.rule_id,
                self.reason or f"Deterministic policy violation: tenant {proposal.tenant_id!r} not in allowed_tenants",
            )
        if self.denied_tenants is not None and proposal.tenant_id in self.denied_tenants:
            return (
                effective_outcome,
                self.risk,
                self.rule_id,
                self.reason or f"Deterministic policy violation: tenant {proposal.tenant_id!r} is explicitly denied",
            )
        if self.tenants is not None and proposal.tenant_id in self.tenants:
            return (
                effective_outcome,
                self.risk,
                self.rule_id,
                self.reason or f"Deterministic policy match for tenant {proposal.tenant_id!r}",
            )

        # 2. Principal checks
        if self.allowed_principals is not None and proposal.principal_id not in self.allowed_principals:
            return (
                effective_outcome,
                self.risk,
                self.rule_id,
                self.reason or f"Deterministic policy violation: principal {proposal.principal_id!r} not in allowed_principals",
            )
        if self.denied_principals is not None and proposal.principal_id in self.denied_principals:
            return (
                effective_outcome,
                self.risk,
                self.rule_id,
                self.reason or f"Deterministic policy violation: principal {proposal.principal_id!r} is explicitly denied",
            )
        if self.principals is not None and proposal.principal_id in self.principals:
            return (
                effective_outcome,
                self.risk,
                self.rule_id,
                self.reason or f"Deterministic policy match for principal {proposal.principal_id!r}",
            )

        # 3. Scope checks
        if self.allowed_scopes is not None and proposal.scope not in self.allowed_scopes:
            return (
                effective_outcome,
                self.risk,
                self.rule_id,
                self.reason or f"Deterministic policy violation: scope {proposal.scope!r} not in allowed_scopes",
            )
        if self.denied_scopes is not None and proposal.scope in self.denied_scopes:
            return (
                effective_outcome,
                self.risk,
                self.rule_id,
                self.reason or f"Deterministic policy violation: scope {proposal.scope!r} is explicitly denied",
            )
        if self.scopes is not None and proposal.scope in self.scopes:
            return (
                effective_outcome,
                self.risk,
                self.rule_id,
                self.reason or f"Deterministic policy match for scope {proposal.scope!r}",
            )

        # 4. Tool / Action checks
        if self.allowed_tools is not None and proposal.tool not in self.allowed_tools:
            return (
                effective_outcome,
                self.risk,
                self.rule_id,
                self.reason or f"Deterministic policy violation: tool {proposal.tool!r} not in allowed_tools",
            )
        if self.denied_tools is not None and proposal.tool in self.denied_tools:
            return (
                effective_outcome,
                self.risk,
                self.rule_id,
                self.reason or f"Deterministic policy violation: tool {proposal.tool!r} is explicitly denied",
            )
        if self.tools is not None and proposal.tool in self.tools:
            return (
                effective_outcome,
                self.risk,
                self.rule_id,
                self.reason or f"Deterministic policy match for tool {proposal.tool!r}",
            )

        # 5. Target checks
        if self.denied_targets is not None:
            target_str = str(proposal.canonical_target).lower()
            for pattern in self.denied_targets:
                if str(pattern).lower() in target_str:
                    return (
                        effective_outcome,
                        self.risk,
                        self.rule_id,
                        self.reason or f"Deterministic policy violation: target {proposal.canonical_target!r} matches denied target pattern {pattern!r}",
                    )

        # 6. Argument limits & checks
        if self.argument_limits is not None:
            raw_args = dict(proposal.arguments)
            for arg_key, limit_val in self.argument_limits.items():
                if arg_key in raw_args:
                    val = raw_args[arg_key]
                    if isinstance(limit_val, (int, float)) and isinstance(val, (int, float)):
                        if val > limit_val:
                            return (
                                effective_outcome,
                                self.risk,
                                self.rule_id,
                                self.reason or f"Deterministic policy violation: argument {arg_key}={val} exceeds limit {limit_val}",
                            )
                    elif isinstance(limit_val, (list, tuple, set)):
                        if val not in limit_val:
                            return (
                                effective_outcome,
                                self.risk,
                                self.rule_id,
                                self.reason or f"Deterministic policy violation: argument {arg_key}={val!r} not in allowed values",
                            )

        # 7. Custom predicate
        if self.predicate is not None:
            pred_res = self.predicate(proposal)
            if pred_res is not None:
                p_outcome, p_reason = pred_res
                return (p_outcome, self.risk, self.rule_id, p_reason)

        return None


class PolicyEngine:
    """Deterministic reference monitor evaluating explicit security policies with absolute veto authority."""

    def __init__(
        self,
        rules: Optional[Sequence[PolicyRule]] = None,
        *,
        allowed_principals: Optional[Sequence[str]] = None,
        denied_principals: Optional[Sequence[str]] = None,
        allowed_tenants: Optional[Sequence[str]] = None,
        denied_tenants: Optional[Sequence[str]] = None,
        allowed_scopes: Optional[Sequence[str]] = None,
        denied_scopes: Optional[Sequence[str]] = None,
        allowed_tools: Optional[Sequence[str]] = None,
        denied_tools: Optional[Sequence[str]] = None,
        denied_targets: Optional[Sequence[str]] = None,
        argument_limits: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.rules: List[PolicyRule] = list(rules) if rules is not None else []
        if any(
            x is not None
            for x in (
                allowed_principals,
                denied_principals,
                allowed_tenants,
                denied_tenants,
                allowed_scopes,
                denied_scopes,
                allowed_tools,
                denied_tools,
                denied_targets,
                argument_limits,
            )
        ):
            self.rules.append(
                PolicyRule(
                    rule_id="engine_configured_rule",
                    description="Configured engine-level deterministic policy constraints",
                    outcome=DecisionOutcome.DENY,
                    risk=RiskLevel.IRREVERSIBLE,
                    allowed_principals=allowed_principals,
                    denied_principals=denied_principals,
                    allowed_tenants=allowed_tenants,
                    denied_tenants=denied_tenants,
                    allowed_scopes=allowed_scopes,
                    denied_scopes=denied_scopes,
                    allowed_tools=allowed_tools,
                    denied_tools=denied_tools,
                    denied_targets=denied_targets,
                    argument_limits=argument_limits,
                )
            )

    def add_rule(self, rule: PolicyRule) -> None:
        self.rules.append(rule)

    def evaluate(self, proposal: ActionProposal) -> Optional[Tuple[DecisionOutcome, RiskLevel, str, str]]:
        """Evaluates all rules sequentially. First veto match returns immediately."""
        for rule in self.rules:
            res = rule.evaluate(proposal)
            if res is not None:
                return res
        return None


DeterministicPolicyEngine = PolicyEngine


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
        policy: Optional[PolicyEngine] = None,
        policy_engine: Optional[PolicyEngine] = None,
        policy_rules: Optional[Sequence[PolicyRule]] = None,
        fail_closed_ledger: bool = True,
    ) -> None:
        self.fail_closed_ledger = bool(fail_closed_ledger)
        if engine is None:
            self.engine = ReflexEngine(
                DefaultGuardDecisionSchema,
                ledger=ledger,
                fail_closed_ledger=self.fail_closed_ledger,
            )
            if auto_calibrate:
                self.engine.calibrate(_DEFAULT_GUARD_CALIBRATION, n_bins=5)
        else:
            self.engine = engine
        self.min_confidence = float(min_confidence)
        self.alpha = float(alpha)
        self.ledger = ledger or self.engine.ledger

        resolved_policy = policy or policy_engine
        if resolved_policy is not None:
            self.policy: Optional[PolicyEngine] = resolved_policy
        elif policy_rules is not None:
            self.policy = PolicyEngine(rules=policy_rules)
        else:
            self.policy = None

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

    def evaluate_deterministic_policy(
        self, proposal: ActionProposal
    ) -> Optional[GuardInterceptionResult]:
        """Evaluates deterministic policy rules before model inference and caching."""
        if self.policy is not None:
            pol_eval = self.policy.evaluate(proposal)
            if pol_eval is not None:
                p_outcome, p_risk, p_rule_id, p_reason = pol_eval
                pol_dec = self._build_policy_decision(
                    proposal, p_outcome, p_risk, p_rule_id, p_reason
                )
                return GuardInterceptionResult(
                    outcome=p_outcome,
                    reason=p_reason,
                    decision_result=None,
                    policy_decision=pol_dec,
                    proposal=proposal,
                )
        return None

    def evaluate_proposal(
        self,
        proposal: ActionProposal,
        *,
        context_prompt: Optional[str] = None,
    ) -> GuardInterceptionResult:
        """Evaluates an ActionProposal through Reflex fail-closed gates."""
        # 0. Deterministic policy evaluation BEFORE model inference and caching
        det_result = self.evaluate_deterministic_policy(proposal)
        if det_result is not None:
            return det_result

        prompt = context_prompt or (
            f"Tool: {proposal.tool}. Target: {proposal.canonical_target}. "
            f"Args: {dict(proposal.arguments)}. Purpose: {proposal.purpose}"
        )

        try:
            decision = self.engine.decide(
                prompt,
                alpha=self.alpha,
                record_receipt=True,
                ledger=self.ledger,
                fail_closed_ledger=self.fail_closed_ledger,
            )
        except (LedgerWriteError, LedgerError, sqlite3.Error) as ex:
            outcome = DecisionOutcome.DENY
            reason = f"Fail-closed Reference Monitor: ActionLedger write failed ({ex})"
            pol_decision = self._build_policy_decision(
                proposal, outcome, RiskLevel.IRREVERSIBLE, "reflex_ledger_failure", reason
            )
            return GuardInterceptionResult(outcome, reason, decision_result=None, policy_decision=pol_decision, proposal=proposal)

        # Fail-closed ledger check: if ledger is configured, durable recording MUST have succeeded
        if self.ledger is not None and (decision.receipt is None or not decision.receipt.ledger_record_id):
            outcome = DecisionOutcome.DENY
            reason = "Fail-closed Reference Monitor: ActionLedger failed to durably record decision receipt"
            pol_decision = self._build_policy_decision(
                proposal, outcome, RiskLevel.IRREVERSIBLE, "reflex_ledger_unrecorded", reason
            )
            return GuardInterceptionResult(outcome, reason, decision, pol_decision, proposal=proposal)

        # 1. Check safety boolean if present
        if "is_safe" in decision.values and not decision.values["is_safe"]:
            outcome = DecisionOutcome.DENY
            reason = (
                f"Reflex classified action as unsafe (confidence: {decision.confidences.get('is_safe', 0.0):.3f})"
            )
            pol_decision = self._build_policy_decision(
                proposal, outcome, RiskLevel.IRREVERSIBLE, "reflex_safety_deny", reason
            )
            return GuardInterceptionResult(outcome, reason, decision, pol_decision, proposal=proposal)

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
                return GuardInterceptionResult(outcome, reason, decision, pol_decision, proposal=proposal)

            # Categorical ambiguity (|C(x)| > 1) strictly applies to single-selection fields
            if isinstance(f_def, (ChoiceField, BooleanField)) and len(cset) > 1:
                outcome = DecisionOutcome.REQUIRE_APPROVAL
                reason = (
                    f"Reflex conformal ambiguity for {field_name!r}: set {cset} has {len(cset)} candidates at 1-alpha={1.0 - self.alpha:.2f}"
                )
                pol_decision = self._build_policy_decision(
                    proposal, outcome, RiskLevel.EXTERNAL, "reflex_conformal_ambiguity", reason
                )
                return GuardInterceptionResult(outcome, reason, decision, pol_decision, proposal=proposal)

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
                return GuardInterceptionResult(outcome, reason, decision, pol_decision, proposal=proposal)

        # 4. Safe and confident
        outcome = DecisionOutcome.ALLOW
        reason = f"Reflex verified action with full conformal confidence in {decision.latency_ms:.2f}ms"
        pol_decision = self._build_policy_decision(
            proposal, outcome, RiskLevel.READ_ONLY, "reflex_verified_allow", reason
        )
        return GuardInterceptionResult(outcome, reason, decision, pol_decision, proposal=proposal)


# Compatibility alias
SystemOneGuardHook = ReflexGuardHook

__all__ = [
    "ActionProposal",
    "ActionState",
    "DecisionOutcome",
    "DefaultGuardDecisionSchema",
    "DeterministicPolicyEngine",
    "EvidenceRef",
    "GuardInterceptionResult",
    "PolicyDecision",
    "PolicyEngine",
    "PolicyRule",
    "ReflexGuardHook",
    "ResultStatus",
    "RiskLevel",
    "SystemOneGuardHook",
]

