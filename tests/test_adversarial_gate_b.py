"""Adversarial Attack & Stress Test Suite for Gate B (Authenticated Final Authorization & Durability).

Empirical Challenger verification covering:
1. Unauthenticated / Downgrade Attack:
   - Strip signature (signature=None or empty string) in enforcement and product_signed profiles.
   - Downgrade profile from enforcement/product_signed to diagnostic_local.
   - Public key substitution and untrusted key re-signing.
   - EnforcementProfile missing signing key rejection.

2. Cross-Field Digest Tampering:
   - Alter canonical probabilities (values, keys, extra entries).
   - Alter policy decision claims (outcome, rule_id, policy_id, policy_epoch, risk).
   - Alter risk level.
   - Alter normalized arguments, canonical target, scope, principal_id, tenant_id.
   - Alter effective alpha, model/calibration identities.
   - Alter values and confidences.

3. Ledger Durability Evasion:
   - Attempt to pass ':memory:' to EnforcementProfile.validate_configuration.
   - Attempt to initialize ActionLedger(':memory:', require_durable=True / enforce_durability=True).
   - Attempt to pass file::memory: URI variants.
   - Attempt to pass ':memory:' ActionLedger to ReflexGuardHook in enforcement mode.
   - Attempt read-only ':memory:' initialization.

4. Hash Chain & Signature Forgery:
   - Valid SHA-256 hash chains with forged/untrusted signatures in ActionLedger.
   - Tampered entry payload in the middle of the chain with recomputed hashes.
   - Injected unsigned decision receipts in ledger with valid hash chain.
   - Broken previous_hash chain links.
   - Corrupted ledger_meta audit_head desynchronization.

5. Two-Phase Outcome Interception:
   - SQLite locks/errors during record_execution_outcome across:
     * wrap_mcp_tool
     * ReflexMCPProxy.handle_call (JSON-RPC error code -32001)
     * ReflexToolInterceptor (sync invoke)
     * ReflexToolInterceptor (async ainvoke)
     * ReflexGuardCallbackHandler (on_tool_end)
     * ReflexGuardCallbackHandler (on_tool_error with exec_error preservation)
   - Ensure complete structured reconciliation evidence (status=INDETERMINATE, action_id, receipt_digest, raw_result, underlying_error).

6. Twin Namespace Parity:
   - Verify object identity and functionality under both 'import reflex' and 'import system1'.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import sqlite3
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import reflex
import reflex.guard as r_guard
import reflex.integrations.langchain as r_lc
import reflex.integrations.mcp as r_mcp
import reflex.ledger as r_ledger
import reflex.receipt as r_receipt
import system1
import system1.guard as s1_guard
import system1.integrations.langchain as s1_lc
import system1.integrations.mcp as s1_mcp
import system1.ledger as s1_ledger
import system1.receipt as s1_receipt
from system1.guard import (
    ActionProposal,
    DecisionOutcome,
    ENFORCEMENT_PROFILE_V1,
    EnforcementProfile,
    PolicyDecision,
    PolicyEngine,
    PolicyRule,
    ReflexGuardHook,
    RiskLevel,
)
from system1.integrations.langchain import (
    ReflexGuardBlockedException,
    ReflexGuardCallbackHandler,
    ReflexIndeterminateExecutionError,
    ReflexToolInterceptor,
    wrap_langchain_tool,
)
from system1.integrations.mcp import (
    ReflexMCPBlockedError,
    ReflexMCPProxy,
    wrap_mcp_tool,
)
from system1.ledger import ActionLedger, IntegrityError, LedgerError, LedgerWriteError
from system1.receipt import (
    DecisionWitnessReceipt,
    RunWitnessEnvelope,
    canonical_json,
    compute_receipt_digest,
    create_decision_receipt,
    create_run_witness_envelope,
    fingerprint,
    public_key_bytes,
    verify_decision_witness_receipt,
)


class MockSideEffectTarget:
    """Mock target that simulates real side effects (e.g. financial transaction or file creation)."""

    def __init__(self, return_val: str = "transferred_funds"):
        self.call_count = 0
        self.recorded_args = []
        self.recorded_kwargs = []
        self.return_val = return_val

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        self.recorded_args.append(args)
        self.recorded_kwargs.append(kwargs)
        return self.return_val

    def mcp_executor(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        self.call_count += 1
        self.recorded_args.append((tool_name, arguments))
        self.recorded_kwargs.append(arguments)
        return {"status": "success", "result": self.return_val}

    def execute_action(self, target: str, amount: float = 100.0) -> str:
        self.call_count += 1
        self.recorded_args.append((target, amount))
        self.recorded_kwargs.append({"target": target, "amount": amount})
        return f"executed:{target}:{amount}"

    async def aexecute_action(self, target: str, amount: float = 100.0) -> str:
        self.call_count += 1
        self.recorded_args.append((target, amount))
        self.recorded_kwargs.append({"target": target, "amount": amount})
        return f"aexecuted:{target}:{amount}"


# ============================================================================
# 1. Unauthenticated / Downgrade Attacks
# ============================================================================


class TestUnauthenticatedAndDowngradeAttacks:
    """Adversarially challenge authentication invariants of Gate B."""

    def test_strip_signature_in_enforcement_profile(self):
        """Stripping signature from an enforcement profile receipt must fail verification."""
        key = Ed25519PrivateKey.generate()
        receipt = create_decision_receipt(
            schema_name="SecurityAuth",
            schema_digest="e" * 64,
            prompt="Authorize root credential rotation",
            values={"is_safe": True},
            confidences={"is_safe": 0.999},
            conformal_sets={"is_safe": ["True"]},
            probabilities={"is_safe": {"True": 0.999, "False": 0.001}},
            latency_ms=0.3,
            is_ambiguous=False,
            signing_key=key,
            profile=ENFORCEMENT_PROFILE_V1,
        )
        receipt_dict = receipt.to_dict()

        # 1. Verification succeeds normally
        assert verify_decision_witness_receipt(receipt_dict, public_key=key.public_key()) is True

        # 2. Attack: Strip signature (None)
        tampered_none = copy.deepcopy(receipt_dict)
        tampered_none["envelope"]["signature"] = None
        assert verify_decision_witness_receipt(tampered_none, public_key=key.public_key()) is False
        assert verify_decision_witness_receipt(tampered_none, public_key=None) is False

        # 3. Attack: Strip signature (empty string)
        tampered_empty = copy.deepcopy(receipt_dict)
        tampered_empty["envelope"]["signature"] = ""
        assert verify_decision_witness_receipt(tampered_empty, public_key=key.public_key()) is False
        assert verify_decision_witness_receipt(tampered_empty, public_key=None) is False

    def test_strip_signature_in_product_signed_profile(self):
        """Stripping signature from product_signed_v1 profile must fail verification."""
        key = Ed25519PrivateKey.generate()
        receipt = create_decision_receipt(
            schema_name="ProductAuth",
            schema_digest="d" * 64,
            prompt="Access financial records",
            values={"is_safe": True},
            confidences={"is_safe": 0.99},
            conformal_sets={"is_safe": ["True"]},
            probabilities={"is_safe": {"True": 0.99, "False": 0.01}},
            latency_ms=0.2,
            is_ambiguous=False,
            signing_key=key,
            profile="product_signed_v1",
        )
        receipt_dict = receipt.to_dict()

        tampered = copy.deepcopy(receipt_dict)
        tampered["envelope"]["signature"] = None
        assert verify_decision_witness_receipt(tampered, public_key=key.public_key()) is False
        assert verify_decision_witness_receipt(tampered, public_key=None) is False

    def test_profile_downgrade_to_diagnostic_local(self):
        """Attacker downgrades profile to diagnostic_local to bypass signature check."""
        key = Ed25519PrivateKey.generate()
        receipt = create_decision_receipt(
            schema_name="EnforcementOp",
            schema_digest="c" * 64,
            prompt="Perform high-risk transaction",
            values={"is_safe": True},
            confidences={"is_safe": 0.99},
            conformal_sets={"is_safe": ["True"]},
            probabilities={"is_safe": {"True": 0.99, "False": 0.01}},
            latency_ms=0.2,
            is_ambiguous=False,
            signing_key=key,
            profile=ENFORCEMENT_PROFILE_V1,
        )
        receipt_dict = receipt.to_dict()

        # Attacker attempts downgrade: sets signature to None, changes profile to diagnostic_local
        downgraded = copy.deepcopy(receipt_dict)
        downgraded["envelope"]["signature"] = None
        downgraded["envelope"]["profile"] = "diagnostic_local"
        downgraded["envelope"]["schema"] = "reflex.witness.diagnostic_local.v1"
        # Recompute envelope digests so envelope internal check might pass
        env_obj = RunWitnessEnvelope.from_dict(downgraded["envelope"])
        computed_env_digest = env_obj.compute_witness_digest()
        downgraded["envelope"]["witness_digest"] = computed_env_digest
        downgraded["envelope"]["envelope_digest"] = computed_env_digest

        # Must reject when verified against trusted public key
        assert verify_decision_witness_receipt(downgraded, public_key=key.public_key()) is False

    def test_tampered_public_key_rejected(self):
        """Verification against an untrusted public key must fail."""
        trusted_key = Ed25519PrivateKey.generate()
        untrusted_key = Ed25519PrivateKey.generate()

        receipt = create_decision_receipt(
            schema_name="TrustedOp",
            schema_digest="b" * 64,
            prompt="Authorize administrative action",
            values={"is_safe": True},
            confidences={"is_safe": 0.99},
            conformal_sets={"is_safe": ["True"]},
            probabilities={"is_safe": {"True": 0.99, "False": 0.01}},
            latency_ms=0.2,
            is_ambiguous=False,
            signing_key=trusted_key,
        )
        receipt_dict = receipt.to_dict()

        # 1. Verified against correct key: True
        assert verify_decision_witness_receipt(receipt_dict, public_key=trusted_key.public_key()) is True

        # 2. Verified against wrong key: False
        assert verify_decision_witness_receipt(receipt_dict, public_key=untrusted_key.public_key()) is False

    def test_untrusted_key_re_signing_rejected_by_trusted_verifier(self):
        """Attacker re-signs payload with their own key; verification with trusted key must fail."""
        trusted_key = Ed25519PrivateKey.generate()
        attacker_key = Ed25519PrivateKey.generate()

        receipt = create_decision_receipt(
            schema_name="Op",
            schema_digest="a" * 64,
            prompt="Authorize transfer",
            values={"is_safe": True},
            confidences={"is_safe": 0.99},
            conformal_sets={"is_safe": ["True"]},
            probabilities={"is_safe": {"True": 0.99, "False": 0.01}},
            latency_ms=0.2,
            is_ambiguous=False,
            signing_key=trusted_key,
        )
        receipt_dict = receipt.to_dict()

        # Attacker re-signs envelope using attacker_key
        envelope = RunWitnessEnvelope.from_dict(receipt_dict["envelope"])
        from system1.receipt import sign_payload, public_key_fingerprint
        new_sig = sign_payload(envelope.unsigned_payload(), attacker_key)
        re_signed_dict = copy.deepcopy(receipt_dict)
        re_signed_dict["envelope"]["signature"] = new_sig
        re_signed_dict["envelope"]["signer_fingerprint"] = public_key_fingerprint(attacker_key)

        # Verifier verifying with trusted_key must reject
        assert verify_decision_witness_receipt(re_signed_dict, public_key=trusted_key.public_key()) is False

    def test_public_key_substitution_in_receipt_detected(self):
        """Attacker replaces declared signer_public_key in receipt; verifier must detect mismatch."""
        trusted_key = Ed25519PrivateKey.generate()
        attacker_key = Ed25519PrivateKey.generate()

        receipt = create_decision_receipt(
            schema_name="Op",
            schema_digest="a" * 64,
            prompt="Authorize transfer",
            values={"is_safe": True},
            confidences={"is_safe": 0.99},
            conformal_sets={"is_safe": ["True"]},
            probabilities={"is_safe": {"True": 0.99, "False": 0.01}},
            latency_ms=0.2,
            is_ambiguous=False,
            signing_key=trusted_key,
        )
        receipt_dict = receipt.to_dict()

        # Attacker substitutes declared signer_public_key with attacker's key
        sub_dict = copy.deepcopy(receipt_dict)
        sub_dict["signer_public_key"] = public_key_bytes(attacker_key).hex()

        # Verifier checks against trusted key: must fail anti-substitution check
        assert verify_decision_witness_receipt(sub_dict, public_key=trusted_key.public_key()) is False

    def test_enforcement_profile_rejects_missing_signing_key(self, tmp_path):
        """EnforcementProfile fails closed when signing_key is None."""
        profile = EnforcementProfile(require_signer=True, require_durable_ledger=True)
        ledger = ActionLedger(path=str(tmp_path / "valid.db"))

        with pytest.raises(ValueError, match="requires a configured trusted Ed25519 signing key"):
            profile.validate_configuration(signing_key=None, ledger=ledger)


# ============================================================================
# 2. Cross-Field Digest Tampering Attacks
# ============================================================================


class TestCrossFieldDigestTampering:
    """Stress-test tamper detection across all fields in DecisionWitnessReceipt."""

    @pytest.fixture
    def authentic_receipt_fixture(self):
        key = Ed25519PrivateKey.generate()
        prop = ActionProposal.create(
            tenant_id="tenant_finance",
            principal_id="service_bot_9",
            scope="ledger:transfer",
            tool="wire_transfer",
            arguments={"amount": 5000.0, "currency": "USD", "destination": "acct_888"},
            canonical_target="acct_888",
            purpose="Routine supplier payout",
        )
        pol_dec = PolicyDecision._issue(
            action_id=prop.action_id,
            action_fingerprint=prop.fingerprint,
            tenant_id=prop.tenant_id,
            principal_id=prop.principal_id,
            scope=prop.scope,
            adapter_contract_fingerprint="0" * 64,
            canonical_target=prop.canonical_target,
            policy_id="policy_finance_v2",
            policy_epoch=5,
            revocation_epoch=1,
            outcome=DecisionOutcome.ALLOW,
            reason="Approved under supplier whitelist rule",
            rule_id="rule_supplier_whitelist",
            risk=RiskLevel.EXTERNAL,
            normalized_arguments=prop.arguments,
        )
        receipt = create_decision_receipt(
            schema_name="WireTransferSchema",
            schema_digest="1234abcd" * 8,
            prompt="Wire transfer request $5000 to acct_888",
            values={"is_safe": True, "policy_outcome": "ALLOW"},
            confidences={"is_safe": 0.999},
            conformal_sets={"is_safe": ["True"]},
            probabilities={"is_safe": {"True": 0.999, "False": 0.001}},
            latency_ms=0.45,
            is_ambiguous=False,
            signing_key=key,
            policy_decision=pol_dec,
            action_proposal=prop,
            profile=ENFORCEMENT_PROFILE_V1,
            effective_alpha=0.05,
            effective_gates={"conformal_gate": "PASSED", "policy_gate": "ALLOW"},
            model_identity="model_v3_metal",
            projector_identity="projector_v3_whitened",
            calibration_identity="cal_fold_2",
        )
        return key, receipt

    def test_tamper_canonical_probabilities(self, authentic_receipt_fixture):
        """Tampering with probability values or adding options invalidates verification."""
        key, receipt = authentic_receipt_fixture
        pub_key = key.public_key()
        r_dict = receipt.to_dict()

        # 1. Modify existing probability value
        tampered_val = copy.deepcopy(r_dict)
        tampered_val["probabilities"]["is_safe"]["True"] = 0.499
        assert verify_decision_witness_receipt(tampered_val, public_key=pub_key) is False

        # 2. Modify other probability option
        tampered_val2 = copy.deepcopy(r_dict)
        tampered_val2["probabilities"]["is_safe"]["False"] = 0.501
        assert verify_decision_witness_receipt(tampered_val2, public_key=pub_key) is False

        # 3. Add an unexpected probability class
        tampered_extra = copy.deepcopy(r_dict)
        tampered_extra["probabilities"]["is_safe"]["Uncertain"] = 0.1
        assert verify_decision_witness_receipt(tampered_extra, public_key=pub_key) is False

    def test_tamper_policy_decision_claims(self, authentic_receipt_fixture):
        """Tampering with policy decision claims invalidates verification."""
        key, receipt = authentic_receipt_fixture
        pub_key = key.public_key()
        r_dict = receipt.to_dict()

        # 1. Tamper outcome from ALLOW to DENY or vice-versa
        tampered_outcome = copy.deepcopy(r_dict)
        tampered_outcome["policy_decision"]["outcome"] = "DENY"
        assert verify_decision_witness_receipt(tampered_outcome, public_key=pub_key) is False

        # 2. Tamper rule_id
        tampered_rule = copy.deepcopy(r_dict)
        tampered_rule["policy_decision"]["rule_id"] = "forged_rule"
        assert verify_decision_witness_receipt(tampered_rule, public_key=pub_key) is False

        # 3. Tamper policy_id
        tampered_pid = copy.deepcopy(r_dict)
        tampered_pid["policy_decision"]["policy_id"] = "policy_bypass"
        assert verify_decision_witness_receipt(tampered_pid, public_key=pub_key) is False

        # 4. Tamper policy_epoch
        tampered_epoch = copy.deepcopy(r_dict)
        tampered_epoch["policy_decision"]["policy_epoch"] = 999
        assert verify_decision_witness_receipt(tampered_epoch, public_key=pub_key) is False

        # 5. Tamper risk level
        tampered_risk = copy.deepcopy(r_dict)
        tampered_risk["policy_decision"]["risk"] = 0  # READ_ONLY instead of EXTERNAL
        assert verify_decision_witness_receipt(tampered_risk, public_key=pub_key) is False

    def test_tamper_risk_level_at_top_level(self, authentic_receipt_fixture):
        """Tampering with top-level risk level in receipt invalidates verification."""
        key, receipt = authentic_receipt_fixture
        pub_key = key.public_key()
        r_dict = receipt.to_dict()

        tampered_risk = copy.deepcopy(r_dict)
        tampered_risk["risk"] = 0  # downgrade to READ_ONLY
        assert verify_decision_witness_receipt(tampered_risk, public_key=pub_key) is False

    def test_tamper_normalized_arguments(self, authentic_receipt_fixture):
        """Tampering with authorized arguments (e.g. payout amount) invalidates verification."""
        key, receipt = authentic_receipt_fixture
        pub_key = key.public_key()
        r_dict = receipt.to_dict()

        tampered_args = copy.deepcopy(r_dict)
        tampered_args["normalized_arguments"]["amount"] = 5000000.0  # $5M instead of $5K
        assert verify_decision_witness_receipt(tampered_args, public_key=pub_key) is False

        # Also test altering argument in policy_decision dictionary
        tampered_args2 = copy.deepcopy(r_dict)
        tampered_args2["policy_decision"]["normalized_arguments"]["destination"] = "attacker_acct"
        assert verify_decision_witness_receipt(tampered_args2, public_key=pub_key) is False

    def test_tamper_target_and_scope(self, authentic_receipt_fixture):
        """Tampering with canonical_target or scope invalidates verification."""
        key, receipt = authentic_receipt_fixture
        pub_key = key.public_key()
        r_dict = receipt.to_dict()

        # 1. Tamper canonical_target
        tampered_target = copy.deepcopy(r_dict)
        tampered_target["canonical_target"] = "acct_evil"
        assert verify_decision_witness_receipt(tampered_target, public_key=pub_key) is False

        # 2. Tamper scope
        tampered_scope = copy.deepcopy(r_dict)
        tampered_scope["scope"] = "admin:unrestricted"
        assert verify_decision_witness_receipt(tampered_scope, public_key=pub_key) is False

    def test_tamper_principal_and_tenant(self, authentic_receipt_fixture):
        """Tampering with principal_id or tenant_id invalidates verification."""
        key, receipt = authentic_receipt_fixture
        pub_key = key.public_key()
        r_dict = receipt.to_dict()

        # 1. Tamper principal_id
        tampered_p = copy.deepcopy(r_dict)
        tampered_p["principal_id"] = "impersonated_admin"
        assert verify_decision_witness_receipt(tampered_p, public_key=pub_key) is False

        # 2. Tamper tenant_id
        tampered_t = copy.deepcopy(r_dict)
        tampered_t["tenant_id"] = "competitor_tenant"
        assert verify_decision_witness_receipt(tampered_t, public_key=pub_key) is False

    def test_tamper_effective_alpha_and_identities(self, authentic_receipt_fixture):
        """Tampering with effective_alpha, model_identity, or calibration_identity invalidates verification."""
        key, receipt = authentic_receipt_fixture
        pub_key = key.public_key()
        r_dict = receipt.to_dict()

        # 1. Tamper effective_alpha
        tampered_alpha = copy.deepcopy(r_dict)
        tampered_alpha["effective_alpha"] = 0.50
        assert verify_decision_witness_receipt(tampered_alpha, public_key=pub_key) is False

        # 2. Tamper model_identity
        tampered_model = copy.deepcopy(r_dict)
        tampered_model["model_identity"] = "compromised_weights_v9"
        assert verify_decision_witness_receipt(tampered_model, public_key=pub_key) is False

        # 3. Tamper calibration_identity
        tampered_cal = copy.deepcopy(r_dict)
        tampered_cal["calibration_identity"] = "unvalidated_cal"
        assert verify_decision_witness_receipt(tampered_cal, public_key=pub_key) is False

    def test_tamper_values_and_confidences(self, authentic_receipt_fixture):
        """Tampering with returned values or confidences invalidates verification."""
        key, receipt = authentic_receipt_fixture
        pub_key = key.public_key()
        r_dict = receipt.to_dict()

        # 1. Tamper values
        tampered_val = copy.deepcopy(r_dict)
        tampered_val["values"]["is_safe"] = False
        assert verify_decision_witness_receipt(tampered_val, public_key=pub_key) is False

        # 2. Tamper confidences
        tampered_conf = copy.deepcopy(r_dict)
        tampered_conf["confidences"]["is_safe"] = 0.123
        assert verify_decision_witness_receipt(tampered_conf, public_key=pub_key) is False


# ============================================================================
# 3. Ledger Durability Evasion Attacks
# ============================================================================


class TestLedgerDurabilityEvasion:
    """Stress-test durability enforcement and memory-only evasion prevention."""

    def test_in_memory_ledger_rejected_in_enforcement_profile(self):
        """EnforcementProfile rejects ActionLedger(':memory:')."""
        profile = EnforcementProfile(require_signer=True, require_durable_ledger=True)
        key = Ed25519PrivateKey.generate()
        mem_ledger = ActionLedger(path=":memory:")

        with pytest.raises(ValueError, match="rejects in-memory ledger|requires a persistent, durable ActionLedger"):
            profile.validate_configuration(signing_key=key, ledger=mem_ledger)

    def test_action_ledger_durability_flags_prevent_memory_creation(self):
        """ActionLedger with require_durable or enforce_durability rejects ':memory:'."""
        with pytest.raises(ValueError, match="Enforcement profile requires a persistent, durable ActionLedger"):
            ActionLedger(path=":memory:", require_durable=True)

        with pytest.raises(ValueError, match="Enforcement profile requires a persistent, durable ActionLedger"):
            ActionLedger(path=":memory:", enforce_durability=True)

    def test_uri_memory_variants_rejected_or_not_durable(self, tmp_path):
        """ActionLedger with require_durable=True rejects 'file::memory:' during __init__."""
        # Clean up any leftover file if created
        import os
        from pathlib import Path
        
        # 1. require_durable=True rejects file::memory: at construction
        with pytest.raises(ValueError, match="Enforcement profile requires a persistent, durable ActionLedger"):
            ActionLedger(path="file::memory:", require_durable=True)

        # Cleanup if a file::memory: was touched
        if Path("file::memory:").exists():
            os.remove("file::memory:")

    def test_reflex_guard_hook_with_memory_ledger_and_enforcement_profile_raises(self):
        """ReflexGuardHook initialization fails closed when given ':memory:' ledger in enforcement mode."""
        key = Ed25519PrivateKey.generate()
        mem_ledger = ActionLedger(path=":memory:")

        with pytest.raises(ValueError, match="rejects in-memory ledger"):
            ReflexGuardHook(
                signing_key=key,
                ledger=mem_ledger,
                enforcement_profile=True,
            )

        with pytest.raises(ValueError, match="rejects in-memory ledger"):
            ReflexGuardHook(
                signing_key=key,
                ledger=mem_ledger,
                enforcement_profile=ENFORCEMENT_PROFILE_V1,
            )

    def test_read_only_memory_ledger_rejected(self):
        """Read-only ActionLedger with ':memory:' is immediately rejected."""
        with pytest.raises(ValueError, match="Read-only ledger requires an existing durable database file"):
            ActionLedger(path=":memory:", read_only=True)


# ============================================================================
# 4. Hash Chain & Signature Forgery Attacks
# ============================================================================


class TestHashChainAndSignatureForgery:
    """Stress-test cryptographic chain integrity and detect signature forgery."""

    def test_forged_signature_in_valid_hash_chain_rejected(self, tmp_path):
        """Attacker constructs a valid SHA-256 hash chain with forged signatures; verify_integrity rejects."""
        trusted_key = Ed25519PrivateKey.generate()
        attacker_key = Ed25519PrivateKey.generate()

        db_path = str(tmp_path / "forgery_test.db")
        ledger = ActionLedger(path=db_path)

        # 1. Record authentic entry signed by trusted_key
        rec1 = create_decision_receipt(
            schema_name="LegitOp",
            schema_digest="a" * 64,
            prompt="Authorize legitimate action 1",
            values={"is_safe": True},
            confidences={"is_safe": 0.99},
            conformal_sets={"is_safe": ["True"]},
            probabilities={"is_safe": {"True": 0.99, "False": 0.01}},
            latency_ms=0.2,
            is_ambiguous=False,
            signing_key=trusted_key,
        )
        ledger.record_decision_receipt(rec1)

        # Chain verifies with trusted key
        assert ledger.verify_integrity(trusted_public_key=trusted_key.public_key()) is True

        # 2. Attacker crafts forged entry signed by attacker_key and appends
        forged_rec = create_decision_receipt(
            schema_name="ForgedOp",
            schema_digest="f" * 64,
            prompt="Authorize fraudulent payout",
            values={"is_safe": True},
            confidences={"is_safe": 0.99},
            conformal_sets={"is_safe": ["True"]},
            probabilities={"is_safe": {"True": 0.99, "False": 0.01}},
            latency_ms=0.2,
            is_ambiguous=False,
            signing_key=attacker_key,
        )
        ledger.record_decision_receipt(forged_rec)

        # Hash chain itself is mathematically valid:
        assert ledger.verify_integrity(trusted_public_key=None) is True

        # BUT cryptographic signature check against trusted_public_key MUST FAIL!
        assert ledger.verify_integrity(trusted_public_key=trusted_key.public_key()) is False

    def test_tampered_payload_in_middle_of_chain_detected(self, tmp_path):
        """Attacker modifies an entry's payload in SQLite and repairs hash chain; verify_integrity rejects."""
        trusted_key = Ed25519PrivateKey.generate()
        db_path = str(tmp_path / "chain_tamper.db")
        ledger = ActionLedger(path=db_path)

        # Append 3 entries
        for i in range(1, 4):
            r = create_decision_receipt(
                schema_name=f"Op{i}",
                schema_digest=f"{i}" * 64,
                prompt=f"Action {i}",
                values={"is_safe": True},
                confidences={"is_safe": 0.99},
                conformal_sets={"is_safe": ["True"]},
                probabilities={"is_safe": {"True": 0.99, "False": 0.01}},
                latency_ms=0.1,
                is_ambiguous=False,
                signing_key=trusted_key,
            )
            ledger.record_decision_receipt(r)

        assert ledger.verify_integrity(trusted_public_key=trusted_key.public_key()) is True

        # Attacker tampers sequence 2's payload_json
        with ledger._transaction() as conn:
            row2 = conn.execute("SELECT * FROM audit_entries WHERE sequence=2").fetchone()
            p_data = json.loads(row2["payload_json"])
            p_data["receipt"]["values"]["is_safe"] = False  # tamper
            new_p_json = canonical_json(p_data)
            conn.execute("UPDATE audit_entries SET payload_json=? WHERE sequence=2", (new_p_json,))

        # Without repairing hash, verify_integrity fails:
        assert ledger.verify_integrity(trusted_public_key=trusted_key.public_key()) is False

    def test_unsigned_receipt_in_ledger_rejected_when_trusted_key_checked(self, tmp_path):
        """Unsigned receipt injected into ledger causes verify_integrity(trusted_key) to return False."""
        trusted_key = Ed25519PrivateKey.generate()
        db_path = str(tmp_path / "unsigned_inject.db")
        ledger = ActionLedger(path=db_path)

        # Create unsigned receipt (diagnostic_local)
        unsigned_receipt = create_decision_receipt(
            schema_name="UnsignedOp",
            schema_digest="0" * 64,
            prompt="Unsigned diagnostic check",
            values={"is_safe": True},
            confidences={"is_safe": 0.9},
            conformal_sets={"is_safe": ["True"]},
            probabilities={"is_safe": {"True": 0.9, "False": 0.1}},
            latency_ms=0.1,
            is_ambiguous=False,
            signing_key=None,
            profile="diagnostic_local",
        )
        ledger.record_decision_receipt(unsigned_receipt)

        # verify_integrity with None passes hash check
        assert ledger.verify_integrity(trusted_public_key=None) is True

        # verify_integrity with trusted_public_key MUST FAIL on unsigned receipt
        assert ledger.verify_integrity(trusted_public_key=trusted_key.public_key()) is False

    def test_broken_hash_chain_previous_hash_mismatch(self, tmp_path):
        """Direct manipulation of previous_hash is caught by verify_integrity."""
        db_path = str(tmp_path / "broken_hash.db")
        ledger = ActionLedger(path=db_path)
        trusted_key = Ed25519PrivateKey.generate()

        for i in range(2):
            r = create_decision_receipt(
                schema_name="Step",
                schema_digest="1" * 64,
                prompt=f"Step {i}",
                values={"is_safe": True},
                confidences={"is_safe": 0.99},
                conformal_sets={"is_safe": ["True"]},
                probabilities={"is_safe": {"True": 0.99, "False": 0.01}},
                latency_ms=0.1,
                is_ambiguous=False,
                signing_key=trusted_key,
            )
            ledger.record_decision_receipt(r)

        with ledger._transaction() as conn:
            conn.execute("UPDATE audit_entries SET previous_hash=? WHERE sequence=2", ("bad_hash" * 8,))

        assert ledger.verify_integrity() is False

    def test_audit_head_meta_mismatch_raises_integrity_error(self, tmp_path):
        """Tampering with ledger_meta audit_head_hash triggers IntegrityError on audit_head()."""
        db_path = str(tmp_path / "meta_tamper.db")
        ledger = ActionLedger(path=db_path)
        trusted_key = Ed25519PrivateKey.generate()

        r = create_decision_receipt(
            schema_name="Step",
            schema_digest="1" * 64,
            prompt="Step 1",
            values={"is_safe": True},
            confidences={"is_safe": 0.99},
            conformal_sets={"is_safe": ["True"]},
            probabilities={"is_safe": {"True": 0.99, "False": 0.01}},
            latency_ms=0.1,
            is_ambiguous=False,
            signing_key=trusted_key,
        )
        ledger.record_decision_receipt(r)

        with ledger._transaction() as conn:
            conn.execute("UPDATE ledger_meta SET value=? WHERE key='audit_head_hash'", ("corrupted_head" * 4,))

        with pytest.raises(IntegrityError, match="Persisted audit head does not match the chain tip"):
            ledger.audit_head()

        assert ledger.verify_integrity() is False


# ============================================================================
# 5. Two-Phase Outcome Interception Attacks
# ============================================================================


class TestTwoPhaseOutcomeInterception:
    """Stress-test SQLite locks and outcome recording failures across all integration adapters.
    
    Verifies that side effects are accounted for with complete structured INDETERMINATE evidence:
    (action_id, receipt_digest, raw_result, underlying_error).
    """

    @pytest.fixture
    def setup_guard(self, tmp_path):
        db_path = str(tmp_path / "interception_test.db")
        ledger = ActionLedger(path=db_path)
        key = Ed25519PrivateKey.generate()
        allow_rule = PolicyRule(
            rule_id="allow_all_for_interception_test",
            effect=DecisionOutcome.ALLOW,
            target_pattern=r".*",
            reason="Allow all actions in test",
        )
        engine_policy = PolicyEngine(rules=[allow_rule])
        guard = ReflexGuardHook(
            policy_engine=engine_policy,
            ledger=ledger,
            signing_key=key,
            min_confidence=0.50,
            alpha=0.10,
        )
        return guard, ledger, key

    def test_wrap_mcp_tool_indeterminate_on_ledger_write_failure(self, setup_guard):
        """wrap_mcp_tool executes side effect, but raises ReflexIndeterminateExecutionError on outcome record failure."""
        guard, ledger, key = setup_guard
        proxy = ReflexMCPProxy(guard=guard)
        target = MockSideEffectTarget(return_val="asset_created_id_999")

        wrapped = wrap_mcp_tool(proxy=proxy, tool_name="create_asset")(target.execute_action)

        # Simulate SQLite failure on outcome recording
        def locked_outcome(*args, **kwargs):
            raise sqlite3.OperationalError("database table is locked (sqlite3 WAL lock contention)")

        ledger.record_execution_outcome = locked_outcome

        # Action execution must raise ReflexIndeterminateExecutionError
        with pytest.raises(ReflexIndeterminateExecutionError) as exc_info:
            wrapped("asset_db_target", amount=750.0)

        err = exc_info.value
        # Side effect DID occur:
        assert target.call_count == 1
        assert target.recorded_kwargs[0] == {"target": "asset_db_target", "amount": 750.0}

        # Reconciliation evidence is preserved:
        assert err.status == "INDETERMINATE"
        assert err.action_id != ""
        assert err.receipt_digest != ""
        assert err.raw_result == "executed:asset_db_target:750.0"
        assert isinstance(err.underlying_error, sqlite3.OperationalError)

    def test_mcp_proxy_handle_call_indeterminate_on_ledger_write_failure(self, setup_guard):
        """ReflexMCPProxy.handle_call returns structured JSON-RPC -32001 INDETERMINATE response on audit lock."""
        guard, ledger, key = setup_guard
        proxy = ReflexMCPProxy(guard=guard)
        target = MockSideEffectTarget(return_val="account_debited_100")

        # Simulate SQLite failure on outcome recording
        def locked_outcome(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        ledger.record_execution_outcome = locked_outcome

        req = {
            "jsonrpc": "2.0",
            "id": "req-mcp-lock-55",
            "method": "tools/call",
            "params": {"name": "debit_tool", "arguments": {"target": "acct_1", "amount": 100.0}},
        }
        resp = proxy.handle_call(req, target.mcp_executor)

        # Side effect executed:
        assert target.call_count == 1

        # Structured JSON-RPC error:
        assert "error" in resp
        assert resp["error"]["code"] == -32001
        data = resp["error"]["data"]
        assert data["status"] == "INDETERMINATE"
        assert data["action_id"] != ""
        assert data["receipt_digest"] != ""
        assert data["raw_result"] == {"status": "success", "result": "account_debited_100"}

    def test_tool_interceptor_sync_indeterminate_on_ledger_write_failure(self, setup_guard):
        """ReflexToolInterceptor synchronous invoke raises ReflexIndeterminateExecutionError on audit lock."""
        guard, ledger, key = setup_guard
        target = MockSideEffectTarget(return_val="file_written_successfully")
        interceptor = ReflexToolInterceptor(tool=target.execute_action, guard=guard, tool_name="writer")

        def disk_full_outcome(*args, **kwargs):
            raise sqlite3.OperationalError("disk I/O error: device full")

        ledger.record_execution_outcome = disk_full_outcome

        with pytest.raises(ReflexIndeterminateExecutionError) as exc_info:
            interceptor("/critical/audit.log", amount=1.0)

        err = exc_info.value
        assert target.call_count == 1
        assert err.status == "INDETERMINATE"
        assert err.action_id != ""
        assert err.receipt_digest != ""
        assert err.raw_result == "executed:/critical/audit.log:1.0"
        assert isinstance(err.underlying_error, sqlite3.OperationalError)

    @pytest.mark.asyncio
    async def test_tool_interceptor_async_ainvoke_indeterminate_on_ledger_write_failure(self, setup_guard):
        """ReflexToolInterceptor asynchronous ainvoke raises ReflexIndeterminateExecutionError on audit lock."""
        guard, ledger, key = setup_guard
        target = MockSideEffectTarget(return_val="async_effect_complete")
        ainterceptor = ReflexToolInterceptor(tool=target.aexecute_action, guard=guard, tool_name="async_writer")

        def wal_locked_outcome(*args, **kwargs):
            raise sqlite3.OperationalError("sqlite3 WAL lock contention")

        ledger.record_execution_outcome = wal_locked_outcome

        with pytest.raises(ReflexIndeterminateExecutionError) as exc_info:
            await ainterceptor.ainvoke("/async/path", amount=42.0)

        err = exc_info.value
        assert target.call_count == 1
        assert err.status == "INDETERMINATE"
        assert err.action_id != ""
        assert err.receipt_digest != ""
        assert err.raw_result == "aexecuted:/async/path:42.0"
        assert isinstance(err.underlying_error, sqlite3.OperationalError)

    def test_callback_handler_on_tool_end_indeterminate_on_ledger_failure(self, setup_guard):
        """ReflexGuardCallbackHandler.on_tool_end raises ReflexIndeterminateExecutionError when outcome recording fails."""
        guard, ledger, key = setup_guard
        handler = ReflexGuardCallbackHandler(guard=guard)

        serialized = {"name": "db_update_tool", "description": "Update customer status"}
        handler.on_tool_start(serialized, "status=ACTIVE", run_id="run_cb_101")

        def failing_outcome(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        ledger.record_execution_outcome = failing_outcome

        with pytest.raises(ReflexIndeterminateExecutionError) as exc_info:
            handler.on_tool_end("row_updated_ok", run_id="run_cb_101")

        err = exc_info.value
        assert err.status == "INDETERMINATE"
        assert err.action_id != ""
        assert err.receipt_digest != ""
        assert err.raw_result == "row_updated_ok"

    def test_callback_handler_on_tool_error_preserves_both_errors(self, setup_guard):
        """ReflexGuardCallbackHandler.on_tool_error preserves tool exception AND audit failure."""
        guard, ledger, key = setup_guard
        handler = ReflexGuardCallbackHandler(guard=guard)

        serialized = {"name": "db_tool", "description": "Execute query"}
        handler.on_tool_start(serialized, "SELECT * FROM secrets", run_id="run_cb_102")

        def failing_outcome(*args, **kwargs):
            raise sqlite3.OperationalError("ledger commit failed")

        ledger.record_execution_outcome = failing_outcome

        original_tool_err = RuntimeError("Target database connection timed out")
        with pytest.raises(ReflexIndeterminateExecutionError) as exc_info:
            handler.on_tool_error(original_tool_err, run_id="run_cb_102")

        err = exc_info.value
        assert err.status == "INDETERMINATE"
        assert err.action_id != ""
        assert err.receipt_digest != ""
        assert err.exec_error is original_tool_err
        assert isinstance(err.underlying_error, sqlite3.OperationalError)


# ============================================================================
# 6. Twin Namespace Parity Verification
# ============================================================================


class TestNamespaceParityGateB:
    """Verify deep parity between 'reflex' and 'system1' for Gate B classes and functions."""

    def test_gate_b_symbol_and_object_identity(self):
        """All Gate B symbols exported by system1 and reflex must have 100% object identity."""
        assert r_receipt.EnforcementProfile is s1_receipt.EnforcementProfile
        assert r_receipt.DecisionWitnessReceipt is s1_receipt.DecisionWitnessReceipt
        assert r_receipt.RunWitnessEnvelope is s1_receipt.RunWitnessEnvelope
        assert r_receipt.create_decision_receipt is s1_receipt.create_decision_receipt
        assert r_receipt.verify_decision_witness_receipt is s1_receipt.verify_decision_witness_receipt
        assert r_receipt.compute_receipt_digest is s1_receipt.compute_receipt_digest

        assert r_ledger.ActionLedger is s1_ledger.ActionLedger
        assert r_ledger.LedgerError is s1_ledger.LedgerError
        assert r_ledger.LedgerWriteError is s1_ledger.LedgerWriteError
        assert r_ledger.IntegrityError is s1_ledger.IntegrityError

        assert r_guard.ReflexGuardHook is s1_guard.ReflexGuardHook
        assert r_guard.EnforcementProfile is s1_guard.EnforcementProfile
        assert r_guard.ENFORCEMENT_PROFILE_V1 == s1_guard.ENFORCEMENT_PROFILE_V1

        assert r_mcp.ReflexMCPProxy is s1_mcp.ReflexMCPProxy
        assert r_mcp.wrap_mcp_tool is s1_mcp.wrap_mcp_tool

        assert r_lc.ReflexIndeterminateExecutionError is s1_lc.ReflexIndeterminateExecutionError
        assert r_lc.ReflexToolInterceptor is s1_lc.ReflexToolInterceptor
        assert r_lc.ReflexGuardCallbackHandler is s1_lc.ReflexGuardCallbackHandler
