"""Reflex Proof-Carrying Cryptographic Decision Receipts.

Self-contained cryptographic decision verification using Ed25519 signatures,
deterministic canonical JSON hashing, and tamper-evident RunWitnessEnvelope structures.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC, Set as SetABC
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

DECISION_WITNESS_PROFILE = "reflex.witness.decision.v1"
WITNESS_PROFILES = {
    "diagnostic_local": "reflex.witness.diagnostic_local.v1",
    "product_signed_v1": "reflex.witness.product_signed.v1",
}


# ============================================================================
# Canonical Cryptographic Primitives & JSON Serialization
# ============================================================================

def utc_now() -> str:
    """Return a canonical UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="microseconds")


def freeze(value: Any) -> Any:
    """Recursively freeze JSON-shaped data."""
    if isinstance(value, MappingABC):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical mappings require string keys")
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical data cannot contain NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool, bytes, Enum)):
        return value
    if is_dataclass(value):
        return value
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def thaw(value: Any) -> Any:
    """Convert immutable data into ordinary JSON-shaped values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, datetime):
        aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return aware.astimezone(UTC).isoformat(timespec="microseconds")
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return thaw(value.to_dict())
    if is_dataclass(value):
        return {f.name: thaw(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, MappingABC):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical mappings require string keys")
        return {key: thaw(item) for key, item in sorted(value.items())}
    if isinstance(value, SetABC) and not isinstance(value, (str, bytes)):
        items = [thaw(item) for item in value]
        return sorted(items, key=lambda item: canonical_json(item))
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        return [thaw(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical data cannot contain NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize a value using deterministic canonical JSON representation."""
    return json.dumps(
        thaw(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    """UTF-8 bytes of canonical JSON representation."""
    return canonical_json(value).encode("utf-8")


def fingerprint(value: Any) -> str:
    """SHA-256 hex digest of canonical JSON bytes."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _canonical_probabilities(
    probs: Optional[Mapping[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """Normalizes and canonicalizes probability distributions deterministically."""
    if not probs or not isinstance(probs, MappingABC):
        return {}
    result: Dict[str, Dict[str, float]] = {}
    for field_name, p_dist in sorted(probs.items()):
        if isinstance(p_dist, MappingABC):
            result[str(field_name)] = {
                str(opt): round(float(p), 12) for opt, p in sorted(p_dist.items())
            }
    return result


def public_key_bytes(key: Ed25519PrivateKey | Ed25519PublicKey) -> bytes:
    """Extract raw 32 bytes from an Ed25519 private or public key."""
    public = key.public_key() if isinstance(key, Ed25519PrivateKey) else key
    return public.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def public_key_fingerprint(key: Ed25519PrivateKey | Ed25519PublicKey) -> str:
    """SHA-256 digest of raw public key bytes."""
    return hashlib.sha256(public_key_bytes(key)).hexdigest()


def load_private_key(path_or_bytes: str | Path | bytes) -> Ed25519PrivateKey:
    """Loads an Ed25519 private key from PEM bytes or file path."""
    raw = path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_bytes()
    key = serialization.load_pem_private_key(raw, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("Signing key must be an Ed25519 private key")
    return key


def load_public_key(path_or_bytes: str | Path | bytes) -> Ed25519PublicKey:
    """Loads an Ed25519 public key from PEM bytes, file path, or raw bytes."""
    if isinstance(path_or_bytes, (str, Path)) and Path(path_or_bytes).is_file():
        raw = Path(path_or_bytes).read_bytes()
    elif isinstance(path_or_bytes, str):
        raw = path_or_bytes.encode("utf-8")
    else:
        raw = path_or_bytes

    if len(raw) == 32:
        return Ed25519PublicKey.from_public_bytes(raw)
    try:
        key = serialization.load_pem_public_key(raw)
        if isinstance(key, Ed25519PublicKey):
            return key
    except Exception:
        pass
    try:
        hex_clean = raw.decode("utf-8").strip()
        if len(hex_clean) == 64:
            return Ed25519PublicKey.from_public_bytes(bytes.fromhex(hex_clean))
    except Exception:
        pass
    raise TypeError("Trusted key must be an Ed25519 public key")


def sign_payload(payload: Any, key: Ed25519PrivateKey) -> str:
    """Signs canonical payload bytes with an Ed25519 private key, returning Base64 string."""
    return base64.b64encode(key.sign(canonical_bytes(payload))).decode("ascii")


def verify_payload(
    payload: Any,
    signature: str,
    key: Ed25519PublicKey | bytes | str,
) -> bool:
    """Verifies an Ed25519 signature against canonical payload bytes."""
    try:
        if isinstance(key, str):
            clean = key.strip()
            if len(clean) == 64:
                public = Ed25519PublicKey.from_public_bytes(bytes.fromhex(clean))
            else:
                public = Ed25519PublicKey.from_public_bytes(base64.b64decode(clean, validate=True))
        elif isinstance(key, bytes):
            public = Ed25519PublicKey.from_public_bytes(key)
        else:
            public = key
        public.verify(base64.b64decode(signature, validate=True), canonical_bytes(payload))
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


# ============================================================================
# RunWitnessEnvelope
# ============================================================================

@dataclass(frozen=True)
class RunWitnessEnvelope:
    """Cryptographic evidence envelope binding mutation intent, observations, and signatures."""

    schema: str = "reflex.witness.product_signed.v1"
    mutation_intent: Dict[str, Any] = field(default_factory=dict)
    guard_receipt: Dict[str, Any] = field(default_factory=dict)
    effect_observation: Dict[str, Any] = field(default_factory=dict)
    readback_observation: Dict[str, Any] = field(default_factory=dict)
    truth_ledger_head: str = ""
    artifact_checksums: Dict[str, str] = field(default_factory=dict)
    closure_receipt_fingerprint: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    signature: Optional[str] = None
    signer_fingerprint: Optional[str] = None
    profile: str = "product_signed_v1"

    @property
    def envelope_digest(self) -> str:
        return self.compute_witness_digest()

    def unsigned_payload(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "profile": self.profile,
            "timestamp": self.timestamp,
            "mutation_intent": self.mutation_intent,
            "guard_receipt": self.guard_receipt,
            "effect_observation": self.effect_observation,
            "readback_observation": self.readback_observation,
            "truth_ledger_head": self.truth_ledger_head,
            "artifact_checksums": self.artifact_checksums,
            "closure_receipt_fingerprint": self.closure_receipt_fingerprint,
        }

    def compute_witness_digest(self) -> str:
        payload = self.unsigned_payload()
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        digest = self.compute_witness_digest()
        d["witness_digest"] = digest
        d["envelope_digest"] = digest
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RunWitnessEnvelope:
        valid_field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_field_names}
        return cls(**filtered)


def create_run_witness_envelope(
    mutation_intent: Dict[str, Any],
    guard_receipt: Dict[str, Any],
    effect_observation: Dict[str, Any],
    readback_observation: Dict[str, Any],
    truth_ledger_head: str,
    artifact_checksums: Dict[str, str],
    closure_receipt_fingerprint: str,
    *,
    profile: str = "product_signed_v1",
    signing_key: Optional[Ed25519PrivateKey] = None,
) -> RunWitnessEnvelope:
    """Constructs and optionally signs a RunWitnessEnvelope."""
    if profile == "product_signed_v1" and signing_key is None:
        raise ValueError("product_signed_v1 profile requires an Ed25519 signing_key")

    if effect_observation and "effect_digest" not in effect_observation:
        effect_observation = dict(effect_observation)
        effect_observation["effect_digest"] = hashlib.sha256(
            json.dumps(effect_observation, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
    if readback_observation and "readback_digest" not in readback_observation:
        readback_observation = dict(readback_observation)
        readback_observation["readback_digest"] = hashlib.sha256(
            json.dumps(readback_observation, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    schema = WITNESS_PROFILES.get(profile, "reflex.witness.product_signed.v1")
    envelope = RunWitnessEnvelope(
        schema=schema,
        profile=profile,
        mutation_intent=mutation_intent,
        guard_receipt=guard_receipt,
        effect_observation=effect_observation,
        readback_observation=readback_observation,
        truth_ledger_head=truth_ledger_head,
        artifact_checksums=artifact_checksums,
        closure_receipt_fingerprint=closure_receipt_fingerprint,
    )
    if signing_key is not None:
        envelope = sign_run_witness_envelope(envelope, signing_key)
    return envelope


def sign_run_witness_envelope(
    envelope: RunWitnessEnvelope,
    signing_key: Ed25519PrivateKey,
) -> RunWitnessEnvelope:
    """Signs the unsigned payload of a RunWitnessEnvelope using Ed25519."""
    payload = envelope.unsigned_payload()
    sig = sign_payload(payload, signing_key)
    fp = public_key_fingerprint(signing_key)
    return RunWitnessEnvelope(
        schema=envelope.schema,
        profile=envelope.profile,
        mutation_intent=envelope.mutation_intent,
        guard_receipt=envelope.guard_receipt,
        effect_observation=envelope.effect_observation,
        readback_observation=envelope.readback_observation,
        truth_ledger_head=envelope.truth_ledger_head,
        artifact_checksums=envelope.artifact_checksums,
        closure_receipt_fingerprint=envelope.closure_receipt_fingerprint,
        timestamp=envelope.timestamp,
        signature=sig,
        signer_fingerprint=fp,
    )


def verify_run_witness_envelope(envelope: Dict[str, Any] | RunWitnessEnvelope) -> Tuple[bool, str]:
    """Fast-path verification checking schema conformance and internal hash consistency."""
    if isinstance(envelope, RunWitnessEnvelope):
        env_dict = envelope.to_dict()
    elif isinstance(envelope, dict):
        env_dict = envelope
    else:
        return False, f"Expected RunWitnessEnvelope or dict, got {type(envelope).__name__}"

    obj = RunWitnessEnvelope.from_dict(env_dict)
    expected_digest = obj.compute_witness_digest()
    if env_dict.get("witness_digest") and env_dict["witness_digest"] != expected_digest:
        return False, "envelope witness_digest mismatch"
    if env_dict.get("envelope_digest") and env_dict["envelope_digest"] != expected_digest:
        return False, "envelope envelope_digest mismatch"
    return True, "valid"


# ============================================================================
# DecisionWitnessReceipt
# ============================================================================

@dataclass(frozen=True)
class DecisionWitnessReceipt:
    """Proof-carrying cryptographic decision receipt."""

    decision_id: str
    schema_name: str
    schema_digest: str
    prompt: str
    prompt_digest: str
    values: Dict[str, Any]
    confidences: Dict[str, float]
    conformal_sets: Dict[str, List[str]]
    probabilities: Dict[str, Dict[str, float]]
    latency_ms: float
    is_ambiguous: bool
    timestamp: str
    truth_ledger_head: str = ""
    ledger_record_id: Optional[str] = None
    signer_public_key: Optional[str] = None
    envelope: Optional[RunWitnessEnvelope] = None

    def unsigned_payload(self) -> Dict[str, Any]:
        """Returns deterministic dictionary representation for canonical hashing."""
        return {
            "decision_id": self.decision_id,
            "schema_name": self.schema_name,
            "schema_digest": self.schema_digest,
            "prompt_digest": self.prompt_digest,
            "values": self.values,
            "confidences": self.confidences,
            "conformal_sets": self.conformal_sets,
            "probabilities": _canonical_probabilities(self.probabilities),
            "latency_ms": round(self.latency_ms, 4),
            "is_ambiguous": self.is_ambiguous,
            "timestamp": self.timestamp,
            "truth_ledger_head": self.truth_ledger_head,
        }

    def compute_digest(self) -> str:
        """Computes SHA-256 hash over the canonical unsigned payload."""
        return compute_receipt_digest(self.unsigned_payload())

    @property
    def digest(self) -> str:
        """Convenience property for compute_digest()."""
        return self.compute_digest()

    @property
    def receipt_id(self) -> str:
        """Convenience alias for decision_id."""
        return self.decision_id

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "decision_id": self.decision_id,
            "schema_name": self.schema_name,
            "schema_digest": self.schema_digest,
            "prompt": self.prompt,
            "prompt_digest": self.prompt_digest,
            "values": self.values,
            "confidences": self.confidences,
            "conformal_sets": self.conformal_sets,
            "probabilities": self.probabilities,
            "latency_ms": self.latency_ms,
            "is_ambiguous": self.is_ambiguous,
            "timestamp": self.timestamp,
            "truth_ledger_head": self.truth_ledger_head,
            "ledger_record_id": self.ledger_record_id,
            "receipt_digest": self.compute_digest(),
        }
        if self.signer_public_key is not None:
            d["signer_public_key"] = self.signer_public_key
        if self.envelope is not None:
            d["envelope"] = self.envelope.to_dict()
        return d


def compute_receipt_digest(receipt_data: Mapping[str, Any]) -> str:
    """Computes canonical SHA-256 digest from a receipt dictionary or unsigned payload."""
    payload = {
        "decision_id": str(receipt_data.get("decision_id", "")),
        "schema_name": str(receipt_data.get("schema_name", "")),
        "schema_digest": str(receipt_data.get("schema_digest", "")),
        "prompt_digest": str(receipt_data.get("prompt_digest", "")),
        "values": receipt_data.get("values", {}),
        "confidences": {k: float(v) for k, v in (receipt_data.get("confidences") or {}).items()},
        "conformal_sets": {k: list(v) for k, v in (receipt_data.get("conformal_sets") or {}).items()},
        "probabilities": _canonical_probabilities(receipt_data.get("probabilities")),
        "latency_ms": round(float(receipt_data.get("latency_ms", 0.0)), 4),
        "is_ambiguous": bool(receipt_data.get("is_ambiguous", False)),
        "timestamp": str(receipt_data.get("timestamp", "")),
        "truth_ledger_head": str(receipt_data.get("truth_ledger_head", "")),
    }
    canonical_bytes_val = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(canonical_bytes_val).hexdigest()


def create_decision_receipt(
    *,
    schema_name: str,
    schema_digest: str,
    prompt: str,
    values: Mapping[str, Any],
    confidences: Mapping[str, float],
    conformal_sets: Mapping[str, Sequence[str]],
    probabilities: Mapping[str, Mapping[str, float]],
    latency_ms: float,
    is_ambiguous: bool,
    truth_ledger_head: str = "",
    ledger_record_id: Optional[str] = None,
    signing_key: Optional[Ed25519PrivateKey] = None,
) -> DecisionWitnessReceipt:
    """Builds a DecisionWitnessReceipt and generates a signed RunWitnessEnvelope."""
    decision_id = f"dec_{os.urandom(16).hex()}"
    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    ts = datetime.now(UTC).isoformat()

    unbound_receipt = DecisionWitnessReceipt(
        decision_id=decision_id,
        schema_name=schema_name,
        schema_digest=schema_digest,
        prompt=prompt,
        prompt_digest=prompt_digest,
        values=dict(values),
        confidences={k: float(v) for k, v in confidences.items()},
        conformal_sets={k: list(v) for k, v in conformal_sets.items()},
        probabilities={k: {str(opt): float(p) for opt, p in p_dist.items()} for k, p_dist in probabilities.items()},
        latency_ms=float(latency_ms),
        is_ambiguous=bool(is_ambiguous),
        timestamp=ts,
        truth_ledger_head=truth_ledger_head,
        ledger_record_id=ledger_record_id,
        envelope=None,
    )

    envelope_payload_mutation = {
        "decision_id": decision_id,
        "schema_name": schema_name,
        "schema_digest": schema_digest,
        "prompt_digest": prompt_digest,
        "values": dict(values),
    }

    envelope_payload_guard = {
        "confidences": {k: float(v) for k, v in confidences.items()},
        "conformal_sets": {k: list(v) for k, v in conformal_sets.items()},
        "probabilities": _canonical_probabilities(probabilities),
        "is_ambiguous": is_ambiguous,
        "latency_ms": round(float(latency_ms), 4),
    }

    signer_pub_hex: Optional[str] = None
    if signing_key is not None:
        signer_pub_hex = public_key_bytes(signing_key).hex()

    envelope = create_run_witness_envelope(
        mutation_intent=envelope_payload_mutation,
        guard_receipt=envelope_payload_guard,
        effect_observation={
            "type": "reflex_decision_evaluation",
            **({"signer_public_key": signer_pub_hex} if signer_pub_hex else {}),
        },
        readback_observation={"evaluated": True},
        truth_ledger_head=truth_ledger_head,
        artifact_checksums={"schema_digest": schema_digest, "prompt_digest": prompt_digest},
        closure_receipt_fingerprint=unbound_receipt.compute_digest(),
        profile="product_signed_v1" if signing_key is not None else "diagnostic_local",
        signing_key=signing_key,
    )

    return DecisionWitnessReceipt(
        decision_id=decision_id,
        schema_name=schema_name,
        schema_digest=schema_digest,
        prompt=prompt,
        prompt_digest=prompt_digest,
        values=dict(values),
        confidences={k: float(v) for k, v in confidences.items()},
        conformal_sets={k: list(v) for k, v in conformal_sets.items()},
        probabilities={k: {str(opt): float(p) for opt, p in p_dist.items()} for k, p_dist in probabilities.items()},
        latency_ms=float(latency_ms),
        is_ambiguous=bool(is_ambiguous),
        timestamp=ts,
        truth_ledger_head=truth_ledger_head,
        ledger_record_id=ledger_record_id,
        signer_public_key=signer_pub_hex,
        envelope=envelope,
    )


def verify_decision_witness_receipt(
    receipt_data: Mapping[str, Any],
    public_key: Optional[Union[Ed25519PublicKey, str, bytes]] = None,
) -> bool:
    """Independently verifies an emitted decision receipt and its signed envelope."""
    envelope_data = receipt_data.get("envelope")
    if not envelope_data:
        raise ValueError("Receipt contains no RunWitnessEnvelope")

    envelope = RunWitnessEnvelope.from_dict(envelope_data)

    # 1. Verify envelope digest
    expected_digest = envelope.compute_witness_digest()
    if envelope.envelope_digest != expected_digest:
        return False
    if "witness_digest" in envelope_data and envelope_data["witness_digest"] != expected_digest:
        return False
    if "envelope_digest" in envelope_data and envelope_data["envelope_digest"] != expected_digest:
        return False

    # 2. Verify Ed25519 signature and enforce authentication profile
    if envelope.signature is None or envelope.signature == "":
        # Unsigned envelope: reject if caller expected a signed receipt or if profile requires signing
        if public_key is not None:
            return False
        if envelope.profile == "product_signed_v1":
            return False
        if envelope.profile != "diagnostic_local":
            return False
    else:
        # Signed envelope
        key_obj: Optional[Ed25519PublicKey] = None
        if public_key is not None:
            if isinstance(public_key, Ed25519PublicKey):
                key_obj = public_key
            elif isinstance(public_key, str):
                pub_clean = public_key.strip()
                if len(pub_clean) == 64:
                    try:
                        key_obj = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_clean))
                    except Exception:
                        pass
                if key_obj is None:
                    try:
                        key_obj = load_public_key(pub_clean)
                    except Exception:
                        pass
            elif isinstance(public_key, bytes):
                if len(public_key) == 32:
                    try:
                        key_obj = Ed25519PublicKey.from_public_bytes(public_key)
                    except Exception:
                        pass
                else:
                    try:
                        key_obj = load_public_key(public_key)
                    except Exception:
                        pass
            else:
                try:
                    key_obj = load_public_key(public_key)
                except Exception:
                    pass

            if key_obj is None:
                raise ValueError("Invalid public_key provided")
        else:
            pub_hex = (
                receipt_data.get("signer_public_key")
                or receipt_data.get("public_key")
                or (envelope.effect_observation.get("signer_public_key") if envelope.effect_observation else None)
            )
            if pub_hex:
                try:
                    key_obj = Ed25519PublicKey.from_public_bytes(bytes.fromhex(str(pub_hex)))
                except Exception:
                    pass

            if key_obj is None:
                raise ValueError("Receipt is cryptographically signed; valid public_key is required for verification")

        # Anti-substitution check: if signer_public_key is declared in receipt, it must match key_obj
        if receipt_data.get("signer_public_key"):
            try:
                declared_bytes = bytes.fromhex(str(receipt_data["signer_public_key"]))
                if declared_bytes != public_key_bytes(key_obj):
                    return False
            except Exception:
                return False

        # Anti-substitution check: if effect_observation declares signer_public_key, it must match key_obj
        if envelope.effect_observation and envelope.effect_observation.get("signer_public_key"):
            try:
                obs_bytes = bytes.fromhex(str(envelope.effect_observation["signer_public_key"]))
                if obs_bytes != public_key_bytes(key_obj):
                    return False
            except Exception:
                return False

        unsigned = envelope.unsigned_payload()
        if not verify_payload(unsigned, envelope.signature, key_obj):
            return False
        if envelope.signer_fingerprint != public_key_fingerprint(key_obj):
            return False

    # 3. Verify cross-field integrity between receipt and envelope
    mutation = envelope.mutation_intent
    if mutation.get("decision_id") != receipt_data.get("decision_id"):
        return False
    if mutation.get("schema_name") and mutation.get("schema_name") != receipt_data.get("schema_name"):
        return False
    if mutation.get("schema_digest") != receipt_data.get("schema_digest"):
        return False
    if mutation.get("prompt_digest") != receipt_data.get("prompt_digest"):
        return False
    if "values" in mutation and mutation.get("values") != receipt_data.get("values"):
        return False

    guard = envelope.guard_receipt or {}
    if "confidences" in guard and guard.get("confidences") != receipt_data.get("confidences"):
        return False
    if "conformal_sets" in guard and guard.get("conformal_sets") != receipt_data.get("conformal_sets"):
        return False
    if "is_ambiguous" in guard and bool(guard.get("is_ambiguous")) != bool(receipt_data.get("is_ambiguous")):
        return False
    if "probabilities" in guard:
        if _canonical_probabilities(guard.get("probabilities")) != _canonical_probabilities(receipt_data.get("probabilities")):
            return False

    # 4. Verify prompt hash integrity if prompt text is present
    if "prompt" in receipt_data and "prompt_digest" in receipt_data:
        computed_prompt_hash = hashlib.sha256(str(receipt_data["prompt"]).encode("utf-8")).hexdigest()
        if computed_prompt_hash != receipt_data["prompt_digest"]:
            return False

    # 5. Verify closure receipt fingerprint matches receipt payload
    computed_receipt_digest = compute_receipt_digest(receipt_data)
    if envelope.closure_receipt_fingerprint is not None:
        if envelope.closure_receipt_fingerprint != computed_receipt_digest:
            return False
    if "receipt_digest" in receipt_data:
        if receipt_data["receipt_digest"] != computed_receipt_digest:
            return False

    return True


def save_private_key(key: Ed25519PrivateKey, path: str | Path) -> None:
    """Saves an Ed25519 private key in PEM format to a file with owner-only (0o600) permissions."""
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p.parent, 0o700)
    except OSError:
        pass
    p.write_bytes(pem)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


def save_public_key(key: Ed25519PublicKey | Ed25519PrivateKey, path: str | Path) -> None:
    """Saves an Ed25519 public key in PEM format to a file."""
    pub = key.public_key() if isinstance(key, Ed25519PrivateKey) else key
    pem = pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pem)


def save_keypair(key: Ed25519PrivateKey, directory: str | Path) -> Tuple[Path, Path]:
    """Saves identity.key and identity.pub in the specified directory."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    key_path = d / "identity.key"
    pub_path = d / "identity.pub"
    save_private_key(key, key_path)
    save_public_key(key, pub_path)
    return key_path, pub_path


__all__ = [
    "DECISION_WITNESS_PROFILE",
    "DecisionWitnessReceipt",
    "RunWitnessEnvelope",
    "canonical_bytes",
    "canonical_json",
    "compute_receipt_digest",
    "create_decision_receipt",
    "create_run_witness_envelope",
    "fingerprint",
    "freeze",
    "load_private_key",
    "load_public_key",
    "public_key_bytes",
    "public_key_fingerprint",
    "save_keypair",
    "save_private_key",
    "save_public_key",
    "sign_payload",
    "sign_run_witness_envelope",
    "thaw",
    "utc_now",
    "verify_decision_witness_receipt",
    "verify_payload",
    "verify_run_witness_envelope",
]


