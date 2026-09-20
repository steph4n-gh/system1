"""System 1 gRPC Server — language-agnostic sidecar deployment mode.

Exposes the System 1 decision engine over gRPC so polyglot agent stacks
(TypeScript, Go, Rust) can call Decide, Guard, VerifyReceipt, and
HealthCheck without embedding the Python runtime.

Design notes:
- All gRPC imports are lazy / guarded behind ``try … except ImportError``
  so the rest of the system1 package remains importable without grpcio.
- The server implementation avoids generated protobuf stubs by default and
  instead uses grpcio's *generic-handler* API with hand-rolled
  serialization.  If compiled stubs are available they will be preferred.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent import futures
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("reflex.grpc")

# ---------------------------------------------------------------------------
# Lazy gRPC imports — the module stays importable even without grpcio.
# ---------------------------------------------------------------------------
try:
    import grpc
    from grpc import ServicerContext, StatusCode

    _GRPC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GRPC_AVAILABLE = False

# ---------------------------------------------------------------------------
# Lazy protobuf message imports — we generate stubs at build time but fall
# back to manual JSON serialisation when stubs are missing.
# ---------------------------------------------------------------------------
_STUBS_AVAILABLE = False
try:
    from system1.proto import system1_pb2, system1_pb2_grpc  # type: ignore[attr-defined]

    _STUBS_AVAILABLE = True
except Exception:
    pass


def grpc_available() -> bool:
    """Return ``True`` if ``grpcio`` is installed."""
    return _GRPC_AVAILABLE


# ---------------------------------------------------------------------------
# Schema registry helpers
# ---------------------------------------------------------------------------

def _load_schema_by_name(name: str, custom_schemas: Optional[Dict[str, Any]] = None):
    """Resolve a schema name to a ``DecisionSchema`` instance."""
    from system1.guard import DefaultGuardDecisionSchema
    from system1.cli import DefaultTriageSchema

    if custom_schemas and name in custom_schemas:
        return custom_schemas[name]

    builtins = {
        "guard": DefaultGuardDecisionSchema,
        "guardrail": DefaultGuardDecisionSchema,
        "triage": DefaultTriageSchema,
        "default": DefaultTriageSchema,
    }
    if name.lower() in builtins:
        return builtins[name.lower()]()

    raise ValueError("Unknown schema name; register custom schemas at server startup")


# ---------------------------------------------------------------------------
# Servicer implementation
# ---------------------------------------------------------------------------

_BaseServicer = system1_pb2_grpc.SystemOneServiceServicer if _STUBS_AVAILABLE else object


class SystemOneServiceServicer(_BaseServicer):
    """Implements the four SystemOneService RPCs on top of the in-process engine."""

    def __init__(
        self,
        schemas: Optional[Dict[str, Any]] = None,
        signing_key: Optional[Any] = None,
        ledger: Optional[Any] = None,
        policy_engine: Optional[Any] = None,
    ) -> None:
        from system1.engine import SystemOneEngine
        from system1.guard import SystemOneGuardHook

        self._custom_schemas = dict(schemas or {})
        self._signing_key = signing_key
        self._ledger = ledger
        self._policy_engine = policy_engine
        self._engines: Dict[str, SystemOneEngine] = {}
        self._guard_hooks: Dict[str, SystemOneGuardHook] = {}

        # Pre-warm engines for explicitly registered schemas.
        for name, schema in self._custom_schemas.items():
            self._engines[name] = SystemOneEngine(
                schema,
                signing_key=self._signing_key,
                ledger=self._ledger,
            )

    # -- helpers ----------------------------------------------------------

    def _get_engine(self, schema_name: str):
        from system1.engine import SystemOneEngine

        key = self._schema_key(schema_name, "triage")
        if key not in self._engines:
            schema = _load_schema_by_name(key, self._custom_schemas)
            self._engines[key] = SystemOneEngine(
                schema,
                signing_key=self._signing_key,
                ledger=self._ledger,
            )
        return self._engines[key]

    def _get_guard_hook(self, schema_name: str):
        from system1.guard import SystemOneGuardHook

        key = self._schema_key(schema_name, "guard")
        if key not in self._guard_hooks:
            engine = self._get_engine(key)
            self._guard_hooks[key] = SystemOneGuardHook(
                engine=engine,
                signing_key=self._signing_key,
                ledger=self._ledger,
                policy=self._policy_engine,
            )
        return self._guard_hooks[key]

    def _schema_key(self, name: str, default: str) -> str:
        """Keep remote names within the finite startup registry, including aliases."""
        key = name or default
        if key in self._custom_schemas:
            return key
        aliases = {"triage": "triage", "default": "triage", "guard": "guard", "guardrail": "guard"}
        canonical = aliases.get(key.lower())
        if canonical is None:
            raise ValueError("Unknown schema name; register custom schemas at server startup")
        return canonical

    # -- Decide -----------------------------------------------------------

    def Decide(self, request, context):
        """Evaluate a prompt against a typed decision schema."""
        is_proto = _is_proto_call(request, context)
        try:
            prompt = request.prompt if hasattr(request, "prompt") else request.get("prompt", "")
            schema_name = (request.schema_name if hasattr(request, "schema_name") else request.get("schema_name", "")) or "triage"
            alpha = (request.alpha if hasattr(request, "alpha") else request.get("alpha", 0.0)) or 0.05
            telemetry_raw = request.telemetry if hasattr(request, "telemetry") else request.get("telemetry", {})

            if not prompt:
                context.set_code(StatusCode.INVALID_ARGUMENT)
                context.set_details("prompt is required")
                return _empty_decide_response(as_protobuf=is_proto)

            engine = self._get_engine(schema_name)
            telemetry = dict(telemetry_raw) if telemetry_raw else None

            result = engine.decide(prompt, alpha=alpha, telemetry=telemetry, record_receipt=True)

            return _build_decide_response(result, as_protobuf=is_proto)

        except Exception as exc:
            logger.exception("Decide RPC failed")
            context.set_code(StatusCode.INTERNAL)
            context.set_details(str(exc))
            return _empty_decide_response(as_protobuf=is_proto)

    # -- Guard ------------------------------------------------------------

    def Guard(self, request, context):
        """Run a prompt through the fail-closed guard reference monitor."""
        is_proto = _is_proto_call(request, context)
        t0 = time.perf_counter()
        try:
            prompt = request.prompt if hasattr(request, "prompt") else request.get("prompt", "")
            schema_name = (request.schema_name if hasattr(request, "schema_name") else request.get("schema_name", "")) or "guard"
            alpha = (request.alpha if hasattr(request, "alpha") else request.get("alpha", 0.0)) or 0.05

            if not prompt:
                context.set_code(StatusCode.INVALID_ARGUMENT)
                context.set_details("prompt is required")
                return _empty_guard_response(as_protobuf=is_proto)

            guard = self._get_guard_hook(schema_name)
            guard.alpha = alpha
            from system1.guard import ActionProposal, DecisionOutcome

            proposal = ActionProposal.create(
                tenant_id="default",
                principal_id="grpc_client",
                scope="execution",
                tool=schema_name,
                arguments={"prompt": prompt},
                purpose=prompt or f"Evaluate proposal for {schema_name}",
            )

            res = guard.evaluate_proposal(proposal, context_prompt=prompt)

            if res.outcome == DecisionOutcome.ALLOW:
                outcome_str = _GUARD_OUTCOME_ALLOW
            elif res.outcome == DecisionOutcome.DENY:
                outcome_str = _GUARD_OUTCOME_DENY
            else:
                outcome_str = _GUARD_OUTCOME_REQUIRE_APPROVAL

            reason = res.reason
            decision = res.decision_result
            measured_latency_ms = max(0.001, (time.perf_counter() - t0) * 1000.0)
            if decision is None:
                from system1.engine import DecisionResult
                receipt = getattr(res, "receipt", None)
                if receipt is None and getattr(res, "policy_decision", None) is not None:
                    receipt = getattr(res.policy_decision, "receipt", None)
                if receipt is None:
                    from system1.receipt import create_decision_receipt
                    chosen_profile = (
                        "product_signed_v1"
                        if self._signing_key is not None
                        else "diagnostic_local"
                    )
                    receipt = create_decision_receipt(
                        schema_name=schema_name,
                        schema_digest=hashlib.sha256(schema_name.encode("utf-8")).hexdigest(),
                        prompt=prompt,
                        values={"policy_outcome": outcome_str},
                        confidences={"policy": 1.0},
                        conformal_sets={},
                        probabilities={},
                        latency_ms=measured_latency_ms,
                        is_ambiguous=False,
                        truth_ledger_head=self._ledger.head_hash() if self._ledger else "",
                        signing_key=self._signing_key,
                        profile=chosen_profile,
                    )
                    if self._ledger is not None:
                        try:
                            self._ledger.record_decision_receipt(receipt)
                        except Exception:
                            pass
                decision = DecisionResult(
                    schema_name=schema_name,
                    schema_digest="",
                    prompt=prompt,
                    values={"policy_outcome": outcome_str},
                    confidences={},
                    conformal_sets={},
                    probabilities={},
                    is_ambiguous=False,
                    latency_ms=measured_latency_ms,
                    alpha=alpha,
                    receipt=receipt,
                )

            return _build_guard_response(outcome_str, reason, decision, as_protobuf=is_proto)

        except Exception as exc:
            logger.exception("Guard RPC failed")
            context.set_code(StatusCode.INTERNAL)
            context.set_details(str(exc))
            return _empty_guard_response(as_protobuf=is_proto)

    # -- VerifyReceipt ----------------------------------------------------

    def VerifyReceipt(self, request, context):
        """Cryptographically verify a previously emitted decision receipt."""
        from system1.receipt import verify_decision_witness_receipt

        is_proto = _is_proto_call(request, context)
        try:
            receipt_bytes = request.receipt_json if hasattr(request, "receipt_json") else request.get("receipt_json", b"")
            pub_key_hex = (request.public_key_hex if hasattr(request, "public_key_hex") else request.get("public_key_hex", "")) or ""

            if not receipt_bytes:
                context.set_code(StatusCode.INVALID_ARGUMENT)
                context.set_details("receipt_json is required")
                return _build_verify_response(False, error="receipt_json is required", as_protobuf=is_proto)

            receipt_data = json.loads(receipt_bytes)
            pub_key = pub_key_hex if pub_key_hex else None

            verified = verify_decision_witness_receipt(receipt_data, public_key=pub_key)

            return _build_verify_response(
                verified,
                decision_id=receipt_data.get("decision_id", ""),
                receipt_digest=receipt_data.get("receipt_digest", ""),
                as_protobuf=is_proto,
            )

        except Exception as exc:
            logger.exception("VerifyReceipt RPC failed")
            return _build_verify_response(False, error=str(exc), as_protobuf=is_proto)

    # -- HealthCheck ------------------------------------------------------

    def HealthCheck(self, request, context):
        """Lightweight liveness / readiness probe."""
        is_proto = _is_proto_call(request, context)
        schemas = list(self._engines.keys()) + [
            k for k in self._custom_schemas if k not in self._engines
        ]
        return _build_health_response(schemas, as_protobuf=is_proto)


# ---------------------------------------------------------------------------
# Guard logic (mirrors SystemOneGuardHook.evaluate_proposal without needing
# a full ActionProposal object)
# ---------------------------------------------------------------------------

_GUARD_OUTCOME_ALLOW = "ALLOW"
_GUARD_OUTCOME_DENY = "DENY"
_GUARD_OUTCOME_REQUIRE_APPROVAL = "REQUIRE_APPROVAL"


def _apply_guard_logic(decision, min_confidence: float, alpha: float):
    """Return (outcome_str, reason) applying the same fail-closed rules as
    ``SystemOneGuardHook.evaluate_proposal``."""
    from system1.core import BooleanField, ChoiceField

    # 1. Safety boolean
    if "is_safe" in decision.values and not decision.values["is_safe"]:
        conf = decision.confidences.get("is_safe", 0.0)
        return _GUARD_OUTCOME_DENY, f"System 1 classified action as unsafe (confidence: {conf:.3f})"

    # 2. Conformal ambiguity / OOD
    engine_schema = None
    try:
        # access schema from the engine that produced the decision
        engine_schema = decision  # we pass schema fields via decision.conformal_sets
    except Exception:
        pass

    for field_name, cset in decision.conformal_sets.items():
        if len(cset) == 0:
            return _GUARD_OUTCOME_REQUIRE_APPROVAL, (
                f"System 1 detected out-of-distribution input for field {field_name!r} (empty conformal set)"
            )
        if len(cset) > 1:
            return _GUARD_OUTCOME_REQUIRE_APPROVAL, (
                f"System 1 conformal ambiguity for {field_name!r}: set {cset} has {len(cset)} candidates at 1-alpha={1.0 - alpha:.2f}"
            )

    # 3. Minimum confidence
    for field_name, conf in decision.confidences.items():
        if conf < min_confidence:
            return _GUARD_OUTCOME_REQUIRE_APPROVAL, (
                f"System 1 calibrated confidence {conf:.3f} for {field_name!r} below required threshold {min_confidence:.3f}"
            )

    # 4. All clear
    return _GUARD_OUTCOME_ALLOW, f"System 1 verified action with full conformal confidence in {decision.latency_ms:.2f}ms"


# ---------------------------------------------------------------------------
# Response builders — support both genuine compiled protobuf messages and
# lightweight namespace objects for testing/generic handlers.
# ---------------------------------------------------------------------------

class _SimpleNamespace:
    """Minimal attribute container that serialises to dict for JSON."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        items = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"_SimpleNamespace({items})"


class _ConformalSetMsg:
    def __init__(self, members: List[str]):
        self.members = list(members)


def _is_proto_call(request, context) -> bool:
    """Determine whether the current invocation expects a genuine compiled protobuf response."""
    if not _STUBS_AVAILABLE:
        return False
    if hasattr(request, "DESCRIPTOR"):
        return True
    req_types = (
        getattr(system1_pb2, "DecideRequest", type(None)),
        getattr(system1_pb2, "GuardRequest", type(None)),
        getattr(system1_pb2, "VerifyReceiptRequest", type(None)),
        getattr(system1_pb2, "HealthCheckRequest", type(None)),
    )
    if isinstance(request, req_types):
        return True
    if context is not None:
        ctx_cls = type(context).__name__
        if "Fake" not in ctx_cls and "Mock" not in ctx_cls and hasattr(context, "abort"):
            ctx_mod = getattr(type(context), "__module__", "")
            if "grpc" in ctx_mod:
                return True
    return False


def _empty_decide_response(as_protobuf: bool = False):
    if as_protobuf and _STUBS_AVAILABLE:
        return system1_pb2.DecideResponse()
    return _SimpleNamespace(
        schema_name="",
        schema_digest="",
        prompt="",
        values={},
        confidences={},
        conformal_sets={},
        latency_ms=0.0,
        is_ambiguous=False,
        escalated=False,
        receipt_json=b"",
        escalated_fields=[],
    )


def _build_decide_response(result, as_protobuf: bool = False):
    values = {}
    for k, v in result.values.items():
        values[k] = json.dumps(v) if not isinstance(v, str) else v

    receipt_bytes = b""
    try:
        receipt_bytes = json.dumps(result.receipt.to_dict(), default=str).encode("utf-8")
    except Exception:
        pass

    if as_protobuf and _STUBS_AVAILABLE:
        csets = {}
        for k, v in result.conformal_sets.items():
            members = [str(m) for m in v] if isinstance(v, (list, tuple, set)) else [str(v)]
            csets[k] = system1_pb2.ConformalSet(members=members)
        return system1_pb2.DecideResponse(
            schema_name=result.schema_name or "",
            schema_digest=result.schema_digest or "",
            prompt=result.prompt or "",
            values=values,
            confidences={k: float(v) for k, v in result.confidences.items()},
            conformal_sets=csets,
            latency_ms=float(result.latency_ms),
            is_ambiguous=bool(result.is_ambiguous),
            escalated=bool(result.escalated_fields),
            receipt_json=receipt_bytes,
            escalated_fields=list(result.escalated_fields),
        )

    conformal_sets = {}
    for k, v in result.conformal_sets.items():
        conformal_sets[k] = _ConformalSetMsg(v)

    return _SimpleNamespace(
        schema_name=result.schema_name,
        schema_digest=result.schema_digest,
        prompt=result.prompt,
        values=values,
        confidences=dict(result.confidences),
        conformal_sets=conformal_sets,
        latency_ms=result.latency_ms,
        is_ambiguous=result.is_ambiguous,
        escalated=bool(result.escalated_fields),
        receipt_json=receipt_bytes,
        escalated_fields=list(result.escalated_fields),
    )


def _empty_guard_response(as_protobuf: bool = False):
    if as_protobuf and _STUBS_AVAILABLE:
        return system1_pb2.GuardResponse()
    return _SimpleNamespace(
        outcome=0,
        reason="",
        values={},
        confidences={},
        conformal_sets={},
        latency_ms=0.0,
        receipt_json=b"",
    )


_OUTCOME_TO_INT = {
    _GUARD_OUTCOME_ALLOW: 1,
    _GUARD_OUTCOME_DENY: 2,
    _GUARD_OUTCOME_REQUIRE_APPROVAL: 3,
}


def _build_guard_response(outcome_str, reason, decision, as_protobuf: bool = False):
    values = {}
    for k, v in decision.values.items():
        values[k] = json.dumps(v) if not isinstance(v, str) else v

    receipt_bytes = b""
    try:
        receipt_bytes = json.dumps(decision.receipt.to_dict(), default=str).encode("utf-8")
    except Exception:
        pass

    outcome_int = _OUTCOME_TO_INT.get(outcome_str, 0)

    if as_protobuf and _STUBS_AVAILABLE:
        csets = {}
        for k, v in decision.conformal_sets.items():
            members = [str(m) for m in v] if isinstance(v, (list, tuple, set)) else [str(v)]
            csets[k] = system1_pb2.ConformalSet(members=members)
        return system1_pb2.GuardResponse(
            outcome=outcome_int,
            reason=reason or "",
            values=values,
            confidences={k: float(v) for k, v in decision.confidences.items()},
            conformal_sets=csets,
            latency_ms=float(decision.latency_ms),
            receipt_json=receipt_bytes,
        )

    conformal_sets = {}
    for k, v in decision.conformal_sets.items():
        conformal_sets[k] = _ConformalSetMsg(v)

    return _SimpleNamespace(
        outcome=outcome_int,
        reason=reason,
        values=values,
        confidences=dict(decision.confidences),
        conformal_sets=conformal_sets,
        latency_ms=decision.latency_ms,
        receipt_json=receipt_bytes,
    )


def _build_verify_response(verified: bool, decision_id: str = "", receipt_digest: str = "", error: str = "", as_protobuf: bool = False):
    if as_protobuf and _STUBS_AVAILABLE:
        return system1_pb2.VerifyReceiptResponse(
            verified=bool(verified),
            decision_id=decision_id or "",
            receipt_digest=receipt_digest or "",
            error=error or "",
        )
    return _SimpleNamespace(
        verified=verified,
        decision_id=decision_id,
        receipt_digest=receipt_digest,
        error=error,
    )


def _build_health_response(schemas: List[str], as_protobuf: bool = False):
    try:
        from system1 import __version__ as _ver
    except ImportError:
        _ver = "0.2.2"
    if as_protobuf and _STUBS_AVAILABLE:
        return system1_pb2.HealthCheckResponse(
            status=system1_pb2.HealthCheckResponse.SERVING,
            loaded_schemas=schemas or [],
            version=_ver,
        )
    return _SimpleNamespace(
        status=1,  # SERVING
        loaded_schemas=schemas,
        version=_ver,
    )



# ---------------------------------------------------------------------------
# Generic gRPC method handler (no compiled stubs needed)
# ---------------------------------------------------------------------------

def _make_generic_handler(servicer):
    """Build a ``grpc.GenericRpcHandler`` that dispatches JSON-over-gRPC."""

    # Fallback: manual handler using JSON serialization
    class _Handler(grpc.GenericRpcHandler):
        def service(self, handler_call_details):
            method = handler_call_details.method
            if method.endswith("/Decide"):
                return grpc.unary_unary_rpc_method_handler(
                    _json_handler(servicer.Decide),
                    request_deserializer=_deserialize_json,
                    response_serializer=_serialize_json,
                )
            elif method.endswith("/Guard"):
                return grpc.unary_unary_rpc_method_handler(
                    _json_handler(servicer.Guard),
                    request_deserializer=_deserialize_json,
                    response_serializer=_serialize_json,
                )
            elif method.endswith("/VerifyReceipt"):
                return grpc.unary_unary_rpc_method_handler(
                    _json_handler(servicer.VerifyReceipt),
                    request_deserializer=_deserialize_json,
                    response_serializer=_serialize_json,
                )
            elif method.endswith("/HealthCheck"):
                return grpc.unary_unary_rpc_method_handler(
                    _json_handler(servicer.HealthCheck),
                    request_deserializer=_deserialize_json,
                    response_serializer=_serialize_json,
                )
            return None

    return _Handler()


def _deserialize_json(data: bytes):
    """Deserialize incoming bytes as JSON dict."""
    if not data:
        return {}
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {}


def _serialize_json(msg) -> bytes:
    """Serialize a response namespace to JSON bytes."""
    if isinstance(msg, dict):
        return json.dumps(msg, default=str).encode("utf-8")

    d = {}
    for k, v in msg.__dict__.items():
        if isinstance(v, bytes):
            import base64
            d[k] = base64.b64encode(v).decode("ascii")
        elif isinstance(v, _ConformalSetMsg):
            d[k] = {"members": v.members}
        elif isinstance(v, dict):
            serialized = {}
            for dk, dv in v.items():
                if isinstance(dv, _ConformalSetMsg):
                    serialized[dk] = {"members": dv.members}
                else:
                    serialized[dk] = dv
            d[k] = serialized
        else:
            d[k] = v
    return json.dumps(d, default=str).encode("utf-8")


def _json_handler(method):
    """Wrap an RPC method to accept dict requests."""
    class _DictRequest:
        def __init__(self, data):
            self._data = data if isinstance(data, dict) else {}

        def __getattr__(self, name):
            if name.startswith("_"):
                return object.__getattribute__(self, name)
            return self._data.get(name, "" if name != "telemetry" else {})

        def get(self, name, default=None):
            return self._data.get(name, default)

    def handler(request, context):
        if isinstance(request, dict):
            request = _DictRequest(request)
        return method(request, context)

    return handler


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def serve(
    host: str = "127.0.0.1",
    port: int = 50051,
    schemas: Optional[Dict[str, Any]] = None,
    max_workers: int = 10,
    block: bool = True,
    signing_key: Optional[Any] = None,
    ledger: Optional[Any] = None,
    policy_engine: Optional[Any] = None,
) -> Any:
    """Start the System 1 gRPC server.

    Parameters
    ----------
    host : str
        Host address to bind to (default '127.0.0.1' for local loopback).
    port : int
        TCP port to listen on (default 50051).
    schemas : dict, optional
        Mapping of schema name → ``DecisionSchema`` instance.
    max_workers : int
        Thread-pool size for the gRPC server.
    block : bool
        If True, block until the server is terminated.
    signing_key : Ed25519PrivateKey, optional
        Cryptographic signing key for receipts.
    ledger : ActionLedger, optional
        Persistent audit ledger.
    policy_engine : PolicyEngine, optional
        Deterministic policy reference monitor.

    Returns
    -------
    grpc.Server
        The running server instance (useful when ``block=False``).
    """
    if not _GRPC_AVAILABLE:
        raise ImportError(
            "grpcio is required for the gRPC server.  Install with: "
            "pip install 'system1[grpc]'"
        )

    servicer = SystemOneServiceServicer(
        schemas=schemas,
        signing_key=signing_key,
        ledger=ledger,
        policy_engine=policy_engine,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    if _STUBS_AVAILABLE:
        system1_pb2_grpc.add_SystemOneServiceServicer_to_server(servicer, server)
    else:
        handler = _make_generic_handler(servicer)
        server.add_generic_rpc_handlers([handler])

    listen_addr = f"{host}:{port}"
    bound_port = server.add_insecure_port(listen_addr)
    server.port = bound_port
    server.start()

    logger.info("System 1 gRPC server listening on %s", listen_addr)
    loaded = list((schemas or {}).keys()) or ["triage (default)"]
    logger.info("Loaded schemas: %s", ", ".join(loaded))
    print(f"System 1 gRPC server started on {listen_addr}")
    print(f"Loaded schemas: {', '.join(loaded)}")

    if block:
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            print("\nShutting down System 1 gRPC server…")
            server.stop(grace=5)

    return server


__all__ = [
    "SystemOneServiceServicer",
    "grpc_available",
    "serve",
]
