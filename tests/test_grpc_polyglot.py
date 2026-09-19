"""Tests for Polyglot gRPC / Protobuf Packaging, Wire Serialization, and Twin-Namespace Symmetry.

Verifies:
1. Live gRPC server communication with compiled SystemOneServiceStub over binary wire.
2. All 4 gRPC RPCs: Decide, Guard, VerifyReceipt, HealthCheck.
3. Binary protobuf serialization/deserialization fidelity (SerializeToString / ParseFromString).
4. Twin-namespace packaging symmetry:
   - reflex.integrations.observability and reflex.integrations.otel mirror system1.integrations
   - reflex.proto and system1.proto package assets (.proto files exist and match)
   - reflex.promotion and system1.promotion exports
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Any, Dict
import pytest
try:
    import grpc
    from system1.proto import system1_pb2, system1_pb2_grpc, system1_proto_path as s1_proto_path
    from reflex.proto import system1_pb2 as r_pb2, system1_pb2_grpc as r_pb2_grpc, system1_proto_path as system1_proto_path
    _GRPC_AVAILABLE = True
except ImportError:
    _GRPC_AVAILABLE = False

if not _GRPC_AVAILABLE:
    pytest.skip("grpcio not installed", allow_module_level=True)

from system1.core import BooleanField, ChoiceField, DecisionSchema
from system1.engine import SystemOneEngine
from system1.grpc_server import SystemOneServiceServicer


@pytest.fixture(scope="module")
def simple_engine():
    schema = DecisionSchema(
        fields={
            "action": ChoiceField(["allow", "deny", "escalate"], description="Action"),
            "is_malicious": BooleanField(description="Malicious probe"),
        },
        schema_name="auth_policy",
    )
    return SystemOneEngine(schema, dimension=64)


@pytest.fixture(scope="module")
def live_grpc_server(simple_engine):
    """Spawns an in-process live gRPC server on a dynamic local port."""
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=2))
    from system1.grpc_server import SystemOneServiceServicer
    servicer = SystemOneServiceServicer(schemas={"auth_policy": simple_engine.schema})
    servicer._engines["auth_policy"] = simple_engine
    system1_pb2_grpc.add_SystemOneServiceServicer_to_server(servicer, server)

    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    yield f"127.0.0.1:{port}", servicer
    server.stop(grace=None)


def test_grpc_decide_wire_call(live_grpc_server):
    """Verify Decide RPC receives protobuf request and returns compiled protobuf response."""
    addr, servicer = live_grpc_server
    channel = grpc.insecure_channel(addr)
    stub = system1_pb2_grpc.SystemOneServiceStub(channel)

    req = system1_pb2.DecideRequest(
        prompt="Authorization request from user bob",
        schema_name="auth_policy",
        alpha=0.05,
    )

    resp = stub.Decide(req)
    channel.close()

    assert isinstance(resp, system1_pb2.DecideResponse)
    assert resp.schema_name == "auth_policy"
    assert resp.prompt == "Authorization request from user bob"
    assert "action" in resp.values
    assert resp.values["action"] in ("allow", "deny", "escalate")
    assert "is_malicious" in resp.values
    assert resp.latency_ms > 0.0
    assert len(resp.receipt_json) > 0


def test_grpc_guard_wire_call(live_grpc_server):
    """Verify Guard RPC executes policy evaluation over binary gRPC."""
    addr, servicer = live_grpc_server
    channel = grpc.insecure_channel(addr)
    stub = system1_pb2_grpc.SystemOneServiceStub(channel)

    req = system1_pb2.GuardRequest(
        prompt="Safe health inspection probe",
        schema_name="auth_policy",
        alpha=0.05,
    )

    resp = stub.Guard(req)
    channel.close()

    assert isinstance(resp, system1_pb2.GuardResponse)
    assert resp.outcome in (
        system1_pb2.ALLOW,
        system1_pb2.DENY,
        system1_pb2.REQUIRE_APPROVAL,
    )
    assert "action" in resp.values
    assert len(resp.receipt_json) > 0


def test_grpc_verify_receipt_wire_call(live_grpc_server, simple_engine):
    """Verify VerifyReceipt RPC validates cryptographic receipts over gRPC."""
    addr, servicer = live_grpc_server
    channel = grpc.insecure_channel(addr)
    stub = system1_pb2_grpc.SystemOneServiceStub(channel)

    # Generate a genuine signed receipt from engine
    import json
    res = simple_engine.decide("Check receipt validity", record_receipt=True)
    assert res.receipt is not None
    receipt_bytes = json.dumps(res.receipt.to_dict()).encode("utf-8")

    req = system1_pb2.VerifyReceiptRequest(
        receipt_json=receipt_bytes,
        public_key_hex="",
    )

    resp = stub.VerifyReceipt(req)
    channel.close()

    assert isinstance(resp, system1_pb2.VerifyReceiptResponse)
    assert resp.verified is True
    assert resp.decision_id == res.receipt.decision_id
    assert resp.receipt_digest == res.receipt.digest


def test_grpc_health_check_wire_call(live_grpc_server):
    """Verify HealthCheck RPC returns serving status and loaded schemas."""
    addr, servicer = live_grpc_server
    channel = grpc.insecure_channel(addr)
    stub = system1_pb2_grpc.SystemOneServiceStub(channel)

    req = system1_pb2.HealthCheckRequest()
    resp = stub.HealthCheck(req)
    channel.close()

    assert isinstance(resp, system1_pb2.HealthCheckResponse)
    assert resp.status == system1_pb2.HealthCheckResponse.SERVING
    assert "auth_policy" in resp.loaded_schemas
    assert resp.version in ("0.1.0", "0.1.2", "1.0.0")


def test_protobuf_binary_serialization_round_trip():
    """Verify exact binary serialization and deserialization of protobuf messages."""
    original_req = system1_pb2.DecideRequest(
        prompt="Binary wire test prompt with special characters: ⚡ α = 0.05",
        schema_name="enterprise_policy_v2",
        alpha=0.01,
        telemetry={"user_id": "usr_99182", "ip": "10.0.1.42"},
    )

    # Serialize to standard wire bytes
    wire_bytes = original_req.SerializeToString()
    assert isinstance(wire_bytes, bytes)
    assert len(wire_bytes) > 0

    # Deserialize back
    parsed_req = system1_pb2.DecideRequest()
    parsed_req.ParseFromString(wire_bytes)

    assert parsed_req.prompt == original_req.prompt
    assert parsed_req.schema_name == original_req.schema_name
    assert abs(parsed_req.alpha - original_req.alpha) < 1e-6
    assert dict(parsed_req.telemetry) == dict(original_req.telemetry)


def test_twin_namespace_observability_mirrors():
    """Verify reflex.integrations exports observability and otel without ModuleNotFoundError."""
    import sys
    orig_prom = sys.modules.get("prometheus_client")
    orig_otel = sys.modules.get("opentelemetry")
    orig_otel_trace = sys.modules.get("opentelemetry.trace")

    try:
        try:
            import prometheus_client
        except ImportError:
            import types
            shim = types.ModuleType("prometheus_client")
            class _FakeMetric:
                def __init__(self, *a, **kw):
                    self._labelnames = kw.get("labelnames", [])
                    self._value = 0.0
                    self._children: Dict[tuple, Any] = {}

                def labels(self, **kw):
                    key = tuple(sorted(kw.items()))
                    if key not in self._children:
                        self._children[key] = _FakeMetric()
                    return self._children[key]

                def inc(self, amount=1):
                    self._value += amount

                def observe(self, value):
                    self._value += 1

                def set(self, value):
                    self._value = value
            class _Registry: pass
            shim.Counter = _FakeMetric
            shim.Gauge = _FakeMetric
            shim.Histogram = _FakeMetric
            shim.CollectorRegistry = _Registry
            shim.start_http_server = lambda *a, **kw: None
            sys.modules["prometheus_client"] = shim

        try:
            import opentelemetry
        except ImportError:
            import types
            otel = types.ModuleType("opentelemetry")
            otel_trace = types.ModuleType("opentelemetry.trace")
            class _FakeSpan:
                def set_attribute(self, k, v): pass
                def set_status(self, s): pass
                def __enter__(self): return self
                def __exit__(self, *a): pass
            class _FakeTracer:
                def start_as_current_span(self, n, kind=None): return _FakeSpan()
            class _FakeStatus:
                def __init__(self, c, d=None): pass
            class _FakeStatusCode:
                OK = 0
                ERROR = 1
            class _SpanKind:
                INTERNAL = 0
            otel_trace.get_tracer = lambda n, tracer_provider=None: _FakeTracer()
            otel_trace.Tracer = _FakeTracer
            otel_trace.TracerProvider = type(None)
            otel_trace.SpanKind = _SpanKind
            otel_trace.Status = _FakeStatus
            otel_trace.StatusCode = _FakeStatusCode
            otel.trace = otel_trace
            sys.modules["opentelemetry"] = otel
            sys.modules["opentelemetry.trace"] = otel_trace

        import reflex.integrations.observability as r_obs
        import reflex.integrations.otel as r_otel
        import system1.integrations.observability as s_obs
        import system1.integrations.otel as s_otel

        assert hasattr(r_obs, "SystemOneMetricsExporter")
        assert hasattr(s_obs, "SystemOneMetricsExporter")
        assert r_obs.SystemOneMetricsExporter is s_obs.SystemOneMetricsExporter

        assert hasattr(r_otel, "SystemOneOTelInstrumentor")
        assert hasattr(s_otel, "SystemOneOTelInstrumentor")
        assert r_otel.SystemOneOTelInstrumentor is s_otel.SystemOneOTelInstrumentor
    finally:
        # Restore sys.modules so downstream test files are not contaminated
        if orig_prom is None:
            sys.modules.pop("prometheus_client", None)
            sys.modules.pop("system1.integrations.observability", None)
            sys.modules.pop("reflex.integrations.observability", None)
        else:
            sys.modules["prometheus_client"] = orig_prom
        if orig_otel is None:
            sys.modules.pop("opentelemetry", None)
            sys.modules.pop("system1.integrations.otel", None)
            sys.modules.pop("reflex.integrations.otel", None)
        else:
            sys.modules["opentelemetry"] = orig_otel
        if orig_otel_trace is None:
            sys.modules.pop("opentelemetry.trace", None)
        else:
            sys.modules["opentelemetry.trace"] = orig_otel_trace


def test_twin_namespace_proto_assets():
    """Verify that .proto files exist in both namespaces and are identical."""
    s1_path = Path(s1_proto_path())
    r_path = Path(system1_proto_path())

    assert s1_path.is_file(), f"Proto file missing at {s1_path}"
    assert r_path.is_file(), f"Proto file missing at {r_path}"

    s1_content = s1_path.read_text(encoding="utf-8")
    r_content = r_path.read_text(encoding="utf-8")

    assert s1_content == r_content
    assert 'syntax = "proto3";' in s1_content
    assert "service SystemOneService" in s1_content
    # Confirm deprecated option is removed
    assert "python_generic_services" not in s1_content


def test_twin_namespace_promotion_symmetry():
    """Verify promotion symbols match across reflex and system1 namespaces."""
    import reflex.compat.typesafe as r_compat
    import system1.compat.typesafe as s_compat

    promotion_symbols = [
        "CutoverPartition",
        "DriftDetector",
        "PromotionPolicy",
        "PromotionReport",
        "evaluate_promotion_eligibility",
        "partition_cutover_history",
        "compute_wilson_score_lower",
    ]
    for symbol in promotion_symbols:
        assert hasattr(r_compat, symbol), f"reflex.compat.typesafe missing export: {symbol}"
        assert hasattr(s_compat, symbol), f"system1.compat.typesafe missing export: {symbol}"
        assert getattr(r_compat, symbol) is getattr(s_compat, symbol)
