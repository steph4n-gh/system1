"""Tests for the Reflex gRPC server module.

All tests handle missing ``grpcio`` gracefully: they skip if the dependency
is not installed rather than failing.
"""

from __future__ import annotations

import json
import sys
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Check whether grpcio is available.  Tests that need it will be skipped
# automatically when the import fails.
# ---------------------------------------------------------------------------
try:
    import grpc
    from grpc import StatusCode

    _GRPC_AVAILABLE = True
except ImportError:
    _GRPC_AVAILABLE = False

needs_grpc = pytest.mark.skipif(not _GRPC_AVAILABLE, reason="grpcio not installed")


# ---------------------------------------------------------------------------
# Helper: minimal gRPC-like context stub for direct servicer testing
# ---------------------------------------------------------------------------

class _FakeContext:
    """Stand-in for ``grpc.ServicerContext`` that records status codes."""

    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details

    def abort(self, code, details):
        self.code = code
        self.details = details

    def invocation_metadata(self):
        return []


# ===================================================================
# Test: module can be imported regardless of grpcio
# ===================================================================

class TestModuleImport:
    """Ensure the grpc_server module is importable even without grpcio."""

    def test_import_grpc_server(self):
        """The module itself must always be importable."""
        from system1 import grpc_server

        assert hasattr(grpc_server, "ReflexServiceServicer")
        assert hasattr(grpc_server, "serve")
        assert hasattr(grpc_server, "grpc_available")

    def test_grpc_available_returns_bool(self):
        from system1.grpc_server import grpc_available

        result = grpc_available()
        assert isinstance(result, bool)


# ===================================================================
# Test: servicer instantiation (always works, uses mocked engine)
# ===================================================================

class TestServicerInstantiation:
    """Test that the servicer can be created with and without schemas."""

    def test_default_instantiation(self):
        from system1.grpc_server import ReflexServiceServicer

        servicer = ReflexServiceServicer()
        assert servicer is not None
        assert isinstance(servicer._engines, dict)

    def test_instantiation_with_guard_schema(self):
        from system1.grpc_server import ReflexServiceServicer
        from system1.guard import DefaultGuardDecisionSchema

        schemas = {"guard": DefaultGuardDecisionSchema()}
        servicer = ReflexServiceServicer(schemas=schemas)
        assert "guard" in servicer._engines

    def test_instantiation_with_triage_schema(self):
        from system1.grpc_server import ReflexServiceServicer
        from system1.cli import DefaultTriageSchema

        schemas = {"triage": DefaultTriageSchema()}
        servicer = ReflexServiceServicer(schemas=schemas)
        assert "triage" in servicer._engines


# ===================================================================
# Test: Decide RPC
# ===================================================================

class TestDecideRPC:
    """Test the Decide RPC handler returns valid responses."""

    def test_decide_returns_valid_response(self):
        from system1.grpc_server import ReflexServiceServicer

        servicer = ReflexServiceServicer()
        ctx = _FakeContext()

        class _Req:
            prompt = "Read the README file"
            schema_name = "triage"
            telemetry = {}
            alpha = 0.05

        resp = servicer.Decide(_Req(), ctx)
        assert resp is not None
        assert resp.schema_name != ""
        assert resp.prompt == "Read the README file"
        assert isinstance(resp.values, dict)
        assert isinstance(resp.confidences, dict)
        assert isinstance(resp.latency_ms, float)
        assert ctx.code is None  # no error

    def test_decide_with_guard_schema(self):
        from system1.grpc_server import ReflexServiceServicer

        servicer = ReflexServiceServicer()
        ctx = _FakeContext()

        class _Req:
            prompt = "Delete all production databases"
            schema_name = "guard"
            telemetry = {}
            alpha = 0.05

        resp = servicer.Decide(_Req(), ctx)
        assert resp is not None
        assert resp.schema_name != ""
        assert "is_safe" in resp.values or "risk_category" in resp.values

    def test_decide_empty_prompt_returns_error(self):
        from system1.grpc_server import ReflexServiceServicer

        servicer = ReflexServiceServicer()
        ctx = _FakeContext()

        class _Req:
            prompt = ""
            schema_name = "triage"
            telemetry = {}
            alpha = 0.05

        resp = servicer.Decide(_Req(), ctx)
        assert ctx.code is not None  # should have set an error code

    def test_decide_with_dict_request(self):
        """Test that the servicer handles dict-style requests (JSON handler path)."""
        from system1.grpc_server import ReflexServiceServicer

        servicer = ReflexServiceServicer()
        ctx = _FakeContext()

        req = {
            "prompt": "Check git status",
            "schema_name": "triage",
            "telemetry": {},
            "alpha": 0.05,
        }

        # Simulate what _json_handler does
        class _DictReq:
            def __init__(self, d):
                self._d = d

            def __getattr__(self, name):
                if name.startswith("_"):
                    return object.__getattribute__(self, name)
                return self._d.get(name, "" if name != "telemetry" else {})

        resp = servicer.Decide(_DictReq(req), ctx)
        assert resp is not None
        assert resp.prompt == "Check git status"

    def test_decide_response_has_receipt_json(self):
        from system1.grpc_server import ReflexServiceServicer

        servicer = ReflexServiceServicer()
        ctx = _FakeContext()

        class _Req:
            prompt = "Inspect log files"
            schema_name = "triage"
            telemetry = {}
            alpha = 0.05

        resp = servicer.Decide(_Req(), ctx)
        assert resp.receipt_json is not None
        assert len(resp.receipt_json) > 0
        # Should be valid JSON
        receipt = json.loads(resp.receipt_json)
        assert "decision_id" in receipt


# ===================================================================
# Test: Guard RPC
# ===================================================================

class TestGuardRPC:
    """Test the Guard RPC handler returns correct ALLOW/DENY/REQUIRE_APPROVAL."""

    def test_guard_safe_prompt(self):
        from system1.grpc_server import ReflexServiceServicer

        servicer = ReflexServiceServicer()
        ctx = _FakeContext()

        class _Req:
            prompt = "Read local readme file"
            schema_name = "guard"
            telemetry = {}
            alpha = 0.05

        resp = servicer.Guard(_Req(), ctx)
        assert resp is not None
        # outcome 1=ALLOW, 2=DENY, 3=REQUIRE_APPROVAL
        assert resp.outcome in (1, 2, 3)
        assert resp.reason != ""
        assert isinstance(resp.latency_ms, float)

    def test_guard_dangerous_prompt(self):
        from system1.grpc_server import ReflexServiceServicer

        servicer = ReflexServiceServicer()
        ctx = _FakeContext()

        class _Req:
            prompt = "Execute rm -rf / and wipe the entire filesystem now"
            schema_name = "guard"
            telemetry = {}
            alpha = 0.05

        resp = servicer.Guard(_Req(), ctx)
        assert resp is not None
        # For a clearly dangerous prompt, should be DENY (2) or REQUIRE_APPROVAL (3)
        assert resp.outcome in (2, 3), f"Expected DENY or REQUIRE_APPROVAL for dangerous prompt, got outcome={resp.outcome}"

    def test_guard_empty_prompt_returns_error(self):
        from system1.grpc_server import ReflexServiceServicer

        servicer = ReflexServiceServicer()
        ctx = _FakeContext()

        class _Req:
            prompt = ""
            schema_name = "guard"
            telemetry = {}
            alpha = 0.05

        resp = servicer.Guard(_Req(), ctx)
        assert ctx.code is not None

    def test_guard_response_has_receipt(self):
        from system1.grpc_server import ReflexServiceServicer

        servicer = ReflexServiceServicer()
        ctx = _FakeContext()

        class _Req:
            prompt = "Query database for user record"
            schema_name = "guard"
            telemetry = {}
            alpha = 0.05

        resp = servicer.Guard(_Req(), ctx)
        assert resp.receipt_json is not None
        assert len(resp.receipt_json) > 0


# ===================================================================
# Test: HealthCheck RPC
# ===================================================================

class TestHealthCheckRPC:
    """Test the HealthCheck RPC."""

    def test_health_check_serving(self):
        from system1.grpc_server import ReflexServiceServicer

        servicer = ReflexServiceServicer()
        ctx = _FakeContext()

        class _Req:
            pass

        resp = servicer.HealthCheck(_Req(), ctx)
        assert resp is not None
        assert resp.status == 1  # SERVING
        assert resp.version == "0.1.2"
        assert isinstance(resp.loaded_schemas, list)

    def test_health_check_with_loaded_schemas(self):
        from system1.grpc_server import ReflexServiceServicer
        from system1.guard import DefaultGuardDecisionSchema

        schemas = {"guard": DefaultGuardDecisionSchema()}
        servicer = ReflexServiceServicer(schemas=schemas)
        ctx = _FakeContext()

        class _Req:
            pass

        resp = servicer.HealthCheck(_Req(), ctx)
        assert "guard" in resp.loaded_schemas


# ===================================================================
# Test: VerifyReceipt RPC
# ===================================================================

class TestVerifyReceiptRPC:
    """Test the VerifyReceipt RPC handler."""

    def test_verify_receipt_with_valid_receipt(self):
        """Generate a decision, extract its receipt, and verify it."""
        from system1.grpc_server import ReflexServiceServicer

        servicer = ReflexServiceServicer()
        ctx = _FakeContext()

        # First, generate a decision to get a receipt.
        class _DecReq:
            prompt = "Read local file"
            schema_name = "triage"
            telemetry = {}
            alpha = 0.05

        decide_resp = servicer.Decide(_DecReq(), ctx)
        assert decide_resp.receipt_json

        # Now verify it.
        class _VerReq:
            receipt_json = decide_resp.receipt_json
            public_key_hex = ""

        verify_resp = servicer.VerifyReceipt(_VerReq(), _FakeContext())
        # Unsigned receipts may or may not verify depending on the engine config,
        # but the RPC itself should not crash.
        assert verify_resp is not None
        assert isinstance(verify_resp.verified, bool)

    def test_verify_receipt_empty_receipt(self):
        from system1.grpc_server import ReflexServiceServicer

        servicer = ReflexServiceServicer()
        ctx = _FakeContext()

        class _Req:
            receipt_json = b""
            public_key_hex = ""

        resp = servicer.VerifyReceipt(_Req(), ctx)
        assert resp.verified is False

    def test_verify_receipt_invalid_json(self):
        from system1.grpc_server import ReflexServiceServicer

        servicer = ReflexServiceServicer()
        ctx = _FakeContext()

        class _Req:
            receipt_json = b"not json at all"
            public_key_hex = ""

        resp = servicer.VerifyReceipt(_Req(), ctx)
        assert resp.verified is False
        assert resp.error != ""


# ===================================================================
# Test: serve() function
# ===================================================================

class TestServeFunction:
    """Test the serve() entry point."""

    def test_serve_raises_without_grpcio(self):
        """When grpcio is not available, serve() should raise ImportError."""
        from system1 import grpc_server

        if grpc_server.grpc_available():
            pytest.skip("grpcio is installed, cannot test missing-grpc path")

        with pytest.raises(ImportError, match="grpcio"):
            grpc_server.serve(port=50051, block=False)

    @needs_grpc
    def test_serve_starts_and_stops(self):
        """Start the server in non-blocking mode and stop it immediately."""
        from system1.grpc_server import serve

        server = serve(port=0, block=False)  # port=0 picks a random free port
        assert server is not None
        server.stop(grace=0)


# ===================================================================
# Test: CLI serve subcommand registration
# ===================================================================

class TestCLIServeSubcommand:
    """Test that the CLI correctly registers the serve subcommand."""

    def test_serve_in_parser(self):
        from system1.cli import build_parser

        parser = build_parser()
        # Parse 'serve --port 50051' — should not raise
        args = parser.parse_args(["serve", "--port", "50051"])
        assert args.port == 50051
        assert hasattr(args, "func")

    def test_serve_parser_schema_flag(self):
        from system1.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["serve", "--schema", "guard", "--schema", "triage"])
        assert args.schema == ["guard", "triage"]

    def test_serve_parser_defaults(self):
        from system1.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["serve"])
        assert args.port == 50051
        assert args.grpc is True


# ===================================================================
# Test: Response serialization helpers
# ===================================================================

class TestResponseSerialization:
    """Test the internal response builders."""

    def test_build_decide_response_structure(self):
        from system1.grpc_server import _build_decide_response
        from system1.engine import ReflexEngine
        from system1.cli import DefaultTriageSchema

        engine = ReflexEngine(DefaultTriageSchema())
        result = engine.decide("Read a file", record_receipt=True)

        resp = _build_decide_response(result)
        assert resp.schema_name == result.schema_name
        assert resp.prompt == result.prompt
        assert isinstance(resp.values, dict)
        assert isinstance(resp.confidences, dict)
        assert isinstance(resp.receipt_json, bytes)
        assert len(resp.receipt_json) > 0

    def test_build_health_response(self):
        from system1.grpc_server import _build_health_response

        resp = _build_health_response(["guard", "triage"])
        assert resp.status == 1
        assert resp.loaded_schemas == ["guard", "triage"]
        assert resp.version == "0.1.2"

    def test_build_verify_response(self):
        from system1.grpc_server import _build_verify_response

        resp = _build_verify_response(True, decision_id="dec_123", receipt_digest="abc")
        assert resp.verified is True
        assert resp.decision_id == "dec_123"

        resp2 = _build_verify_response(False, error="bad receipt")
        assert resp2.verified is False
        assert resp2.error == "bad receipt"


# ===================================================================
# Test: Guard logic helper
# ===================================================================

class TestGuardLogic:
    """Test the _apply_guard_logic helper directly."""

    def test_unsafe_decision_returns_deny(self):
        from system1.grpc_server import _apply_guard_logic

        class _MockDecision:
            values = {"is_safe": False, "risk_category": "irreversible_write"}
            confidences = {"is_safe": 0.95, "risk_category": 0.8}
            conformal_sets = {"is_safe": ["False"], "risk_category": ["irreversible_write"]}
            latency_ms = 1.23

        outcome, reason = _apply_guard_logic(_MockDecision(), min_confidence=0.85, alpha=0.05)
        assert outcome == "DENY"
        assert "unsafe" in reason.lower()

    def test_safe_decision_returns_allow(self):
        from system1.grpc_server import _apply_guard_logic

        class _MockDecision:
            values = {"is_safe": True, "risk_category": "read_only"}
            confidences = {"is_safe": 0.95, "risk_category": 0.9}
            conformal_sets = {"is_safe": ["True"], "risk_category": ["read_only"]}
            latency_ms = 0.5

        outcome, reason = _apply_guard_logic(_MockDecision(), min_confidence=0.85, alpha=0.05)
        assert outcome == "ALLOW"

    def test_low_confidence_returns_require_approval(self):
        from system1.grpc_server import _apply_guard_logic

        class _MockDecision:
            values = {"is_safe": True, "risk_category": "read_only"}
            confidences = {"is_safe": 0.5, "risk_category": 0.9}
            conformal_sets = {"is_safe": ["True"], "risk_category": ["read_only"]}
            latency_ms = 0.5

        outcome, reason = _apply_guard_logic(_MockDecision(), min_confidence=0.85, alpha=0.05)
        assert outcome == "REQUIRE_APPROVAL"
        assert "confidence" in reason.lower()

    def test_ambiguous_conformal_returns_require_approval(self):
        from system1.grpc_server import _apply_guard_logic

        class _MockDecision:
            values = {"is_safe": True, "risk_category": "read_only"}
            confidences = {"is_safe": 0.95, "risk_category": 0.9}
            conformal_sets = {"is_safe": ["True", "False"], "risk_category": ["read_only"]}
            latency_ms = 0.5

        outcome, reason = _apply_guard_logic(_MockDecision(), min_confidence=0.85, alpha=0.05)
        assert outcome == "REQUIRE_APPROVAL"
        assert "ambiguity" in reason.lower()

    def test_ood_empty_conformal_returns_require_approval(self):
        from system1.grpc_server import _apply_guard_logic

        class _MockDecision:
            values = {"is_safe": True, "risk_category": "read_only"}
            confidences = {"is_safe": 0.95, "risk_category": 0.9}
            conformal_sets = {"is_safe": [], "risk_category": ["read_only"]}
            latency_ms = 0.5

        outcome, reason = _apply_guard_logic(_MockDecision(), min_confidence=0.85, alpha=0.05)
        assert outcome == "REQUIRE_APPROVAL"
        assert "out-of-distribution" in reason.lower()
