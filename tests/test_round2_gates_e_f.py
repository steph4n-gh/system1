"""Comprehensive contract test suite for Gate E (True Zero-Egress) and Gate F (Packaging & gRPC).

Validates:
- Gate E:
  - ZeroEgressViolationError raised before socket creation.
  - Fail-closed configuration validation (passthrough, auto_cutover without handler, allow_cloud_fallback).
  - compare() and AsyncTypeSafeClient.compare() enforcement under zero_egress=True.
  - Elimination of silent synthetic baseline fallback when fallback_baseline=False.
  - Structured egress audit logging with host, port, payload bytes, operation.
  - Zero socket connection attempts verified via socket patching.
- Gate F:
  - Twin namespace parity between system1 and reflex for ZeroEgressViolationError and egress logs.
  - pyproject.toml bounds: protobuf>=5.26.1, grpcio>=1.62.0, grpcio-tools>=1.62.0.
  - grpc_server.serve() default loopback host="127.0.0.1".
  - CLI serve subcommand --host flag support.
  - SystemOneServiceServicer.Guard() proposal evaluation with ActionProposal, PolicyEngine, signing_key, and ActionLedger.
"""

import inspect
import json
from pathlib import Path
import socket
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from system1.cli import build_parser
import system1.compat.typesafe as s1_typesafe
from system1.compat.typesafe import (
    AsyncTypeSafeClient,
    Choice,
    TypeSafeClient,
    ZeroEgressViolationError,
    call_real_typesafe_api,
    clear_egress_audit_log,
    compare,
    get_egress_audit_log,
)
from system1.grpc_server import SystemOneServiceServicer, serve
from system1.guard import DecisionOutcome, PolicyEngine, PolicyRule, RiskLevel
from system1.guard import DefaultGuardDecisionSchema
from system1.ledger import ActionLedger
from system1.receipt import verify_decision_witness_receipt


# ============================================================================
# Gate E: True Zero-Egress Enforcement (Features F19–F21)
# ============================================================================

class TestZeroEgressGatekeeper:
    """Feature F19: Socket-level zero-egress enforcement and fail-closed gatekeeping."""

    def test_zero_egress_violation_error_raised_before_socket_connect(self, monkeypatch):
        """Verify call_real_typesafe_api raises ZeroEgressViolationError with 0 socket calls."""
        socket_connect_calls = []
        original_connect = socket.socket.connect

        def mock_connect(self, *args, **kwargs):
            socket_connect_calls.append(args)
            return original_connect(self, *args, **kwargs)

        monkeypatch.setattr(socket.socket, "connect", mock_connect)

        with pytest.raises(ZeroEgressViolationError, match="zero_egress=True"):
            call_real_typesafe_api(
                "Classify request",
                {"tier": Choice("Tier", {"fast": "Fast", "smart": "Smart"})},
                api_key="test-api-key",
                base_url="https://api.typesafe.ai/v1",
                zero_egress=True,
            )

        assert len(socket_connect_calls) == 0, "Socket connect was called despite zero_egress=True!"

    def test_incompatible_configuration_validation(self):
        """Verify TypeSafeClient fail-closed configuration validation."""
        # 1. passthrough with zero_egress=True
        with pytest.raises(ValueError, match="mode='passthrough' requires network egress"):
            TypeSafeClient(mode="passthrough", zero_egress=True)

        # 2. auto_cutover without baseline_handler with zero_egress=True
        with pytest.raises(ValueError, match="mode='auto_cutover' with zero_egress=True requires a local baseline_handler"):
            TypeSafeClient(mode="auto_cutover", cutover_threshold=5, zero_egress=True)

        # 3. allow_cloud_fallback with zero_egress=True
        with pytest.raises(ValueError, match="allow_cloud_fallback=True cannot be combined with zero_egress=True"):
            TypeSafeClient(mode="local", zero_egress=True, allow_cloud_fallback=True)

    def test_sync_and_async_compare_raises_zero_egress_violation(self):
        """Feature F20: compare() raises ZeroEgressViolationError under zero_egress=True."""
        client = TypeSafeClient(mode="local", zero_egress=True)
        questions = {"tier": Choice("Tier", {"fast": "Fast", "smart": "Smart"})}

        # Sync client.compare
        with pytest.raises(ZeroEgressViolationError, match="zero_egress=True"):
            client.compare("Evaluate performance difference", questions)

        # Top-level compare function
        with pytest.raises(ZeroEgressViolationError, match="zero_egress=True"):
            compare("Evaluate performance difference", questions, zero_egress=True)

    @pytest.mark.asyncio
    async def test_async_compare_raises_zero_egress_violation(self):
        """AsyncTypeSafeClient.compare() raises ZeroEgressViolationError under zero_egress=True."""
        async_client = AsyncTypeSafeClient(mode="local", zero_egress=True)
        questions = {"tier": Choice("Tier", {"fast": "Fast", "smart": "Smart"})}

        with pytest.raises(ZeroEgressViolationError, match="zero_egress=True"):
            await async_client.compare("Evaluate async performance difference", questions)

    def test_elimination_of_silent_synthetic_teacher_fallback(self):
        """Feature F21: When fallback_baseline=False, transport failure raises explicit error."""
        # When zero_egress=False and fallback_baseline=False, connection to unreachable host raises error
        with pytest.raises(Exception) as exc_info:
            call_real_typesafe_api(
                "Evaluate prompt without synthetic fallback",
                {"tier": Choice("Tier", {"fast": "Fast", "smart": "Smart"})},
                api_key="test-api-key",
                base_url="http://127.0.0.1:59999/v1",
                zero_egress=False,
                fallback_baseline=False,
                timeout=0.2,
            )
        assert not isinstance(exc_info.value, ZeroEgressViolationError)

    def test_egress_audit_logging(self):
        """Verify egress audit logging records host, port, bytes, operation."""
        clear_egress_audit_log()
        assert len(get_egress_audit_log()) == 0

        # Execute call with zero_egress=False and fallback_baseline=True
        call_real_typesafe_api(
            "Egress audit logging test prompt",
            {"status": Choice("Status", {"ok": "OK", "err": "Error"})},
            api_key="test-api-key",
            base_url="https://api.typesafe.ai/v1",
            zero_egress=False,
            fallback_baseline=True,
            timeout=0.5,
        )

        log = get_egress_audit_log()
        assert len(log) >= 1
        event = log[-1]
        assert "timestamp" in event
        assert event["host"] == "api.typesafe.ai"
        assert event["port"] == 443
        assert event["bytes_out"] > 0
        assert event["destination_host"] == "api.typesafe.ai"
        assert event["destination_port"] == 443
        assert event["payload_bytes"] > 0
        assert event["operation_type"] == "typesafe_wan_call"
        assert "/v1" in event["endpoint"]

        clear_egress_audit_log()
        assert len(get_egress_audit_log()) == 0


# ============================================================================
# Gate F: Packaging, Protobuf Alignment & gRPC Truth (Features F22–F25)
# ============================================================================

class TestPackagingAndProtobufAlignment:
    """Features F22–F25: Namespace twin parity, pyproject bounds, loopback server, and Guard evaluation."""

    def test_twin_namespace_parity(self):
        """Feature F22: system1 and reflex namespaces export identical Gate E classes and functions."""
        import reflex
        import reflex.compat.typesafe as system1_typesafe
        import system1

        # ZeroEgressViolationError
        assert system1.ZeroEgressViolationError is s1_typesafe.ZeroEgressViolationError
        assert reflex.ZeroEgressViolationError is system1_typesafe.ZeroEgressViolationError
        assert system1.ZeroEgressViolationError is reflex.ZeroEgressViolationError

        # Egress audit log helpers
        assert s1_typesafe.get_egress_audit_log is system1_typesafe.get_egress_audit_log
        assert s1_typesafe.clear_egress_audit_log is system1_typesafe.clear_egress_audit_log

    def test_pyproject_dependency_bounds(self):
        """Feature F22: pyproject.toml defines required minimum dependency bounds."""
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        assert pyproject_path.exists()
        content = pyproject_path.read_text()

        assert "protobuf>=7.35.1" in content
        assert "grpcio>=1.80.0" in content
        assert "grpcio-tools>=1.80.0" in content

    def test_grpc_server_serve_default_loopback_and_cli_host(self):
        """Feature F23: serve() defaults to 127.0.0.1 and CLI supports --host."""
        sig = inspect.signature(serve)
        assert "host" in sig.parameters
        assert sig.parameters["host"].default == "127.0.0.1"

        # CLI parser inspection
        parser = build_parser()
        args = parser.parse_args(["serve", "--host", "127.0.0.1", "--port", "50051"])
        assert args.host == "127.0.0.1"
        assert args.port == 50051

    def test_grpc_guard_proposal_evaluation_with_policy_and_ledger(self):
        """Feature F23: SystemOneServiceServicer.Guard evaluates ActionProposal through PolicyEngine."""
        signing_key = Ed25519PrivateKey.generate()
        public_key_hex = signing_key.public_key().public_bytes_raw().hex()
        ledger = ActionLedger(":memory:")

        policy_engine = PolicyEngine(rules=[
            PolicyRule(
                rule_id="rule_allow_status",
                tools=["status_probe"],
                outcome=DecisionOutcome.ALLOW,
                effect=DecisionOutcome.ALLOW,
                risk=RiskLevel.READ_ONLY,
            ),
            PolicyRule(
                rule_id="rule_deny_rm",
                tools=["destructive_wipe"],
                outcome=DecisionOutcome.DENY,
                effect=DecisionOutcome.DENY,
                risk=RiskLevel.IRREVERSIBLE,
            ),
        ])

        servicer = SystemOneServiceServicer(
            signing_key=signing_key,
            ledger=ledger,
            policy_engine=policy_engine,
            schemas={tool: DefaultGuardDecisionSchema() for rule in policy_engine.rules for tool in rule.tools},
        )

        class FakeContext:
            def set_code(self, code):
                pass
            def set_details(self, details):
                pass

        # 1. ALLOW outcome via policy
        class AllowReq:
            prompt = "Check system status"
            schema_name = "status_probe"
            telemetry = {}
            alpha = 0.05

        allow_resp = servicer.Guard(AllowReq(), FakeContext())
        assert allow_resp.outcome == 1  # ALLOW
        assert len(allow_resp.receipt_json) > 0
        allow_receipt = json.loads(allow_resp.receipt_json.decode("utf-8"))
        assert verify_decision_witness_receipt(allow_receipt, public_key=public_key_hex) is True

        # 2. DENY outcome via policy
        class DenyReq:
            prompt = "Delete system partition"
            schema_name = "destructive_wipe"
            telemetry = {}
            alpha = 0.05

        deny_resp = servicer.Guard(DenyReq(), FakeContext())
        assert deny_resp.outcome == 2  # DENY
        assert len(deny_resp.receipt_json) > 0
        deny_receipt = json.loads(deny_resp.receipt_json.decode("utf-8"))
        assert verify_decision_witness_receipt(deny_receipt, public_key=public_key_hex) is True

        # 3. Durable recording in ledger
        assert ledger.audit_head()[0] >= 2
        assert ledger.verify_integrity() is True
