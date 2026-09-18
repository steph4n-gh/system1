"""Integration tests exercising ReflexService gRPC server with an externally generated client.

This test compiles `src/system1/proto/reflex.proto` into python stubs at runtime using
`grpc_tools.protoc`, dynamically loads the stubs, connects to a live in-process gRPC server
on loopback, and validates Decide, Guard, VerifyReceipt, and HealthCheck RPCs with true
cryptographic receipt and signature verifications.
"""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import grpc
import pytest

from system1.grpc_server import serve
from system1.guard import DecisionOutcome, PolicyEngine, PolicyRule, RiskLevel
from system1.ledger import ActionLedger
from system1.receipt import verify_decision_witness_receipt


def test_grpc_external_generated_client_roundtrip():
    """Verify that an externally compiled gRPC client can interact with ReflexService over loopback."""
    from grpc_tools import protoc

    repo_root = Path(__file__).resolve().parent.parent
    proto_dir = repo_root / "src" / "system1" / "proto"
    proto_file = proto_dir / "reflex.proto"

    assert proto_file.exists(), f"Proto file not found at {proto_file}"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 1. Compile reflex.proto into temp_dir
        args = [
            "grpc_tools.protoc",
            f"-I{proto_dir}",
            f"--python_out={temp_dir}",
            f"--grpc_python_out={temp_dir}",
            str(proto_file),
        ]
        exit_code = protoc.main(args)
        assert exit_code == 0, "protoc compilation failed"

        pb2_file = temp_path / "reflex_pb2.py"
        pb2_grpc_file = temp_path / "reflex_pb2_grpc.py"
        assert pb2_file.exists(), "reflex_pb2.py was not generated"
        assert pb2_grpc_file.exists(), "reflex_pb2_grpc.py was not generated"

        # 2. Dynamically import generated modules
        sys.path.insert(0, temp_dir)
        try:
            spec_pb2 = importlib.util.spec_from_file_location("ext_reflex_pb2", pb2_file)
            ext_reflex_pb2 = importlib.util.module_from_spec(spec_pb2)
            sys.modules["ext_reflex_pb2"] = ext_reflex_pb2
            sys.modules["reflex_pb2"] = ext_reflex_pb2
            spec_pb2.loader.exec_module(ext_reflex_pb2)

            spec_grpc = importlib.util.spec_from_file_location("ext_reflex_pb2_grpc", pb2_grpc_file)
            ext_reflex_pb2_grpc = importlib.util.module_from_spec(spec_grpc)
            sys.modules["ext_reflex_pb2_grpc"] = ext_reflex_pb2_grpc
            spec_grpc.loader.exec_module(ext_reflex_pb2_grpc)

            # 3. Configure and spin up live gRPC server on loopback
            signing_key = Ed25519PrivateKey.generate()
            public_key_hex = signing_key.public_key().public_bytes_raw().hex()
            ledger = ActionLedger(":memory:")
            policy_engine = PolicyEngine(rules=[
                PolicyRule(
                    rule_id="rule_allow_safe",
                    tools=["safe_tool"],
                    outcome=DecisionOutcome.ALLOW,
                    effect=DecisionOutcome.ALLOW,
                    risk=RiskLevel.READ_ONLY,
                ),
                PolicyRule(
                    rule_id="rule_deny_rm",
                    tools=["rm_rf"],
                    outcome=DecisionOutcome.DENY,
                    effect=DecisionOutcome.DENY,
                    risk=RiskLevel.IRREVERSIBLE,
                ),
            ])

            server = serve(
                host="127.0.0.1",
                port=0,
                block=False,
                signing_key=signing_key,
                ledger=ledger,
                policy_engine=policy_engine,
            )
            port = server.port

            try:
                # 4. Connect external client via gRPC channel
                with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                    stub = ext_reflex_pb2_grpc.ReflexServiceStub(channel)

                    # RPC 1: HealthCheck
                    health_req = ext_reflex_pb2.HealthCheckRequest()
                    health_resp = stub.HealthCheck(health_req)
                    assert health_resp.status == ext_reflex_pb2.HealthCheckResponse.SERVING
                    assert health_resp.version == "0.1.1"

                    # RPC 2: Decide
                    decide_req = ext_reflex_pb2.DecideRequest(
                        prompt="Prioritize urgent customer inquiry",
                        schema_name="triage",
                    )
                    decide_resp = stub.Decide(decide_req)
                    assert decide_resp.schema_name in ("triage", "DefaultTriageSchema")
                    assert decide_resp.latency_ms >= 0.0
                    assert len(decide_resp.receipt_json) > 0
                    decide_receipt_dict = json.loads(decide_resp.receipt_json.decode("utf-8"))
                    assert verify_decision_witness_receipt(decide_receipt_dict, public_key=public_key_hex) is True

                    # RPC 3: Guard - ALLOWED proposal
                    guard_req_allow = ext_reflex_pb2.GuardRequest(
                        prompt="Execute harmless status check",
                        schema_name="safe_tool",
                    )
                    guard_resp_allow = stub.Guard(guard_req_allow)
                    assert guard_resp_allow.outcome == ext_reflex_pb2.ALLOW
                    assert len(guard_resp_allow.receipt_json) > 0
                    guard_receipt_dict = json.loads(guard_resp_allow.receipt_json.decode("utf-8"))
                    assert verify_decision_witness_receipt(guard_receipt_dict, public_key=public_key_hex) is True

                    # RPC 3b: Guard - DENIED proposal
                    guard_req_deny = ext_reflex_pb2.GuardRequest(
                        prompt="Execute recursive deletion",
                        schema_name="rm_rf",
                    )
                    guard_resp_deny = stub.Guard(guard_req_deny)
                    assert guard_resp_deny.outcome == ext_reflex_pb2.DENY
                    assert len(guard_resp_deny.receipt_json) > 0
                    deny_receipt_dict = json.loads(guard_resp_deny.receipt_json.decode("utf-8"))
                    assert verify_decision_witness_receipt(deny_receipt_dict, public_key=public_key_hex) is True

                    # RPC 4: VerifyReceipt - valid receipt
                    verify_req_valid = ext_reflex_pb2.VerifyReceiptRequest(
                        receipt_json=guard_resp_allow.receipt_json,
                        public_key_hex=public_key_hex,
                    )
                    verify_resp_valid = stub.VerifyReceipt(verify_req_valid)
                    assert verify_resp_valid.verified is True
                    assert verify_resp_valid.decision_id != ""

                    # RPC 4b: VerifyReceipt - tampered receipt fails
                    tampered_receipt = json.loads(guard_resp_allow.receipt_json.decode("utf-8"))
                    tampered_receipt["envelope"]["signature"] = "00" * 64
                    verify_req_invalid = ext_reflex_pb2.VerifyReceiptRequest(
                        receipt_json=json.dumps(tampered_receipt).encode("utf-8"),
                        public_key_hex=public_key_hex,
                    )
                    verify_resp_invalid = stub.VerifyReceipt(verify_req_invalid)
                    assert verify_resp_invalid.verified is False

                    # Verify that the ActionLedger recorded the receipts
                    assert ledger.audit_head()[0] >= 2
                    assert ledger.verify_integrity() is True

            finally:
                server.stop(grace=0)

        finally:
            if temp_dir in sys.path:
                sys.path.remove(temp_dir)
            sys.modules.pop("ext_reflex_pb2", None)
            sys.modules.pop("ext_reflex_pb2_grpc", None)
            sys.modules.pop("reflex_pb2", None)
