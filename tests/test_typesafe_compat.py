"""Tests for TypeSafe AI (Jev) drop-in compatibility layer."""

import asyncio
import sys
import pytest
from pathlib import Path

import system1.compat.typesafe as system1_typesafe
from system1.compat.typesafe import (
    AsyncClient,
    AsyncTypeSafeClient,
    Choice,
    ChoiceAnswer,
    Client,
    MultiChoice,
    MultiChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    TypeSafeClient,
    TypeSafeResponse,
    Usage,
    batch_system_one,
    batch_systemone,
    patch_typesafe,
    system_one,
    systemone,
    DotDict,
)


def test_typesafe_module_exports():
    """Verify system1.compat.typesafe exports all expected drop-in classes and functions."""
    expected_exports = {
        "TypeSafeClient",
        "Client",
        "AsyncTypeSafeClient",
        "AsyncClient",
        "Choice",
        "MultiChoice",
        "Noul",
        "Score",
        "TypeSafeResponse",
        "SystemOneResponse",
        "ChoiceAnswer",
        "NoulAnswer",
        "ScoreAnswer",
        "MultiChoiceAnswer",
        "Usage",
        "system_one",
        "systemone",
        "batch_system_one",
        "batch_systemone",
        "evaluate",
        "patch_typesafe",
        "DotDict",
        "call_real_typesafe_api",
        "create_typesafe_baseline_response",
        "compare",
    }
    for item in expected_exports:
        assert hasattr(system1_typesafe, item)
        assert item in system1_typesafe.__all__


def test_question_types_construction():
    """Verify Choice, MultiChoice, Noul, and Score question constructors and serialization."""
    # Choice with mapping
    c1 = Choice("Select routing target", criteria={"fast": "Fast route", "slow": "Slow route"})
    d1 = c1.to_dict()
    assert d1["type"] == "choice"
    assert d1["criteria"] == {"fast": "Fast route", "slow": "Slow route"}

    # Choice with list
    c2 = Choice("Pick color", criteria=["red", "green", "blue"])
    assert c2.options == ("red", "green", "blue")

    # MultiChoice with mapping
    mc1 = MultiChoice(
        "Detect policy violations",
        criteria={
            "DATA_EXFILTRATION": "Outbound data transfer",
            "PII_LEAK": "Accessing customer personal data",
        },
    )
    mcd1 = mc1.to_dict()
    assert mcd1["type"] == "multi_choice"
    assert mcd1["criteria"]["DATA_EXFILTRATION"] == "Outbound data transfer"
    assert "PII_LEAK" in mc1.options

    # MultiChoice with list
    mc2 = MultiChoice("Tags", criteria=["tagA", "tagB"])
    assert mc2.options == ("tagA", "tagB")

    # Noul
    n = Noul("Requires escalated human review?", criteria={"true": "Review needed", "false": "Safe"})
    nd = n.to_dict()
    assert nd["type"] == "noul"
    assert nd["criteria"]["true"] == "Review needed"

    # Score
    s = Score("Rate difficulty", criteria=["Easy", "Medium", "Hard"])
    sd = s.to_dict()
    assert sd["type"] == "score"
    assert sd["criteria"] == ["Easy", "Medium", "Hard"]
    assert "min_value" not in sd  # Official ordinal wire contract


def test_typesafe_client_sync_evaluation():
    """Verify TypeSafeClient executes single-pass evaluation with Jev-compatible response."""
    client = TypeSafeClient()

    questions = {
        "target_tier": Choice("Target model tier", criteria={
            "local_small": "Fast local routine lookup or syntax transform",
            "frontier_reasoning": "Deep math proofs, theorem proving, architecture",
        }),
        "needs_deep_reasoning": Noul("Requires multi-step logic?"),
        "complexity_score": Score("Task complexity", min_value=0.0, max_value=3.0),
    }

    # Fast task
    res_fast = client.systemone("Format json indentation and clean whitespace", questions)
    assert isinstance(res_fast, TypeSafeResponse)
    assert res_fast.local_execution is True
    assert res_fast.usage.total_tokens == 0
    assert res_fast.latency_ms < 50.0

    # Verify answers dict and dot-notation
    assert "target_tier" in res_fast.answers
    assert res_fast.answers.target_tier.choice in ("local_small", "frontier_reasoning")
    assert 0.0 <= res_fast.answers.target_tier.confidence <= 1.0
    assert len(res_fast.answers.target_tier.conformal_set) >= 1

    # Noul answers
    assert "needs_deep_reasoning" in res_fast.answers
    assert isinstance(res_fast.answers.needs_deep_reasoning.value, bool)
    assert 0.0 <= res_fast.answers.needs_deep_reasoning.noul <= 1.0

    # Score answers
    assert "complexity_score" in res_fast.answers
    assert 0.0 <= res_fast.answers.complexity_score.score <= 3.0

    # Hard task
    res_hard = client.systemone(
        "Prove the Riemann Hypothesis and derive non-trivial zero distribution on the critical line",
        questions,
    )
    assert res_hard.answers.target_tier.choice == "frontier_reasoning"
    assert res_hard.answers.needs_deep_reasoning.value is True


def test_typesafe_client_raw_dict_questions():
    """Verify TypeSafeClient works with raw dictionary questions as sent over Jev HTTP API."""
    client = TypeSafeClient()

    raw_questions = {
        "verdict": {
            "type": "choice",
            "instructions": "Security verdict",
            "criteria": {
                "ALLOW": "Safe read-only operation",
                "BLOCK": "Destructive command or credential leak",
            },
        },
        "is_safe": {
            "type": "noul",
            "instructions": "Is command safe?",
            "criteria": {"true": "Safe", "false": "Dangerous"},
        },
        "risk": {
            "type": "score",
            "instructions": "Risk score",
            "min_value": 0.0,
            "max_value": 10.0,
        },
    }

    res = client.systemone("rm -rf / --no-preserve-root", raw_questions)
    assert res.answers.verdict.choice == "BLOCK"
    assert res.answers.is_safe.value is False
    assert res.answers.risk.score > 5.0


def test_typesafe_client_batch_systemone():
    """Verify batched prompt evaluation."""
    client = TypeSafeClient()
    questions = {
        "route": Choice(
            "Route",
            criteria={
                "fs": "local filesystem file read or write",
                "net": "remote network socket or websocket handshake",
            },
        ),
    }
    prompts = [
        "Read configuration file config.yaml",
        "Establish WebSocket handshake to upstream server",
    ]
    results = client.batch_systemone(prompts, questions)
    assert len(results) == 2
    assert results[0].answers.route.choice == "fs"
    assert results[1].answers.route.choice == "net"


@pytest.mark.asyncio
async def test_async_typesafe_client():
    """Verify AsyncTypeSafeClient executes non-blocking evaluation."""
    async_client = AsyncTypeSafeClient()
    questions = {
        "action": Choice("Action", criteria={"open": "open door", "close": "close door"}),
        "urgent": Noul("Is urgent"),
    }

    res = await async_client.systemone("Please unlock and open the front entrance immediately", questions)
    assert res.answers.action.choice == "open"
    assert res.local_execution is True

    batch_res = await async_client.batch_systemone(["open it", "shut it down"], questions)
    assert len(batch_res) == 2


def test_patch_typesafe_injection():
    """Verify patch_typesafe correctly replaces the typesafe namespace."""
    with patch_typesafe() as unpatcher:
        import typesafe
        from typesafe import TypeSafeClient as TSClient, Choice as TSChoice

        assert TSClient is TypeSafeClient
        assert TSChoice is Choice

        cli = typesafe.Client()
        r = cli.systemone("Hello", {"greet": typesafe.Choice("Greeting", criteria=["hi", "bye"])})
        assert r.answers.greet.choice in ("hi", "bye")

    # After exiting context, original state is restored
    assert "typesafe" not in sys.modules or sys.modules["typesafe"] is None or not hasattr(sys.modules["typesafe"], "_is_system1_patched")


def test_patch_typesafe_submodule_and_response_exports():
    """Verify typesafe.compat attribute and TypeSafeResponse are accessible."""
    with patch_typesafe():
        import typesafe
        assert hasattr(typesafe, "compat")
        assert typesafe.compat.TypeSafeClient is TypeSafeClient
        assert typesafe.compat.TypeSafeResponse is TypeSafeResponse
        assert typesafe.TypeSafeResponse is TypeSafeResponse

        from typesafe import TypeSafeResponse as TSR
        assert TSR is TypeSafeResponse


def test_patch_typesafe_decorator_usage():
    """Verify patch_typesafe works cleanly as a function decorator."""
    @patch_typesafe()
    def inner_test():
        import typesafe
        assert hasattr(typesafe, "Client")
        client = typesafe.Client()
        assert isinstance(client, TypeSafeClient)
        return True

    assert inner_test() is True
    # State restored
    assert "typesafe" not in sys.modules or sys.modules["typesafe"] is None or not hasattr(sys.modules["typesafe"], "_is_system1_patched")


def test_typesafe_boolean_and_multichoice_questions():
    """Verify raw dict questions supporting boolean and multi_choice."""
    client = TypeSafeClient()
    questions = {
        "is_admin": {"type": "boolean", "instructions": "Is administrative action?"},
        "flags": {"type": "multi_choice", "criteria": ["debug", "dry_run", "audit"]},
    }
    resp = client.systemone("Run sudo administrative debug trace", questions)
    assert "is_admin" in resp.answers
    assert resp.answers.is_admin.type == "noul"
    assert isinstance(resp.answers.is_admin.value, bool)
    assert "flags" in resp.answers
    assert resp.answers.flags.type == "multi_choice"
    assert isinstance(resp.answers.flags.choices, list)


def test_patch_typesafe_concurrent_safety():
    """Verify patch_typesafe behaves correctly under concurrent multi-threaded execution."""
    import threading

    errors = []

    def worker():
        try:
            for _ in range(20):
                with patch_typesafe():
                    import typesafe
                    assert typesafe.TypeSafeClient is TypeSafeClient
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Threading errors encountered: {errors}"


def test_patch_typesafe_sdk_drop_in_workflow():
    """Verify drop-in compatibility for code using 'import typesafe_sdk' and 'client.system_one'."""
    with patch_typesafe():
        import typesafe_sdk
        from typesafe_sdk import (
            AsyncClient,
            AsyncTypeSafeClient,
            Choice,
            ChoiceAnswer,
            Client,
            Noul,
            NoulAnswer,
            Score,
            ScoreAnswer,
            SystemOneResponse,
            TypeSafeClient,
            TypeSafeResponse,
            Usage,
            system_one,
        )

        assert Client is TypeSafeClient
        assert AsyncClient is AsyncTypeSafeClient
        assert SystemOneResponse is TypeSafeResponse
        assert ChoiceAnswer is DotDict or issubclass(ChoiceAnswer, dict)
        assert NoulAnswer is DotDict or issubclass(NoulAnswer, dict)
        assert ScoreAnswer is DotDict or issubclass(ScoreAnswer, dict)
        assert Usage is DotDict or issubclass(Usage, dict)

        client = TypeSafeClient()
        questions = {
            "tier": Choice("Model tier", criteria={"local_small": "Fast local", "frontier": "Frontier"}),
            "urgent": Noul("Is urgent?"),
            "risk": Score("Risk rating", min_value=0.0, max_value=5.0),
        }

        # Verify client.system_one() method alias
        resp = client.system_one(
            state="Route urgent theorem verification task to high-reasoning engine",
            questions=questions,
        )
        assert isinstance(resp, SystemOneResponse)
        assert isinstance(resp, TypeSafeResponse)
        assert resp.local_execution is True
        assert resp.usage.total_tokens == 0
        assert resp.latency_ms < 50.0

        # Answers verification
        assert "tier" in resp.answers
        assert resp.answers.tier.choice in ("local_small", "frontier")
        assert 0.0 <= resp.answers.tier.confidence <= 1.0
        assert len(resp.answers.tier.conformal_set) >= 1

        assert "urgent" in resp.answers
        assert isinstance(resp.answers.urgent.value, bool)
        assert 0.0 <= resp.answers.urgent.noul <= 1.0

        assert "risk" in resp.answers
        assert 0.0 <= resp.answers.risk.score <= 5.0

        # Verify client.batch_system_one() method alias
        batch_resp = client.batch_system_one(
            ["Prompt one", "Prompt two"],
            questions=questions,
        )
        assert len(batch_resp) == 2
        assert isinstance(batch_resp[0], SystemOneResponse)
        assert isinstance(batch_resp[1], SystemOneResponse)

        # Verify top-level functional system_one() API
        func_resp = system_one(
            state="Direct functional call",
            questions=questions,
        )
        assert isinstance(func_resp, SystemOneResponse)
        assert func_resp.answers.tier.choice in ("local_small", "frontier")

    # Verify clean unpatch restoration
    for mod_name in ("typesafe", "typesafe.compat", "typesafe_sdk", "typesafe_sdk.compat"):
        assert mod_name not in sys.modules or not hasattr(sys.modules[mod_name], "_is_system1_patched")


def test_typesafe_sdk_compat_submodule_and_decorator():
    """Verify typesafe_sdk.compat submodule and decorator usage."""
    @patch_typesafe()
    def run_under_patch():
        import typesafe_sdk
        import typesafe_sdk.compat
        from typesafe_sdk.compat import TypeSafeClient as TSCompatClient, SystemOneResponse as TSCompatResp

        assert TSCompatClient is TypeSafeClient
        assert TSCompatResp is SystemOneResponse
        assert hasattr(typesafe_sdk, "compat")
        assert typesafe_sdk.compat.TypeSafeClient is TypeSafeClient
        return True

    assert run_under_patch() is True
    assert "typesafe_sdk" not in sys.modules or not hasattr(sys.modules["typesafe_sdk"], "_is_system1_patched")


@pytest.mark.asyncio
async def test_async_typesafe_client_system_one_alias():
    """Verify system_one and batch_system_one aliases on AsyncTypeSafeClient."""
    async_client = AsyncTypeSafeClient()
    questions = {
        "verdict": Choice("Action", criteria={"allow": "Allow", "deny": "Deny"}),
        "is_safe": Noul("Is safe?"),
    }

    resp = await async_client.system_one("Read local file safely", questions)
    assert isinstance(resp, SystemOneResponse)
    assert resp.answers.verdict.choice in ("allow", "deny")
    assert isinstance(resp.answers.is_safe.value, bool)

    batch_resp = await async_client.batch_system_one(["PR 1", "PR 2"], questions)
    assert len(batch_resp) == 2
    assert isinstance(batch_resp[0], SystemOneResponse)


def test_patch_typesafe_preserves_preexisting_modules():
    """Verify patch_typesafe restores any module that was already present in sys.modules."""
    import types
    fake_mod = types.ModuleType("typesafe_sdk")
    fake_mod.custom_attr = "original_sentinel"
    sys.modules["typesafe_sdk"] = fake_mod

    try:
        with patch_typesafe():
            import typesafe_sdk
            assert hasattr(typesafe_sdk, "_is_system1_patched")
            assert hasattr(typesafe_sdk, "TypeSafeClient")

        # After unpatch, the pre-existing fake module should be restored
        assert sys.modules.get("typesafe_sdk") is fake_mod
        assert getattr(sys.modules.get("typesafe_sdk"), "custom_attr", None) == "original_sentinel"
    finally:
        sys.modules.pop("typesafe_sdk", None)


def test_typesafe_compat_edge_cases_and_error_handling():
    """Verify error paths, boundary conditions, and response serialization."""
    import json
    client = TypeSafeClient()

    # 1. Non-string state raises TypeError
    with pytest.raises(TypeError, match="State must be text"):
        client.system_one(12345, {"q": Choice("Q", criteria=["a", "b"])})

    # 2. Empty questions dict raises ValueError (schema has no fields)
    with pytest.raises(ValueError, match="At least one question"):
        client.system_one("Prompt", {})

    # 3. Invalid question specification type raises TypeError
    with pytest.raises(TypeError, match="Cannot construct question field"):
        client.system_one("Prompt", {"bad": 42})

    # 4. DotDict attribute behaviors
    dd = DotDict({"a": 1, "nested": {"b": 2}})
    assert dd.a == 1
    assert dd.nested.b == 2
    dd.new_key = 3
    assert dd["new_key"] == 3
    del dd.new_key
    assert "new_key" not in dd
    with pytest.raises(AttributeError):
        _ = dd.non_existent_key
    with pytest.raises(AttributeError):
        del dd.non_existent_key

    # 5. Full JSON serialization of response
    resp = client.system_one("Test prompt", {"flag": Noul("Is true?")})
    d = resp.to_dict()
    assert isinstance(d, dict)
    encoded = json.dumps(d)
    assert "input_tokens" in encoded
    assert "local_execution" in encoded


def test_importlib_reload_patched_modules():
    """Verify importlib.reload works without error on patched modules and submodules."""
    import importlib
    with patch_typesafe():
        import typesafe
        import typesafe.compat
        import typesafe.client
        import typesafe_sdk
        import typesafe_sdk.compat
        import typesafe_sdk.client

        reloaded_sdk = importlib.reload(typesafe_sdk)
        assert reloaded_sdk is typesafe_sdk
        assert reloaded_sdk.TypeSafeClient is TypeSafeClient

        reloaded_compat = importlib.reload(typesafe_sdk.compat)
        assert reloaded_compat is typesafe_sdk.compat
        assert reloaded_compat.TypeSafeClient is TypeSafeClient

        reloaded_client = importlib.reload(typesafe_sdk.client)
        assert reloaded_client is typesafe_sdk.client
        assert reloaded_client.TypeSafeClient is TypeSafeClient

        reloaded_ts = importlib.reload(typesafe)
        assert reloaded_ts is typesafe

        reloaded_ts_compat = importlib.reload(typesafe.compat)
        assert reloaded_ts_compat is typesafe.compat

        reloaded_ts_client = importlib.reload(typesafe.client)
        assert reloaded_ts_client is typesafe.client


def test_patch_typesafe_decorator_multiple_invocations():
    """Verify @patch_typesafe() allows a decorated function to be called repeatedly."""
    call_count = 0

    @patch_typesafe()
    def process_query():
        nonlocal call_count
        call_count += 1
        import typesafe_sdk
        from typesafe_sdk import TypeSafeClient as TSC
        c = TSC()
        res = c.system_one("Format json", {"route": Choice("Route", criteria=["a", "b"])})
        return res.answers.route.choice

    for _ in range(5):
        choice = process_query()
        assert choice in ("a", "b")

    assert call_count == 5
    assert "typesafe_sdk" not in sys.modules or not hasattr(sys.modules["typesafe_sdk"], "_is_system1_patched")


def test_patch_typesafe_decorator_without_parentheses():
    """Verify @patch_typesafe without parentheses works seamlessly."""
    call_count = 0

    @patch_typesafe
    def process_query():
        nonlocal call_count
        call_count += 1
        import typesafe_sdk
        return typesafe_sdk.TypeSafeClient

    for _ in range(3):
        cls = process_query()
        assert cls is TypeSafeClient

    assert call_count == 3
    assert "typesafe_sdk" not in sys.modules or not hasattr(sys.modules["typesafe_sdk"], "_is_system1_patched")


@pytest.mark.asyncio
async def test_patch_typesafe_async_decorator_multiple_invocations():
    """Verify @patch_typesafe() and @patch_typesafe work on coroutines across multiple calls."""
    call_count_parens = 0

    @patch_typesafe()
    async def async_worker():
        nonlocal call_count_parens
        call_count_parens += 1
        import typesafe_sdk
        client = typesafe_sdk.AsyncTypeSafeClient()
        res = await client.system_one("Read file", {"q": Choice("Q", criteria=["x", "y"])})
        return res.answers.q.choice

    for _ in range(3):
        val = await async_worker()
        assert val in ("x", "y")

    assert call_count_parens == 3

    call_count_no_parens = 0

    @patch_typesafe
    async def async_worker_no_parens():
        nonlocal call_count_no_parens
        call_count_no_parens += 1
        import typesafe_sdk
        return typesafe_sdk.AsyncTypeSafeClient

    for _ in range(3):
        cls = await async_worker_no_parens()
        assert cls is AsyncTypeSafeClient

    assert call_count_no_parens == 3
    assert "typesafe_sdk" not in sys.modules or not hasattr(sys.modules["typesafe_sdk"], "_is_system1_patched")


def test_typesafe_client_with_decision_fields():
    """Verify TypeSafeClient accepts DecisionField objects directly with proper answer typing."""
    from system1.schema import ChoiceField, BooleanField, ScoreField, MultiChoiceField

    client = TypeSafeClient()
    questions = {
        "route": ChoiceField(options=["allow", "deny"]),
        "flag": BooleanField(),
        "score": ScoreField(min_value=0.0, max_value=10.0),
        "tags": MultiChoiceField(options=["tag1", "tag2"]),
    }

    resp = client.system_one("Audit request", questions)
    assert isinstance(resp.answers.route, ChoiceAnswer)
    assert resp.answers.route.choice in ("allow", "deny")
    assert resp.answers.route.value in ("allow", "deny")

    assert isinstance(resp.answers.flag, NoulAnswer)
    assert isinstance(resp.answers.flag.value, bool)
    assert 0.0 <= resp.answers.flag.noul <= 1.0

    assert isinstance(resp.answers.score, ScoreAnswer)
    assert 0.0 <= resp.answers.score.score <= 10.0
    assert resp.answers.score.score == resp.answers.score.value

    assert isinstance(resp.answers.tags, MultiChoiceAnswer)
    assert isinstance(resp.answers.tags.choices, list)
    assert resp.answers.tags.choices == resp.answers.tags.value


def test_typesafe_client_case_insensitive_dict_types():
    """Verify raw question dicts with mixed/upper-case types produce correct typed answers."""
    client = TypeSafeClient()
    questions = {
        "route": {"type": "Choice", "criteria": ["alpha", "beta"]},
        "is_ok": {"type": "NOUL", "instructions": "Is OK?"},
        "rating": {"type": "Score", "min_value": 0.0, "max_value": 5.0},
        "flags": {"type": "MULTI_CHOICE", "criteria": ["f1", "f2"]},
    }

    resp = client.system_one("Check status", questions)
    assert isinstance(resp.answers.route, ChoiceAnswer)
    assert resp.answers.route.choice in ("alpha", "beta")
    assert resp.answers.route.value in ("alpha", "beta")

    assert isinstance(resp.answers.is_ok, NoulAnswer)
    assert isinstance(resp.answers.is_ok.value, bool)

    assert isinstance(resp.answers.rating, ScoreAnswer)
    assert 0.0 <= resp.answers.rating.score <= 5.0

    assert isinstance(resp.answers.flags, MultiChoiceAnswer)
    assert isinstance(resp.answers.flags.choices, list)


def test_choice_string_criteria_type_error():
    """Verify passing a raw string for Choice criteria raises TypeError instead of character splitting."""
    with pytest.raises(TypeError, match="Invalid criteria type for Choice: str"):
        Choice("Invalid criteria test", criteria="fast")


def test_batch_system_one_module_apis():
    """Verify top-level functional batch_system_one and batch_systemone APIs."""
    prompts = ["First prompt", "Second prompt"]
    questions = {"action": Choice("Action", criteria=["go", "stop"])}

    r1 = batch_system_one(prompts, questions)
    assert len(r1) == 2
    assert r1[0].answers.action.choice in ("go", "stop")

    r2 = batch_systemone(prompts, questions)
    assert len(r2) == 2
    assert r2[0].answers.action.choice in ("go", "stop")


def test_typesafe_client_submodule_import():
    """Verify legacy projects importing typesafe.client and typesafe_sdk.client succeed."""
    with patch_typesafe():
        import typesafe.client as ts_cli
        import typesafe_sdk.client as sdk_cli
        from typesafe_sdk.client import TypeSafeClient as TSClient, AsyncTypeSafeClient as ATSClient
        from typesafe.client import Client as C, AsyncClient as AC

        assert ts_cli.TypeSafeClient is TypeSafeClient
        assert sdk_cli.TypeSafeClient is TypeSafeClient
        assert TSClient is TypeSafeClient
        assert ATSClient is AsyncTypeSafeClient
        assert C is TypeSafeClient
        assert AC is AsyncTypeSafeClient


def test_create_typesafe_baseline_response():
    """Verify realistic TypeSafe baseline fallback response profile generation."""
    from system1.compat.typesafe import create_typesafe_baseline_response

    questions = {
        "action": Choice("Action", criteria={"allow": "Allow access", "deny": "Deny access"}),
        "is_safe": Noul("Is safe?"),
        "risk": Score("Risk level", min_value=0.0, max_value=3.0),
    }

    baseline = create_typesafe_baseline_response(
        state="Inspect system logs",
        questions=questions,
        measured_latency_ms=215.4,
        egress_bytes=512,
    )

    assert isinstance(baseline, TypeSafeResponse)
    assert baseline.local_execution is False
    assert baseline.baseline_fallback is True
    assert baseline.receipt is None
    assert baseline.latency_ms == 215.4
    assert baseline.usage.total_tokens > 0
    assert baseline.answers.action.choice in ("allow", "deny")
    assert baseline.answers.is_safe.noul > 0.0
    assert 0.0 <= baseline.answers.risk.score <= 3.0
    assert baseline.answers.action.conformal_set is None


def test_call_real_typesafe_api_baseline_fallback():
    """Verify call_real_typesafe_api gracefully falls back to realistic baseline when offline/unauthorized."""
    from system1.compat.typesafe import call_real_typesafe_api

    questions = {
        "tier": Choice("Model tier", criteria=["small", "large"]),
    }

    # Calling with dummy key or no key against real endpoint triggers fallback
    resp, latency_ms, egress_bytes = call_real_typesafe_api(
        state="Route to optimal model",
        questions=questions,
        api_key="test-key-invalid",
        timeout=1.0,
        fallback_baseline=True,
        zero_egress=False,
    )

    assert resp is not None
    assert isinstance(resp, TypeSafeResponse)
    assert latency_ms > 0.0
    assert egress_bytes > 0
    assert resp.answers.tier.choice in ("small", "large")


def test_typesafe_client_compare():
    """Verify TypeSafeClient.compare executes local System 1 alongside cloud API comparison."""
    client = TypeSafeClient(zero_egress=False, fallback_baseline=True)

    questions = {
        "target": Choice("Target tier", criteria={"fast": "Local small", "frontier": "Frontier cloud"}),
        "needs_deep": Noul("Needs deep reasoning?"),
    }

    comparison = client.compare("Fix indentation syntax error", questions)

    assert "local_response" in comparison
    assert "cloud_response" in comparison
    assert comparison.local_egress_bytes == 0
    assert comparison.cloud_egress_bytes > 0
    assert comparison.local_tokens == 0
    assert comparison.cloud_tokens > 0
    assert comparison.local_cost_usd == 0.0
    assert comparison.cloud_cost_usd is None
    assert comparison.speedup_factor is None  # Simulated fallback is not live performance evidence
    assert comparison.local_receipt_verified is False  # No signing key configured
    assert comparison.cloud_receipt_verified is False
    assert comparison.local_response.receipt is not None


@pytest.mark.asyncio
async def test_async_typesafe_client_compare():
    """Verify AsyncTypeSafeClient.compare executes asynchronously."""
    client = AsyncTypeSafeClient(zero_egress=False, fallback_baseline=True)

    questions = {
        "status": Choice("Status", criteria=["ok", "error"]),
    }

    comparison = await client.compare("Check system health", questions)
    assert comparison.speedup_factor is None  # Simulated fallback is not live performance evidence
    assert comparison.local_egress_bytes == 0
    assert comparison.cloud_egress_bytes > 0


def test_top_level_compare_convenience():
    """Verify top-level compare functional API."""
    from system1.compat.typesafe import compare as ts_compare

    questions = {
        "sentiment": Choice("Sentiment", criteria=["positive", "negative"]),
    }

    comp = ts_compare("Great service, thank you!", questions, zero_egress=False, fallback_baseline=True)
    assert comp.speedup_factor is None
    assert comp.local_response.answers.sentiment.choice in ("positive", "negative")


def test_usage_edge_cases_and_token_computation():
    """Verify Usage handles None total_tokens, missing fields, string inputs, and property setter."""
    u1 = Usage({"input_tokens": 12, "output_tokens": 8, "total_tokens": None})
    assert u1.total_tokens == 20
    assert u1["total_tokens"] == 20

    u2 = Usage()
    assert u2.total_tokens == 0
    assert u2["total_tokens"] == 0

    u3 = Usage(input_tokens="15", output_tokens="25")
    assert u3.total_tokens == 40

    u4 = Usage(total_tokens=100)
    assert u4.total_tokens == 100
    u4.total_tokens = 250
    assert u4.total_tokens == 250
    assert u4["total_tokens"] == 250


def test_choice_and_score_criteria_dict_and_list_serialization():
    """Verify Choice serializes criteria as mapping and Score serializes criteria as list."""
    # Choice with list criteria -> dictionary serialization
    c = Choice("Verdict", criteria=["ALLOW", "BLOCK"])
    cd = c.to_dict()
    assert cd["type"] == "choice"
    assert isinstance(cd["criteria"], dict)
    assert cd["criteria"] == {"ALLOW": "ALLOW", "BLOCK": "BLOCK"}

    # Score with min/max only -> list criteria serialization
    s1 = Score("Risk", min_value=0.0, max_value=3.0)
    s1d = s1.to_dict()
    assert s1d["type"] == "score"
    assert isinstance(s1d["criteria"], list)
    assert s1d["criteria"] == ["0", "1", "2", "3"]

    # Score with mapping criteria -> sorted list serialization
    s2 = Score("Urgency", criteria={"0": "Low", "1": "High"})
    s2d = s2.to_dict()
    assert s2d["criteria"] == ["Low", "High"]


def test_typesafe_response_enrichment_for_live_cloud_format():
    """Verify TypeSafeResponse adds value, confidence, and conformal_set to live cloud payloads."""
    raw_cloud_json = {
        "model": "jev-1.13.0",
        "answers": {
            "tier": {
                "type": "choice",
                "choice": "standard",
                "confidence": 0.92,
            },
            "is_bug": {
                "type": "noul",
                "noul": 0.85,
            },
            "severity": {
                "type": "score",
                "score": 2.5,
                "confidence": 0.80,
            },
            "tags": {
                "type": "multi_choice",
                "choices": ["tagA", "tagB"],
            },
        },
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
        },
    }

    resp = TypeSafeResponse(raw_cloud_json)

    # Check Choice
    assert resp.answers.tier.value == "standard"
    assert resp.answers.tier.choice == "standard"
    assert resp.answers.tier.conformal_set is None

    # Check Noul
    assert resp.answers.is_bug.value is True
    assert resp.answers.is_bug.confidence == 0.85
    assert resp.answers.is_bug.conformal_set is None

    # Check Score
    assert resp.answers.severity.value == 2.5
    assert resp.answers.severity.score == 2.5
    assert resp.answers.severity.conformal_set is None

    # Check MultiChoice
    assert resp.answers.tags.value == ["tagA", "tagB"]
    assert resp.answers.tags.choices == ["tagA", "tagB"]
    assert resp.answers.tags.conformal_set is None

    # Check Usage
    assert resp.usage.total_tokens == 120
    assert resp.receipt is None
    assert resp.baseline_fallback is False


def test_compare_flags_and_fallback_baseline_toggle():
    """Verify compare returns is_live and baseline_fallback flags and respects fallback_baseline."""
    client = TypeSafeClient(api_key="", zero_egress=False)
    questions = {"q": Choice("Q", criteria=["a", "b"])}

    # Without real API key, fallback_baseline=True produces baseline_fallback=True, is_live=False
    comp = client.compare("Test state", questions, fallback_baseline=True)
    assert comp.is_live is False
    assert comp.baseline_fallback is True

    # Without an explicit simulation, failed HTTP stays a failure.
    import urllib.error
    with pytest.raises(urllib.error.URLError):
        client.compare("Test state", questions, fallback_baseline=False)
def test_paperclips_dropin_compatibility():
    """Verify that the Universal Paperclips Jev prompt and dynamic choices work seamlessly with drop-in patch."""
    patch_typesafe()
    import typesafe

    client = typesafe.Client()
    prompt = (
        "Play Universal Paperclips. Choose the next action to make progress toward completing the game. "
        "Wait when an ongoing process is likely to improve the state."
    )
    choices = [
        "Wait 1 second",
        "Wire",
        "Lower Price",
        "AutoClippers",
        "Make paperclips",
        "AutoClippers x 10",
        "Raise Price",
        "Processors",
        "Memory",
    ]
    q = typesafe.Choice(instructions=prompt, criteria=choices)
    state = '{"paperclips": 7323, "funds": 181.08, "wire": 1676, "operations": 1000}'

    # Test both client.system_one and client.evaluate
    resp1 = client.system_one(state=state, questions={"next_action": q})
    assert resp1.answers.next_action.choice in choices
    assert resp1.local_execution is True

    resp2 = client.evaluate(state=state, questions={"next_action": q})
    assert resp2.answers.next_action.choice in choices
    assert resp2.local_execution is True


def test_paperclips_compare_and_cutover_compatibility():
    """Verify that Universal Paperclips works with client.compare() and auto-cutover modes."""
    patch_typesafe()
    import typesafe
    from system1.ledger import ActionLedger
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    prompt = (
        "Play Universal Paperclips. Choose the next action to make progress toward completing the game. "
        "Wait when an ongoing process is likely to improve the state."
    )
    choices = [
        "Wait 1 second",
        "Wire",
        "Lower Price",
        "AutoClippers",
        "Make paperclips",
        "AutoClippers x 10",
        "Raise Price",
    ]
    q = typesafe.Choice(instructions=prompt, criteria=choices)
    state = '{"paperclips": 7323, "funds": 181.08, "wire": 1676, "operations": 1000}'

    # 1. Compare mode
    client_comp = typesafe.Client(zero_egress=False, fallback_baseline=True)
    comp = client_comp.compare(
        state=state,
        questions={"next_action": q},
        fallback_baseline=True,
    )
    assert comp.local_response.answers.next_action.choice in choices
    assert comp.cloud_response is not None
    assert comp.cloud_response.answers.next_action.choice in choices
    assert comp.speedup_factor is None

    # 2. Auto-cutover mode
    ledger = ActionLedger(":memory:")
    signing_key = Ed25519PrivateKey.generate()
    demo_policy = typesafe.PromotionPolicy(
        min_agreement_threshold=0.5,
        false_allow_ceiling=0.0,
        require_statistical_bound=False,
    )
    client_cut = typesafe.Client(
        mode="auto_cutover",
        cutover_threshold=3,
        min_agreement_threshold=0.5,
        promotion_policy=demo_policy,
        augment=True, strict_mode=False,
        ledger=ledger,
        signing_key=signing_key,
        zero_egress=False,
        fallback_baseline=True,
    )
    assert not client_cut.is_cutover

    # Query 1 (apprentice)
    r1 = client_cut.system_one(state=state, questions={"next_action": q})
    assert r1.answers.next_action.choice in choices
    assert not client_cut.is_cutover

    # Query 2 (apprentice)
    r2 = client_cut.system_one(state='{"paperclips": 7400, "funds": 190.00, "wire": 1600, "operations": 1050}', questions={"next_action": q})
    assert r2.answers.next_action.choice in choices
    assert not client_cut.is_cutover

    # Query 3 (triggers cutover: 3 disjoint samples, 1 train, 1 calib, 1 val)
    r3 = client_cut.system_one(state='{"paperclips": 7500, "funds": 200.00, "wire": 1500, "operations": 1100}', questions={"next_action": q})
    assert r3.answers.next_action.choice in choices
    assert client_cut.is_cutover

    # Query 4 (100% local metal)
    r4 = client_cut.system_one(state=state, questions={"next_action": q})
    assert r4.answers.next_action.choice in choices
    assert r4.local_execution is True







