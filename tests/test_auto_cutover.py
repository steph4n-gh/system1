"""Tests for Autonomous Migration & Trojan Horse Cutover Engine (system1.compat.typesafe)."""

import asyncio
from pathlib import Path
import pytest

from system1.compat.typesafe import AsyncTypeSafeClient, Choice, Noul, PromotionPolicy, TypeSafeClient
from system1.ledger import ActionLedger


def test_typesafe_client_modes_initialization():
    """Verify TypeSafeClient validates execution modes and configuration."""
    client_local = TypeSafeClient(mode="local")
    assert client_local.mode == "local"
    assert not client_local.is_cutover
    assert client_local.call_count == 0

    client_pass = TypeSafeClient(mode="passthrough", zero_egress=False, fallback_baseline=True)
    assert client_pass.mode == "passthrough"

    client_cutover = TypeSafeClient(mode="auto_cutover", cutover_threshold=25, zero_egress=False, fallback_baseline=True)
    assert client_cutover.mode == "auto_cutover"
    assert client_cutover.cutover_threshold == 25

    with pytest.raises(ValueError, match="Invalid mode"):
        TypeSafeClient(mode="quantum_cloud")  # type: ignore


def test_passthrough_mode_execution():
    """Verify mode='passthrough' proxies directly to cloud / baseline response."""
    client = TypeSafeClient(mode="passthrough", zero_egress=False, fallback_baseline=True)
    questions = {
        "tier": Choice("Model tier", criteria={"fast": "Fast model", "smart": "Reasoning model"}),
    }
    resp = client.systemone("Calculate prime factor", questions)
    assert resp.local_execution is False
    assert resp.model == "jev-latest"
    assert "tier" in resp.answers


def test_auto_cutover_engine_trojan_horse_lifecycle():
    """Verify Trojan Horse auto-cutover: forwards first N calls, fits weights, and flips to 100% local."""
    ledger = ActionLedger(":memory:")
    threshold = 5

    client = TypeSafeClient(
        mode="auto_cutover",
        cutover_threshold=threshold,
        promotion_policy=PromotionPolicy(
            min_agreement_threshold=0.75,
            false_allow_ceiling=0.0,
            require_statistical_bound=False,
        ),
        ledger=ledger,
        zero_egress=False,
        fallback_baseline=True,
        augment=True, strict_mode=False,
    )

    questions = {
        "intent": Choice("Request intent", criteria={
            "refund": "Requesting a refund or chargeback",
            "support": "Technical support for application error",
        }),
        "is_urgent": Noul("Is this urgent?", criteria={"true": "Urgent", "false": "Routine"}),
    }

    test_queries = [
        "Please issue a full refund for invoice #100",
        "App crashed with error code 500",
        "I need my money back immediately",
        "Database connection timed out",
    ]

    # Calls 1 to 4: Should be in passthrough/cloud phase
    for idx, prompt in enumerate(test_queries, start=1):
        resp = client.systemone(prompt, questions)
        assert resp.get("auto_cutover_active") is True
        assert resp.get("is_cutover") is False
        assert client.call_count == idx
        assert not client.is_cutover

    # Call 5: Reaches cutover threshold!
    resp5 = client.systemone("Critical server outage on production cluster", questions)
    assert client.call_count == 5
    # Cutover must now be activated!
    assert client.is_cutover is True

    # Audit log verification
    audit_log = client.cutover_audit_log
    assert len(audit_log) >= 1
    event = audit_log[0]
    assert event["event"] == "trojan_horse_cutover"
    assert event["status"] == "active_100_percent_local"
    assert event["call_count"] == 5
    assert event["samples_collected"] >= 5

    # Verify ActionLedger audit log integrity: 5 passthrough queries + 1 cutover event
    assert ledger.audit_head()[0] >= 6
    assert ledger.verify_integrity() is True

    # Calls 6 and beyond: Must execute 100% locally with 0 token usage and sub-5ms latency
    resp6 = client.systemone("Another refund inquiry for transaction #200", questions)
    assert resp6.local_execution is True
    assert resp6.get("auto_cutover_active") is True
    assert resp6.get("is_cutover") is True
    assert resp6.usage.total_tokens == 0
    assert resp6.latency_ms < 15.0
    assert "intent" in resp6.answers
    assert resp6.answers.intent.choice in ("refund", "support")


def test_auto_cutover_respects_min_agreement_threshold():
    """Verify that auto_cutover defers local cutover if local agreement is below min_agreement_threshold."""
    # Set an impossibly high min_agreement_threshold (e.g. 1.01 or 0.999 when agreement < 50.0)
    client = TypeSafeClient(
        mode="auto_cutover",
        cutover_threshold=3,
        min_agreement_threshold=1.01,  # Impossible threshold to force deferral
        zero_egress=False,
        fallback_baseline=True,
        augment=True, strict_mode=False,
    )
    questions = {
        "tier": Choice("Tier", criteria={"fast": "Fast model", "smart": "Smart reasoning model"}),
    }

    for i in range(3):
        client.systemone(f"Prompt test {i}", questions)

    # Cutover threshold reached, but agreement threshold failed!
    assert client.call_count == 3
    assert client.is_cutover is False

    # Check audit log contains deferred event
    audit_log = client.cutover_audit_log
    assert len(audit_log) >= 1
    deferred_event = audit_log[0]
    assert deferred_event["event"] == "trojan_horse_cutover_deferred"
    assert deferred_event["status"] == "deferred_insufficient_agreement"


def test_auto_cutover_high_concurrency_thread_safety():
    """Attack Target 2: Verify 50 parallel threads calling systemone at cutover threshold is thread-safe."""
    import concurrent.futures

    ledger = ActionLedger(":memory:")
    client = TypeSafeClient(
        mode="auto_cutover",
        cutover_threshold=10,
        promotion_policy=PromotionPolicy(
            min_agreement_threshold=0.75,
            false_allow_ceiling=0.0,
            require_statistical_bound=False,
        ),
        ledger=ledger,
        zero_egress=False,
        fallback_baseline=True,
        augment=True, strict_mode=False,
    )
    questions = {
        "action": Choice("Action", criteria={"allow": "Allow", "deny": "Deny"}),
    }

    def worker(idx: int):
        return client.systemone(f"Concurrent action request {idx}", questions)

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(worker, i) for i in range(50)]
        results = [f.result() for f in futures]

    assert len(results) == 50
    assert client.call_count == 50
    assert client.is_cutover is True
    # Audit log should only contain 1 cutover event
    cutover_events = [e for e in client.cutover_audit_log if e["event"] == "trojan_horse_cutover"]
    assert len(cutover_events) == 1
    # ActionLedger integrity must verify cleanly across all 50 concurrent writes
    assert ledger.audit_head()[0] >= 51
    assert ledger.verify_integrity() is True


def test_manual_distill_and_cutover():
    """Verify manual distill_and_cutover flips execution immediately."""
    demo_policy = PromotionPolicy(
        min_agreement_threshold=0.75,
        false_allow_ceiling=0.0,
        require_statistical_bound=False,
    )
    client = TypeSafeClient(
        mode="auto_cutover",
        cutover_threshold=100,
        promotion_policy=demo_policy,
        zero_egress=False,
        fallback_baseline=True,
        augment=True, strict_mode=False,
    )
    questions = {
        "route": Choice("Route", criteria={"sales": "Sales", "support": "Support"}),
    }

    # Execute calls to satisfy 3-way disjoint partition and train both classes
    client.systemone("Pricing question for enterprise", questions)
    client.systemone("Bug with API webhook", questions)
    client.systemone("Enterprise sales quote", questions)
    client.systemone("Technical support ticket", questions)
    client.systemone("Another pricing quote", questions)
    assert not client.is_cutover

    # Manually trigger cutover early
    success = client.distill_and_cutover(questions)
    assert success is True
    assert client.is_cutover is True

    # Next call executes locally
    resp = client.systemone("Follow up on enterprise pricing", questions)
    assert resp.local_execution is True
    assert resp.get("is_cutover") is True


@pytest.mark.asyncio
async def test_async_typesafe_client_auto_cutover():
    """Verify AsyncTypeSafeClient supports auto_cutover seamlessly."""
    demo_policy = PromotionPolicy(
        min_agreement_threshold=0.75,
        false_allow_ceiling=0.0,
        require_statistical_bound=False,
    )
    client = AsyncTypeSafeClient(
        mode="auto_cutover",
        cutover_threshold=3,
        promotion_policy=demo_policy,
        zero_egress=False,
        fallback_baseline=True,
        augment=True, strict_mode=False,
    )
    questions = {
        "verdict": Choice("Verdict", criteria={"allow": "Allow action", "deny": "Deny action"}),
    }

    for i in range(3):
        r = await client.systemone(f"Request {i}", questions)
        assert r is not None

    assert client.is_cutover is True
    assert client.call_count == 3

    # Call 4: local
    r4 = await client.systemone("Next request after cutover", questions)
    assert r4.local_execution is True
    assert r4.get("is_cutover") is True


@pytest.mark.asyncio
async def test_auto_cutover_model_export(tmp_path: Path):
    """Verify distilled model can be accessed and exported via sync and async clients."""
    from system1.compiler import CompiledSystemOneModel

    demo_policy = PromotionPolicy(
        min_agreement_threshold=0.75,
        false_allow_ceiling=0.0,
        require_statistical_bound=False,
    )

    # 1. Sync client export before cutover should return False
    sync_client = TypeSafeClient(
        mode="auto_cutover",
        cutover_threshold=3,
        promotion_policy=demo_policy,
        zero_egress=False,
        fallback_baseline=True,
        augment=True, strict_mode=False,
    )
    assert sync_client.compiled_model is None
    sync_model_path = str(tmp_path / "sync_export.s1m")
    assert sync_client.export_model(sync_model_path) is False

    questions = {
        "status": Choice("Status", criteria={"pass": "Success", "fail": "Failure"}),
    }
    sync_client.systemone("Success test case 1", questions)
    sync_client.systemone("Success test case 2", questions)
    sync_client.systemone("Success test case 3", questions)

    assert sync_client.is_cutover is True
    assert sync_client.compiled_model is not None
    assert sync_client.export_model(sync_model_path) is True
    assert Path(sync_model_path).exists()

    loaded = CompiledSystemOneModel.load(sync_model_path)
    res = loaded.forward_single("Sample check")
    assert "status" in res.fields

    # 2. Async client export
    async_client = AsyncTypeSafeClient(
        mode="auto_cutover",
        cutover_threshold=3,
        promotion_policy=demo_policy,
        zero_egress=False,
        fallback_baseline=True,
        augment=True, strict_mode=False,
    )
    async_model_path = str(tmp_path / "async_export.s1m")
    assert (await async_client.export_model(async_model_path)) is False

    await async_client.systemone("Success async case 1", questions)
    await async_client.systemone("Success async case 2", questions)
    await async_client.systemone("Success async case 3", questions)

    assert async_client.is_cutover is True
    assert async_client.compiled_model is not None
    assert (await async_client.export_model(async_model_path)) is True
    assert Path(async_model_path).exists()
