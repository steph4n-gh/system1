"""Verify guard enforcement through LangChain's real callback manager."""

import pytest

pytest.importorskip("langchain_core")
from langchain_core.tools import tool

from system1 import ActionLedger, DecisionOutcome, PolicyEngine, PolicyRule, SystemOneGuard
from system1.integrations import SystemOneGuardBlockedException, SystemOneGuardCallbackHandler


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("outcome", [DecisionOutcome.DENY, DecisionOutcome.REQUIRE_APPROVAL])
async def test_blocked_tool_never_executes(asynchronous, outcome):
    calls = []

    @tool
    def sentinel(value: str) -> str:
        """Record a harmless sentinel call."""
        calls.append(value)
        return value

    guard = SystemOneGuard(policy=PolicyEngine(rules=[
        PolicyRule(rule_id="block_sentinel", tools=["sentinel"], outcome=outcome),
    ]))
    handler = SystemOneGuardCallbackHandler(guard=guard)
    config = {"callbacks": [handler]}
    with pytest.raises(SystemOneGuardBlockedException):
        if asynchronous:
            await sentinel.ainvoke({"value": "blocked"}, config=config)
        else:
            sentinel.invoke({"value": "blocked"}, config=config)

    assert calls == []
    assert handler._active_runs == {}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("fails", [False, True])
async def test_authenticated_principal_and_outcome_survive_run_ids(tmp_path, asynchronous, fails):
    """A random LangChain run ID must not replace the configured policy principal."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    @tool
    def sentinel(value: str) -> str:
        """Return a harmless value or simulate an execution failure."""
        if fails:
            raise RuntimeError("sentinel failure")
        return value

    key = Ed25519PrivateKey.generate()
    with ActionLedger(tmp_path / "audit.sqlite", require_durable=True) as ledger:
        guard = SystemOneGuard(
            policy=PolicyEngine(rules=[PolicyRule(
                rule_id="allow_sentinel", tools=["sentinel"],
                outcome=DecisionOutcome.ALLOW, allowed_principals=["worker"],
            )]),
            ledger=ledger, signing_key=key, enforcement_profile=True,
        )
        handler = SystemOneGuardCallbackHandler(guard=guard, principal_id="worker")

        async def invoke():
            config = {"callbacks": [handler]}
            if asynchronous:
                return await sentinel.ainvoke({"value": "ok"}, config=config)
            return sentinel.invoke({"value": "ok"}, config=config)

        if fails:
            with pytest.raises(RuntimeError, match="^sentinel failure$"):
                await invoke()
        else:
            assert await invoke() == "ok"
        assert handler.interceptions[-1].proposal.principal_id == "worker"
        assert handler._active_runs == {}
        assert ledger.audit_head()[0] == 2
        assert ledger.verify_integrity(trusted_public_key=key.public_key())
