"""Opt-in recovery must preserve strict review and reject unsafe/late answers."""
import asyncio

import pytest

from examples.gaming.skill_playground import teacher
from examples.gaming.skill_playground.review_fallback import StrictCourier, permitted_move
from examples.gaming.skill_playground.world import World


class Network:
    def __init__(self, review=True):
        self.review, self.calls = review, 0

    def decide(self, world):
        self.calls += 1
        return {"action": "east", "review": self.review, "calls": 9}


def setup(**options):
    world = World(98200)
    async def expert(w):
        return teacher.decide(w)["action"]
    config = dict(expert=expert, allowed=lambda w: True,
                  validate=permitted_move, expert_budget=8)
    config.update(options)
    return world, StrictCourier(Network(), **config)


@pytest.mark.asyncio
async def test_strict_default_and_local_acceptance():
    world = World(98200)
    strict = StrictCourier(Network())
    result = await strict.decide(world)
    assert result["action"] is None and result["expert_attempts"] == 0
    accepted = await StrictCourier(Network(False)).decide(world)
    assert accepted["action"] == "east" and not accepted["recovered"]


@pytest.mark.asyncio
async def test_recovery_keeps_statistical_review_separate_and_cannot_repeat():
    world, runner = setup()
    expected = teacher.decide(world)["action"]
    result = await runner.decide(world)
    assert result["action"] == expected and result["review"] and result["recovered"]
    second = await runner.decide(world)
    assert second["action"] is None and runner.attempts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("options,reason", [
    ({"expert": None}, "expert_not_configured"),
    ({"allowed": None}, "expert_not_configured"),
    ({"validate": None}, "expert_not_configured"),
    ({"allowed": lambda w: False}, "recovery_not_allowed"),
    ({"allowed": lambda w: 1}, "recovery_not_allowed"),
    ({"expert_budget": 0}, "budget_exhausted"),
])
async def test_no_call_without_contract_permission_and_budget(options, reason):
    world, runner = setup(**options)
    result = await runner.decide(world)
    assert result["action"] is None and result["recovery_reason"] == reason
    assert runner.attempts == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate", [None, "teleport", 7, True, {"private": "value"}])
async def test_rejected_values_never_become_actions(candidate):
    async def expert(w):
        return candidate
    world, runner = setup(expert=expert)
    result = await runner.decide(world)
    assert result["action"] is None and runner.attempts == 1
    assert "private" not in repr(result)
    assert (await runner.decide(world))["action"] is None
    assert runner.attempts == 1


@pytest.mark.asyncio
async def test_failed_and_synchronous_experts_stop():
    async def failed(w):
        raise RuntimeError("private exception details")
    for expert in (failed, lambda w: "east"):
        world, runner = setup(expert=expert)
        result = await runner.decide(world)
        assert result["action"] is None and result["recovery_reason"] == "expert_failed"
        assert "private" not in repr(result)


@pytest.mark.asyncio
async def test_depleted_budget_and_stale_context():
    world, runner = setup(expert_budget=1)
    first = await runner.decide(world)
    world.step(first["action"])
    result = await runner.decide(world)
    assert result["action"] is None and runner.attempts == 1
    async def stale(w):
        w.step("east")
        return "east"
    world, runner = setup(expert=stale)
    assert (await runner.decide(world))["recovery_reason"] == "stale_context"


@pytest.mark.asyncio
async def test_revocation_and_validator_self_close_prevent_answer():
    permission = [True]
    def revoke(w, action):
        permission[0] = False
        return True
    world, runner = setup(allowed=lambda w: permission[0], validate=revoke)
    assert (await runner.decide(world))["action"] is None
    world, runner = setup()
    def close(w, action):
        runner.close()
        return True
    runner.validate = close
    assert (await runner.decide(world))["action"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["close", "move"])
async def test_last_permission_callback_cannot_change_state_and_expose_answer(interruption):
    world, runner = setup()
    checks = [0]
    def allowed(w):
        checks[0] += 1
        if checks[0] == 3:
            if interruption == "close":
                runner.close()
            else:
                w.step("east")
        return True
    runner.allowed = allowed
    assert (await runner.decide(world))["action"] is None


@pytest.mark.asyncio
async def test_async_validator_rejected_without_unawaited_coroutine():
    async def validate(w, action):
        return True
    world, runner = setup(validate=validate)
    result = await runner.decide(world)
    assert result["action"] is None and result["recovery_reason"] == "callback_or_context_failed"
    assert result["calls"] == 9


@pytest.mark.asyncio
async def test_failed_network_has_unknown_calls_and_never_dispatches():
    world, runner = setup()
    def failed(w):
        raise RuntimeError("private network failure")
    runner.network.decide = failed
    result = await runner.decide(world)
    assert result["action"] is None and result["calls"] is None
    assert "private" not in repr(result)


@pytest.mark.asyncio
async def test_timeout_does_not_expose_a_late_answer():
    gate = asyncio.Event()
    finished = asyncio.Event()
    async def expert(w):
        try:
            await gate.wait()
        except asyncio.CancelledError:
            await gate.wait()
        finished.set()
        return "east"
    world, runner = setup(expert=expert, expert_timeout=.005)
    result = await runner.decide(world)
    assert result["action"] is None and result["recovery_reason"] == "expert_timeout"
    gate.set()
    await asyncio.wait_for(finished.wait(), .2)
    await asyncio.sleep(0)
    assert (await runner.decide(world))["action"] is None and runner.attempts == 1


@pytest.mark.asyncio
async def test_concurrent_decide_and_caller_cancellation_do_not_leak():
    started = asyncio.Event()
    async def expert(w):
        started.set()
        await asyncio.sleep(10)
        return "east"
    world, runner = setup(expert=expert)
    first = asyncio.create_task(runner.decide(world))
    await started.wait()
    second = await runner.decide(world)
    assert second["action"] is None and (await first)["action"] is None
    world, runner = setup(expert=expert)
    started.clear()
    task = asyncio.create_task(runner.decide(world))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert (await runner.decide(world))["action"] is None


def test_known_movement_policy_rejects_danger_wall_and_locked_gate():
    world = World(98200, purple=True, purple_dangerous=True)
    world.pos = (1, 1)
    assert not permitted_move(world, "west")
    world.pos = (4, world.gate[1])
    assert not permitted_move(world, "east")
    world.has_key = True
    assert permitted_move(world, "east")
    for surface, expected in (("lava", False), ("purple", False), ("floor", True)):
        world.pos = (6, world.gate[1])
        world.hazards[(7, world.gate[1])] = surface
        assert permitted_move(world, "east", purple_dangerous=True) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["wall", "lava", "purple"])
async def test_unsafe_expert_proposal_rejected(surface):
    async def expert(w):
        return "east"
    world, runner = setup(expert=expert, validate=lambda w, a: permitted_move(w, a, purple_dangerous=True))
    world.pos = (6, world.gate[1])
    world.hazards[(7, world.gate[1])] = surface
    result = await runner.decide(world)
    assert result["action"] is None and result["recovery_reason"] == "candidate_rejected"


@pytest.mark.parametrize("options", [{"expert_budget": -1}, {"expert_budget": True},
                                     {"expert_timeout": 0}, {"expert_timeout": float("inf")}])
def test_constructor_bounds(options):
    with pytest.raises(ValueError):
        StrictCourier(Network(), **options)


@pytest.mark.asyncio
async def test_finished_world_requires_new_episode():
    world, runner = setup()
    world.paused_for_review = True
    result = await runner.decide(world)
    assert result["action"] is None and runner.attempts == 0
