"""Optional strict-review courier fallback; the existing playground is unchanged.

An expert proposes a move. The application independently checks current context
and permission before receiving it. This example does not dispatch actions,
change model confidence, learn labels, or provide a general retry controller.
"""
from __future__ import annotations

import argparse
import asyncio
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
import tempfile
import time
import zipfile

from .world import MOVES, World, flat_prompt


def observed_state(world):
    """Hash application-visible state; polling time and rewards are absent."""
    data = (flat_prompt(world), world.pos, world.ticks, sorted(world.visits.items()),
            world.alive, world.done)
    return sha256(json.dumps(data, sort_keys=True).encode()).digest()


def _checked(callback, *args):
    result = callback(*args)
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError("Permission and validation callbacks must be synchronous")
    return result is True


class StrictCourier:
    """One runner per episode/event loop, with at most one pending expert.

    Recovery requires an async read-only expert, explicit permission callback,
    independent synchronous validator and attempted-call budget. Any failure
    stops this runner. Async experts must cooperate with the event loop; a
    timeout discards late results but cannot kill arbitrary callback code.
    """

    def __init__(self, network, *, expert=None, allowed=None, validate=None,
                 expert_budget=0, expert_timeout=1.0):
        if type(expert_budget) is not int or expert_budget < 0:
            raise ValueError("expert_budget must be a nonnegative integer")
        if (isinstance(expert_timeout, bool) or not isinstance(expert_timeout, (int, float))
                or not math.isfinite(expert_timeout) or expert_timeout <= 0):
            raise ValueError("expert_timeout must be finite and positive")
        self.network = network
        self.expert, self.allowed, self.validate = expert, allowed, validate
        self.limit, self.timeout, self.attempts = expert_budget, expert_timeout, 0
        self._busy = self._halted = False
        self._pending = self._reviewed = None

    def close(self):
        self._halted = True
        if self._pending is not None:
            self._pending.cancel()

    def _stop(self, decision, reason):
        self.close()
        return {**decision, "action": None, "recovered": False,
                "recovery_reason": reason, "expert_attempts": self.attempts}

    def _drain(self, task):
        if not task.cancelled():
            task.exception()
        if self._pending is task:
            self._pending = None

    async def decide(self, world):
        if self._halted or self._busy:
            return self._stop({"review": True, "calls": 0}, "runner_stopped")
        self._busy = True
        decision = {"review": True, "calls": None}
        try:
            if world.done:
                return self._stop({"review": True, "calls": 0}, "episode_finished")
            state = observed_state(world)
            decision = dict(self.network.decide(world))
            if self._halted or observed_state(world) != state:
                return self._stop(decision, "stale_context")
            if not decision["review"]:
                return {**decision, "recovered": False, "recovery_reason": "local_accepted",
                        "expert_attempts": self.attempts}
            if state == self._reviewed:
                return self._stop(decision, "review_already_attempted")
            self._reviewed = state
            if not all(callable(cb) for cb in (self.expert, self.allowed, self.validate)):
                return self._stop(decision, "expert_not_configured")
            if not _checked(self.allowed, world):
                return self._stop(decision, "recovery_not_allowed")
            if self._halted or observed_state(world) != state:
                return self._stop(decision, "stale_context")
            if self.attempts >= self.limit:
                return self._stop(decision, "budget_exhausted")
            self.attempts += 1

            async def invoke():
                answer = self.expert(world)
                if not inspect.isawaitable(answer):
                    raise TypeError("Expert must return an awaitable")
                return await answer

            task = self._pending = asyncio.create_task(invoke())
            task.add_done_callback(self._drain)
            deadline = time.monotonic() + self.timeout
            done, _ = await asyncio.wait({task}, timeout=self.timeout)
            if self._halted:
                return self._stop(decision, "runner_stopped")
            if not done or time.monotonic() >= deadline:
                return self._stop(decision, "expert_timeout")
            if task.cancelled():
                return self._stop(decision, "expert_cancelled")
            try:
                answer = task.result()
            except Exception:
                return self._stop(decision, "expert_failed")
            if observed_state(world) != state:
                return self._stop(decision, "stale_context")
            if not _checked(self.allowed, world):
                return self._stop(decision, "recovery_not_allowed")
            if self._halted or observed_state(world) != state:
                return self._stop(decision, "stale_context")
            if type(answer) is not str or answer not in MOVES or not _checked(self.validate, world, answer):
                return self._stop(decision, "candidate_rejected")
            # Validators may mutate context, revoke permission or close this runner.
            if not _checked(self.allowed, world):
                return self._stop(decision, "recovery_not_allowed")
            if self._halted or observed_state(world) != state or time.monotonic() >= deadline:
                return self._stop(decision, "validation_interrupted")
            return {**decision, "action": answer, "recovered": True,
                    "recovery_reason": "validated_expert", "expert_attempts": self.attempts}
        except asyncio.CancelledError:
            self.close()
            raise
        except Exception:
            return self._stop(decision, "callback_or_context_failed")
        finally:
            self._busy = False


def permitted_move(world, action, *, purple_dangerous=False):
    """Explicit known movement rules, not a learned safety guarantee."""
    if type(action) is not str or action not in MOVES:
        return False
    dx, dy = MOVES[action]
    nxt = world.pos[0] + dx, world.pos[1] + dy
    surface = world.terrain(nxt)
    return (surface not in ("wall", "lava") and
            not (surface == "purple" and purple_dangerous) and
            (nxt != world.gate or world.gate_open or world.has_key))


async def run(network, seed, *, changed=False, enable_rule_expert=False, expert_budget=0):
    from . import teacher
    world = World(seed, purple=changed, purple_dangerous=changed)

    async def expert(current):
        return teacher.decide(current, purple_dangerous=changed)["action"]

    runner = StrictCourier(network, expert=expert if enable_rule_expert else None,
                          allowed=lambda _: enable_rule_expert,
                          validate=lambda w, a: permitted_move(w, a, purple_dangerous=changed),
                          expert_budget=expert_budget)
    actions, reasons, calls, unknown_calls = [], [], 0, 0
    started = time.perf_counter()
    try:
        while not world.done:
            decision = await runner.decide(world)
            actions.append(decision["action"])
            reasons.append(decision["recovery_reason"])
            if decision["calls"] is None:
                unknown_calls += 1
            else:
                calls += decision["calls"]
            if decision["action"] is None:
                world.paused_for_review = True
            else:
                world.step(decision["action"])
        return {"seed": seed, "changed": changed, "completed": world.success,
                "alive": world.alive, "steps": world.ticks, "review_stop": world.paused_for_review,
                "expert_attempts": runner.attempts, "model_calls": calls,
                "model_calls_unknown_turns": unknown_calls,
                "actions": actions, "reasons": reasons,
                "wall_ms": (time.perf_counter() - started) * 1000}
    finally:
        runner.close()


def main():
    from .skills import Network
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=98200)
    parser.add_argument("--changed", action="store_true")
    parser.add_argument("--enable-rule-expert", action="store_true")
    parser.add_argument("--expert-budget", type=int, default=0)
    args = parser.parse_args()
    archive = Path(__file__).with_name("results") / "quality-evidence.zip"
    with tempfile.TemporaryDirectory(prefix="system1-courier-review-") as directory:
        root = Path(directory)
        with zipfile.ZipFile(archive) as z:
            for name in ("objective.s1m", "terrain.s1m", "terrain-corrected.s1m", "movement.s1m"):
                (root / name).write_bytes(z.read(name))
        result = asyncio.run(run(Network(root, corrected=args.changed), args.seed,
                                changed=args.changed, enable_rule_expert=args.enable_rule_expert,
                                expert_budget=args.expert_budget))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
