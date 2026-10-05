"""Compare bounded strict-review policies in the shipped courier workflow.

Saved skills and policies are fixed; fresh map seeds test execution handling,
not new language understanding. The direct rule policy remains a cheap baseline.
"""
import argparse
import asyncio
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from examples.gaming.skill_playground import teacher
from examples.gaming.skill_playground.review_fallback import StrictCourier, observed_state, permitted_move
from examples.gaming.skill_playground.skills import Network
from examples.gaming.skill_playground.world import World


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class CachedObservation:
    """Application answer reuse only while the complete observed state matches."""
    def __init__(self, network):
        self.network, self.state, self.result = network, None, None
        self.evaluations = 0

    def decide(self, world):
        current = observed_state(world)
        if current != self.state:
            self.result = self.network.decide(world)
            self.state = current
            self.evaluations += 1
        return dict(self.result)


class WaitingCourier(StrictCourier):
    """Benchmark-only finite waiting alternatives; no general recovery API."""
    def __init__(self, *args, progress=False, poll_seconds=.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress, self.poll_seconds = progress, poll_seconds
        self.polls, self.wait_ms = 0, 0.0

    async def decide(self, world):
        decision = self.network.decide(world)
        if decision["review"] and self.attempts < self.limit:
            previous = observed_state(world)
            for _ in range(3):
                started = time.perf_counter()
                await asyncio.sleep(self.poll_seconds)
                self.wait_ms += (time.perf_counter() - started) * 1000
                self.polls += 1
                current = observed_state(world)
                if not self.network.decide(world)["review"]:
                    break
                if self.progress and current == previous:
                    break  # Two successive identical reviewed observations.
                previous = current
        return await super().decide(world)


async def episode(network, seed, changed, mode, cap, poll_seconds):
    world = World(seed, purple=changed, purple_dangerous=changed)
    observed = CachedObservation(network)
    native_calls = 0

    async def expert(current):
        nonlocal native_calls
        native_calls += 1
        return teacher.decide(current, purple_dangerous=changed)["action"]

    options = dict(expert=expert if mode != "strict" else None,
                   allowed=lambda _: True,
                   validate=lambda w, a: permitted_move(w, a, purple_dangerous=changed),
                   expert_budget=cap, expert_timeout=1)
    runner = (WaitingCourier(observed, progress=mode == "progress", poll_seconds=poll_seconds, **options)
              if mode in ("bounded_wait", "progress") else StrictCourier(observed, **options))
    initial = world.snapshot()
    initial.pop("seed")
    initial.update(light=world.light, texture=world.texture)
    actions, times, reasons = [], [], []
    unsafe_executed = unsafe_proposed = 0
    started = time.perf_counter()
    try:
        while not world.done:
            turn = time.perf_counter()
            if mode == "full_rules":
                decision = teacher.decide(world, purple_dangerous=changed)
                native_calls += 1
                reason = "known_rules"
            else:
                proposal = observed.decide(world)["action"]
                unsafe_proposed += int(proposal is not None and not permitted_move(world, proposal, purple_dangerous=changed))
                decision = await runner.decide(world)
                reason = decision["recovery_reason"]
            action = decision["action"]
            actions.append(action)
            reasons.append(reason)
            if action is None:
                world.paused_for_review = True
            else:
                unsafe_executed += int(not permitted_move(world, action, purple_dangerous=changed))
                world.step(action)
            times.append((time.perf_counter() - turn) * 1000)
        return {"seed": seed, "changed": changed, "mode": mode, "cap": cap,
                "initial_sha256": digest(initial), "completed": world.success,
                "alive": world.alive, "steps": world.ticks, "review_stop": world.paused_for_review,
                "expert_attempts": runner.attempts, "native_rule_calls": native_calls,
                "model_evaluations": observed.evaluations, "model_calls": observed.evaluations * 9,
                "wait_polls": getattr(runner, "polls", 0), "wait_ms": getattr(runner, "wait_ms", 0.0),
                "unsafe_proposed": unsafe_proposed, "unsafe_executed": unsafe_executed,
                "actions": actions, "actions_sha256": digest(actions), "reasons": reasons,
                "wall_ms": (time.perf_counter() - started) * 1000, "turn_latency_ms": times}
    finally:
        runner.close()


def summarize(rows):
    times = sorted(t for row in rows for t in row["turn_latency_ms"])
    return {"maps": len(rows), "completed": sum(r["completed"] for r in rows),
            "deaths": sum(not r["alive"] for r in rows), "review_stops": sum(r["review_stop"] for r in rows),
            **{key: sum(r[key] for r in rows) for key in
               ("expert_attempts", "native_rule_calls", "model_calls", "wait_polls", "wait_ms",
                "unsafe_proposed", "unsafe_executed")},
            "episode_wall_ms_median": statistics.median(r["wall_ms"] for r in rows),
            "turn_ms_median": statistics.median(times),
            "turn_ms_p95": times[min(len(times) - 1, int(len(times) * .95))]}


async def run(args):
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    models = root / "models"
    models.mkdir()
    archive = ROOT / "examples/gaming/skill_playground/results/quality-evidence.zip"
    if sha256(archive.read_bytes()).hexdigest() != "19d4d89013e4ba5105ff8cb62142480da45467f2623cd5b0cb44d72186e0d6a0":
        raise ValueError("Frozen courier skill archive changed")
    with zipfile.ZipFile(archive) as z:
        for name in ("objective.s1m", "terrain.s1m", "terrain-corrected.s1m", "movement.s1m"):
            (models / name).write_bytes(z.read(name))
    sources = [Path(__file__), ROOT / "tests/test_courier_review_fallback.py"]
    sources += [ROOT / f"examples/gaming/skill_playground/{name}.py"
                for name in ("review_fallback", "world", "skills", "teacher")]
    sources += sorted((ROOT / "src/system1").rglob("*.py"))
    freeze = {"seeds": list(range(args.seed, args.seed + args.maps)), "caps": [0, 1, 8, 256],
              "modes": ["immediate", "bounded_wait", "progress"], "physics_horizon": 180,
              "poll_seconds": args.poll_seconds, "review_limit": 4, "expert_timeout_seconds": 1,
              "python": platform.python_version(), "platform": platform.platform(),
              "archive_sha256": sha256(archive.read_bytes()).hexdigest(),
              "source_sha256": {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in sources},
              "skill_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in models.iterdir()},
              "selection_rule": "Prefer immediate if progress has no paired completion, safety or expert-budget benefit.",
              "scope": "Shipped structured simulator; fresh map seeds, familiar local inputs, no paid expert or deployed workload."}
    (root / "FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")
    networks = {False: Network(models), True: Network(models, corrected=True)}
    rows, parity_misses = [], []
    for changed in (False, True):
        for offset, seed in enumerate(freeze["seeds"]):
            for mode in ("strict", "full_rules"):
                rows.append(await episode(networks[changed], seed, changed, mode, 0, args.poll_seconds))
            for cap in freeze["caps"]:
                modes = freeze["modes"][offset % 3:] + freeze["modes"][:offset % 3]
                paired = []
                for mode in modes:
                    row = await episode(networks[changed], seed, changed, mode, cap, args.poll_seconds)
                    assert row["native_rule_calls"] == row["expert_attempts"] <= cap
                    rows.append(row)
                    paired.append(row)
                for key in ("completed", "alive", "steps", "review_stop", "expert_attempts",
                            "model_calls", "unsafe_executed", "actions_sha256", "initial_sha256"):
                    if len({r[key] for r in paired}) != 1:
                        parity_misses.append({"seed": seed, "changed": changed, "cap": cap, "key": key})
            print(f"Scored seed {seed} changed={changed}", flush=True)
    groups = {f"{'changed' if changed else 'ordinary'}/{mode}/{cap}": summarize([
                  row for row in rows if (row["changed"], row["mode"], row["cap"]) == (changed, mode, cap)])
              for changed in (False, True) for mode in ("strict", "full_rules", *freeze["modes"])
              for cap in ([0] if mode in ("strict", "full_rules") else freeze["caps"])}
    initial = Counter(row["initial_sha256"] for row in rows if row["mode"] == "strict")
    report = {"freeze": freeze, "groups": groups, "rows": rows, "paired_parity_misses": parity_misses,
              "initial_snapshot_multiplicities": dict(initial),
              "limits": ["Generated courier maps and rule-generated teaching; no production qualification.",
                         "Physics does not advance during review; waiting cannot improve this observation.",
                         "Timing includes explicit 1ms scheduling delays, separately recorded; no paid latency savings.",
                         "Full rules counts cheap native invocations, outside hybrid expert-budget accounting.",
                         "Unsafe metrics include known dangerous, wall and locked-gate dispatch; no world-truth action repair.",
                         "Same per-map caps; this is not a global queue-allocation or asynchronous source-fetch test."]}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "summary.json").write_text(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2) + "\n")
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=99300)
    parser.add_argument("--maps", type=int, default=20)
    parser.add_argument("--poll-seconds", type=float, default=.001)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.maps <= 50 or not 0 <= args.poll_seconds <= .1:
        parser.error("Bounded map count and polling interval required")
    asyncio.run(run(args))
