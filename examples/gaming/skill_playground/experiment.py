"""Reproducible teaching, compositional evaluation and isolated correction."""
from collections import Counter
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import platform
import statistics
import time

from . import teacher
from .world import World, MOVES, facts, flat_prompt, move_prompt
from .skills import Network, Single, teach, ask

LIGHTS = ("day", "dusk", "night", "dawn", "bright", "dim", "overcast", "moonlit")
TEXTURES = ("smooth", "rough", "grainy", "polished", "matte", "ridged", "pitted", "bumpy")


def digest(value):
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def row(prompt, label, split):
    return {"input": prompt, "label": label, "split": split}


def lessons():
    # Never teach the locked + return mission. Different map seeds for each split.
    objective, flat = {}, {}
    for split, seeds in (("teach", range(100)), ("calibration", range(100, 150))):
        for seed in seeds:
            for locked, returning in ((False, True), (True, False)):
                world = World(seed, locked=locked, return_home=returning, purple=True)
                while not world.done:
                    decision = teacher.decide(world)
                    text = facts(world)
                    reserved = int(digest(text)[:8], 16) % 4 == 0
                    if reserved == (split == "calibration"):
                        objective.setdefault(text, row(text, decision["target"], split))
                    text = flat_prompt(world)
                    flat.setdefault(text, row(text, decision["action"], split))
                    if decision["action"] is None:
                        world.paused_for_review = True
                    else:
                        world.step(decision["action"])
    terrain = []
    for surface in ("wall", "floor", "lava", "purple"):
        for i, (light, texture) in enumerate(product(LIGHTS, TEXTURES)):
            split = "calibration" if i % 3 == 0 else "teach"
            text = f"surface_{surface} light_{light} texture_{texture}"
            terrain.append(row(text, "dangerous" if teacher.dangerous(surface) else "safe", split))
    movement = []
    for values in product((-1, 0, 1), (-1, 0, 1), (0, 1), range(4)):
        text = move_prompt(values)
        split = "calibration" if int(digest(text)[:8], 16) % 3 == 0 else "teach"
        movement.append(row(text, teacher.preference(values), split))
    data = {"objective": list(objective.values()), "terrain": terrain, "movement": movement}
    # Same total labeled-example budget; the flat model gets extra unique states,
    # not repetitions to make the count look matched. The annotation types differ.
    budgets = Counter(r["split"] for rows in data.values() for r in rows)
    flat_rows = list(flat.values())
    for split in ("teach", "calibration"):
        if sum(r["split"] == split for r in flat_rows) < budgets[split]:
            raise ValueError("Need more unique flat teaching states for the matched budget")
    # Deterministic hash order avoids choosing only the start of each trajectory.
    flat_rows.sort(key=lambda r: digest(r["input"]))
    data["flat"] = [r for split in ("teach", "calibration")
                    for r in [r for r in flat_rows if r["split"] == split][:budgets[split]]]
    return data


def corrected_rows(rows):
    """Replace obsolete purple labels. Retain every other terrain example."""
    result = [dict(r) for r in rows if "surface_purple " not in r["input"]]
    purple = [r for r in rows if "surface_purple " in r["input"]]
    corrections = [r for r in purple if r["split"] == "teach"][:8]
    validation = [r for r in purple if r["split"] == "calibration"]
    result += [{**r, "label": "dangerous"} for r in corrections + validation]
    return result


def run_episode(controller, seed, *, locked=True, returning=True, purple=False, changed=False, trace=False):
    world = World(seed, locked=locked, return_home=returning, purple=purple, purple_dangerous=changed)
    initial = world.snapshot()
    events, times, reviews, calls = [], [], 0, 0
    actions, flags = [], []
    while not world.done:
        start = time.perf_counter()
        decision = controller(world)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        actions.append(decision["action"])
        flags.append(bool(decision["review"]))
        calls += decision["calls"]
        reviews += int(decision["review"])
        # Simulation explicitly executes the prediction even when flagged.
        # The learned terrain gate can stop execution; no world-truth shield.
        if decision["action"] is None:
            world.paused_for_review = True
        else:
            world.step(decision["action"])
        if trace:
            events.append({**decision, "latency_ms": elapsed, "state": world.snapshot()})
    return {"seed": seed, "success": world.success, "alive": world.alive,
            "steps": world.ticks, "paused_for_review": world.paused_for_review, "review_steps": reviews, "model_calls": calls,
            "latencies_ms": times, "actions": actions, "review_flags": flags,
            "initial": initial if trace else None,
            "events": events if trace else None, "final": world.snapshot() if trace else None}


def summarize(rows):
    times = sorted(t for r in rows for t in r["latencies_ms"])
    return {"episodes": len(rows), "successes": sum(r["success"] for r in rows),
            "successes_without_review_flags": sum(r["success"] and not r["review_steps"] for r in rows),
            "deaths": sum(not r["alive"] for r in rows),
            "timeouts": sum(r["alive"] and not r["success"] and not r["paused_for_review"] for r in rows),
            "review_stops": sum(r["paused_for_review"] for r in rows),
            "steps": sum(r["steps"] for r in rows),
            "review_steps": sum(r["review_steps"] for r in rows),
            "model_calls": sum(r["model_calls"] for r in rows),
            "teacher_calls": 0, "median_step_ms": statistics.median(times),
            "p95_step_ms": times[min(len(times)-1, int(len(times)*.95))]}


def experiment(directory, count=40, start_seed=4000):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = lessons()
    lesson_generation_ms = (time.perf_counter()-started)*1000
    (directory / "lessons.json").write_text(json.dumps(data, indent=2) + "\n")
    training = {}
    for name, rows in data.items():
        started = time.perf_counter()
        engine = teach(name, rows, directory / f"{name}.s1m")
        held_out = [r for r in rows if r["split"] == "calibration"]
        # Calibration diagnostics are not independent test accuracy.
        training[name] = {"fit": sum(r["split"] == "teach" for r in rows),
                          "calibration": len(held_out), "bytes": (directory / f"{name}.s1m").stat().st_size,
                          "teaching_ms": (time.perf_counter()-started)*1000,
                          "lessons_sha256": digest(rows)}
        print(f"Taught {name}: {training[name]}", flush=True)
    revised = corrected_rows(data["terrain"])
    (directory / "terrain-corrections.json").write_text(json.dumps(revised, indent=2) + "\n")
    started = time.perf_counter()
    teach("terrain", revised, directory / "terrain-corrected.s1m")
    correction_ms = (time.perf_counter()-started)*1000
    network, corrected, single = Network(directory), Network(directory, corrected=True), Single(directory)
    controllers = {"network": network.decide, "single": single.decide, "rules": teacher.decide,
                   "corrected": corrected.decide}
    suites = {"familiar_unlocked_return": dict(locked=False, returning=True),
              "familiar_locked_collect": dict(locked=True, returning=False),
              "unseen_locked_return": {},
              "changed_purple_rule": dict(purple=True, changed=True)}
    results, details, replays = {}, {}, {}
    for suite, config in suites.items():
        results[suite], details[suite], replays[suite] = {}, {}, {}
        for name, controller in controllers.items():
            rows = [run_episode(controller, seed, **config, trace=seed == start_seed)
                    for seed in range(start_seed, start_seed+count)]
            results[suite][name] = summarize(rows)
            replays[suite][name] = rows[0]
            details[suite][name] = [{k:v for k,v in r.items() if k not in ("initial", "events", "final")} for r in rows]
            print(suite, name, results[suite][name], flush=True)
        if suite == "changed_purple_rule":
            rows = [run_episode(lambda w: teacher.decide(w, True), seed, **config)
                    for seed in range(start_seed, start_seed+count)]
            results[suite]["updated_rules"] = summarize(rows)
            details[suite]["updated_rules"] = [{k:v for k,v in r.items() if k not in ("initial", "events", "final")} for r in rows]
    unchanged = {}
    for name in ("objective", "movement"):
        probes = [r["input"] for r in data[name]]
        unchanged[name] = all(ask(getattr(network, name), p).values == ask(getattr(corrected, name), p).values for p in probes)
    report = {"experiment": "System1 compositional courier, v2 learned terrain gate", "python": platform.python_version(),
              "platform": platform.platform(), "seed_start": start_seed, "seed_count": count,
              "lesson_generation_ms": lesson_generation_ms, "training": training, "teaching_family": ["unlocked_return", "locked_collect"],
              "held_out_family": "locked_return", "results": results, "episodes": details,
              "correction": {"new_fit_labels": 8, "new_calibration_labels": sum(r["split"] == "calibration" and "surface_purple " in r["input"] for r in revised),
                             "teaching_ms": correction_ms, "unchanged_skill_predictions": unchanged},
              "caveats": ["Synthetic structured world and rule-generated labels; not language understanding.",
                          "Equal retained label counts, but intermediate labels and representation differ; rule-generated candidate labels are free, not billed teacher calls.",
                          "Local argmax executes on review flags in this simulator; success is not an autonomy guarantee.",
                          "Visit memory, target-coordinate lookup, a learned-terrain eligibility gate and argmax are explicit application wiring.",
                          "The objective interface deliberately omits the mission termination flag; the flat model receives it.",
                          "Review flags and score intervals do not certify an entire mission or the ranking of scores.",
                          "Whole-mission combinations are held out; familiar local facts and skills may recur."]}
    sources = {p.name: p.read_text() for p in Path(__file__).parent.iterdir() if p.suffix in (".py", ".html")}
    report["source_sha256"] = digest(sources)
    report["skill_sha256"] = {p.name: sha256(p.read_bytes()).hexdigest() for p in directory.glob("*.s1m")}
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (directory / "replays.json").write_text(json.dumps(replays) + "\n")
    (directory / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    return report
