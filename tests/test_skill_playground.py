"""Behavioral checks for compositional teaching and a local, repairable runtime."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.gaming.skill_playground import teacher
from examples.gaming.skill_playground.experiment import lessons, corrected_rows, run_episode
from examples.gaming.skill_playground.skills import Network, teach, ask
from examples.gaming.skill_playground.server import Playground
from examples.gaming.skill_playground.world import World


@pytest.fixture(scope="module")
def taught(tmp_path_factory):
    root = tmp_path_factory.mktemp("courier")
    data = lessons()
    for name, rows in data.items():
        teach(name, rows, root / f"{name}.s1m")
    teach("terrain", corrected_rows(data["terrain"]), root / "terrain-corrected.s1m")
    (root / "lessons.json").write_text(json.dumps(data))
    return root, data


def test_independent_calibration_and_equal_label_budget(taught):
    _, data = taught
    for rows in data.values():
        fit = {r["input"] for r in rows if r["split"] == "teach"}
        calibration = {r["input"] for r in rows if r["split"] == "calibration"}
        assert fit and calibration and not fit & calibration
        assert len(rows) == len(fit) + len(calibration)
    network = Counter(r["split"] for name, rows in data.items() if name != "flat" for r in rows)
    assert network == Counter(r["split"] for r in data["flat"])


def test_composed_mission_uses_only_saved_skills_and_no_teacher(taught, tmp_path, monkeypatch):
    root, _ = taught
    for name in ("objective", "terrain", "movement"):
        shutil.copyfile(root / f"{name}.s1m", tmp_path / f"{name}.s1m")
    network = Network(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Local execution called its teacher")
    for name in ("objective", "dangerous", "preference", "decide"):
        monkeypatch.setattr(teacher, name, forbidden)
    episode = run_episode(network.decide, 2000, trace=True)
    assert episode["success"]
    assert episode["model_calls"] == episode["steps"] * 9
    assert {e["target"] for e in episode["events"]} == {"key", "gate", "supply", "home"}
    assert all(not getattr(network, name).use_cache for name in ("objective", "terrain", "movement"))


def test_local_correction_changes_only_terrain_and_preserves_old_missions(taught):
    root, data = taught
    network, fixed = Network(root), Network(root, corrected=True)
    old = run_episode(network.decide, 2000, purple=True, changed=True)
    new = run_episode(fixed.decide, 2000, purple=True, changed=True)
    assert not old["alive"] and new["success"]
    for name in ("objective", "movement"):
        for row in data[name]:
            assert ask(getattr(network, name), row["input"]).values == ask(getattr(fixed, name), row["input"]).values
    for config in ({"locked":False}, {"returning":False}, {}):
        before = run_episode(network.decide, 2001, **config, trace=True)
        after = run_episode(fixed.decide, 2001, **config, trace=True)
        assert before["success"] == after["success"]
        assert [e["state"] for e in before["events"]] == [e["state"] for e in after["events"]]
    retained = [r for r in data["terrain"] if "surface_purple " not in r["input"]]
    updated = corrected_rows(data["terrain"])
    assert all(r in updated for r in retained)
    assert sum(r["split"] == "teach" and "surface_purple " in r["input"] for r in updated) == 8


def test_live_teaching_retains_other_instances_and_can_restore(taught):
    root, _ = taught
    playground = Playground(root)
    objective, movement = playground.network.objective, playground.network.movement
    hashes = {name: sha256((root / f"{name}.s1m").read_bytes()).hexdigest() for name in ("objective", "movement")}
    state = playground.correct(True)
    assert state["corrected"] and state["teaching"]["new_examples"] == 8
    assert playground.network.objective is objective and playground.network.movement is movement
    for name, expected in hashes.items():
        assert sha256((root / f"{name}.s1m").read_bytes()).hexdigest() == expected
    assert not playground.correct(False)["corrected"]
    for invalid in ({"seed": True}, {"seed": -1}, {"locked": "false"}, {"command": "x"}, []):
        with pytest.raises(ValueError):
            playground.reset(invalid)


def test_physics_cannot_walk_through_locked_gate_or_wall():
    world = World(3000)
    world.pos = (4, world.gate[1])
    world.step("east")
    assert world.pos == (4, world.gate[1]) and not world.gate_open
    world.has_key = True
    world.step("east")
    assert world.pos == world.gate and world.gate_open
    world.pos = (1, 1)
    world.step("west")
    assert world.pos == (1, 1)
    with pytest.raises(ValueError):
        world.step("teleport")


def test_taught_terrain_is_a_gate_and_no_eligible_move_stops(taught, monkeypatch):
    from types import SimpleNamespace
    from examples.gaming.skill_playground import skills
    root, _ = taught
    network = Network(root, corrected=True)
    episode = run_episode(network.decide, 3010, purple=True, changed=True, trace=True)
    assert episode["success"]
    for event in episode["events"]:
        assert event["action"] in event["eligible"]
        assert not event["terrain"][event["action"]]["risk"]
    original = skills.ask
    def no_safe_move(engine, prompt):
        if engine is network.terrain:
            return SimpleNamespace(values={"risk": "dangerous"}, is_ambiguous=False)
        return original(engine, prompt)
    monkeypatch.setattr(skills, "ask", no_safe_move)
    stopped = run_episode(network.decide, 2000)
    assert stopped["paused_for_review"] and stopped["steps"] == 0
    assert stopped["review_steps"] == 1 and stopped["model_calls"] == 9


def test_published_evidence_replays_every_episode_and_matches_sources():
    import statistics
    import zipfile
    from examples.gaming.skill_playground.experiment import digest, summarize
    root = Path(__file__).resolve().parents[1] / "examples/gaming/skill_playground"
    with zipfile.ZipFile(root / "results/courier-evidence.zip") as archive:
        report = json.loads(archive.read("report.json"))
        sources = json.loads(archive.read("sources.json"))
        assert digest(sources) == report["source_sha256"]
        for name, content in sources.items():
            assert (root / name).read_text() == content
        for name, expected in report["skill_sha256"].items():
            assert sha256(archive.read(name)).hexdigest() == expected
        lessons = json.loads(archive.read("lessons.json"))
        for name, rows in lessons.items():
            assert digest(rows) == report["training"][name]["lessons_sha256"]
        assert json.loads((root / "results/summary.json").read_text()) == {k:v for k,v in report.items() if k != "episodes"}
        for suite, controllers in report["episodes"].items():
            for controller, rows in controllers.items():
                assert [r["seed"] for r in rows] == list(range(4000, 4040))
                for row in rows:
                    world = World(row["seed"], locked=suite != "familiar_unlocked_return",
                                  return_home=suite != "familiar_locked_collect",
                                  purple=suite == "changed_purple_rule",
                                  purple_dangerous=suite == "changed_purple_rule")
                    for action in row["actions"]:
                        if action is None:
                            assert not world.done
                            world.paused_for_review = True
                        else:
                            world.step(action)
                    assert world.done
                    assert (world.success, world.alive, world.ticks, world.paused_for_review) == (row["success"], row["alive"], row["steps"], row["paused_for_review"])
                    assert len(row["actions"]) == len(row["latencies_ms"]) == len(row["review_flags"])
                    assert sum(row["review_flags"]) == row["review_steps"]
                    expected_calls = 9 if controller in ("network", "corrected") else 1 if controller == "single" else 0
                    assert row["model_calls"] == expected_calls * len(row["actions"])
                assert summarize(rows) == report["results"][suite][controller]
        for suite in ("familiar_unlocked_return", "familiar_locked_collect", "unseen_locked_return"):
            before = report["episodes"][suite]["network"]
            after = report["episodes"][suite]["corrected"]
            assert [r["actions"] for r in before] == [r["actions"] for r in after]
