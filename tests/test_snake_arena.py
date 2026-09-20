"""Offline checks of teaching, fair execution and replayable evidence."""
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.gaming.snake_arena.__main__ import Arena, RecordedArena, apply_choice, settings
from examples.gaming.snake_arena.game import DIRECTIONS, SnakeGame, hamiltonian_cycle
from examples.gaming.snake_arena.policy import System1Player, lessons, planner_input


def test_cycle_is_adjacent_closed_and_complete():
    for width, height in ((24, 16), (5, 4), (4, 5)):
        cycle = hamiltonian_cycle(width, height)
        assert len(set(cycle)) == width * height
        assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                   for a, b in zip(cycle, cycle[1:] + cycle[:1]))


def test_taught_skill_generalizes_and_survives_reload(tmp_path):
    groups = {key: {row["group"] for row in rows} for key, rows in lessons().items()}
    assert not groups["teach"] & (groups["calibration"] | groups["evaluate"])
    assert not groups["calibration"] & groups["evaluate"]
    player = System1Player(tmp_path / "snake.s1m")
    report = player.teaching
    assert report["reload_identical"]
    assert report["held_out_correct"] == report["held_out_cases"] == 94
    assert report["skill_bytes"] < 20000
    for seed in (2026, 2027, 2028):
        game = SnakeGame(seed=seed)
        for _ in range(200):
            state, questions, plan = planner_input(game)
            result = player.predict(state, questions)
            proposed, executed = apply_choice(result, plan, False)
            assert proposed == executed == plan["preferred"]
            assert result["teacher_calls"] == 0
            game.step(executed)
            assert game.alive and game.cycle_order_valid()


def test_shield_is_explicit_and_does_not_change_raw_choice():
    result = {"probabilities": dict(zip(DIRECTIONS, (.7, .1, .05, .15)))}
    plan = {"safe": ["DOWN", "RIGHT"]}
    assert apply_choice(result, plan, False) == ("UP", "UP")
    assert apply_choice(result, plan, True) == ("UP", "RIGHT")
    with pytest.raises(ValueError, match="invariant"):
        apply_choice(result, {"safe": []}, True)


@pytest.mark.parametrize("probabilities", [{}, {"UP": 1}, dict.fromkeys(DIRECTIONS, 0),
                                         dict.fromkeys(DIRECTIONS, float("nan")),
                                         dict(zip(DIRECTIONS, (1.1, -.1, 0, 0)))])
def test_invalid_probabilities_cannot_move(probabilities):
    with pytest.raises(ValueError, match="probabilities"):
        apply_choice({"probabilities": probabilities}, {"safe": ["UP"]}, False)


@pytest.mark.parametrize("payload", [{"mode": "unknown"}, {"seed": True}, {"seconds": 0},
                                   {"turns": 999999}, {"shield": "false"}, []])
def test_bounded_settings(payload):
    with pytest.raises(ValueError):
        settings(payload)


class PlannerTestDouble:
    """Only a test double, never used as a substitute for a live provider."""
    metadata = {"name": "test"}

    def __init__(self):
        self.inputs = []

    def predict(self, state, questions):
        self.inputs.append((state, questions))
        preferred = next(d for d, value in questions["move"]["criteria"].items() if "Best" in value)
        return {"probabilities": {d: float(d == preferred) for d in DIRECTIONS},
                "needs_review": False, "input_tokens": 0, "output_tokens": 0, "teacher_calls": 0}


def test_equal_inputs_and_export_reconstruct_every_board(tmp_path):
    players = {name: PlannerTestDouble() for name in ("a", "b", "c")}
    arena = Arena(players, tmp_path)
    arena.config = settings({"mode": "turns", "turns": 20, "shield": False})
    arena.started, arena.run_id = time.perf_counter(), "test"
    for name in players:
        arena._play(name)
    arena._finish([])
    report = json.loads((tmp_path / "latest.json").read_text())
    assert not report["summary"]["running"]
    assert report["summary"]["has_replay"]
    assert players["a"].inputs == players["b"].inputs == players["c"].inputs
    for name in players:
        game = SnakeGame(seed=arena.config["seed"])
        for event in report["lanes"][name]["events"]:
            game.step(event["executed"])
            assert list(game.head) == event["head"]
            assert list(game.food) == event["food"]
        assert game.snapshot() == report["summary"]["lanes"][name]["board"]


def test_provider_failure_is_not_replaced_or_exposed(tmp_path):
    class BrokenPlayer(PlannerTestDouble):
        def predict(self, state, questions):
            raise RuntimeError("secret HTTP body")
    arena = Arena({"failed": BrokenPlayer()}, tmp_path)
    arena.started = time.perf_counter()
    arena._play("failed")
    lane = arena.snapshot()["lanes"]["failed"]
    assert lane["status"] == "error" and lane["board"]["ticks"] == 0
    assert "secret" not in lane["error"]


def test_reply_after_deadline_does_not_buy_an_extra_move(tmp_path, monkeypatch):
    ticks = [100.0]
    class LatePlayer(PlannerTestDouble):
        def predict(self, state, questions):
            result = super().predict(state, questions)
            ticks[0] += 31
            return result
    arena = Arena({"late": LatePlayer()}, tmp_path)
    arena.started = ticks[0]
    monkeypatch.setattr("examples.gaming.snake_arena.__main__.time.perf_counter", lambda: ticks[0])
    arena._play("late")
    lane = arena.snapshot()["lanes"]["late"]
    assert lane["board"]["ticks"] == 0
    assert lane["requests"] == lane["late_responses"] == 1


def test_published_traces_replay_to_the_reported_scores_and_boards():
    root = Path(__file__).resolve().parents[1] / "examples/gaming/snake_arena/results"
    manifest = json.loads((root / "manifest.json").read_text())
    sources = json.loads(gzip.decompress((root / "recorded-source.json.gz").read_bytes()))
    source_hash = sha256("".join(sources[name] for name in sorted(sources)).encode()).hexdigest()
    for item in manifest["reports"]:
        raw = gzip.decompress((root / item["file"]).read_bytes())
        assert sha256(raw).hexdigest() == item["raw_sha256"]
        report = json.loads(raw)
        assert report["source_sha256"] == source_hash
        for name, lane in report["lanes"].items():
            game = SnakeGame(seed=report["summary"]["settings"]["seed"])
            assert game.snapshot() == lane["initial"]
            for event in lane["events"]:
                game.step(event["executed"])
                assert list(game.head) == event["head"]
                assert (list(game.food) if game.food else None) == event["food"]
                assert len(game.body) == event["length"]
                assert game.score == event["score"]
            assert game.snapshot() == report["summary"]["lanes"][name]["board"]


def test_recorded_server_cannot_start_live_calls():
    path = Path(__file__).resolve().parents[1] / "examples/gaming/snake_arena/results/turns-seed2026.json.gz"
    arena = RecordedArena(path)
    assert arena.snapshot()["read_only"] and not arena.snapshot()["running"]
    with pytest.raises(ValueError, match="Replay-only"):
        arena.start({})
