"""One shared planner contract; a small taught System1 skill reads its features.

The compact question wording is adapted from mizchi/laya-mlx, Apache-2.0.
See NOTICE.md. The planner, not any model, computes safe routes.
"""
from hashlib import file_digest, sha256
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
import re
import time

from system1 import (BooleanField, ChoiceField, CompiledSystemOneModel, DecisionSchema,
                     System1Engine, SystemOneCompiler, TfidfProjector)
from .game import DIRECTIONS

DESCRIPTIONS = (
    "Blocked. Collision.", "Unsafe. Traps the snake.", "Safe. Slower route.",
    "Safe. Best route to food.", "Safe. Eat food now. Best.",
)


class SnakeDecision(DecisionSchema):
    move = ChoiceField(options=list(DIRECTIONS))
    risk = BooleanField()
    food = BooleanField()


def questions_for(descriptions):
    return {
        "move": {"type": "choice", "instructions": "Choose the best safe move toward food.",
                 "criteria": dict(zip(DIRECTIONS, descriptions))},
        "risk": {"type": "noul", "instructions": "Is a safe route available?"},
        "food": {"type": "noul", "instructions": "Is food reachable through empty cells?"},
    }


def state_for(safe, reachable):
    return f"Safe route: {'yes' if safe else 'no'}. Food reachable through empty cells: {'yes' if reachable else 'no'}."


def planner_input(game):
    moves = game.moves()
    safe = [move for move in moves if move.safe]
    preferred = max(safe, key=lambda move: move.advance).direction if safe else None
    reachable, _ = game.food_reachability()
    descriptions = [DESCRIPTIONS[0 if not m.legal else 1 if not m.safe else
                                 4 if m.eats else 3 if m.direction == preferred else 2] for m in moves]
    return state_for(bool(safe), reachable), questions_for(descriptions), {
        "safe": [m.direction for m in safe], "preferred": preferred,
        "reachable": reachable, "legal": [m.direction for m in moves if m.legal],
    }


def feature_text(state, questions):
    """Keep each word attached to its input field, without interpreting its meaning.

    A bag of words otherwise loses which direction each description describes.
    All providers receive the same information; only this representation differs.
    """
    fields = {f"state_{i}": part for i, part in enumerate(state.split("."))}
    fields.update({f"move_{direction}": description for direction, description in questions["move"]["criteria"].items()})
    return " ".join(f"{key}_{word}" for key, value in fields.items()
                    for word in re.findall(r"\w+", value.casefold()))


def lessons():
    splits = {split: [] for split in ("teach", "calibration", "evaluate")}
    for codes in product(range(5), repeat=4):
        # A unique preferred direction, or a trapped state for Boolean teaching.
        winners = [i for i, value in enumerate(codes) if value >= 3]
        trapped = all(value < 2 for value in codes)
        if len(winners) != 1 and not trapped:
            continue
        group = "-".join(map(str, codes))
        bucket = int(sha256(("snake-lessons-v1:" + group).encode()).hexdigest(), 16) % 10
        split = "calibration" if bucket < 2 else "evaluate" if bucket < 4 else "teach"
        for reachable in (False, True):
            state = state_for(not trapped, reachable)
            questions = questions_for([DESCRIPTIONS[value] for value in codes])
            splits[split].append({"group": group, "state": state, "questions": questions,
                                  "labels": {"move": DIRECTIONS[winners[0]] if winners else None,
                                             "risk": not trapped, "food": reachable}})
    return splits


def teach(path):
    data = lessons()
    def samples(split):
        return {field: [(feature_text(row["state"], row["questions"]), row["labels"][field])
                        for row in data[split] if row["labels"][field] is not None]
                for field in ("move", "risk", "food")}
    started = time.perf_counter()
    projector = TfidfProjector.fit([feature_text(row["state"], row["questions"]) for row in data["teach"]], max_features=512)
    model = SystemOneCompiler(SnakeDecision, projector=projector, backend="numpy", regularization=.1).compile(
        samples("teach"), augment=False, calibration_exemplars=samples("calibration"))
    elapsed = (time.perf_counter() - started) * 1000
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    restored = CompiledSystemOneModel.load(path)
    engines = []
    for skill in (model, restored):
        skill.use_cache = False
        engines.append(System1Engine(skill.schema, model=skill, strict_mode=True, use_cache=False))
    records = []
    for row in data["evaluate"]:
        text = feature_text(row["state"], row["questions"])
        before, after = [engine.decide(text, record_receipt=False) for engine in engines]
        assert (before.values, before.probabilities, before.conformal_sets, before.is_ambiguous) == (
            after.values, after.probabilities, after.conformal_sets, after.is_ambiguous)
        records.append({"group": row["group"], "labels": row["labels"], "prediction": after.values,
                        "needs_review": after.is_ambiguous,
                        "correct": all(after.values[field] == label for field, label in row["labels"].items() if label is not None)})
    report = {"teacher": "Deterministic planner vocabulary; synthetic instruction examples, not raw-board reasoning",
              "counts": {key: len(rows) for key, rows in data.items()}, "teaching_ms": elapsed,
              "skill_bytes": path.stat().st_size, "held_out_correct": sum(r["correct"] for r in records),
              "held_out_cases": len(records), "held_out_reviews": sum(r["needs_review"] for r in records),
              "reload_identical": True, "records": records,
              "lessons_sha256": sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()}
    path.with_suffix(".teaching.json").write_text(json.dumps(report, indent=2) + "\n")
    return engines[1], report


class System1Player:
    def __init__(self, path):
        self.engine, self.teaching = teach(Path(path))
        self.metadata = {"name": "System1", "runtime": "NumPy · local CPU", "model": "Taught planner-reading skill",
                         "teaching": {k: v for k, v in self.teaching.items() if k != "records"}}

    def predict(self, state, questions):
        result = self.engine.decide(feature_text(state, questions), record_receipt=False)
        return {"probabilities": result.probabilities["move"], "needs_review": result.is_ambiguous,
                "answers": result.values, "input_tokens": 0, "output_tokens": 0, "teacher_calls": 0}


class LayaPlayer:
    def __init__(self, path):
        from laya_mlx import Agent
        # This comparison names a specific English model; reject different weights.
        with (Path(path) / "model.safetensors").open("rb") as weights:
            weight_hash = file_digest(weights, "sha256").hexdigest()
        if weight_hash != "b9c07bf14be2fa5c78a9193a3e6d840ac80e89e62fc40f425834c3d8a6eaa3de":
            raise ValueError("Expected the pinned English Laya-MLX checkpoint documented in README.md")
        self.agent = Agent(Path(path), device="gpu", dtype="float16", batch_size=3,
                           compile=True, pad_to_multiple=16, cache_prompts=True)
        config = json.loads((Path(path) / "mlx_config.json").read_text())
        self.metadata = {"name": "Laya", "runtime": "MLX · local GPU", "model": "Laya English · 421M",
                         "checkpoint": config, "source": "aac6fef/laya-mlx", "revision": "20aed815fc6acde75733882e7ec0e3f28aeb9717",
                         "weight_sha256": weight_hash, "versions": {name: version(name) for name in ("mlx", "laya-mlx", "tokenizers")},
                         "optimization": "compiled, 16-token padding, tokenization prefix cache; fresh inference each move"}

    def predict(self, state, questions):
        result = self.agent.predict(state, questions)
        return {"probabilities": result["answers"]["move"]["probabilities"], "needs_review": None,
                "answers": result["answers"], "input_tokens": result["usage"]["input_tokens"],
                "output_tokens": result["usage"]["output_tokens"], "teacher_calls": 0}


class JevPlayer:
    def __init__(self, key):
        if not key:
            raise ValueError("Set TYPESAFE_API_KEY in the server environment or --credentials file")
        self.key = key
        self.metadata = {"name": "Jev", "runtime": "TypeSafe · remote API", "model": "jev-latest"}

    def predict(self, state, questions):
        from system1.compat.typesafe import call_real_typesafe_api
        result, _, _ = call_real_typesafe_api(state, questions, api_key=self.key, zero_egress=False,
                                             fallback_baseline=False, timeout=15)
        data = dict(result)
        self.metadata["model"] = data.get("model", self.metadata["model"])
        usage = data.get("usage", {})
        return {"probabilities": data["answers"]["move"]["probabilities"], "needs_review": None,
                "answers": data["answers"], "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0), "teacher_calls": 1}
