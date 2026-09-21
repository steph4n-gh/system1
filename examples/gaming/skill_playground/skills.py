"""Three ordinary saved System1 skills and transparent Python wiring."""
from pathlib import Path
from system1 import ChoiceField, DecisionSchema, ScoreField, System1Engine
from system1.compiler import CompiledSystemOneModel, SystemOneCompiler
from system1.core.text import TfidfProjector
from .world import MOVES, facts, terrain_prompt, move_facts, move_prompt, flat_prompt


class Objective(DecisionSchema):
    target = ChoiceField(options=["key", "gate", "supply", "home"])


class Terrain(DecisionSchema):
    risk = ChoiceField(options=["safe", "dangerous"])


class Movement(DecisionSchema):
    preference = ScoreField(min_value=0, max_value=1)


class Flat(DecisionSchema):
    action = ChoiceField(options=list(MOVES))


SCHEMAS = {"objective": Objective, "terrain": Terrain, "movement": Movement, "flat": Flat}


def teach(name, rows, path):
    """Rows are plain JSON: input, label, split. No template augmentation."""
    fit = [row for row in rows if row["split"] == "teach"]
    calibration = [row for row in rows if row["split"] == "calibration"]
    schema = SCHEMAS[name]
    field = next(iter(schema().fields))
    projector = TfidfProjector.fit([row["input"] for row in fit], max_features=512)
    compiler = SystemOneCompiler(schema, projector=projector, regularization=.01)
    model = compiler.compile({field: [(r["input"], r["label"]) for r in fit]}, augment=False,
                             calibration_exemplars={field: [(r["input"], r["label"]) for r in calibration]})
    model.save(path)
    restored = load(path)
    model.use_cache = False
    original = System1Engine(model.schema, model=model, strict_mode=True, use_cache=False)
    for row in calibration:
        before, after = ask(original, row["input"]), ask(restored, row["input"])
        if (before.values, before.probabilities, before.conformal_sets, before.is_ambiguous) != (after.values, after.probabilities, after.conformal_sets, after.is_ambiguous):
            raise AssertionError("Saved skill changed its decision during reload")
    return restored


def load(path):
    model = CompiledSystemOneModel.load(Path(path))
    model.use_cache = False
    return System1Engine(model.schema, model=model, strict_mode=True, use_cache=False)


def ask(engine, prompt):
    return engine.decide(prompt, record_receipt=False)


class Network:
    def __init__(self, directory, corrected=False):
        root = Path(directory)
        self.objective = load(root / "objective.s1m")
        self.terrain = load(root / ("terrain-corrected.s1m" if corrected else "terrain.s1m"))
        self.movement = load(root / "movement.s1m")

    def decide(self, world):
        selected = ask(self.objective, facts(world))
        target = selected.values["target"]
        scores, risks = {}, {}
        review = selected.is_ambiguous
        for action, (dx, dy) in MOVES.items():
            surface = world.terrain((world.pos[0] + dx, world.pos[1] + dy))
            terrain = ask(self.terrain, terrain_prompt(world, surface))
            risk = terrain.values["risk"] == "dangerous"
            movement = ask(self.movement, move_prompt(move_facts(world, getattr(world, target), action, risk)))
            scores[action] = movement.values["preference"]
            risks[action] = {"surface": surface, "risk": risk, "review": terrain.is_ambiguous}
            review |= terrain.is_ambiguous or movement.is_ambiguous
        # This gate uses only the taught classifier, never world truth.
        eligible = [a for a in MOVES if not risks[a]["risk"] and not risks[a]["review"]]
        action = max(eligible, key=scores.get) if eligible else None
        return {"action": action, "eligible": eligible, "target": target, "scores": scores,
                "terrain": risks, "review": bool(review or not eligible), "calls": 9}


class Single:
    def __init__(self, directory):
        self.engine = load(Path(directory) / "flat.s1m")

    def decide(self, world):
        result = ask(self.engine, flat_prompt(world))
        return {"action": result.values["action"], "target": None,
                "scores": result.probabilities["action"], "review": result.is_ambiguous, "calls": 1}
