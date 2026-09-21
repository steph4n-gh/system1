"""Explicit rule teacher and handwritten baseline. Never called by local skills.

This is a greedy courier policy with visit memory, not shortest-path planning.
It is deliberately a strong, cheap baseline for this small structured world.
"""
import math
from .world import MOVES, move_facts


def objective(world):
    if world.has_supply:
        return "gate" if world.pos[0] > 5 else "home"
    if not world.gate_open:
        return "gate" if world.has_key else "key"
    return "gate" if world.pos[0] < 5 else "supply"


def dangerous(surface, purple_dangerous=False):
    return surface in ("wall", "lava") or (surface == "purple" and purple_dangerous)


def preference(values):
    px, py, risk, visits = values
    logit = .25 * (1.5 * px + py - 6 * risk - 2 * visits)
    return 1 / (1 + math.exp(-logit))


def decide(world, purple_dangerous=False):
    target = objective(world)
    scores, eligible = {}, []
    for action, (dx, dy) in MOVES.items():
        surface = world.terrain((world.pos[0] + dx, world.pos[1] + dy))
        if not dangerous(surface, purple_dangerous):
            eligible.append(action)
        scores[action] = preference(move_facts(world, getattr(world, target), action,
                                                dangerous(surface, purple_dangerous)))
    return {"action": max(eligible, key=scores.get) if eligible else None, "target": target, "scores": scores,
            "review": not eligible, "calls": 0}
