"""Courier world physics and observations. No planner or teacher in this module."""
from collections import Counter
from dataclasses import dataclass, field
import random

MOVES = {"east": (1, 0), "south": (0, 1), "west": (-1, 0), "north": (0, -1)}


@dataclass
class World:
    seed: int
    locked: bool = True
    return_home: bool = True
    purple_dangerous: bool = False
    purple: bool = False
    width: int = 11
    height: int = 9
    ticks: int = 0
    has_key: bool = False
    has_supply: bool = False
    alive: bool = True
    delivered: bool = False
    paused_for_review: bool = False
    visits: Counter = field(default_factory=Counter)

    def __post_init__(self):
        rng = random.Random(self.seed)
        self.gate = (5, rng.randrange(2, 7))
        self.home = (1, rng.randrange(2, 7))
        self.key = (3, rng.randrange(2, 7))
        self.supply = (9, rng.randrange(2, 7))
        self.pos = self.home
        self.gate_open = not self.locked
        self.visits[self.pos] = 1
        # Isolated patches leave space to go around; never cover a task object.
        protected = {self.home, self.key, self.supply, self.gate, (4, self.gate[1]), (6, self.gate[1])}
        candidates = [(x, y) for x in (2, 7, 8) for y in range(1, 8) if (x, y) not in protected]
        rng.shuffle(candidates)
        self.hazards = {candidates[0]: "lava", candidates[1]: "lava"}
        if self.purple:
            # A changed surface rule on a frequently used lane, with a detour.
            self.hazards[(7, self.gate[1])] = "purple"
        self.light = ("day", "dusk", "night")[self.seed % 3]
        self.texture = ("smooth", "rough", "grainy")[self.seed % 3]

    @property
    def done(self):
        return self.paused_for_review or not self.alive or self.delivered or (self.has_supply and not self.return_home) or self.ticks >= 180

    @property
    def success(self):
        return self.alive and (self.delivered or (self.has_supply and not self.return_home))

    def terrain(self, pos):
        x, y = pos
        if not (0 < x < self.width - 1 and 0 < y < self.height - 1):
            return "wall"
        if x == self.gate[0] and pos != self.gate:
            return "wall"
        return self.hazards.get(pos, "floor")

    def step(self, action):
        if self.done:
            raise ValueError("Episode has ended")
        if action not in MOVES:
            raise ValueError("Unknown move")
        dx, dy = MOVES[action]
        nxt = self.pos[0] + dx, self.pos[1] + dy
        terrain = self.terrain(nxt)
        self.ticks += 1
        if terrain != "wall" and (nxt != self.gate or self.gate_open or self.has_key):
            self.pos = nxt
            if terrain == "lava" or (terrain == "purple" and self.purple_dangerous):
                self.alive = False
            if nxt == self.key:
                self.has_key = True
            if nxt == self.gate and self.has_key:
                self.gate_open = True
            if nxt == self.supply:
                self.has_supply = True
            if nxt == self.home and self.has_supply:
                self.delivered = True
        self.visits[self.pos] += 1

    def snapshot(self):
        return {"seed": self.seed, "pos": self.pos, "home": self.home, "key": self.key,
                "gate": self.gate, "supply": self.supply, "has_key": self.has_key,
                "gate_open": self.gate_open, "has_supply": self.has_supply,
                "return_home": self.return_home, "locked": self.locked,
                "alive": self.alive, "success": self.success, "done": self.done,
                "paused_for_review": self.paused_for_review,
                "ticks": self.ticks, "width": self.width, "height": self.height,
                "tiles": [[self.terrain((x, y)) for x in range(self.width)] for y in range(self.height)]}


def facts(world):
    """Only observed facts. Pair features expose interactions, not target labels."""
    side = "left" if world.pos[0] < 5 else "right" if world.pos[0] > 5 else "gate"
    flags = [f"side_{side}", f"key_{int(world.has_key)}", f"open_{int(world.gate_open)}",
             f"cargo_{int(world.has_supply)}"]
    pairs = [a + "__" + b for i, a in enumerate(flags) for b in flags[i + 1:]]
    return " ".join(flags + pairs + [f"x_{world.pos[0]}", f"y_{world.pos[1]}", f"gate_y_{world.gate[1]}"])


def terrain_prompt(world, surface):
    return f"surface_{surface} light_{world.light} texture_{world.texture}"


def move_facts(world, target, action, risk):
    dx, dy = MOVES[action]
    tx, ty = target
    px = dx * ((tx > world.pos[0]) - (tx < world.pos[0]))
    py = dy * ((ty > world.pos[1]) - (ty < world.pos[1]))
    visits = min(3, world.visits[(world.pos[0] + dx, world.pos[1] + dy)])
    return px, py, int(risk), visits


def move_prompt(values):
    px, py, risk, visits = values
    names = {-1: "away", 0: "level", 1: "toward"}
    return f"horizontal_{names[px]} vertical_{names[py]} risk_{risk} visits_{visits}"


def flat_prompt(world):
    """Same raw facts, all target offsets and adjacent surfaces; no taught outputs."""
    parts = [facts(world), f"return_{int(world.return_home)}"]
    for name in ("key", "gate", "supply", "home"):
        tx, ty = getattr(world, name)
        for action, (dx, dy) in MOVES.items():
            nxt = world.pos[0] + dx, world.pos[1] + dy
            values = move_facts(world, (tx, ty), action, False)
            parts.extend(f"{name}_{action}_{token}" for token in move_prompt(values).split())
            parts.append(f"{action}_surface_{world.terrain(nxt)}")
    parts.append(f"light_{world.light} texture_{world.texture}")
    return " ".join(parts)
