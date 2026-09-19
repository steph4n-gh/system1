# First-use game decisions

Run from the repository root:

```bash
python benchmarks/quality/evaluate_zero_shot.py --projectors
```

The command blocks network access, compilation, teaching, and calibration. It
disables decision caching and writes `.system1/zero-shot/report.json`. It exits
unsuccessfully if any game case fails, any Pokémon action is illegal, or the raw
routing control falls below its 8/12 baseline. The regression runs in pytest too.
`--projectors` adds an optional comparison of two existing text projectors.

## What changed

The default Pokémon agent ranks **the moves currently available**, using type
effectiveness, power, the attacker's and defender's stats, same-type attack bonus,
and nominal accuracy. It heals below 25% HP when a healing item is available.
It excludes exhausted moves, absent items, fainted or active party members as
switch targets, and escape from trainer battles. With no PP it uses Struggle.
Move names and slot positions do not supply the strategy.

A supplied engine or taught skill retains its legal strategy. Invalid actions
and moves receive a legal fallback, and the override requests review. Telemetry
keeps the original model suggestion separately; a game-rule decision has no
model confidence or conformal set assigned to it.

Paperclips now compares clip price with the observed cost **per unit of wire**,
rather than assuming a 1,000-unit spool. Phase-two purchases must fit the observed
clip budget. Missing prices are unavailable; live reads include spool size and
drone, factory, solar-farm, and battery costs. A failed live observation stops
execution instead of substituting a fabricated starting state. The accelerated
mock explicitly uses zero purchase costs by default and honors supplied costs.

These are explicit application rules using known game facts. Paperclips already
used a handwritten policy; Pokémon's default demo now does too. They improve
first-use behavior without teaching a model or changing the core classifier.
The teaching, observation, cutover, and saved-skill APIs are unchanged.

## Frozen cases and results

The [cases](cases.json) were frozen before these changes. The
[baseline](baseline.json) was measured at the `v1.0.0` commit
`b45b6f9f71df82178bb7b1e35fdf9a6e324b987d`. Both reports use case-file SHA-256
`3fc27d78d71476f487dcf962452e2fdc0124f0fb01f19e5ef216b1c4c008b651`.
The [new report](results.json) includes every prediction and local timing.

| Workload | Before | After |
|---|---:|---:|
| Pokémon first decisions | 6/24 correct | 24/24 correct |
| Paperclips first decisions | 8/12 correct | 12/12 correct |
| Raw routing control | 8/12 correct | 8/12 correct |

The Pokémon cases rotate move slots, exhaust PP, remove a slot, vary opponent
types, and vary HP/item availability. Paperclips varies inventory, demand, spool
size, and purchase affordability. All 24 resulting Pokémon actions are legal.
The raw Pokémon model suggestion remains 6/24 correct: the gain comes from the
game policy. Review requests are retained in the report; correct suggestions do
not establish calibrated automatic acceptance.

These are small authored regression cases with explicit immediate-damage/healing
labels, not an independent gameplay benchmark. They do not measure long-term
strategy, status-move tactics, campaign completion, a real ROM, or speed records.
The Pokémon damage ranking uses nominal accuracy even though the simplified
simulator does not roll accuracy misses. Paperclips' mock accelerates production
and purchases; its progress is not evidence of real-game economics or timing.

## Existing text projector comparison

One fixed comparison uses the existing 100-case intent-routing and security-triage
development datasets, 384 features, NumPy, strict mode, no caching, and no
teaching. It does not tune parameters or change the library default.

| Raw predictions | Default projector | Existing hybrid projector |
|---|---:|---:|
| Intent routing | 51/100 correct | 60/100 correct |
| Security triage | 50/100 correct | 56/100 correct |

**Both projectors request review on every case.** Accepted accuracy is undefined,
not 100%. These development datasets have informed prior work, so the results
are evidence for an experiment, not a deployment guarantee.

The existing option can be tried on a task's own data without adding a dependency:

```python
from system1 import HybridProjector, System1Engine

engine = System1Engine(
    YourDecisionSchema,
    projector=HybridProjector(dimension=384, backend="numpy"),
    dimension=384,
    backend="numpy",
    strict_mode=True,
)
```

HybridProjector combines hashed lexical features with existing curated semantic
anchors. It is not a pretrained language model. Teaching representative examples
and checking fresh cases remains the path to a reliable reusable skill.
