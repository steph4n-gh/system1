# Teach a piece. Change the behavior.

An experimental courier playground made from three ordinary System1 skills:
choose an objective, classify adjacent terrain, then rank eligible moves.
Teach the pieces separately, combine them on a withheld mission, and correct
one piece while keeping the others unchanged. No API key, LLM or new dependency.

## Run it

From the repository root, with System1 installed:

```bash
python -m examples.gaming.skill_playground
python -m examples.gaming.skill_playground --serve
```

Open http://127.0.0.1:8789/. The first command generates lessons, teaches and
reloads the skills, and evaluates 40 maps. The second opens a local HTTP service
using those artifacts. `--serve` also performs that preparation if the default
output directory has no report. Use `--output-dir`, `--episodes`, `--seed` and
`--port` to change the experiment or local service.

1. Run the default mission. Purple is now dangerous, but the original lesson
   still says it is safe. The three lanes show actual local calls.
2. Select **Teach 8 purple examples**. This actually recompiles and reloads the
   terrain skill, using eight new fitting labels and 22 separate calibration
   labels. It replaces the obsolete purple examples and retains other surfaces.
3. Run the same mission again. Only the skill network receives that correction;
   the single model and original handwritten policy keep their old knowledge.
4. Switch mission or map, or restore the original lesson. Single-step mode shows
   the chosen objective, terrain decisions and movement scores. Crossed-out
   scores were excluded by the learned terrain classifier.

The key unlocks the gate when entered. Supplies are picked up on contact; `H`
is home. Maps have isolated hazard patches, a single gate and fixed room roles.
The display advances one decision per lane per animation tick; it is not a speed
race. Timings measure the actual complete controller call, excluding animation.

## What is learned, and what is code?

| Part | Input | Learned output |
|---|---|---|
| Objective | Inventory, gate state, room and position; explicit pair features | Key, gate, supplies or home |
| Terrain | Surface appearance, lighting and texture | Safe or dangerous |
| Movement | Direction relative to the selected target, predicted risk, local visit count | A preference score for one candidate move |

All three use the existing NumPy ridge compiler and fitted TF-IDF features.
Each `.s1m` carries its schema, features, weights and calibration. Each tick uses
one objective call, four terrain calls and four movement calls. Exact and fuzzy
decision caches are disabled. There is no teacher call during local execution.

The application supplies observable coordinates, categorical facts and short
visit memory. It looks up the chosen target's coordinates, excludes moves the
terrain skill marks dangerous **or uncertain**, and takes the highest remaining
movement score. If no move qualifies it stops for review. This gate uses learned
predictions; it never consults the world's true hazard labels. Physics still
prevents movement through walls and locked gates.

Other review flags are recorded but do not stop the simulator. Score intervals
do not certify a ranking, and individual skill calibration does not establish
whole-mission accuracy. Completing a simulated mission is different from
completing it without review flags.

The objective skill deliberately does not receive the mission termination flag:
its job is to select the next object from inventory and gate state. The world
ends a collect-only mission when supplies are collected. This interface design
is part of the experiment, not something System1 discovered automatically.

## Teaching and evidence

[Measured results](RESULTS.md) include the failed first design and the revised
network. Teaching uses two mission families: unlocked gate + return, and locked
gate + collect. Locked gate + return is withheld as a full sequence. Local skill
inputs can recur; this is composition of familiar abilities, not unfamiliar
concepts, raw pixels, or natural-language reasoning.

Fitting map seeds are 0–99; calibration candidates use 100–149. Repeated effective
inputs stay in one partition. Objective input groups are reserved for calibration
before fitting. Terrain and movement lessons are separate synthetic microtasks
with disjoint effective fitting/calibration inputs. Development used seeds
2000–2009. The first evaluation used 3000–3039; after discovering the danger
penalty failure, the revised gate was evaluated on fresh seeds 4000–4039.

The single System1 model sees the same available world facts, all target offsets,
adjacent terrain and visit counts, plus the mission flag. It directly predicts
an action. It receives the same **total retained fitting and calibration label
counts** as the network, using additional unique examples from the teaching
families. Intermediate supervision and feature representations differ, so this
is not a controlled demonstration that every cascade outperforms every single
model. The handwritten baseline is the teaching policy itself; it is also
updated explicitly in the offline changed-terrain comparison.

Generated `lessons.json` contains editable `input`, `label` and `split` records.
For example:

```json
{"input": "surface_purple light_day texture_rough", "label": "dangerous", "split": "teach"}
```

To teach a file directly, reuse the same helper as the GUI:

```python
import json
from pathlib import Path
from examples.gaming.skill_playground.skills import teach

rows = json.loads(Path("my-terrain-lessons.json").read_text())
terrain = teach("terrain", rows, Path("terrain.s1m"))
```

Use separate examples for fitting and calibration, include both labels, and
check the **whole mission** after changing any skill. The correction control is
a small fixed demonstration, not a general annotation application.

## Scope

This adds an example, not a new core API or a new neural architecture. The
composition and interfaces are written by us. Automatic discovery of missing
skills, automatic graph growth, transfer to unrelated environments and stronger
end-to-end uncertainty handling remain research questions. Classical policies
are the simpler choice when the rules are already known and easily maintained.
