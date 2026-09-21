# Teach a better Pokémon battle decision

This experimental lab composes two ordinary System1 skills: one scores available
moves, then the other decides whether to attack or heal. A live GUI lets you run a
battle, correct only the healing lesson, and try again. It also compares both healing lessons from identical saved battle states in a
user-supplied Pokémon Red/Blue ROM, including actual potion-menu control.

The useful result is local, isolated teaching. This is not a new neural architecture,
a full-game player, or a demonstration of learning Pokémon from pixels.

## Run it

From a repository checkout:

```bash
python -m pip install -e .
python -m examples.gaming.pokemon_teaching --serve
```

Open the printed local URL, normally `http://127.0.0.1:8790/`. Run the illustrative
survival encounter, seed **8004**. The original skill loses. Click **Teach the
correction**, then run again: only healing is recompiled, and the corrected skill
wins. **Restore original lesson** reverses it. Every turn makes fresh model calls;
the GUI does not replay recorded decisions. The result table is recorded evidence.

The correction replaces **21 fitting labels and 6 separate calibration labels**.
All other lessons are retained. The move engine, its saved file and its predictions
for identical inputs remain unchanged. A changed healing action can change later
observations, so subsequent battle trajectories are not generally identical.

Enable **Stop before a decision flagged for review** to test actual abstention.
With it disabled, the simulation executes flagged predictions and counts any
legal-action overrides. Baseline lanes retain their original lessons; the table
also includes separately measured, corrected single-model and rule baselines.

For the optional real emulator:

```bash
python -m pip install -e '.[gameboy]'
python -m examples.gaming.pokemon_teaching --serve --rom '/path/to/Pokemon Red.gb'
```

In **The same save. One changed skill**, click **Load same battle**, then
**Run comparison**. Initially both lanes use the original lesson. Click **Teach healing** in that section: both games reset to the identical starting save, and only
the right lane’s healing engine changes. Run again. The default `threat_slow-37`
illustrates a loss becoming a win; it was selected after evaluating the entire
fixed suite. **Restore original healing** reverses it. The ROM section has its own
**Stop for review** checkbox, enabled by default. Click **Load same battle** to
apply a different fixture or review setting.

The optional `gameboy` extra includes
PyBoy and Pillow for the live screen. Unmodified English Red/Blue binaries are
checked against the hashes published by
[pret/pokered](https://github.com/pret/pokered/blob/master/roms.sha1).
The ROM is read through an in-memory file, and the user's battery save is neither
loaded nor overwritten. No cartridge data, saves, or sprites are distributed.

## What the lesson means

An example is an observation paired with the desired answer. The original healing
lesson says to use an available potion below 25% HP. That misses a simple case:
27/80 HP is above the threshold, but the next attack can still knock the player out.
The correction keeps the critical-HP rule and adds healing when a normal incoming
hit could be lethal, a potion restores more than that hit, and attacking is unlikely
to finish the opponent immediately.

The generated `lessons.json` contains plain rows:

```json
{"input": "observed game facts and their pair features", "label": "use_item", "split": "teach"}
```

`input` is the observation; `label` is what you want System1 to choose; `split`
keeps fitting examples separate from calibration checks. The GUI applies a
predefined correction using `corrected_rows()` in [battle.py](battle.py).
It is not a free-form conversational teacher. To teach another rule, edit the
JSON rows and use the same helper:

```python
from pathlib import Path
import json
from examples.gaming.pokemon_teaching.battle import teach

rows = json.loads(Path('.system1/pokemon-teaching/lessons.json').read_text())['healing']
# Edit labels and provide separate calibration observations for the intended rule.
engine = teach('healing', rows, Path('.system1/my-healing.s1m'))
```

Move inputs include power, accuracy, level, attack/defense ratio, type effectiveness
and same-type bonus. Healing inputs include HP, known damage estimates, inventory
and their relationships. These are explicit game facts. System1 learns the bounded
choice from these inputs; it does not discover the damage formula or infer hidden
opponent moves. The example uses NumPy CPU and fitted TF-IDF, with caching and
receipt generation disabled. No LLM or teacher runs during model-controlled play.

## Measured results and limits

Final exploratory cohort: **60 seeds, 8000–8059**, per family, on Apple M4 Pro,
Python 3.13 / NumPy. Every controller receives the same initial encounter and RNG
seed. The simulator is the existing `run_battle_simulation`: the player acts first,
damage/status mechanics are simplified, and each battle has a 25-turn limit.
These are simulator results, not cartridge campaign win rates.

| Encounter family | Two skills before | Two skills after | Corrected single model | Corrected rules |
|---|---:|---:|---:|---:|
| Healthy start | 49/60 | 51/60 | 48/60 | 51/60 |
| Low-health start | 36/60 | 37/60 | 35/60 | 37/60 |
| Mixed | 40/60 | 40/60 | 39/60 | 40/60 |
| Constructed survival stress test | 13/60 | 44/60 | 27/60 | 44/60 |

The survival family deliberately exposes the HP-threshold weakness: 21–35 HP,
a strong neutral attack, 50–70 enemy HP, and two healing items. It was added after
broader experiments showed limited gains, without filtering seeds by outcomes.
The large gain applies to this constructed family. **Two survival seeds regressed:
8032 and 8055.** There were 33 improved seeds; seed 8004 was selected afterward to
illustrate one improvement in the GUI. These experiments do not establish a broad
or statistically significant advantage.

Strict runs stop on every review flag. Corrected wins with that requirement were
**49/60 healthy, 31/60 low-health, 35/60 mixed, and 44/60 survival**. On the low-health
and mixed sets, the correction reduced completion without review compared with
the original skill, despite equal or slightly higher overall wins. Thus this is an
experimental lesson, not a recommendation to promote it to unattended use.

The two skills used **1,494 fitting labels and 480 calibration labels**, exactly
the retained budgets of each single-model baseline. Effective input strings are
grouped before splitting. Intermediate supervision and annotation difficulty are
not equivalent. Encounter seeds are unseen; familiar local skill inputs can recur.
The lesson generator uses seeds 0–1199, with additional single-model examples from
100000 onward to match the retained budgets. There are no paid teacher calls.

The corrected pair occupies **57,906 bytes**. A complete decision took approximately
**0.33–0.44 ms median**; the single-model baselines took **0.19–0.24 ms**, and the
handwritten policies **0.007–0.010 ms**. Known rules remain much cheaper. Recompiling,
saving and reloading the corrected healing skill took approximately **43 ms**.
Teaching time excludes human labeling; skill sizes exclude the installed runtime.

The survival gain consumes more resources: **41 → 109 potions across 60 episodes**
when flagged predictions are executed. The report includes turns, items, review
flags, legal overrides, model calls, timing and per-seed changes for every baseline.

## Actual cartridge battle experiment

The original, unmodified starter-rival compatibility run is retained: **five turns,
8/20 HP remaining, no review flags and no teacher calls**. Select `starter` in the
GUI for this encounter. It has no healing items.

The new comparison uses **18 controlled save-state fixtures**: six conditions at
three initial idle timings (37, 71 and 109 frames). The conditions are healthy,
critical HP, a threatening slower opponent, a threatening faster opponent,
Super Potion availability, and no items. Each fixture changes starting HP, bag
contents, and enemy attack/speed in the first rival battle. The stronger opponent
stats are artificial stress inputs, not legal level-5 stats or naturally reached
campaign encounters. These are six conditions with timing variants, not 18
independent real-world encounters.

Every controller loads the **same complete emulator state**, including RNG and
hardware timing, for each fixture. After setup there are no RAM writes: controllers
press buttons, and the cartridge handles damage, misses, critical hits, stat
changes, move order and item effects. Different button sequences can change later
random draws. The existing saved skills are frozen; these fixtures are not used for
fitting or calibration. The suite definition is in [rom_experiment.py](rom_experiment.py).

| Controller | Wins / 18 | Turns played | Potions used | Review stops |
|---|---:|---:|---:|---:|
| Two skills, original | 6 | 75 | 5 | 0 (flags executed) |
| Two skills, corrected | 7 | 135 | 20 | 0 (flags executed) |
| Original, stop for review | 2 | 60 | 4 | 4 |
| Corrected, stop for review | 6 | 132 | 20 | 1 |
| Original handwritten rule | 6 | 75 | 5 | 0 |
| Corrected handwritten rule | 7 | 135 | 20 | 0 |

There are **two improvements (`threat_slow-37`, `threat_slow-71`) and one regression
(`threat_slow-109`)**. Teaching transfers to real item use, but overall gains are
small and consume more resources. Strict wins are wins completed without review,
not a measure of classifier accuracy. Some losses remain confidently wrong as
strategic choices; calibration cannot certify that a short-term policy wins games.

Both frozen single-model baselines won **0/18**. The original stopped on an
unavailable move in 12 cases and lost six battles; the corrected version stopped
on an unavailable move in 15 cases and lost three. They were taught four-slot
simulator encounters and do not reliably transfer to the starter’s two available
moves. Unsupported outputs stop visibly, with no hidden replacement action.
This exposes a limitation of these particular frozen baselines, not a general
claim that multiple heads outperform all single models.

The pair took about **0.6 ms per decision** in the first run; handwritten rules
about **0.02 ms**, and single-model predictions about **0.4 ms**. These measure
inference only, excluding emulation, rendering, setup and teaching. Exact measured
timings accompany the results. All runs made zero teacher/API calls.

### What the failures taught us

In simulator seeds 8032 and 8055, the correction uses the second potion earlier.
It delays attacking, changes the subsequent random sequence, and still loses to
critical hits. In actual fixture `threat_slow-109`, earlier healing also produces
a different battle trajectory: attacks make less progress while potions run out.
The original wins when flags execute; the corrected lesson loses. We preserve
these traces rather than teaching exceptions keyed to seed names.

The lesson knows immediate estimated damage and HP; it does not plan several
turns ahead or account fully for game-specific status and speed effects. In
particular, a predicted finishing attack is not guaranteed to land before a faster
opponent. That is a concrete boundary for the next teaching experiment, not a
reason to grow the skill network yet.

### Controller details

The strict reader in [red.py](red.py) uses observed stats, types, moves, PP and bag
contents. Enemy moves are privileged RAM facts. The controller chooses the same
item preference exposed to the healing skill (Super Potion, otherwise Potion),
finds it in the real bag, selects the active party member, and verifies that exactly
one requested item was consumed. It handles shifted bag positions after use.
Strict mode stops before sending buttons for a flagged decision. The GUI preserves
the move-engine object and file when replacing the healing engine.

RAM layouts and item effects follow
[pret/pokered](https://github.com/pret/pokered/blob/master/engine/items/item_effects.asm);
emulation uses the [PyBoy API](https://docs.pyboy.dk/). No ROM, emulator save, battery
save or cartridge screenshot is published in the evidence archive.

## Evidence and reproduction

```bash
python -m examples.gaming.pokemon_teaching --episodes 60 --seed 8000
python -m pytest -q tests/test_pokemon_teaching.py
```

[Summary](results/summary.json) · [Exact files and traces](results/battle-evidence.zip)
· [Earlier rounds, including regressions](results/earlier-rounds.json.gz)

The archive contains the actual saved skills, lesson rows, all 1,680 controller
episodes, source snapshot and hashes, plus numeric ROM observations. Tests verify
hashes, replay every simulator action, reconcile reported totals, prohibit teacher
calls during saved-skill execution, and verify isolated live correction.

The earlier rounds remain visible: an aggressive two-hit healing policy on seeds
4000–4059 improved low-health outcomes but caused healthy-battle regressions;
the narrowed policy on seeds 7000–7059 provided little broad gain and also had a
regression. Those results motivated the explicitly targeted survival experiment.
The courier's separate uncertainty follow-up is documented
[here](../skill_playground/RESULTS.md#objective-teaching-follow-up).

Reproduce the real-engine comparison with a local ROM (optional gameboy extra):

```bash
python -m examples.gaming.pokemon_teaching --rom-suite --rom '/path/to/Pokemon Red.gb'
```

The command reuses the saved skills in `--output-dir`, or prepares them if absent.
It writes the full report and local-only checkpoints there. It runs every declared
fixture once for each of eight controllers; an error stops the command instead of
retrying or substituting a successful result.

[Real-engine summary](results/rom-summary.json) ·
[144 battle traces, frozen skills and source hashes](results/rom-evidence.zip)

Normal CI checks hashes, report totals, identical initial state across controllers,
legal dispatch and review stops without requiring a ROM. For the optional actual
emulator integration tests, set `SYSTEM1_TEST_POKEMON_ROM` to the local ROM path and
run `python -m pytest -q tests/test_pokemon_rom.py`. Those tests exercise item use,
isolated correction, checkpoint reuse and the retained regression.
