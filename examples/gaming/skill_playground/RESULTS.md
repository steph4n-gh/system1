# Courier composition experiment

Measured on Apple M4 Pro / macOS, Python 3.13, NumPy CPU. All local inference,
with caches and receipts disabled. Forty fresh map seeds, 4000–4039, for each
mission family. A 180-move limit applies equally to all controllers.

| Mission | Three skills | Single System1 | Handwritten policy |
|---|---:|---:|---:|
| Familiar: unlocked gate + return | 40/40 | 29/40 | 40/40 |
| Familiar: locked gate + collect | 40/40 | 37/40 | 40/40 |
| Withheld combination: locked gate + return | 40/40 | 0/40 | 40/40 |
| Changed purple rule, original knowledge | 0/40 | 0/40 | 0/40 |
| Changed purple rule, terrain skill corrected | 40/40 | Not corrected | 40/40 with explicitly updated rules |

The improvement is **0/40 → 40/40** after changing only the terrain skill.
Eight new fitting labels and 22 new calibration labels were supplied. Every
objective and movement probe returned identical predictions before and after;
the two familiar families and unchanged-terrain composite missions retained
identical trajectories. Loading a fresh agent requires the saved skills only.

The network used 762 fitting labels and 247 calibration labels in total. The
single model received exactly the same respective counts. These are retained
labels, not equivalent annotation difficulty: the network gets intermediate
labels and a deliberately factored interface. Labels were produced by a small
handwritten policy; no paid teacher was invoked. Generating the candidate lessons
is measured separately from fitting and serialization.

The three skills total about **24 KB**. A complete network step makes nine model
calls and takes about **1 ms** median. The single model is about **0.2 ms** and
the handwritten policy about **0.003 ms**. This experiment demonstrates useful
composition and isolated correction; it does not beat known rules on speed.
Exact bytes, timings, label hashes and per-suite results are in
[the summary](results/summary.json).

## Review flags limit the autonomy claim

Of the 40 successful withheld-combination missions, **23** completed without
any review flags. After the terrain correction, **19/40** changed-terrain missions
completed without any flags. The simulator executes objective/movement
predictions despite flags; the learned terrain gate does exclude uncertain or
dangerous terrain. There were no gate-induced review stops in this evaluation.
These numbers do not establish reliable unattended operation, and there is no
claim of statistical significance from one small structured environment.

## Retained failure and the resulting fix

The first independent cohort, seeds 3000–3039, used a numerical danger penalty.
It completed 40/40 original composite missions and 39/40 changed-terrain missions
after correction. On seed 3010, the movement score outweighed the correctly
predicted danger of lava after other routes accumulated visit penalties.
The decision was flagged for review but the simulation executed it.

We retained [that run, lessons and source](results/initial-penalty-run.json.gz),
then made the terrain decision an explicit eligibility gate. The revised
handwritten baseline uses the equivalent known-rule gate. We evaluated the
revised network on fresh seeds 4000–4039 and added the original failing map as a
regression test. The final cohort was rerun only to add complete compact action,
latency and review traces; its outcomes remained unchanged.

## Reproduce and inspect

```bash
python -m examples.gaming.skill_playground --episodes 40 --seed 4000
python -m pytest -q tests/test_skill_playground.py
```

[The evidence archive](results/courier-evidence.zip) contains the exact saved
skills, full report, per-episode moves/review flags/timings, detailed first-map
replays, fitting/calibration examples, correction examples and source snapshot.
Tests verify archive hashes and replay every recorded action against world
physics, then reconcile outcomes and timings with the reported aggregates.

The runtime tests additionally block every teacher function during local play,
load an independent agent from copied `.s1m` files alone, reproduce the failure
and correction, and check that the other skills and old mission trajectories
remain unchanged. The GUI uses fresh model calls, not these recorded traces.

This is a positive result for a deliberately designed skill decomposition in a
small synthetic world. It does not show that System1 discovers that decomposition,
learns arbitrary navigation, or outperforms other modular-learning approaches.


## Objective teaching follow-up

A separate round retains the original model and adds local observations of
returning with key, open gate and cargo: **104 fitting and 45 calibration labels**.
No thresholds changed. On fresh seeds **6000–6039**, stopping before every flagged
action improved successful completion from **22/40 to 31/40**, and from **19/40 to
28/40** with the corrected terrain lesson. Both versions still completed 40/40
when flagged predictions were executed; all those action trajectories were unchanged.

The remaining review stops are real. Empty conformal sets can occur even when the
highest-scoring class is correct; per-decision calibration does not guarantee an
entire mission. This round improves useful acceptance without establishing reliable
unattended operation. Development used seeds 5000–5009; added lessons are isolated
inventory-state observations, not complete trajectories through the withheld mission.

Run `python -m examples.gaming.skill_playground.quality` to reproduce this round.
The [follow-up summary](results/quality-summary.json) and
[hashed lessons, skills and traces](results/quality-evidence.zip) preserve strict
stops and the unchanged original comparison. The original evidence above remains
frozen. The next [Pokémon teaching experiment](../pokemon_teaching/README.md) tests
isolated correction in a second workload, with broader and targeted battle families.
