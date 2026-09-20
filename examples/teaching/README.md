# Teach a bounded skill

New to teaching? Start with the [first-skill walkthrough](../../docs/guides/first_skill.md)
and its eight editable [message-and-answer pairs](first_skill.json).

Each example teaches one decision, saves a `.s1m` skill, reloads it, and checks
new inputs. Everything runs locally using the existing NumPy compiler. There
are no API keys, downloads, language model calls, or additional dependencies.

From the repository root, with System 1 installed:

```bash
python examples/support_triage.py
python examples/model_routing.py
python examples/agent_guard.py
python examples/banking_support.py
```

Each command prints every evaluation decision and writes its skill and full
report under `.system1/examples/<example>/`. Use `--output-dir /your/path` to
choose a different parent directory. Reuse a saved skill directly:

```bash
system1 decide "Please correct the invoice address" --model .system1/examples/support_triage/skill.s1m --json
system1 decide "Summarize these meeting notes" --model .system1/examples/model_routing/skill.s1m --json
system1 decide "Inspect the local source files" --model .system1/examples/agent_guard/skill.s1m --json
```

## What each skill learns

| Example | Decision | Teaching cases | Calibration cases | Evaluation cases |
|---|---|---:|---:|---:|
| [Support triage](../support_triage.py) | Billing, product support, account security, or sales | 192 | 112 | 104 |
| [Model routing](../model_routing.py) | Mechanical transform, ordinary chat, or deeper reasoning | 156 | 156 | 96 |
| [Agent guard](../agent_guard.py) | Inspect, change, or restricted operation | 334 | 125 | 132 |
| [Banking support](../banking_support.py) | Card arrival, lost/stolen card, or withdrawal fees | 287 | 125 | 120 |

The current files include six additional routing lessons and 28 operation lessons
for close distinctions. They are marked `quality_round` and are locally authored.
The [contrast comparison](../../benchmarks/quality/contrast_round.md) keeps their
fresh evaluation separate from teaching and records the remaining errors. Support
teaching is unchanged. The later confidence round adds 60 routing calibration
cases marked `2026-09-confidence`; the historical contrast comparison excludes
those rows to preserve its original results. No new APIs or settings are needed.

The banking example uses externally labeled public queries, with all selected
official test rows kept separate. See its [source and license](BANKING77.md).
The [1.0.1 evidence](../../docs/releases/1.0.1.md) reports current calibration and
banking results, including the harder routing cohort that misses the 95% target.

The agent example additionally demonstrates an explicit permission rule: a local
configuration lookup is allowed, a private-key lookup is denied, and the signed
ledger is verified. The classifier's suggestions do not grant permissions. No
operation described by the example dataset is executed.

The model example applies a stated routing policy. It does not make claims about
specific language models' capabilities or call any of those models.

## The teaching step

All four use the same existing API:

```python
skill = SystemOneCompiler(Schema, dimension=2048, regularization=0.1).compile(
    examples,
    augment=False,
    calibration_exemplars=calibration_examples,
)
skill.save("skill.s1m")
```

The short example scripts declare their output labels. The shared
[measurement helper](../_teaching_demo.py) contains the teach/save/load steps and
reporting. To teach a different policy, edit the labeled cases in the matching
JSON file and rerun. See the [teaching guide](../../docs/guides/training_experts.md)
for a complete Python example and CLI dataset formats.

These files are demonstration bundles containing three splits, policy notes,
and case groups. The helper converts each split to the compiler's field mapping;
a whole bundle is not itself a CLI `--dataset` file.

## Current primary results (unchanged in 1.0.2)

| Skill | Accepted locally | Correct among accepted |
|---|---:|---:|
| Support triage | 92/104 | 91/92 |
| Model routing | 94/96 | 94/94 |
| Operation triage | 126/132 | 126/126 |
| Banking support | 118/120 | 115/118 |

The [1.0.2 quality round](../../benchmarks/quality/quality_round/README.md) adds
new authored probes and retains a rejected routing candidate. It does not change
these primary lesson files. Banking and SMS failures on new phrasing are disclosed
separately from these existing test results.

The [current primary report](../../benchmarks/quality/results/release_1_0_1.json)
and [banking report](../../benchmarks/quality/workloads/results.json) retain the
complete measurements. Broader [public-workload evidence](../../benchmarks/quality/workloads/README.md)
adds assistant and SMS tasks, a classical baseline and live Jev/Gemini takeovers.
The tables below retain the original 1.0 run; operation acceptance has since changed.

## Recorded 1.0 run

Measured on Apple M4 Pro with Python 3.13.5. These timings cover fitting and
calibration once labeled examples exist. They exclude authoring/reviewing data,
interpreter startup, imports, and application tool execution.

| Skill | Teach + calibrate | Saved size | Raw correct | Accepted locally | Correct among accepted | Decision median |
|---|---:|---:|---:|---:|---:|---:|
| Support triage | 198.8 ms | 31.9 KiB | 98/104 | 92/104 (88.5%) | 91/92 (98.9%) | 0.504 ms |
| Model routing | 175.4 ms | 24.9 KiB | 95/96 | 94/96 (97.9%) | 94/94 (100%) | 0.509 ms |
| Operation triage | 212.2 ms | 24.9 KiB | 131/132 | 125/132 (94.7%) | 125/125 (100%) | 0.459 ms |

Raw reports: [support](results/support_triage.json),
[routing](results/model_routing.json), [operations](results/agent_guard.json).
They include every prediction and error, confusion matrices, counts, dataset
hashes, environment, and 95% Wilson intervals. Accepted-accuracy intervals are
94.1–99.8%, 96.1–100%, and 97.0–100%, respectively. These descriptive intervals
assume independent cases; shared authorship and vocabulary limit that assumption.
In particular, the support result does not establish 95% population precision.

The helper uses 2048 fixed features, ridge regularization 0.1, examples-only
teaching, alpha 0.05, and strict LAC uncertainty sets. Regularization was selected
using teaching-only cross-validation. The library defaults remain 384 features
and regularization 1.0. Both inference caches and receipts are disabled for these
decision timings. The separate guard's authorization and ledger work is excluded.

**One accepted support answer was wrong.** Review decisions and errors remain in
the reports. The example guard separately permits one explicit configuration
lookup and denies a private-key lookup; classifier labels grant no permission.

## Actual Jev comparison

```bash
python benchmarks/quality/evaluate_release.py
```

The command separately checks current manual teaching and replays
[1,313 actual Jev responses](../../benchmarks/quality/results/jev_observations.json)
for the original 1.0 baseline. The Jev comparison excludes the new locally authored
`quality_round` lessons. It
teaches each baseline skill from Jev's labels, calibrates on separate responses, saves and
reloads, then blocks socket connections while evaluating locally. It checks
identical answers, probabilities, prediction sets, and review behavior after
reload. Each of the three skills reaches the same accepted counts and correctness
shown above. Jev agrees with every evaluation label, but disagrees with three
operation-teaching labels; those differences are preserved.

This is teaching from recorded live responses, separate from automatic stream
promotion. The [live structured-ticket example](../../benchmarks/quality/results/jev_live_cutover.json)
proves that second path: 357 teacher observations, default-gate promotion, then
40/40 correct and accepted local answers with no further teacher calls.

The comparison runs offline by default and is checked in CI. `--refresh-teacher`
collects only missing responses using `TYPESAFE_API_KEY` and makes billable HTTP
calls. See the [release evidence](../../docs/releases/1.0.md) for methods and limits.

## Data separation and development history

The three original skills use explicit AI-authored demonstration cases, not customer
traffic or an independent benchmark. Banking uses public queries with original
intent annotations. Each file states its label policy and contains:

- `teach`: examples used to fit the decision head.
- `calibration`: separate examples, divided between temperature fitting and
  conformal calibration (56/56 support, 78/78 routing, 63/62 operations and banking).
- `evaluate`: examples used after saving and reloading, never fitted or calibrated.

The loader rejects repeated normalized prompts and groups crossing these splits.
These checks catch accidental reuse; they do not prove semantic independence.
Authored read/edit contrast pairs in the operation teaching data share object
families, with family groups kept together. No examples are generated at runtime.

Each of the three original evaluations retains 24 cases as `0.2.2 reference`. Before
expanding teaching, we froze 80 new support cases, 72 routing cases, and 72
operation cases as `stable held-out`. Support and routing met the targets on the
first evaluation. Operation acceptance informed further teaching improvements,
so its 72 cases are now development evidence. After selecting the final teaching
data and settings, we authored a separate 36-case `stable confirmation` cohort:
35/36 were accepted and all 35 accepted answers were correct. That small cohort
remains authored demonstration evidence, not an independent security evaluation.

Recorded targets are at least 95% correctness among accepted answers and 80%
acceptance. Passing point estimates is not a population guarantee. For deployment,
use representative traffic, retain errors and review rates, and collect fresh
evaluation cases whenever previous results have informed an improvement.
