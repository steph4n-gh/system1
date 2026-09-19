# Teach the three primary skills

Each example teaches one decision, saves a `.s1m` skill, reloads it, and checks
new inputs. Everything runs locally using the existing NumPy compiler. There
are no API keys, downloads, language model calls, or additional dependencies.

From the repository root, with System 1 installed:

```bash
python examples/support_triage.py
python examples/model_routing.py
python examples/agent_guard.py
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

| Example | Decision | Teaching cases | Calibration cases | Unseen cases |
|---|---|---:|---:|---:|
| [Support triage](../support_triage.py) | Billing, product support, account security, or sales | 64 | 48 | 24 |
| [Model routing](../model_routing.py) | Mechanical transform, ordinary chat, or deeper reasoning | 54 | 48 | 24 |
| [Agent guard](../agent_guard.py) | Inspect, change, or restricted operation | 54 | 48 | 24 |

The agent example additionally demonstrates an explicit permission rule: a local
configuration lookup is allowed, a private-key lookup is denied, and the signed
ledger is verified. The classifier's suggestions do not grant permissions. No
operation described by the example dataset is executed.

The model example applies a stated routing policy. It does not make claims about
specific language models' capabilities or call any of those models.

## The teaching step

All three use the same existing API:

```python
skill = SystemOneCompiler(Schema, dimension=2048).compile(
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

## Recorded run

Measured on the review machine (Apple M4 Pro, Python 3.13.5). These timings cover
fitting and calibration once the labeled examples exist; they exclude authoring
and reviewing examples, interpreter startup, and imports.

| Skill | Teach + calibrate | Saved size | Starter → taught accuracy | Decision median |
|---|---:|---:|---:|---:|
| Support triage | 158.5 ms | 32.2 KiB | 62.5% → 87.5% (21/24) | 0.500 ms |
| Model routing | 134.8 ms | 24.8 KiB | 54.2% → 91.7% (22/24) | 0.558 ms |
| Operation triage | 127.3 ms | 24.8 KiB | 37.5% → 100% (24/24) | 0.478 ms |

Raw reports: [support](results/support_triage.json),
[routing](results/model_routing.json), [operations](results/agent_guard.json).
They include every prediction and error, per-label confusion matrices, teaching
counts, dataset hashes, save/load times, and environment details.

Both the starter and taught models use 2048 features in this comparison, with
ridge regularization 1.0 for teaching. The library default remains 384. The
saved skill is used for evaluation. Both inference caches and receipt generation
are disabled for decision timings; the separate guard's authorization and ledger
work is not included in that measurement. Settings and labels were fixed before
this first recorded evaluation; the examples were not revised to remove its errors.

**Raw accuracy is not the same as answering without review.** At alpha 0.05 with
strict uncertainty checks, support answered 1/24 cases without review and
operation triage answered 2/24. All three accepted answers were correct.
Routing requested review for all 24, so its accepted accuracy is undefined.
The strict set rule now inverts the calibrated cumulative-probability score;
empty sets request review too. Acceptance remains a limitation of these small
natural-language demonstrations. More examples do not automatically guarantee
a useful acceptance rate.

For observation-driven teaching and successful default-gate takeover on a simpler
structured workload, run [observe_routing.py](../observe_routing.py). Its
[separate report](results/observed_routing.json) uses a rule-based offline teacher
and synthetic tickets, not these natural-language cases.

## How the data is separated

The JSON files contain explicit, AI-authored demonstration cases, not real
customer traffic or independent benchmarks. Each file states its label policy.
The three splits are fixed in the file before teaching:

- `teach` supplies examples to fit the decision head.
- `calibration` supplies separate examples to assess uncertainty. The compiler
  divides these into 24 temperature and 24 conformal calibration cases.
- `evaluate` is used only after saving and reloading the skill.

Each case has a scenario group. The loader rejects groups or normalized prompts
that cross these boundaries, and rejects repeated prompts within a split. These
checks catch accidental reuse, but do not establish semantic independence:
single-author examples share vocabulary, topics, and style across splits.
No templates are expanded and no examples are generated during teaching.

These small, focused demonstrations show the mechanics and measured benefit of
teaching. They do not replace testing on representative user data. Retain the
errors, review rates, and provenance when sharing the results. Once an evaluation
case informs an improvement, treat it as development data and use fresh cases
for the next independent evaluation.
