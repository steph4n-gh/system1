# Correct, compare, and adopt a skill

A teaching session keeps the lessons, a working skill, and a proposed revision
together. **Correct a lesson → build a candidate → compare answers → explicitly
adopt.** Editing a lesson or assessing a candidate does not replace the skill
already serving your application.

This additive workflow is available in **System 1 1.1.0 and newer**:

```bash
python -m pip install 'system1>=1.1.0'
```

The first supported workflow is text classification with one
`ChoiceField` with ambiguity escalation enabled (the schema default). A field
that disables escalation is rejected because the session requires review of
uncertain answers. Existing compiler and engine APIs, numerical algorithms, and
`.s1m` format remain unchanged; no additional dependency is required. Existing
hashed skills remain readable. The helper's TF-IDF skills need a 1.1.0-or-newer
reader. This helper uses the existing TF-IDF projector with at most 1,024 features,
ridge regularization
0.1, and strict decisions at `alpha=0.05`. These choices are confined to the helper;
the compiler's existing defaults remain unchanged.

For the experimental visual version, use a repository checkout and run
`python -m examples.teaching_by_doing`. Use
**Review candidate** to check a proposed document-filing skill and **Adopt
candidate** to activate a passing revision. The
[workspace guide](../../examples/teaching_by_doing/README.md) explains its authored
sample data and retained historical evidence.

## Start with the decision and its lessons

For this example, payment questions belong to `billing` and broken software
belongs to `support`. An example pairs the message with the answer you want.

Keep three groups separate:

| Group | Session split | Purpose |
|---|---|---|
| Teach the decision | `teach` | Fit the numerical head |
| Recognize uncertainty | `calibrate` | Set review behavior on separate examples |
| Compare old and new answers | `evaluate` | Measure a candidate on examples outside teaching and calibration |

A session stores labels and source notes; it does not obtain correct answers for
you. Related customer threads and near-duplicate messages should stay in one
group. Once you use an evaluation result to decide a correction, that check is
regression evidence. Keep fresh, independently labeled cases for later confirmation.

Create `lessons.json` using this layout, replacing the short illustrative lists
with representative examples of **both** choices in each group:

```json
{
  "teach": [
    ["Refund my payment", "billing"],
    ["The application crashes", "support"]
  ],
  "calibrate": [
    ["Please correct my invoice", "billing"],
    ["The software fails during startup", "support"]
  ],
  "evaluate": [
    ["Why was I billed twice?", "billing"],
    ["The checkout screen freezes", "support"]
  ]
}
```

These six rows demonstrate the file format, not a reliably taught skill. The
report may withhold adoption because of mistakes or insufficient acceptance.
There is no universal number of examples that makes a task reliable.

## Use the Python workflow

```python
import json
from pathlib import Path
from system1 import ChoiceField, DecisionSchema, TeachingSession

class SupportRoute(DecisionSchema):
    team = ChoiceField(options=["billing", "support"])

session = TeachingSession(".system1/support-session", SupportRoute)
lessons = json.loads(Path("lessons.json").read_text())
session.record_many([
    {"input": text, "label": label, "split": split, "source": "human"}
    for split, pairs in lessons.items()
    for text, label in pairs
])
report = session.assess()
print(json.dumps(report, indent=2))
```

`assess()` compiles, saves, and reloads a separate candidate. It compares the
candidate and the current adopted skill, if one exists, on the same evaluation
rows using strict review. All three splits are required for assessment.
An initial session can be assessed before it has an adopted skill. Assessment
requires teaching examples for every choice, at least two calibration examples,
and evaluation examples. These are input requirements, not a useful evidence
budget: tiny calibration sets will usually withhold decisions.

Reopen the same session with `TeachingSession(".system1/support-session")`.
The retained records are available through `session.records`; `session.report`
and `session.snapshot()` expose the assessment and current state. The session
also exposes `current_path` and `candidate_path` for its saved artifacts.
`snapshot()` distinguishes an adopted candidate from a stale pending assessment.
Records and artifacts are ordinary local files: keep this a single-writer session,
not a shared multi-user database. Inputs are limited to 8,192 characters.

## Read the before-and-after report

Three questions matter:

- **Does the raw guess choose the right label?** `raw_accuracy` counts all
  evaluation rows, including reviewed guesses. Weak raw accuracy points to
  incomplete or inconsistent lessons, an unclear task, or inadequate features.
- **How much work can proceed without review?** `coverage` is the accepted
  fraction. `accepted_errors` counts wrong guesses that would be acted on.
  Accurate guesses with little acceptance call for checking calibration and
  representative evidence, not silently ignoring review.
- **Did the correction break earlier useful behavior?** `regressions` counts
  cases that the adopted skill previously answered correctly without review but
  the candidate now gets wrong **or sends to review**. `raw_regressions` separately
  counts previously correct raw guesses that became wrong.

The `candidate` and `incumbent` summaries also report counts, review rate, accepted
correct answers, and raw correct answers. `cases` lists each checked input and
expected label, both predictions, review decisions, prediction sets and regression
flags, so you can inspect the actual changed answers. `diagnostics` explains
observed patterns; `reasons` lists failed checks. These are descriptive measurements, not proof of
what caused a failure, out-of-distribution detection, or a confidence guarantee.
For a skill controlling a workflow or game, check the actual completed task too:
correct answers on a fixed list do not establish successful execution.

The default demonstration policy requires raw accuracy of at least 0.8, coverage
of at least 0.5, no accepted errors, and no regressions. You can state your own
policy explicitly:

```python
report = session.assess(
    min_accuracy=0.8,
    min_coverage=0.5,
    max_accepted_errors=0,
    max_regressions=0,
)
```

Choose thresholds for your task before reviewing candidate outcomes. Passing
these finite checks is not certification of production accuracy. Do not lower a
threshold merely to make an unsuccessful candidate pass.

## Correct one lesson

Suppose the skill suggests `billing` for **“The payment page crashes.”** Under our
rule the answer is `support` because the software is broken:

```python
session.record("The payment page crashes", "support", source="human")
report = session.assess()
print(json.dumps(report, indent=2))
```

The default split is `teach`. Recording the same normalized input in the same
split replaces its earlier label; it does not add contradictory copies. An input
already assigned to another split is rejected, so a correction cannot quietly
become its own evaluation evidence. `session.remove(text)` explicitly removes a
record. If moving a known case into teaching, supply different evaluation cases
and disclose that the original case is now a lesson.

A failed candidate leaves the adopted skill available. Keep representative older
lessons while adding a few different examples of the confusing distinction, then
check both teams again. One corrected sentence does not establish generalization.

## Adopt deliberately

After reviewing a passing report, activate that exact candidate:

```python
session.adopt()
answer = session.predict("Please send my payment receipt")
print(answer)
```

`adopt()` requires a passing assessment with unchanged lessons, candidate, and
current skill. Editing data or replacing an artifact makes the earlier approval
stale: assess again. `session.engine` and `session.predict()` use the adopted skill,
never an unapproved candidate. Before the first adoption `session.engine` is
`None`, and `session.predict()` asks you to assess and adopt first. Predictions
can still need review after adoption.

This explicit adoption is separate from the existing
[automatic teacher-promotion policy](../typesafe.md). It does not change that
adapter or turn a predicted label into permission to execute an action.

## The same workflow from the CLI

Export the schema once:

```python
Path("schema.json").write_text(json.dumps(SupportRoute().to_dict()))
```

Then run:

```bash
system1 teach .system1/support-session --schema schema.json
system1 teach .system1/support-session --dataset lessons.json
system1 teach .system1/support-session --record "The payment page crashes" --label support
system1 teach .system1/support-session --assess --json
```

The dataset format is the `teach` / `calibrate` / `evaluate` object shown above,
with arrays of `[text, label]` pairs. It is different from the lower-level
compiler's field-mapping format. `--split` selects a split for one record and
defaults to `teach`; `--remove "text"` removes a record. With no action flag,
`system1 teach .system1/support-session --json` reports the session state.
The CLI returns exit code 0 for success and 2 when an assessment fails its checks
(with the report retained). Handled operational or data errors return 1;
argument syntax errors also return 2.

After reviewing a passing assessment:

```bash
system1 teach .system1/support-session --adopt --json
```

For existing lower-level compilation, numerical telemetry, multiple fields, or
other output types, continue using the [teaching API guide](training_experts.md).
The session helper deliberately starts with one text-choice task.

For a measured real-document example, see the
[BBC correction-workflow experiment](../../benchmarks/quality/document_workflow/README.md).
Targeted feedback reduced accepted mistakes on its separate holdout, with more
reviews; every candidate still failed the default zero-error adoption checks.
That result illustrates why a quality improvement and permission to adopt are
separate decisions.
