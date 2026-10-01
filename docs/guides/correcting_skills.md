# Correct, compare, and adopt a skill

A teaching session keeps the lessons, a working skill, and a proposed revision
together. **Correct a lesson → build a candidate → compare answers → explicitly
adopt.** Editing a lesson or assessing a candidate does not replace the skill
already serving your application.

This additive workflow is available in **System 1 1.1.0 and newer**:

```bash
python -m pip install 'system1>=1.2.0'
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


## Independent qualification

Version 1.2.0 adds a separate one-shot qualification step. Choose the workload
policy before inspecting the cohort, keep this data outside fitting, calibration,
and recurring evaluation, and collect independently reviewed representative rows.

```python
report = session.assess(require_qualification=True)
# Only a passing development assessment can be qualified.
fresh = json.loads(Path("fresh-cases.json").read_text())
qualification = session.qualify(
    fresh, source="Reserved independently reviewed requests; collection dates and method",
    min_accepted_accuracy=.95, min_coverage=.8, confidence=.95,
)
print(qualification["candidate"], qualification["reasons"])
# Review the passing evidence before explicitly calling session.adopt().
```

Each fresh row is an object with `input`, `label`, and optional `group`. Related
records can have a nonempty `group` in all session splits, but a group cannot cross
splits. Qualification requires one row per supplied independent group. Omitting
groups is a declaration that individual rows are independent; the software cannot
verify that, detect all paraphrases, or establish label correctness.

Both accepted-correctness and coverage lower bounds must pass. The exact one-sided
binomial bounds share the specified confidence error budget across all attempts
in this session. The first request fixes the confidence target. Attempt `k`
allocates `(1 - confidence) / (2 * k * (k + 1))` to each bound, so repeated
qualification cannot reset the error budget. A few perfect answers
are insufficient; zero acceptance leaves accepted correctness undefined. These
bounds assume independent representative units and a fixed candidate. They are
not a promise under arbitrary changes in traffic. Development bounds are descriptive
only because checks may be correlated and have influenced repeated revisions.

The request, source, policy, and cohort are frozen before the first prediction.
A failed or interrupted attempt consumes its rows/groups across this session,
including later candidates. They cannot be imported as lessons or replaced with
another policy. Other copies of a session and semantic overlap cannot be controlled
by software; reserve genuinely fresh data rather than copying inspected tests.
The confidence scope does not extend to separate session directories. Adoption
rechecks the exact frozen evidence when qualification is required.
Default `assess()` retains the existing development-only approval policy.

```bash
system1 teach .system1/support-session --assess --require-qualification
system1 teach .system1/support-session --qualification fresh-cases.json --source "Independent collection"
system1 teach .system1/support-session --adopt
```

## Approved revisions and recovery

```python
print(session.history)
revision = session.history[0]["revision"]
session.rollback(revision)  # Explicitly restore approved bytes, keeping all lessons.
print(session.decision_details("Please refund my payment"))
```

`history` retains artifact digests, approval dates, and whether qualification
was part of the approval. Recovery validates the artifact, schema, approval policy
and required qualification. A missing or altered archive is refused. Rollback
invalidates the pending assessment even if the restored skill was its original
incumbent. Reassess before another adoption; restore is not automatic promotion.
An existing 1.1 approved skill is retained during the first 1.2 adoption with its
legacy status when its earlier assessment is unavailable.

```bash
system1 teach .system1/support-session --history
system1 teach .system1/support-session --rollback FULL_APPROVED_REVISION_DIGEST
system1 teach .system1/support-session --predict "Please refund my payment" --json
```

## Choose the fitting method

Ridge remains the default. A new session can opt into the existing logistic solver:

```python
session = TeachingSession(".system1/logistic-session", SupportRoute, choice_solver="logistic")
```

Install `system1[teaching]` for its SciPy fitting dependency. The chosen setting is
retained, and reopening with a different supplied setting is refused. Default
teaching, qualification, rollback, and inference on saved logistic heads need no
SciPy. Old version-one session state reopens without rewriting; missing solver
settings mean ridge. Reassess old active candidate reports before adoption. Use
1.2.0 for directories that now contain the new session contracts.
