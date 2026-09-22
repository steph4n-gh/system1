# Teach by doing: organize a sample inbox

File a document in **Finance**, **People** or **Projects**. That action records
the visible text and the folder you chose as an explicit lesson. Review a
candidate, compare it with the approved skill, then adopt it explicitly. The
interface uses the shared `TeachingSession` lifecycle and adds no dependencies.
This workflow is unreleased; use the current source checkout, not the published
1.0.3 package on its own.

## Try the workspace

From a repository checkout with System1 installed ([setup](../../README.md#quickstart)):

```bash
python -m examples.teaching_by_doing
```

Open **http://127.0.0.1:8791/**. The server binds to localhost. All sample documents
are fictional text previews; no real files are opened, moved or uploaded.

1. Select a document under **Show how**. Drag its card to a folder, or click the
   folder. Both actions record a fitting lesson. Include examples of each folder.
2. Use **Calibrate** to label different documents for uncertainty calibration.
   These answers stay separate from fitting and candidate comparison.
3. Click **Review candidate**. Inspect candidate and approved-skill accuracy,
   accepted errors, coverage, review count and regressions. Expand the case
   details to see wrong answers, reviewed cases and regressions.
4. Click **Adopt candidate** only after reviewing a passing report. Until then,
   predictions and downloads continue to use the approved skill. Lesson edits,
   failed candidates and server restarts do not disable that skill.
5. Click **Try sample batch** or **Compare with sample answers**. The 30 filing
   samples are recurring development checks; six unusual requests are separate
   preview-only challenges. Their text cannot become teaching lessons.
6. **Try your own text** accepts a title and excerpt. **Suggest a folder** does
   not save a lesson. To teach from your text, select **Your correct folder** and
   click **Save my lesson**. The selection is never filled from a prediction.
7. Correct a recorded lesson by choosing another folder, then review and adopt a
   new candidate. Repeated choices replace the same lesson instead of inflating
   the count. Your saved texts also appear under **Show how** for correction or
   removal.

For a quick tour, **Add sample session** fills missing choices using the authored
sample policy. It preserves your corrections and labels the source of each
choice. The sample supplies 18 fitting lessons and 42 calibration examples;
System1 splits those into 21 temperature and 21 conformal examples. The separate
30 filing checks never enter fitting or calibration. Three fitting lessons alone
do not establish readiness: assessment also requires separate calibration, and
poor accuracy or low acceptance coverage can prevent adoption.

The demo uses the helper's default gates: at least 80% raw accuracy, at least 50%
acceptance coverage, no accepted errors and no regressions on the declared
checks. A regression is a previously accepted correct answer becoming wrong or
requiring review. Passing is a minimum check, not an automatic adoption or a
claim of readiness outside this sample policy. Different personal policies may
conflict with the authored sample answers.

## What the action teaches

The observation is `title + "\n" + excerpt`. The answer is your selected folder.
Document IDs, data splits and reference answers are never model features.
TF-IDF features are fitted only on demonstrated text, followed by a small
numerical decision head. Compilation uses `augment=False`: it generates no
additional labels and calls no LLM or teacher API.

The sample policy files by the work requested. An invoice for a project belongs
in Finance; a software bug in an invoice screen belongs in Projects. Learned text
features can transfer between examples, but this is not general language
understanding or a guarantee for unfamiliar requests.

The fixed sample batch is reserved for recurring development checks. Its answers
cannot be recorded as lessons in this demo, including through **Try your own
text**. Reusing these checks to choose candidates does not create fresh test
evidence. Use independently collected documents to evaluate a real application.

## Save, export and reuse

Choices and their display metadata persist in
`.system1/teaching-by-doing/actions.json`. The shared helper stores fitting,
calibration and development-check records in `session.json`, the proposed skill
in `candidate.s1m`, its comparison in `assessment.json`, and the approved skill in
`current.s1m`. Adoption checks that the data, candidate, current skill and report
still match. Edits require a new assessment before adoption.

An older `filing.s1m` is preserved as the current skill when its schema matches
this demo exactly. The original file and any historical `report.json` remain
unchanged. Current inbox previews write `preview-report.json`.

Keep this demo's directory separate from CLI/Python teaching sessions. The demo
owns its action metadata; if another writer changes the helper's lessons, reopening
refuses the mismatch instead of silently overwriting them. Use one process per
session directory.

The page downloads the action log, ordinary teaching JSON and the approved skill.
The JSON separates `exemplars` from `calibration_exemplars`, each containing a
`folder` list of `[text, chosen_folder]` pairs. The action log also preserves
whether each choice was manual or supplied by the sample session.

Use the saved skill in another process without the GUI or lesson file:

```python
from system1 import CompiledSystemOneModel, System1Engine

skill = CompiledSystemOneModel.load(".system1/teaching-by-doing/current.s1m")
engine = System1Engine(skill.schema, model=skill, strict_mode=True, use_cache=False)
answer = engine.decide(
    "Supplier invoice\nPlease arrange payment for the attached invoice.",
    record_receipt=False,
)
print(answer.values["folder"], "Needs review:", answer.is_ambiguous)
```

Your application must honor review before acting. `.s1m` carries the learned
features, head and calibration; it does not contain an LLM or this web server.
`--output-dir PATH` starts or resumes a separate demonstration session;
`--port PORT` changes the local port.

## Review a sample session from the command line

Prepare a candidate in an empty directory:

```bash
python -m examples.teaching_by_doing --evaluate --output-dir .system1/filing-evaluation
```

This writes a candidate and its report without activation. Open the same output
directory in the page to review and adopt it:

```bash
python -m examples.teaching_by_doing --output-dir .system1/filing-evaluation
```

For an explicitly requested sample adoption and full preview in another empty
directory, use:

```bash
python -m examples.teaching_by_doing --evaluate --adopt --output-dir .system1/filing-preview
```

Both evaluation commands refuse to overwrite an existing session or evidence.
[documents.json](documents.json) contains the fixed, disjoint inputs and sample
policy. The command describes repeated checks as development feedback.

## Evidence and limits

The separate [real-document workflow test](../../benchmarks/quality/document_workflow/README.md)
uses five BBC news categories and 400 untouched articles. Targeted feedback halves
accepted mistakes from 16 to 8 while sending more articles to review; all candidates
fail the unchanged zero-error development gate. That result does not qualify
this demo's Finance/People/Projects policy or show successful adoption on real data.

### Historical sample evidence

The measurements below describe the original pre-lifecycle demonstration.
[results/report.json](results/report.json) and the evidence archive are preserved
unchanged. They are historical evidence, not a measurement of the new interface,
helper overhead, or repeated candidate selection.

| Measurement | Recorded result |
|---|---:|
| Correct best guesses, 30 new filing documents | 29/30 |
| Accepted filing suggestions correct | 27/27 |
| Ordinary documents sent to review | 3/30 |
| Unusual requests sent to review | 5/6 |
| Teaching and calibration | 12.4 ms |
| Median local decision, all 36 requests | 0.127 ms |
| Saved skill | 15,817 bytes |
| Teacher/API calls during prediction | 0 |

These authored examples are **not a representative real-world benchmark**.
“Leave form interface” was incorrectly guessed as People, but sent to review.
“Kitchen notes,” a recipe, was incorrectly accepted as Projects. Including that
challenge, the virtual inbox files 28 documents: 27 match the intended filing
policy and one should have waited for review. Confidence is not correctness.

Timing uses NumPy on an Apple M4 Pro CPU, with cache and receipts disabled.
Teaching time excludes human filing and saving/reloading; prediction time
excludes startup, HTTP, browser rendering and downstream actions. The skill size
excludes the installed runtime. Hardware and timing details accompany the
[complete report](results/report.json).

The [evidence archive](results/filing-evidence.zip) preserves all actions,
every prediction and review set, the saved skill, source snapshot and SHA-256
manifest. Tests replay every prediction from that saved skill. The demo has no
PDF extraction, OCR, arbitrary folder setup or desktop observation. Those are
separate application concerns, not capabilities established by these results.
