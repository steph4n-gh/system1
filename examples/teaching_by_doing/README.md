# Teach by doing: organize a sample inbox

File a document in **Finance**, **People** or **Projects**. That action records
the document's visible text and the folder you chose: an ordinary labelled
example, without editing JSON. Teach from those choices, then try a different
batch. This is an experimental teaching interface on the existing System1 API;
it adds no core abstractions or dependencies.

## Try the workspace

From a repository checkout with System1 installed ([setup](../../README.md#quickstart)):

```bash
python -m examples.teaching_by_doing
```

Open **http://127.0.0.1:8791/**. The server binds to localhost. All documents are
fictional text previews; no real files are opened, moved or uploaded.

1. Select a document under **Show how**. Drag its card to a folder, or click the
   folder. Both actions record the same lesson. Repeat for different kinds of
   documents, including at least one example of each folder.
2. Use **Check examples** to file separate documents. These answers calibrate
   when the skill should ask for help; they are not an independent test score.
3. Click **Teach from these choices**, then **Try fresh batch**. Accepted
   suggestions appear in virtual folders; ambiguous ones appear in **Review**.
4. Click **Compare with sample answers** to reveal every expected answer,
   including failures. **Try your own text** also accepts a title and excerpt;
   this preview does not save the text or turn predictions into lessons.
5. To correct a recorded example, select it and choose a different folder.
   Teach again. Refiling replaces that example, rather than counting duplicates.

For a quick tour, **Add sample session** fills missing choices using the authored
sample policy. Existing choices are preserved. The page identifies how many
choices came from you and how many came from the sample session. This is an
explicit shortcut, not a claim that you demonstrated those actions.

Three demonstrations can create a skill, but without separate checks all 36 new
documents go to review. More examples and representative checks matter. The full
sample supplies 18 fitting examples and 42 checks; System1 splits the checks into
21 temperature and 21 conformal examples.

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

The fresh batch is reserved for evaluation. Its answers cannot be recorded as
lessons in this demo. Correct earlier demonstrations to improve the skill; use
new, independently collected documents to evaluate a real application. The
fixed sample score becomes development feedback once you have inspected it.

## Save, export and reuse

Choices persist in `.system1/teaching-by-doing/actions.json`. Teaching creates
`filing.s1m` in the same directory. Changing or removing a choice disables the old
skill until you teach again, including after restarting the server.

The page downloads the action log, ordinary teaching JSON and the saved skill.
The JSON separates `exemplars` from `calibration_exemplars`, each containing a
`folder` list of `[text, chosen_folder]` pairs. The action log also preserves
whether each choice was manual or supplied by the sample session.

Use the saved skill in another process without the GUI or lesson file:

```python
from system1 import CompiledSystemOneModel, System1Engine

skill = CompiledSystemOneModel.load(".system1/teaching-by-doing/filing.s1m")
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

## Measured sample session

Reproduce in a fresh directory:

```bash
python -m examples.teaching_by_doing --evaluate --output-dir .system1/filing-evaluation
```

The command refuses to overwrite existing demonstrations. It explicitly loads
the authored sample session, teaches, saves/reloads, then evaluates 30 new
documents and six unusual requests. [documents.json](documents.json) contains
the fixed, disjoint inputs, their purposes and the sample policy.

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
