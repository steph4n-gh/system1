# Teach your first skill

Teaching System 1 means **showing it examples of a decision you already know how
to make**. You do not need an LLM, an API key or a GPU. We will teach one small
job: choose the team that should receive a customer message.

Our rule is simple: payment questions go to `billing`; broken software goes to
`support`. You choose that rule. System 1 learns to apply it to other messages.

If you prefer a visual first lesson, try the
[document teaching workspace](../../examples/teaching_by_doing/README.md).
You choose folders for fictional documents, and the page records those actions
as examples. It also lets you correct a choice and download the learned skill.
The walkthrough below teaches the same lifecycle using an editable JSON file.

## What is a label?

A label is just **the answer you want**. These are three teaching examples:

| Message System 1 will see | Answer you want |
|---|---|
| Refund my payment | `billing` |
| Correct the invoice | `billing` |
| Fix a software crash | `support` |

You are showing complete messages and their correct destination. You do not need
to highlight keywords, write a prompt for an LLM or describe an algorithm.
The answer names must match the two choices exactly: `billing` or `support`.

## Run the first lesson

From a repository checkout with System 1 installed, run:

```bash
python examples/teach_skill.py
```

If you have not installed the project yet, use the
[README setup commands](../../README.md#quickstart), then return here.

The script reads [first_skill.json](../../examples/teaching/first_skill.json),
teaches from its eight examples, saves a skill and tries a different message.
The file uses this simple layout:

```json
{
  "team": [
    ["Refund my payment", "billing"],
    ["Correct the invoice", "billing"],
    ["Subscription charged twice", "billing"],
    ["Payment receipt requested", "billing"],
    ["Debug this exception", "support"],
    ["Fix a software crash", "support"],
    ["API request fails", "support"],
    ["Investigate error logs", "support"]
  ]
}
```

`team` names the decision. Each pair contains a message followed by the answer
you want. That is the whole teaching format for this example.

The first run prints:

```text
Suggested team: billing
Needs review: True
```

`billing` is its best guess. **Needs review means a person should still decide.**
Eight examples show the process, but do not provide enough separate evidence for
reliable automatic routing. Saving a skill does not certify that it is ready.

The saved `.s1m` file contains the reusable skill. System 1 applies learned
patterns to new messages; it does not need the JSON file at decision time.

## Teach your own messages

Copy the lesson file so you can edit it. The first run creates `.system1/`, which
is ignored by Git:

```bash
cp examples/teaching/first_skill.json .system1/my-lessons.json
```

Open `.system1/my-lessons.json` in a text editor. Replace or add messages that
people actually send to your team, keeping the same two possible answers. Then:

```bash
python examples/teach_skill.py --lessons .system1/my-lessons.json --prompt "Please send my payment receipt"
```

Each run rereads your lessons and saves the newly taught skill. Editing the JSON
alone does not change an already saved skill. Use `--output .system1/candidate.s1m`
to save a candidate separately while keeping the previous skill.

For a different job, define different output choices as described in the
[teaching API guide](training_experts.md#teach-your-data).

## How do I know it is ready?

Keep some real messages out of the lessons. Decide their correct teams yourself,
then compare System 1's answers. Track two separate things:

- **How often it is right when it answers without review.** Count confident mistakes too.
- **How often it needs review.** A skill that asks about every message is not yet doing the job automatically.

The complete workflow uses examples in three ways:

| In plain language | Name used in the code | Purpose |
|---|---|---|
| Show it how to decide | Teaching | Learn the decision |
| Help it recognize uncertainty | Calibration | Decide when to ask for review |
| Give it a final check | Evaluation | Measure behavior on messages kept out of the first two groups |

The starter script handles a small calibration split automatically. Its tiny
file leaves too little evidence, so review is expected. For a fuller working
example with separate groups and a report, run:

```bash
python examples/support_triage.py
```

That example includes four department choices and shows both correct decisions
and mistakes. Its larger file is a demonstration, not a substitute for your own
messages. There is no magic number of examples: start with the task's common
cases and confusing distinctions, then use the separate checks to decide what
is missing. Never teach from your final check and still call that same check new.

If many answers need review, gather more varied lessons and separate checking
examples. If confident answers are wrong, revisit the rule and the lessons before
allowing automatic decisions. Reteaching changes the skill, so check it again.

## Reuse the saved skill

After the first run, the saved file can answer new messages without reading the
lesson file or calling a teacher:

```python
from system1 import CompiledSystemOneModel, System1Engine

skill = CompiledSystemOneModel.load(".system1/support-route.s1m")
engine = System1Engine(skill.schema, model=skill, strict_mode=True)
answer = engine.decide("Please refund this payment", record_receipt=False)
print(answer.values["team"])
print("Needs review:", answer.is_ambiguous)
```

The starter still needs review. Saving and reopening a skill preserves that
behavior; it does not approve the answer. Your application chooses what to do
when review is requested. Keep the earlier file while checking a revised
candidate, then choose which saved skill your application loads.

## What does correcting a mistake mean?

Suppose a new message says **“The payment page crashes when I open it.”** and
System 1 suggests `billing`. Under our rule the right answer is `support`: the
problem is broken software, even though it mentions payment.

Add this pair to your lessons, remembering the comma between JSON entries:

```json
["The payment page crashes when I open it.", "support"]
```

If that exact message is already present with the wrong answer, **replace the
wrong answer**. Do not leave two conflicting lessons. Add a few different real
examples of the same distinction, rather than copying the same sentence many
times. Keep ordinary billing and ordinary support messages too.

Rerun the teaching command. Then try **different** messages, such as “The checkout
screen freezes” and “Why was I billed twice?” Showing that the corrected sentence
now works is useful, but it does not show that the skill handles new cases.
A correction can also affect other decisions, so check both teams again.

## Can someone else supply the lessons?

Yes. A colleague, reviewed historical decisions, a rule in your existing software,
a conventional classifier, Jev, Gemini or another API can provide the same
message-and-answer pairs. JSON is simply a way to store them. The
[observation workflow](../typesafe.md) collects answers from a running teacher
and keeps that teacher responsible until a candidate qualifies for local takeover.

System 1 remains a small local decision skill regardless of who taught it.
