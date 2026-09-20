# Teaching a decision skill

If you are new to labeling examples or correcting a skill, start with
[Teach your first skill](first_skill.md). It explains those terms using a copyable
message-and-answer file. This page covers the underlying API and data contracts.

System 1 learns a particular decision from labeled examples, saves it as a small
`.s1m` file, and applies it locally. A person, an existing system, or a reasoning
model can supply the examples. An LLM is not required.

The compiler fits small numerical decision heads over fixed text features. It
does not train a language model or acquire general language understanding.
The product loop is **teach or observe → validate → run locally → save and reuse**. Review mistakes and revalidate when the skill changes.

## Try one skill

For complete demonstrations with separate teaching, calibration, and evaluation
cases, run [the four teaching examples](../../examples/teaching/README.md).
The smaller example below introduces the API with just eight labeled cases.

From the repository root:

```bash
python examples/teach_skill.py
system1 decide "Please refund this payment" --model .system1/support-route.s1m --json
```

The [example](../../examples/teach_skill.py) teaches eight billing/support examples,
saves the skill, reloads it, and evaluates a new sentence. It deliberately reports
that review is needed: eight examples demonstrate the API, not reliable coverage
of a support workload.

The essential Python calls are:

```python
from system1 import ChoiceField, DecisionSchema, System1Engine
from system1.compiler import CompiledSystemOneModel, SystemOneCompiler

class SupportRoute(DecisionSchema):
    team = ChoiceField(options=["billing", "support"])

examples = {"team": [
    ("Refund my payment", "billing"),
    ("Correct the invoice", "billing"),
    ("Subscription charged twice", "billing"),
    ("Payment receipt requested", "billing"),
    ("Debug this exception", "support"),
    ("Fix a software crash", "support"),
    ("API request fails", "support"),
    ("Investigate error logs", "support"),
]}

skill = SystemOneCompiler(SupportRoute).compile(examples, augment=False)
skill.save("support-route.s1m")

loaded = CompiledSystemOneModel.load("support-route.s1m")
engine = System1Engine(loaded.schema, model=loaded, strict_mode=True)
decision = engine.decide("Please refund this payment")
print(decision.values)
print("Needs review:", decision.is_ambiguous)
```

`augment=False` means only the supplied examples are used. Invalid labels,
unknown fields, and missing classes are errors. Python's existing
`compile(...)` default still augments with schema-derived templates for backward
compatibility; explicitly disable augmentation when teaching your own skill.

With `strict_mode=True`, the engine uses held-out calibration evidence to form
prediction sets. Always inspect `decision.is_ambiguous`: a best-guess value is
still returned when the skill needs review. Your application chooses how to ask
a person or call System 2.

## Choose text features for a taught skill

The default hashed features remain useful without fitting a vocabulary. For a
text classification task with teaching examples, you can instead use the optional
`TfidfProjector`. It learns which words and adjacent word pairs to retain, weighs
common words less, and normalizes each document. It is classical text processing,
not a language model. Both teaching and inference use only NumPy.

```python
from system1 import TfidfProjector

# Fit the vocabulary ONLY on the examples reserved for teaching.
projector = TfidfProjector.fit([text for text, label in examples["team"]])
calibration = {"team": [
    ("Please correct my invoice total", "billing"),
    ("The application crashes during startup", "support"),
    # Supply enough separate, representative examples for useful calibration.
]}
skill = SystemOneCompiler(SupportRoute, projector=projector).compile(
    examples, augment=False, calibration_exemplars=calibration,
)
skill.save("support-route.s1m")
loaded = CompiledSystemOneModel.load("support-route.s1m")
```

The default vocabulary cap is 2,048 terms. Text is lowercased and limited to its
first 8,192 characters; tokens use Unicode word characters with at least two
characters. Terms longer than 128 characters are ignored. Unseen words are
ignored, and an input with no known features requires review. That check alone
does not detect every out-of-scope input containing familiar words. TF-IDF does
not support recency weighting.

Vocabulary and IDF weights are frozen and embedded in `.s1m`. Refit and recompile
when teaching new vocabulary; online head corrections do not expand it. Keep
vocabulary fitting, head fitting, calibration, and evaluation appropriately
separated. In particular, fitting vocabulary before asking the compiler to
split those same rows internally would expose calibration text to vocabulary
selection; use explicit separate calibration as above. Old readers continue to
read old skills but cannot load the new TF-IDF feature type.

See the [email comparison](../../examples/inbox_zero/README.md) for measured gains,
remaining accepted errors, and the distinction between authored and real-world
evidence. The tiny snippet above only demonstrates the API.

## Teach your data

Define a narrow task and consistent labels. Include ordinary cases, confusing
near-matches, and examples of each label. For instance, distinguish an invoice
refund from a request for a new product quote using your own routing rules.

The CLI accepts a JSON field mapping:

```json
{
  "team": [
    ["Refund my payment", "billing"],
    ["Investigate this software crash", "support"]
  ]
}
```

It also accepts records of the form
`{"prompt": "...", "labels": {"team": "billing"}}`, with optional `telemetry`.
The two rows above illustrate the format; use a representative collection for
an actual skill. Export the schema with
`Path("schema.json").write_text(json.dumps(SupportRoute().to_dict()))`
after importing `json` and `Path`.

```bash
system1 compile --schema schema.json --dataset examples.json --output support-route.s1m --json
system1 decide "Please refund this payment" --model support-route.s1m --json
```

With `--dataset`, the CLI uses only those examples by default. Missing files and
malformed examples fail instead of falling back to invented data. `--augment`
explicitly opts into synthetic templates; compilation without a dataset remains
a synthetic demo.

## Check before relying on it

Keep three uses of examples separate:

1. **Teach:** examples used to fit the skill.
2. **Calibrate:** separate examples used to assess uncertainty.
3. **Evaluate:** unseen examples used to measure the finished skill.

The compiler reserves about 25% of the supplied unique inputs for calibration by
default. For categorical and Boolean fields it divides that portion again between temperature
and conformal calibration; MultiChoice also separates temperature fitting from
its joint assignment score. An input is the normalized prompt plus its optional
numeric telemetry. Repeated inputs, ignoring prompt whitespace and case and
mapping key order, stay in the same partition. A repeated question with a different
numeric state can be a different example. This does not establish independence:
the compiler does not detect paraphrases, related customer threads, shared source
documents, or adjacent observations from one game run.

For related examples, split those groups yourself before compilation. Pass
`calibration_exemplars=held_out_examples` in Python, or
`--calibration-dataset calibration.json` in the CLI. Keep the final evaluation
examples outside both files. Explicit teaching and calibration inputs must be
disjoint. A Python `calibration_split=0` produces an uncalibrated skill;
strict mode then asks for review.

For a concrete numeric example, run
`python examples/gaming/teach_paperclips_wire.py` from the repository root.
Its [lesson file](../../examples/teaching/paperclips_wire.json) teaches the same
question with different wire supplies; its [walkthrough](../../examples/gaming/PAPERCLIPS_TEACHING.md)
shows the saved skill, a supervised live purchase, and the remaining boundary errors.

Check both accuracy and how often the skill can answer without review. Examine
mistakes by label, especially costly mistakes. A small calibration set or a weak
skill can mean every input needs review. There is no guaranteed example count
that makes a skill reliable, and calibration is not proof of correct permissions.

For current measured behavior, use the [public-workload evidence](../../benchmarks/quality/workloads/README.md).
The [teacher adapter guide](../typesafe.md) covers observing Jev or another
callback, including the completed Gemini demonstration. A bundle such as
`examples/teaching/support_triage.json` contains `teach`, `calibration` and
`evaluate` splits; its example helper converts these into compiler field mappings.
Pass a field mapping or record list, not the whole bundle, to CLI `--dataset`.

To reproduce the smaller historical development comparison:

```bash
python benchmarks/quality/evaluate_teaching.py
```

See [results and limitations](../../benchmarks/quality/README.md#teaching-comparison).
The default feature dimension remains 384. The comparison also measures 2048;
larger features can reduce collisions but do not supply language understanding.
Use workload evidence before changing this setting.

## Improve the same skill

Review a mistake, correct its label, add representative examples of that case,
and compile again. Recheck with separate calibration and evaluation examples.
Saving the revised skill uses the same runtime. Version 1.0 writes format v2
and reads older v1 skills; older runtimes must be upgraded before reading v2.

Online correction methods invalidate the changed heads' calibration in memory
and in saved artifacts. Strict mode then requires review until recalibration. The
simplest reproducible workflow is to retain your examples and recompile.

The broader [expert examples](../../examples/train_expert.py) explore synthetic
bootstrapping, online updates, and composition. They are optional. Teaching one
useful skill does not require any of those mechanisms.
