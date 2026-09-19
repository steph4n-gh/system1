# Teaching a decision skill

System 1 learns a particular decision from labeled examples, saves it as a small
`.s1m` file, and applies it locally. A person, an existing system, or a reasoning
model can supply the examples. An LLM is not required.

The compiler fits small numerical decision heads over fixed text features. It
does not train a language model or acquire general language understanding.
The product loop is **show examples → save the skill → use it → review mistakes**.

## Try one skill

For complete demonstrations with separate teaching, calibration, and evaluation
cases, run [the three primary examples](../../examples/teaching/README.md).
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

The compiler reserves about 25% of the supplied unique prompts for calibration by
default. For choice fields it divides that portion again between temperature
and conformal calibration. Repeated prompts, ignoring whitespace and case, stay
in the same partition. It does not detect paraphrases, related customer threads,
or shared source documents.

For related examples, split those groups yourself before compilation. Pass
`calibration_exemplars=held_out_examples` in Python, or
`--calibration-dataset calibration.json` in the CLI. Keep the final evaluation
examples outside both files. Explicit teaching and calibration prompts must be
disjoint. A Python `calibration_split=0` produces an uncalibrated skill;
strict mode then asks for review.

Check both accuracy and how often the skill can answer without review. Examine
mistakes by label, especially costly mistakes. A small calibration set or a weak
skill can mean every input needs review. There is no guaranteed example count
that makes a skill reliable, and calibration is not proof of correct permissions.

Run the repository's small development comparison with:

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
Saving the revised skill uses the same file format and runtime.

The existing online correction methods remain available for experiments; after
changing a skill, old calibration evidence may no longer describe it. The
simplest reproducible workflow is to retain your examples and recompile.

The broader [expert examples](../../examples/train_expert.py) explore synthetic
bootstrapping, online updates, and composition. They are optional. Teaching one
useful skill does not require any of those mechanisms.
