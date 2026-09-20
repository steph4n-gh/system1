# Teach one decision from Paperclips

The real browser game exposed a useful product issue: asking the same question
with different numeric states was treated as a repeated teaching example.
Explicit calibration rejected it, and automatic splitting could reserve no
calibration examples. The compiler now groups the prompt **and telemetry**.
An exact repeated input still cannot cross the teaching/calibration boundary.

This small example uses the existing compiler, numeric telemetry and `.s1m`
format. It adds no runtime dependency and is separate from the handwritten
[three-phase game policy](paperclips_speedrun.py).

## The lesson

The question is always “Should we restock wire?” A label is the desired answer:

```json
{"wire": 0, "label": "buy_wire"}
{"wire": 539, "label": "hold"}
```

Those two supplies were observed in the actual browser game. The editable
[lesson file](../teaching/paperclips_wire.json) retains their full visible states
and adds 22 authored teaching examples. The preference is to restock below
250 inches and otherwise hold. This is an illustrative preference, not a proven
optimal game strategy. A direct rule is simpler for this exact threshold; the
example demonstrates teaching a numeric preference, not an advantage over that rule.

Wire is divided by 1,000 before being supplied as telemetry. A constant reference
feature preserves magnitude when the existing projector normalizes the vector.
The model learns from the labels; the numeric preparation does not return the answer.

## Teach, save, check

From the repository root:

```bash
python examples/gaming/teach_paperclips_wire.py
```

The script uses 24 teaching examples and 48 separate calibration examples, saves
`.system1/paperclips/wire.s1m`, reloads it, then checks 16 further wire supplies.
Every calibration and evaluation case is authored, not an independent live run.
Related observations from one run should stay together when assessing game performance.

The [recorded report](../teaching/results/paperclips_wire.json) retains every result:

| Measurement | Recorded result |
|---|---:|
| Teach and calibrate | 38.9 ms |
| Saved skill | 2,455 bytes |
| Correct suggestions | 14/16 |
| Accepted without review | 15/16 |
| Correct among accepted | 14/15 |
| Median decision time | 0.237 ms |

Measured locally on Apple M4 Pro, Python 3.13.5, NumPy CPU. Decision time includes
the example's affordability check but excludes imports, browser actions and API
calls. Caches and receipts were disabled. No LLM or teacher API was called.

**Two boundary mistakes remain.** At 250 inches the skill suggested buying without
review, contrary to the stated preference. At 252 inches it also preferred buying,
but requested review. This candidate is not qualified for unattended control;
a prediction set is not proof that the answer is correct. These cases are now
development evidence. Any revision informed by them needs fresh evaluation cases.

## Reuse the skill

```python
from system1 import System1Engine
from system1.compiler import CompiledSystemOneModel
from examples.gaming.teach_paperclips_wire import suggest

skill = CompiledSystemOneModel.load(".system1/paperclips/wire.s1m")
engine = System1Engine(skill.schema, model=skill, strict_mode=True, use_cache=False)
print(suggest(engine, wire=0, funds=100, wire_cost=20))
```

The application blocks unaffordable purchases and returns `review` when the skill
is ambiguous. An actual browser adapter must also recheck that the current Wire
button is enabled before clicking. The example script does not control a browser.

In the [supervised live check](../teaching/results/paperclips_live_session.json),
the saved skill suggested `buy_wire` at zero wire. The operator reviewed the
suggestion and clicked the real game button. The supply rose to 1,499 inches
after one clip was made. On that observed supply, the same saved skill suggested
`hold`. This demonstrates portable local advice connected to a real action;
it is not autonomous gameplay, automatic promotion, a speedrun, or a completed game.
