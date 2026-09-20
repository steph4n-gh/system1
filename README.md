<p align="center">
  <img src="https://raw.githubusercontent.com/steph4n-gh/system1/main/assets/system1-logo.jpg" alt="System 1 logo" width="128" />
</p>

# System 1: Teach a decision. Run it locally.

**System 1 is not an LLM.** Show it examples of a repeatable decision—such as
which team should receive a message—then check the skill and reuse it locally.
It uses small numerical decision heads: no language-model weights to download,
no token generation, and no GPU required. Optional **MLX support** runs
decision-head matrix operations on Apple Silicon.

**Choose a decision → show examples → check answers → save and reuse → improve.**

[![CI](https://github.com/steph4n-gh/system1/actions/workflows/ci.yml/badge.svg)](https://github.com/steph4n-gh/system1/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/system1.svg)](https://pypi.org/project/system1/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://github.com/steph4n-gh/system1/blob/main/pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](https://github.com/steph4n-gh/system1/blob/main/LICENSE)

[Performance](#why-it-is-fast) · [Start teaching](#quickstart) · [Automatic teachers](#observe-a-teacher-then-take-over) · [Evidence](#benchmarks) · [Integrations](#integrations) · [Documentation](https://github.com/steph4n-gh/system1/blob/main/docs/README.md)

## Why it is fast

**Small skills make fast local decisions.** After teaching, System 1 applies a
compact numerical decision head. Each request avoids a teacher API round trip
and language-model token generation. The saved skill needs no language-model
weights or model server.

| Benefit | Measured on our three public-data workloads |
|---|---|
| **Sub-millisecond decisions** | **0.34–0.44 ms** median per local decision |
| **Small, portable skills** | **20.4–47.0 KiB** saved `.s1m` files |
| **Quick teaching** | **0.23–1.77 seconds** to teach and calibrate once labeled examples exist |
| **No ongoing teacher calls** | **Zero teacher API calls** during local evaluation |

These are NumPy CPU measurements on an Apple M4 Pro, excluding startup, receipt
signing and downstream work. Skill-file sizes exclude the installed runtime and
dependencies. Optional MLX supports Apple Silicon acceleration; its speed benefit
depends on the workload. See the [reproducible results and quality limits](https://github.com/steph4n-gh/system1/blob/main/benchmarks/quality/workloads/README.md).

The payoff is useful for repeated routing, classification and structured choices.
Quality depends on the taught task; uncertain answers still need review.

## Quickstart

Use Python 3.11 or newer. To follow the editable lesson, start from a repository checkout:

```bash
git clone https://github.com/steph4n-gh/system1.git
cd system1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

### 1. Choose one decision

We will choose between `billing` and `support`. Payment questions go to billing;
broken software goes to support. You decide the rule.

### 2. Show it examples

An example is an input paired with **the answer you want**. That answer is what
“label” means:

| Message | Answer you want |
|---|---|
| Refund my payment | `billing` |
| Correct the invoice | `billing` |
| Fix a software crash | `support` |

The [editable lesson file](https://github.com/steph4n-gh/system1/blob/main/examples/teaching/first_skill.json)
contains eight such pairs. Run it:

```bash
python examples/teach_skill.py
```

This teaches a small skill, saves `.system1/support-route.s1m`, reloads it and tries
“Please refund this payment.” No API key or teacher service is involved.

### 3. Check the answer

The starter prints:

```text
Suggested team: billing
Needs review: True
```

`billing` is its best guess. **Needs review means a person should still decide.**
Eight examples show the process; they do not establish reliable automatic routing.
To judge a skill, check different messages: count both wrong answers and requests
for review. A saved file alone does not mean the skill is ready.

For a complete, larger demonstration with separate teaching, uncertainty checks
and final evaluation, run `python examples/support_triage.py`. It routes to four
departments and reports every result, including mistakes. The
[first-skill walkthrough](https://github.com/steph4n-gh/system1/blob/main/docs/guides/first_skill.md)
explains how to do this with your own messages.

### 4. Reuse the saved skill

The file is the portable skill. Load it without reteaching or calling the teacher:

```python
from system1 import CompiledSystemOneModel, System1Engine

skill = CompiledSystemOneModel.load(".system1/support-route.s1m")
engine = System1Engine(skill.schema, model=skill, strict_mode=True)
answer = engine.decide("Please refund this payment", record_receipt=False)
print(answer.values["team"])
print("Needs review:", answer.is_ambiguous)
```

The application honors the review flag before using an answer automatically.
Saving and reloading preserves uncertainty; it does not turn the starter's
uncertain answer into an approved one.

### 5. Improve with a correction

Copy the lessons and edit their answers or add varied real messages:

```bash
cp examples/teaching/first_skill.json .system1/my-lessons.json
python examples/teach_skill.py --lessons .system1/my-lessons.json --output .system1/candidate.s1m --prompt "The payment page crashes"
```

For that message, our desired answer is `support`: the issue is broken software.
If a lesson has the wrong answer, replace it. Rerun teaching, then check different
messages for **both** teams before adopting the candidate. This keeps the earlier
skill while you evaluate the revision. Follow the
[worked correction](https://github.com/steph4n-gh/system1/blob/main/docs/guides/first_skill.md#what-does-correcting-a-mistake-mean)
and [teaching API guide](https://github.com/steph4n-gh/system1/blob/main/docs/guides/training_experts.md)
for your own output choices and separate checking data.

## Observe a teacher, then take over

Once the lesson format makes sense, another source can supply the answers:
a colleague, reviewed historical decisions, existing rules, a conventional
classifier, Jev, Gemini or another API. JSON stores lessons; a teacher callback
supplies them while a program runs. An LLM can teach a skill without the resulting
skill becoming an LLM or inheriting the teacher's general knowledge.

```bash
python examples/observe_routing.py
```

This demonstrates the same journey automatically: observe answers, teach,
validate, switch to local decisions, then save and reload. Its default teacher
is an inspectable rule over synthetic tickets. An optional live Jev mode and
custom callbacks use the same [observation workflow](https://github.com/steph4n-gh/system1/blob/main/docs/typesafe.md).

The teacher stays responsible until the candidate qualifies. Local execution
can still request review. Promotion depends on evidence, not a promised number
of turns. Use `PromotionPolicy(min_accepted_agreement=.95)` when you need the
stronger accepted-agreement qualification added in 1.0.2; teacher agreement and
independent correctness are separate checks.

## Benchmarks

The [published public-data evaluation](https://github.com/steph4n-gh/system1/blob/main/benchmarks/quality/workloads/README.md)
measures three fixed bounded tasks. These 1,301 cases were kept out of teaching
and calibration in the original experiment:

| Explicitly taught skill | Answered without review | Correct among those answers | Median local decision |
|---|---:|---:|---:|
| Banking support, three intents | 118/120 | 115/118 (97.5%) | 0.378 ms |
| Assistant commands, six intents | 173/180 | 172/173 (99.4%) | 0.344 ms |
| SMS spam triage | 980/1,001 | 957/980 (97.7%) | 0.444 ms |

The report includes original labels, fixed splits, every error and a classical
TF-IDF/logistic-regression baseline. That baseline matched raw correctness on
banking and assistant, scored lower on SMS, and ran faster locally on all three.

The unreleased [real-email experiment](https://github.com/steph4n-gh/system1/blob/main/benchmarks/quality/public_email/README.md)
uses public SpamAssassin labels: a representative teaching recipe accepts
618/632 held-out spam/ham decisions, with 603/618 correct (97.6%), after 0.94 seconds
of teaching. Median local decisions take 0.216 ms and the saved skill is 78.2 KiB.
It retains 15 accepted mistakes and an earlier source-shift failure (89.9% accepted
correctness). The representative split was designed after that failure; it is not
new blinded evidence or qualification of the seven-category Inbox Zero pilot.

Automatic observation is measured separately. Under the original default policy,
actual **Jev and Gemini 2.5 Flash** each enabled assistant takeover after 358
observations: 171/180 old official test requests accepted locally, 167/171 correct,
with no further teacher calls. Saved/reloaded results matched. Recorded teacher
responses can be replayed offline.

The [1.0.2 quality round](https://github.com/steph4n-gh/system1/blob/main/benchmarks/quality/quality_round/README.md)
uses stronger qualification and a fuller SMS stream: 2,961 observations,
954/980 correct accepted old-test decisions (97.3%). Banking and assistant remain
deferred under that policy. **Fresh authored SMS probes reach only 27/33 correct
accepted answers (81.8%)**. New banking phrasing also reveals weaknesses, and a
routing teaching candidate was not adopted after mixed results. The old tests are
now regression evidence; the new authored probes are diagnostics, not independent
customer evidence. These limitations remain visible.

To check the primary teaching examples and the recorded Jev comparison:

```bash
python benchmarks/quality/evaluate_release.py
```

This runs offline and fails if a primary demonstration misses its 95% accepted
correctness / 80% acceptance point targets. See the [quality index](https://github.com/steph4n-gh/system1/blob/main/benchmarks/quality/README.md)
for all reproduction commands, confidence intervals and historical experiments.

## Integrations

The project is packaged on PyPI as `system1`. To add it to an existing Python project:

```bash
python -m pip install system1
```

After checking the skill on your task, connect it to your application. The core
supports choice, boolean, multi-choice and score outputs; uncertainty tells the
application when to request review.

| Need | Next step |
|---|---|
| Teach a different task or use your own data | [Teaching API and file formats](https://github.com/steph4n-gh/system1/blob/main/docs/guides/training_experts.md) |
| Learn by observing a running teacher | [Teacher callbacks and automatic takeover](https://github.com/steph4n-gh/system1/blob/main/docs/typesafe.md) |
| Use Apple Silicon decision-head operations | Install `system1[metal]`; see [backend scope](https://github.com/steph4n-gh/system1/blob/main/docs/architecture/technical_specification.md) |
| Connect LangChain, MCP, ASGI or gRPC | [Integration guide](https://github.com/steph4n-gh/system1/blob/main/docs/integrations.md) |
| Pilot local email categories in Inbox Zero | [Working integration and measured quality limits](https://github.com/steph4n-gh/system1/blob/main/examples/inbox_zero/README.md) |
| Monitor local decisions | [Metrics and tracing](https://github.com/steph4n-gh/system1/blob/main/docs/observability/README.md) |

### Policy guard

If a decision would execute a tool, add an explicit permission policy. The
[executable guard quickstart](https://github.com/steph4n-gh/system1/blob/main/docs/guides/policy_guard.md)
shows `PolicyEngine`, an enforcement guard and a durable ledger. Optional
software Ed25519 signatures (RFC 8032) and a SHA-256 hash-chained ledger provide
audit evidence. A classifier's suggested answer alone is not permission to act.
See [deployment boundaries](https://github.com/steph4n-gh/system1/blob/main/docs/deployment.md)
for identity, network and enforcement responsibilities.

## Examples and research

The [Snake arena](examples/gaming/snake_arena/README.md) puts System1, Laya-MLX
and Jev side by side in a live GUI. Replay the published runs without an API key,
or run fresh comparisons. In the recorded 30-second race, System1 filled the board
in 13.44 seconds; three equal-move runs matched Jev's preferred-move choices.
All engines receive planner hints, so this demonstrates a tiny taught skill's
execution cost, not independent game reasoning. See the
[full measurements and limitations](examples/gaming/snake_arena/RESULTS.md).

The [example catalog](https://github.com/steph4n-gh/system1/blob/main/examples/README.md)
links every teaching, observation, game and integration example with its scope.
The [architecture](https://github.com/steph4n-gh/system1/blob/main/docs/architecture/technical_specification.md),
[technical brief](https://github.com/steph4n-gh/system1/blob/main/docs/paper/system1_technical_brief.md)
and [whitepaper](https://github.com/steph4n-gh/system1/blob/main/docs/paper/system1_whitepaper.md)
explain the implementation and evidence. The papers are maintainer-authored,
not peer-reviewed publications. Established numerical methods power the product;
the practical value is the complete path from examples to a checked local skill.

## Release and development

**Version 1.0.3:** authenticated audit receipts, checked ledger inclusion, bounded
saved-skill loading and registered gRPC schemas. Includes the numeric teaching
fix and supervised Paperclips example.
See the [release report](https://github.com/steph4n-gh/system1/blob/main/docs/releases/1.0.3.md)
and [1.0 migration guide](https://github.com/steph4n-gh/system1/blob/main/docs/releases/1.0.md).
The legacy `reflex` import namespace remains a compatibility alias.

```bash
python -m pip install -e '.[dev,langchain]'
python -m pytest tests/ -q
```

CI checks Linux/macOS tests, documentation, built distributions and real
LangChain dispatch. See [contributing](https://github.com/steph4n-gh/system1/blob/main/CONTRIBUTING.md)
and [security reporting](https://github.com/steph4n-gh/system1/blob/main/SECURITY.md).

[Apache License 2.0](https://github.com/steph4n-gh/system1/blob/main/LICENSE).
