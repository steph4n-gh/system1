# System 1: technical brief

**System 1 1.0.3 · 20 September 2026 · Maintainer-authored implementation brief**

System 1 turns a repeatable decision into a small reusable local skill. Define the
outputs, teach from examples or observe a teacher, validate the skill, then reuse
it through the same local runtime. A teacher can be a person, JSON data, a rule,
Jev, Gemini or another callback. System 1 fits numerical decision heads; it does
not fine-tune or run a language model.

**System 1 is not an LLM:** the built-in path needs no language-model weights,
token generation or GPU. Optional MLX supports Apple Silicon decision-head
operations; the results below use NumPy, and speedups must be measured for the
workload. Its narrow output schema and learned feature weights
are also its limits; it does not acquire the teacher's general knowledge.

## How it works

Fixed text features and optional numerical telemetry feed NumPy decision heads.
The optional TF-IDF projector fits a vocabulary on teaching text and freezes it
for inference, retaining the existing ridge head and NumPy-only runtime.
Separate examples calibrate uncertainty. Explicit teaching produces a `.s1m`
file; automatic observation keeps the teacher answering until promotion checks
pass. A saved skill preserves the schema, feature configuration, weights and
uncertainty behavior. New inputs are evaluated through those shared features,
not only looked up in a cache.

The local result includes a value and review information. The application chooses
how to handle review requests. Automatic takeover removes the teacher from the
normal decision path, but does not mean every subsequent input is accepted.

## Demonstrated behavior

On three fixed public-data tasks, explicitly taught skills accepted 96.1–98.3%
of 1,301 held-out cases, with 97.5–99.4% correctness among accepted decisions.
Teaching plus calibration took 0.23–1.77 seconds once labeled data existed;
saved skills were 20.4–47.0 KiB and median local decisions took 0.34–0.44 ms.
These are measurements from one machine, excluding data preparation, startup,
receipts and downstream actions.

On a separate automatic-observation path, actual Jev and Gemini each taught the
six-intent assistant skill in 358 observations. With network access disabled,
each skill accepted 171/180 unseen requests and got 167/171 accepted decisions
right. Nine requests needed review. Saved/reloaded outputs matched.

Banking did not qualify for automatic takeover with the available observations.
SMS taught from observed dataset labels promoted under the normal 80% gates but
missed the independent 95% correctness target; the live Jev SMS run did not promote.
The classical TF-IDF/logistic-regression baseline ran faster locally. Full counts,
methods, errors, teacher recordings and reproduction commands are in the
[workload report](../../benchmarks/quality/workloads/README.md).

The later [1.0.2 quality round](../../benchmarks/quality/quality_round/README.md)
adds an opt-in 95% accepted-agreement lower-bound requirement. With fuller local
label observation, SMS takes over after 2,961 cases and gets 954/980 accepted old
test decisions right (97.3%). Banking and assistant remain deferred. Fresh authored
SMS probes reach only 27/33 correct accepted decisions (81.8%). A proposed routing
teaching change is retained as an experiment because its tradeoffs failed the
adoption criteria. These limits remain part of the evidence.

## What the product adds

The practical value is the integrated path from an existing teacher to a validated,
portable local skill, with explicit review behavior and optional audit evidence.
Ridge regression, feature hashing, conformal prediction and digital signatures
are established techniques. These results support the bounded tasks measured;
they do not establish universal provider parity or a novel learning algorithm.

For application permissions, `PolicyEngine` evaluates explicit rules and
`SystemOneGuard(enforcement_profile=True)` requires a permission grant, signing
key and durable ledger. A classifier answer alone is not authorization. Software
Ed25519 receipts and a SHA-256-chained SQLite ledger support auditing; applications
must authenticate callers, protect keys and enforce tool boundaries.

Version 1.0.3 requires independent receipt verification keys, binds ledger IDs in
signed envelopes, checks exact inclusion and full ledger history, and bounds
saved-skill loading. gRPC resolves only registered schemas. See the
[security release notes](../releases/1.0.3.md) for compatibility changes.

## Start here

From an installed repository checkout:

```bash
python examples/support_triage.py
python examples/observe_routing.py
python benchmarks/quality/evaluate_release.py
```

These run locally; the observation tutorial uses an inspectable rule teacher.
For your own task, follow the [teaching guide](../guides/training_experts.md).
The [architecture reference](../architecture/technical_specification.md),
[teacher adapter](../typesafe.md), [deployment boundaries](../deployment.md) and
[documentation index](../README.md) describe the supported interfaces.


The unreleased [Inbox Zero pilot](../../examples/inbox_zero/README.md) now includes
a TF-IDF feature option and broader lessons. It improves the original email
result and speed, but its new quality evidence is synthetic, includes an accepted
mistake, and does not qualify real-mailbox automation. Its recorded results are
separate from the public-data and teacher comparisons above.

The additional [real-email recipe](../../benchmarks/quality/public_email/README.md)
teaches spam/ham from public SpamAssassin labels in 0.94 seconds and saves a
78.2 KiB local skill. Its representative grouped evaluation accepts 618/632
decisions, 603 correct (97.6%), at 0.216 ms median latency. This retrospective
experiment follows a preserved source-shift failure at 89.9% accepted correctness;
it supports a bounded spam/ham skill, not seven-category Inbox Zero automation.
