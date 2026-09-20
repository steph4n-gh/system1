# System 1: technical brief

**System 1 1.0.1 · 19 September 2026 · Maintainer-authored implementation brief**

System 1 turns a repeatable decision into a small reusable local skill. Define the
outputs, teach from examples or observe a teacher, validate the skill, then reuse
it through the same local runtime. A teacher can be a person, JSON data, a rule,
Jev, Gemini or another callback. System 1 fits numerical decision heads; it does
not fine-tune or run a language model.

## How it works

Fixed text features and optional numerical telemetry feed NumPy decision heads.
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
