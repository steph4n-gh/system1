# Observe a teacher and reuse a local skill

System 1's TypeSafe adapter supports the basic Jev decision workflow: typed
questions in, typed answers out. A teacher can be Jev, another API or LLM through a callback, or labeled local data.
The adapter observes answers, teaches a small decision head, validates it, and
switches the same call site to local execution. It does not fine-tune an LLM.

## Start with the complete offline example

```bash
python examples/observe_routing.py
```

The teacher in this example is a deterministic rule over synthetic structured
support tickets. It is deliberately easy to inspect. The example uses normal
promotion defaults, disconnects the teacher after promotion, evaluates 40 fresh
tickets, saves `routing.s1m`, and checks identical answers, probabilities,
prediction sets, and review decisions after reload. It writes a complete JSON
report under `.system1/observe-routing/`.

The recorded offline and live Jev runs both promoted after 357 observations:
215 fitting cases, 71 calibration cases, and 71 validation cases. Calibration
splits again between temperature and conformal scoring. All 71 validation
decisions agreed and were accepted on attempt ten; all 40 subsequent local
cases were correct and accepted. The saved skill is about 5 KB. The live run
took 116 seconds to observe and teach, then about 0.4 ms per local decision.

These inputs reuse a small topic vocabulary with new ticket IDs: this proves the
lifecycle on that structured workload. Ticket IDs do not establish independent
real-world scenario diversity. See the [live evidence](../benchmarks/quality/results/jev_live_cutover.json).

To observe real Jev answers with the same example:

```bash
# Set TYPESAFE_API_KEY in your environment first. This makes billable API calls.
python examples/observe_routing.py --teacher jev --output-dir .system1/observe-jev
```

There is no simulated fallback in that mode. Insufficient evidence leaves the
teacher active. The fresh evaluation compares against the example's stated routing policy; it does
not call Jev again after disconnection.

## Public-data takeover with Jev and Gemini

On the six-intent assistant task, actual Jev 1.13.0 and Gemini 2.5 Flash each
qualified after 358 observations. With teacher callbacks and network access
disabled, each skill accepted 171/180 unseen official test requests, with 167/171
correct. Nine requests required review. Saving and reloading preserved results.

The [workload report](../benchmarks/quality/workloads/README.md) includes the
protocol, real response records and offline replay commands. Banking did not
promote with the available 412 observations. SMS observed from original labels
promoted but missed the independent 95% correctness target; live Jev SMS did not
promote within 1,000 observations. The normal 80% promotion thresholds do not
qualify a skill for a separate 95% accepted-correctness requirement.

## Integrate the same lifecycle

```python
from system1.compat.typesafe import Client, Choice

questions = {
    "queue": Choice(
        instructions="Route the request by its topic.",
        criteria={"billing": "Invoices and payments", "access": "Login and passwords"},
    )
}

# Jev answers until the observed skill passes validation.
with Client(mode="auto_cutover", zero_egress=False) as client:
    response = client.systemone(state={"topic": "invoice"}, questions=questions)
    print(response.choices.queue.choice)
    print(client.is_cutover)
    # Call systemone for each real request. Export succeeds only after promotion.
    if client.is_cutover:
        client.export_model("queue.s1m")
```

For another teacher, pass `baseline_handler=teacher` to the same constructor.
It receives `(state, questions)` and returns a response mapping containing
`answers`, with each answer carrying `type` and `choice`, `noul`, or `score`
(or the adapter's normalized `value`). A callback is preferred when supplied.
Its network behavior remains the application's responsibility; `zero_egress`
blocks the adapter's own HTTP transport, not arbitrary callback code.

`Client(mode="auto_cutover", model_path="queue.s1m")` reopens the validated skill
without a teacher or API key. The questions must match its complete saved schema.
The file restores built-in deterministic/hybrid projectors and their settings,
weights, temperature, conformal scores, and the adapter's evaluated strict-mode
setting. A mismatched projector or schema raises an error. External projectors
must be supplied explicitly; implement `projector_digest()` to validate their
configuration. Older files without projector metadata retain legacy loading
behavior and cannot verify the original feature space.

## What promotion means

The default observation path uses **only observed labels** (`augment=False`).
A missing class or insufficient disjoint evidence defers promotion. Normalized
repeated prompts, request IDs and lineage groups must not cross fitting,
calibration and validation folds. Supply group/lineage identifiers for related
requests; the runtime cannot discover every paraphrase automatically.

The normal policy checks complete-decision agreement, critical false-allows,
at least 80% agreement, and at least 80% local acceptance. All fields must agree, including exact
MultiChoice sets. Related observations count as one validation group; repeating
an easy request cannot inflate either agreement or acceptance. Agreement among
accepted decisions must also meet the threshold.

For the statistical gate, both agreement and acceptance must pass exact
one-sided binomial lower bounds. At validation attempt `k`, each bound spends
`(1 - confidence) / (2 * k * (k + 1))`; the total across attempts is bounded by
`1 - confidence`. Each attempt needs an entirely fresh validation block. This
controls repeated checks only under independent, representative validation
groups and a candidate fixed before its validation. Correlated templates,
adaptive traffic, or changed workloads do not satisfy those assumptions merely
because their IDs differ. The Wilson lower bound remains a diagnostic and an
additional conservative gate.

Validation and drift state are per schema. `cutover_threshold=50` is the earliest
attempt, not a promise that 50 observations suffice. The default statistical
policy may need hundreds of examples. Deployment validation remains necessary.

Promotion enables local execution, which can still return `is_ambiguous` or
`abstain`. Applications must honor review requests. Cloud fallback is off by
default and cannot be combined with `zero_egress=True`. HTTP errors propagate
unless the caller explicitly opts into a labeled simulation with
`fallback_baseline=True`. `compare()` reports no speedup for simulated responses
and does not manufacture pricing or performance data after a failed request.

Explicit `PromotionPolicy` objects can override the defaults, including opting
out of the acceptance check when `min_local_acceptance=0`. The older enterprise
showcases explicitly retain synthetic augmentation, heuristic non-strict gates,
and relaxed promotion settings. Their output demonstrates integration mechanics;
use the flagship example for the normal observation path.

## Supported SDK surface and differences

The reference is the [official Python SDK at commit 2ce5c65](https://github.com/typesafe-ai/typesafe-sdk-python/tree/2ce5c65f13646cab6e6f782328194c9d85f3300a).
This is compatibility with its basic decision API, not its entire transport or
Pydantic model contract.

| Surface | Behavior |
|---|---|
| `Client`, `AsyncClient` | Sync/async context managers, `close()` / `aclose()`, `systemone`, constructor model default and request override. |
| Structured `state` | Strings, JSON objects and arrays. Canonical JSON text supplies local features; HTTP preserves the original JSON value. |
| Questions | `Choice`, `Noul`, `NoulCriteria`, ordinal `Score`; mappings and objects exposing `model_dump()` also work. JSON-valued descriptions/instructions are retained on the wire. |
| Responses | `.answers`, `.choices`, `.nouls`, `.scores`, nested attribute access and token usage. Responses are mutable dicts, not the SDK's immutable Pydantic models. |
| Ordinal `Score` | Two to ten ordered levels; actual per-level probabilities, integer-keyed legend, and probability-weighted mean score. Every level participates in the local head. Observed scalar scores teach the nearest level, so this is not full teacher-distribution distillation. |
| Local extensions | Explicit-range continuous scores, `MultiChoice`, receipts, uncertainty sets, local batch helpers and portable skills. These are not promises about Jev's HTTP API. |
| Unsupported extensions | SDK retry policies, custom HTTP clients/transports, custom headers, extra request bodies, custom response models, and model-management endpoints. Supplied transport/response options raise `NotImplementedError`. |

Local confidence is the local head's probability, not a promise to reproduce
Jev's confidence computation or semantic quality. For ordinal scores, receipts
bind the underlying chosen level and level probabilities; the returned mean
score is derived from that distribution. Strict calibration is marginal under
appropriate exchangeability assumptions, not a guarantee for each accepted
answer. Newly taught examples-only skills use the standard LAC score
`1 - probability(label)`. Legacy and schema-augmented skills retain APS cumulative
probability scores. Both invert their saved score in strict mode; an empty set
also requires review. See the
[conformal prediction procedure](https://arxiv.org/html/2107.07511v6#S2.SS1).


## Saved skills and corrections in 1.0

New `.s1m` files use format v2 so older runtimes cannot silently interpret LAC
scores as APS. System 1 1.0 reads v1 artifacts using their original APS semantics;
0.2.x cannot read v2. Upgrade readers before distributing newly saved skills.
Invalid versions, schema identity, shapes, and nonfinite weights or temperatures
are rejected. Keep a copy of the original artifact when migrating.

`engine.calibrate(...)` saves the resulting evidence into the compiled model.
`learn_from_tier2(...)` invalidates the changed heads' calibration, including in
saved files. Strict decisions request review until those heads are recalibrated
on separate examples. Calibration, correction, inference, and serialization of a
shared compiled model are synchronized. Retaining examples and recompiling is
the simplest reproducible update workflow.

The [release comparison](releases/1.0.md) separately measures three natural-language
skills taught from actual recorded Jev responses. It does not claim all three
automatically promoted from a live stream or that local confidence equals Jev's.
