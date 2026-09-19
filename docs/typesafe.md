# Observe a teacher and reuse a local skill

System 1's TypeSafe adapter supports the basic Jev decision workflow: typed
questions in, typed answers out. A teacher can be Jev or an existing callback.
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

On the review machine, promotion occurred after 190 observations: 114 fitting
cases, 38 calibration cases, and 38 validation cases. The calibration cases
split again into temperature fitting and conformal calibration. Validation
agreement was 38/38 and 37/38 responses were accepted. All 40 subsequent local
cases were correct and accepted. The saved skill was about 5 KB. These inputs
reuse a small topic vocabulary with new ticket IDs: this demonstrates the
lifecycle, **not open-ended language understanding or Jev quality parity**.
Do not interpret the ticket IDs as independent real-world scenario diversity.

To observe real Jev answers with the same example:

```bash
# Set TYPESAFE_API_KEY in your environment first. This makes billable API calls.
python examples/observe_routing.py --teacher jev --output-dir .system1/observe-jev
```

There is no simulated fallback in that mode. Insufficient evidence leaves the
teacher active. No live Jev quality result is included in this release. The
fresh evaluation compares against the example's stated routing policy; it does
not call Jev again after disconnection.

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

The normal policy checks held-out agreement, its Wilson lower bound, critical
false-allows, and at least 80% local acceptance. Agreement among accepted
responses must also meet the agreement threshold. Validation is per schema;
adding an unrelated question creates a different skill. `cutover_threshold=50`
is the earliest attempt, not a promise that 50 observations suffice. Repeated
validation attempts and changing traffic require independent deployment
validation; these checks are not a sequential statistical guarantee.

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
answer. Strict prediction sets invert the calibrated cumulative-probability
score; an empty set also requires review. See the
[conformal prediction procedure](https://arxiv.org/html/2107.07511v6#S2.SS1).
