# System 1: teaching bounded decisions for local reuse

**steph4n-gh · System 1 1.1.0 · 22 September 2026**
Maintainer-authored technical whitepaper; not a peer-reviewed publication.
[Source repository](https://github.com/steph4n-gh/system1)

This document describes [System 1 1.1.0](../releases/1.1.0.md). Measurements retain
the original source revisions and evidence linked in each experiment; this
release does not make historical results fresh evaluations.

System 1 is **not an LLM**. Its built-in local path fits small numerical decision
heads and requires no language-model download, token generation or GPU. It can
learn a bounded decision from an LLM's answers without becoming a language model.

## Abstract

System 1 is a local runtime for repeatable structured decisions. It fits small
NumPy decision heads from labeled examples or observed teacher outputs, calibrates
uncertainty, validates automatic promotion, and saves the resulting skill for
reuse. Its demonstrated value is a complete teaching-to-local-execution workflow
using established statistical and software techniques. On three public-data tasks,
explicitly taught skills meet fixed 95% accepted-correctness and 80% acceptance
point targets across 1,301 held-out cases. Actual Jev and Gemini observations each
enable automatic local takeover on a six-intent assistant task. Banking and SMS
observation tests expose limits, retained alongside the successful results.
The later full-scope n8n gauntlet remains unqualified; its teacher-disconnection
rehearsal establishes integration behavior, not successful quality qualification.
A pre-release correction-workflow experiment on real BBC articles reduces accepted
mistakes by requesting more review; every candidate fails the default adoption
policy. These results distinguish successful workflow checks from workload readiness.

## 1. Problem and scope

Applications often repeat a narrow decision: route a support request, identify an
assistant command or triage a message. Once labels and boundaries are stable,
calling a general reasoning service for every instance may be unnecessary.
System 1 teaches that decision and applies it locally. A new request need not
exactly match a prior request, because the decision head operates on shared text
features. This is limited statistical generalization, not newly acquired general
language understanding.

The product loop is **teach → check a candidate → adopt → correct and compare**,
with automatic teacher observation as an alternative source of lessons and
promotion evidence.
A human, JSON file, decision API, LLM or application callback can provide labels.
Teacher outputs are supervision, not automatically ground truth. Independent
expected labels are needed to assess whether the teacher and local skill are right.

The dual-process name is a product metaphor for a fast local skill supported by
an external teacher or reviewer. It is not a claim to reproduce human cognition.

## 2. Implementation

Schemas define Choice, Boolean, MultiChoice and score fields. A deterministic
projector creates SHA-256-derived lexical and subword features, defaulting to 384
dimensions. An optional hybrid projector adds a seeded local subword table with
curated semantic anchors. Neither path downloads a pretrained language model.
An optional fitted TF-IDF projector now supplies word unigram/bigram features
for taught text tasks. It uses the same numerical heads and saves its frozen
vocabulary and IDF weights in the skill; NumPy remains sufficient at runtime.
Numerical telemetry can contribute features for structured state.

For explicit teaching, `SystemOneCompiler(...).compile(examples, augment=False)`
fits regularized linear heads. The default ridge implementation augments features
with a bias, computes regularized normal equations using NumPy, and retains calibration
metadata. Temperature fitting and conformal scoring use separate examples.
The compiler additionally offers opt-in `choice_solver="logistic"` for
choice fields: SciPy minimizes cross-entropy with L2-regularized weights and an
unpenalized bias during teaching. It saves the existing NumPy head format. Ridge
remains the default, including automatic observation. See the
[teaching guide](../guides/training_experts.md#optional-classification-fitting).
The inference path returns typed values, probabilities and review signals.
Detailed defaults and code links are in the
[architecture reference](../architecture/technical_specification.md).

`System1Engine(..., strict_mode=True)` uses the saved uncertainty evidence.
The direct Python engine's legacy default is non-strict; the normal observation
adapter and saved-model CLI use strict review. Examples-only categorical/Boolean
skills use LAC scores, while legacy and augmented skills retain APS semantics.
MultiChoice calibrates complete assignments rather than treating any selected
list as a categorical singleton. Model corrections invalidate changed heads'
calibration; separate recalibration is required before strict acceptance resumes.

Format-v2 `.s1m` artifacts preserve schema, feature configuration, weights,
temperature and score semantics. Tests check values, probabilities, sets and
review behavior across saving and loading. The current reader also understands
legacy v1 artifacts. A smaller skill file does not include the Python runtime or
installed dependencies.

Optional research adapters may supply features from fixed pretrained text
encoders. Those require separate encoder weights and execution, even though the
decision head does not generate language or update the encoder. Their reports
count both parts; the default projector measurements cannot be applied to them.

### Retained lessons, candidate assessment and explicit adoption

`TeachingSession`, introduced in 1.1.0, adds a reproducible correction workflow for
one text `ChoiceField`, with ambiguity escalation enabled so uncertain answers
request review. It retains explicitly labeled `teach`, `calibrate` and
`evaluate` records. A same-input correction replaces a record; normalized inputs
cannot cross splits. Related threads and paraphrases still require caller-managed
grouping. The helper uses the existing fitted TF-IDF projector (at most 1,024
features), ridge regularization 0.1, and strict inference at `alpha=0.05`. These
are helper-specific choices, not changes to existing compiler or engine defaults.

`assess()` compiles a separate candidate, saves/reloads it and compares it with
the approved skill on the same evaluation rows. The report exposes raw accuracy,
acceptance coverage, accepted errors and per-case predictions/review sets.
A useful regression is an earlier accepted-correct case that becomes wrong or
requires review; raw correctness regressions are also counted separately.
`adopt()` requires a passing report and unchanged data, candidate and current
artifact. Editing lessons or failing assessment does not replace the approved
skill. Existing algorithms, `.s1m` format and dependencies are unchanged.

The defaults require raw accuracy ≥0.8, coverage ≥0.5, no accepted errors and no
useful regressions. They are demonstration policy, not statistical production
qualification. Checks reused to guide corrections are recurring development
evidence, even though excluded from fitting and calibration. Diagnostics help
distinguish wrong raw choices from withheld choices; they do not establish causal
failure explanations or out-of-distribution detection. For decisions acting in a
workflow, fixed-list accuracy also cannot establish successful task completion.
The [correction guide](../guides/correcting_skills.md) is the authoritative API and
CLI walkthrough; the [document workspace](../../examples/teaching_by_doing/README.md)
uses this same lifecycle.

## 3. Automatic teacher observation

The TypeSafe-compatible adapter records valid teacher answers by schema. It fits
a candidate from observation groups, calibrates it on separate evidence, and
checks fresh held-out validation groups. Related requests must not span those
partitions. The caller can supply lineage identifiers; the runtime does not
magically recognize every paraphrase or common source.

The normal promotion policy requires 80% agreement and 80% local acceptance,
with 95% statistical confidence and zero observed critical false-allows. It
checks complete decisions and accepted-decision agreement. Exact one-sided
binomial bounds use a summable error budget across repeated attempts, and each
attempt requires fresh validation groups. Those statistical interpretations need
independent representative groups and a candidate fixed before validation.

The 50-observation default trigger is the earliest attempt, not a guaranteed
handoff count. A failure keeps the teacher answering. Successful promotion enables
local computation; uncertain local decisions still require application review.
The [adapter contract](../typesafe.md) documents fallback, callback and SDK limits.

## 4. Public-data evaluation

The [protocol](../../benchmarks/quality/workloads/protocol.json) was committed as
`b5e5246` before the first new assistant/SMS evaluation. Banking repeats the fixed
1.0.1 task. No teaching cases, model settings, thresholds or core code were changed
in response to these results. Public sources, licensing, checksums, partitions,
every prediction and reproduction commands are in the
[complete workload evidence](../../benchmarks/quality/workloads/README.md).

Explicit teaching uses 2,048 features, ridge regularization 0.1, supplied examples
only, alpha 0.05 and strict review. Network access, caches and receipts are disabled
for local evaluation, and saved/reloaded results must match.

| Task | Fitting / calibration / test cases | Raw correct | Accepted | Correct among accepted |
|---|---:|---:|---:|---:|
| Banking, three intents | 287 / 125 / 120 | 115/120 | 118/120 | 115/118 (97.5%) |
| Assistant, six intents | 600 / 120 / 180 | 175/180 | 173/180 | 172/173 (99.4%) |
| SMS spam triage | 3,061 / 1,068 / 1,001 | 972/1,001 | 980/1,001 | 957/980 (97.7%) |

These meet the predefined point targets on their stated tasks. Descriptive Wilson
intervals and per-class errors remain in the report; passing a point target does
not establish the same accuracy on future traffic. The banking interval for
accepted correctness extends below 95%.

A TF-IDF/unigram-bigram logistic-regression baseline uses the same fitting rows
and conformal half. It matches raw correctness on banking and assistant and is
lower on SMS, while fitting and evaluating faster locally on all three tasks.
System 1's saved artifacts are smaller in this comparison, but the serialization
formats differ. No universal classifier or speed superiority follows.

Teaching plus calibration took 226, 239 and 1,767 ms. Median local decisions took
0.378, 0.344 and 0.444 ms on the review machine, Python 3.13.5/NumPy 2.5.3.
Those timings exclude label preparation, imports, startup, audit work and
application actions. No service-level or hardware-independent bound is asserted.

## 5. Live teachers and unsuccessful takeovers

Observation uses the unchanged normal gate and compiler regularization 1.0,
reserving fitting, calibration and validation evidence. It stops at promotion or
1,000 available observations. Test cases never enter the teacher stream.

| Task / teacher | Observations | Promoted | Correct among accepted test decisions |
|---|---:|---|---:|
| Assistant / original labels | 358 | Yes | 167/171, out of 180 cases |
| Assistant / actual Jev 1.13.0 | 358 | Yes | 167/171, out of 180 cases |
| Assistant / actual Gemini 2.5 Flash | 358 | Yes | 167/171, out of 180 cases |
| Banking / original labels | 412 | No | — |
| Banking / actual Jev | 412 | No | — |
| SMS / original labels | 557 | Yes | 897/963 (93.1%), out of 1,001 cases |
| SMS / actual Jev | 1,000 | No | — |

Both live assistant teachers match all observed original labels. After promotion,
teacher callbacks and sockets are disabled. Each local skill accepts 95% of unseen
assistant requests at 97.7% correctness among accepted decisions and preserves
results after reload. This demonstrates teacher independence for one bounded skill.
It does not establish parity with either provider's complete capabilities.

The banking stream does not qualify with its available evidence. The SMS dataset
teacher qualifies under the normal 80% gate but misses the independent 95%
accepted-correctness target. The live Jev SMS stream does not qualify within the
cap. These results distinguish gate eligibility from workload suitability.

The report retains one Gemini HTTP 503 and the completed response cache. Recorded
responses reproduce promotion and quality entirely offline without API keys.
Teacher and local latency cohorts differ, so their timings are not a matched-input
speedup benchmark. No dollar savings are inferred from token usage.

### 1.0.2 qualification follow-up

The [follow-up protocol and results](../../benchmarks/quality/quality_round/README.md)
add opt-in accepted-agreement qualification without changing the original policy.
With fuller label observation, SMS promotes after 2,961 cases and reaches 954/980
correct accepted old-test decisions (97.3%). That reused test is regression
evidence. Fresh authored SMS probes reach only 27/33 (81.8%); banking and assistant
remain deferred under the stronger policy, and a routing revision is not adopted.
The [evidence index](../../benchmarks/quality/README.md) retains these failures,
recipes and comparisons alongside the original results.

### Real-document correction workflow (pre-release experiment)

The [BBC experiment](../../benchmarks/quality/document_workflow/README.md) evaluates
the new session workflow on all five news topics using published labels as
simulated human feedback. A frozen protocol groups duplicate and near-duplicate
articles before separating 100 initial lessons, 200 calibration examples,
200 recurring adoption checks, a feedback pool and 400 final holdout articles.
Exactly 64 wrong or reviewed examples are added after inspecting 400 feedback
labels. An ordinary-addition control receives the same number of lessons, without
targeting errors; it does not match the cost of inspecting labels. All candidates
and adoption outcomes are frozen before the final holdout is scored.

On that holdout, initial → targeted raw correctness is 369→373/400. The one-point
gain has a paired 95% bootstrap interval of −1.25 to +3.5 percentage points.
Accepted mistakes fall 16→8 while accepted decisions fall 373→337 and correct
accepted decisions fall 357→329. The ordinary-addition control has 372/400 raw
correct, 310/314 accepted correct and four accepted mistakes. These comparisons
demonstrate differing review/quality tradeoffs, not a clear raw-quality or
sample-efficiency advantage for targeted feedback.

The targeted candidate meets the separate 95% accepted-correctness / 80% coverage
holdout targets, with a 95.39–98.79% Wilson interval for accepted correctness.
However, the recurring adoption checks retain 8, 4 and 1 accepted mistakes for
the initial, targeted and ordinary candidates: all fail the unchanged default
zero-error gate. Nothing is adopted, and no approved incumbent exists in this
run. The experiment tests stale/failed rejection, not preservation of an already
serving skill or successful adoption/reopen prediction parity. Published evidence
includes every outcome and source hash. Historical news-topic routing does not
qualify private document filing, unfamiliar-topic rejection or traffic drift.

## 6. Statistical and operational boundaries

Split-conformal prediction provides marginal prediction-set coverage when the
scoring model is independent of the calibration fold and calibration/test cases
are exchangeable. It does not guarantee correctness conditional on accepting a
singleton, simultaneous coverage across all fields, or adversarial robustness.
The [mathematical note](conformal_gating.md) states the assumptions and distinguishes
this established result from promotion policy and tool permissions.

Explicit permissions are evaluated by `PolicyEngine`. With
`SystemOneGuard(enforcement_profile=True)`, authorization requires a matching
permission grant, signing key and durable ledger. Applications must authenticate
callers and prevent execution from bypassing the guard. Statistical labels do
not grant permission.

Optional receipts use software Ed25519 signatures, as described in
[RFC 8032](https://www.rfc-editor.org/rfc/rfc8032.html). The SQLite ledger is a
linear SHA-256 hash chain, not a Merkle tree or hardware attestation. Signatures
verify recorded payloads under a trusted key; they do not prove correct external
execution or complete event capture on a compromised host.

In 1.0.3, authentication requires an independent key; unsigned diagnostics expose
only an integrity check. The signed envelope binds the committed ledger ID,
guards check exact inclusion, and writes reverify the whole chain. Saved-skill
resource budgets and a closed gRPC schema registry constrain input handling.
See the [security release and migration](../releases/1.0.3.md).

Local decisions require no teacher API, but callbacks, tools, network services
and telemetry exporters can communicate externally. The package is not a process
sandbox or a regulatory certification. See [deployment boundaries](../deployment.md).

## 7. Interpretation and remaining work

The evidence supports narrow routing and classification tasks taught from
representative examples. The useful integration is the transition from an
existing teacher to a validated, portable local skill. Feature hashing, ridge
regression, conformal prediction, recursive least squares and signatures are
established components; no new general learning algorithm or first-of-its-kind
claim is made here.

Public datasets may have appeared in provider pretraining. Related paraphrases
and SMS campaigns can remain correlated despite duplicate checks. Task subsets
above do not evaluate all BANKING77/CLINC150 intents or out-of-scope detection, and
historical SMS data does not measure modern phishing. Scripted gaming gains are
separate [first-use policy evidence](../../benchmarks/quality/zero_shot/README.md),
not learned strategic competence or verified speed records.

Later experiments widen the scope without establishing universal quality. The
[full-scope n8n gauntlet](../../benchmarks/quality/n8n_gauntlet/README.md) still fails
accepted accuracy on all-intent CLINC and BANKING77, plus unfamiliar-input
rejection on CLINC. Its teacher-disconnection recording is an integration
rehearsal using an unqualified artifact; independent confirmation remains missing.
The [Inbox Zero pilot](../../examples/inbox_zero/README.md) improves authored email
results but retains accepted errors and lacks independent mailbox qualification.
A separate binary [public-email experiment](../../benchmarks/quality/public_email/README.md)
retains an 89.9% accepted-correctness source-shift failure and a later retrospective
97.6% result, with slightly better raw correctness from its conventional baseline.
Neither result qualifies seven-category mailbox automation or modern drift.

The [document workspace](../../examples/teaching_by_doing/README.md) demonstrates
explicit lessons, corrections, candidate comparison and adoption on recurring
authored checks. Courier and Pokémon labs retain bounded composition tests,
baselines and regressions. Snake receives planner hints. None establishes general
agency. The [evidence index](../../benchmarks/quality/README.md) and
[example catalog](../../examples/README.md) preserve complete protocols, raw
results, reproduction commands and each scope. They are the entry points for
research history, rather than treating later experiments as revisions of older
measurements.

Suggested citation: steph4n-gh (2026), *System 1: teaching bounded decisions for
local reuse*, version 1.1.0. Cite the source revision and linked evidence when
quoting measurements; earlier unsupported benchmark tables have been withdrawn.
