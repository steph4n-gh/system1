# System 1: teaching bounded decisions for local reuse

**steph4n-gh · System 1 1.0.2 · 19 September 2026**
Maintainer-authored technical whitepaper; not a peer-reviewed publication.
[Source repository](https://github.com/steph4n-gh/system1)

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

## 1. Problem and scope

Applications often repeat a narrow decision: route a support request, identify an
assistant command or triage a message. Once labels and boundaries are stable,
calling a general reasoning service for every instance may be unnecessary.
System 1 teaches that decision and applies it locally. A new request need not
exactly match a prior request, because the decision head operates on shared text
features. This is limited statistical generalization, not newly acquired general
language understanding.

The product loop is **teach or observe → validate → run locally → save and reuse**.
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
Numerical telemetry can contribute features for structured state.

For explicit teaching, `SystemOneCompiler(...).compile(examples, augment=False)`
fits regularized linear heads. The implementation augments features with a bias,
computes regularized normal equations using NumPy, and retains calibration
metadata. Temperature fitting and conformal scoring use separate examples.
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

## 1.0.2 qualification follow-up

The [follow-up protocol and full results](../../benchmarks/quality/quality_round/README.md)
retain the earlier experiments above and add an explicit 95% accepted-agreement
requirement. Both its point estimate and exact lower bound must pass; three exact
bounds share the repeated-attempt confidence budget. This is opt-in and does not
change the original default policy. The adapter also exposes the compiler's
existing regularization parameter.

With regularization 0.1, a first attempt at 400 observations and a fuller available
stream, the original-label SMS skill promotes after 2,961 observations. On the
old test, 954/980 accepted decisions are correct (97.3%), improving on the earlier
897/963 (93.1%). This test is now regression evidence, not a fresh independent
experiment. Banking and assistant do not qualify under the stronger requirement.
On 40 new authored SMS probes the promoted skill gets only 27/33 accepted answers
right. Unchanged explicitly taught banking and SMS skills likewise miss the
accepted-correctness target on new authored probes. A 36-lesson routing candidate
reduces accepted errors on the older difficult cohort but misses fresh coverage
and introduces a primary-cohort error; it is not adopted. No general quality
claim follows from the older passing results.

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
do not evaluate all BANKING77/CLINC150 intents or out-of-scope detection, and
historical SMS data does not measure modern phishing. Scripted gaming gains are
separate [first-use policy evidence](../../benchmarks/quality/zero_shot/README.md),
not learned strategic competence or verified speed records.

The next quality work is to make promotion match the application's required
accepted accuracy, teach the documented confusing routing cases, and reevaluate
on fresh representative data. The existing compiler, callback and evaluation
workflow can support that work without a new provider or training framework.

Suggested citation: steph4n-gh (2026), *System 1: teaching bounded decisions for
local reuse*, version 1.0.2. Cite the source revision and linked evidence when
quoting measurements; earlier unsupported benchmark tables have been withdrawn.
