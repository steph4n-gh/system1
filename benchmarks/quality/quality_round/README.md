# 1.0.2 quality round

System 1 is not an LLM. This round uses the existing NumPy decision heads,
recorded/public labels, and a small authored teaching experiment. No language
model, GPU, model download, new runtime dependency or live teacher call is involved.

The [protocol](protocol.json), [106 fresh authored probes](cases.json) and routing
candidate were committed in `f4b3e58` before the first evaluation. The
[candidate lessons](model_routing_candidate.json) retain that exact content; they
were **not adopted** into the primary example after the results below. The
[published report](results.json) preserves predictions, errors, review decisions,
promotion reports, timings and file sizes. Earlier [public results](../workloads/README.md)
and [1.0.1 routing inputs](../results/model_routing_1_0_1.json) remain unchanged.

## Stricter automatic takeover

`PromotionPolicy(min_accepted_agreement=.95)` adds an explicit target for
agreement among accepted decisions. With statistical checking enabled, the
point estimate and exact lower bound must both meet that target. Overall
agreement and acceptance retain their existing 80% requirements. The three exact
bounds share the confidence budget, including repeated-attempt spending. The
original default policy remains available for compatibility; a 95% target is an
explicit choice. Agreement with a teacher is not independent correctness.

This round uses 2,048 features, ridge regularization 0.1, no augmentation, strict
review and the existing stream order. The first attempt waits until 400
observations, with up to 4,129 available rows. Fitting, calibration and validation
remain disjoint; unsuccessful attempts need fresh validation blocks.

| Original-label teacher | Observations consumed | Result | Last accepted agreement / exact lower bound |
|---|---:|---|---:|
| Banking, three intents | 412 | Deferred | 98.6% / 90.9% |
| Assistant, six intents | 720 | Deferred | 99.2% / 92.9% |
| SMS spam | 2,961 | Promoted | 99.0% / 96.4% |

High point agreement alone does not qualify the banking or assistant stream.
Their available samples do not establish the requested lower bound. Banking's
last validation acceptance is 90%, but its acceptance lower bound is also short
of the 80% requirement. The last reports concern the most recent evaluated
candidate, not every observation consumed afterward.

The SMS skill accepts **980/1,001** old official test cases and gets **954/980
(97.3%)** right, versus **897/963 (93.1%)** in the earlier default-policy takeover.
Its 95% Wilson interval for accepted correctness is 96.1–98.2%. It saves to 19,361
bytes and reloads with identical outputs and review behavior; the local median
was 0.54 ms on this run. Observing the local label callback and evaluating ten
candidates took 18.9 seconds. That is local replay time, not API latency or the
cost of preparing labels. The recipe changes sample count, regularization and
qualification together, so this is not an ablation attributing improvement to
one setting. The old test has already been inspected and is regression evidence.

On **40 fresh authored SMS probes**, the promoted skill accepts 33 and gets only
27/33 right (81.8%): it fails the quality target. The probes deliberately balance
20 personal messages and 20 promotions, unlike the historical dataset. This
shows why a passing historical test and promotion report are not guarantees
under changed language or class balance.

## Explicit teaching and routing experiment

Unchanged explicitly taught banking and SMS skills retain their public-test
results. The new authored probes reveal weaker behavior:

| Explicit skill / cohort | Accepted | Correct among accepted |
|---|---:|---:|
| Banking / new authored phrasing | 29/30 | 25/29 (86.2%) |
| SMS / new authored messages | 36/40 | 30/36 (83.3%) |
| Existing routing / new authored probes | 27/36 | 26/27 (96.3%) |
| Candidate routing / new authored probes | 28/36 | 28/28 (100%) |

The candidate adds 36 lessons distinguishing mechanical transformations,
ordinary explanations and actual proof/correctness work. Calibration stays fixed.
On the older 48-case difficult routing cohort it reduces accepted errors from
three to one, with acceptance falling from 44 to 39. On the original 96 cases it
introduces one accepted error where there were none. On the fresh probes it
accepts only 77.8%, missing the 80% coverage target. **The candidate is retained
for diagnosis; the primary routing lesson file stays unchanged.**

The new probes and lessons share authorship. Their novelty relative to the
previous runs does not make them independent customer evidence. All point
targets, misses and intervals are retained. There is no claim of general language
understanding, universal provider parity or modern phishing detection.

## Reproduce

From a repository checkout with System 1 installed:

```bash
python benchmarks/quality/prepare_workloads.py
python benchmarks/quality/evaluate_quality_round.py
```

The first command obtains and verifies the pinned public datasets, using the
ignored source cache when present. The second runs offline with socket
connections blocked; it writes `.system1/quality-102/report.json` and saved skills.
A deferred promotion is an experimental result, so the runner does not treat it
as an execution failure. It asserts data hashes, split separation and saved/reloaded
behavior. Runtime regression tests separately check the qualification contract.

The next data work is to collect independently reviewed, representative banking
phrasing and SMS promotions, teach on a development portion, and reserve a new
cohort before evaluating. Reusing these disclosed probes for teaching would make
them development cases, not a fresh test.
