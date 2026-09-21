# n8n decision takeover experiment

**First held-out evaluation failed; not qualified for takeover.** The goal is to teach a bounded
classification skill, independently qualify it, connect that exact saved skill
to self-hosted n8n, and record the teacher being disconnected. This directory
does not yet contain a passing final evaluation or an integration recording.

The latest [frozen regression check](results/latest-regression.md), after the
boundary, interaction and input-feature experiments, still fails. CLINC System1
accepts 3,956/4,500 supported requests, gets 3,903/3,956 correct (98.66%), and falsely
accepts 46/1,000 unfamiliar requests (4.6%). Banking gets 2,492/2,524 accepted
requests correct (98.73%) at 81.95% coverage. Complete p95 is 2.65/3.48 ms. The
stronger baselines accept 59.33%/54.29%, below 80%, and the CLINC baseline also
fails unfamiliar rejection. All 17,160 outcomes are retained. These are already
observed original cohorts, not fresh qualification.

The subsequent [direct OOS teaching comparison](results/direct-oos.md) also
retains the incumbents. Teaching 674 synthetic unfamiliar examples to the existing
OOS intent improves raw OOS recognition but reduces supported correctness and
coverage. All 12,380 development decisions are retained, with no new teacher calls
and no routing differences after saving. A report-writer failure after measurement
was recovered from the original journal without repeating timings. This does not
resolve either full-workload failure or supply independent confirmation.

The subsequent [semantic-plus-word head comparison](results/fused-heads.md) has
mixed results: raw CLINC correctness improves from 2,864 to 2,889 out of 2,995,
while banking falls from 1,787 to 1,773 out of 1,960. All 10,110 development
outcomes and four saved heads are retained. Only CLINC advances to a separately
declared full review comparison; simple confidence/margin policies still fail
80% coverage at the quality targets. No current adapter is replaced and no new
complete-adapter latency or qualification is claimed by that head probe.

The [completed fused-review follow-up](results/fused-review.md) fails coverage:
2,375/2,995 supported requests accepted (79.30%), with 2,367 correct (99.66%)
and 1/100 unfamiliar false acceptance. Complete p95 is 2.80 ms. All 3,095 saved
adapter responses match selection, with zero teacher calls or runtime errors.
The higher raw head accuracy does not improve useful coverage. All incumbents
remain; banking quality and independent full-scope confirmation are unresolved.

The next [taught nonlinear feature comparison](results/shallow-heads.md) also
fails to advance. A 128-unit feature transform leaves CLINC raw correctness
unchanged at 2,864/2,995 and reduces banking from 1,787 to 1,782 out of 1,960.
Its best simple-threshold coverage is 75.03% / 63.83%, below 80%. All 5,055 new
outcomes, both saved bundles and fixed-budget teaching warnings are retained.
No encoder weights, core runtime or current candidate changed; there is no new
adapter timing or qualification claim.

The [fixed 12-layer MiniLM comparison](results/minilm12.md) also fails its
declared screen. CLINC gains 20 raw correct decisions but its best simple
confidence/margin coverage falls to 70.18%; banking loses 15 raw correct decisions
while that frontier rises to 68.88%. All 5,055 outcomes and exact heads are
retained. Saved-head p95 is 2.69 / 4.39 ms, excluding learned review and adapter
validation. Those numbers are not complete-adapter qualification. Existing
candidates remain selected and the full failures remain unresolved.

The [completed original-only banking comparison](results/human-only.md) removes
the 924 earlier synthetic lessons from both methods and reduces accepted coverage:
System1 moves from 80.66% to 79.95%, and the baseline from 51.79% to 49.54%.
System1 gains three raw correct decisions but loses fourteen accepted correct
decisions. Both candidates fail 80% coverage; all incumbents remain. Complete
p95 is 4.38 / 0.68 ms, with all 3,920 routes matching selection, zero runtime
errors and zero teacher calls. Exact artifacts and offline replay are retained.

The [completed teaching-consistency diagnostic](results/teaching-consistency.md)
checks all 18,045 original fitting examples with predictions made outside their
teaching folds. Both classifiers disagree with the supplied label in 121 CLINC
and 182 banking cases; these are review candidates, not verified label errors.
The source audit verifies all 28,080 fitting/calibration/development rows with
zero import mismatches. It also inspects all 42 current accepted development
errors, showing both annotation concerns and genuine decision weaknesses.
Original labels, final classifiers and full-workload targets stay fixed.

The next [declared correction comparison](CORRECTION_PROTOCOL.md) tests only the
seven proposed fitting-label changes in both methods, with every calibration
and development label unchanged. It must reproduce the published controls before
evaluating corrected candidates. No quality improvement is claimed before it runs.

The [earlier regression failure](results/observed-regression.md) and original
held-out failure remain unchanged. Development gains below have not resolved
the full quality objective.

The next [declared boundary-teaching round](results/boundary-teaching.md) records
227 real Gemini calls and 3,287 retained teaching examples across the full scope.
Generated labels remain unverified teaching material. The
[completed fixed comparison](results/boundary-development.md) retains both
existing System1 candidates: new CLINC lessons improve raw correctness but reduce
coverage at the quality thresholds, and banking falls below 80% coverage. The
conventional CLINC baseline improves from 29.45% to 59.10% coverage, still short of
80%. All 16,300 runtime decisions match selection. This does not supply fresh
confirmation or qualification.

The subsequent [review-feature interaction experiment](results/polynomial-review.md)
retains all eight conditions and 20,220 development decisions. Its selected banking
System1 review accepts one additional request (1,581/1,960), with the same 15
accepted mistakes; this is not a meaningful quality breakthrough. The banking
baseline improves to 51.79% coverage. Both CLINC incumbents remain selected,
including the stronger boundary-taught baseline at 59.10%. No original test or
reserved-human input was scored, and neither qualification nor the recording is
complete.

A subsequent [disagreement diagnostic](results/review-disagreement.md) also
retains a failure: requiring semantic/lexical agreement reduces banking coverage
to 76.28%, while agreement without learned review accepts 6/100 unfamiliar CLINC
requests. It reuses recorded development responses and makes no joint-latency or
qualification claim; no ensemble was added.

The earlier [input-feature review comparison](results/context-review.md) adds no
encoder or teacher call. CLINC System1 gains 16 accepted development cases, but
also one mistake: 2,584/2,610 correct (99.0038%), with 1/100 unfamiliar acceptance
and 2.78 ms p95. Banking ties its incumbent; stronger baseline incumbents remain.
All four artifacts and 10,110 runtime decisions are retained. This modest
selection gain does not resolve the observed-test failure or establish fresh
qualification.

The [protocol](PROTOCOL.md) and [source/split manifest](manifest.json) were
committed in `803650c` before any official-test scoring. All 150 CLINC intents
and all 77 BANKING77 intents remain in scope. Required targets are 80% local
acceptance, 99% correctness among accepted requests, at most 1% CLINC out-of-scope
false acceptance, and complete local-adapter p95 below 5 ms. None has been lowered.

The candidates and evaluator were frozen and pushed in `41a0d36` before the first
official run. The [complete result](results/official-test.md) is retained:

| Official test / System1 | Supported accepted | Correct among accepted supported | OOS falsely accepted | Complete p95 | Failed gate |
|---|---:|---:|---:|---:|---|
| CLINC150 + OOS | 3,757/4,500 (83.5%) | 3,721/3,757 (99.04%) | 59/1,000 (5.9%) | 1.82 ms | OOS rejection |
| BANKING77 | 2,516/3,080 (81.7%) | 2,485/2,516 (98.77%) | No native OOS cohort | 4.39 ms | Accepted accuracy |

The conventional baseline is faster (0.83/0.71 ms p95) but accepts only
23.4%/43.4% of supported requests. Every official row, accepted error, per-intent
result and Wilson interval is in the [machine-readable report](results/official-test.json).
Inference ran with OS networking denied and made zero teacher calls. Neither
System1 candidate qualifies. A follow-up needs a declared experiment and fresh
confirmation; these observed tests cannot be reused as independent evidence.
The [follow-up requirements and source audit](FOLLOW_UP.md) keep that next
experiment separate from this failure.

The [later review-teaching experiments](results/consistent-features.md) improve
development coverage to 86.6% on CLINC and 80.6% on banking at the quality targets,
with complete p95 of 2.69/4.24 ms. They also found and corrected a mismatch between
batched quantized-MiniLM teaching features and single-request inference. The
corrected CLINC candidate matches selection on all 3,095 development requests.
These results do not supersede the failed official test or establish qualification;
fresh full-scope confirmation remains outstanding.

## What development has established

The original small lexical classifier does not meet this broader task. Simply
raising its confidence threshold loses most local coverage. Conventional
TF-IDF/logistic regression also misses the targets. A fixed local encoder helps;
cross-entropy fitting is substantially better than ridge on those features.
This motivated the optional Python compiler's `choice_solver="logistic"` method.
It needs SciPy during teaching and retains the existing saved-head format and
NumPy inference. Ridge remains the default. See the
[teaching guide](../../../docs/guides/training_experts.md).

Both **original development** candidates cleared the targets through the complete local
adapter, including the saved head, strict System1 review, fixed encoder and an
additional learned/distribution guard:

| Development workload | Supported accepted | Correct among accepted | OOS falsely accepted | Complete p95 |
|---|---:|---:|---:|---:|
| CLINC, fixed MiniLM, one encoder thread | 2,512/2,995 (83.9%) | 2,495/2,512 (99.32%) | 1/100 | 1.84 ms |
| BANKING77, fixed BGE-small, four encoder threads | 1,575/1,960 (80.4%) | 1,560/1,575 (99.05%) | No native OOS cohort | 4.38 ms |

Banking's margin is narrow. Its original reliability check reached 77.6% coverage;
adding 924 generated contrast examples reached 78.9%. Three-fold predictions on
original fitting rows then supplied independent correctness examples for the
review model. Each fold excluded its own rows and all generated examples, whose
prompt lineage could otherwise leak the withheld examples. The review model was
fitted on those correctness examples plus the reserved calibration fold. Its
regularization and strict-review settings were selected on development data.
The one- and two-thread banking runs failed latency (6.23 and 5.36 ms p95); their
results remain published. No encoder weights were updated for these candidates.
The exploratory encoder-adaptation run reaches 1,530/1,960 accepted (78.1%) with
1,515/1,530 correct (99.0%). That adaptation updates pretrained weights; it is
research evidence, not a requirement for ordinary System1 teaching or an adopted
integration path. The main route continues with a fixed encoder and taught head.

These are development results: thresholds/configurations were
selected using development outcomes. They are **not** held-out qualifications,
statistical guarantees or claims that a standard System1 install achieves these
numbers. The subsequent official-test failures above take precedence over this
development evidence. The [candidate freeze and evaluation recipe](FREEZE.md)
bound the exact saved candidates, conventional baselines and source files before
the first test run.
The strengthened TF-IDF baseline selects calibration-taught review models as
well as probability and margin policies. Its best development coverage is 22.1%
on CLINC and 42.0% on banking at the quality thresholds, still below 80%.

The subsequent [matched review-teaching comparison](results/baseline-review.md)
gives the baseline the same additional review information and 18 policy choices
as System1's follow-up. Coverage improves to 29.45% / 44.39%, with 0.88 / 0.54 ms
p95, while both baselines still miss the 80% coverage target. All configurations,
saved review parameters, complete runtime decisions and failures are retained.
These are development comparisons, not new official-test results.

The [real n8n development workflow](N8N.md) already uses the exact saved CLINC
candidate. It preserves item identity and explicit review, including when the
local service is stopped. This is integration evidence, not the final qualified
teacher-disconnection recording.

The separate [development disconnection rehearsal](DISCONNECTION_REHEARSAL.md)
loads the saved banking review candidate. Two real Gemini fallback calls were
followed by stopping the teacher process: ten subsequent n8n decisions completed
with zero new provider attempts, preserving local answers and explicit reviews.
This validates integration plumbing, not full-scope quality or teacher accuracy.

Retained results:

- [Initial feasibility failure](results/initial-feasibility.json), with its
  [original probe](results/initial-feasibility.py).
- [Lexical heads and conventional baselines](results/lexical-development.json).
- Fixed [MiniLM](results/minilm-development.json) and
  [BGE-small](results/bge-small-development.json) encoder probes.
- [Combined lexical/encoder features](results/hybrid-development.json).
- [Distance rejection with the actual System1 logistic compiler](results/gating-development.json).
- [Nonlinear feature probe](results/kernel-development.json),
  [nearby-example agreement](results/agreement-development.json), and
  [calibration-taught reliability](results/reliability-development.json).
- [MPNet-base](results/mpnet-base-development.json): the larger encoder does not
  solve banking and its encoder-only p95 exceeds 5 ms on both workloads.
- Exploratory encoder adaptation on [banking](results/adapted-banking77.json) and
  [CLINC](results/adapted-clinc150.json), including every measured epoch.
- [Teacher connectivity](results/teacher-connectivity.json): one actual Jev and
  one actual Gemini request. These are connectivity checks, **not** teacher
  distillation or measured n8n executions.
- [Generated contrast lessons](results/contrast-lessons.json),
  [complete teacher requests/responses](results/contrast-teacher.json),
  [usage and cost estimate](results/contrast-teacher-summary.json), and
  [matched development comparisons](results/contrast-development.json).
- [Request-length reliability follow-up](results/contrast-length-development.json):
  adding a word-count feature did not close the quality gap.
- [Out-of-fold reliability](results/crossfit-development.json) and its
  [regularization follow-up](results/crossfit-refined-development.json).
- [Linear/prototype combination](results/prototype-development.json): retained
  failure, not adopted by the runtime candidates.
- Complete saved-adapter runs for [CLINC](results/runtime-development.json) and
  banking with [one](results/bank-runtime-development-1.json),
  [two](results/bank-runtime-development-2.json) and
  [four](results/bank-runtime-development-4.json) encoder threads.
- OS network-blocked replay of all [3,095 CLINC](results/runtime-isolation.json)
  and [1,960 banking](results/bank-runtime-isolation.json) development decisions.
- [Frozen conventional baseline selection](results/baseline-development.json),
  with all 20 attempted policies and complete saved-adapter development outcomes.

Original examples come from the pinned fitting, calibration or development
folds; the additional generated lessons are identified separately. Fitted feature vectors may be cached to avoid repeated preprocessing in
research runs; recorded single-input encoder timings execute the encoder again.
There is no test-response cache. Head-fit timings exclude initial feature
extraction. Exploratory encoder timings exclude the System1 engine and adapter,
and some ran alongside other development processes; they cannot prove the latency
gate. Final timings follow the protocol without other experiment jobs; ordinary
desktop applications remain open.

## Reproduce development

From a checkout with System1 installed:

```bash
python benchmarks/quality/n8n_gauntlet/prepare.py
python benchmarks/quality/n8n_gauntlet/develop.py
```

The lexical comparison additionally uses scikit-learn and threadpoolctl. For the
optional encoder experiments, install ONNX Runtime and tokenizers in a research
environment, and obtain the pinned files with the Hugging Face `hf` CLI:

```bash
hf download sentence-transformers/all-MiniLM-L6-v2 tokenizer.json tokenizer_config.json config.json README.md 1_Pooling/config.json onnx/model_qint8_arm64.onnx --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --local-dir .system1/n8n-gauntlet/minilm
hf download BAAI/bge-small-en-v1.5 tokenizer.json tokenizer_config.json config.json README.md 1_Pooling/config.json onnx/model.onnx --revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a --local-dir .system1/n8n-gauntlet/bge-small
python benchmarks/quality/n8n_gauntlet/encoder_probe.py minilm
python benchmarks/quality/n8n_gauntlet/encoder_probe.py bge-small
python benchmarks/quality/n8n_gauntlet/hybrid_probe.py
python benchmarks/quality/n8n_gauntlet/gating_probe.py
python benchmarks/quality/n8n_gauntlet/kernel_probe.py
python benchmarks/quality/n8n_gauntlet/agreement_probe.py
python benchmarks/quality/n8n_gauntlet/reliability_probe.py
python benchmarks/quality/n8n_gauntlet/runtime_probe.py
```

`encoder_probe.py --extended` retains a separate report for additional C=100/1000
head fits. All these runners open only the three development folds. `prepare.py`
uses test text solely to exclude normalized duplicates and preserves every
official test row.

MiniLM's ONNX file is 23,026,053 bytes; BGE-small's is 133,093,490 bytes, before
tokenizers, runtime libraries and taught heads. These externally pretrained
encoders are optional research dependencies, not System1's built-in hashed
features. They run locally without text generation. Their pretraining corpora
have not been independently audited for benchmark overlap. Upstream model cards:
[MiniLM (Apache-2.0)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
[BGE-small (MIT)](https://huggingface.co/BAAI/bge-small-en-v1.5).

The adaptation probes additionally used PyTorch 2.14.0 and Transformers 4.57.6,
with the matching pinned MiniLM `model.safetensors` downloaded into the same
directory. Commands are `python benchmarks/quality/n8n_gauntlet/adapt_encoder.py
banking77` and the corresponding `clinc150` run. They use only fitting labels for
gradient updates; calibration remains separate. These slower experiments must
not be described as instant teaching or as proof that language-model fine-tuning
is required. No pretrained encoder weights or credentials are committed. The
selected decision heads, review guards and conventional baselines are retained
under `artifacts` for exact evaluation without refitting.

## Recorded contrast teaching

The [contrast-example procedure](TEACHING_ADDENDUM.md) was committed in `679470a`
before its first API call. Gemini generated 924 lessons in 77 requests, using only
fitting examples and category names. No generated lesson overlapped a reserved
normalized group. This is synthetic augmentation, not blind teacher classification
or a passing result. For example, its disposable-card lessons explicitly mention
temporary or single-use numbers, while ordinary virtual-card lessons ask how to
obtain a virtual card. Those labels come from generation instructions, not a human
audit of all generated examples.

Generation took 180.3 seconds and used 25,644 input and 17,510 output tokens. The
published standard-price estimate is $0.05147 using Gemini 2.5 Flash's $0.30 per
million input and $2.50 per million output tokens, checked September 21, 2026.
Actual account tier and billed cost are unavailable. See
[Google's pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-2.5-flash).

```bash
# Replay published teacher responses; no API credential or new calls needed.
python benchmarks/quality/n8n_gauntlet/teach_contrasts.py
python benchmarks/quality/n8n_gauntlet/contrast_probe.py
python benchmarks/quality/n8n_gauntlet/contrast_probe.py --length-feature
python benchmarks/quality/n8n_gauntlet/crossfit_probe.py
python benchmarks/quality/n8n_gauntlet/crossfit_probe.py --refine
python benchmarks/quality/n8n_gauntlet/prototype_probe.py
python benchmarks/quality/n8n_gauntlet/bank_runtime_probe.py --threads 4
```

The first comparison wrote all eight configurations and then aborted during native
library shutdown; its [complete result and failure](results/contrast-development-first.json)
are preserved. Releasing encoder sessions before threadpool teardown produced a
clean run with exactly identical quality results. Offline replay also reproduced
the lesson file byte-for-byte while socket connections were blocked. These checks
do not replace the required final saved-skill isolation and n8n workflow tests.
