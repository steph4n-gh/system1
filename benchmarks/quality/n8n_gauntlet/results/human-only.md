# Removing synthetic banking lessons reduces accepted coverage

Experiment 2n was [declared](../HUMAN_ONLY_PROTOCOL.md) and pushed in
`b1e6cb8e63873e325cd61c6f3ce2ec5fd080803d` before fitting. It removes the earlier
924 generated banking lessons from both final classifiers and their similarity
prototypes. **Neither original-only candidate replaces its incumbent.** Both
fall below the unchanged 80% development coverage target. The full CLINC and
banking [observed-test failures](latest-regression.md) remain unresolved.

| Full 1,960-request development cohort | Augmented control: accepted / correct | Original-only: accepted / correct | Coverage change | Raw correct: control → original-only |
|---|---:|---:|---:|---:|
| System1 | 1,581 / 1,566 | 1,567 / 1,552 | 80.66% → **79.95%** | 1,787 → 1,790 |
| Conventional TF-IDF/logistic | 1,015 / 1,005 | 971 / 962 | 51.79% → **49.54%** | 1,699 → 1,685 |

Original-only accepted accuracy is 99.04% for System1 and 99.07% for the baseline.
These are development point estimates, not population guarantees; all Wilson
intervals remain in the reports. BANKING77 has no native unfamiliar cohort here,
so unfamiliar rejection is unmeasured, not perfect. The full goal still requires
CLINC's unfamiliar false acceptance <=1% alongside its other targets.

Removing the synthetic lessons fixes 17 raw System1 mistakes and introduces 14;
for the baseline it fixes 24 and introduces 38. Fourteen of System1's fifteen
accepted mistakes are shared with the augmented control. A small raw-accuracy
gain therefore does not establish a better review decision. Under this fixed
procedure, the extra lessons modestly improve accepted coverage; the result
does not establish that generated labels are generally correct or beneficial.

## Controlled teaching and complete runtime

Both methods retain all 6,026 original fitting rows, all 1,960 calibration rows,
all 77 labels and the original three review folds. The control review is refit
from the reconstructed fold/calibration lessons and reproduces all 1,960
published control decisions per method. Control intent heads are unchanged and
their old timings are not rerun. Both new review heads use the same 35 numerical
features, predicted-category indicator and C=.1. System1 retains fixed BGE,
regularization .01 and strict alpha .075; the baseline retains TF-IDF and C=10.

All 3,920 new requests run through loaded standalone adapters under OS network
denial, after 100 warmups per method. Every route matches development selection;
there are no runtime errors, response caches, receipts, teacher calls or new
API costs. Timings include request validation, actual text projection,
classification, review and response construction, excluding HTTP/n8n transport.

| New original-only adapter | System1 | Conventional baseline |
|---|---:|---:|
| Complete p50 / p95 | 2.752 / 4.384 ms | 0.472 / 0.677 ms |
| Artifact load | 414.71 ms | 35.45 ms |
| Recorded intent-fit stage | 1.511 s | 3.683 s |
| Review logistic fit | 6.97 ms | 9.62 ms |
| Total preparation and fitting | 6.339 s | 12.688 s |
| Standalone artifact | 8,731,167 bytes | 11,268,194 bytes |
| Additional encoder/tokenizer | 133,804,886 bytes | None |

Process imports and source/network preflight take a separately recorded
1,112.77 ms. Teaching-feature caches were already populated; those teaching
times do not include first-time encoding. Total preparation includes the
three original-fold models, control-review reproduction, new head/review,
saving and scalar verification. The System1 intent-fit stage includes model
construction and calibration/development probability preparation; the baseline
stage measures its logistic fit. Neither is an end-to-end first-install time.
The conventional baseline is faster in this run but accepts much less work.
Old control timings are historical references, not a paired speed experiment.

## Retained evidence

The [development report](human-only-development.json) records the exact fitting
and calibration groups, source hashes, original folds, control reproduction,
teaching times, thresholds and all selected outcomes. The
[complete runtime report](human-only-runtime.json) retains every response,
latency, accepted mistake, per-category metric, interval and unchanged incumbent.
No original test or human reserve was opened or scored.

Both complete artifacts are published under `artifacts/`, with manifest hashes:

- `banking77-system1-human-only`:
  `d4bf2478cb582f5e6325468fe31d8136887d48e42607cae536332886ba739892`.
- `banking77-baseline-human-only`:
  `93af6b3f984581d7b66159c2b1c1239901ef778cd42d75c1d7a0e8ba262bdeb2`.

The [record audit](human-only-audit.json) verifies 3,920 new outcomes, source and
artifact identities, all 147 frozen files and eleven declared sources. It
recomputes metrics, intervals, timing and selection without model inference.
The [offline replay](human-only-portability.json) reproduces 100 System1
responses with SciPy/sklearn imports blocked, reviews two changed contracts and
rejects six malformed requests. Seven existing numerical-review tests pass.
An initial import-only preflight mistakenly imported the research regression
runner, which requires SciPy; the corrected check verifies the portable adapter
directly. No fitted result or timing was changed to fix that check.

This is a rejected development ablation, not independent confirmation. All four
incumbents stay selected. The n8n workflow still uses its explicitly unqualified
rehearsal artifact; the qualified recording is unfinished. PR #3 stays open and
no version bump or release is made.
