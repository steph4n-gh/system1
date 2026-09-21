# Seven teaching corrections produce small development changes

Experiment 2p applies six proposed CLINC fitting-label corrections and one
banking correction to both System1 and the conventional baseline. **Banking
gains one accepted request per method; CLINC keeps its previous candidates.**
This does not resolve the [full observed-test failures](latest-regression.md),
establish independent confirmation or qualify the n8n recording.

The [procedure](../CORRECTION_PROTOCOL.md), runner and exact proposals were
committed in `ce753012cc7f763ca7f32cc7f6523730681e073d` before fitting. The
proposals are assistant-authored and not independently verified annotations.
They are an overlay on the original teaching rows: every text, group, order and
original fold assignment remains fixed. No calibration, development or test
label changes. The historical proposal file remains unchanged; this report
records its application. These are seven corrections to existing teaching
sets, not classifiers taught from only seven examples.

| Complete development cohort | Control: accepted / correct | Corrected: accepted / correct | Corrected coverage | Corrected accepted accuracy | Raw correct: control → corrected |
|---|---:|---:|---:|---:|---:|
| CLINC System1, 2,995 supported + 100 unfamiliar | 2,610 / 2,584 | 2,609 / 2,583 | 87.11% | 99.0034% | 2,864 → 2,865 |
| CLINC TF-IDF/logistic, same cohort | 1,770 / 1,762 | 1,723 / 1,715 | 57.53% | 99.54% | 2,720 → 2,720 |
| Banking System1, 1,960 requests | 1,581 / 1,566 | 1,582 / 1,567 | 80.71% | 99.05% | 1,787 → 1,787 |
| Banking TF-IDF/logistic, same cohort | 1,015 / 1,005 | 1,016 / 1,006 | 51.84% | 99.02% | 1,699 → 1,702 |

Every CLINC condition falsely accepts 1/100 unfamiliar requests. Banking has no
native unfamiliar cohort. Accepted accuracy above counts supported requests;
CLINC's unfamiliar acceptance is an additional error. The baseline still fails
the 80% coverage target on both workloads. All intervals and per-category
counts are retained in the runtime report; these are development point targets,
not 99% lower-confidence-bound guarantees.

The banking System1 gain comes from routing changes, with no raw classification
changes: fourteen of its fifteen accepted errors overlap the control. CLINC
System1 fixes one raw mistake but loses accepted coverage. The CLINC baseline
fixes eight raw mistakes and introduces eight; banking's baseline fixes six
and introduces three. These small, mixed changes do not show that annotation
correction explains the remaining quality failures.

## Retained control failure and repair

The [first attempt](corrected-teaching-attempt1.json) stopped at the banking
System1 control. All decisions matched, but confidence and review-score
differences reached 0.000967707 and 0.000811984, exceeding the declared <=1e-4
tolerance. No runtime measurements or selection followed that failure.

The comparison runner had used single-request features to rebuild a final head
originally taught using batched BGE features. The
[diagnostic](corrected-teaching-control-diagnostic.json) restores the original
hash-bound teaching caches and reproduces the incumbent weights, biases,
temperature and prototypes exactly. Repair commit
`4b57955ede3b88ca50d2cbbcdbb9c60f2ba980ef` restores that final teaching path
for both banking conditions. Review folds, review calibration/development and
actual serving retain single-request features; serving still uses four threads.
No tolerance, parameter or target changes.

The repaired run reuses all four completed CLINC fit records and both saved
CLINC artifacts byte-for-byte, with their original provenance, then completes
banking. All four controls reproduce their published routes; maximum review
score difference is 0.000007160. The failed checkpoint and its original source
revision remain auditable. The repair changes this experiment's reconstruction,
not the frozen product or any published incumbent.

## Actual saved-adapter runtime

All 10,110 development requests run once through the four saved adapters after
100 warmups each, under OS network denial with response caches and receipts
disabled. Every route matches development selection. There are zero runtime
errors, teacher API calls or new API costs. No first-attempt or control adapter
was retimed. Assistant-session proposal-authoring cost is not measured; prior
teacher-call costs remain in their original reports.

| Corrected adapter | Complete p50 / p95 | Artifact load | Standalone bytes | Additional encoder/tokenizer bytes |
|---|---:|---:|---:|---:|
| CLINC System1 | 2.190 / 2.863 ms | 302.23 ms | 17,372,071 | 23,492,300 |
| CLINC TF-IDF/logistic | 0.697 / 0.919 ms | 97.59 ms | 35,721,647 | 0 |
| Banking System1 | 2.846 / 4.486 ms | 360.87 ms | 10,050,155 | 133,804,886 |
| Banking TF-IDF/logistic | 0.509 / 0.697 ms | 40.18 ms | 13,219,089 | 0 |

These timings include validation, real text encoding/projection, classification,
strict/learned review and response construction. HTTP/n8n transport and
first-install downloads are excluded. Import and source/network preflight take
a separate 1,120.37 ms. Environment: Apple M4 Pro, macOS 27 arm64, Python 3.13.5;
exact package versions are recorded. The baseline is faster but accepts less
work. Incumbent timings are historical references, not paired speed comparisons.

| Corrected fit | Intent-fit stage | Review fit | Total preparation and fitting |
|---|---:|---:|---:|
| CLINC System1 | 1.668 s | 37.91 ms | 7.810 s |
| CLINC baseline | 10.422 s | 19.50 ms | 30.733 s |
| Banking System1 | 2.069 s | 6.45 ms | 6.308 s |
| Banking baseline | 4.021 s | 9.59 ms | 12.272 s |

Feature caches were populated already. These are not first-time teaching costs.
Total preparation includes original-fold heads, review construction, saving and
scalar checks; the banking intent stage also restores the original batch cache
and teaching encoder. CLINC retains 12,019 original fitting rows; its baseline
also uses the existing 1,696 generated positives. Banking retains 6,026 original
and 924 generated lessons. Review settings and negative sources match the
specified incumbent methods. No encoder weights are updated.

## Evidence and selection

The [development report](corrected-teaching-development.json) retains all eight
fits, original groups/folds, seven overlay changes per method pair, control
checks, source hashes and provenance. The
[runtime report](corrected-teaching-runtime.json) retains all responses,
latencies, accepted mistakes, metrics, intervals and selections. The
[audit](corrected-teaching-audit.json) verifies all 10,110 runtime records and
10,110 reconstructed control records, all four complete artifacts, 147 frozen
files, 17 current sources and 15 historical source identities. The
[offline replay](corrected-teaching-portability.json) matches 100 actual
responses per System1 adapter with SciPy/sklearn imports blocked; each also
reviews two changed contracts and rejects six malformed requests.

Four standalone bundles are published under `artifacts/`:

- `clinc150-system1-corrected`: `53da8338f56b6853d468e48174c4cc7ece5bd9e6bffe877bf3bec50adaa01dab`.
- `clinc150-baseline-corrected`: `fd26ac820b43716521cd8b6fe6bb4a58527e1ae4137eca5a8c660da08161d032`.
- `banking77-system1-corrected`: `f1ec08c2d5859d29b09686176a7d29df1059dbd195597b2cbe77b34e4da9aaae`.
- `banking77-baseline-corrected`: `de400c9077a68b02d620a9867efa06b39988dacd9b18ba9dcf6631e6e82251db`.

The declared higher-coverage rule selects both corrected banking candidates as
development incumbents; the conventional candidate still fails coverage.
Both stronger CLINC incumbents remain selected. Original tests and human
reserves were not opened or scored. All original full-workload targets and
fresh-confirmation requirements remain. The n8n workflow remains the explicitly
unqualified rehearsal; the qualified teacher-disconnection recording is still
unfinished. PR #3 stays open, with no version bump, merge or release.
