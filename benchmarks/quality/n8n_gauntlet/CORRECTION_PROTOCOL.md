# Seven fitting-only teaching corrections: experiment 2p

Commit this procedure, implementation and exact proposal file before fitting.
Apply only the seven assistant-proposed corrections retained after experiment
2o: six original CLINC fitting labels and one original banking fitting label.
These are unverified teaching proposals, not independent annotations. Preserve
the historical proposal file and all original cohort files; record their use as
an overlay in this experiment. Never change an evaluation label to match a model.

## Fixed data and methods

Retain every original fitting text, group, source index and row order. Change
only the seven labels in the proposed overlay. All calibration and development
texts/labels, all 150+OOS CLINC categories and 77 banking categories stay fixed.
The original tests and human reserves remain unopened and unscored. Keep the
original three review-fold assignments by group, even where a corrected label
changes its class. Each fold model still excludes its held-out fold entirely.

For each of the four current method/workload selections, fit an unchanged-label
control and a corrected-label condition through the same runner. Preserve the
existing teaching material and selected review method:

| Method | Final intent teaching | Review teaching and settings |
|---|---|---|
| System1 CLINC | 12,019 original rows; fixed MiniLM, numerical regularization .1, strict alpha .05 | Original three-fold correctness + 3,020 calibration + 2,000 Wikipedia negatives; seven standardized signals, predicted-category indicator and existing 384 input features; review C=.1 |
| Baseline CLINC | Original rows + existing 1,696 synthetic supported lessons; TF-IDF/logistic C=10 | Original three-fold correctness + calibration + existing 674 synthetic unfamiliar negatives; seven standardized signals + predicted category, review C=1 |
| System1 banking | 6,026 original rows + existing 924 synthetic lessons; fixed BGE/four threads, numerical regularization .01, strict alpha .075 | Original three-fold correctness + 1,960 calibration; 35 standardized numerical terms + predicted category, review C=.1 |
| Baseline banking | Same original and 924 synthetic rows; TF-IDF/logistic C=10 | Same review rows/feature expansion as System1 banking, review C=.1 |

All lexical heads use word unigram/bigram TF-IDF, sublinear TF and max_iter=1000.
Fit each temporary fold's vocabulary and IDF on its two original fitting folds.
No generated example fits a temporary fold head or prototype. Final prototypes
use the respective final intent-teaching rows. Corrected fitting labels affect
intent weights, prototype grouping and fitting-fold correctness targets; they
do not alter calibration truth or any generated lesson. Supported correctness
is a positive review target; mistakes and OOS are negative. No new negative
source, review feature, encoder update, parameter search or teacher API call.

Text features are unchanged by a label correction, so reuse individual-request
feature caches keyed to the original text records; record that reuse. Rebuild
the numerical heads and review from the declared labels. Review class indicators
and CLINC System1 input features remain unscaled, as in the selected controls.

Before advancing each corrected fit, require the unchanged-label control to
reproduce every published development suggestion, prediction set (System1),
accepted/review route and category, with maximum review-score and confidence
differences <=1e-4. Checkpoint the control before asserting. Retain a mismatch
as a failure and do not select a replacement from that failed comparison.
Controls use their existing published artifact identities and are not retimed;
retain all newly reconstructed control outcomes and fitting evidence.

## Saved adapter and selection

Select a corrected candidate's review threshold only on the unchanged development
cohort using the existing maximum-coverage frontier and midpoint procedure:
>=99% correctness among accepted supported requests and <=1% CLINC unfamiliar
false acceptance. Strict ambiguity and OOS suggestions still require review.
The complete development pass also requires >=80% supported coverage, p95 <5 ms
and no runtime errors. Compare with the incumbent list in
`results/human-only-runtime.json`; a replacement must have higher coverage at
the same quality/latency targets and identical saved-runtime routing. Ties keep
the incumbent. Keep stronger baselines and all prior failed experiments visible.

Save the four corrected adapters as standalone, hash-bound artifacts. System1
retains its existing S1M/NumPy inference and fixed external encoder; baseline
artifacts retain their vocabulary, IDF, weights and sparse prototypes. Request
validation and review behavior remain the existing adapter contract.

After 100 warmups per adapter, measure every original development request once:
3,095 CLINC requests per method and 1,960 banking requests per method, totaling
10,110 complete requests. Deny networking at OS level, disable response caches
and receipts, and include actual text projection, prediction, strict/learned
review and response construction. Record every outcome, accepted error,
per-category count, Wilson interval, p50/p95, startup/load and fitting times,
artifact/encoder bytes, environment and teacher API calls/cost. Assistant-session
proposal-authoring cost is unmeasured. No first-install or HTTP/n8n latency is
claimed. Checkpoint measurements before comparison; never retime for nicer results.

Verify all 147 frozen files and committed sources before fitting/measurement.
Audit all records, overlay membership and artifact bytes; replay 100 requests
per saved System1 candidate with fitting imports blocked, including changed
contracts and malformed inputs. Outputs are exclusive; retain failures.

This remains development on already observed cohorts, not independent
confirmation. The full original failed scope, original test labels and targets,
fresh full-scope confirmation requirement and exact qualified n8n recording
remain unchanged. PR #3 stays open; no release or qualification claim.

After committing, from the repository root:

```sh
sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python benchmarks/quality/n8n_gauntlet/corrected_teaching.py fit --folder .system1/n8n-gauntlet/corrected-teaching-development
sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python benchmarks/quality/n8n_gauntlet/corrected_teaching.py measure --folder .system1/n8n-gauntlet/corrected-teaching-development
```
