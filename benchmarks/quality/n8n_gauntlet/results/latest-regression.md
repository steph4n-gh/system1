# Latest saved candidates still fail the observed-test quality gates

**Neither System1 workload is qualified.** The latest selected review heads meet coverage and latency targets but miss accepted accuracy; CLINC also misses unfamiliar-input rejection. The stronger conventional baselines improve coverage, yet remain below 80%, and the newer CLINC baseline also misses unfamiliar rejection. These are the already-observed original tests, not independent confirmation.

The [experiment-2h protocol](../LATEST_REGRESSION_PROTOCOL.md), [147-file freeze](../latest-regression-freeze.json), evaluator and successful [64-response development preflight](latest-regression-preflight.json) were committed and pushed in `8d80e476e9e7261fe1f845a8bb7a3065943870d8` before this run. There was no new fitting, threshold selection, relabeling or example removal. Every original row is retained: 4,500 supported and 1,000 unfamiliar CLINC inputs, plus 3,080 banking inputs, each scored by System1 and its baseline.

The [complete 17,160-decision report](latest-regression.json) includes every original text/label, response, accepted error, per-intent result and timing. Each percentage below includes a count/denominator and two-sided 95% Wilson interval. Accuracy among accepted supported requests and unfamiliar false acceptance have separate denominators.

| Workload / method | Supported coverage | Correct among accepted supported | Unfamiliar falsely accepted | Complete p95 | Failed point gates |
|---|---|---|---|---:|---|
| clinc150 / system1 | 3,956/4,500 = 87.91% (86.93–88.83%) | 3,903/3,956 = 98.66% (98.25–98.97%) | 46/1,000 = 4.60% (3.47–6.08%) | 2.655 ms | accepted accuracy, unfamiliar rejection |
| clinc150 / baseline | 2,670/4,500 = 59.33% (57.89–60.76%) | 2,652/2,670 = 99.33% (98.94–99.57%) | 25/1,000 = 2.50% (1.70–3.66%) | 0.850 ms | coverage, unfamiliar rejection |
| banking77 / system1 | 2,524/3,080 = 81.95% (80.55–83.27%) | 2,492/2,524 = 98.73% (98.22–99.10%) | No native cohort | 3.483 ms | accepted accuracy |
| banking77 / baseline | 1,672/3,080 = 54.29% (52.52–56.04%) | 1,663/1,672 = 99.46% (98.98–99.72%) | No native cohort | 0.603 ms | coverage |

## What changed from the prior observed regression

CLINC System1 accepts 3,956 supported requests instead of 3,912: 42 additional correct decisions and two additional mistakes in aggregate. Supported accepted accuracy declines from 98.70% to 98.66%. Unfamiliar false acceptance rises from 41/1,000 to 46/1,000 (4.6%), above the unchanged limit of 10. Counting all accepted requests, including unfamiliar ones, it gets 3,903/4,002 correct (97.53%). The latest development selection did not resolve either quality deficiency.

Banking System1 accepts 2,524 requests instead of 2,525, with 2,492 correct and the same total of 32 accepted mistakes. Accepted accuracy remains 98.73%, below 99%. Its one-case development gain over the earlier review does not carry over as a gain on this cohort. Raw intent correctness is unchanged: 4,281/4,500 for CLINC and 2,850/3,080 for banking, consistent with unchanged final intent heads.

The conventional CLINC baseline now accepts 2,670 supported cases instead of 1,357, with 2,652 correct and 18 supported mistakes. Its supported accuracy is 99.33%, but coverage is only 59.33% and unfamiliar acceptance rises from 8/1,000 to 25/1,000 (2.5%). Including unfamiliar requests, accepted accuracy is 2,652/2,695 (98.40%). Banking baseline coverage improves from 1,422/3,080 (46.17%) to 1,672/3,080 (54.29%), with 1,663 correct and nine mistakes. It passes accepted accuracy but still misses coverage. The stronger baselines are retained honestly, including their new failures.

All changes above compare counts on identical original cohorts with the [earlier observed regression](observed-regression.md). Neither run is independent confirmation. Relative to the [first held-out failure](official-test.md), System1 CLINC rejects more unfamiliar requests (46 false acceptances versus 59), but still fails the 1% gate; supported accepted accuracy is now below 99%. Banking remains below 99%. No single favorable metric is substituted for the joint targets.

## Exact artifacts and measurement

- clinc150 / system1: [clinc150-system1-context](../artifacts/clinc150-system1-context/manifest.json), SHA-256 `d63b1376cbe8f47c14488d679383e35b91ec6b598f1297de60e78b654b22d45d`.
- clinc150 / baseline: [clinc150-baseline-positive-and-unsupported](../artifacts/clinc150-baseline-positive-and-unsupported/manifest.json), SHA-256 `3fc70d9b9073502e85bd1306f0e9f67eb7537c7eaa1c7e1cabe61ef27ab7ba98`.
- banking77 / system1: [polynomial-banking77-system1-quadratic](../artifacts/polynomial-banking77-system1-quadratic/manifest.json), SHA-256 `77e120f186cda6a9403122cb2ae478c822538a8b1feaf8e558cb520eba8d16e4`.
- banking77 / baseline: [polynomial-banking77-baseline-quadratic](../artifacts/polynomial-banking77-baseline-quadratic/manifest.json), SHA-256 `7a6539ca70005d2b553dfb9154e76859f621d6d1e50c4acd0baa537cffcf20ca`.

System1 uses the CLINC input-context review from 2g and banking quadratic review from 2f. The baselines use the CLINC boundary-taught artifact from 2e and banking quadratic review from 2f. Parent intent/prototype files and fixed encoders are bound in the freeze. The CLINC baseline had additional generated teaching lessons; the chosen System1 intent head did not adopt them. The source reports retain that difference.

Each candidate received 100 development warmups, then one timed decision per original test request. Encoding/vectorization, intent prediction, review and response construction are included; HTTP/n8n transport and loading are separate. OS networking was denied, there were zero teacher calls and runtime errors, and response caching/receipts were disabled. No timing run was repeated. The machine was an Apple M4 Pro with one BLAS thread, MiniLM with one encoder thread and BGE-small with four; package versions are in the full report. Differences from prior-run timings are not a controlled speedup claim.

| Workload / method | Required artifact bytes, including parent | External encoder/tokenizer bytes | Load | Warm p50 | Warm maximum |
|---|---:|---:|---:|---:|---:|
| clinc150 / system1 | 17,383,260 | 23,492,300 | 284.4 ms | 2.113 ms | 4.056 ms |
| clinc150 / baseline | 35,717,463 | 0 | 93.6 ms | 0.658 ms | 2.613 ms |
| banking77 / system1 | 10,059,841 | 133,804,886 | 331.8 ms | 2.653 ms | 7.409 ms |
| banking77 / baseline | 11,484,597 | 0 | 36.7 ms | 0.478 ms | 0.843 ms |

Full process import plus Git/hash freeze verification took 2958.9 ms before loading candidates. It is not decision latency. Runtime libraries are excluded from artifact sizes. This regression does no teaching and adds zero API calls or cost; the earlier real teacher usage and estimates remain published.

The [separate arithmetic/source audit](latest-regression-audit.json) verifies every original text, label and normalized group, recomputes metrics, Wilson intervals, per-intent results and latency percentiles, and checks artifact identities/sizes and all 147 frozen files. The first failed report and the previous observed failure remain byte-identical. All six CI jobs pass on the freeze commit.

The frozen run can be reproduced into a fresh output path, explicitly as observed-test evidence:

```bash
sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  python benchmarks/quality/n8n_gauntlet/latest_regression.py observed \
  --output .system1/n8n-gauntlet/latest-regression-reproduction.json
python benchmarks/quality/n8n_gauntlet/audit_latest_regression.py
```

## Consequence

The recent review-feature changes do not justify a success claim or a qualified recording. Repeated development selection has not translated into meeting the larger-cohort quality gates. Do not respond by quietly changing thresholds against these observed tests, reducing categories, dropping difficult rows or presenting another development pass as confirmation.

Further quality work needs a better-supported teaching or representation change, and independently established fresh confirmation remains necessary. The reserved human OOS and paraphrase sources remain unscored; neither supplies the missing full supported scope. The existing n8n rehearsal is still useful integration evidence, but the exact qualified artifact and requested recording do not yet exist. PR #3 stays open; no release is made.
