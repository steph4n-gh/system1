# Review feature interactions: a one-case System1 banking gain

**This experiment does not resolve qualification.** System1 banking accepts one additional development request with the same total of 15 accepted mistakes. CLINC System1 does not improve under the declared maximum-coverage selection. The banking conventional baseline improves more, while remaining below 80% coverage. The original held-out failure and subsequent observed-test failure remain authoritative.

The [protocol](../POLYNOMIAL_REVIEW_PROTOCOL.md) and implementation were committed in `05951d27521407bf602346e19ab181f3921c8362` before fitting. Final intent heads, encoders and prototypes were unchanged. Each method received a fresh linear review control and a review head with the same seven signals plus 28 pairwise products. All review fits used C=.1 and the same original-only review folds; both CLINC methods received the same 2,000 pinned Wikipedia negatives. No new teacher calls or encoder updates were made.

The [complete fitting report](polynomial-development.json) and [complete runtime report](polynomial-runtime.json) retain all eight conditions. Each percentage below includes a count/denominator and a two-sided 95% Wilson interval. These are selected development estimates, not error guarantees for unseen traffic.

| Workload / method / review | Supported coverage | Correct among accepted supported | Unfamiliar falsely accepted | Complete p95 |
|---|---|---|---|---:|
| clinc150 / system1 / linear-control | 2,594/2,995 = 86.61% (85.34–87.78%) | 2,569/2,594 = 99.04% (98.58–99.35%) | 1/100 = 1.00% (0.18–5.45%) | 2.701 ms |
| clinc150 / system1 / quadratic | 2,569/2,995 = 85.78% (84.48–86.98%) | 2,547/2,569 = 99.14% (98.71–99.43%) | 1/100 = 1.00% (0.18–5.45%) | 2.724 ms |
| clinc150 / baseline / linear-control | 315/2,995 = 10.52% (9.47–11.67%) | 314/315 = 99.68% (98.22–99.94%) | 1/100 = 1.00% (0.18–5.45%) | 0.778 ms |
| clinc150 / baseline / quadratic | 930/2,995 = 31.05% (29.42–32.73%) | 928/930 = 99.78% (99.22–99.94%) | 1/100 = 1.00% (0.18–5.45%) | 0.799 ms |
| banking77 / system1 / linear-control | 1,579/1,960 = 80.56% (78.75–82.25%) | 1,564/1,579 = 99.05% (98.44–99.42%) | No native cohort | 4.272 ms |
| banking77 / system1 / quadratic | 1,581/1,960 = 80.66% (78.86–82.35%) | 1,566/1,581 = 99.05% (98.44–99.42%) | No native cohort | 4.305 ms |
| banking77 / baseline / linear-control | 870/1,960 = 44.39% (42.20–46.60%) | 862/870 = 99.08% (98.20–99.53%) | No native cohort | 0.578 ms |
| banking77 / baseline / quadratic | 1,015/1,960 = 51.79% (49.57–53.99%) | 1,005/1,015 = 99.01% (98.20–99.46%) | No native cohort | 0.623 ms |

## Selection and limitations

The CLINC System1 linear control reproduces the incumbent’s 2,594 accepted requests and 25 supported mistakes. Quadratic review accepts 2,569, with 22 supported mistakes. Both falsely accept 1/100 unfamiliar requests. The declared maximum-coverage selection retains the incumbent; fewer mistakes at lower coverage do not establish a generally worse or better model.

Banking System1 quadratic review accepts 1,581/1,960 (80.66%), versus 1,580 (80.61%) for the prior incumbent and 1,579 for the new linear control. It gets 1,566 accepted requests correct; all three have 15 accepted mistakes. The one-case gain over the incumbent is 0.051 percentage points. It wins the declared selection but is too small to call a meaningful quality breakthrough. Raw intent correctness stays unchanged: 2,864/2,995 CLINC and 1,787/1,960 banking.

The banking baseline accepts 1,015/1,960 (51.79%), up from 870 (44.39%), with 10 instead of eight accepted mistakes. Accepted accuracy remains above 99%. Its new quadratic review is retained. CLINC baseline quadratic review beats this experiment’s linear control, but **the stronger boundary-taught CLINC baseline remains selected** at 1,770/2,995 (59.10%) coverage and 1,762 correct. That incumbent had the additional experiment-2e teaching lessons; the current fixed-parent comparison did not. Do not quote the weaker 31.05% result as the best conventional baseline.

All eight manifests are published byte-for-byte under `artifacts/polynomial-*`. They reference existing hash-bound parent artifacts, which contain the unchanged intent heads/prototypes. The selected new banking artifacts are:

- system1: [banking77-system1-quadratic](../artifacts/polynomial-banking77-system1-quadratic/manifest.json), SHA-256 `77e120f186cda6a9403122cb2ae478c822538a8b1feaf8e558cb520eba8d16e4`.
- baseline: [banking77-baseline-quadratic](../artifacts/polynomial-banking77-baseline-quadratic/manifest.json), SHA-256 `7a6539ca70005d2b553dfb9154e76859f621d6d1e50c4acd0baa537cffcf20ca`.

Load System1 children with `PolynomialCandidate` from `polynomial_runtime.py`, and conventional children with `PolynomialBaseline` from `polynomial_review.py`. The existing n8n rehearsal remains bound to its originally measured banking artifact; it has not silently switched to this unqualified candidate.

## Runtime and teaching accounting

All 20,220 saved-adapter decisions match their selection suggestions and accept/review routing. There were zero runtime errors, zero teacher calls, no response caching and no receipts. OS network access was denied. Each condition had 100 untimed development warmups, followed by one timed request per development row. Timings include encoding/vectorization, the intent head, review and response construction. All original development rows and labels are retained.

The [independent arithmetic/source audit](polynomial-audit.json) recomputes counts, intervals, per-intent results, accepted errors and timing percentiles; it checks all eight child/parent identities and all 125 previously frozen files. Floating-point scores from fitting and individual runtime execution differ by up to 0.00000716; no routing changes result. This is not a claim of bit-identical fit/runtime arithmetic.

| Workload / method / review | Parent plus child bytes | New child bytes | External encoder/tokenizer bytes | Load | Review fit |
|---|---:|---:|---:|---:|---:|
| clinc150 / system1 / linear-control | 17,373,437 | 11,757 | 23,492,300 | 277.4 ms | 14.6 ms |
| clinc150 / system1 / quadratic | 17,375,653 | 13,973 | 23,492,300 | 237.8 ms | 28.1 ms |
| clinc150 / baseline / linear-control | 28,709,566 | 11,196 | 0 | 79.0 ms | 15.1 ms |
| clinc150 / baseline / quadratic | 28,711,772 | 13,402 | 0 | 77.6 ms | 28.2 ms |
| banking77 / system1 / linear-control | 10,057,590 | 8,405 | 133,804,886 | 308.2 ms | 5.0 ms |
| banking77 / system1 / quadratic | 10,059,841 | 10,656 | 133,804,886 | 291.7 ms | 7.4 ms |
| banking77 / baseline / linear-control | 11,482,358 | 7,919 | 0 | 33.8 ms | 4.8 ms |
| banking77 / baseline / quadratic | 11,484,597 | 10,158 | 0 | 32.4 ms | 8.6 ms |

Review-fit times exclude construction of the review lessons and feature extraction. The full fitting report records preparation times and feature-cache use; cached features are research inputs, never cached benchmark responses. The final intent heads were reused, while disposable original-only fold heads supplied the review lessons. Runtime libraries are excluded from artifact sizes. Both banking methods retain the earlier 924 generated teaching examples in their parent intent heads; this round adds zero API calls and zero API cost. Earlier teaching costs remain in the linked reports.

Measurements used the Apple M4 Pro desktop, one BLAS thread, MiniLM with one encoder thread and BGE-small with four. Library versions and the 882.7-ms process import/preflight time are in the runtime record. Ordinary desktop apps remained open. Incumbent timings come from prior runs and are not a controlled speed comparison.

## Import repair and saved-artifact verification

The original CI run [35643732829](https://github.com/steph4n-gh/system1/actions/runs/35643732829) failed because numerical tests imported the benchmark adapter and required optional `threadpoolctl`. No measured result was discarded. Commit `f28336d` moved the four numerical functions unchanged into `polynomial_scores.py`; the audit verifies their parsed function bodies against the measured revision. Neither fitted parameters nor recorded timings were replaced. The repaired [CI run](https://github.com/steph4n-gh/system1/actions/runs/35644863601) passes all six jobs.

Seven focused tests pass locally, including comparison with scikit-learn within 1e-12, JSON parameter validation and an import check blocking optional benchmark libraries. Core CI skips the two scikit-learn oracle cases when that optional fitting dependency is absent; the independent-import and parameter-validation cases still run. The [saved-artifact portability replay](polynomial-portability.json) additionally loads all four System1 children with SciPy/scikit-learn imports blocked and OS networking denied. Each reproduces 100 recorded development responses, reviews two changed contracts, and rejects six malformed requests. This replay makes no new latency claim.

After preparing the pinned data and encoders, reproduce the declared fitting and measurement in a fresh directory:

```bash
sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  python benchmarks/quality/n8n_gauntlet/polynomial_review.py fit \
  --folder .system1/n8n-gauntlet/polynomial-reproduction
sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  python benchmarks/quality/n8n_gauntlet/polynomial_review.py measure \
  --folder .system1/n8n-gauntlet/polynomial-reproduction
python benchmarks/quality/n8n_gauntlet/audit_polynomial_review.py
sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  python benchmarks/quality/n8n_gauntlet/check_polynomial_portability.py
```

No official-test or reserved-human request was scored here. Repeated development selection can overfit, and these development passes do not overturn the [observed regression failure](observed-regression.md). Independent full-scope confirmation and a recording using the exact qualified skill remain outstanding. PR #3 stays open; no release or qualification claim is made.
