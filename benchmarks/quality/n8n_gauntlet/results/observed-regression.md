# Observed-test regression: quality still fails

**The saved follow-up candidates do not meet all original targets.** This is a regression audit on the previously observed original tests, not an independent qualification. The [protocol](../OBSERVED_REGRESSION_PROTOCOL.md), evaluator and 125-file freeze were committed and pushed in `9a8e90f590db02ba4a39fd0c5dcec5a05ac59739` before scoring. No fitting, threshold selection, new teaching calls or label changes occurred.

All 5,500 CLINC and 3,080 banking requests were scored once per method, with 100 development warmups. The [complete report](observed-regression.json) retains all 17,160 decisions, accepted errors, per-intent results and timings. The [arithmetic audit](observed-regression-audit.json) independently recomputes all quality counts, Wilson intervals, gate results and timing percentiles, verifies original text/labels and checks every frozen file. It runs no models. All 109 original frozen files and the [first failed report](official-test.md) remain unchanged.

Quality cells show count, point estimate and two-sided 95% Wilson interval. Accepted supported accuracy and unfamiliar false acceptance have separate denominators. These are measured cohort results, not guarantees for future traffic.

| Workload / method | Supported coverage | Correct among accepted supported | Unfamiliar falsely accepted | Complete p95 |
|---|---|---|---|---:|
| clinc150 / system1 | 3,912/4,500 = 86.93% (85.92–87.89%) | 3,861/3,912 = 98.70% (98.29–99.01%) | 41/1,000 = 4.10% (3.04–5.51%) | 2.846 ms |
| clinc150 / baseline | 1,357/4,500 = 30.16% (28.83–31.51%) | 1,354/1,357 = 99.78% (99.35–99.92%) | 8/1,000 = 0.80% (0.41–1.57%) | 0.888 ms |
| banking77 / system1 | 2,525/3,080 = 81.98% (80.58–83.30%) | 2,493/2,525 = 98.73% (98.22–99.10%) | No native cohort | 3.635 ms |
| banking77 / baseline | 1,422/3,080 = 46.17% (44.41–47.93%) | 1,415/1,422 = 99.51% (98.99–99.76%) | No native cohort | 0.667 ms |

CLINC System1 increases coverage from 3,757 to 3,912 supported requests, but accepted supported mistakes increase from 36 to 51. Unfamiliar false acceptances improve from 59 to 41 out of 1,000; the unchanged target permits at most 10. Both supported accepted accuracy and unfamiliar rejection fail. Across supported and unfamiliar inputs together, accepted correctness is **3,861/3,953 = 97.67%**, with a 95% Wilson interval of 97.15%–98.10%.

Banking System1 increases accepted requests from 2,516 to 2,525, with accepted mistakes increasing from 31 to 32. Its 98.73% accepted accuracy still misses 99%. There is no native banking unfamiliar cohort, so this result supplies no banking unfamiliar-input guarantee.

The matched TF-IDF baselines improve coverage to 30.16% and 46.17%, while meeting the measured accuracy and CLINC unfamiliar targets. They remain faster than System1, but still fail both 80% coverage requirements. System1 raw supported accuracy is 4,281/4,500 (95.13%) and 2,850/3,080 (92.53%); baseline raw accuracy is 4,067/4,500 (90.38%) and 2,693/3,080 (87.44%). The result is a quality/coverage/latency tradeoff, not an across-the-board win.

## Runtime and teaching accounting

All four runs recorded **zero teacher calls and zero runtime errors**, with OS networking denied (permission-denied control, errno 1). No response cache or receipts were used. Timings include validation, projection, intent prediction, review and response construction; HTTP/n8n overhead is separate. The Apple M4 Pro ran MiniLM with one encoder thread and BGE-small with four; BLAS used one thread. Exact runtime versions are in the report. Ordinary desktop applications remained open; no other experiment jobs ran concurrently. Differences from earlier timing runs are not controlled speedup measurements.

| Saved candidate | Manifest SHA-256 | Required artifact bytes | Additional encoder/tokenizer bytes | Load | p50 |
|---|---|---:|---:|---:|---:|
| clinc150 / system1 | `94abc591501bf5fa9335d5e8708c86d7576c0123b2bd854d638dd029029b4dba` | 17,361,680 | 23,492,300 | 295.6 ms | 2.231 ms |
| clinc150 / baseline | `4ba07476da4405070812155e3153738d469df184c3e7ec0d56943aee63558c29` | 28,709,983 | 0 | 90.9 ms | 0.672 ms |
| banking77 / system1 | `956d19c5444e66a238645abfe72e01f2b8b18d6dff2e80d57d40e87b4b305767` | 10,049,185 | 133,804,886 | 387.1 ms | 2.743 ms |
| banking77 / baseline | `4d0cc90e08a7423e299e233e38122556723f03e45437c46dcf4cc246dba471be` | 11,482,669 | 0 | 37.3 ms | 0.487 ms |

Process imports and freeze/preflight verification took 2977.1 ms before artifact-load timing. Baseline bytes include its shared original intent files as well as the additional review manifest. Runtime libraries are not included in artifact sizes.

These exact candidates reuse the previously recorded public labels, CLINC review negatives and banking Gemini teaching. The [77-call teaching run](contrast-teacher-summary.json) generated 924 lessons in 180.3 seconds, used 43,154 tokens and had a published-price estimate of $0.05147 (actual billed cost unavailable). This regression audit adds **no teacher calls or API cost**. Teaching/compilation times remain separately recorded in [consistent CLINC features](consistent-features.md), [banking review teaching](review-teaching.md) and the [matched baseline comparison](baseline-review.md). Neither adopted encoder was fine-tuned.

## Consequence

The apparent development passes did not carry over to the larger observed cohorts. Improving batch/single-request consistency fixed a real implementation discrepancy; it did not solve generalization or establish reliable rejection. The narrow development margins are insufficient grounds for a qualified takeover claim.

Keep both original taxonomies and every quality target. Further improvement needs a separately declared teaching or review experiment using fitting/calibration/development data, followed by independent full-scope confirmation. The 151 human unfamiliar requests and the human paraphrase supplement remain reserved and unscored. Neither can supply full-scope confirmation alone. Do not retune against this report and call another score on these same tests fresh.

The [real n8n disconnection rehearsal](../DISCONNECTION_REHEARSAL.md) still demonstrates that saved local decisions survive teacher shutdown. It does not establish the missing quality result. A qualified skill and the requested qualified recording remain outstanding; PR #3 stays open.

After the pinned data/encoder setup, reproduce the observed regression with a fresh output path:

```bash
sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  python benchmarks/quality/n8n_gauntlet/observed_regression.py observed \
  --output .system1/n8n-gauntlet/observed-regression-replication.json
python benchmarks/quality/n8n_gauntlet/audit_observed_regression.py
```

Any repetition remains observed-test replication. The audit command checks the retained public result without rerunning inference. The [64-response evaluator preflight](observed-regression-preflight.json) matched prior saved development outputs exactly; it was not a quality test.
