# Matched review teaching improves the conventional baseline

**Development only.** The first official gauntlet remains failed. No official
test, human OOS reserve or new human paraphrase reserve was scored in experiment
2c. The original 150/77-intent scope and all qualification targets remain intact.

The [protocol](../BASELINE_REVIEW_PROTOCOL.md) was committed and pushed in
`fec6877752c32c11b95c3ef7034fd2847e2e7fce` before fitting. Both final baseline
intent classifiers retain their original C=10 weights, vocabulary, IDF, fitting
labels and prototypes. Only the learned review policy changes. The baseline gets
the same three review-teaching folds, category features and 18 policy choices as
System1, including the option to use the same 2,000 extra CLINC negatives. Each
original saved baseline also remains eligible as an incumbent.

## Complete saved-adapter results

| Development workload and method | Accepted supported requests | Correct among accepted supported | Unfamiliar false accepts | Complete single-request p95 |
|---|---:|---:|---:|---:|
| CLINC System1, existing experiment 2b | 2,594/2,995 (86.61%) | 2,569/2,594 (99.04%) | 1/100 | 2.69 ms |
| CLINC matched TF-IDF baseline | 882/2,995 (29.45%) | 881/882 (99.89%) | 1/100 | 0.88 ms |
| Banking System1, existing experiment 2a | 1,580/1,960 (80.61%) | 1,565/1,580 (99.05%) | No native cohort | 4.24 ms |
| Banking matched TF-IDF baseline | 870/1,960 (44.39%) | 862/870 (99.08%) | No native cohort | 0.54 ms |

The conventional baseline is faster but fails the 80% coverage target on both
workloads. Its selected review heads improve coverage from 22.10% to 29.45% on
CLINC and from 42.04% to 44.39% on banking. Both use predicted-category features:
review C=1 for CLINC and C=.1 for banking. The selected CLINC policy uses no extra
Wikipedia negatives; all six corresponding extra-negative alternatives are
retained, including their lower coverage. Raw supported accuracy is unchanged at
2,692/2,995 for CLINC and 1,699/1,960 for banking.

These are separate development timing runs on the same Apple M4 Pro, not a paired
API benchmark or fresh quality confirmation. System1's numbers come from the
already retained [CLINC](consistent-review-runtime-development.json) and
[banking](review-runtime-development.json) runs. No System1 policy or result was
changed here. Neither method's selected development result establishes general
accuracy. In particular, one unfamiliar acceptance out of 100 has a 95% Wilson
interval of approximately 0.18%–5.45%.

## Evidence and accounting

- [All 20 configuration entries and selected outcomes](baseline-review-development.json):
  18 new policies plus the two original incumbents; all 5,055 selected development
  decisions match the saved adapter with no routing mismatches. Numerical review
  features see 15,039 CLINC examples without extra negatives, 17,039 with them,
  and 7,986 banking examples. Fold heads exclude their own rows and all generated
  lessons; the final banking intent head retains its original 924 Gemini lessons.
- [Complete runtime report](baseline-review-runtime.json): 100 warmups per
  workload, fresh text vectorization for every request, no response cache, no
  receipts, zero teacher calls, and OS network denial verified with EPERM.
  P50 is 0.657 ms for CLINC and 0.449 ms for banking. Artifact loads take 86.5 and
  39.2 ms; numerical-module imports take another 593.4 ms in this research process.
- [Published-artifact audit](baseline-review-audit.json): all 5,055 responses
  replay identically from the public files with networking denied. Each workload
  reviews both changed contracts, rejects six malformed inputs and rejects two
  artifact mutations. All 109 original frozen files remain unchanged.
  A separate end-to-end refit reproduced both manifests as parsed JSON objects,
  all 20 policy frontiers and all 5,055 decisions and scores. JSON key order differs
  from the initial layout conversion, so refit byte hashes and returned artifact
  IDs differ; the published artifacts retain their measured identities.
- Review development took 22.67 seconds for CLINC and 10.61 seconds for banking,
  including fold fitting, feature extraction, policy selection, saving and
  replay. These are not isolated head-fitting times. Fold/individual gate times,
  runtime versions, thresholds and Wilson intervals are retained in the reports.
  No new API teaching cost was incurred.

The [CLINC review manifest](../artifacts/clinc150-baseline-review/manifest.json)
is 11,613 bytes; the [banking manifest](../artifacts/banking77-baseline-review/manifest.json)
is 8,230 bytes. They contain numerical review parameters and bind their original
baseline manifests by SHA-256. They reuse the existing intent files, so publishing
this comparison does not duplicate approximately 40 MB of unchanged artifacts.
Including those required files, the complete baselines occupy 28,709,983 and
11,482,669 bytes respectively. They need no separate neural encoder.

An initial serialization copied whole baseline directories and replaced only
the review coefficients inside the weights archive. Its [timing summary](baseline-review-initial-layout-runtime.json)
is retained. The compact representation was verified against all 5,055 initial
decisions and scores before timing it again; only artifact identities changed.
No refitting or threshold adjustment was involved in that storage change.

## Reproduce

Prepare the pinned original folds and review negatives as documented in the
experiment README. Install the research dependencies (NumPy 2.5.3, SciPy 1.18.1,
scikit-learn 1.9.1 and threadpoolctl 3.7.0 in the recorded run). From the repository
root on macOS, use a fresh output folder:

```bash
sandbox-exec -p '(version 1) (allow default) (deny network*)' python benchmarks/quality/n8n_gauntlet/baseline_review.py fit --folder .system1/n8n-gauntlet/my-baseline-review
sandbox-exec -p '(version 1) (allow default) (deny network*)' python benchmarks/quality/n8n_gauntlet/baseline_review.py measure --folder .system1/n8n-gauntlet/my-baseline-review
```

The fit command refuses an existing run directory; measurement refuses an
existing runtime report. The original baseline artifacts and source hashes must
still match experiment 1's freeze. To load published artifacts without fitting,
use `baseline_review.ReviewBaseline` with either review-manifest directory and
retain its referenced original baseline directory alongside it. These are
experimental benchmark adapters, not new default product behavior.
