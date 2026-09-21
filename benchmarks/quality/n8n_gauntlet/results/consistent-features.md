# Consistent CLINC teaching features: development experiment 2b

**The feature mismatch is fixed; qualification remains unproven.** The
[prospective protocol](../CONSISTENT_FEATURES_PROTOCOL.md) was committed and
pushed in `a517756` before fitting. It declared one fixed configuration, not an
expanded search. Encoder weights, categories, source labels, group boundaries,
head regularization .1, strict alpha .05, review C=.1 and the selected 2,000
unsupported teaching examples were unchanged.

The correction uses the encoder's individual-request operation when preparing
all CLINC teaching, calibration and selection features. Its cache identity is
different from batched features. The intent and review heads were rebuilt from
those consistent features; the pretrained encoder was not fine-tuned.

The saved complete adapter matches selection's predicted category and review
decision for **all 3,095 development requests**, with zero mismatches. It runs
with OS networking denied, receipts disabled and no response cache. Each timed
request executes the encoder, decision head, strict uncertainty check and learned
review guard.

| Development measurement | Result | Two-sided 95% Wilson interval |
|---|---:|---:|
| Supported coverage | 2,594/2,995 = 86.61% | 85.34%–87.78% |
| Correct among accepted supported | 2,569/2,594 = 99.04% | 98.58%–99.35% |
| Unfamiliar falsely accepted | 1/100 = 1.00% | 0.18%–5.45% |
| Complete single-request p50 / p95 | 2.09 / 2.69 ms | Not a proportion |

The [teaching/selection record](consistent-review-teaching-development.json) and
[complete runtime outcomes](consistent-review-runtime-development.json) retain
all counts, errors, timings and hashes. These are still selected development
results. In particular, one error among only 100 unfamiliar requests does not
establish a reliable 1% future error rate; experiment 1 already demonstrated that
limitation. No official test or reserved human OOS request was evaluated again.

Single-request feature preparation took about 20.4 seconds across 20,134 distinct
fitting, calibration, development and additional-negative rows. The repeated
original-fitting lookup was a separate cache hit. Head/cross-fit work and review
feature preparation took 6.28 seconds; fitting the small review head took 14.7 ms.
Those components are recorded separately rather than described as an instant
end-to-end teaching time. No new API calls or API charges were incurred.

The [saved CLINC development artifact](../artifacts/clinc-review-consistent/manifest.json)
has manifest SHA-256
`94abc591501bf5fa9335d5e8708c86d7576c0123b2bd854d638dd029029b4dba`.
Its head, review prototypes/weights and manifest total 17,361,680 bytes. The
unchanged encoder/tokenizer add 23,492,300 bytes, before runtime libraries.
Artifact loading took 261 ms in this run. The measured hardware is the same
Apple M4 Pro desktop used in the earlier experiments.

Banking keeps the experiment-2a [saved review candidate](../artifacts/banking-review/manifest.json):
1,580/1,960 development requests accepted, 1,565 correct and 4.24 ms p95. Its
manifest SHA-256 is
`956d19c5444e66a238645abfe72e01f2b8b18d6dff2e80d57d40e87b4b305767`.
That small development improvement is not a new held-out banking result.

Reproduce this CLINC comparison after the original pinned dataset/encoder setup:

```bash
python benchmarks/quality/n8n_gauntlet/prepare_review_teaching.py
python benchmarks/quality/n8n_gauntlet/review_teaching.py --consistent-clinc
sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  python benchmarks/quality/n8n_gauntlet/review_runtime.py --consistent-clinc
```

The public saved artifacts can also be loaded with `ReviewCandidate` from
`review_runtime.py`, without repeating teaching. They are experimental adapters,
not qualified n8n replacements. The first failed official result, original
artifacts and all experiment-2a failures remain unchanged. The 151 extra human
OOS examples remain reserved. Independent full-scope confirmation and the
qualified teacher-disconnection recording are still required.

[Public artifact portability checks](review-artifact-portability.json) replayed
100 complete decisions per workload identically with SciPy/scikit-learn imports
blocked and OS networking denied. Each artifact also reviewed two changed
contracts and rejected six malformed requests. The teaching dependencies are not
needed to load and use these saved review candidates.
