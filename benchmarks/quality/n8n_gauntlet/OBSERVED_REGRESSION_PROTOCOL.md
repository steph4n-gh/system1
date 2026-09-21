# Observed-test regression audit: experiment 2d

This audit answers whether the saved follow-up candidates resolve the failures
on the original cohorts. **Those cohorts were already evaluated in experiment
1. This is not an independent test, qualification, or permission to announce a
successful gauntlet.** Even if every point target passes here, full-scope fresh
confirmation and the qualified recording remain outstanding.

Freeze the current artifacts and evaluator in Git before this run. No fitting,
threshold selection, label correction, example removal or API calls occur:

- CLINC System1: `artifacts/clinc-review-consistent`, experiment 2b.
- Banking System1: `artifacts/banking-review`, experiment 2a.
- Both conventional baselines: the published `*-baseline-review` manifests from
  experiment 2c and their hash-bound original intent artifacts.

Retain all original test rows: 4,500 supported CLINC requests plus 1,000 unfamiliar
ones, and 3,080 banking requests. Original data hashes, labels, categories and
point targets remain unchanged: at least 80% supported coverage, at least 99%
correct among accepted supported requests, at most 1% CLINC unfamiliar false
acceptance, and complete warm single-request p95 below 5 ms. The first failed
report remains unchanged and is separately hash-bound by this audit.

Use the existing scoring, Wilson intervals and pinned split reader. The evaluator
loads each exact saved candidate, runs 100 development warmups, then evaluates
every original test request once. Include vectorization/encoding, review and
response construction; no response cache or receipts. Record load/startup time,
runtime/hardware versions, all errors, per-intent results, accepted mistakes,
artifact/encoder bytes and zero teacher calls under verified OS network denial.
Count shared original baseline artifacts in their complete required size.

Before freezing, preflight the evaluator on the first 16 development requests per
workload, verifying responses against the already retained complete development
reports. This checks the evaluator, not quality. After the freeze is committed,
perform one full observed-test audit. Exclusive report/journal creation preserves
partial or failed runs. Any replication must use new output paths and still be
marked as observed-test evidence. No successful gate may set `qualified=true`.

Publish the full result and direct counts alongside experiment 1, including
regressions. Do not remeasure merely to improve a timing result. Do not retune in
response to this audit and describe the same tests as unseen. Further teaching
would require another declared development experiment and independent full-scope
confirmation. Neither reserved human OOS cases nor human paraphrases are opened.
