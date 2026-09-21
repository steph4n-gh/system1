# Fused review misses the coverage target

Experiment 2k completes the review and saved-adapter check permitted by the
[fused-head comparison](fused-heads.md). Its [fixed procedure](../FUSED_REVIEW_PROTOCOL.md)
was committed and pushed in `29e229acb69ba7a8a1cd1e5fcea9290c823279c8` before
fitting. **Coverage is 79.30%, below the unchanged 80% target. All four incumbent
selections remain.** Neither workload qualifies; the
[full observed regression failure](latest-regression.md) remains authoritative.

| CLINC150 development measurement | Result | Target |
|---|---:|---:|
| Supported accepted | 2,375/2,995 (79.30%) | At least 80% — failed |
| Correct among accepted supported | 2,367/2,375 (99.66%) | At least 99% |
| Unfamiliar falsely accepted | 1/100 (1%) | At most 1% |
| Complete warm adapter p50 / p95 | 2.163 / 2.796 ms | p95 below 5 ms |

The adapter measured all 3,095 development requests once after 100 warmups.
OS networking was denied, response caching and receipts were disabled, and
there were zero runtime errors or selection/routing mismatches. It made zero
teacher calls and incurred zero new API cost. The [runtime record](fused-review-runtime.json)
retains every outcome, all nine accepted mistakes (eight supported and one OOS),
per-intent results, Wilson intervals, environment and exact artifact identity.
These are development-selected point estimates, not independent guarantees.

Raw supported correctness remains 2,889/2,995, identical to the fixed fused
intent head. Better raw classification did not produce better useful coverage:
the incumbent accepts 2,610 supported development requests at the declared
quality targets, versus 2,375 here. Banking's rejected fusion and original
quality failure remain unresolved; its complete scope is still required.
The matched and stronger conventional baselines remain in the recorded
comparison. Prior candidates' timings are explicitly from their earlier runs.

## Teaching and runtime evidence

The intent head and lexical configuration are byte-identical to experiment 2j.
Only the review head is newly taught: seven existing numerical signals plus the
predicted category, using 12,019 original out-of-fold predictions, 3,020 original
calibration predictions and 2,000 pinned Wikipedia negatives. Each temporary
fold fits its vocabulary, IDF and numerical intent head only on its two fitting
folds. No synthetic lesson, original test or reserved human request is used.
The encoder stays fixed and runs once per request.

The [development report](fused-review-development.json) records 11.65 seconds
of review preparation and a 12.80 ms review fit. Semantic feature caches were
already populated; these are not cold end-to-end teaching times. The separate
runtime process records 958.96 ms for imports/network preflight and 295.46 ms
for artifact loading. Its complete warm timing includes feature extraction,
the System1 head, strict and learned review, and response construction; it
excludes HTTP/n8n transport and startup.

The exact standalone bundle is published in
`artifacts/clinc150-fused-review`: unchanged `intent.s1m`, `lexical.json`,
review/prototype `scope.npz` and their binding manifest. It totals 18,582,807 bytes,
plus the external 23,492,300-byte encoder/tokenizer. Manifest SHA-256:
`c2144c382e66453d2eb0bd6b365ed44a6b930fa522c1deb2a8d28e17f075cb48`.

The [record audit](fused-review-audit.json) checks all 147 previously frozen
files, ten declared sources, three reconstructed fold vocabularies, all 3,095
recorded outcomes and unchanged selections. Maximum selection/runtime review
score difference is 9.02e-7, with identical routing. It performs no model
inference or test access. The [portability check](fused-review-portability.json)
reproduces 100 recorded responses with SciPy and scikit-learn imports blocked
and OS networking denied, checks two changed contracts and six malformed
requests, and makes no new timing claim.

The next requirement is a justified improvement to teaching or representation,
followed by full-scope independent confirmation. Another threshold chosen from
these failures cannot supply that confirmation. The existing n8n demonstration
is still an unqualified rehearsal; a qualified saved skill and recording remain
outstanding. PR #3 remains open, with no release or lowered target.
