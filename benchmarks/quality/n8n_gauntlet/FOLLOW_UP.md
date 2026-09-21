# Follow-up after experiment 1

The [first held-out result](results/official-test.md) is a failed qualification,
not a new development baseline to silently overwrite. The original 150-intent
CLINC scope, 77-intent banking scope, 80% coverage, 99% accepted accuracy, 1%
CLINC unfamiliar false acceptance and sub-5-ms p95 remain required. The n8n
workflow and disconnection recording remain unfinished deliverables.

The later [experiment-2d regression audit](results/observed-regression.md) froze
the updated saved candidates before scoring the already-observed original
cohorts. It still fails: CLINC supported accepted accuracy is 98.70% with 41/1,000
unfamiliar acceptances; banking accepted accuracy is 98.73%. Both meet coverage
and latency targets. This regression evidence supersedes any impression that
the development passes below resolved quality. It is not independent confirmation.

The two concrete deficiencies are unfamiliar-input rejection and banking
distinctions. Keep the encoders fixed and first investigate better teaching of
those decisions. Another encoder, ensemble, larger framework or new product
abstraction is not justified by this result alone.

## Evidence needed before another qualification

The official tests are now observed. A later score on them is follow-up evidence,
not independent confirmation. Do not change their labels, discard ambiguous
rows, reduce the categories or select a few successful workflows as a substitute
for the full gauntlet.

A new experiment must declare its additional teaching examples, calibration and
selection procedure, and a genuinely separate confirmation source covering both
complete supported taxonomies and unfamiliar requests. Freeze the candidate and
all confirmation data before scoring. Teacher-generated examples can help teach;
teacher agreement on synthetic data must not be substituted for independently
established correctness. Independently authored/adjudicated requests may be
needed if a suitable untouched public source cannot be found.

The [source audit](results/confirmation-source-audit.json) compares normalized
groups from two pinned upstream alternatives against every original fold:

- [CLINC OOS-plus](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_oos_plus.json)
  adds 151 distinct unfamiliar groups relative to the prepared original folds.
  Its supported validation/test data and unfamiliar test data add **zero** groups.
- [CLINC Wikipedia augmentation](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/binary_wiki_aug.json)
  adds 14,359 unfamiliar training groups, but no new validation or test groups.
  These are not a fresh full-scope confirmation set or necessarily representative
  of near-boundary user requests.

The audit opened no model and scored no requests. It records URLs, file hashes,
row counts and overlap counts. After that audit, the prospectively declared
[review-teaching experiment](REVIEW_TEACHING_PROTOCOL.md) allocated 2,000
Wikipedia negatives to teaching and reserved all 151 extra human OOS groups.
That reserve remains unscored; its hashes are in [the separate data manifest](review-data-manifest.json).

The [review-teaching result](results/review-teaching.md) retained a CLINC failure
caused by batched versus single-request quantized-encoder features. The subsequent
[consistency correction](results/consistent-features.md) rebuilt CLINC teaching
features through the same individual-request operation as inference and verified
zero selection/runtime mismatches on the full development cohort. Both current
development candidates clear the targets; neither has new independent full-scope
confirmation. The original official-test failure remains authoritative.

A [further source search](CONFIRMATION_SOURCES.md) identified a human CLINC
paraphrase supplement with 2,977 normalized groups absent from the original folds.
It covers only 35 numeric categories and has one unresolved label mapping; it is
reserved unscored and does not supply the missing full-scope confirmation.

The [matched baseline review comparison](results/baseline-review.md) subsequently
gave TF-IDF the same review-teaching folds, category features and negative-data
choices. Its development coverage improved to 29.45% / 44.39%, with faster
sub-millisecond p95, but both workloads still missed the 80% coverage target.
This closes the follow-up comparison gap; it does not supply independent
confirmation for System1 or alter the original official results.

## Completed review experiments and remaining work

The [boundary-teaching round](results/boundary-development.md) completed the
proposed supported/unsupported teaching comparison, with 227 recorded Gemini
calls. It improved CLINC raw correctness but did not win the declared System1
coverage selection, and it reduced banking coverage below 80%. It substantially
improved the conventional CLINC baseline, which remains part of every comparison.

The [feature-interaction experiment](results/polynomial-review.md) then kept the
intent heads fixed and added 28 products of existing numerical review signals.
It gains just one accepted banking development request over the System1 incumbent,
with the same 15 accepted mistakes; CLINC does not improve under the declared
selection. The banking baseline improves from 44.39% to 51.79% coverage. All eight
conditions, saved parameters and 20,220 runtime outcomes are retained. Neither
result resolves the observed-test failure or supplies independent confirmation.

Further implementation should first identify useful information missing from the
current review decision, rather than assume that more examples or more feature
products will solve the gap. A bounded diagnostic can compare the already-saved
semantic and lexical predictions on development to establish whether their
disagreements identify errors while preserving enough coverage. This is a
diagnostic, not authorization to change thresholds from observed test outcomes
or add an ensemble without evidence. The missing independent full-scope
confirmation source remains a separate requirement.

Before any new run, pin the additional data and a small fixed set of review-policy
settings in a separate experiment protocol. Any new teacher call must retain its
real prompt, response, model, usage and cost; classification prompts must not
contain the expected label. Failures remain in the record. A development gain
alone cannot authorize the qualified demonstration.

The final integration must load the exact newly qualified artifact, run real
n8n requests with item identity preserved, show live-teacher status and call
counts, then disconnect that teacher and continue with local decisions plus
explicit review. The existing local-classifier shutdown test does not prove
teacher disconnection and must not be presented as doing so.

The [development disconnection rehearsal](DISCONNECTION_REHEARSAL.md) subsequently
verified the actual teacher-process shutdown, preserved local answers and review,
and retained real API usage and n8n results. The newly qualified artifact and
qualified recording are still missing; this rehearsal does not waive those steps.
