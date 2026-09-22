# Real-document correction workflow: frozen protocol

Recorded on 2026-09-22, before generating any classifier predictions for this
experiment. One run, one correction round, no parameter search or gate changes.

## Question and data

Does the public `TeachingSession` correction → assessment → adoption workflow
improve routing of unfamiliar documents, and does it preserve an approved skill
when a candidate is ineligible?

Use all five categories of the BBC news corpus: business, entertainment,
politics, sport and tech. These are real published news articles, not customer
office documents. The [original UCD source](http://mlg.ucd.ie/datasets/bbc.html)
was unavailable at preparation time. Use the 2,225-row CSV from this
[pinned mirror](https://github.com/codehax41/BBC-Text-Classification/blob/9a64d059d04fe11489692d7214ef4561e30c4673/bbc-text.csv),
SHA-256 `fdaee0f7451cd8db2709d00e992886fe1c387ee332b8c7b7c1554ed3d3e3382e`.
The mirror's text is already lowercased. Do not redistribute article text in
the result report. Original article rights remain with the BBC.

## Preparation and five separate roles

Use only the `text` column as model input; category/path/row IDs never become
features. Normalize whitespace and clip each input to 8,192 characters.
Before splitting, group exact normalized duplicates and near duplicates by
five-word shingles: Jaccard similarity at least 0.5 **or** containment at least
0.8. Keep one stable representative per connected component; discard components
with conflicting labels. This limits duplicated-story leakage, but cannot
establish that every related news event is separated.

For each class, sort representatives by SHA-256 of
`20260922:split:<input digest>`, then take, in order:

| Role | Per class | Purpose |
|---|---:|---|
| Initial lessons | 20 | Fit the first skill |
| Calibration | 40 | Estimate uncertainty; fixed throughout |
| Adoption checks | 40 | Recurring `evaluate` examples; fixed throughout |
| Final holdout | 80 | Separate, balanced, previously unscored documents |
| Feedback pool | remainder | Simulated human corrections / extra lessons |

Stop if deduplication leaves too few representatives. Save the split manifest
and protocol hash before fitting. The holdout must never enter a teaching
session. It is fresh to this experiment, not a new corpus or prospective traffic.

## Fixed workflow and control

1. Use unchanged `TeachingSession` defaults: TF-IDF up to 1,024 features, ridge
   regularization 0.1, strict decisions at alpha 0.05, no inference caches.
2. Assess with unchanged default gates: raw accuracy ≥80%, acceptance ≥50%,
   zero accepted errors and zero regressions on the 200 adoption checks.
   Adopt the initial skill only if eligible. Otherwise retain the failure;
   do not bypass the gate to manufacture a working incumbent.
3. Order feedback by SHA-256 of `20260922:feedback:<input digest>` and inspect
   the first 400. Record the published label for the first at most 100 cases
   that the initial skill gets wrong **or** sends to review. These are simulated
   human-supplied lessons, not self-labels and not a live user study. Disclose
   all inspected labels, selected lessons and actual corrected mistakes.
4. Assess once with the same calibration, adoption checks and gates. Verify
   that edits and assessment preserve the current artifact, stale adoption is
   rejected, failed adoption preserves current, and successful adoption installs
   the exact assessed bytes. Reopen the session and check prediction parity.
5. As a control, start from the identical initial session and add the same
   number of hash-ordered feedback examples without targeting mistakes/review.
   This matches added lessons, not label-inspection cost. Assess/adopt using
   exactly the same policy. No model or hyperparameter search.
6. Freeze all artifact hashes and adoption outcomes before scoring the final
   holdout. Score initial, corrected and ordinary-addition candidates together,
   including rejected candidates. Report which artifact actually serves.

## Measurements and interpretation

Report raw correct/total, accepted correct/accepted, accepted errors, acceptance
coverage, reviews, per-class results, and paired improvements/regressions on
the same holdout. Include Wilson 95% intervals for raw and accepted accuracy,
and a paired bootstrap interval for the change in raw accuracy and correct
accepted decisions per incoming document (5,000 draws, seed 20260922).

An independent workload target is ≥95% accepted correctness at ≥80% coverage;
show both point results and whether the accepted-correctness Wilson lower bound
reaches 95%. These targets do not change the helper's adoption gates. A passing
development gate is not production qualification. A rejected improvement is a
workflow-policy finding; it must not be presented as an adopted improvement.

Record local assessment/adoption time, prediction latency including text
encoding, artifact size, runtime versions, exact source/protocol/script hashes,
split IDs and frozen per-case outcomes. No LLM, teacher API, new production
dependency or source-code tuning during this experiment. Holdout results are
opened once after freezing; later runs are reproduction, not fresh evidence.
