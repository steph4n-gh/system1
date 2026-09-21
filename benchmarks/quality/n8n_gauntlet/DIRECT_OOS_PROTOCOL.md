# Direct unfamiliar-input teaching: development experiment 2i

Commit this protocol and runner before fitting. The latest experiment-2h
regression still fails. This experiment tests a simpler teaching change: use the
existing OOS intent for additional unfamiliar examples. No encoder weights,
runtime format, product API or model architecture change.

All original targets and the full 150-intent CLINC plus native OOS / 77-intent
banking scope remain required. This comparison changes CLINC only. Banking's
incumbents and its failed observed accuracy remain unchanged. This cannot
complete the full goal or qualify a teacher-disconnection recording.

## Fixed comparison

Fit two conditions for each of System1 and TF-IDF/logistic, four in total:

1. **Positive control:** original 12,019 CLINC fitting rows (including 80 native
   OOS examples), plus the 1,696 supported experiment-2e teaching examples.
2. **Direct OOS:** exactly the control rows plus the 674 retained experiment-2e
   synthetic unfamiliar examples, labeled with the already existing `oos` choice.

Those 674 labels remain unverified synthetic teaching material. This explicitly
changes the experiment-2e restriction for this new experiment; it neither
rewrites the old protocol nor treats generated labels as clean human truth.
Use every retained example with no semantic filtering, relabeling, resampling,
class weighting, threshold grid or additional generation. No teacher calls.

Use the same pinned MiniLM encoder, single-request features and one encoder
thread for System1; logistic regularization .1 and strict alpha .05. The lexical
comparison uses TF-IDF word unigrams/bigrams, sublinear TF and C=10 logistic
intent fitting. Fit its vocabulary/IDF only on that condition's fitting rows.
Both methods receive the identical teaching rows. Refit the existing intent head
and prototypes, retaining the existing saved formats and runtime loaders.

Review teaching remains the seven standardized numerical signals plus one-hot
predicted category, with C=.1 for System1 and C=1 for the baseline. Preserve the
same original per-label/hash three folds; each temporary head and prototype set
excludes its own held-out fold and **all generated examples** because of prompt
lineage. Add original calibration correctness and the pinned 2,000 Wikipedia
negative review examples. Both conditions and both methods receive that exact
review data. Wikipedia examples never enter intent fitting. None of the 674
synthetic OOS examples enters review teaching directly, avoiding in-sample
negative predictions as review evidence. The final head does determine the
calibration/Wikipedia numerical review signals, as in earlier experiments.

Reconstruct the prior System1 positive-only control's original fold groups,
development routing and metrics; require score differences below 1e-10. Its
procedure is unchanged. The lexical control now receives Wikipedia review
examples too; disclose that it is not a replay of experiment-2e's lexical control.
Use unchanged maximum-development-coverage selection subject to >=99% supported
accepted accuracy and <=1% OOS false acceptance. Always review OOS suggestions.
Compare with the selected candidates in `results/context-runtime.json`; retain
incumbents on ties, then prefer the positive control. Do not replace a candidate
unless its complete saved runtime matches selection, has no errors and p95 <5 ms.
The >=80% supported coverage target remains required for a development pass.

## Measurement and boundaries

Preserve the 147-file latest-regression freeze. Verify original split hashes,
teaching source hashes and committed procedure before fitting. No original test,
human OOS reserve or human paraphrase reserve is opened or scored. Use the
existing feature caches only for teaching; runtime executes each request anew.

Reload all four artifacts, perform 100 development warmups, and measure every
one of the 3,095 development requests once per condition under OS network denial,
without response caches or receipts. Retain complete outcomes, accepted errors,
per-intent counts, Wilson intervals, p50/p95, load/import time, fitting preparation
and head-fit time, complete artifact and external encoder sizes, environment,
zero new teacher calls and zero new API cost. Preserve partial failures and do
not repeat timings for a better number. Prior incumbent timings remain from
their published runs. Publish all four artifact identities and both System1
artifacts, plus any new winning baseline, with regeneration recipes for the rest.

Audit recorded arithmetic, rows, lineage, routing, hashes and selection without
rerunning inference. Check the saved System1 artifacts with optional fitting
imports blocked, including malformed inputs and changed contracts. These are
development outcomes, not independent confirmation. The small development OOS
cohort and repeatedly used development data cannot establish future guarantees.
All earlier failures remain published. Independent full-scope confirmation is
still missing; keep PR #3 open with no merge, release or lowered target.

## Reproduction

From the repository root, after committing the exact procedure:

```sh
sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python benchmarks/quality/n8n_gauntlet/direct_oos_teaching.py fit --folder .system1/n8n-gauntlet/direct-oos-development
sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python benchmarks/quality/n8n_gauntlet/direct_oos_teaching.py measure --folder .system1/n8n-gauntlet/direct-oos-development
```

Outputs are exclusive: use a new directory for a labeled replication rather
than overwriting this run. A replication on these rows is not fresh evidence.
