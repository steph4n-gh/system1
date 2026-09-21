# Remove synthetic banking lessons: experiment 2n

Commit this procedure and implementation before fitting. The comparison changes
one teaching-data choice: remove the earlier 924 generated BANKING77 lessons
from the final intent head and its similarity prototypes. Keep all 6,026
original fitting examples, 1,960 calibration examples, 1,960 development
examples and all 77 categories. Apply the same removal to the conventional
baseline. Generated labels are not verified human truth. Earlier raw results
(1,786 original-only versus 1,787 augmented) came from different experiments;
they motivate this ablation but do not establish its result.

## Fixed comparison

Fit one original-only System1 candidate: the same pinned 384-dimensional BGE
encoder with four CPU threads, individual-request features, logistic numerical
head regularization .01, original calibration and strict alpha .075. Fit one
original-only conventional baseline: word unigram/bigram TF-IDF, sublinear TF,
logistic C=10 and max_iter=1000. Fit vocabulary and IDF on original fitting rows
only. Neither encoder weights nor System1 runtime APIs change.

For both methods keep the existing quadratic review: seven numerical signals,
their 28 products, and one-hot predicted category. Standardize the 35 numerical
features only; fit review logistic C=.1, max_iter=1000. No parameter search.
Use the exact original per-label/hash three-fold correctness lessons, plus
correctness on the original calibration fold from the respective final head.
Fold heads and prototypes exclude their held-out fitting fold and every
generated lesson. There are 7,986 review targets and no new negative source.

Reuse `polynomial_review.prepare` to reconstruct the original-only fold review
lessons. Its augmented-parent calibration and development signals are used only
to verify that refitting the unchanged quadratic control review reproduces the
published 2f outcomes. Retain the first 6,026 out-of-fold review rows; replace
the remaining calibration signals and correctness targets with those from the
new original-only head. Recompute development signals and strict eligibility.
The published augmented intent heads remain fixed; controls are not retimed.

For each new candidate, select the maximum development coverage at >=99%
accepted supported accuracy using the unchanged midpoint threshold procedure.
Compare against the full incumbent list in `results/fused-review-runtime.json`.
Select a replacement only with higher accepted supported coverage, the same
quality targets, no runtime errors or selection/runtime routing mismatches,
and complete warm single-request p95 <5 ms. Ties keep the incumbent. Coverage
>=80% remains mandatory for a development pass. There is no raw-accuracy proxy
gate: this experiment measures the actual reviewed adapter.

## Evidence and limits

Save each complete standalone artifact and hash-bound manifest, with the exact
teaching groups and source hashes in the report. System1 saves its numerical
head, review arrays and fitting prototypes; the baseline saves its vocabulary,
IDF, intent/review weights and fitting prototypes. Reconstruct projection at
runtime; teaching-vector caches must not serve requests.

Verify the 147 frozen files and committed sources before fitting and measuring.
Under OS network denial, load each saved artifact, warm up with the first 100
development requests, then measure all 1,960 requests once. Include validation,
projection, prediction, strict/learned review and response construction. Disable
response caches and receipts. Record all 3,920 outcomes, errors, accepted
mistakes, per-category counts, Wilson intervals, p50/p95, startup/load and
teaching times, artifact/encoder bytes, environment and zero new calls/cost.
Checkpoint measured results before comparison; never rerun for nicer timing.
Replay 100 System1 responses with SciPy/sklearn imports blocked, and check
changed contracts and malformed requests. Audit all recorded results and files.

This is development, not independent confirmation. No original test or reserved
human example is opened or scored. CLINC150 and its native unfamiliar examples
remain in the full goal with the current failed results unchanged. Banking's
latest observed-test failure remains evidence regardless of this comparison.
The original full gauntlet, fresh full-scope confirmation and a recording using
the exact qualified n8n skill are still required. PR #3 stays open; no release.

After committing, from the repository root:

```sh
sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python benchmarks/quality/n8n_gauntlet/human_only.py fit --folder .system1/n8n-gauntlet/human-only-development
sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python benchmarks/quality/n8n_gauntlet/human_only.py measure --folder .system1/n8n-gauntlet/human-only-development
```

Outputs are exclusive. Later reproductions require a new directory and must
identify reused development data as replication, not fresh confirmation.
