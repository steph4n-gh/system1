# Matched baseline review teaching: development experiment 2c

Commit this protocol before fitting. Give the conventional TF-IDF/logistic
baseline the same review-teaching information and policy search as System1's
experiment 2a. System1's saved 2b CLINC and 2a banking candidates remain unchanged.
This is development evidence, not qualification; retain experiment 1's failed
official results and all original targets and categories.

Keep each selected original baseline's TF-IDF vocabulary, IDF, C=10 intent
weights, fitting examples and label order. Final banking prototypes include the
same 924 recorded Gemini lessons. Change only review teaching and its development
threshold. Preserve the original saved baselines and their results.

Use three original-fitting folds assigned exactly as in experiment 2a: for each
label, sort by normalized group hash and assign positions modulo three. Fit a
separate TF-IDF bigram/sublinear vectorizer and C=10 logistic classifier on each
fold's other two parts. Held-out rows and all generated lessons must be excluded
from that fold's vocabulary, IDF, weights and nearest-example prototypes. Reject
unconverged fits at 1,000 iterations. These temporary heads provide review examples
and never replace the saved final intent classifier.

Combine these out-of-fold review examples with the original calibration fold's
predictions from the frozen final baseline. Correct supported predictions are
positive; all mistakes and all unfamiliar requests are negative. Use the same
seven reliability features as System1, with cosine similarities computed in the
baseline's own TF-IDF space. Fit the numerical scaler only on review teaching
examples. No development labels fit the scaler, intent head or review head.

Compare the same 18 configurations: zero or 2,000 additional pinned Wikipedia
negatives for CLINC (banking adds none); seven numerical features alone or plus
an unscaled one-hot predicted category; logistic review C=.01, .1, 1. Use exactly
`review-negative-fit.json` and verify its hash against `review-data-manifest.json`.
Those extra examples teach only review, as in System1's experiment 2a/2b. No new
API calls occur. Neither official tests nor the reserved human OOS/paraphrase
data is opened by fitting, selection or timing.

Run each policy on every original development row. OOS predictions always go to
review; do not impose System1-specific conformal eligibility on this conventional
baseline. Use the existing unchanged 99% supported accepted accuracy / 1% CLINC
unfamiliar acceptance frontier. Select maximal supported coverage, breaking ties
by retaining the original saved baseline first, then zero negatives, shared
features, and ascending review C. The original saved baseline is an incumbent
with its original threshold, so extra teaching cannot silently replace it with
a worse development choice. Retain all 18 new configurations plus two incumbents,
including quality at 80% coverage and failures. These remain optimistic
development selections, not independent confirmation.

Save selected candidates separately with bound source/artifact hashes, sizes,
teaching times and all selected development outcomes. Verify save/reload routing
against selection for every development request. Then, in a separate process
with OS networking denied, load the artifacts and measure the complete adapter
on individual requests after 100 warmups. Include text vectorization, review,
contract validation and response construction; no response cache or API calls.
Publish p50/p95, load time, counts and Wilson intervals alongside the already
retained System1 development results. Do not change a policy after timing.

Full-scope independent confirmation remains missing. This comparison cannot
authorize the qualified n8n recording or replace any original quality target.
