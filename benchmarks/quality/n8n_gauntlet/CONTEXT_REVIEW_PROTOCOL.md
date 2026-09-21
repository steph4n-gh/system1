# Input-aware review: development experiment 2g

Commit this protocol, implementation and numerical preflight before fitting.
The previous feature-interaction experiment gains only one accepted banking
development case. A simple semantic/lexical agreement veto drops banking below
80% coverage. This experiment instead lets the existing review head use the
input features already computed by its own classifier. It adds no second encoder,
intent classifier, teacher call, new teaching source or encoder update.

## Fixed data and comparison

Retain all 150 CLINC intents, its unfamiliar inputs and all 77 banking intents.
Use the same exact parent intent heads, prototypes, encoders, vocabularies, IDFs
and original-only review folds as experiment 2f. Reuse its `prepare` procedure:
three per-label/hash original-fitting folds, with each temporary classifier and
prototype set excluding its held-out fold and all generated lessons; add the
original calibration correctness examples. Both CLINC methods receive the same
2,000 pinned Wikipedia negatives; banking receives none. Correct supported
predictions are positive review lessons; mistakes and unfamiliar inputs are
negative. No experiment-2e lessons enter this fixed-parent comparison. The
banking parent intent heads still contain the earlier 924 generated lessons.

Fit exactly one new review head per method/workload, four in total:

- Standardize the original seven numerical review signals using only the
  review-teaching rows, as in the linear control.
- Append the unscaled one-hot predicted category.
- Append the input's existing unit-normalized feature vector, unscaled:
  384-dimensional frozen encoder features for System1, or the baseline's sparse
  TF-IDF vector using its saved final vocabulary and IDF. Both methods receive
  their own existing input representation. The lexical review representation
  therefore uses a vocabulary fitted on the complete parent fitting text, while
  its correctness targets still come from classifiers excluding the held-out
  fold. No held-out labels enter the per-fold intent head.
- Fit logistic review with C=.1 and max_iter=1000, with no grid or other feature
  interactions. Save numerical weights and reference the hash-bound parent.

Every System1 input vector must use the same single-request encoder operation
and thread settings as runtime. The baseline reuses its existing sparse vector;
no dense vocabulary-sized vector is constructed at runtime. Inference adds one
dot product to the current review logit, and keeps the existing response format.
No new product dependency or core API is introduced.

Reproduce the four experiment-2f linear controls' development scores from their
published parameters and reconstructed features before each new fit. Require
the same fold groups, row counts, negative-target counts and parent identities;
retain any disagreement and stop rather than silently refit a different control.
Compare new candidates with those controls and every selected incumbent from
`results/polynomial-runtime.json`, including the stronger boundary-taught CLINC
baseline and both selected banking quadratic reviews. The CLINC baseline
incumbent's additional 2e teaching data remains disclosed.

Keep strict System1 alpha .05 / .075 and always review OOS suggestions. Each
condition uses the unchanged development frontier: maximum supported coverage
at at least 99% accepted supported accuracy and at most 1% CLINC unfamiliar false
acceptance. Ties retain the incumbent. A new development winner must also match
saved-runtime routing, produce no runtime errors and have complete p95 below
5 ms. The 80% coverage target remains required for a development pass. Do not
use original tests or reserved-human inputs to fit, select or validate this run.

## Measurement and interpretation

Before fitting, test JSON save/load scores against scikit-learn on authored dense
and sparse numerical examples and reject invalid context parameters. Numerical
helpers must import without optional fitting or benchmark dependencies.

Reload all four new candidates, perform 100 development warmups, then measure
each development input once under OS network denial with no response cache or
receipts. Record every outcome, accepted error, per-intent count, Wilson interval,
complete p50/p95, load/import time, teaching preparation/fit time, parent and child
artifact sizes, external encoder sizes, hardware/packages and zero teacher calls
and new API costs. Preserve failures and partial output. Do not repeat a timing
run to improve the result. Published incumbent timings remain from earlier runs.
Replay saved System1 artifacts with SciPy/scikit-learn imports blocked; check
changed contracts, malformed inputs and matching recorded responses.

The original held-out failure, observed-test regression failure and all earlier
results remain unchanged. This is another development experiment and cannot
qualify the skill or authorize the qualified n8n recording. Independent full-scope
confirmation is still missing. Keep PR #3 open with no release or lowered target.
