# Review teaching: development experiment 2a

Declare this bounded experiment before fitting its review heads. Experiment 1
and its failed official result remain unchanged. This experiment does not score
an official test or qualify a takeover. The full 150/77-category objective and
80% coverage, 99% accepted accuracy, 1% CLINC unfamiliar acceptance and sub-5-ms
complete-adapter p95 targets remain unchanged.

## Hypothesis and fixed scope

The saved intent head already supports fast, useful classification. A taught
review decision may reject more errors if it sees additional unsupported inputs
and learns which predicted categories are harder. Compare these small review
heads without changing the selected intent heads, encoders or strict conformal
settings. No encoder fine-tuning, ensemble or new product dependency is added.

Use the exact experiment-1 MiniLM CLINC head (regularization .1, alpha .05) and
BGE-small banking head (regularization .01, alpha .075). Fitting/calibration/
development rows remain those in the original pinned manifest. Banking's final
head retains its 924 recorded generated lessons. No new live teacher call occurs.

Teach review using three deterministic, per-label/hash folds of original fitting
rows. Each fold's head and nearest-example features exclude that fold and all
generated lessons. Add review examples from original calibration predictions
using the saved final head. The target is true only for a **correct supported**
prediction; an unfamiliar request is always false, even when predicted as OOS.

## Additional negative data and reserve

Use the pinned `binary_wiki_aug.json` SHA-256
`0e059889d4771dd50463f4332374d8de8389c5828cb4d85b754b4ce8339b8544`
from CLINC commit `828f8093932c8fe6ca7936c3d2e52903b1c523de`.
Use only its training rows labeled `oos`. Exclude every normalized group from
all original folds of both workloads, and every additional OOS-plus group.
Deduplicate, sort groups by hash, and take the first 2,000. These encyclopedia
sentences are declared unsupported teaching data, not a realistic replacement
for difficult unfamiliar user requests. Compare with **zero** additional rows.

Reserve every group in `data_oos_plus.json` not present in the original folds
(151 groups in the prior source audit). Pin that source SHA-256
`bfcca9ae515623541dc1983c94c4ed7cae9d26b42ae47d74b972e51bb6f7a21f`.
The preparation script may read them only for group exclusion and saving the
reserve. No encoder/model sees them, and no outcomes are scored. They do not
provide independent supported-input confirmation across either taxonomy.

The preparation step verifies original split hashes and writes a separate data
manifest before fitting. Original official test rows participate only in the
already-declared normalized-group exclusion, never fitting or selection.

## Fixed review-head comparison

Use the existing seven reliability features: top probability log odds, top-two
margin, entropy, nearest own-class similarity and its rival gap, and mean top-five
own-class similarity and its rival gap. Compare those alone with those plus a
one-hot predicted category. Fit `StandardScaler` on the seven numerical review
teaching features, leave one-hot category values unscaled, and fit logistic
regression with C in `.01, .1, 1`, maximum 1,000 iterations. Reject unconverged
fits. For CLINC compare zero and 2,000 extra negatives; banking adds none.
This is 12 CLINC and six banking configurations, with no adaptive grid extension.

Measure each review score on all original development rows, with the fixed
strict System1 decision as an eligibility requirement and OOS predictions always
reviewed. Select thresholds only from development outcomes. Report the same
quality frontiers as experiment 1, including every failure and performance at
80% coverage. These are optimistic development frontiers, not independent tests.
If selecting a candidate for subsequent runtime work, maximize supported coverage
at both quality targets, then prefer the earliest configuration in this order:
zero extra negatives first, shared features before category features, C ascending.

Only a development improvement justifies measuring a saved complete adapter.
Do not measure cached vectors as runtime latency. Include all runtime guards and
encoder cost in any subsequent complete-adapter measurement.

## Qualification remains separate

Keep the original official results intact and do not rerun them during this
selection. A follow-up qualification requires a separately declared, frozen
candidate and independently established full-scope confirmation data. Finding
or collecting that supported-input confirmation remains unfinished. Neither
Wikipedia negatives, reserved OOS-only cases, nor synthetic teacher agreement
can stand in for the full original objective. No qualified recording follows
from this development experiment alone.
