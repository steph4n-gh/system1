# Review feature interactions: development experiment 2f

Commit this protocol and implementation before fitting. The completed boundary
teaching round did not win the declared System1 selection. It improved CLINC raw
accuracy, with a coverage/accepted-accuracy tradeoff, and reduced banking coverage.
This experiment asks whether interactions among existing review signals improve
the accept/review decision without changing the intent classifier or encoder.
The full 150/77 scope and every original target remain unchanged.

## Fixed comparison

Use the exact saved System1 intent heads/prototypes from `clinc-review-consistent`
and `banking-review`. Use the original C=10 TF-IDF intent heads/prototypes from
`clinc150-baseline` and `banking77-baseline`, whose intent fitting labels match
those System1 heads: original fitting rows, plus the earlier 924 banking lessons.
Never refit or change these final intent heads, encoders, vocabularies or IDFs.
All CLINC individual-request features must match runtime projection; banking also
uses individual-request features with its saved four-thread encoder settings.

Reconstruct the same three per-label/hash original-fitting review folds. Each
temporary intent head and its prototypes exclude its held-out fold and all
generated lessons, whose prompts contain original fitting examples. Use System1
logistic regularization .1 for CLINC / .01 for banking, or a fresh per-fold TF-IDF
vectorizer and C=10 logistic head. Add original calibration correctness examples
from the unchanged final classifier. Correct supported decisions are positive;
all mistakes and unfamiliar inputs are negative. Add the same 2,000 pinned
Wikipedia review negatives to both CLINC methods. Banking adds no negatives.
No experiment-2e generated lessons enter this particular comparison.

Compare exactly two review representations per method/workload:

1. The existing seven numerical signals: top-probability log odds, probability
   margin, entropy, own-class nearest/top-five cosine similarity and their
   margins against the nearest competing category.
2. Those same seven signals plus all 28 products `x[i]*x[j]` for `i <= j`.

Fit a separate StandardScaler on each representation's review-teaching rows,
then append the unscaled one-hot predicted category. Fit logistic review C=.1,
max_iter=1000 in all eight cases. This includes a fresh linear control for each
quadratic condition. No hyperparameter grid, new feature family, second intent
model, encoder adaptation or API calls. Save only numerical review parameters;
reference hash-bound original parent artifacts instead of copying unchanged heads.
System1 inference remains one feature expansion, scaling, dot product and sigmoid
after its existing signals. Teaching uses existing optional research dependencies.

Preserve System1 strict alpha .05 (CLINC) / .075 (banking); OOS guesses always go
to review. The conventional baseline keeps its own eligibility. Select each
condition's threshold using the existing development frontier: maximum supported
coverage with at least 99% accepted supported accuracy and at most 1% CLINC OOS
false acceptance. Do not relabel, remove examples or retune using original tests.

Compare with the retained best development incumbents as additional choices:
System1's existing 2b CLINC and 2a banking, **the stronger 2e CLINC baseline**
(`clinc150-baseline-positive-and-unsupported`), and the 2c banking baseline. The
stronger baseline has access to the recorded 2e lessons that its own development
selection retained; publish that difference. Do not silently substitute a weaker
baseline. Ties keep an incumbent, then prefer the linear control. New winners
must preserve the quality targets, runtime equivalence and sub-5-ms complete p95.

## Verification and interpretation

Preflight feature expansion and save/load scores against scikit-learn on authored
numerical data; these checks must not open benchmark tests or reserve data.
Retain every attempted fit and full development outcome, including failures.
Reload all eight candidates, run 100 development warmups, then measure every
development input once with OS networking denied, no response cache and no
receipts. Compare saved runtime routing/suggestions with selection on every row;
retain any mismatch. Record accepted errors, per-intent metrics, count/denominator,
Wilson intervals, complete p50/p95, load/import time, fitting/feature time, parent
and additional artifact sizes, hardware/packages and zero new teacher calls.
Do not remeasure merely to improve a timing result. Check System1's review runtime
can load without scikit-learn/SciPy imports, as for its current numerical head.

The first official failure, observed regression and all earlier teaching results
remain unchanged. Neither original tests nor reserved human sources are scored
here. Another development pass cannot qualify the skill or authorize the qualified
n8n recording; independent full-scope confirmation is still missing. Keep PR #3
open and do not release, lower targets or replace the scope with an easier task.
