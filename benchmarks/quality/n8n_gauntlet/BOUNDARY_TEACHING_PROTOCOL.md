# Boundary teaching: experiment 2e

Declare this procedure before request preparation, API calls or development fits.
The first official failure and experiment-2d observed regression stay unchanged.
All 150 CLINC intents, all 77 banking intents and every original quality/coverage/
latency target remain required. This experiment is development, not qualification.

## Teaching source and bounded generation

Use the existing fixed MiniLM encoder for CLINC and fixed BGE-small for banking.
Project fitting examples individually, matching the runtime operation; never
update encoder weights. Compute a normalized centroid for each supported category
from original fitting rows only. Its three nearest supported centroids define
neighboring categories. Choose five target fitting examples with the smallest
own-centroid minus strongest-neighbor cosine margin (group hash breaks ties),
and three hash-first fitting examples from each neighbor. These are geometrically
close examples, not independently established model mistakes.

For each supported category, ask Gemini 2.5 Flash for exactly 12 varied requests
whose wording clearly distinguishes that category from its neighbors. Include
the complete category-name list to avoid treating another supported category as
unsupported. For each CLINC category also request exactly eight related but
unsupported requests, each with a short reason why none of the supported intents
fits. Rotate eight original fitting OOS examples through those prompts in hash
order as additional context. No banking OOS examples are generated: that would
introduce a new taxonomy or evaluation scope. Category names are legitimate
generation conditions; this is not blind teacher classification.

Use temperature .7, disabled thinking, JSON output and at most 4,096 output tokens.
At most one call per category: **227 calls**, at most 2,724 supported proposals
and 1,200 unsupported proposals. No retries, regeneration, model fallback or
quality-dependent extra calls. Record an exclusive attempt file before every
request; an interrupted/failed attempt consumes its slot and is never repeated.
Stop on authorization/rate-limit failure. Resume may issue only unattempted,
hash-identical requests. Secrets are read from the ignored environment file and
sent only in the API header, never prompts, URLs, logs or published evidence.

Commit the protocol/code, prepare and commit all exact requests, source hashes
and encoder identities, then make live calls. Request preparation accesses only
fitting rows for context/geometry. Calibration/development/test/reserved texts
must never appear in teacher prompts. The immutable original split hashes remain
required. Tests and reserves may contribute normalized hashes solely to duplicate
exclusion, never to features, generation, predictions or selection.

Retain raw successful responses, exact requests, model versions, usage, latency,
HTTP status/error type and every rejection. Reject malformed arrays/counts,
empty/oversized texts, missing unsupported reasons, and normalized duplicates
against every original fold of both workloads, earlier generated lessons,
Wikipedia teaching rows, reserved human OOS/paraphrases and this generated batch.
Do not repair labels manually or filter by development/test success. All retained
labels remain **unverified synthetic teaching labels**, including OOS rationales;
format validation is not semantic adjudication. Publish all attempted calls and
token usage; estimate standard-price cost from Google's published rates separately
from unavailable actual billing. Failed calls without usage have unknown cost.

## Fixed development comparison

Keep original fitting/calibration/development boundaries. Compare each incumbent
with one new positive-teaching condition; CLINC additionally has a condition
adding generated unsupported examples to review teaching. Generated supported
examples fit the intent head and join its prototypes. Generated unsupported
examples teach the review head only, like existing Wikipedia negatives; do not
fit them as if they were a clean human OOS category.

Use System1 logistic regularization .1 for CLINC and .01 for banking. Banking
retains its earlier 924 Gemini lessons. Rebuild the same three per-label/hash
review folds on original fitting examples, excluding all generated examples and
their prompt lineage from fold heads and prototypes. Add original calibration
correctness examples. Use the same seven numerical review features plus predicted
category, C=.1, and existing 2,000 CLINC Wikipedia negatives. Preserve strict
alpha .05 (CLINC) / .075 (banking). No new encoder, feature family or hyperparameter
search. Generated examples do not become evaluation labels.

The conventional TF-IDF/logistic comparison receives the identical new fitting
and review data. Keep its C=10 intent fit and its selected review configuration:
predicted-category features, C=1 for CLINC / .1 for banking, no Wikipedia negatives
in the positive-only control; include the same generated negatives in the CLINC
negative condition. Refit TF-IDF only on the declared augmented fitting rows.
Retain every condition and the incumbent; a worse result does not replace it.

Use the existing development frontier: maximize supported coverage subject to
99% supported accepted accuracy and at most 1% CLINC unfamiliar false acceptance.
Ties prefer the incumbent, then positive-only teaching. Record raw quality,
selected thresholds, complete outcomes, all errors and Wilson intervals. This
small development OOS cohort cannot establish future 1% rejection performance.
Freeze/save any selected candidate and verify complete runtime decisions with
OS network denial, no response cache and 100 development warmups, reporting
complete p50/p95, load/import costs and full artifact/encoder sizes separately.

No original-test or reserved-human scoring occurs in this experiment. A gain
still requires independent full-scope confirmation before any qualified takeover
recording. That source has not yet been obtained. The existing n8n rehearsal
continues to be marked unqualified. No merge, release or target reduction.
