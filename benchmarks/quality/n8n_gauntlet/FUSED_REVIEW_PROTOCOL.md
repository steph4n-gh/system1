# Complete review for the fused CLINC head: experiment 2k

Commit this procedure and implementation before fitting review or measuring the
adapter. Experiment 2j's fixed raw-gain rule advances CLINC only: word features
improve its raw supported correctness by 25 requests but worsen banking by 14.
Banking's rejected fusion, original scope and failed quality remain unchanged.
This follow-up does not narrow the full two-workload goal or qualify either one.

## Fixed head and review teaching

Use the exact saved `fused-clinc150-semantic-plus-words` intent head and lexical
configuration from 2j. Verify every byte against its manifest; do not refit its
intent head, vocabulary, IDF, temperature or conformal calibration. Its original
CLINC fitting rows and fixed MiniLM remain the only intent-teaching data. No
generated 2e/2i lesson, new teacher call, encoder update or new dependency.

Teach exactly one review head with the existing seven numerical signals and
one-hot predicted category: standardize the seven numbers, keep the category
indicator unscaled, fit logistic C=.1 with max_iter=1000. No grid, new review
feature or rescue condition. Preserve strict alpha .05 and review OOS suggestions.

Use the original per-label/hash three fitting folds. For each temporary fused
intent head, fit the 2,048-term portable TF-IDF vocabulary and IDF **only on its
two fitting folds**. Compile its numerical head with regularization .1 and the
original calibration fold. Neither the excluded fold nor any generated example
fits its vocabulary, IDF, intent weights or prototypes. Their out-of-fold
predictions supply supported-correctness review targets. Add correctness on the
original calibration fold from the fixed final head, and the pinned 2,000
Wikipedia negative review examples. Correct supported predictions are positive;
mistakes and OOS are negative. Wikipedia never teaches the intent head.

For the four similarity signals, retain 384-dimensional semantic prototypes
from the applicable fitting rows only. Do not create 2,432-dimensional prototype
storage or run a second encoder. At runtime recover the semantic component by
normalizing the first 384 entries of the fused decision embedding. The other
three signals come from the fused head's class probabilities. Use this same
operation for review fitting, with a scalar numerical consistency preflight.

Keep all saved-head decisions identical to the published 2j development record
before selecting review. Select maximum supported development coverage subject
to >=99% supported accepted accuracy and <=1% OOS false acceptance, with the
unchanged midpoint threshold procedure. Compare with the full incumbent list in
`results/direct-oos-runtime.json`; ties retain the incumbent. Retain the matching
data linear semantic review and conventional baseline results as references,
including the stronger CLINC baseline's disclosed additional teaching data.

## Complete saved adapter measurement

Save a standalone artifact: unchanged intent head and lexical JSON, numerical
review/prototype NPZ, and manifest binding the files, external encoder, contract
and source head identity. The runtime uses the existing System1 engine and
response format. Reconstruct features when loading; cached teaching vectors
must not serve inference. No response cache or receipts.

Verify all 147 prior frozen files and the committed procedure/2j artifacts before
fitting. No original tests or human reserves are opened or scored. After saving,
run exactly 100 development warmups and measure all 3,095 development requests
once under OS network denial. Record complete outcomes, accepted mistakes,
per-intent counts, Wilson intervals, p50/p95, startup/load time, full artifact and
encoder sizes, review preparation/fit time, environment and zero new calls/cost.
Checkpoint the result before comparison so a reporting failure cannot erase it.
Do not repeat timing to improve the number. Reproduce recorded responses with
fitting imports blocked, and check malformed inputs and changed contracts.

A new development selection requires identical saved-runtime routing, no errors,
complete p95 <5 ms and higher supported coverage at the same quality targets.
The >=80% coverage gate still applies to any development pass. Earlier component
timings are not joint latency; this measurement includes projection, numerical
head, strict review, learned review and response construction.

This is development, not independent confirmation. All prior failures and
banking requirements remain. A qualified n8n takeover still requires the full
unchanged gauntlet, fresh full-scope confirmation and that exact qualified saved
artifact. PR #3 remains open; no release, lowered target or qualification claim.

From the repository root after committing the exact procedure:

```sh
sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python benchmarks/quality/n8n_gauntlet/fused_review.py fit --folder .system1/n8n-gauntlet/fused-review-development
sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python benchmarks/quality/n8n_gauntlet/fused_review.py measure --folder .system1/n8n-gauntlet/fused-review-development
```

Outputs are exclusive. A later reproduction must use a new directory and label
its reused development rows as a replication, not fresh evidence.
