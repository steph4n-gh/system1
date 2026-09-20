# Three workloads, two live teachers

This is the preserved first public-workload experiment. The later
[1.0.2 quality round](../quality_round/README.md) adds stricter qualification and
new diagnostics; it does not rewrite the original results below.

System 1.0.1 meets the fixed 95% accepted-correctness and 80% local-acceptance
targets on three public-data teaching tasks. On the assistant task, both actual
Jev and Gemini observations produce a validated local takeover. The classical
baseline is faster locally, and two other takeover tests expose limitations.
All outcomes are retained in the [complete report](results.json).

The [protocol](protocol.json) was committed as `b5e5246` before the first
assistant/SMS model evaluation. Workload selection, data splits, model settings,
stream order, observation limits, and quality targets were fixed in advance.
Banking repeats the published 1.0.1 task; assistant commands and SMS are new
evaluations for this project. No teaching data, calibration, thresholds, or core
code were changed in response to these results.

## Teaching from independent labels

| Workload | Teaching / calibration / test | Raw correct | Accepted locally | Correct among accepted |
|---|---:|---:|---:|---:|
| Three banking-support queues | 287 / 125 / 120 | 115/120 | 118/120 (98.3%) | 115/118 (97.5%) |
| Six assistant commands | 600 / 120 / 180 | 175/180 | 173/180 (96.1%) | 172/173 (99.4%) |
| SMS spam triage | 3,061 / 1,068 / 1,001 | 972/1,001 | 980/1,001 (97.9%) | 957/980 (97.7%) |

The banking decisions are card arrival, lost/stolen card, and cash withdrawal
fees. Assistant commands are weather, translation, timers, calendars,
calculations, and playing music. SMS labels are the original `ham` and `spam`.
The examples only classify requests; no banking, messaging, or assistant action
is executed.

All three use 2,048 fixed features, ridge regularization 0.1, supplied examples
only, alpha 0.05, and strict review. The compiler fits separate temperature and
conformal calibration folds. Evaluation blocks network connections and checks
identical answers, probabilities, uncertainty sets, and review flags before and
after saving. Every test input is outside the teaching and calibration groups.

| Workload | Teach + calibrate | Saved skill | Median local decision |
|---|---:|---:|---:|
| Banking | 226 ms | 24.9 KiB | 0.378 ms |
| Assistant | 239 ms | 47.0 KiB | 0.344 ms |
| SMS | 1,767 ms | 20.4 KiB | 0.444 ms |

These are measurements on the review machine, Python 3.13.5 and NumPy 2.5.3.
Inference is measured one request at a time with caches and receipts disabled.
Timings exclude imports, interpreter startup, preparing/reviewing labels, and
downstream application actions. Skill sizes exclude the installed Python runtime
and libraries. The JSON report includes the environment and per-request timings.

## A real classical baseline

The comparison uses scikit-learn 1.9.1: word unigram/bigram TF-IDF with sublinear
term frequency, followed by logistic regression (`C=1`, `max_iter=1000`). It gets
the identical teaching rows and conformal-calibration half, with standard LAC
scores and alpha 0.05. It does not fit temperature on the other half. Neither
model is tuned against test results.

| Workload | Majority raw correct | System 1 starter raw correct | Taught System 1 raw correct | Classical raw correct |
|---|---:|---:|---:|---:|
| Banking, 120 cases | 40 | 106 | 115 | 115 |
| Assistant, 180 cases | 30 | 112 | 175 | 175 |
| SMS, 1,001 cases | 881 | 496 | 972 | 949 |

| Workload | System 1 accepted / correct | Classical accepted / correct | System 1 / classical median latency |
|---|---:|---:|---:|
| Banking | 118 / 115 | 114 / 112 | 0.378 / 0.196 ms |
| Assistant | 173 / 172 | 169 / 166 | 0.344 / 0.200 ms |
| SMS | 980 / 957 | 997 / 948 | 0.444 / 0.201 ms |

The classical baseline fits faster as well: approximately 19, 19, and 79 ms.
Its compressed serialized bundles are 36,649, 76,636, and 445,694 bytes;
serialization formats differ, and neither figure includes dependencies.
System 1 matches raw correctness on two tasks, improves it on SMS, and produces
smaller artifacts in this comparison. This does not establish superiority over
a tuned classical model or other packages. System 1's demonstrated additional
value is the integrated teaching, validation, takeover, and portable-skill path.

## Actual Jev and Gemini takeover

The observer receives answers through the existing teacher callback and uses the
normal automatic-promotion policy. It streams up to 1,000 rows from teaching
and calibration data in a fixed hash order, stopping when promotion succeeds.
Official/frozen test cases never enter that stream. No request includes its
expected dataset label when sent to Jev or Gemini.

| Workload / teacher | Observations | Promoted? | Subsequent accepted / test cases | Correct among accepted |
|---|---:|---|---:|---:|
| Assistant / dataset labels | 358 | Yes | 171/180 | 167/171 (97.7%) |
| Assistant / live Jev 1.13.0 | 358 | Yes | 171/180 | 167/171 (97.7%) |
| Assistant / live Gemini 2.5 Flash | 358 | Yes | 171/180 | 167/171 (97.7%) |
| Banking / dataset labels | 412 | No | — | — |
| Banking / live Jev | 412 | No | — | — |
| SMS / dataset labels | 557 | Yes | 963/1,001 | 897/963 (93.1%; target missed) |
| SMS / live Jev | 1,000 | No | — | — |

Both live assistant teachers matched all 358 observed dataset labels. After
promotion, their callbacks and network access were disabled. Each saved skill
then processed all 180 unseen official test requests locally, accepting 171 and
requesting review for nine. Reloading preserved complete answers and review
behavior. **Zero teacher calls after takeover does not mean every request can
be accepted automatically: the measured unattended fraction is 95%.**

Jev's median successful observation latency was 319 ms; Gemini's was 525 ms.
Local assistant decisions took about 0.4 ms. Observation and local timings use
different cohorts, so they are illustrative measurements, not a matched-request
speedup benchmark. Gemini had one HTTP 503; 229 successful responses were retained,
then 129 further responses completed the 358-example stream. Its final process's
68.7-second duration excludes the interrupted first process. The report records
the failure and API-reported token usage; no prices or dollar savings are inferred.

This demonstrates teaching from both a decision API and a general-purpose LLM.
The callback converts their outputs to one label per request; no language model
is downloaded or fine-tuned by System 1. The teachers' confidence distributions
are not copied into the local model.

## Important failures and boundaries

- Banking did not meet the acceptance requirement for automatic promotion with
  the available 412 observations. Its explicitly taught skill succeeds, but that
  is a different path with more fitting data and regularization 0.1. Automatic
  observation retains the compiler's default regularization 1.0 and reserves
  validation data as well as calibration. We did not force a takeover.
- SMS with dataset labels passed the normal promotion policy, then achieved only
  93.1% accepted correctness on the independent test set. The normal gate checks
  lower bounds against 80% agreement and 80% acceptance; it is not a 95% correctness
  qualification. The independent 95% target catches this distinction. Jev's SMS
  run stayed with the teacher through the 1,000-observation cap.
- SMS's stronger explicitly taught result still contains three accepted legitimate
  messages labeled spam and 20 accepted spam messages labeled legitimate. Raw
  spam recall is 80%, versus 56.7% for this classical baseline; correctly accepted
  spam accounts for 75% of all spam. The baseline has no legitimate-to-spam errors.
  Overall accuracy alone is inadequate for choosing an automatic blocking policy.
- Banking and assistant scopes cover three and six intents, respectively. They
  do not measure all 77/150 source intents or out-of-scope detection. The SMS
  corpus is historical; these numbers do not establish modern phishing coverage.
- Confidence intervals are descriptive. Related paraphrases and SMS campaigns can
  remain correlated across splits despite normalized-duplicate checks. Provider
  pretraining exposure to these public datasets is unknown. These results are
  evidence for the stated tasks, not a universal capability or accuracy guarantee.

## Reproduce

From this repository with System 1 installed:

```bash
python benchmarks/quality/prepare_workloads.py
uv run --with 'scikit-learn==1.9.1' python benchmarks/quality/evaluate_workloads.py
python benchmarks/quality/observe_workloads.py assistant_commands --teacher jev --offline
python benchmarks/quality/observe_workloads.py assistant_commands --teacher gemini --offline
```

The first command downloads public sources, verifies their pinned SHA-256 hashes,
and writes the frozen JSON teaching files under `.system1/workloads/data/`.
The comparison-only scikit-learn dependency is not added to System 1's package.
The last two commands replay the [actual recorded teacher responses](records/)
with socket connections blocked; no API keys are needed. Replace the workload
with `banking_support` or `sms_triage`, and use `--teacher dataset` for original
labels. A missing recorded response fails offline replay.

To make new live calls, omit `--offline` and provide `TYPESAFE_API_KEY` or
`GEMINI_API_KEY` through the environment. Existing records are reused. The runner
records only response evidence, model names, timing, and usage; no credentials
are written. It never makes teacher calls on the evaluation cohort.

The supervised runner exits unsuccessfully if any System 1 workload misses the
95%/80% targets. The observation runner reports `promoted` and `meets_test_targets`
separately: finishing an experiment successfully is not the same as qualifying
the skill. First-run results remain in the committed report even when subsequent
offline replays have different timings or make zero new HTTP requests.

## Source attribution and changes

- Banking: [BANKING77 by Casanueva et al. (2020)](https://github.com/PolyAI-LDN/task-specific-datasets),
  CC BY 4.0. The existing [source and split documentation](../../../examples/teaching/BANKING77.md)
  applies; its official 120-query three-intent test slice is unchanged.
- Assistant: [CLINC150 by Larson et al. (2019)](https://github.com/clinc/oos-eval),
  *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*,
  [CC BY 3.0](CLINC_LICENSE.md), revision `828f8093932c8fe6ca7936c3d2e52903b1c523de`.
  We select the six named intents, preserve original labels and all 180 official
  test queries, and use official train/validation splits for teaching/calibration.
  No cross-split normalized duplicates were found. Other intents and the OOS
  benchmark are outside this experiment.
- SMS: [Almeida and Hidalgo, SMS Spam Collection (2011), UCI](https://doi.org/10.24432/C5CC84),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  From 5,574 source rows, normalize to lowercase word tokens and retain the first
  row per normalized duplicate, removing 444 repeated rows. Conflicting-label
  groups would be excluded; none occurred. Split each group's SHA-256 integer
  modulo ten: buckets 0–5 teach, 6–7 calibrate, 8–9 test. Text and labels remain
  unchanged; source indices and group hashes are added. This is a fixed split
  created for this experiment, not an official UCI test split.

The source-preparation regression checks preserve evaluation rows and labels,
keep normalized duplicates together, reject changed source files, and retain
minority-class failures in reported metrics. Recorded teacher responses and
full first-run outcomes allow independent inspection without using either API.
