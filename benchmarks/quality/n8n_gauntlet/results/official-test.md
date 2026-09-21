# First official-test evaluation: failed qualification

The candidates frozen and pushed in `41a0d3634831ec00f104bcd33540a9b613007184` were evaluated once on all official rows. **Neither workload passes every preregistered target.** No candidate or threshold changed after this evaluation. The n8n integration remains a development experiment, and a qualified teacher-disconnection recording has not been made.

All 4,500 supported and 1,000 unfamiliar CLINC requests and all 3,080 BANKING77 requests are included for both methods. The [complete report](official-test.json) contains every response, accepted error, per-intent result, latency, artifact identity and environment. The [frozen recipe](../FREEZE.md) and [protocol](../PROTOCOL.md) define the gates.

An [arithmetic and preservation audit](official-test-audit.json) independently
recomputed the counts, accepted errors and p95 from the saved outcomes and
verified every original text/label plus all 109 frozen file hashes. It ran no
models and did not re-evaluate or select a candidate.

## Held-out results

Quality cells show count, point estimate and two-sided 95% Wilson interval. Accepted accuracy here uses supported inputs; unfamiliar acceptance is measured separately. These intervals are not guarantees for future traffic.

| Workload / method | Supported coverage | Correct among accepted supported | Unfamiliar falsely accepted | Complete adapter p95 |
|---|---|---|---|---:|
| clinc150 / system1 | 3,757/4,500 = 83.49% (82.38–84.55%) | 3,721/3,757 = 99.04% (98.68–99.31%) | 59/1,000 = 5.90% (4.60–7.54%) | 1.825 ms |
| clinc150 / baseline | 1,054/4,500 = 23.42% (22.21–24.68%) | 1,052/1,054 = 99.81% (99.31–99.95%) | 8/1,000 = 0.80% (0.41–1.57%) | 0.828 ms |
| banking77 / system1 | 2,516/3,080 = 81.69% (80.28–83.01%) | 2,485/2,516 = 98.77% (98.26–99.13%) | Not available | 4.392 ms |
| banking77 / baseline | 1,337/3,080 = 43.41% (41.67–45.17%) | 1,331/1,337 = 99.55% (99.02–99.79%) | Not available | 0.709 ms |

CLINC meets supported coverage, supported accepted accuracy and latency, but falsely accepts **59/1,000 unfamiliar requests (5.9%)**, exceeding the 1% limit. The 36 incorrect accepted supported decisions and 59 unfamiliar acceptances are all errors. Across both input types its accepted-decision correctness is **3,721/3,816 (97.51%)**. Reporting 99.04% without its supported-input denominator and the unfamiliar-input failure would be misleading.

Banking meets coverage and latency, but **2,485/2,516 accepted decisions are correct (98.77%)**, below 99%. There are 31 accepted mistakes. BANKING77 has no native unfamiliar cohort; this result provides no banking out-of-scope guarantee.

The conventional TF-IDF baseline is faster and more conservative: it accepts 23.4% of supported CLINC requests and 43.4% of banking requests. It misses the 80% coverage target on both. This experiment supports a coverage/latency tradeoff, not a claim that System1 beats every conventional classifier on speed. Raw supported accuracy is 95.09% versus 90.38% on CLINC and 92.53% versus 87.44% on banking.

## What the errors show

The frozen CLINC development policy accepted one of only 100 unfamiliar examples; that estimate had a 95% Wilson interval of 0.18%–5.45% even before accounting for development selection. Eighty unfamiliar examples taught the head and 20 were reserved for calibration. The final failure shows that passing this small development cohort was insufficient evidence of reliable unfamiliar-input rejection.

Banking mistakes include close distinctions such as card arrival versus delivery estimates, supported currencies versus exchanging currencies, and a virtual card failing versus a generic card failure. These observations are diagnosis, not permission to relabel benchmark errors or tune against the test. Some official labels appear surprising; every original row and label remains unchanged.

## Runtime and teaching accounting

The evaluator required an OS permission-denied control connection and ran all decisions with networking denied. **Zero teacher calls, zero runtime errors and no response cache** were recorded. Each adapter received 100 untimed development warmups; each official request was then timed individually. The complete adapter includes validation, encoding, uncertainty/review checks and response construction. Receipts were disabled. HTTP/n8n overhead is measured separately in the [integration evidence](n8n-integration.json).

CLINC used a frozen MiniLM encoder with one CPU thread; banking used frozen BGE-small with four intra-op CPU threads per request. BLAS used one thread. Neither selected encoder was fine-tuned. This is an Apple M4 Pro desktop run with ordinary applications open and no concurrent experiment jobs, not an isolated-server latency promise.

| Saved candidate | Head/guard/manifest bytes | Additional encoder/tokenizer bytes | Artifact load |
|---|---:|---:|---:|
| clinc150 | 1,258,035 | 23,492,300 | 247.1 ms |
| banking77 | 10,048,195 | 133,804,886 | 347.9 ms |

Process imports and freeze/preflight verification took 2188.0 ms before artifact-load timing. Runtime libraries are additional dependencies, not included in the artifact byte totals.

CLINC used 12,019 public fitting labels and 3,020 public calibration labels. Banking used 6,026 public fitting labels, 1,960 calibration labels and 924 Gemini-generated teaching examples; both baselines received the same final fitting rows as their corresponding System1 candidate. Fitting/compilation timings in the report exclude cached feature extraction and are not complete end-to-end teaching times.

The banking teaching run made 77 Gemini 2.5 Flash requests in 180.3 seconds, using 25,644 input and 17,510 output tokens. The published-price estimate is $0.05147; actual billed cost is unavailable. It generated examples conditioned on category labels, not blind labels for a test. [Requests, usage and pricing evidence](contrast-teacher-summary.json) and [all recorded responses](contrast-teacher.json) remain available. No new API calls were used in final evaluation.

## Consequence for the next experiment

The original goal and targets remain unchanged. The next work must improve unfamiliar-input teaching/rejection and banking distinctions, then obtain independent confirmation across the full scope. These official tests are now observed: a later run on them can diagnose or replicate, but cannot be called fresh confirmation. A new experiment must declare its teaching, selection and independent confirmation data before scoring. The existing artifacts and this failed result remain immutable evidence.
