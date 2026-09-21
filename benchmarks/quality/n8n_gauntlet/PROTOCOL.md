# n8n decision takeover: protocol v1

This protocol is frozen before this experiment's first official-test evaluation.
Development results, including failures, remain part of the record. Passing is
an empirical result on these cohorts, not a promise about arbitrary workflows.

## Claim and scope

Teach one bounded, single-label intent skill, qualify it, connect **that saved
artifact** to a self-hosted n8n workflow, and disconnect the teacher. Accepted
requests run locally; uncertain or unsupported requests go to an explicit review
branch when the teacher is unavailable. This replaces a classification call,
not text generation, arbitrary reasoning, or an entire agent. Category or
instruction changes require a new candidate and qualification.

The primary workload is **all 150 CLINC intents**, with its original out-of-scope
requests. Transfer uses **all 77 BANKING77 intents**. Neither workload is reduced
to a convenient subset. These are English, single-intent benchmarks; BANKING77
has no native out-of-scope cohort. Earlier project experiments already inspected
six CLINC intents and three banking intents. We disclose that exposure rather
than describing every test example as new to the project.

## Unchanged pass criteria

Each workload must meet both supported-input targets. CLINC must also meet the
out-of-scope target. All targets must pass together for a qualified demonstration.

| Measurement | Target |
|---|---:|
| Supported requests accepted locally | at least 80% |
| Correct labels among those accepted | at least 99% |
| CLINC out-of-scope requests accepted as a supported intent | at most 1% |
| Warm, single-request local decision latency p95 | below 5 ms |

Publish numerator/denominator and two-sided 95% Wilson intervals, including for
out-of-scope false acceptance. Targets refer to point estimates; they are not
99% statistical lower-bound guarantees. Publish per-intent results and every
accepted error, including bad outcomes. Empty acceptance cannot pass.

## Sources and separation

`prepare.py` pins URLs and SHA-256 checksums for CLINC's `data_full.json` and
BANKING77's original train/test CSVs. Licenses are
[CLINC CC BY 3.0](../workloads/CLINC_LICENSE.md) and
[BANKING77 CC BY 4.0](../../../examples/teaching/BANKING77_LICENSE.md).

Preserve every official test row: 4,500 supported and 1,000 out-of-scope CLINC
requests, and 3,080 supported banking requests. Test texts are used only for
normalized-duplicate exclusion until the candidate is frozen. They never fit
features, weights, thresholds, teaching examples, or selection decisions.

Normalize with Unicode `\w+`, case folding and single-space joining, then SHA-256.
Reserve test groups first and remove their occurrences from earlier splits.
For CLINC, reserve official validation (3,000 supported and 100 out-of-scope
before exclusions), then deduplicate the combined train/out-of-scope-train rows.
For each label, sort remaining groups by hash: the first 20 become calibration,
the rest fitting examples. For banking, group and sort the original training
rows per label: first `floor(0.2 * count)` calibration, next the same number
development, remainder fitting (the smallest source class has only 35 rows).
Conflicting-label groups in training are excluded and counted. All folds are
group-disjoint. Test rows are never silently dropped to improve a score.

Development data may select representation, fitting method, regularization,
uncertainty policy and thresholds. Log every attempted configuration and result.
Calibration is separate from fitting; development is separate from both. The
existing compiler may further split calibration for temperature/conformal fits.
The initial feasibility probe used validation-only duplicate exclusion and is
retained as an exploratory baseline, not an official-test result.

Before evaluating the official tests, commit the chosen settings, acceptance
rules, source revision, artifact hashes and exact reproduction commands. Do not
change the candidate in response to those results and reuse the same test as
fresh confirmation. If it fails, retain the failure; a follow-up needs a declared
new experiment and independent confirmation. Never lower these targets or call
a development-set score a pass.

## Comparisons, cost and timing

Run TF-IDF/logistic regression on identical fitting labels and split boundaries
as a conventional baseline. Give it calibration/development selection too, and
publish its settings and failures. Report raw quality as well as selective
quality/coverage. Do not imply System1 invented intent routing or distillation.

Public human labels are a reproducible first teacher. A live API teacher is a
separate measured experiment: save its model/settings, actual answers, calls,
errors, latency and available usage/cost. Dataset correctness and agreement with
that teacher are separate measurements. Never substitute dataset labels for
unavailable live responses while claiming an API-teaching run. Teacher prompts
must not contain the expected label.

Record compilation time, number of teaching labels, package/runtime versions,
hardware, artifact bytes and any separately required encoder files. If a
pretrained encoder is used, identify and count it; a tiny decision head does not
make its encoder free. No test-result or semantic-response cache is allowed.

Time the complete local adapter decision, including input projection, uncertainty
and scope checks, from an already loaded artifact. Use 100 untimed development
warmups followed by every official test request once; report p50 and p95. Exclude
process/import startup, measure and publish it separately. HTTP/n8n round-trip
latency is a separate measurement, including serialization and workflow costs.
API comparisons include network latency and use matched inputs; never present a
batch measurement as single-request latency. Document receipt settings.

## Reproducible integration and recording

Use the exact qualified skill (verified by SHA-256), preserving n8n item pairing
and explicit review routing. Test save/reload equivalence, changed categories or
instructions, malformed input, unfamiliar requests and teacher failure. A local
decision must not need or silently call an API. Block network connections during
headless inference; separately verify teacher call counts in the workflow.

The recording shows a real n8n execution, the skill identity, teacher status and
call count, accepted novel requests, and at least one review outcome. Disconnect
the live teacher and execute additional requests. Publish the importable workflow,
setup commands, evidence and recording; exclude credentials. If quality fails,
show the failure instead of recording an allegedly qualified takeover.
