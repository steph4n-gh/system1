# Latest saved-candidate regression: experiment 2h

This is a separately declared regression check of the development selections
after experiments 2e–2g. The original tests were observed in experiments 1 and
2d. **No outcome here is independent confirmation or qualification**, even if
all point targets pass. Do not fit, select thresholds or modify labels here.

Freeze these exact public artifacts and their loader code in Git before scoring:

- CLINC System1: `artifacts/clinc150-system1-context`, selected in 2g.
- Banking System1: `artifacts/polynomial-banking77-system1-quadratic`, selected
  in 2f and retained on the 2g tie.
- CLINC conventional baseline: `artifacts/clinc150-baseline-positive-and-unsupported`,
  selected in 2e and retained subsequently.
- Banking conventional baseline: `artifacts/polynomial-banking77-baseline-quadratic`,
  selected in 2f and retained subsequently.

Bind their parent artifacts, all 125 previously frozen files, the earlier failed
reports, latest development comparison reports, this protocol and evaluator in
`latest-regression-freeze.json`. Before committing the freeze, check the first
16 original development responses per method/workload against their published
runtime reports. Preflight opens no test or reserved-human requests. The runner
must verify the exact committed freeze and its bound files before test access.

Use every original test row and label without corrections or filtering: 4,500
supported plus 1,000 unfamiliar CLINC inputs, and 3,080 banking inputs. Keep the
original point targets: supported coverage at least 80%, accepted supported
accuracy at least 99%, CLINC unfamiliar false acceptance at most 1%, and full
warm single-request adapter p95 below 5 ms. No teacher calls, response caching
or receipts; verify OS network denial. Warm each candidate with 100 development
requests, then measure every original test request once. Retain runtime errors
as explicit reviews and count them as a failed runtime-error gate.

Record every response, group, original text/label, candidate identity, per-intent
counts, accepted mistakes, count/denominator and Wilson intervals, complete
p50/p95/max, load time, full process import/freeze-preflight time, packages,
hardware, parent/child/encoder sizes and zero teacher calls/new API cost. Use
exclusive report/journal creation to retain failed or partial runs. Do not
remeasure to improve latency. Preserve all earlier reports unchanged.

Compare directly with the first held-out failure and 2d observed regression,
including improvements and regressions for both System1 and the stronger
baselines. No new workload, reduced taxonomy or relabeled cohort substitutes
for the original scope. The earlier teacher costs remain published; this audit
adds no teaching. The 151 human OOS requests and human paraphrase supplement
remain unscored. Independent full-scope confirmation, the exact qualified n8n
artifact and its disconnection recording are still required. Keep PR #3 open.
