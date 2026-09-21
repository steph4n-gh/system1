# Official-test candidate freeze: experiment 1

This recipe is selected on fitting, calibration and development data only. The
official tests have not been scored at freeze time. The unchanged targets are
80% supported coverage, 99% accepted accuracy, at most 1% CLINC out-of-scope false
acceptance and complete warm adapter p95 below 5 ms. Both workloads must pass.
The [protocol](PROTOCOL.md) controls interpretation, including prior limited test
exposure and the prohibition on tuning against the final tests.

[freeze.json](freeze.json) binds every selected artifact and relevant Python
source, the pinned data manifest, package configuration and this recipe. The
evaluator resolves the immutable Git commit containing that file, checks its
committed contents, then checks every bound file against that same commit before
opening a test. Runtime source before this freeze is commit
`016df20bd0c747ce892f6bbcc7b9eeab09127da1`; the baseline and evaluator are added
in the freeze commit. Development reports retain all attempted configurations.

## Fixed candidates

| Workload | Saved System1 candidate | Policy | Encoder CPU threads |
|---|---|---|---:|
| CLINC150 + OOS | `artifacts/clinc-development` | strict conformal alpha .05; confidence >= .8961238765292784; density >= -658.9875995494192; OOS prediction always reviewed | 1 |
| BANKING77 | `artifacts/banking-development` | strict conformal alpha .075; learned reliability >= .9167952200817726 | 4 |

CLINC uses frozen MiniLM and a logistic choice head with regularization .1.
Banking uses frozen BGE-small, regularization .01 and a seven-feature reliability
head with C=.1. That gate is taught with three folds of out-of-fold original
fitting predictions plus calibration predictions. Generated lessons are excluded
from the out-of-fold heads, preserving their prompt-example separation. All
6,950 original/generated banking fitting examples become the final head's lessons.
No pretrained encoder is fine-tuned in these selected candidates.

Each workload's TF-IDF/logistic baseline receives the identical final fitting
rows. C=1 and C=10, probability/margin thresholds and calibration-taught
seven-feature reliability gates with C=.01/.1/1 were compared on development.
Both selected baselines use intent C=10 and gate C=1. Their saved manifests bind
vocabulary, IDF, coefficients, prototypes and thresholds. Banking's generated
lessons reduce the simpler baseline's coverage; that failed comparison remains
in the earlier reports rather than disappearing from the record.

The [baseline development report](results/baseline-development.json) records all
20 policy configurations and save/reload outcomes. An initial script run finished
CLINC, then stopped at a JSON-container parsing error before banking fitting. The
corrected run completed both; no official test was involved in either run.

## Frozen procedure

Download the pinned sources and prepare splits with `prepare.py`. Download the
two encoder assets using the revisions in the [experiment README](README.md).
Use the saved artifacts; recompilation can change serialization hashes and is
not necessary to reproduce their evaluation. The required research environment
is Python 3.13.5, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, ONNX Runtime
1.30.0, tokenizers 0.22.2 and threadpoolctl 3.7.0 on macOS/Apple M4 Pro. The
evaluator records actual versions and hardware. Latency is hardware-specific.
No other experiment or fitting jobs run during final timing; ordinary desktop
applications remain open. This is a local desktop measurement, not an isolated
server benchmark.

Before the freeze commit, run the evaluator on all development inputs only:

```bash
sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  .venv/bin/python benchmarks/quality/n8n_gauntlet/evaluate.py --preflight \
  --output .system1/n8n-gauntlet/evaluator-preflight.json
```

After committing and pushing this freeze, make the first official evaluation:

```bash
sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  .venv/bin/python benchmarks/quality/n8n_gauntlet/evaluate.py --official \
  --output benchmarks/quality/n8n_gauntlet/results/official-test.json
```

An OS-denied control connection is required. For each workload, evaluate System1
then the baseline, with 100 untimed development warmups followed by every official
test row exactly once per adapter. Limit BLAS to one thread; the banking encoder
uses its four declared intra-op threads for each single request. No batching,
response cache, teacher call or receipt signing occurs. Per-request timing covers
input validation, encoding, the decision, uncertainty/scope checks and response
construction. Journal writes occur after timing; HTTP/n8n timing is separate.

Publish full counts, Wilson intervals, every accepted error, per-intent metrics,
all row outcomes, startup/preflight and artifact-load times, and head/guard/encoder
bytes. The JSONL journal preserves interrupted-run observations; neither output
is overwritten. Independent reproductions should choose a new ignored output
path. Reusing these tests is replication, not new confirmation.

Artifact manifests intentionally retain their development status and exact
hashes. Qualification is a separate result about those immutable artifacts.
If any target fails, publish that failure and do not present the n8n development
integration as a qualified takeover. Do not retune and rescore these tests as
fresh evidence. A next attempt requires a declared experiment and independent
confirmation while preserving the full workload and original targets.
