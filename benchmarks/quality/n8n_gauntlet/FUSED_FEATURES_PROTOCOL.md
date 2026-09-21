# Semantic plus word features: experiment 2j, head comparison

Commit this protocol, feature helper, runner and authored numerical tests before
fitting. Experiment 2i's synthetic OOS teaching harms supported distinctions.
The earlier agreement diagnostic separately found lexical predictions correct
on 56 CLINC and 61 banking development inputs where System1 was wrong. Those
counts are not an oracle or evidence that a consensus policy works. This next
comparison asks whether one learned head can use both representations.

Use the existing portable `TfidfProjector`; do not introduce another encoder,
dependency, network call, LLM fine-tuning, core API or saved-head format. Before
building a new review/runtime path, fit exactly four intent candidates:

| Workload | Control | New condition |
|---|---|---|
| All 150 CLINC intents plus native OOS | pinned 384-dimensional MiniLM | same MiniLM + 2,048 TF-IDF features |
| All 77 BANKING77 intents | pinned 384-dimensional BGE-small | same BGE + 2,048 TF-IDF features |

Use original CLINC fitting rows; banking additionally retains the earlier 924
Gemini supported lessons. Neither experiment-2e nor experiment-2i lessons enter
these fits. The controls and new candidates receive exactly the same labels.
Original fit/calibration/development boundaries and hashes remain unchanged;
no original test or reserved human inputs are opened or scored.

Fit the portable word unigram/bigram vocabulary and IDF exclusively on each
workload's declared fitting text, with the existing default `max_features=2048`.
Use unscaled unit semantic and unit lexical vectors concatenated and then
L2-normalized; neither half gets a tuned coefficient. An all-unknown lexical
input has a zero lexical half and a normalized semantic half. This is 2,432
dimensions, with no encoder-weight update. Every semantic vector uses the same
individual-request operation as runtime: one encoder thread for CLINC, four for
banking. Existing feature caches may accelerate teaching only.

Keep System1 logistic regularization .1 for CLINC and .01 for banking, no feature
count/weight/regularization grid. Compile using original calibration data and
the existing temperature/conformal procedure. Strict alpha remains .05 / .075.
Save each numerical intent head with the real external projector identity and
the lexical configuration, where applicable. Record encoder identities and all
artifact hashes. Reconstruct each saved projector and head and verify a small
fixed 16-input development replay against recorded probabilities and prediction
sets; replay is serialization verification, not a quality subset or latency run.

Report every development input's raw suggestion, confidence, margin, conformal
set and strict-review eligibility. Report raw supported correctness, OOS
suggestions, and the unchanged optimistic probability/margin coverage frontiers
subject to >=99% accepted supported accuracy and <=1% CLINC unfamiliar acceptance,
with and without strict eligibility. These exploratory frontiers select on
development labels and are **not qualified policies or replacements for the
existing learned review**. They cannot establish an adapter p95 or a full pass.
Retain all four outcomes, fit/preparation times, bytes, failures, source hashes,
environment and zero new teacher calls/cost. No inference-speed claim is made
from cached vectors or this head comparison.

Compare raw outcomes with the already recorded matching-data TF-IDF/logistic
baseline and controls. Also retain the stronger selected baselines; the CLINC
incumbent has extra experiment-2e supported teaching data, which must be disclosed.
Do not refit identical conventional baselines simply for a new timing number.

Advance fusion to a separately declared complete review/adapter comparison for
a workload only if its raw supported-correct count is strictly higher than its
semantic control. Otherwise retain the failure and existing selection; no
hyperparameter rescue. A raw gain alone is not adoption or qualification. The
next stage, if justified, must use original-only review cross-fitting with lexical
vocabulary/IDF fitted only inside each fitting fold, exclude generated prompt
lineage, use calibration correctness, and measure the complete saved adapter.

All 147 previously frozen files and all failures remain unchanged. Both full
workloads still need >=80% coverage, >=99% accepted accuracy, <=1% CLINC unfamiliar
false acceptance and complete p95 <5 ms together. Independent full-scope
confirmation, exact-qualified-artifact n8n integration and the qualified
teacher-disconnection recording remain required. Keep PR #3 open; no release.

After committing the procedure, from the repository root:

```sh
sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python benchmarks/quality/n8n_gauntlet/fused_head_probe.py --folder .system1/n8n-gauntlet/fused-head-development
```

The output directory is exclusive. A later replication must use a new directory
and cannot become independent confirmation on these already-used development rows.
