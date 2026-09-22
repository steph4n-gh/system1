# System 1 Decision Engine — Quality Benchmark Suite

Use this index to distinguish a working product flow from a qualified workload.
**Raw accuracy, accepted correctness, acceptance coverage, and execution cost
answer different questions.** A passing recurring check is regression evidence;
it is not fresh confirmation after its outcomes have informed development.

| Workload / evidence | Status and scope | Baseline or comparison |
|---|---|---|
| [Three bounded public-data skills](workloads/README.md) — three banking intents, six assistant intents, SMS | Original fitting/calibration/test splits; all three meet the stated 95% accepted-correctness / 80% acceptance point targets. These fixed tests are now regression evidence. | Matched TF-IDF/logistic regression is faster on all three; raw quality ties banking/assistant and is lower on SMS. |
| [Actual Jev/Gemini observation](workloads/README.md) | Both teach the six-intent assistant takeover after 358 observations. Other teacher/task combinations defer or miss independent quality targets. | Original-label teacher and saved/reloaded local outputs; provider and local timing cohorts are not a matched speedup test. |
| [Stronger qualification / fresh authored probes](quality_round/README.md) | Fuller SMS teaching improves the old test; new SMS accepted correctness is 27/33 (81.8%). Banking/assistant remain deferred; routing revision rejected. | Original recipes and unchanged skills; authored probes are diagnostics, not customer evidence. |
| [Public SpamAssassin mail](public_email/README.md) | Source shift fails at 89.9% accepted correctness; subsequent retrospective grouped split reaches 603/618 (97.6%). Binary spam/ham only. | Conventional classifier has slightly better raw correctness on the representative split. |
| [Inbox Zero pilot](../../examples/inbox_zero/README.md) | Upstream compatibility and authored synthetic email; accepted mistakes remain. Review-only, without independent mailbox qualification. | Earlier hashed skill, broader TF-IDF skill, and conventional classifier. |
| [Full-scope n8n gauntlet](n8n_gauntlet/README.md) | All 150 CLINC / 77 banking intents remain unqualified: accepted accuracy fails on both; unfamiliar rejection fails on CLINC. Rehearsal proves integration only. | Matched conventional and research candidates; exact artifacts, failures and costs retained. |
| [Document teaching workspace](../../examples/teaching_by_doing/README.md) | Demonstrates lessons, corrections, candidate comparison and explicit adoption on recurring authored checks. Historical sample evidence includes an accepted unusual-input error. | Approved skill versus candidate on the same checks; no independent production qualification. |
| [Real-document correction workflow](document_workflow/README.md) — five BBC news topics | Fresh 400-article holdout: targeted feedback halves accepted errors (16→8), while coverage falls from 93.25% to 84.25%. All candidates fail the default zero-error development adoption gate. | Initial skill and equal-count ordinary extra lessons; no adopted skill or proven raw-accuracy gain. |
| [Games and composed skills](../../examples/README.md) | Snake planner hints, courier wiring, and bounded Pokémon/Paperclips policies have separate scopes. No general agency or independent game-reasoning claim. | Per-experiment rules, planner accounting, retained-label baselines and paired runs; inspect each report. |

The new [correction workflow](../../docs/guides/correcting_skills.md) makes these
distinctions visible while improving a single text-choice skill. Its default
adoption thresholds are a demonstration policy, not a new quality benchmark or
production guarantee. Use representative labeled traffic and task-specific
thresholds to qualify your own skill. Keep evaluation evidence out of fitting and
calibration, and check completed workflow outcomes when decisions cause actions.

Recent reproduction paths follow. Historical benchmark tables and raw reports
remain below and in their linked directories; no failed result has been removed.

## Full-scope n8n gauntlet: unqualified

The [gauntlet index](n8n_gauntlet/README.md) retains the preregistered protocol,
all experiments and failures for 150 CLINC and 77 BANKING77 intents. Both workloads
still miss accepted correctness in the latest full-scope regression; CLINC also
misses unfamiliar-input rejection. The later development comparisons do not
replace that evidence. Independent confirmation remains unavailable.

The [n8n workflow](n8n_gauntlet/N8N.md) and
[live-teacher disconnection rehearsal](n8n_gauntlet/DISCONNECTION_REHEARSAL.md)
demonstrate integration behavior with an unqualified saved skill. They do not
satisfy the qualified-recording goal. Use a Git checkout for full research
reproduction; raw results and research artifacts are excluded from distributions.

## Real public email, without mailbox access

The [SpamAssassin experiment](public_email/README.md) downloads checksum-pinned
real emails, teaches a saved local spam/ham skill, and compares it with a
conventional classifier. It retains both a source-shift failure and a successful
representative grouped split. Reproduce with
`python benchmarks/quality/evaluate_public_email.py --download --representative`.
This separate task does not qualify the Inbox Zero pilot's seven-category routing.

## 1.0.2 qualification and teaching-quality round

[Protocol, results and limits](quality_round/README.md) cover the new opt-in
accepted-agreement bound, fuller SMS observation and 106 fresh authored probes.
The stronger SMS recipe improves the old test result but fails the new authored
SMS cohort. Banking/assistant remain deferred, and the routing candidate is not
adopted. Reproduce with `python benchmarks/quality/evaluate_quality_round.py`
after preparing the public datasets. No live teacher calls or new dependencies.

## Three public workloads and two live teachers

The [workload evidence](workloads/README.md) measures banking support, six assistant
commands, and SMS spam triage against a TF-IDF/logistic-regression baseline.
It also records actual Jev and Gemini observation-to-local takeover, preserving
unsuccessful promotions and the SMS case that misses the independent quality
target. The protocol was committed before the new evaluations; recorded teacher
responses replay offline without API keys.

## Confidence and external banking data

```bash
python benchmarks/quality/evaluate_confidence.py
```

Checks calibration changes with unchanged routing weights, keeps prior development
and fresh confirmation cohorts separate, and evaluates a fixed three-intent slice
of BANKING77's official test split. Network calls are blocked and saved/reloaded
behavior must match. See [1.0.1 methods, results, and limits](../../docs/releases/1.0.1.md)
and the [complete report](results/confidence_round.json).

The fresh routing cohort misses the 95% accepted-correctness target, explicitly
recorded as `all_cohorts_meet_targets: false`. The command's regression checks
allow that disclosed miss while requiring useful acceptance, no increase in
accepted routing errors per cohort, and target attainment on original routing
and banking. Exit success does not mean every cohort meets the quality target.

## Stable release evaluation

```bash
python benchmarks/quality/evaluate_release.py
```

Checks the three current manually taught skills alongside the original 1.0 skills
taught from recorded live Jev answers. New locally authored contrast lessons are
excluded from the recorded teacher baseline. No API key is needed. Local evaluation blocks network access and
checks saved/reloaded behavior. The command exits unsuccessfully if any skill
misses 95% accepted correctness or 80% acceptance on these authored cases.

See the [1.0 evidence and limitations](../../docs/releases/1.0.md),
[current primary results](results/release_1_0_1.json), [original 1.0 results](results/stable_release.json), and
[data and development history](../../examples/teaching/README.md).
These point-estimate targets do not certify population accuracy. The seed and
five-fold results below are retained as historical 0.2.2 comparisons.

## Teaching close distinctions

```bash
python benchmarks/quality/evaluate_contrasts.py
```

Reproduces the frozen comparison of original teaching with 34 additional lessons, checks saved/reloaded
behavior offline, and reports diagnostic and fresh confirmation cases separately.
Later confidence-calibration additions are excluded. See the [results and remaining errors](contrast_round.md).

## First-use game decisions

```bash
python benchmarks/quality/evaluate_zero_shot.py --projectors
```

Checks frozen Pokémon and Paperclips decisions with teaching and network access
blocked, plus an unchanged raw routing control. The optional projector comparison
measures the existing hybrid text projector. See the
[results and limitations](zero_shot/README.md): game-policy improvements are
reported separately from raw classifier quality.

## Historical seed benchmarks

The remainder of this page describes the original 0.2.2 development benchmarks. Recorded values are historical snapshots; rerunning with current code can differ. The public-workload and release evaluations above are the current evidence entry points.

## Benchmarks

| # | Benchmark | Type | Classes / Range | Examples | Key Metrics |
|---|-----------|------|----------------|----------|-------------|
| 1 | **Security Triage** | 3-class classification | ALLOW / QUARANTINE / BLOCK | 100 | Accuracy, per-class P/R/F1, confusion matrix |
| 2 | **Intent Routing** | 4-class classification | technical_support / billing / sales / escalate | 100 | Accuracy, per-class P/R/F1, confusion matrix |
| 3 | **Threat Scoring** | Continuous regression | 0.0 – 10.0 | 100 | MAE, RMSE, Pearson r |

All benchmarks also report latency statistics (P50, P90, P95, P99, min, max).

## Dataset Composition

### Security Triage
- 35 benign read/diagnostic/exec operations (ALLOW): 27 benign reads, 4 benign exec, 4 benign diagnostic
- 30 borderline modification/network/install/exec operations (QUARANTINE)
- 35 destructive/exfiltration/exploit operations (BLOCK): including 10 prompt injection attempts, 6 destructive, 5 exfiltration, 7 exploit, and others

### Intent Routing
- 25 clear billing queries, 32 technical support queries (25 clear + 7 ambiguous/edge), 25 sales queries (20 clear + 5 ambiguous/edge), 18 escalation cases (15 clear + 3 ambiguous/edge)
- 10 ambiguous multi-intent queries, 5 vague queries, 6 edge cases

### Threat Scoring
- 20 benign operations (score ≈ 0–0.5)
- 15 low-risk operations (score ≈ 1.5–2.5)
- 20 moderate-risk operations (score ≈ 3.5–5.5)
- 20 high-risk/malicious operations (score ≈ 8.5–10)
- 10 prompt injection attempts (score ≈ 8.5–10)
- 15 borderline cases (score ≈ 2.5–7)

## How to Run

```bash
# Run all benchmarks
python3 benchmarks/quality/run_quality_benchmarks.py

# Run a specific benchmark
python3 benchmarks/quality/run_quality_benchmarks.py --bench security
python3 benchmarks/quality/run_quality_benchmarks.py --bench intent
python3 benchmarks/quality/run_quality_benchmarks.py --bench scoring

# Custom output path
python3 benchmarks/quality/run_quality_benchmarks.py --output my_results.json
```

Run from the repository root. The script automatically adds `src/` to `sys.path`.

## Teaching comparison

The main runner above measures schema-derived starter classifiers. To check what
the existing compiler learns from task examples, run:

```bash
python benchmarks/quality/evaluate_teaching.py
```

The [recorded comparison](results/teaching_review.json) uses five fixed,
label-stratified folds (seed 42). Each of the 100 examples per task is evaluated
once, outside the examples used to teach or calibrate that fold's skill. Normalized
duplicate prompts stay together. There are no workflow or paraphrase-group labels,
so related cases may cross folds and inflate results. These development datasets
have already informed implementation choices; they are not an untouched release test.

Each taught skill uses only supplied examples, ridge regularization 1.0, and the
default 25% calibration split. For choice fields, temperature and conformal
calibration use separate parts of that split. We compare the unchanged default
384 features with 2048 features. No parameter search is run by this script.

| Task | Starter, 384 features | Taught, 384 features | Taught, 2048 features |
|---|---|---|---|
| Security triage accuracy | 50% | 68% | 71% |
| Intent routing accuracy | 51% | 61% | 68% |
| BLOCK predicted as ALLOW, out of 35 | 8 | 6 | 0 |

**All variants requested review on all examples** in strict mode at alpha 0.05.
The taught folds have only 9–10 conformal calibration examples; the resulting
quantiles retain all choices here. Accepted accuracy is therefore undefined
(`null` in the report), not 100%. Zero accepted errors with zero accepted decisions
does not establish useful automatic routing or security enforcement.

The 2048-feature skills were about 26 KB for triage and 34 KB for routing, with
median decision times of 0.396 ms and 0.509 ms respectively on the review machine.
Both caches and receipt generation were disabled. These are local measurements,
not end-to-end service guarantees. The report includes environment details,
dataset hashes, fold sizes, per-example predictions, review rates, and latency.

The comparison demonstrates that teaching helps raw classification. To establish
a useful skill, collect representative task examples, keep related workflows
together, and check the resulting skill against a separate evaluation set.

## Seed benchmark output

Results are printed as human-readable tables to stdout **and** saved as JSON to `benchmarks/quality/results/`. Each run produces a timestamped JSON file containing:

- Per-benchmark accuracy, precision, recall, F1 (macro and micro)
- Full confusion matrices
- Latency statistics (P50, P90, P95, P99)
- Per-example raw results (prompt, expected, predicted, confidence, latency)
- Regression metrics (MAE, RMSE, Pearson r) for the scoring benchmark

## Directory Structure

```
benchmarks/quality/
├── README.md                       # This file
├── run_quality_benchmarks.py       # Main runner script
├── datasets/
│   ├── security_triage.json        # 100 labeled security triage examples
│   ├── intent_routing.json         # 100 labeled intent routing examples
│   └── threat_scoring.json         # 100 labeled threat scoring examples
└── results/                        # Auto-created, holds JSON output
    └── quality_results_*.json
```

## Schemas

The historical benchmark schemas use field types also demonstrated by `examples/support_triage.py` and `examples/autonomous_agent_firewall_showcase.py`:

- **SecurityTriageSchema**: `ChoiceField` with 3 options (ALLOW/QUARANTINE/BLOCK) and detailed multi-line descriptions per option
- **IntentRoutingSchema**: `ChoiceField` with 4 options (technical_support/billing/sales/escalate) with detailed descriptions
- **ThreatScoringSchema**: `ScoreField` with range [0.0, 10.0] with low/high boundary descriptions

## Metrics Definitions

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Accuracy | correct / total | Overall correctness |
| Precision | TP / (TP + FP) | How many predictions for a class were correct |
| Recall | TP / (TP + FN) | How many actual examples of a class were found |
| F1 | 2 × P × R / (P + R) | Harmonic mean of precision and recall |
| Macro-F1 | mean(F1 per class) | Unweighted average across classes |
| Micro-F1 | F1 on aggregated TP/FP/FN | Aggregate counts; equals accuracy for single-label multiclass classification |
| MAE | mean(abs(actual - predicted)) | Average absolute prediction error |
| RMSE | √mean((actual - predicted)²) | Root mean square error |
| Pearson r | correlation coefficient | Linear correlation between predicted and actual |
