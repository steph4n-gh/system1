# System 1 Decision Engine — Quality Benchmark Suite

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
[recorded results](results/stable_release.json), and
[data and development history](../../examples/teaching/README.md).
These point-estimate targets do not certify population accuracy. The seed and
five-fold results below are retained as historical 0.2.2 comparisons.

## Teaching close distinctions

```bash
python benchmarks/quality/evaluate_contrasts.py
```

Compares the original teaching with 34 additional lessons, checks saved/reloaded
behavior offline, and reports diagnostic and fresh confirmation cases separately.
See the [results and remaining errors](contrast_round.md).

## First-use game decisions

```bash
python benchmarks/quality/evaluate_zero_shot.py --projectors
```

Checks frozen Pokémon and Paperclips decisions with teaching and network access
blocked, plus an unchanged raw routing control. The optional projector comparison
measures the existing hybrid text projector. See the
[results and limitations](zero_shot/README.md): game-policy improvements are
reported separately from raw classifier quality.

## Overview

This benchmark suite measures the **decision quality** (accuracy, precision, recall, F1, MAE, RMSE, correlation) of the System 1 non-autoregressive decision engine. It complements the existing latency benchmarks by providing the first published accuracy/quality metrics.

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

The benchmark schemas mirror the patterns from `examples/support_triage.py` and `examples/autonomous_agent_firewall_showcase.py`:

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
| Micro-F1 | F1 on aggregated TP/FP/FN | Weighted by class frequency |
| MAE | mean(|actual - predicted|) | Average absolute prediction error |
| RMSE | √mean((actual - predicted)²) | Root mean square error |
| Pearson r | correlation coefficient | Linear correlation between predicted and actual |
