# Reflex Decision Engine — Quality Benchmark Suite

## Overview

This benchmark suite measures the **decision quality** (accuracy, precision, recall, F1, MAE, RMSE, correlation) of the Reflex non-autoregressive decision engine. It complements the existing latency benchmarks by providing the first published accuracy/quality metrics.

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

## Output

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
