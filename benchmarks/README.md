# Benchmarks and evidence

Use the [quality suite](quality/README.md) for current evaluation commands and the
[public-workload report](quality/workloads/README.md) for banking, assistant and
SMS results, the classical baseline and actual Jev/Gemini takeover recordings.
All reported quality must distinguish raw correctness, local acceptance and
correctness among accepted decisions. Failed targets stay in the evidence.

[run_triple_crown_benchmark.py](run_triple_crown_benchmark.py) is a historical
three-domain simulation inspired by OpenHands, Instructor and Semantic Router
use cases. It does not execute those packages or establish drop-in compatibility
with them. Its cloud baseline timings, nominal labels and token costs are
synthetic. The archived [scorecard](results/triple_crown_scorecard.json) retains
that provenance and its historical numbers; its speedup and savings fields must
not be quoted as measured provider results.

The current workload runner compares against an actual local TF-IDF/logistic
regression baseline. Live teacher observations are recorded separately, and
teacher/local timings on different inputs are not presented as matched speedups.
