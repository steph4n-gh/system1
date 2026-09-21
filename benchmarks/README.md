# Benchmarks

The [full-scope n8n gauntlet](quality/n8n_gauntlet/README.md) covers all 150 CLINC
and 77 BANKING77 intents. Quality gates still fail; development improvements and
the real teacher-disconnection rehearsal do not establish qualification. Full
artifacts and raw results stay in Git and are excluded from Python distributions.

The [1.0.2 quality round](quality/quality_round/README.md) adds stricter qualification,
a measured SMS takeover improvement and fresh authored failures. System 1 remains
a small local decision runtime, not an LLM.

The unreleased [real-email experiment](quality/public_email/README.md) adds public
SpamAssassin teaching and evaluation with no credentials: 603/618 accepted
decisions correct on a representative grouped split, with a separate failed
source-shift experiment retained. Neither result qualifies seven-category routing.

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
