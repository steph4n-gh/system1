# Real documents: correction workflow result

**Targeted feedback reduced accepted mistakes, but the default adoption gates
rejected every candidate.** This is a measured quality/coverage tradeoff and a
working refusal path, not a successful end-to-end adoption demonstration on
real data or a general classifier breakthrough.

The single run on 2026-09-22 used the [frozen protocol](PROTOCOL.md), an unreleased
`TeachingSession` checkout based on `5c00910`, and real BBC news articles from a
checksum-pinned mirror. All five topics were included. Duplicate/near-duplicate
grouping reduced 2,225 source rows to 2,057 representatives before splitting.
The 400-document balanced holdout was scored only after the three candidates,
their adoption outcomes and hashes were frozen. It never entered a session.

## What happened on the untouched holdout

| Candidate | Raw correct | Accepted correct / accepted | Accepted mistakes | Answered without review | Sent to review |
|---|---:|---:|---:|---:|---:|
| Initial 100 lessons | 369/400 (92.25%) | 357/373 (95.71%) | 16 | 93.25% | 27 |
| Add 64 targeted feedback lessons | 373/400 (93.25%) | 329/337 (97.63%) | 8 | 84.25% | 63 |
| Add 64 ordinary lessons | 372/400 (93.00%) | 310/314 (98.73%) | 4 | 78.50% | 86 |

The targeted candidate halved accepted mistakes while asking for 36 additional
reviews. Its raw accuracy gain was only one percentage point: the paired 95%
bootstrap interval was **−1.25 to +3.5 points**, which includes no improvement.
Correct accepted decisions fell from 357 to 329: fewer mistakes came with less
automatic throughput. This is not an across-the-board win.

Targeted feedback delivered 19 more correct accepted decisions than ordinary
extra lessons, but made four more accepted mistakes. The ordinary control used
the same added-lesson count, not the same label-inspection cost. These results
do not establish that targeted teaching is generally more sample-efficient.

The targeted candidate met the predeclared ≥95% accepted-correctness / ≥80%
coverage target on this holdout. Its accepted-correctness Wilson 95% interval
was **95.39–98.79%**. The initial candidate met the point targets but its lower
bound was 93.15%; the ordinary candidate missed the coverage target. These are
single-run, in-distribution historical news results, not production qualification.

## What the actual workflow allowed

The independent **200 recurring adoption checks** found 8, 4 and 1 accepted
mistakes for the initial, targeted and ordinary candidates respectively. All
failed the unchanged default requirement of **zero accepted mistakes**. No
candidate was adopted and no `current.s1m` was created. The final holdout was
not used to override that decision.

The run verified rejection of failed and stale candidates, unchanged current
artifact state during edits/assessment, and deduplication of repeated lessons.
Since no initial skill qualified, this run preserved an **absent** incumbent;
it did not exercise an already-serving skill or successful public-session
prediction/reopen parity. Those positive paths remain covered by the earlier
automated and authored-demo tests. Candidate files were saved, reloaded and
scored through the existing engine.

This identifies a product-policy question for review: a useful classifier with
a nonzero error budget can still be ineligible under the demonstration defaults.
The API already accepts explicit workload thresholds. We did not relax them,
change core behavior, or search for a passing recipe after seeing results.

## Teaching effort and speed

Each candidate started with 100 lessons, **200 separate calibration examples**
and **200 separate adoption checks**. We inspected a fixed 400-document feedback
batch using the published category as a simulated human answer. Exactly 64
documents were wrong or ambiguous, so all 64 became targeted lessons: 43 were
wrong predictions and 50 were reviewed cases, with overlap between those counts.
These were new feedback examples, not intentionally corrupted initial labels.
This was not a live human teaching study or a classifier taught from just 64 labels.

Building, saving, reloading and assessing the targeted candidate took **188 ms**;
checking its refused adoption took another **59 ms**. Loaded-engine predictions,
including text encoding, had **0.168 ms median / 0.405 ms p95** latency on this
machine. Its artifact was **51,880 bytes**. These are uncached, single-process
local timings, not `TeachingSession.predict()` timings or an LLM speed comparison.
No teacher APIs were called; network connections were blocked during fitting
and scoring. No product dependency or classifier implementation changed.

## Reproduce and inspect

From a development checkout containing this unreleased helper:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/quality/document_workflow/run.py \
  --download --output .system1/document-workflow-reproduction
```

Use a fresh output directory; an existing manifest is never overwritten. This
downloads the pinned CSV and verifies its SHA-256, prepares groups/splits,
runs the exact workflow, freezes candidates and then scores the holdout.
An exit code of zero means the experiment completed with its invariants intact;
it does **not** mean a candidate passed adoption. Subsequent runs reproduce
known evidence; they are not fresh confirmation.

- [Machine-readable summary](results/summary.json): workflow outcomes, all
  metrics, per-class results, intervals, timings and source hashes.
- [Evidence archive](results/evidence.zip): the original full report, split and
  group manifest, pre-prediction provenance, and pre-holdout freeze record.
  Includes hashed document IDs, per-case outcomes and the exact measured runtime,
  runner and protocol source snapshot; no article bodies.
- Raw corpus, text-bearing sessions and candidate artifacts remain under the
  ignored local output directory. The runner pins the source and hashes every
  System1 Python source file because the package version alone does not identify
  this unreleased change.

The source snapshot precedes a review fix that rejects session schemas with
ambiguity escalation disabled. This experiment used the normal enabled setting;
the guard does not change its behavior, and the holdout was not rerun. The
snapshot preserves the exact source corresponding to the recorded hashes.

The corpus is old and drawn from one publisher; text in the mirror is already
lowercased, nine documents were clipped, and the duplicate heuristic cannot
guarantee separation of every related event. Category routing of news does not
validate Finance/People/Projects filing, extraction from PDFs, unfamiliar-topic
rejection, temporal drift, or performance on private company documents. No
parameters were changed in response to this holdout.
