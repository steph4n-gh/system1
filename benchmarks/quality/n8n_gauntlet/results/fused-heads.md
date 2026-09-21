# Word features help CLINC raw accuracy, but hurt banking

The fixed [experiment-2j procedure](../FUSED_FEATURES_PROTOCOL.md) was committed
and pushed in `f008fbfe0d16711604cbcd54fee03ed091c9f450` before fitting. It combines
the frozen 384-dimensional semantic encoder with System1's existing portable
2,048-dimensional TF-IDF word features. One existing logistic numerical head
learns from their normalized concatenation. There is no extra encoder, teacher
call, pretrained-weight update, new dependency or core API change.

| Workload | Semantic control: raw supported correct | Semantic + words: raw supported correct | Change | Advance to full review comparison? |
|---|---:|---:|---:|---|
| All CLINC150 supported intents | 2,864/2,995 (95.63%) | 2,889/2,995 (96.46%) | +25 | Yes |
| All BANKING77 intents | 1,787/1,960 (91.17%) | 1,773/1,960 (90.46%) | −14 | No |

For CLINC, raw OOS recognition also rises from 51/100 to 60/100. These are raw
suggestions, not final accepted accuracy or guarantees. Banking's regression is
retained and the preregistered advancement rule rejects its fused candidate.
**Neither workload's current integration candidate is replaced by this probe.**
The full [observed-test regression failure](latest-regression.md) remains current.

The result does not establish a passing rejection policy. With strict System1
review alone, CLINC falsely accepts 17/100 unfamiliar requests for the control
and 11/100 for fusion; both miss 1%. Applying confidence or margin thresholds on
development data at the quality targets yields these optimistic best coverages:

| Workload / condition | Confidence frontier | Margin frontier |
|---|---:|---:|
| CLINC semantic control | 2,335/2,995 (77.96%) | 2,346/2,995 (78.33%) |
| CLINC semantic + words | 2,108/2,995 (70.38%) | 2,063/2,995 (68.88%) |
| Banking semantic control | 1,223/1,960 (62.40%) | 1,227/1,960 (62.60%) |
| Banking semantic + words | 1,212/1,960 (61.84%) | 1,222/1,960 (62.35%) |

The same counts result with the additional strict-eligibility restriction. All
are below 80%; the CLINC raw gain comes with worse simple-threshold coverage.
These are development-selected frontiers, not independently qualified policies.
They do not include the existing learned review guard. A separately declared
complete review/adapter comparison is justified for CLINC by the prespecified
raw-gain rule; it may still fail. Banking's complete scope and failed quality
remain required, with its incumbent unchanged.

## Teaching and comparison boundaries

CLINC uses its original 12,019 fitting rows, including native OOS. Banking uses
its original 6,026 fitting rows plus the earlier 924 synthetic supported lessons.
Both conditions receive identical labels. Vocabulary and IDF see fitting text
only. No experiment-2e/2i additions, original-test inputs or human reserves enter
these fits. Encoders use individual-request projection; no encoder weights change.

The recorded TF-IDF/logistic baseline with matching intent-teaching labels gets
2,692/2,995 CLINC and 1,699/1,960 banking raw supported decisions correct. It is
retained without redundant refitting or new timing. The stronger selected CLINC
baseline has 1,696 extra experiment-2e supported lessons and gets 2,720/2,995 raw
correct; that additional teaching is not hidden. All stronger baseline selections
and the failed joint quality result stay unchanged.

The [complete report](fused-head-development.json) retains **10,110 development
outcomes**, every raw suggestion, strict eligibility, prediction set, confidence,
margin, coverage frontier, Wilson interval, source hash and artifact identity.
Fits and verification ran with OS networking denied: zero teacher calls and zero
new API cost. No complete-adapter latency was measured; cached feature scoring
must not be presented as end-to-end speed.

Head fit times were 1.70 / 4.12 seconds for CLINC control/fusion and 1.79 / 3.43
seconds for banking. Additional fused feature preparation took 332.37 / 221.83 ms
for CLINC / banking. These exclude semantic feature preparation: cached vectors
were reused where present, while 6,950 banking teaching rows were newly projected
in 15.53 seconds. These are not cold end-to-end teaching totals.

The four saved head/configuration bundles are 233,237 / 1,454,112 / 123,754 /
793,554 bytes in CLINC-control, CLINC-fusion, banking-control, banking-fusion order.
The external encoder/tokenizer adds 23,492,300 bytes for CLINC or 133,804,886 bytes
for banking. All four exact bundles, including the losing banking candidate, are
published under `artifacts/fused-{dataset}-{condition}`. They are intent heads,
not complete qualified adapters with learned review.

Reloading every saved head and reconstructed projector reproduces suggestions,
strict eligibility and conformal sets on 16 fixed development inputs each.
Maximum top-probability difference is below 2e-8. These 64 replays verify saving
and projection, not quality on a selected subset and not latency. Three authored
tests also verify feature weighting, unknown-word behavior, serialized vocabulary
identity, recency rejection and import without optional libraries.

The [record audit](fused-head-audit.json) verifies all 10,110 recorded outcomes
against original development groups/labels, recomputes metrics and frontiers,
checks all four bundles and eight declared sources, and confirms all 147 earlier
frozen files are unchanged. The next steps remain full review/runtime validation,
resolution of both workloads' failures, independent full-scope confirmation and
the qualified n8n disconnection recording. PR #3 stays open; no release.

The subsequent [complete review and adapter check](fused-review.md) has now
finished: 79.30% supported coverage fails the unchanged 80% target, despite
99.66% accepted accuracy and 2.80 ms p95. The candidate does not replace the
incumbent. The raw gain above was insufficient for a useful coverage gain.
