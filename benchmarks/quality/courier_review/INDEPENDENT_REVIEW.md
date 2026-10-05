# Independent review: courier strict-review fallback

Reviewed implementation commit `7ad7213e29846f07e9d7ec9a66c72dec32d60e8c` before scoring confirmation seeds 99300–99319. The source, tests and prospective protocol were fixed before this independent run. No fitting, calibration or policy changes followed the scores.

**Conclusion:** no remaining material correctness or safety bug was found within the documented contract: one runner per episode/event loop, trusted read-only asynchronous expert, separate synchronous permission and current-context validator, attempted-call cap, and cooperative timeout. This is a limited example review, not a general security certification. The evidence supports immediate fallback for this loop; progress tracking does not earn its complexity.

## Reviewed behavior and failure controls

The new standalone `examples/gaming/skill_playground/review_fallback.py` preserves the existing playground's defaults and model uncertainty. It returns a proposal without dispatching. A recovered proposal keeps statistical review visible and needs exact-`True` permission and validation in unchanged current context. Denied, missing, failed, invalid, dangerous, blocked, stale or late answers stop the runner. Attempts count failures; a stopped runner cannot retry to reset the budget. Timeout discards late answers and cannot kill arbitrary blocking callback code.

Independent focused verification: **39 tests passed**, covering the new callback and existing courier paths. Seven additional adversarial probes passed: expert self-cancellation, permission/validator exceptions, validator context mutation, permission-triggered close before dispatch, a cancellation-ignoring late answer with task draining, and locked-gate rejection. The review found exception telemetry undercounting and a stopped-report aggregation failure before confirmation; both were fixed and probed before the cohort. Failed model execution now reports unknown calls separately from measured work. No rejected expert data or exception text is exposed as an action.

Independent permissions and `permitted_move` enforce explicit known toy-world rules, including walls and locked gates. No hidden hazard labels, future rewards, teacher answer, or confidence threshold are used to repair local predictions. This example is not a general action-permission boundary or proof that accepted model predictions are always safe.

## Independent confirmation

The public benchmark `benchmarks/quality/courier_review/run.py` was run once on seeds 99300–99319 in both ordinary and corrected-purple conditions, under the committed prospective protocol. Every hybrid had the same per-map cap and 180-action horizon. The known rule teacher and an independent movement validator were identical across policies. The waiting baselines reused a model answer while the complete observation matched; neither was charged redundant inference.

| Per-map cap | Strict stop | Each hybrid | Native fallback calls, each hybrid |
|---|---:|---:|---:|
| 0 | 27/40 | 27/40 | 0 |
| 1 | 27/40 | 35/40 | 13 |
| 8 | 27/40 | 40/40 | 19 |
| 256 | 27/40 | 40/40 | 19 |

All three hybrids matched **every per-map action trajectory, completion, death/stop outcome, expert attempt, native invocation, model call and unsafe dispatch** at every cap. Zero unsafe proposals/dispatches or deaths were observed. At cap eight, immediate fallback added no waits, progress added 19, and plain bounded waiting added 57. These are prescribed scheduling waits, not paid-workload savings.

Direct known rules completed 40/40 with 1,410 cheap native invocations and no model calls. This is the stronger cheap baseline when rules are already available. Recovery is an explicit application pattern; it does not improve classifier quality over the existing simulator that executes flagged argmax predictions.

Independent audit replayed all **560 recorded comparisons** against world physics, reconciled group counters and medians, verified source/skill manifests and exact paired actions/resources, and checked 1,410 direct-rule actions plus 153 hybrid expert actions across all policy/cap arms against the existing teacher. There were zero parity misses. The confirmation report's uncompressed SHA-256 is `436458f967948ccc77cd6631b19bdd918f077d06cd7e03744780d028297200ba`; its freeze SHA-256 is `7382e46c8c9a8e637087b72edf2a1eb6e1ae9a96b8c961fbb5993a9e3a949610`.

## Independence and timing limits

Removing seed metadata leaves 40 distinct complete geometry/appearance worlds, generated from **20 base geometries and 20 base role layouts**. Ordinary/changed terrain are paired conditions, not 40 independent mission concepts. Five role layouts recur in original fitting/calibration map seeds; isolated additional-objective, old evaluation and development cohorts also share some role layouts. Exact complete/base geometries did not overlap the source-declared fitting/calibration, objective-extra, development, previous evaluation or preliminary-selection ranges. Familiar effective model inputs and vocabulary still recur. Seed holdout is not new language understanding or customer qualification.

Full-suite activity overlapped the benchmark on the same host. Wall-clock measurements are descriptive and include the explicitly selected 1 ms poll delays. No isolated latency comparison, portable speedup, paid API saving or production throughput claim follows. Policy selection rests on exact paired outcomes, calls and safety together with known imposed waits.

The implementation owner also reports a supported full-suite run of **1,419 passed, 22 skipped**, including the previously blocked localhost wire cases. That full-suite result is separate from the independent focused tests and confirmation audit above.

## Frozen implementation hashes

| Relative source | SHA-256 |
|---|---|
| `examples/gaming/skill_playground/review_fallback.py` | `98b0b15c0ed240707851cc839819c2c8b8c12f516e163fec68c4b3d40b9f481d` |
| `tests/test_courier_review_fallback.py` | `e8e1af2663745df55b8a7f2ab940b8555e1105812f07fce2ddf4e277c01ce00c` |
| `benchmarks/quality/courier_review/run.py` | `a9cb2878f1070339d81c49405fc302786db977d68dd0039d7fab4624a703656f` |
| `benchmarks/quality/courier_review/PROTOCOL.md` | `23848c3d261b6ff5378a44d3e4ef55f4f417a166572d681f9d4d69802487f776` |

Reviewer attestation: independent agent `workflow_review`, 2026-10-05. This is a textual review record, not a cryptographic signature or merge approval.
