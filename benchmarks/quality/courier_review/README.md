# Courier review recovery: choose immediate fallback

Progress tracking did not earn its complexity in the shipped courier workflow.
Physics advances only after an action, so waiting cannot improve a reviewed
observation. Immediate fallback, progress-after-two and plain bounded waiting
produce identical actions, calls and completion at equal per-map expert budgets.

The optional [strict courier adapter](../../../examples/gaming/skill_playground/review_fallback.py)
provides immediate fallback only. It requires an explicitly configured async
expert, a separate permission callback, an independent current-context validator,
an attempted-call cap and a timeout. Failed, denied, invalid, stale or late
answers stop the episode. A successful fallback keeps statistical `review`
visible; it does not raise confidence, grant permission or teach a label.
The existing playground, core engine, Guard and email defaults are unchanged.

```bash
# Strict stop; no expert configured or called:
python -m examples.gaming.skill_playground.review_fallback --seed 98200

# Explicitly enable the existing cheap local rule expert:
python -m examples.gaming.skill_playground.review_fallback \
  --seed 98200 --enable-rule-expert --expert-budget 8

# Use the already-corrected terrain skill and explicit changed rule:
python -m examples.gaming.skill_playground.review_fallback \
  --seed 98200 --changed --enable-rule-expert --expert-budget 8
```

This loads existing frozen skills into temporary local storage. There are no
downloads, credentials, new dependencies or paid teacher calls. Custom callbacks
must be read-only and cooperate with the event loop. Timeout discards late
results but cannot kill blocking code. The adapter returns proposals for its
caller to dispatch; it provides no durable retry, restart or exactly-once ledger.
Use one runner per episode and call `close()` when finished. The independent
validator must enforce the application's current schema, safety and permissions.
`permitted_move` implements known toy-world rules, not a learned safety guarantee.
If model execution fails before returning telemetry, call count is unknown;
the command reports those turns separately from the known model-call subtotal.

## Preliminary selection

Seeds 98200–98219 provided 40 complete geometry/appearance worlds, but only 19
base object layouts, paired across ordinary and corrected purple terrain.

| Per-map expert cap | Strict stop | Each fallback policy | Actual fallback calls across 40 maps |
|---|---:|---:|---:|
| 0 | 28/40 | 28/40 | 0 |
| 1 | 28/40 | 38/40 | 12 |
| 8 | 28/40 | 40/40 | 14 |
| 256 | 28/40 | 40/40 | 14 |

All hybrid actions, completion and calls matched per map. At cap eight, progress
added 14 waits and plain bounded waiting 42; immediate added none. There were no
dangerous dispatches or deaths. Timing differences largely follow the selected
poll delay and do not establish production savings. All 560 recorded comparisons
were independently replayed and their counters reconciled. These maps selected
the simpler policy and are now regression data.

Direct known rules completed 40/40 with 1,434 cheap native invocations and no
model calls. They were much faster than nine-model-call courier steps. Use direct
rules when they are available and easily maintained. Strict recovery is an
application pattern, not improved classifier quality over the existing simulator
that executes flagged argmax predictions.

## Confirmation and reproduction

An independent reviewer froze the committed implementation and scored seeds
99300–99319 once in both terrain conditions. All hybrid trajectories, model
calls, expert attempts and safety outcomes matched per map at every cap.

| Per-map expert cap | Strict stop | Each fallback policy | Actual fallback calls across 40 maps |
|---|---:|---:|---:|
| 0 | 27/40 | 27/40 | 0 |
| 1 | 27/40 | 35/40 | 13 |
| 8 | 27/40 | 40/40 | 19 |
| 256 | 27/40 | 40/40 | 19 |

No unsafe proposal/dispatch, death or paired difference occurred. These forty
complete worlds use twenty base geometries with paired terrain conditions.
Five role layouts recur in fitting/calibration maps; familiar effective inputs
also recur. This is fresh map-seed execution evidence, not forty independent
new concepts, classifier qualification or real customer traffic.

| Cap-eight policy | Median episode elapsed | Actual rule invocations | Model calls | Waiting polls |
|---|---:|---:|---:|---:|
| Immediate | 24.31 ms | 19 | 14,022 | 0 |
| Progress | 26.36 ms | 19 | 14,022 | 19 |
| Plain bounded wait | 26.89 ms | 19 | 14,022 | 57 |
| Direct known rules, separate cheap baseline | 0.15 ms | 1,410 | 0 | 0 |

Full-suite activity overlapped this shared-host run, and waiting includes chosen
one-millisecond sleeps. These elapsed times are descriptive; they establish no
isolated speedup, production latency saving or paid API benefit. The decision
rests on identical outcomes/resources and unnecessary imposed waits. Direct
known rules remain strongest when their knowledge is available.

The [independent review](INDEPENDENT_REVIEW.md) replayed all 560 comparisons,
checked every native rule answer and reconciled counters and source/skill hashes.
[Summary and freeze](results/SUMMARY.json),
[complete compressed report](results/confirmation.json.gz), and
[machine-readable review](results/INDEPENDENT_REVIEW.json) retain the results.
The gzip SHA-256 is
`a9d00a8127d94ed15bcbfb3c6ad82da006044e21b3e0395ae6cd9783dba4d699`.
Complete trajectories remain in Git; the installable source archive includes
the compact reports and reproducible runner.

After scoring, CI exposed a test collection difference between the console
`pytest` command and `python -m pytest`. The new test module now adds the checkout
root to its import path, matching the existing courier tests. Callback, model,
benchmark and policy code are unchanged. Archived hashes retain the exact source
used for confirmation; this test-setup correction does not create another fresh
quality run or replace the frozen observations.

The [prospective protocol](PROTOCOL.md) fixes the independent seeds, budgets,
controls and selection rule. Run it in a fresh output directory:

```bash
python benchmarks/quality/courier_review/run.py \
  --seed 99300 --maps 20 --output-dir .system1/courier-review-99300
python -m pytest -q tests/test_courier_review_fallback.py tests/test_skill_playground.py
```

The runner freezes runtime/source and skill hashes before scoring and retains
complete actions and counters. Reusing published seeds reproduces observed
evidence; it does not create another held-out cohort. The short progress/wait
comparators live only in this benchmark; no general progress-monitor API enters
the package.

Rollback: stop invoking this module or omit its expert callback.
`StrictCourier(network)` requires review without automatic fallback. Existing
callers continue using their original paths. No saved skill, service
configuration, core policy or model confidence is rewritten.
