# Direct OOS teaching did not improve the selected skill

Experiment 2i was declared and pushed in `b31c7db34a6f112abe1e0167a78c4c0bb96f34b4`
before fitting. Adding 674 synthetic unfamiliar examples to the existing OOS
intent improves raw unfamiliar recognition, but reduces supported correctness
and useful coverage. **Keep all four incumbents.** This is development evidence;
the [latest observed regression](latest-regression.md) still fails, and fresh
full-scope confirmation and a qualified disconnection recording remain missing.

## Complete development outcomes

Both methods receive the same original fitting rows, 1,696 supported synthetic
lessons and, in the direct-OOS condition only, 674 unfamiliar synthetic lessons.
Review uses original-only cross-fitting, calibration correctness and the same
2,000 Wikipedia negatives. No generated OOS example directly teaches review.
The fixed encoder, logistic heads, saved formats and runtime loaders are unchanged.
All generated labels remain unverified teaching material. See the
[prospective protocol](../DIRECT_OOS_PROTOCOL.md).

| Method / condition | Supported accepted | Correct among accepted | OOS falsely accepted | Complete p95 |
|---|---:|---:|---:|---:|
| System1 positive control | 2,582/2,995 (86.21%) | 2,561/2,582 (99.19%) | 1/100 | 2.727 ms |
| System1 direct OOS | 2,537/2,995 (84.71%) | 2,518/2,537 (99.25%) | 1/100 | 2.877 ms |
| TF-IDF positive control | 730/2,995 (24.37%) | 729/730 (99.86%) | 1/100 | 0.791 ms |
| TF-IDF direct OOS | 724/2,995 (24.17%) | 723/724 (99.86%) | 1/100 | 0.868 ms |

Each condition measured all 3,095 development requests after 100 warmups, with
OS networking denied, no response cache and no receipts: **12,380 recorded
decisions, zero runtime/selection routing differences, zero inference errors,
zero new teacher calls and zero new API cost**. These point targets and thresholds
were selected on development data; they are not independent guarantees.
All outcomes, accepted mistakes, per-intent counts and Wilson intervals are in
[the recovered runtime report](direct-oos-runtime.json).

System1's raw OOS predictions improve from 48/100 to 78/100. However, supported
inputs incorrectly predicted OOS rise from 5 to 40, and raw supported correctness
falls from 2,878/2,995 to 2,842/2,995: five previous mistakes are fixed and 41
previously correct predictions are broken. The two conditions falsely accept
the same single unfamiliar request after review. This is a rejection tradeoff,
not a solution to the gauntlet. The lexical comparison shows the same direction:
raw OOS recognition rises from 25 to 58, while supported-to-OOS errors rise from
11 to 87. Its raw supported correctness falls from 2,720 to 2,664.

The unchanged System1 control reconstructs the experiment-2e positive-only
control exactly, including all routing and scores (maximum score difference 0).
The lexical control differs from 2e by receiving Wikipedia review negatives too;
its earlier stronger artifact remains selected. The existing CLINC selections
accept 2,610/2,995 for System1 and 1,770/2,995 for TF-IDF. Neither new condition
beats them. All banking candidates, data, failures and selections remain unchanged.

## Reporting failure and recovery

The original runtime command exited 1 **after all measurements**, while assembling
the incumbent comparison: the older polynomial report's folder name differs from
its published artifact path. Its final JSON file was empty, but its flushed
12,380-row journal and all four printed metric/timing summaries were intact.
The [failure log](direct-oos-runtime-failure.log) is retained. Recovery verifies
every recorded latency and outcome against those summaries and the fit selection;
it performs **zero new model decisions**. The corrected lookup uses the frozen
published-candidate map, and future runs checkpoint every completed condition.
Fitting functions are AST-identical to the declaration commit.

The original process import and per-candidate load timings were lost and remain
`null` in the recovered report. A separately labeled [fresh-process probe](direct-oos-startup.json)
records 620.09 ms for imports/network preflight and candidate loads of
281.71 / 248.27 / 92.29 / 93.30 ms in table order. It makes no decisions. These
are not replacements for the original missing timings. Environment metadata and
the explicit network probe in the recovered report were also captured during
recovery; the original run passed its mandatory network-denial check before
inference. No warm timing was repeated to improve its number.

## Teaching, artifacts and verification

The [fit report](direct-oos-development.json) preserves all four fits and feature
preparation evidence. Intent fits took 1.65 / 2.58 / 8.56 / 10.45 seconds; review
fits took 14.94 / 11.37 / 21.58 / 24.88 ms in table order. Total development
preparation/fit/selection took 7.33 / 22.92 / 26.69 / 28.70 seconds. These totals
use existing feature caches where available and are **not cold teaching costs**.
The direct-OOS System1 condition newly encoded 14,389 teaching rows in 14.60 s.
The original 227-call generation cost remains in the
[teacher record](boundary-teaching.md); this experiment incurs none.

Complete standalone artifact sizes are 19,777,111 / 20,737,431 / 35,717,457 /
37,047,118 bytes. Each System1 artifact additionally needs the same 23,492,300-byte
external encoder/tokenizer. Both exact System1 artifacts are published in
`artifacts/clinc150-system1-positive-control` and `artifacts/clinc150-system1-direct-oos`.
Both losing baseline manifests, identities and complete outcomes are published;
their full artifacts remain in the ignored experiment directory and can be
regenerated with the committed recipe. They do not replace the stronger baseline.

The [record audit](direct-oos-audit.json) verifies all 147 frozen files, nine
declared source identities, original development rows, lineage counts, arithmetic,
routing, saved artifacts and unchanged selection. It verifies the historical
runner from Git because the reporting-only repair follows the failed run.
Selection/runtime review scores differ by at most 3.981e-6 for System1 and
4.45e-16 for TF-IDF; routing matches throughout. The
[portability check](direct-oos-portability.json) replays 100 recorded responses
per System1 artifact with SciPy/scikit-learn imports blocked, checks two changed
contracts and six malformed inputs per artifact, and makes no new latency claim.
No original test or human reserve was scored. PR #3 remains open; no release.
