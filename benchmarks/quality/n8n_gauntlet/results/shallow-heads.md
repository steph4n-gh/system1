# Taught nonlinear features do not improve this comparison

The [experiment-2l procedure](../SHALLOW_FEATURES_PROTOCOL.md) was committed and
pushed in `872333784a8cd0bdccde6a23fd0147e0e0aea5c0` before fitting. It teaches
128 ReLU features from the existing fitting examples, joins them with the fixed
384-dimensional encoder output, and compiles System1's existing numerical head
on the resulting 512-vector. **Neither workload advances.** There is no core
runtime change, new teacher call or replacement of the current candidates.

| Full development workload | Matching semantic control: raw correct | Taught nonlinear features: raw correct | Change |
|---|---:|---:|---:|
| CLINC150 supported requests | 2,864/2,995 (95.63%) | 2,864/2,995 (95.63%) | 0 |
| BANKING77 | 1,787/1,960 (91.17%) | 1,782/1,960 (90.92%) | −5 |

The fixed advancement rule requires at least ten additional raw correct
supported decisions and an improvement in the better confidence/margin coverage
frontier. Neither candidate satisfies it. These frontiers select a threshold
on development labels at 99% supported accepted accuracy and at most 1% CLINC
unfamiliar false acceptance:

| Condition | Confidence frontier coverage | Margin frontier coverage |
|---|---:|---:|
| CLINC recorded semantic control | 2,335/2,995 (77.96%) | 2,346/2,995 (78.33%) |
| CLINC taught nonlinear features | 2,224/2,995 (74.26%) | 2,247/2,995 (75.03%) |
| Banking recorded semantic control | 1,223/1,960 (62.40%) | 1,227/1,960 (62.60%) |
| Banking taught nonlinear features | 1,229/1,960 (62.70%) | 1,251/1,960 (63.83%) |

All remain below 80%. The new banking frontier gains slightly but its raw
correctness falls. Strict System1 review alone gives 97.48% accepted supported
accuracy and 18/100 unfamiliar false acceptance for CLINC, and 92.45% accepted
accuracy for banking. Those are failed standalone policies, not replacements
for the existing learned review guard. No complete adapter was built or timed
for these rejected candidates.

## Teaching, retained records and limits

Both workloads receive exactly the labels used by their
[recorded controls](fused-heads.md): original CLINC fitting data, and original
banking fitting data plus the earlier 924 synthetic supported lessons. No new
source, original test or reserved human request is used. Encoder weights remain
fixed and each text projection uses one encoder call.

The feature network runs the declared 100 Adam epochs with one 128-unit layer,
then only its first layer is used as the transform. The output layer is retained
as teaching provenance but is not used by the saved System1 head. Both fits
emit a maximum-iteration warning, preserved verbatim with all 100 training-loss
values. A fixed epoch budget was declared; convergence is not claimed and no
additional fit or epoch selection rescues the result.

Feature teaching takes 4.03 seconds for CLINC and 1.96 seconds for banking.
Subsequent numerical head fits take 2.28 and 2.22 seconds; transforming cached
semantic vectors takes 153.46 and 87.96 ms. Semantic caches were already populated,
so these are not cold end-to-end teaching totals or inference measurements.

The [complete record](shallow-head-development.json) preserves all **5,055 new
development outcomes**, both fits, warning/loss histories, source identities,
strict prediction sets, frontiers and Wilson intervals. The run denied OS
networking and incurred zero teacher calls and zero new API cost. The matching
TF-IDF comparison and all stronger incumbents remain in the
[audit](shallow-head-audit.json); the stronger CLINC baseline's additional 1,696
supported teaching examples remain disclosed.

Both exact saved bundles are published under `artifacts/shallow-clinc150` and
`artifacts/shallow-banking77`. They total 563,580 and 380,773 bytes, respectively,
including the saved head, transform/provenance NPZ and manifest. The external
encoder/tokenizer adds 23,492,300 or 133,804,886 bytes. The manifests identify:

- CLINC: `c819267b80178dba73c18a17187dd2b1bb3bebd62737d528b66996856c46d1a1`.
- Banking: `bb95fb1c74ba4173a7663bcac9e5afa086e6e4a5078a01c749cf022d1d0adb7b`.

The audit recomputes all reported metrics and comparisons, verifies every
development group and label, all 147 earlier frozen files, fifteen declared
sources and both saved bundles. It makes no model decisions. Exported hidden
features match the fitted operation on 32 fitting vectors per workload within
1.5e-8. Reloading each saved head and projector reproduces 16 fixed development
decisions through actual text encoding, with unchanged suggestions, sets and
strict eligibility; maximum confidence difference is below 8e-9. These replays
verify serialization, not quality on a selected subset or latency.

Seven focused shallow/fused feature tests pass, including authored checks for
the nonlinear operation, a zero hidden vector, a single encoder call, immutable
parameters, saved identity and import without optional fitting libraries.
This failed experiment does not resolve the [full regression failure](latest-regression.md),
supply independent full-scope confirmation, or qualify the n8n recording.
The additional transform has not earned a product implementation. PR #3 stays open.
