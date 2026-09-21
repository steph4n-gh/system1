# Boundary teaching comparison: existing System1 candidates retained

**The new lessons do not win the declared System1 selection.** CLINC raw correctness improves, with tradeoffs in accepted accuracy and coverage; banking coverage drops below 80%. The conventional CLINC baseline improves substantially. These are development comparisons, not independent qualification or evidence that the original failures have been resolved.

The [procedure](../BOUNDARY_TEACHING_PROTOCOL.md) was declared before generation. The [recorded teacher data](boundary-teaching.md) and exact comparison implementation were committed and pushed in `110dc7f36ea82de48f92a5c4d1c47b76cd88710b` before fitting. All encoder weights, original fold boundaries, quality targets and fixed parameter choices were preserved. The [complete fit/selection report](boundary-development.json) and [complete runtime report](boundary-runtime.json) retain every condition, outcome, accepted error, per-intent result, threshold, identity and timing.

Each table cell includes count, point estimate and two-sided 95% Wilson interval. Supported accepted accuracy and unfamiliar acceptance have separate denominators. These are selected development estimates, not future error guarantees.

| Workload / method / new lessons | Supported coverage | Correct among accepted supported | Unfamiliar falsely accepted | Complete p95 |
|---|---|---|---|---:|
| clinc150 / system1 / positive-only | 2,582/2,995 = 86.21% (84.93–87.40%) | 2,561/2,582 = 99.19% (98.76–99.47%) | 1/100 = 1.00% (0.18–5.45%) | 2.764 ms |
| clinc150 / system1 / positive-and-unsupported | 2,472/2,995 = 82.54% (81.14–83.86%) | 2,456/2,472 = 99.35% (98.95–99.60%) | 1/100 = 1.00% (0.18–5.45%) | 2.816 ms |
| clinc150 / baseline / positive-only | 1,144/2,995 = 38.20% (36.47–39.95%) | 1,143/1,144 = 99.91% (99.51–99.98%) | 1/100 = 1.00% (0.18–5.45%) | 0.904 ms |
| clinc150 / baseline / positive-and-unsupported | 1,770/2,995 = 59.10% (57.33–60.85%) | 1,762/1,770 = 99.55% (99.11–99.77%) | 1/100 = 1.00% (0.18–5.45%) | 0.869 ms |
| banking77 / system1 / positive-only | 1,548/1,960 = 78.98% (77.12–80.73%) | 1,533/1,548 = 99.03% (98.41–99.41%) | No native cohort | 4.353 ms |
| banking77 / baseline / positive-only | 725/1,960 = 36.99% (34.88–39.15%) | 718/725 = 99.03% (98.02–99.53%) | No native cohort | 0.683 ms |

## What changed

CLINC System1 raw correctness rises from 2,864/2,995 (95.63%) to 2,878/2,995 (96.09%). With positive examples alone it accepts 2,582 requests, 12 fewer than the incumbent, while accepted correctness rises from 99.04% to 99.19%. Adding the unsupported teaching examples accepts 2,472 requests at 99.35% supported accepted accuracy. Both still accept 1/100 development unfamiliar requests. Their higher accepted accuracy comes with lower coverage; the preregistered maximum-coverage selection retains the incumbent. That selection does not prove the incumbent generalizes better.

Banking System1 keeps the same total raw correctness, 1,787/1,960 (91.17%), but accepts 1,548 requests instead of the incumbent’s 1,580. Coverage is 78.98%, below 80%. The existing banking candidate remains selected.

The CLINC conventional baseline improves from 882/2,995 (29.45%) accepted to 1,770/2,995 (59.10%) when given the additional supported and unsupported teaching. It gets 1,762 accepted supported cases correct and falsely accepts 1/100 unfamiliar inputs. It remains faster, but fails the 80% coverage target. The banking baseline accepts 725 requests instead of 870, so its incumbent remains selected. All these gains and regressions are retained; the baseline is not frozen at its earlier weaker result.

The comparison therefore does not justify further bulk generation using this same procedure as a demonstrated System1 fix. It also does not establish that generated teaching never helps: CLINC raw prediction improves, and its supported accuracy/coverage tradeoff changes. Generated unsupported labels remain unverified. No claims are made about why individual lessons help or hurt without a separate diagnostic.

## Runtime, portability and accounting

All 16,300 complete saved-adapter development decisions match their selection predictions and review outcomes. Each condition received 100 untimed development warmups, then one timed request per original development row. OS networking was denied; there were zero teacher calls, zero runtime errors, no response cache and no receipts. Each request includes encoding/vectorization, intent prediction, review and response construction. A separate [arithmetic and preservation audit](boundary-development-audit.json) verified the labels/groups, decisions, metrics, intervals and timing percentiles and confirmed all 125 previously frozen files remain unchanged.

These runs used the same Apple M4 Pro desktop, MiniLM with one encoder thread and BGE-small with four, and one BLAS thread. All fitting features used individual-request projection. Runtime/library versions and import/preflight time are in the full runtime record. Ordinary desktop applications remained open; the live teacher process had finished before these timing measurements. Incumbent timings come from retained earlier runs and are not a controlled speed comparison.

| Workload / method / condition | Artifact bytes | Extra encoder/tokenizer bytes | Load | Intent fit | Review fit |
|---|---:|---:|---:|---:|---:|
| clinc150 / system1 / positive-only | 19,777,106 | 23,492,300 | 290.8 ms | 1692.8 ms | 15.2 ms |
| clinc150 / system1 / positive-and-unsupported | 19,777,116 | 23,492,300 | 249.4 ms | 1692.8 ms | 20.5 ms |
| clinc150 / baseline / positive-only | 35,717,455 | 0 | 91.4 ms | 8871.5 ms | 15.6 ms |
| clinc150 / baseline / positive-and-unsupported | 35,717,463 | 0 | 91.0 ms | 8871.5 ms | 27.0 ms |
| banking77 / system1 / positive-only | 11,357,781 | 133,804,886 | 313.0 ms | 2130.9 ms | 4.7 ms |
| banking77 / baseline / positive-only | 14,313,568 | 0 | 38.3 ms | 5154.2 ms | 4.7 ms |

Intent-fit and review-fit times exclude feature preparation and earlier cross-validation work. Conditions sharing one intent fit repeat that same measurement in the table; they are not independent refits. The complete fit report records feature preparation/cache use and cumulative development elapsed time separately. Runtime libraries are additional dependencies, excluded from artifact sizes.

The 227-call teacher run used 333,462 tokens and 516 seconds, with a published standard-price estimate of $0.2777502 and unknown actual billing. Its 3,287 retained lessons include 1,696 CLINC supported, 674 CLINC unsupported and 917 banking supported examples. The banking fits also retain the previously recorded 924 generated lessons. The current fitting/runtime comparison adds no API calls or charges. Every generated lesson is excluded from the original-only review fold heads and prototypes; calibration and development boundaries remain intact.

The new selected conventional CLINC artifact is [saved here](../artifacts/clinc150-baseline-positive-and-unsupported/manifest.json), with manifest SHA-256 `3fc70d9b9073502e85bd1306f0e9f67eb7537c7eaa1c7e1cabe61ef27ab7ba98`. Load it with `BoundaryBaseline` from `boundary_development.py`. Its public files are byte-identical to the measured files. The other newly fitted artifacts remain in the ignored experiment directory; the frozen code, original sources and public teacher responses reproduce their fits. All six identities and complete decisions are in the public reports. The existing System1 artifacts and banking baseline remain selected and published under their original identities.

After preparing the pinned data/encoders, reproduce all fixed fits and runtime decisions in a fresh directory:

```bash
sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  python benchmarks/quality/n8n_gauntlet/boundary_development.py fit \
  --folder .system1/n8n-gauntlet/boundary-development-reproduction
sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  python benchmarks/quality/n8n_gauntlet/boundary_development.py measure \
  --folder .system1/n8n-gauntlet/boundary-development-reproduction
```

No official test or reserved-human input was scored in this experiment. The [observed regression failure](observed-regression.md) remains the larger-cohort quality evidence. Independent full-scope confirmation and the qualified n8n recording are still missing. The next investigation must address the remaining quality gap without treating more generated examples or another development pass as proof of success. PR #3 remains open.
