# Input features help CLINC review slightly; qualification remains unfinished

**This is a development gain, not a resolved quality gauntlet.** Giving the review head the input's existing features raises System1 CLINC acceptance from 2,594 to 2,610 cases, with one additional supported mistake. Banking ties the incumbent. Both new conventional baseline variants remain weaker than their retained incumbents. No original-test or reserved-human input was scored.

The [experiment-2g protocol](../CONTEXT_REVIEW_PROTOCOL.md), numerical preflight and implementation were committed and pushed in `55c4cc0bf0102f0df7a616cda263f3efd90f76c2` before fitting. Each classifier reuses its existing input vector in a single added review dot product: 384 encoder dimensions for System1, or its saved sparse TF-IDF representation for the baseline. The original seven standardized review signals and category features remain, with review C=.1. There is no second encoder/classifier, encoder update, new teaching source or API call.

The [complete fitting record](context-development.json) and [complete saved-adapter record](context-runtime.json) retain all four conditions. The four prior linear controls reproduce their original predictions and scores within 3.89e-16, with matching fold groups, parent identities, row counts and negative targets. The full 150/77 scope and all targets remain unchanged. Each percentage below includes a count/denominator and two-sided 95% Wilson interval.

| Workload / method | Supported coverage | Correct among accepted supported | Unfamiliar falsely accepted | Complete p95 |
|---|---|---|---|---:|
| clinc150 / system1 | 2,610/2,995 = 87.15% (85.90–88.30%) | 2,584/2,610 = 99.00% (98.54–99.32%) | 1/100 = 1.00% (0.18–5.45%) | 2.782 ms |
| clinc150 / baseline | 303/2,995 = 10.12% (9.09–11.25%) | 302/303 = 99.67% (98.15–99.94%) | 1/100 = 1.00% (0.18–5.45%) | 0.795 ms |
| banking77 / system1 | 1,581/1,960 = 80.66% (78.86–82.35%) | 1,566/1,581 = 99.05% (98.44–99.42%) | No native cohort | 4.275 ms |
| banking77 / baseline | 872/1,960 = 44.49% (42.30–46.70%) | 864/872 = 99.08% (98.20–99.53%) | No native cohort | 0.596 ms |

## Selection and interpretation

CLINC System1 accepts 16 more supported requests than its prior incumbent: 15 additional correct decisions and one additional mistake in aggregate. Coverage rises by 0.53 percentage points, from 86.61% to 87.15%, while supported accepted accuracy falls from 99.04% to 99.0038%, only just above the 99% point target. It still falsely accepts 1/100 development unfamiliar requests. Raw intent correctness is unchanged at 2,864/2,995. The declared maximum-coverage procedure selects the new review head; this does not establish better generalization or improved unfamiliar-input rejection.

Banking accepts 1,581/1,960 with 1,566 correct, tying the selected quadratic review from experiment 2f. The tie rule retains that incumbent. The new head accepts two more requests than the earlier linear control, but does not improve the best retained coverage. Raw intent correctness stays at 1,787/1,960.

The new lexical review accepts 303 CLINC cases versus its linear control's 315, and 872 banking cases versus 870. Neither approaches its strongest incumbent: the boundary-taught CLINC baseline retains 1,770/2,995 (59.10%) coverage; the quadratic banking baseline retains 1,015/1,960 (51.79%). Those stronger baselines remain the comparison. The CLINC incumbent had additional experiment-2e generated lessons; this fixed-parent experiment did not.

All four new child manifests are published unchanged. Their `base_artifact` and parent hashes bind the original saved heads/prototypes, vocabulary/IDF or encoder identity. The new selected [CLINC System1 artifact](../artifacts/clinc150-system1-context/manifest.json) has SHA-256 `d63b1376cbe8f47c14488d679383e35b91ec6b598f1297de60e78b654b22d45d`. Load System1 children with `ContextCandidate` from `context_runtime.py`, and the conventional children with `ContextBaseline` from `context_review.py`. The existing n8n workflows remain bound to their originally demonstrated artifacts; no unqualified replacement was silently installed.

## Measurements, teaching cost and checks

All 10,110 complete single-request adapter decisions match selection's suggested labels and accept/review routing. They contain zero teacher calls and runtime errors, with OS networking denied, no response cache and no receipts. Each condition had 100 untimed development warmups, then one timed request per original development row. Runtime includes feature extraction, intent prediction, review and response construction. No timing run was repeated to improve its result.

The [arithmetic and source audit](context-audit.json) independently recomputes the retained counts, intervals, accepted errors, per-intent metrics and latency percentiles. It verifies all 125 earlier frozen files, nine recorded source files and every public child/parent identity. Selection and runtime review scores differ by at most 0.00000590, without changing any routing decision. Saved System1 inference [replays with SciPy/scikit-learn imports blocked](context-portability.json): 100 recorded responses per new artifact, two changed contracts reviewed and six malformed inputs rejected, with networking denied. This replay makes no latency claim.

| Workload / method | Context dimensions | Parent plus child bytes | New child bytes | External encoder/tokenizer bytes | Load | Review fit |
|---|---:|---:|---:|---:|---:|---:|
| clinc150 / system1 | 384 | 17,383,260 | 21,580 | 23,492,300 | 278.7 ms | 36.1 ms |
| clinc150 / baseline | 27,253 | 29,433,928 | 735,558 | 0 | 89.6 ms | 60.6 ms |
| banking77 / system1 | 384 | 10,067,598 | 18,413 | 133,804,886 | 304.3 ms | 16.2 ms |
| banking77 / baseline | 20,761 | 11,983,849 | 509,410 | 0 | 35.4 ms | 31.5 ms |

Review-fit timings exclude review-lesson construction and feature preparation. Original-only temporary fold heads were rebuilt for correctness labels, while the final intent heads stayed unchanged. Preparing the review lessons took 5.21/16.90 seconds for CLINC System1/baseline and 3.67/7.87 seconds for banking; existing System1 feature caches were reused. Additional input-context preparation took 37.2/98.8/17.4/64.2 ms in the table's order. These cached research timings are not cold end-to-end teaching claims. Runtime libraries are additional dependencies, excluded from artifact sizes.

This round makes zero new API calls and incurs zero new API cost. Both banking parent heads retain the earlier 924 generated lessons; previous teaching costs remain disclosed in the earlier reports. Correctness targets use original-only cross-validation folds plus separate original calibration rows; generated examples never enter the fold heads or prototype sets. Both CLINC methods use the same 2,000 pinned Wikipedia negatives. The baseline's context vector uses the saved final vocabulary, fitted on the parent fitting text, while correctness labels still come from classifiers excluding the held-out fold.

Measurements used the Apple M4 Pro desktop, one BLAS thread, MiniLM with one encoder thread and BGE-small with four. Packages are pinned in the runtime record. The original runtime field `process_import_preflight_ms` records 128.3 ms starting at the shared measurement-module import; it excludes earlier imports by the context runner and is **not full startup time**. A separate [fresh-process startup check](context-startup.json) records 652.3 ms for the full module import/network preflight and 728.1 ms including Python startup/shutdown. Neither number includes candidate loading. This limitation is retained explicitly rather than replacing the original record. Incumbent decision timings come from prior runs and are not a controlled speed comparison.

Fifteen focused context/polynomial tests pass locally, including numerical oracle cases with dense and sparse teaching matrices, JSON reload and invalid-parameter checks. Optional-library import tests run without fitting dependencies; core CI skips numerical oracle cases when scikit-learn is absent. All six CI jobs pass on the preregistered implementation commit. The shared baseline scoring refactor also [preserves 400 prior recorded responses](context-adapter-preflight.json) before any new fit.

After preparing pinned data and encoders, reproduce in a fresh directory:

```bash
sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  python benchmarks/quality/n8n_gauntlet/context_review.py fit \
  --folder .system1/n8n-gauntlet/context-reproduction
sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  python benchmarks/quality/n8n_gauntlet/context_review.py measure \
  --folder .system1/n8n-gauntlet/context-reproduction
python benchmarks/quality/n8n_gauntlet/audit_context_review.py
sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  python benchmarks/quality/n8n_gauntlet/check_polynomial_portability.py --context
```

Repeated development selection is optimistic. This result does not supersede the [observed regression failure](observed-regression.md), supply independent full-scope confirmation, or authorize the qualified recording. The quality objective and PR #3 remain open.
