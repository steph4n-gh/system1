# Stable release checklist

The product promise is: observe a teacher, teach a bounded skill, validate it,
take over locally, and save and reuse the skill. This checklist tracks the single
stabilization effort following 0.2.2. An unchecked item is not release evidence.

- [x] Review and fix the supported core lifecycle: Choice, Boolean/Noul,
  MultiChoice, continuous and ordinal scores, calibration, correction, persistence,
  automatic promotion, per-schema isolation, teacher failures, and drift behavior.
- [x] Make support triage, model routing, and operation triage useful with explicit
  teaching and separate evaluation examples, with development reuse disclosed and
  a fresh operation-triage confirmation cohort. Record accuracy, accepted accuracy,
  acceptance, confidence intervals, teaching time, artifact size, and local latency.
  Routing targets: at least 95% accepted accuracy and 80% acceptance. Keep existing
  evaluation labels fixed; disclose authored data and avoid claims of universal
  language understanding or security detection.
- [x] Observe real Jev responses, validate, disconnect the teacher, and compare
  unseen local answers with both Jev and separately specified expected labels.
  Record actual successful/failed calls and costs only when returned by the API.
  Keep credentials out of artifacts. Verify no local-mode network calls.
- [x] Verify all supported saved/reloaded outputs and uncertainty, safe correction
  followed by revalidation, and explicit permission-denial enforcement.
- [x] Review documented integrations and every example; repair supported behavior
  and clearly identify experimental or scripted demonstrations.
- [x] Pass targeted bug regressions, the existing full test suite, package checks,
  fresh-install quickstarts, and old-skill compatibility checks.
- [x] Update documentation, supported API boundaries, migration guidance,
  and measured release results.
Release gate: require green CI on the exact release commit before tagging
v1.0.0 and verify the triggered PyPI publication. The authoritative status is in
[GitHub Actions](https://github.com/steph4n-gh/system1/actions), checked after the
release commit is pushed; local checks alone do not fulfill it.

Execution uses the existing compiler, runtime, examples, pytest suite, and GitHub
workflows. No new runtime dependency or provider framework is planned. Git author
and committer identity for this repository is `steph4n-gh`.


Evidence: [1.0 release report](releases/1.0.md), [quality reproduction](../benchmarks/quality/results/stable_release.json),
[live handoff](../benchmarks/quality/results/jev_live_cutover.json), and
[typed API checks](../benchmarks/quality/results/jev_typed_contract.json).
Local verification includes the full pytest suite, wheel/sdist and strict Twine
checks, fresh-wheel examples and README policy execution outside the checkout,
and cross-version loading against the published 0.2.2 package. CI repeats the
suite on Linux/Python 3.11–3.14 and macOS/Python 3.13, plus package and quality checks.
