# Changelog

## 1.0.1 — 2026-09-19

- Add 60 routing calibration cases that move two of three previously accepted mistakes to review, preserving fitted weights and raw predictions. Retain the fresh 48-case result that still misses the 95% accepted-correctness target.
- Add a JSON teach/save/local-reuse banking-support demo using original BANKING77 labels and a fixed three-intent official test slice: 118/120 accepted, 115/118 correct. Include source provenance, the CC BY 4.0 data license, and reproducible split verification.
- Add six routing and 28 operation teaching lessons for close distinctions. Preserve the original Jev observations and historical comparisons separately from new local teaching.
- Improve Pokémon's legal move selection and immediate damage/healing rules, and Paperclips' wire-unit pricing, purchase affordability, and handling of failed observations. Report game-rule gains separately from raw classifier results.
- Verify calibration results, untouched banking evaluation, and exact saved/reloaded behavior offline in the existing test suite. No new runtime dependencies or public API changes.

See [release evidence and remaining limitations](docs/releases/1.0.1.md).

## 1.0.0 — 2026-09-19

- Stabilize the bounded-skill lifecycle: teach or observe, validate, run locally, save, reload, and correct with fresh calibration. No new runtime dependencies.
- Use standard LAC uncertainty scores for examples-only teaching, preserving legacy APS behavior. Separate temperature and conformal evidence for categorical, Boolean and MultiChoice outputs; allow a validated empty MultiChoice assignment.
- Persist recalibration, invalidate changed heads after corrections, serialize model operations coherently, and prevent approximate compiled-cache matches from changing distinct inputs. Bind cached-decision receipts to the current request and honor receipt opt-out.
- Require complete decisions and independent evidence groups for promotion. Check agreement and useful local acceptance with exact binomial bounds, fresh validation blocks, and a finite error budget across repeated attempts. Keep drift state per schema and reject malformed teacher answers before recording evidence.
- Write `.s1m` format v2 with explicit score semantics and validated metadata. Read v1 files with their original APS behavior; older readers must be upgraded for v2.
- Expand the three primary teaching examples and publish reproducible quality, review rates, confidence intervals, and actual Jev comparisons. Add the offline comparison to existing CI. Preserve development history and remaining accepted errors.
- Verify a live Jev automatic takeover after 357 observations, followed by 40/40 correct accepted local decisions and identical reloaded behavior. Verify live mixed Choice/Noul/ordinal Score responses.
- Remove inflated default guard calibration and unsupported simulation performance claims. Keep tool permission under explicit policies; keep optional neural and gaming examples experimental.

See [release evidence and migration](docs/releases/1.0.md).

## 0.2.2 — 2026-09-19

- Preserve the complete validated skill through takeover and reload: projector settings, schema identity, calibration, distributions and review behavior. Gate normal promotion on useful local acceptance as well as teacher agreement.
- Make observed-only teaching and strict review the TypeSafe adapter defaults. Group related observations across all supplied lineage identifiers; insufficient evidence continues using the teacher.
- Support SDK context managers, JSON object/array state, typed response accessors and ordinal score distributions. Document the supported contract and reject unsupported transport/response options.
- Correct strict conformal sets to invert their calibrated cumulative-probability scores; retain conservative review when calibration is insufficient.
- Add the complete offline observation-to-local-reuse example, with optional real Jev observation. Remove fabricated speedups after HTTP failures, counter-only campaign takeover, reconstructed probability distributions and unsupported example claims.

- Teach a skill from supplied examples with `compile(..., augment=False)` or CLI `--dataset`, rejecting malformed labels instead of inventing replacements. Keep repeated prompts out of separate calibration partitions and preserve uncalibrated status when no calibration examples exist.
- Load saved skills directly with `system1 decide --model skill.s1m`, using strict uncertainty gating. Add a small teaching example and reproducible comparison on unseen examples.
- Replace the three primary showcases with focused, offline teaching demonstrations: explicit labeled datasets, separate calibration and evaluation cases, saved skills, and measured quality, review frequency, size, and speed. Keep agent permissions under deterministic policy rules.
- Propagate LangChain callback denials and approval requirements through synchronous and asynchronous tool dispatch. Preserve the configured principal instead of substituting a run ID.
- Handle non-object JSON, malformed multipart text, disconnects, and one-time body replay in the ASGI gateway; respect structured escalation signals.
- Return actual receipt digests in gateway headers and MCP errors.
- Add real LangChain dispatch regression tests, built-distribution checks, Python 3.14 CI coverage, and release-tag validation.
- Keep legacy proto assets accessible without optional gRPC/protobuf dependencies in minimal wheel installs.
- Update package build requirements and the lockfile; include examples and test support files in source distributions.
- Replace unsupported launch claims with reproducible benchmark results, executable quickstarts, and explicit deployment boundaries.

## 0.2.1

Previous published baseline. See the [repository history](https://github.com/steph4n-gh/system1/commits/main/) for earlier changes.
