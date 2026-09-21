# Changelog

## Unreleased

- Add a real n8n development workflow bound to a saved CLINC candidate, with item pairing, explicit review and local-service failure checks. Freeze full-scope candidates and conventional baselines before official evaluation. Retain the failed held-out result: CLINC reaches 83.5% coverage and 99.04% supported accepted accuracy at 1.82 ms p95, but accepts 5.9% of unfamiliar requests; banking reaches 81.7% coverage at 4.39 ms p95, but 98.77% accepted accuracy. Neither qualifies; the live-teacher disconnection recording remains pending.
- Add opt-in `choice_solver="logistic"` to the Python compiler for choice skills. SciPy is an optional teaching dependency; saved heads retain the existing NumPy runtime, calibration and portable format. Keep ridge as the default and record failed full-scope development experiments for the n8n quality gauntlet.
- Honor `record_receipt=False` on fresh decisions as well as cache hits, serialize absent receipts as null, and skip unused cache digests. Compute conformal tail counts by binary search while preserving inclusive ties and prediction sets.

- Add an experimental teach-by-doing document workspace: filing actions become explicit examples, corrections invalidate stale skills, separate checks calibrate review, and lessons/actions/skills export for reuse. Include complete authored sample evidence: 29/30 filing guesses correct, 27/27 accepted ordinary suggestions correct, and one of six unusual requests incorrectly accepted. No new core API or dependency.

- Make the takeover lifecycle test fail on any post-promotion teacher call, replacing a brittle single-call wall-clock limit. Retain the separate warm-latency checks.

- Add real Pokémon Red/Blue potion control, paired save-state replays in the teaching GUI, and 144 measured controller runs across 18 explicitly edited battle fixtures. Preserve the modest 6/18 → 7/18 gain, strict review results, resource costs and regression; keep the move skill and prior simulator evidence unchanged.

- Add a two-skill Pokémon teaching lab with live healing correction, equal-budget original/corrected single-model baselines, strict review runs and retained regressions. Include a bounded real Red/Blue ROM controller with observed RAM facts and no demo defaults; add Pillow to the optional gameboy extra for its live screen.
- Improve the courier objective lesson and measure strict completion on a fresh cohort: 22/40 → 31/40, and 19/40 → 28/40 with corrected terrain. Preserve the original experiment and its evidence.

- Bound an existing concurrent reader/writer regression workload so a writer timeout cannot leave CI waiting forever for unbounded readers. Use synchronized startup, 1,024 reads and all 30 checked updates.

- Add an experimental courier teaching playground: three saved skills, live isolated terrain corrections, held-out mission composition, matched retained-label baselines and complete recorded evidence. Preserve the initial danger-penalty failure and gate movement using learned terrain eligibility. Keep review limitations explicit; no new core API or runtime dependency.

- Add a three-player Snake GUI comparing a taught System1 skill, real Laya-MLX and Jev. Include timed and equal-move modes, explicit planner/shield accounting, replay without API access, and four complete measured traces.
- Add a reproducible public SpamAssassin teaching recipe requiring no mailbox or API credentials. Freeze source hashes and disjoint group splits, compare a conventional baseline offline, and retain both the failed collection-shift result (89.9% accepted correctness) and the subsequent representative split (603/618 accepted answers correct, 0.216 ms median). Keep this binary task distinct from the seven-category Inbox Zero pilot.

- Add optional NumPy-only `TfidfProjector` for taught text skills, with a frozen vocabulary carried in `.s1m`, bounded loader validation, and review for inputs with no known features. Preserve the default hashed projector.
- Broaden email teaching and retain both the original failed result and the new TF-IDF result: 83/84 correct on a fresh authored set, one accepted mistake, and about 10× faster direct decisions. Keep review-only default because upstream acceptance and independent mailbox validation remain incomplete.

- Add an opt-in Inbox Zero category-classification pilot: editable lessons, saved local skill, authenticated HTTP service, and a separately licensed upstream provider patch. Preserve review on uncertainty, contract changes, and service failures.
- Publish all 35 upstream compatibility results and a conventional baseline. The initial System1 lesson accepts 5/35 cases and remains review-only by default; no claim of production takeover.

## 1.0.3 — 2026-09-20

- Require an independent trusted public key for receipt authentication. Expose `check_decision_receipt_integrity()` for unsigned diagnostics; comparison results no longer claim unsigned receipts are authenticated.
- Bind the committed ledger record ID in the signed envelope while preserving the core receipt digest. Check exact ledger inclusion before guard acceptance, and recheck the full history before every ledger write.
- Restrict gRPC schema selection to built-ins and the startup registry, with canonical built-in aliases. Remote names cannot trigger imports, local file loading, or unlimited cache growth.
- Bound `.s1m` file/header sizes, ZIP entries and expanded bytes, NPY shapes and types, calibration counts, projector dimensions, and runtime allocations before loading. Preserve ordinary v1/v2 saved-skill roundtrips.
- Fix numeric teaching/calibration split handling and add a supervised Paperclips skill with separate evaluation.
- Add attack regressions and document the receipt and gRPC migration in [the release notes](docs/releases/1.0.3.md).

## 1.0.2 — 2026-09-19

- Add a beginner walkthrough and editable teaching file explaining labels, corrections and readiness. Let the minimal example read custom lessons, try a supplied message and save a separate candidate. Organize the README and documentation around that journey; keep advanced policy/integration examples in linked guides.

- Add opt-in `PromotionPolicy(min_accepted_agreement=.95)`: require point agreement and an exact lower bound among accepted validation groups, sharing the repeated-attempt confidence budget. Preserve the original default policy.
- Expose the compiler's existing regularization parameter through sync/async observation clients; retain its default of 1.0.
- Publish a fuller original-label SMS observation run: 2,961 observations, 954/980 correct accepted old-test decisions (97.3%). Preserve stricter banking/assistant deferrals and the new authored SMS failure.
- Retain 106 new diagnostic probes and a rejected routing teaching candidate. Keep primary lesson files unchanged; do not claim all quality gaps are solved.
- Make the product distinction explicit: System 1 is not an LLM; it teaches bounded local decision heads without language-model weights or token generation.


- Publish three fixed public-data workload evaluations and actual Jev/Gemini takeover records, including unsuccessful promotions and a classical baseline. All teacher recordings replay offline; no new runtime dependency.
- Synchronize the README, guides, architecture and papers with current behavior and evidence. Separate historical reports from current instructions, correct example output claims, and retire unsupported promotional graphics.
- Clarify experimental container templates and repair their binding/probe instructions. Add lightweight documentation link/version checks to CI.
- Make the concurrent-cutover regression independent of teaching-fold arrival order while retaining 50 simultaneous callers and ledger checks.

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
