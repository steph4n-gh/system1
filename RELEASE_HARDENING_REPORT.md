# Reflex / System 1 Runtime Hardening — Release Verification Report

**Document Version**: 1.0.0-final  
**Release Target**: Reflex / System 1 Hardening (v0.1.0)  
**Date**: 2026-09-18T16:15:00Z  
**Author**: Integrator Worker (`teamwork_preview_worker_integrator`)  
**Repository**: `/Volumes/Storage/reflex`  
**Overall Release Status**: **APPROVED & CERTIFIED FOR PRODUCTION**  
**Independent Verification Status**:
- **Reviewer 1 (R1 & R2)**: **APPROVE**
- **Reviewer 2 (R3–R6)**: **APPROVE**
- **Challenger 1 Round 2 (Invariants 1–5 Adversarial)**: **CONFIRM**
- **Challenger 2 (Invariants 6–10 Adversarial)**: **CONFIRM**
- **Forensic Auditor 1 (Codebase Integrity & Anti-Cheating)**: **CLEAN** (0 integrity violations, 0 facades, 0 mocks in production)
- **Regression Invariant Test Suite**: **684 passed, 2 skipped** (100% pass rate, 0 regressions from 506-test baseline)
- **Distribution Packaging**: Wheel and sdist built cleanly with twin-namespace protobuf/gRPC assets

---

## 1. Executive Summary

Reflex / System 1 is an ultra-low-latency dual-process cognitive runtime designed to bridge fast non-autoregressive decision execution on local metal (System 1 fast reflex) with strategic deliberative governance (System 2 deliberate planner). Operating strictly within a sub-2ms local latency budget and guaranteeing zero external network egress, Reflex provides drop-in compatibility across twin namespaces (`reflex` and `system1`) alongside seamless TypeSafe AI client integration.

Prior to this hardening release, comprehensive architectural auditing revealed ten critical regression failure invariants spanning authorization gating, cryptographic attestation, conformal calibration bounds, cache invalidation, cutover generalization, and polyglot wire serialization. Under certain adversarial conditions, these flaws could permit unauthenticated tool execution, blind policy reference monitors through argument truncation, accept tampered receipts, serve stale pre-update decisions from Tier 0 cache, or promote overfitted models based on memorized training exemplars.

This release successfully resolves all 10 regression failure invariants and implements complete hardening across all 6 requirement areas (R1 through R6). Through multi-agent implementation, rigorous peer review, two rounds of adversarial challenge stress testing, and an independent forensic audit, the codebase has been proven to maintain authentic, mathematically sound logic with zero facades, mocks, or shortcuts. Automated test coverage expanded from 506 baseline passed tests to 684 passed tests (178 newly authored invariant and adversarial tests), maintaining a 100% pass rate with zero regressions.

---

## 2. Comprehensive Breakdown of Requirement Areas (R1–R6)

### R1. Authorization & Tool Execution Gating (P0)
- **JSON-RPC Protocol Gating (`ReflexMCPProxy`)**:
  `ReflexMCPProxy.handle_call()` enforces strict protocol segregation. Only JSON-RPC messages with `method == "tools/call"` can enter the tool dispatch pipeline. Non-tool methods (`ping`, `initialize`, `tools/list`) are intercepted prior to any execution logic and serviced with protocol-compliant JSON-RPC response structures, guaranteeing exactly 0 invocations of the underlying tool executor.
- **Untruncated Argument Binding & Parameter Inspection**:
  The legacy 256-character string truncation (`[:256]`) in LangChain integrations was eliminated. The runtime inspects target functions via `inspect.signature()` in both `wrap_mcp_tool` and `ReflexToolInterceptor`, binding positional `*args` and default values into formal parameter names. Unbound positional arguments are preserved under `_args`. Payloads exceeding 4096 characters are safely hashed (`sha256:{digest}`) for bounded target naming while preserving the complete, untruncated payload in `proposal.arguments["input"]` and the policy context prompt.
- **Deterministic Pre-Inference Policy Engine Reference Monitor**:
  `PolicyEngine` and `PolicyRule` were introduced in `src/system1/guard.py` to enforce deterministic security controls (scoping by tenant, principal, action pattern, target regex, denied target lists, and argument limits). `ReflexGuardHook.evaluate_proposal()` evaluates deterministic policy rules *before* model inference and cache retrieval. Any policy denial or approval requirement exerts immediate, absolute veto authority that cannot be bypassed by high model confidence, margin dominance heuristics, or teacher feedback.
- **Harmless Sentinel Execution**:
  Sentinel tools verify 0 invocations across all test suites for all `DENY`, `REQUIRE_APPROVAL`, malformed requests, or ledger errors.

### R2. Cryptographic Attestation & Fail-Closed Ledger Integrity (P0)
- **Authenticated Receipt Verification & Anti-Substitution**:
  `verify_decision_witness_receipt()` strictly forbids unauthenticated verification. In `product_signed_v1` or whenever a public key is supplied, any missing (`signature=None`) or blank signature is rejected immediately. Declared public keys in `receipt["signer_public_key"]`, `envelope.signer_public_key`, and `envelope.effect_observation["signer_public_key"]` must strictly match the verified public key bytes, eliminating unauthenticated key substitution attacks.
- **Canonical Probabilities Binding & Tamper Detection**:
  Canonical normalized probabilities (`_canonical_probabilities`) are bound into `unsigned_payload()`, the SHA-256 receipt digest, and `RunWitnessEnvelope.guard_receipt`. Canonical precision is normalized to 12 decimal places (`round(float(p), 12)`). Cross-field validation checks canonical probabilities between receipt and envelope. Any perturbation down to $10^{-12}$ breaks the digest and envelope verification, forcing `verify_decision_witness_receipt()` to return `False`.
- **Fail-Closed Audit Ledger (`LedgerWriteError`)**:
  In fail-closed mode (`fail_closed_ledger=True`), write failures caused by read-only databases, disk exhaustion, or SQLite lock contention immediately raise `LedgerWriteError`. `ReflexGuardHook` verifies that `decision.receipt.ledger_record_id is not None`. If unrecorded, it forces `DecisionOutcome.DENY`, eliminating silent exception swallowing and preventing tool execution without durable audit trails.
- **Two-Phase Execution Outcome Chaining**:
  `ActionLedger.record_execution_outcome()` logs post-execution statuses (`SUCCEEDED`, `FAILED`, `INDETERMINATE`) cryptographically chained via SHA-256 to the preceding authorization receipt digest. If tool execution completes but outcome recording fails, MCP proxy returns error code `-32001` with status `INDETERMINATE`.

### R3. Mathematically Coherent Conformal Calibration & Set Gating (P0)
- **Empty-Set Resolution on Identical Vectors**:
  Resolved the counterexample where 100 identical probability vectors `[0.4, 0.35, 0.25]` labeled `'A'` returned an empty prediction set at $\alpha=0.05$. The erroneous heuristic cutoff `min_top_prob = max(1e-6, 1.0 - q_hat - 1e-7)` was completely removed from `ConformalPredictor.predict_set()`. Cumulative mass accumulation proceeds greedily from the top candidate, guaranteeing that any valid distribution on the probability simplex contains at least candidate `'A'`. Empty sets are strictly restricted to true out-of-distribution (OOD) inputs ($\sum p_i < q_{\hat{}} - 10^{-7}$ and $\sum p_i < 0.5$).
- **Conservative Small-Sample Order Statistics**:
  In `RegressionConformalPredictor.predict_interval()`, when sample size $n$ is insufficient to resolve the order statistic ($k = \lceil (n+1)(1-\alpha) \rceil > n$), the predictor returns the full feasible domain $[min\_value, max\_value]$ with `margin = val_range`. The arbitrary `margin = val_range / 2.0` heuristic was eliminated, restoring formal finite-sample $1-\alpha$ coverage.
- **Score Harmonization & `.s1m` Serialization**:
  Adaptive Prediction Set (APS) non-conformity scores are computed consistently across `calibration.py`, `compiler.py`, and `engine.py`. Calibration score arrays are persisted under `{name}_calib_scores` in `.s1m` NPZ buffers and faithfully restored during engine initialization, preserving empirical distribution-free quantiles across serialization round-trips.
- **Strict Cardinality Gating**:
  In strict mode (`strict=True`), margin dominance heuristics are disabled. Any prediction set with cardinality $|C(x)| \ne 1$ (empty or multi-label) sets `is_ambiguous = True` and forces escalation to System 2.

### R4. Cache Lifecycle & Online Learning Semantic Invalidation (P0)
- **Online Learning Cache Invalidation**:
  Upon online weight updates via recursive least-squares (Sherman-Morrison rank-1 update in `ReflexEngine.learn_from_tier2()`), `self.model_version` increments, `cache.evict_prompt(prompt)` purges matching prompt keys, and `cache.invalidate_prior_versions(self.model_version)` invalidates all cached entries from previous versions. Repeat queries immediately reflect the newly learned target without requiring manual cache flushes.
- **Comprehensive Cache Key Context Binding**:
  Cache keys strictly encode full parameter context:
  `s:{schema_digest}|v:{model_version}|sc:{policy_scope}|a:{alpha:.6f}|m:{margin_threshold:.6f}|st:{strict}`.
  Altering risk parameters ($\alpha$, margin threshold, strict mode) or security scopes produces immediate cache misses, preventing cross-parameter cache bleeding.
- **Deep Defensive Copying**:
  `SemanticReflexCache` and `ReflexEngine` enforce `copy.deepcopy()` across all cached and returned decision dictionaries (`values`, `confidences`, `conformal_sets`, `probabilities`, `margins`). Callers cannot mutate internal cache memory.

### R5. Rigorous Cutover Promotion & Generalization Validation (P1)
- **Decoupled 3-Way Disjoint Partitioning**:
  Eliminated in-sample cutover evaluation on training history. `partition_cutover_history()` partitions teacher history into 60% training, 20% calibration, and 20% held-out validation. `partition.assert_disjoint()` enforces strict disjointness with zero object identity overlap. Promotion eligibility is evaluated exclusively on held-out validation exemplars.
- **Statistical Promotion Criteria & Zero-Tolerance Safety Ceilings**:
  `evaluate_promotion_eligibility()` enforces minimum sample counts, empirical agreement thresholds, and finite-sample Wilson score confidence lower bounds. A strict zero-tolerance false-allow ceiling (`false_allow_ceiling = 0.0`) is enforced across 9 critical security classes (`{"deny", "unsafe", "fraud", "block", "high_risk", "malicious", "critical", "require_approval", False}`). A single false allow on a critical action immediately aborts promotion.
- **Drift Detection & Zero-Egress Offline Abstention**:
  `DriftDetector` tracks rolling ambiguity and OOD rates across a sliding window. In privacy-restricted `zero_egress=True` mode, drifted queries trigger offline abstention (`REQUIRE_APPROVAL`, `abstain=True`, 0 egress bytes) rather than falling back to external cloud governors.

### R6. Benchmark Provenance, Packaging & Polyglot gRPC Interop (P1)
- **Benchmark Provenance Labeling**:
  In `benchmarks/run_triple_crown_benchmark.py`, simulated cloud WAN latencies and token estimates are explicitly labeled as `SYNTHETIC_SIMULATED` across scorecards, tables, and JSON metadata. Genuine local execution is labeled `MEASURED_LIVE`.
- **Genuine Compiled Protobuf & gRPC Wire Stubs**:
  Cleaned `src/system1/proto/reflex.proto` and compiled official stubs `reflex_pb2.py` and `reflex_pb2_grpc.py`. `ReflexServiceServicer` serializes genuine binary Protobuf messages over the gRPC wire, verified with in-process stub roundtrips.
- **Twin-Namespace Packaging Symmetry**:
  Mirror modules `observability.py` and `otel.py` were added to `src/reflex/integrations/`. Package data in `pyproject.toml` packages all `.proto` files and compiled stubs in both `system1` and `reflex` namespaces. Clean wheel (`.whl`) and source distributions (`.tar.gz`) build and install cleanly.
- **Documentation & Whitepaper Alignment**:
  `README.md`, `docs/architecture/technical_specification.md`, and `docs/paper/reflex_whitepaper.md` were reconciled, clearly distinguishing software Ed25519 receipts from hardware enclaves, specifying local vs fallback egress paths, and providing exact mathematical proofs.

---

## 3. In-Depth Root Causes, Line-Level Patches, and Before/After Evidence for All 10 Invariants

### Invariant 1: MCP Proxy Dispatch of Non-Tool Calls
- **Root Cause**:
  `ReflexMCPProxy.handle_call()` did not verify that incoming JSON-RPC requests had `method == "tools/call"`. Requests with `method="ping"` or `method="initialize"` that supplied tool arguments fell through to tool evaluation or failed unpredictably. Additionally, non-dict arguments (e.g. `"rm -rf /"` or `12345`) caused unhandled `AttributeError: 'str' object has no attribute 'get'`.
- **Line-Level Patch Locations**:
  - `src/system1/integrations/mcp.py:177-186`: In `intercept_jsonrpc()`, validate `isinstance(arguments, dict)`. Return code `-32602` if invalid.
  - `src/system1/integrations/mcp.py:217-235`: In `handle_call()`, intercept `method != "tools/call"`, returning protocol responses for `ping`, `initialize`, and `tools/list`.
  - `src/system1/integrations/mcp.py:257-266`: In `handle_call()`, validate `isinstance(arguments, dict)` before parameter extraction.
- **Before Evidence**:
  - `handle_call({"method": "ping", "params": {"name": "tool"}})` dispatched to the tool executor or threw an internal error.
  - `handle_call({"method": "tools/call", "params": {"name": "tool", "arguments": "rm -rf /"}})` crashed with unhandled `AttributeError`.
- **After Evidence**:
  - `handle_call` returns `{"jsonrpc": "2.0", "id": req_id, "result": {}}` on `ping` with exactly 0 executor invocations.
  - Non-dict arguments return `{"error": {"code": -32602, "message": "Invalid params: arguments must be an object"}}` with 0 executor invocations.
  - Verified in `tests/test_authorization_invariants.py:test_invariant_1_*` and `tests/test_adversarial_invariants_challenger_1.py:test_invariant_1_*`.

### Invariant 2: Authorization Argument Truncation & Parameter Binding
- **Root Cause**:
  - `src/system1/integrations/langchain.py` sliced input strings with `str(input_str[:256])`, blinding security classifiers to dangerous commands appended beyond index 256.
  - Positional arguments passed to tool interceptors were not bound to named schema parameters.
  - When raw input strings exceeding 4096 characters were passed to `canonical_target`, `ActionProposal.create()` raised an unhandled `ValueError` via `_required_target()`.
- **Line-Level Patch Locations**:
  - `src/system1/integrations/langchain.py:88-101`: Removed `[:256]` slice; implemented SHA-256 target hashing for strings $>4096$ chars; preserved full payload in `proposal.arguments["input"]` and `context_prompt`.
  - `src/system1/integrations/langchain.py:134-147, 155-168, 246-258`: Added `inspect.signature` parameter binding and target hashing to `ReflexToolInterceptor`.
  - `src/system1/integrations/mcp.py:109-113, 343-356`: Added signature binding to `wrap_mcp_tool` and target hashing to `evaluate_mcp_call`.
- **Before Evidence**:
  - Input `"safe_prefix " + "malicious_command" * 50` was truncated at 256 characters, dropping the malicious command from policy inspection.
  - Payloads $>4096$ characters crashed tool execution with `ValueError: canonical_target must be non-empty, bounded, and contain no NUL`.
- **After Evidence**:
  - Full payloads (tested up to 100,000 characters) are preserved in `proposal.arguments["input"]` and context prompts without character truncation.
  - Bounded canonical target hashing (`sha256:{hex}`) prevents `ValueError` while maintaining deterministic target identity.
  - Positional arguments are mapped to named parameters with default values instantiated.
  - Verified in `tests/test_authorization_invariants.py:test_invariant_2_*` and `tests/test_adversarial_invariants_challenger_1.py:test_invariant_2_*`.

### Invariant 3: Receipt Verification Passes When Envelope Signature is `None`
- **Root Cause**:
  In `src/system1/receipt.py:verify_decision_witness_receipt()`, missing signatures (`signature=None` or `signature=""`) were not rejected when `profile == "product_signed_v1"` or when `public_key` was provided. Attackers could strip signatures or substitute unauthenticated public keys.
- **Line-Level Patch Locations**:
  - `src/system1/receipt.py:566-574`: Reject missing/empty signatures if `public_key is not None`, `envelope.profile == "product_signed_v1"`, or `envelope.profile != "diagnostic_local"`.
  - `src/system1/receipt.py:626-643`: Check `receipt["signer_public_key"]` and `envelope.effect_observation["signer_public_key"]` against `public_key_bytes(key_obj)`.
- **Before Evidence**:
  - A receipt signed with Ed25519 was modified with `receipt["envelope"]["signature"] = None`; `verify_decision_witness_receipt(receipt, public_key=key.public_key())` returned `True`.
  - Declaring an attacker's public key in the receipt payload was accepted without verification against the signing key.
- **After Evidence**:
  - `verify_decision_witness_receipt` strictly returns `False` when `signature is None`, empty, malformed, or verified against an untrusted/substituted public key.
  - Verified in `tests/test_attestation_invariants.py:test_invariant_3_*`.

### Invariant 4: Receipt Verification Passes When Reported Probabilities Are Altered
- **Root Cause**:
  - `probabilities` were omitted from `unsigned_payload()` and digest calculation.
  - Initial normalization used 6 decimal places (`round(float(p), 6)`), causing perturbations $\le 5\times 10^{-7}$ (e.g. $+10^{-7}$) to round away and pass verification undetected.
- **Line-Level Patch Locations**:
  - `src/system1/receipt.py:114-126`: In `_canonical_probabilities()`, upgraded normalization precision to 12 decimal places: `round(float(p), 12)`.
  - `src/system1/receipt.py:376-391, 442, 501`: Included canonical probabilities in `unsigned_payload()`, `compute_receipt_digest()`, and `envelope.guard_receipt`.
  - `src/system1/receipt.py:670-672, 681-684`: Cross-field verification checks canonical probabilities between receipt and envelope.
- **Before Evidence**:
  - Altering `receipt["probabilities"]["action"]["allow"] += 0.05` left `compute_receipt_digest()` unchanged, and verification succeeded.
  - Altering `receipt["probabilities"]["action"]["allow"] += 1e-7` was truncated by 6-decimal rounding, returning `True`.
- **After Evidence**:
  - Any perturbation down to $10^{-12}$ (tested across $\pm 10^{-7}, \pm 10^{-8}, \pm 10^{-9}, \pm 10^{-10}, \pm 10^{-11}, \pm 10^{-12}$) alters canonical JSON bytes, invalidates the digest, and causes `verify_decision_witness_receipt()` to return `False`.
  - Verified in `tests/test_attestation_invariants.py:test_invariant_4_*` and `tests/test_adversarial_invariants_challenger_1.py:test_invariant_4_*`.

### Invariant 5: Read-Only / Corrupt Ledger Executes Tool Without Durable Recording
- **Root Cause**:
  In `ReflexEngine.decide()` and `ReflexGuardHook.evaluate_proposal()`, SQLite ledger write errors (e.g., read-only database, disk full) were swallowed in `try/except` blocks, allowing actions to execute without audit logging.
- **Line-Level Patch Locations**:
  - `src/system1/ledger.py:30`: Defined `LedgerWriteError`.
  - `src/system1/engine.py:487-488, 560-561, 773-774, 816-817`: Added `fail_closed_ledger: bool = False`; raised `LedgerWriteError` on ledger failure.
  - `src/system1/guard.py:762-777`: Caught `LedgerWriteError`, `LedgerError`, and `sqlite3.Error`, and checked `decision.receipt.ledger_record_id is not None`. If unrecorded, forced `DecisionOutcome.DENY`.
  - `src/system1/ledger.py:251-323`: Added `record_execution_outcome()` to chain post-execution results to authorization receipts.
- **Before Evidence**:
  - When SQLite database was marked read-only (`chmod 444`), `decide()` logged an exception to stderr and returned an approval. The tool executor ran with zero ledger records.
- **After Evidence**:
  - When ledger write fails, `engine.decide()` raises `LedgerWriteError` in fail-closed mode.
  - `ReflexGuardHook` catches write errors, detects missing `ledger_record_id`, and immediately returns `DENY`. Sentinel executor call count remains 0.
  - Post-execution outcomes (`SUCCEEDED`, `FAILED`, `INDETERMINATE`) are cryptographically linked to the authorization receipt digest.
  - Verified in `tests/test_attestation_invariants.py:test_invariant_5_*`.

### Invariant 6: ConformalPredictor Empty Prediction Set on Identical Vectors
- **Root Cause**:
  In `src/system1/calibration.py:465-471`, `predict_set()` computed:
  `min_top_prob = max(1e-6, 1.0 - q_hat - 1e-7)` and skipped accumulation if `top_prob < min_top_prob`.
  When calibrated on 100 identical vectors `[0.4, 0.35, 0.25]` labeled `'A'`, the empirical quantile was $q_{\hat{}} = 0.40$, resulting in `min_top_prob = 0.60`. Since `top_prob = 0.40 < 0.60`, the loop was bypassed and an empty set `()` was returned, violating the $1-\alpha$ coverage guarantee on in-distribution inputs.
- **Line-Level Patch Locations**:
  - `src/system1/calibration.py:487-505`: Removed `min_top_prob`. Accumulated mass greedily starting from the top prediction candidate. If `total_mass >= q_hat - 1e-7` and `total_mass >= 0.5`, candidate `'A'` is included.
  - `src/system1/compiler.py:389-390, 447-448`: Serialized sorted calibration scores under `{name}_calib_scores` in `.s1m` NPZ buffers.
  - `src/system1/engine.py:275-280`: Restored calibration scores from `.s1m` into `conformal_predictors`.
- **Before Evidence**:
  - Evaluating identical vector `[0.4, 0.35, 0.25]` at $\alpha=0.05$ returned `prediction_set = ()` and `is_empty = True`.
- **After Evidence**:
  - Prediction set contains candidate `'A'` (`prediction_set = ('A',)`), guaranteeing empirical coverage $\ge 1 - \alpha$.
  - Deserializing compiled `.s1m` preserves exact non-conformity scores and identical prediction sets.
  - Verified in `tests/test_conformal_invariants.py:test_invariant_6_*` and `tests/test_adversarial_invariants_6_to_10.py:test_invariant_6_*`.

### Invariant 7: Order Statistic Exceeding Calibration Sample Count Returns Arbitrary Interval
- **Root Cause**:
  In `src/system1/calibration.py:615-623`, `RegressionConformalPredictor.predict_interval()` returned `margin = val_range / 2.0` when order statistic $k = \lceil (n+1)(1-\alpha) \rceil > n$. On domain $[0.0, 100.0]$ with $\hat{y} = 20.0$, this returned $[0.0, 70.0]$, arbitrarily truncating $[70.0, 100.0]$ without theoretical justification.
- **Line-Level Patch Locations**:
  - `src/system1/calibration.py:626-635`: When $k > n$, set `margin = val_range`, `low = self.min_value`, `high = self.max_value`.
- **Before Evidence**:
  - Sample size $n=5$ at $\alpha=0.05$ ($k=6 > 5$) on $[0.0, 100.0]$ with prediction $20.0$ returned interval $[0.0, 70.0]$.
- **After Evidence**:
  - When $k > n$, predictor returns the complete feasible domain $[0.0, 100.0]$ with `margin = 100.0`, preserving valid distribution-free coverage.
  - Verified in `tests/test_conformal_invariants.py:test_invariant_7_*` and `tests/test_adversarial_invariants_6_to_10.py:test_invariant_7_*`.

### Invariant 8: Cache Returns Stale Pre-Update Decision After Online Learning
- **Root Cause**:
  In `src/system1/engine.py:learn_from_tier2()`, Sherman-Morrison rank-1 updates modified head covariance and weight matrices, but `self.model_version` was not incremented, the prompt was not evicted, and prior cache entries were not purged. Subsequent queries retrieved stale pre-update decisions from the Tier 0 cache.
- **Line-Level Patch Locations**:
  - `src/system1/engine.py:911-925`: Increment `self.model_version += 1`, call `self.cache.evict_prompt(prompt)`, and call `self.cache.invalidate_prior_versions(self.model_version)`.
  - `src/system1/cache.py:270-295`: Implemented `evict_prompt()` and `invalidate_prior_versions()`.
- **Before Evidence**:
  - Performing 10 online learning updates (`learn_from_tier2`) with teacher target `'ALLOW'` on a previously denied prompt still returned `'DENY'` on repeat queries due to stale Tier 0 cache hits.
- **After Evidence**:
  - Immediate repeat queries return post-update decision `'ALLOW'` without manual cache flush.
  - Verified across 10-step and 15-step alternating cycles in `tests/test_cache_invariants.py:test_invariant_8_*` and `tests/test_adversarial_invariants_6_to_10.py:test_invariant_8_*`.

### Invariant 9: Cache Bypasses Validation When Alpha/Margin Parameters Change
- **Root Cause**:
  In `src/system1/cache.py`, cache keys were generated solely from prompt and telemetry digests. Risk parameters (`alpha`, `margin_threshold`, `strict`, `policy_scope`, `schema_digest`, `model_version`) were omitted from cache keys. Queries under strict risk bounds were satisfied by cached decisions evaluated under lenient bounds.
- **Line-Level Patch Locations**:
  - `src/system1/cache.py:91-104, 155-165`: Bound execution context into cache keys:
    `s:{schema_digest}|v:{model_version}|sc:{policy_scope}|a:{alpha:.6f}|m:{margin_threshold:.6f}|st:{strict}`.
  - `src/system1/cache.py:303, 349`: Enforced `copy.deepcopy()` on cache storage and retrieval.
  - `src/system1/engine.py:568-586, 751-762`: Deep defensive copying of all returned decision dictionaries.
- **Before Evidence**:
  - Decision cached under $\alpha=0.50$ returned a cache hit for request with $\alpha=0.01$.
  - Mutating returned `decision.values["action"] = "EXPLOIT"` corrupted subsequent cache hits.
- **After Evidence**:
  - Changing $\alpha$, margin threshold, strict mode, or policy scope causes a cache miss and forces re-evaluation.
  - External mutation of decision dictionaries does not affect internal cache state.
  - Verified in `tests/test_cache_invariants.py:test_invariant_9_*` and `tests/test_adversarial_invariants_6_to_10.py:test_invariant_9_*`.

### Invariant 10: Cutover Promotes Based on Training History Agreement Rather Than Generalization
- **Root Cause**:
  In `src/system1/compat/typesafe.py:_distill_and_cutover_locked()`, apprentice-to-metal cutover agreement was evaluated in-sample on `self._history` (the training exemplars). Overfitted models that memorized noise achieved 100% agreement and cut over to production, despite failing completely on unseen distributions. Additionally, no false-allow ceilings existed for critical security actions.
- **Line-Level Patch Locations**:
  - `src/system1/compat/typesafe.py:116-202`: Implemented `partition_cutover_history()` (60% train, 20% calib, 20% held-out val) and enforced `partition.assert_disjoint()`.
  - `src/system1/compat/typesafe.py:206-369`: Implemented `PromotionPolicy`, `evaluate_promotion_eligibility()`, and `compute_wilson_score_lower()`. Enforced `false_allow_ceiling = 0.0` on critical classes (`{"deny", "unsafe", "fraud", "block", "high_risk", "malicious", "critical", "require_approval", False}`).
  - `src/system1/compat/typesafe.py:375-430, 2185-2240`: Added `DriftDetector` and offline abstention in zero-egress mode.
- **Before Evidence**:
  - Synthesized training data memorizing random tokens to `'allow'` achieved 100% in-sample agreement and promoted the model, despite 0% agreement on validation data.
  - A model predicting `'allow'` on a critical `'deny'` target was promoted because overall accuracy exceeded 80%.
- **After Evidence**:
  - Overfitted models are strictly rejected on held-out validation (`agreement_rate = 0.0 < 0.80`).
  - Any single false allow on a critical action immediately aborts promotion.
  - Verified in `tests/test_cutover_invariants.py:test_invariant_10_*` and `tests/test_adversarial_invariants_6_to_10.py:test_invariant_10_*`.

---

## 4. Summary of Independent Verification Outcomes

| Verification Agent | Role | Scope | Verdict | Key Findings / Highlights |
|---|---|---|---|---|
| **Reviewer 1** (`teamwork_preview_reviewer_1`) | Reviewer, Critic | R1 (Authorization) & R2 (Attestation & Ledger) | **APPROVE** | Verified Invariants 1–5; confirmed pre-inference reference monitor veto authority; confirmed inspect.signature binding without character truncation; confirmed non-None signature enforcement and anti-substitution; confirmed SQLite WAL fail-closed ledger error handling and two-phase outcome chaining. Zero regressions across 559 tests. |
| **Reviewer 2** (`teamwork_preview_reviewer_2`) | Reviewer, Critic | R3 (Calibration), R4 (Cache), R5 (Cutover), R6 (Packaging & gRPC) | **APPROVE** | Verified Invariants 6–10; confirmed cumulative mass accumulation on probability simplex; confirmed small-sample conservative bounds; confirmed score serialization in `.s1m`; confirmed cache versioning and deep copying; verified 3-way disjoint partition and zero-tolerance safety ceilings; confirmed genuine binary gRPC Protobuf stubs and wheel packaging data; verified `SYNTHETIC_SIMULATED` benchmark provenance. |
| **Challenger 1 Round 2** (`teamwork_preview_challenger_1_r2`) | Critic, Specialist | Adversarial verification of Invariants 1–5 | **CONFIRM** | Tested 18/18 epsilon probability perturbations down to $10^{-12}$ (all strictly rejected); tested large payloads up to 100,000 characters without `ValueError`; tested 10 malformed non-dict MCP argument types returning `-32602` with 0 executor invocations. Confirmed all Round 1 defects remediated. All 41 adversarial tests in `test_adversarial_invariants_challenger_1.py` passed. |
| **Challenger 2** (`teamwork_preview_challenger_2`) | Critic, Specialist | Adversarial verification of Invariants 6–10 | **CONFIRM** | Audited probability simplex boundaries, degenerate uniform distributions, and Dirichlet Monte Carlo trials; verified small-sample regression bounds across 5 continuous domains; verified 15 online update cycles with 0 stale reads; confirmed parameter bleeding isolation and cache immutability; confirmed memorization rejection and zero-tolerance false-allow ceilings across 9 critical classes. All 84 adversarial tests in `test_adversarial_invariants_6_to_10.py` passed. |
| **Auditor 1** (`teamwork_preview_auditor_1`) | Forensic Auditor | Full Codebase Integrity & Anti-Cheating | **CLEAN** | Zero integrity violations, zero facades, zero dummy stub returns, zero production mocks (`unittest.mock` count = 0 in `src/system1` and `src/reflex`), zero rigged assertions (`assert True` count = 0 in `tests/`), zero pre-populated verification logs. All implementations maintain genuine state and algorithmic behavior. 100% test pass rate across entire suite. |

---

## 5. Test Suite Statistics & Evolution

```
========================================================================================
Test Suite Evolution Summary:
----------------------------------------------------------------------------------------
Baseline (Initial Repository):           506 passed, 2 skipped
Milestone M1 & M2 (R1 & R2 Invariants):  +23 tests  (529 passed, 2 skipped)
Milestone M3 & M4 (R3 & R4 Invariants):  +14 tests  (543 passed, 2 skipped)
Milestone M5 & M6 (R5 & R6 Invariants):  +16 tests  (559 passed, 2 skipped)
Challenger 1 Adversarial & Remediation:  +41 tests  (600 passed, 2 skipped)
Challenger 2 Adversarial (Invariants 6-10): +84 tests (684 passed, 2 skipped)
----------------------------------------------------------------------------------------
FINAL TEST SUITE TOTAL:                  684 passed, 2 skipped in 30.58s
NET TEST INCREASE:                       +178 newly authored invariant & stress tests
PASS RATE:                               100.0% (0 failures, 0 errors, 0 regressions)
========================================================================================
```

### Breakdown of Invariant & Adversarial Test Suites:
1. `tests/test_authorization_invariants.py`: 12 tests (Invariant 1, Invariant 2, policy veto, sentinel execution)
2. `tests/test_attestation_invariants.py`: 11 tests (Invariant 3, Invariant 4, Invariant 5, outcome chaining)
3. `tests/test_conformal_invariants.py`: 8 tests (Invariant 6, Invariant 7, `.s1m` serialization, strict mode)
4. `tests/test_cache_invariants.py`: 6 tests (Invariant 8, Invariant 9, defensive copying)
5. `tests/test_cutover_invariants.py`: 8 tests (Invariant 10, Wilson bounds, safety ceilings, drift abstention)
6. `tests/test_grpc_polyglot.py`: 8 tests (gRPC stubs, binary Protobuf wire roundtrips, packaging parity)
7. `tests/test_adversarial_invariants_challenger_1.py`: 41 tests (Epsilon tampering down to $10^{-12}$, large payloads to 100k chars, MCP malformed argument types)
8. `tests/test_adversarial_invariants_6_to_10.py`: 84 tests (Simplex boundaries, small-sample bounds, online cache invalidation, parameter bleeding, held-out cutover gating)

*Note on Skipped Tests (2)*:
- `tests/test_grpc_server.py::test_grpc_requires_grpcio`: Skipped when `grpcio` is installed (verifies fallback behavior when `grpcio` is omitted).
- `tests/test_pokemon_kaizo_speedrun.py::test_pokemon_kaizo_speedrun_headless_loop`: Skipped when optional PyBoy GUI emulator bindings are not installed.

---

## 6. Exact Verified Execution Commands

To independently reproduce, inspect, and certify all findings of this release:

### 6.1 Execute Full Regression Test Suite (684 Tests)
```bash
python3 -m pytest tests/ -v
```
*Expected Output*: `684 passed, 2 skipped in ~30s`.

### 6.2 Execute Invariant Test Suites
```bash
# R1 & R2 Authorization and Attestation
python3 -m pytest tests/test_authorization_invariants.py tests/test_attestation_invariants.py -v

# R3 & R4 Conformal Calibration and Cache Lifecycle
python3 -m pytest tests/test_conformal_invariants.py tests/test_cache_invariants.py -v

# R5 & R6 Cutover Generalization and Polyglot gRPC
python3 -m pytest tests/test_cutover_invariants.py tests/test_grpc_polyglot.py -v
```

### 6.3 Execute Adversarial Challenger Suites
```bash
# Challenger 1: Adversarial Attestation & Authorization Suite (41 tests)
python3 -m pytest tests/test_adversarial_invariants_challenger_1.py -v

# Challenger 2: Adversarial Calibration, Cache, and Cutover Suite (84 tests)
python3 -m pytest tests/test_adversarial_invariants_6_to_10.py -v
```

### 6.4 Verify Invariant 4 Epsilon Probability Tampering Down to $10^{-12}$
```bash
python3 -c "
import copy
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from reflex import create_decision_receipt, verify_decision_witness_receipt

key = Ed25519PrivateKey.generate()
pub = key.public_key()
receipt = create_decision_receipt(
    schema_name='SecuritySchema', schema_digest='a'*64, prompt='test auth',
    values={'is_safe': True}, confidences={'is_safe': 0.95},
    conformal_sets={'is_safe': ['True']},
    probabilities={'is_safe': {'True': 0.95, 'False': 0.05}},
    latency_ms=1.0, is_ambiguous=False, signing_key=key
)
base = receipt.to_dict()
for eps in [1e-7, -1e-7, 1e-9, -1e-9, 1e-12, -1e-12]:
    t = copy.deepcopy(base)
    t['probabilities']['is_safe']['True'] += eps
    assert verify_decision_witness_receipt(t, public_key=pub) is False
print('Invariant 4 Verified: All epsilon perturbations strictly rejected!')
"
```

### 6.5 Verify Twin-Namespace Symmetry & Parity
```bash
python3 -c "
import reflex, system1
assert reflex.__version__ == system1.__version__
assert reflex.LedgerWriteError is system1.LedgerWriteError
assert reflex.PolicyRule is system1.PolicyRule
assert reflex.PolicyEngine is system1.PolicyEngine
assert reflex.ConformalPredictor is system1.ConformalPredictor
assert reflex.ReflexMCPProxy is system1.ReflexMCPProxy
assert reflex.ReflexGuardCallbackHandler is system1.ReflexGuardCallbackHandler
assert reflex.verify_decision_witness_receipt == system1.verify_decision_witness_receipt
print('Twin-namespace symmetry verified successfully!')
"
```

### 6.6 Verify Distribution Packaging & Wheel Build
```bash
python3 -m build --wheel
python3 -c "
import zipfile
z = zipfile.ZipFile('dist/system1-0.1.0-py3-none-any.whl')
names = z.namelist()
assert any('reflex/proto/reflex.proto' in n for n in names)
assert any('system1/proto/reflex_pb2.py' in n for n in names)
assert any('reflex/integrations/observability.py' in n for n in names)
assert any('system1/compat/typesafe.py' in n for n in names)
print('Wheel packaging verified: all proto and mirror modules included!')
"
```

---

## 7. Conclusion & Release Sign-Off

The Reflex / System 1 Runtime Hardening release (v0.1.0) satisfies all functional, architectural, cryptographic, and mathematical requirements set forth in `ORIGINAL_REQUEST.md` and `PROJECT.md`. All 10 regression failure invariants are closed with line-level evidence and verified by independent peer reviewers and adversarial challengers. The runtime is certified for production deployment.
