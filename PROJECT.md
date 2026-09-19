# Project: System 1 / System 1 Runtime Hardening

## Architecture
System 1 / System 1 is a dual-process cognitive architecture providing ultra-low-latency, non-autoregressive decision making on local metal (System 1 fast reflex) coupled with conformal ambiguity gating and online distillation from strategic deliberative planners (System 2 deliberate governor).

Key Hardening Architecture Pillars:
1. **Authorization & Tool Execution Gating (R1)**:
   - JSON-RPC protocol gating in `SystemOneMCPProxy`: strictly route `tools/call` to tool dispatch; handle protocol requests (`ping`, `initialize`) with protocol responses and 0 tool invocations.
   - Comprehensive argument binding in LangChain and MCP tool wrappers: inspect parameter signatures, bind defaults, unpack positional `_args`, evaluate nested arguments without character truncation.
   - Deterministic policy enforcement in `SystemOneGuardHook`: evaluate principal, tenant, scope, target, action, and limits with absolute veto before statistical inference and caching.
   - Harmless sentinel execution: zero side effects on any rejection, error, or unauthenticated request.
2. **Cryptographic Attestation & Audit Ledger Integrity (R2)**:
   - Authenticated receipt verification: strictly reject missing/None signatures, profile downgrades, malformed keys, and unauthenticated key substitution in `verify_decision_witness_receipt`.
   - Comprehensive claim binding in signed receipts: bind canonical action, target, principal, tenant, scope, outcome, digests, gating parameters, nonces, and reported `probabilities`.
   - Fail-closed audit ledger profile: raise `LedgerWriteError` upon SQLite write failures; verify durable recording before granting tool execution permissions.
   - Cryptographic chaining of execution outcomes: link post-execution outcomes back to authorization receipts.
3. **Mathematically Coherent Conformal Calibration & Set Gating (R3)**:
   - Exact conformal prediction set construction: eliminate erroneous `min_top_prob = 1 - q_hat` cutoff on APS cumulative scores; resolve empty set on `[0.4, 0.35, 0.25]`.
   - Conservative finite-sample order statistics: when $k = \lceil (n+1)(1-\alpha) \rceil > n$, return conservative full feasible domain $[min\_value, max\_value]$.
   - Cross-module score and serialization harmonization: compute APS scores consistently across `calibration.py`, `compiler.py`, and `engine.py`; persist scores in `.s1m` NPZ buffers.
   - Strict prediction-set cardinality gating: escalate to System 2 whenever $|C(x)| \ne 1$; disallow margin heuristics in strict mode.
4. **Cache Lifecycle & Semantic Invalidation (R4)**:
   - Online learning invalidation: evict prompt and invalidate prior model caches upon Sherman-Morrison rank-1 updates (`learn_from_tier2`), model recompilation, and policy mutations.
   - Complete cache key binding: bind prompt, schema, model version, policy scope, alpha, margin, and strict mode.
   - Defensive copying: deepcopy returned decision dictionaries to prevent internal cache corruption.
5. **Rigorous Cutover Promotion & Generalization Validation (R5)**:
   - Split Conformal exchangeability: partition history into distinct held-out training, calibration, and validation folds.
   - Statistical promotion criteria: enforce minimum sample thresholds, statistical acceptance confidence intervals, and zero false-allow tolerance on critical security actions.
   - Post-promotion drift detection and offline abstention in zero-egress environments.
6. **Benchmark Provenance, Packaging & Polyglot gRPC Interop (R6)**:
   - Benchmark provenance: explicitly label simulated vs measured provider latencies in triple-crown benchmarks.
   - Production gRPC & Protobuf: fix `reflex.proto` syntax, compile standard Python gRPC stubs, support binary protobuf frames.
   - Twin-namespace packaging: export complete mirror modules (`observability`, `otel`), include `.proto` in wheel distributions.
   - Documentation alignment: reconcile software Ed25519 vs hardware enclaves, clarify mathematical theorems.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | MCP JSON-RPC Gating | Restrict `handle_call` to `tools/call`; return protocol responses for `ping`/`initialize` with 0 tool executions (Invariant 1) | M1 | Survey (E1) |
| F2 | Tool Argument Binding & Truncation Removal | Eliminate `[:256]` slice, inspect signatures for defaults, bind positional `_args` and nested dictionaries (Invariant 2) | M1 | Survey (E1) |
| F3 | Deterministic Policy Engine | Add deterministic reference monitor (principal, tenant, scope, target, action, limits) overriding inference/cache | M1 | Survey (E1) |
| F4 | Harmless Sentinel Execution | Ensure sentinel tools record 0 invocations across all denials, malformed calls, and unrecorded requests | M1 | Survey (E1) |
| F5 | Authenticated Receipt Verification | Enforce non-None signatures for signed profiles, reject key substitution, validate trusted keys (Invariant 3) | M2 | Survey (E1) |
| F6 | Probabilities Digest & Signature Binding | Bind `probabilities` to unsigned payload, digest calculation, envelope, and verification checks (Invariant 4) | M2 | Survey (E1) |
| F7 | Fail-Closed Audit Ledger | Prevent tool execution on ledger write failure/read-only database; eliminate silent exception swallowing (Invariant 5) | M2 | Survey (E1) |
| F8 | Execution Outcome Chaining | Link post-execution results to authorization receipts with durable status recording | M2 | Survey (E1) |
| F9 | Conformal Empty-Set Resolution | Remove erroneous `min_top_prob` lower bound on APS quantile; guarantee valid prediction sets on in-distribution inputs (Invariant 6) | M3 | Survey (E2) |
| F10 | Conservative Small-Sample Bounds | Return full feasible domain $[min\_val, max\_val]$ when order statistic $k > n$ (Invariant 7) | M3 | Survey (E2) |
| F11 | Cross-Module Score Harmonization | Unify APS score calculation across `calibration.py`, `compiler.py`, and `engine.py`; store and restore `.s1m` scores | M3 | Survey (E2) |
| F12 | Strict Prediction-Set Cardinality Gating | Require escalation when $|C(x)| \ne 1$; prevent margin bypass in strict mode | M3 | Survey (E2) |
| F13 | Cache Semantic Invalidation | Invalidate cache on online weight updates (`learn_from_tier2`), bump model version, eliminate stale certified decisions (Invariant 8) | M4 | Survey (E2) |
| F14 | Comprehensive Cache Key Context | Bind schema, model version, policy scope, alpha, margin, and strict mode to cache keys (Invariant 9) | M4 | Survey (E2) |
| F15 | Defensive Decision Copying | Return defensive deep copies of decision dictionaries to prevent internal cache mutation | M4 | Survey (E2) |
| F16 | Held-Out Generalization Validation | Decouple history into training, calibration, and held-out validation partitions for cutover promotion (Invariant 10) | M5 | Survey (E3) |
| F17 | Statistical Promotion Criteria | Enforce per-schema sample sizes, false-allow ceilings, and confidence intervals before cutover | M5 | Survey (E3) |
| F18 | Post-Promotion Drift & Abstention | Provide drift detection and offline abstention in zero-egress mode | M5 | Survey (E3) |
| F19 | Benchmark Provenance Labeling | Explicitly label simulated/randomized vs live measured provider runs in triple-crown benchmarks | M6 | Survey (E3) |
| F20 | Genuine Compiled gRPC & Protobuf | Fix `reflex.proto`, compile genuine protobuf/gRPC stubs, support binary protobuf wire serialization | M6 | Survey (E3) |
| F21 | Twin-Namespace Packaging Symmetry | Add missing `observability` and `otel` mirrors in `src/reflex/integrations/`, package `.proto` in wheel | M6 | Survey (E3) |
| F22 | Documentation & Whitepaper Harmony | Clarify software Ed25519 vs hardware enclaves, qualify mathematical theorems, and verify late-2026 terminology | M6 | Survey (E3) |
| F23 | All 10 Regression Invariant Tests | Implement automated tests covering all 10 invariants in `tests/` | M7 | ORIGINAL_REQUEST |
| F24 | Zero Regression Baseline (506+ Tests) | Verify 100% pass rate across entire existing test suite | M7 | ORIGINAL_REQUEST |
| F25 | Comprehensive Hardening Report | Author `RELEASE_HARDENING_REPORT.md` documenting root causes, patches, before/after evidence, and challenger review | M7 | ORIGINAL_REQUEST |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Authorization | M1 | Authorization & Tool Execution Gating | F1, F2, F3, F4 (Invariants 1 & 2) | None | PLANNED | Tool Execution Gating | F1, F2, F3, F4 (Invariants 1 | M1 | Authorization & Tool Execution Gating | F1, F2, F3, F4 (Invariants 1 & 2) | None | PLANNED | 2) | None | DONE |
| M2 | Cryptographic Attestation | M2 | Cryptographic Attestation & Fail-Closed Ledger | F5, F6, F7, F8 (Invariants 3, 4, 5) | None | PLANNED | Fail-Closed Ledger | F5, F6, F7, F8 (Invariants 3, 4, 5) | None | DONE |
| M3 | Conformal Calibration | M3 | Conformal Calibration & Set Gating | F9, F10, F11, F12 (Invariants 6 & 7) | None | PLANNED | Set Gating | F9, F10, F11, F12 (Invariants 6 | M3 | Conformal Calibration & Set Gating | F9, F10, F11, F12 (Invariants 6 & 7) | None | PLANNED | 7) | None | IN_PROGRESS |
| M4 | Cache Lifecycle | M4 | Cache Lifecycle & Semantic Invalidation | F13, F14, F15 (Invariants 8 & 9) | M3 | PLANNED | Semantic Invalidation | F13, F14, F15 (Invariants 8 | M4 | Cache Lifecycle & Semantic Invalidation | F13, F14, F15 (Invariants 8 & 9) | M3 | PLANNED | 9) | M3 | IN_PROGRESS |
| M5 | Cutover Promotion | M5 | Cutover Promotion & Generalization Validation | F16, F17, F18 (Invariant 10) | M3 | PLANNED | Generalization Validation | F16, F17, F18 (Invariant 10) | M3 | IN_PROGRESS |
| M6 | Benchmark Provenance, Packaging | M6 | Benchmark Provenance, Packaging & Polyglot gRPC | F19, F20, F21, F22 | None | PLANNED | Polyglot gRPC | F19, F20, F21, F22 | None | IN_PROGRESS |
| M7 | Regression Verification, Adversarial Hardening & Report | F23, F24, F25 | M1, M2, M4, M5, M6 | PLANNED |

---

## Interface Contracts
### Authorization & Proxy Contract
- `SystemOneMCPProxy.handle_call(request, executor, ...)`: Only invokes `executor` when JSON-RPC method is `tools/call`. Other methods return protocol responses with 0 executor calls.
- `SystemOneGuardHook.evaluate_proposal(proposal)`: First evaluates deterministic policy engine (`PolicyEngine`). If denied, returns `DENY` with absolute priority.
- Tool argument extraction: Positional arguments, default parameters, and nested arguments are fully preserved in `proposal.arguments` without string length limits.

### Cryptographic Attestation & Ledger Contract
- `verify_decision_witness_receipt(receipt_dict, public_key=...)`: Fails (returns `False` or raises) if signature is missing/None for authenticated profiles.
- `compute_receipt_digest(receipt_dict)` and `unsigned_payload()`: Strictly include canonical `probabilities`. Any alteration in `receipt["probabilities"]` invalidates digest and signature.
- `SystemOneEngine.decide(..., ledger=...)`: Raises `LedgerWriteError` in fail-closed mode if ledger write fails; never silently passes.
- `SystemOneGuardHook.evaluate_proposal`: Checks `receipt.ledger_record_id is not None` before returning `ALLOW`.

### Conformal Calibration & Prediction Set Contract
- `ConformalPredictor.predict_set(probs)`: Computes APS cumulative sums up to $\hat{q}$; does NOT apply `min_top_prob = 1 - q_hat`. Guarantees non-empty prediction sets on in-distribution inputs.
- `RegressionConformalPredictor.predict_interval(val)`: When $k = \lceil (n+1)(1-\alpha) \rceil > n$, returns $[min\_value, max\_value]$.
- `.s1m` Format: Stores `{name}_calib_scores` in NPZ payload; restored upon reload.
- Escalation: Any prediction set with cardinality $\ne 1$ sets `is_ambiguous = True` and triggers System 2 escalation.

### Cache Lifecycle Contract
- `CacheKey`: Binds `(prompt, schema_digest, model_version, policy_scope, alpha, margin_threshold, strict, telemetry)`.
- `SystemOneEngine.learn_from_tier2`: Bumps `model_version`, evicts pre-existing cache entries for prompt and related representations.
- Returned `DecisionResult.values`: Deep copy of internal values dictionary.

### Cutover Promotion Contract
- `TypeSafeClient._distill_and_cutover_locked`: Partitions history into 60% training, 20% calibration, 20% held-out validation. Evaluates promotion agreement solely on held-out validation.
- False-allow tolerance: Exactly 0.0% false-allow rate on critical tool actions.

---

## Code Layout
```
src/
├── reflex/                  # Public twin package (re-exports and mirrors)
│   ├── __init__.py
│   ├── cache.py
│   ├── calibration.py
│   ├── compiler.py
│   ├── engine.py
│   ├── guard.py
│   ├── grpc_server.py
│   ├── ledger.py
│   ├── receipt.py
│   ├── compat/
│   ├── core/
│   └── integrations/
│       ├── langchain.py
│       ├── mcp.py
│       ├── observability.py # Mirror to be added
│       └── otel.py          # Mirror to be added
├── system1/                 # Canonical implementation
│   ├── cache.py
│   ├── calibration.py
│   ├── compiler.py
│   ├── engine.py
│   ├── guard.py
│   ├── grpc_server.py
│   ├── ledger.py
│   ├── receipt.py
│   ├── compat/
│   │   └── typesafe.py
│   ├── proto/
│   │   ├── reflex.proto
│   │   ├── system1_pb2.py
│   │   └── system1_pb2_grpc.py
│   └── integrations/
│       ├── langchain.py
│       └── mcp.py
benchmarks/
│   └── run_triple_crown_benchmark.py
examples/                    # 17 runnable demo and benchmark scripts
tests/                       # 506 existing tests + new regression tests
RELEASE_HARDENING_REPORT.md  # Comprehensive deliverable report
```
