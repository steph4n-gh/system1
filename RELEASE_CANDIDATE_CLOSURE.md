# Reflex / System 1 — Release Candidate Closure Report

**Review Basis**: Bundle `95ab0b2caec62f6e170b0d6f0e726e741354cd8e80e426746c0cf1393ddc411c`  
**Evaluation Date**: September 18, 2026  
**Status**: Release-Candidate Gates Passed for Local On-Metal Decision Firewall Runtime  

This report certifies the resolution of all remaining correctness, authorization, cache fidelity, promotion, and distribution gaps identified in the Round 2 Master Closure Report. Reflex / System 1 is certified as an on-metal, zero-egress decision firewall operating under strict mathematical conformal bounds and Ed25519 cryptographic attestation.

---

## 1. Close the Authorization-to-Execution Evidence Path

### Root Causes & Implementations

1. **Common Authorization-Finalization Path (`guard.py`)**:
   - *Defect*: Model-backed inference receipts were recorded in the ledger, but subsequent policy-bound final receipts created with different digests copied the earlier `ledger_record_id` without persisting the final authorization.
   - *Fix*: Created unified `_finalize_authorization()` method called by all authorization branches (deterministic rule-only, model-backed, and hybrid). It constructs the final authorization record, computes the exact signed digest, writes it to `ActionLedger`, and returns the true persistent record ID.
2. **Fail-Closed Permission Authority (`guard.py`)**:
   - *Defect*: With `enforcement_profile=True`, requests lacking an applicable permission grant fell through to model classification, allowing unpermitted tools (e.g., `sentinel`) to execute.
   - *Fix*: Enforced strict fail-closed reference monitor semantics in `evaluate_proposal()`: if `enforcement_profile` is active and no matching deterministic rule grants `ALLOW`, the request fails closed immediately with `DENY` (`reflex_missing_grant`).
3. **Continuous Ledger Cryptographic Verification (`ledger.py`)**:
   - *Defect*: Earlier ledger payload corruption was detectable by `verify_integrity()` but did not halt subsequent append or outcome operations.
   - *Fix*: Integrated `_verify_integrity_locked()` inside `append()`, `record_decision_receipt()`, and `record_execution_outcome()` within SQLite `BEGIN IMMEDIATE` transactions. Corrupted payloads or broken hash chains prevent any further execution or outcome recording with `IntegrityError`.
4. **Execution Outcomes & Asynchronous Lifecycle (`ledger.py`, `integrations/mcp.py`, `integrations/langchain.py`)**:
   - *Defect*: `record_execution_outcome()` accepted orphan actions and mismatched digests. Coroutine objects from `async def` tools wrapped by synchronous decorators were recorded as `SUCCEEDED` prematurely before execution.
   - *Fix*:
     - `record_execution_outcome()` validates that prior authorization exists for `action_id`, verifies digest matching, and accepts `"INDETERMINATE"` alongside standard terminal statuses.
     - MCP decorator (`wrap_mcp_tool`) inspects return values via `inspect.isawaitable()`; asynchronous tools return an awaiting wrapper that only records `SUCCEEDED` after the coroutine successfully resolves, or records `FAILED` on exception.
     - Integration metadata in MCP and LangChain now binds the actual `receipt.compute_digest()` rather than falling back to `schema_digest`.

---

## 2. Close Exact-Cache and Calibration-Snapshot Semantics

### Root Causes & Implementations

1. **Lossless IEEE-754 Context Identity (`cache.py`)**:
   - *Defect*: Decimal formatting (`:.6f` / `:.8f`) caused alpha collisions (e.g. `0.0594061` vs `0.0594058`) crossing finite-sample conformal quantile boundaries.
   - *Fix*: Refactored `_format_context()` in `cache.py` to use lossless IEEE-754 representation via `float.hex()` for all continuous floating-point parameters (`alpha`, `margin_threshold`, `relative_odds_ratio`, `confidence_floor_tau0`). Distinct float representations never collide.
2. **Supplied Embedding Cache Binding (`cache.py`, `engine.py`)**:
   - *Defect*: Opposite supplied embeddings for identical prompts returned identical cached responses because the initial fast-path did not bind embedding identities.
   - *Fix*: `_make_key()` incorporates `e:{e_dig}` computed from SHA-256 of float32 embedding bytes. `engine.decide()` binds `query_emb` during lookup when an explicit embedding is supplied by the caller, while preserving standard string lookup when no embedding is passed.
3. **Structured Escalation & Regression Uncertainty (`engine.py`, `guard.py`)**:
   - *Defect*: Multi-label and regression uncertainty checks evaluated interval strings (e.g., `['[0.10, 0.90]']`, len=1) as categorical singletons, allowing uncalibrated regression decisions to bypass guard gating.
   - *Fix*: `ReflexGuardHook` explicitly inspects `decision.is_ambiguous` and `decision.escalated_fields`. Any structured escalation or uncalibrated field immediately triggers `REQUIRE_APPROVAL` with `reflex_structured_escalation`.
4. **Online Learning Calibration Invalidation (`engine.py`)**:
   - *Defect*: `learn_from_tier2()` updated model weights while leaving stale calibration scores marked valid.
   - *Fix*: Weight updates increment `model_version`, evict cached entries, and invalidate stale calibration status in strict mode until valid recalibration occurs.

---

## 3. Promote and Export One Complete Validated Runtime

### Root Causes & Implementations

1. **Lineage-Preserving Fold Partitioning (`compat/typesafe.py`)**:
   - *Defect*: `partition_cutover_history()` split records across calibration and validation folds when fewer than 3 distinct groups existed. Overlap checking preferred `request_id` over `group_id`.
   - *Fix*:
     - `CutoverPartition.assert_disjoint()` extracts and verifies disjointness across all lineage identifiers (`group_id`, `lineage_id`, `request_id`, state prompt) across all folds.
     - `partition_cutover_history()` requires at least 3 distinct groups; if `len(groups) < 3`, it raises `ValueError("Insufficient evidence: distinct groups cannot supply the 3 required folds")` rather than manufacturing artificial independence.
2. **Request-Level Statistical Sample Units (`compat/typesafe.py`)**:
   - *Defect*: Requests with multiple correlated fields counted each field as an independent trial, artificially inflating Wilson score sample sizes.
   - *Fix*: `evaluate_promotion_eligibility()` uses `effective_n = len(val_history)` (the number of independent validation requests/groups) as the sample size $n$ for Wilson lower-bound calculation.
3. **Per-Schema Promotion Tracking (`compat/typesafe.py`)**:
   - *Defect*: A global `_has_cutover` flag allowed promotion of one schema to mistakenly certify other unrelated schemas.
   - *Fix*: Promotion state and compiled engines are tracked per schema digest (`_has_cutover_schemas: Set[str]`, `_history: Dict[str, List]`, `_engine_cache: Dict[str, ReflexEngine]`).

---

## 4. Finish Distribution and Align Dependency Bounds

1. **Dependency Alignment (`pyproject.toml`)**:
   - Updated protobuf and gRPC requirements to reflect generated code: `protobuf>=6.31.1`, `grpcio>=1.80.0`, and `grpcio-tools>=1.80.0`.
   - Reconciled all packaging assertions in `tests/test_round2_gates_e_f.py` and `tests/test_adversarial_gate_f.py`.
2. **Repository Hygiene**:
   - Removed all temporary exploratory and test scripts from repository root.
   - Verified that `dist/` builds cleanly without warnings.

---

## 5. Automated Verification Telemetry

### Full Test Suite Execution
```bash
python3 -m pytest tests/ -q
```
- **Total Tests**: 954
- **Passed**: 952
- **Skipped**: 2 (platform-specific conditionals)
- **Failures**: 0
- **Execution Time**: 43.84s

### File Digest Manifest (SHA-256)
```
9dbb9c30eb77b66e70d138674f11936e8fecfa0d36762228fd5aed6cbf5652a1  pyproject.toml
d55dc5559051d33df9c25e36809f43f87cf156ae85fcee93edcdb7860296fb49  src/system1/cache.py
8c295165fc72ec4093562b3fe52a67870cd0852f64804b7c2a2f019e1fefc060  src/system1/calibration.py
61ad8ef73326eb53f9524593cbbd61e56b6e2e615888247143fe7f92d2a2b878  src/system1/compat/typesafe.py
76313345176eeefc0260bd0cc0115ea71324f3fcbd2214c2735361f74282e89f  src/system1/engine.py
73ab9b793026ff4483fac54fb195609c6672395f0459c3a09cdfe0fbea96ee9a  src/system1/guard.py
5b8ba8cf1ad51cd3afbe44c9e49ea8fe5719da00118984395a3397aa80b1beae  src/system1/integrations/langchain.py
33be2b696018a1b3b2548c53619fe392dc4fee9f84bcae288d05b0d7c10f81c0  src/system1/integrations/mcp.py
b0c2a57a78bc3a929074f936adaa378ca86783fc5c8e497d558e94db5c0df3c6  src/system1/ledger.py
944fd72f1f89a706906fe137f631ed09d6c4eecbc31b35a3d6ac7e191be5120e  dist/system1-0.1.1-py3-none-any.whl
e7bdca92492ad525a15b3e270e4151cc15e737bd1e62611fbd6bd72e7b451b86  dist/system1-0.1.1.tar.gz
```

---

## 6. Installed Distribution Artifact Verification

A temporary virtual environment outside the source tree (`/tmp/reflex_rc_isolated_test`) was populated with `system1-0.1.1-py3-none-any.whl` and verified:

1. **Explicit Permission Grant Path**: A policy-granted action (`ALLOW`) executed, was assigned an Ed25519-signed final authorization receipt, was durably recorded in `ActionLedger`, and had its outcome successfully chained (`SUCCEEDED`).
2. **Missing Grant / Unmatched Tool**: An unpermitted tool (`sentinel_exec`) with active `ENFORCEMENT_PROFILE_V1` was rejected with `DENY` (`reflex_missing_grant`).
3. **Corrupted Ledger Tamper-Detection**: Direct payload tampering in the underlying SQLite database was detected by `_verify_integrity_locked()`, and subsequent actions were blocked fail-closed before execution.
4. **Exact Lossless Cache Invariants**:
   - Queries with identical prompt strings but opposite supplied embeddings returned distinct respective cache entries (`north` vs `south`).
   - Distinct alpha values (`0.0594061` vs `0.0594058`) did not collide under `float.hex()` representation.
5. **Group Lineage Integrity**: 10-record histories with only 2 distinct lineage groups were rejected with `ValueError("Insufficient evidence: distinct groups cannot supply the 3 required folds")`.
6. **Zero-Egress Enforcement**: Both `TypeSafeClient(mode="passthrough", zero_egress=True)` at construction and direct transport calls raised `ZeroEgressViolationError`.

---

## 7. Supported Feature Matrix

| Capability | Scope / Guarantee | Validation Status |
| :--- | :--- | :--- |
| **Deterministic Reference Monitor** | Strict precedence (`DENY` > `REQUIRE_APPROVAL` > `ALLOW`), argument limits fail-closed | **VERIFIED** |
| **Ed25519 Decision Receipts** | Final policy-decision binding, SHA-256 request digests, canonical probabilities | **VERIFIED** |
| **Tamper-Evident Action Ledger** | SQLite with cryptographically chained audit hashes and two-phase outcome linking | **VERIFIED** |
| **Tier 0 Exact Reflex Cache** | Lossless IEEE-754 float parameter binding, embedding identity, atomic synchronization | **VERIFIED** |
| **Split Conformal Safety Gate** | Non-conformity calibration sets, structured escalation consumption, OOD detection | **VERIFIED** |
| **Validated Local Cutover** | Lineage-preserving group partitioning, Wilson lower bounds, per-schema promotion | **VERIFIED** |
| **Zero-Egress Transport Boundary** | Incompatible construction fail-close, socket creation blocking across sync/async | **VERIFIED** |
| **Dual Namespace Symmetry** | Identical 107 exported public symbols across `reflex` and `system1` | **VERIFIED** |

---

## 8. Conclusion

**Release-candidate gates passed for the explicit scope of offline authorization hardening, deterministic cache fidelity, runtime validation, and local artifact compilation.**
