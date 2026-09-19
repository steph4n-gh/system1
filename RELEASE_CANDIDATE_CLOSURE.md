# Reflex / System 1 — Release Candidate Closure Report

**Review Basis**: Bundle `a14ec0ff8692fe22c84cbc7b02aa88e49495475ba38749e7f9f212d01c98ddce`  
**Evaluation Date**: September 18, 2026  
**Status**: Release-Candidate Gates Passed for Local On-Metal Decision Firewall Runtime  

This report certifies the resolution of all remaining correctness, authorization, cache fidelity, promotion, and distribution gaps identified across the independent review rounds. Reflex / System 1 is certified as an on-metal, zero-egress decision firewall operating under strict mathematical conformal bounds and Ed25519 cryptographic attestation.

---

## 1. Section A: Calibration Validity and Cache Input Identity

### A1. Invalidate Calibration on Model Changes (`engine.py`)
- **Defect**: `learn_from_tier2()` updated model weights and incremented `model_version`, but conformal predictors remained marked `is_calibrated = True`. In strict evaluation (`strict=True`), inference accepted singletons computed against old calibration distributions.
- **Resolution**:
  - `learn_from_tier2()` now explicitly resets `is_calibrated = False` on standard conformal predictors and regression conformal predictors for all updated fields.
  - Strict inference (`strict=True` or engine configured with `strict_mode=True`) detects uncalibrated state, halts fail-closed, marks decisions as ambiguous (`is_ambiguous = True`), and abstains until valid recalibration occurs.
  - Reader/update barrier tests confirm that active snapshot coherence is maintained across forward passes.
- **Verification**: `tests/test_previous_contracts_adapted.py::test_learning_marks_previous_calibration_stale` PASSED.

### A2. Prevent Explicit Embeddings from Contaminating Implicit Queries (`cache.py`, `engine.py`)
- **Defect**: `_base_to_key` in `cache.py` created generic string lookup aliases. When a query was executed with an explicit caller-supplied embedding, the alias mapped the prompt text to the explicit embedding's cache key. A subsequent query with the same text but NO embedding returned the explicit embedding's result instead of computing fresh semantic features.
- **Resolution**:
  - Added `explicit_embedding: bool` parameter to `ReflexCache.put()`.
  - When `explicit_embedding=True`, `put()` stores the exact key under its embedding-bound digest without updating the `_base_to_key` alias table.
  - Eviction logic in `_remove_key()` deterministically purges any lingering aliases.
  - Normal text queries without caller-supplied embeddings only reuse entries derived from the standard semantic encoder.
- **Verification**: `tests/test_previous_contracts_adapted.py::test_explicit_embedding_does_not_poison_implicit_embedding` PASSED.

---

## 2. Section B: Validate, Promote, Export, and Reload Runtime Artifacts

### B1. Persist Validated Runtime Calibration (`compat/typesafe.py`)
- **Defect**: `TypeSafeClient._distill_and_cutover_locked()` calibrated a live candidate `ReflexEngine`, but `export_model()` serialized the original `CompiledSystemOneModel` containing pre-calibration state, causing a mismatch on artifact reload (e.g. 10 live scores vs 15 reloaded scores).
- **Resolution**:
  - `export_model()` inspects the live candidate `ReflexEngine` associated with the compiled model's schema digest under thread lock.
  - Exact validated temperature scaling values and conformal score/residual arrays are copied into `cm.heads` prior to binary serialization.
  - Artifact reloads preserve exact prediction intervals, conformal set sizes, and escalation thresholds.
- **Verification**: `tests/test_round4_variants.py::test_exported_candidate_preserves_validated_runtime_calibration` PASSED.

### B2. Scored Independent Validation Units (`compat/typesafe.py`)
- **Defect**: `effective_n = len(val_history)` counted records that provided no scored evidence (unlabeled rows) and treated correlated rows from the same group/request as independent trials, artificially inflating the Wilson lower bound (e.g. 1 labeled + 99 unlabeled rows inflated bound from 0.2065 to 0.9630).
- **Resolution**:
  - `evaluate_promotion_eligibility()` groups validation history by stable lineage identifiers (`group_id`, `lineage_id`, `request_id`, or prompt hash).
  - Only groups containing at least one valid labeled field answer are counted as scored units (`unit_has_scored_evidence = True`).
  - `effective_n` is set strictly to the count of independent scored groups (`effective_n = scored_units`). Unlabeled padding rows are ignored, and multiple records within a single group collapse into one unit.
- **Verification**: `tests/test_round4_variants.py::test_statistical_promotion_rejection` PASSED (both unlabeled padding and single-group correlation tests reject promotion).

---

## 3. Section C: Ledger Verification and Execution Outcome Binding

### C1. Trust Anchor Validation on Append (`ledger.py`)
- **Defect**: `ActionLedger.append()` verified existing history integrity, but did not check the incoming record itself against `trusted_public_key`. An empty ledger accepted receipts signed by arbitrary untrusted keys.
- **Resolution**:
  - `append()` validates the incoming receipt signature against `trusted_public_key` using `verify_decision_witness_receipt(payload, public_key=trusted_public_key)` before writing to SQLite.
  - Failure raises `IntegrityError` and leaves the database completely unchanged.
- **Verification**: `tests/test_round4_variants.py::test_ledger_verifies_incoming_receipt_against_anchor` PASSED.

### C2. Outcome Identity & Scope Matching (`ledger.py`, `integrations/`)
- **Defect**: `record_execution_outcome()` verified `action_id` and `receipt_digest` but allowed substitution of `tenant_id`, `principal_id`, and `scope` without validation.
- **Resolution**:
  - `record_execution_outcome()` retrieves the prior authorization entry and verifies that `tenant_id`, `principal_id`, and non-default `scope` strictly match the prior authorization record.
  - Mismatched identity or scope raises `LedgerWriteError("Authorization mismatch")`.
  - Integration adapters (`mcp.py`, `langchain.py`) pass aligned scope parameters matching their initial authorization proposals.
- **Verification**: `tests/test_round4_variants.py::test_record_execution_outcome_enforces_cryptographic_authorization` PASSED.

---

## 4. Section D: Concurrency Counter, Measured Latency, and Quickstart

### D1. Public Counter Contract (`compat/typesafe.py`)
- **Defect**: `TypeSafeClient._call_count` was only incremented during teacher query collection, failing to count post-cutover or local evaluations and causing concurrency test counter mismatches (26 vs 50).
- **Resolution**:
  - Incremented `_call_count` on all successfully evaluated queries across pre-cutover, post-cutover, and local modes under thread synchronization.
  - Added distinct `total_requests` property alongside `teacher_sample_count` to honestly separate evaluated user queries from teacher exemplar collections.
- **Verification**: `tests/test_auto_cutover.py::test_concurrent_auto_cutover` PASSED (50/50 responses, `client.call_count == 50`).

### D2. Measured Authorization Latency (`grpc_server.py`)
- **Defect**: Hardcoded `latency_ms=0.1` was returned in gRPC Guard responses when wrapping policy decisions.
- **Resolution**: Implemented high-resolution monotonic timing via `time.perf_counter()` to record genuine authorization execution latency on every Guard RPC invocation.
- **Verification**: `tests/test_adversarial_gate_f.py` and polyglot gRPC benchmarks PASSED.

### D3. Enforcing Quickstart Documentation (`README.md`)
- **Resolution**: Replaced the classification-only example with an authenticated, durable enforcing quickstart configuring `PolicyEngine`, `ReflexGuard`, `Ed25519PrivateKey`, and persistent WAL `ActionLedger`, explicitly separating deterministic rule policies from statistical conformal classification.

---

## 5. Automated Verification Telemetry

### Full Test Suite Execution
```bash
python3 -m pytest tests/ --junitxml=runs/junit.xml
```
- **Total Tests Collected**: 960
- **Passed**: 958
- **Skipped**: 2 (optional hardware ROM conditionals)
- **Failures**: 0
- **Errors**: 0
- **Execution Time**: 46.06s
- **Machine-Readable Reports**: `runs/junit.xml`, `runs/test_summary.json`

### File Digest Manifest (SHA-256)
```
9dbb9c30eb77b66e70d138674f11936e8fecfa0d36762228fd5aed6cbf5652a1  pyproject.toml
78a20e9cdff3d1e9e15fb0d63e3d92df9b156a93469440bd35346fc22f51ca56  README.md
1e5b3a7a2edc0a06946c8bed19215c4cc9a4d294da7e4fbccd101af1b077f2a3  src/system1/cache.py
8c295165fc72ec4093562b3fe52a67870cd0852f64804b7c2a2f019e1fefc060  src/system1/calibration.py
b98a5b021c3d9576c8a9798fdb2c19213969a47c9a95d3471e7ed5cea5a28348  src/system1/compat/typesafe.py
ef5a28d07a1dfe5f5c39781075ba33c70d5e47a05e66799326fd5e77d7d60e70  src/system1/engine.py
6a3a668c7338811c692bb475094a5266a09365fc860a39027cf008313d019d8b  src/system1/grpc_server.py
a06986b4002b5b57e3607e7aee0c18d9d90dde6826dd8a8fb1aaecc60642366d  src/system1/guard.py
e8f74942438ba1c6884976c681fdf99f1e691ecaa1da2c4acffff169fa33883b  src/system1/integrations/langchain.py
136d0408d282ffbd3fbfe4ec486ba230256d3cdf98ac1fa2c44aaaa1a7e9b5df  src/system1/integrations/mcp.py
64ad9641c87871a872ced21ab734ab4f6ed22829ef07b888827a4657a4a80143  src/system1/ledger.py
ee4fab297b9775dafb44f14a292d455a70494fb9f80ff70a9667d7ab8b13bd13  tests/test_previous_contracts_adapted.py
5d7595a76a9b2eea8c2fd927131a50a7f9e882d0db32becdd78620d7872c7eb7  tests/test_round4_variants.py
b815ff49cf88d3514c6b6d9dfd855f52f61e3a11619098e9f387813990ab256d  dist/system1-0.1.1-py3-none-any.whl
f61f5fc222f1c2c0b3a21242a11c293e775bd206a8b95e9c88979cca6e14ffaf  dist/system1-0.1.1.tar.gz
```

---

## 6. Installed Distribution Artifact Verification

A clean isolated virtual environment (`/tmp/reflex_wheel_env`) outside the repository source tree was populated with `dist/system1-0.1.1-py3-none-any.whl` and verified with `PYTHONPATH=""`:

1. **Acceptance Suite Execution**: Ran `pytest tests/test_previous_contracts_adapted.py tests/test_round4_variants.py` directly against the installed package -> 6 passed in 0.29s.
2. **Sentinel Execution & Effect Counts**: Tested positive authorization allowing tool execution (invoking tool sentinel exactly 1 time) and negative authorization denying unpermitted tool execution (invoking tool sentinel 0 times).
3. **Receipt Outcome Chaining & Ledger Reopening**: Recorded `SUCCEEDED` outcome linked to authorization receipt digest, reopened database with a fresh `ActionLedger` instance, and verified 100% cryptographic hash chain integrity (`reopened.verify_integrity() is True`).
4. **Twin Namespace Parity**: Verified identical exports and class identities between `import reflex` and `import system1`.

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

**release-candidate gates passed for offline authorization hardening, deterministic cache fidelity, runtime validation, and local artifact compilation.**
