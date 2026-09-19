# Test Infrastructure Specification: System 1 / System 1

## 1. Philosophy: Opaque-Box, Requirement-Driven Verification

The test infrastructure for System 1 / System 1 is engineered strictly around **opaque-box, requirement-driven verification**, directly anchored in the specifications set forth in `ORIGINAL_REQUEST.md` and `PROJECT.md`.

In an opaque-box regime:
- **No Reliance on Implementation Details**: Tests evaluate observable system behavior, input-output invariants, interface contracts, performance bounds, and security properties rather than private internal method signatures.
- **Specification Invariance**: Tests remain valid and authoritative regardless of refactoring, caching levers, or algorithmic optimization underneath the public API surface.
- **Progressive Verifiability & Independence**: Every test is self-contained, isolated, deterministic, creates and disposes of its own resources (temporary ledgers, keypairs, mock sockets), and does not depend on test execution order.

---

## 2. Test Design Methodology

Our test design methodology synthesizes four complementary testing formalisms:

### 2.1 Category-Partition Testing
Inputs, system states, and configuration parameters are partitioned into equivalence categories:
- **Execution Namespaces**: `reflex` vs. `system1` dual import trees.
- **Operational Modes**: Standalone decision engine, Reference Monitor guard hook, TypeSafe SDK drop-in client, CLI commands.
- **Input Types**: Canonical natural language prompts, structured tool proposals, empty strings, multi-kilobyte prompt payloads, malformed JSON, and control characters.
- **Risk Tiers**: `READ_ONLY` (0), `REVERSIBLE` (1), `EXTERNAL` (2), `IRREVERSIBLE` (3).
- **Conformal States**: Singleton confident set ($|C(x)| = 1$), ambiguous multi-candidate set ($|C(x)| > 1$), and out-of-distribution empty set ($|C(x)| = 0$).

### 2.2 Boundary Value Analysis (BVA)
System behaviors are rigorously probed at edge boundaries:
- **Confidence & Significance Thresholds**: $\alpha \in \{0.0, 0.001, 0.05, 0.10, 0.99, 1.0\}$; $\text{min\_confidence} \in \{0.0, 0.5, 0.85, 0.99, 1.0\}$.
- **Batch Extremes**: Empty batch ($N = 0$), unitary ($N = 1$), small ($N = 10$), and high-throughput bulk ($N = 500, 1000$).
- **Prompt Size Boundaries**: Empty string ($L = 0$), single character ($L = 1$), standard prompt ($L \approx 100$), and extreme stress prompt ($L > 10,000$).
- **Cryptographic & Hash Validation**: SHA-256 strings of length 63, 64, 65 characters; non-hex characters; empty hashes; corrupted signature payloads.

### 2.3 Pairwise Combinatorial Testing
Interaction faults often manifest at the intersection of two distinct features:
- **Conformal Safety Gate $\times$ Cryptographic Witness Receipts**: Verifying that ambiguous or OOD decisions still issue signed receipts with non-conformal flags and correct digest bindings.
- **SQLite Action Ledger $\times$ Concurrent Multi-Threading**: Verifying append-only SHA-256 hash chaining integrity under concurrent writer threads without deadlock or chain corruption.
- **TypeSafe Drop-in SDK $\times$ Air-Gapped Zero-Network Execution**: Verifying drop-in client calls execute purely through local neural/semantic projectors with hard socket blocks in place.
- **Compiled Model (.s1m) $\times$ Sherman-Morrison Distillation**: Verifying that closed-form rank-1 updates to compiled models update hyperplanes without corrupting binary serializations or cache indices.

### 2.4 Real-World Workload Testing
Realistic mission-critical operational patterns representing actual production deployments:
1. **Autonomous AI Tool Guard Pipeline**: High-frequency interceptor of agent action proposals (file system reads, git commands, database transactions, destructive rm -rf, credential leakage) enforcing fail-closed protection.
2. **Dynamic Gateway Model Router**: Real-time triage classifier routing requests between local sub-2ms reflex execution and cloud LLM escalations based on conformal ambiguity sets.
3. **Auto-Cutover Migration Simulation**: Shadow evaluation and progressive traffic cutover from external LLM APIs to local System 1 runtime, measuring agreement rate, latency drop, and zero data egress.

---

## 3. Feature Inventory & Coverage Mapping

| Feature ID | Feature Description | E2E Test Suite / Tier | Primary Invariants Verified |
|---|---|---|---|
| **F1** | `__all__` Submodule Exports | `tests/e2e/test_tier1_dual_imports.py` | Explicit `__all__` defined across all submodules; import parity |
| **F2** | `reflex.core` & `reflex.neural` Packages | `tests/e2e/test_tier1_dual_imports.py` | Mirror module resolution and symbol equivalence |
| **F3** | Dynamic Submodule Dispatch | `tests/e2e/test_tier1_dual_imports.py` | `getattr(reflex, name)` resolves all 15 submodules identically to `system1` |
| **F4** | `GuardInterceptionResult.allowed` | `tests/e2e/test_tier1_conformal.py`, `test_tier4_real_world.py` | `.allowed == (outcome == DecisionOutcome.ALLOW)` boolean property |
| **F5** | CLI Script Registrations | `tests/e2e/test_tier1_cli.py` | `system1` and `reflex` entrypoint parsing and execution |
| **F6** | Optional Dependencies Handling | `tests/e2e/test_tier2_boundaries.py` | Graceful degradation and optional package handling |
| **F7** | Repository Cleanliness & Isolation | `tests/e2e/test_tier1_zero_network.py` | Clean file descriptors, isolated temp databases and keypairs |
| **F8** | Submodule Parity Automated Tests | `tests/e2e/test_tier1_dual_imports.py` | 15 submodules tested for 100% symbol parity and identity |
| **F10** | Zero External Network Egress | `tests/e2e/test_tier1_zero_network.py` | Hard socket blocking during inference, guard, and ledger operations |
| **F11** | Dual-Process Cognitive Architecture | `tests/e2e/test_tier4_real_world.py` | System 1 fast reflex path vs System 2 escalation trigger |
| **F14** | Sub-2ms Latency Guarantee | `tests/e2e/test_tier1_latency.py` | Empirical warm decision latency < 2.0 ms; L1 cache < 50 µs |
| **F18** | End-to-End Requirement Test Suite | `tests/e2e/` (All Tiers 1–4) | Complete opaque-box test suite verifying all system requirements |

---

## 4. Test Suite Tier Structure

```
tests/e2e/
├── __init__.py
├── conftest.py                       # Global fixtures: isolated sockets, temp DBs, keys
├── test_tier1_dual_imports.py        # Tier 1.1: reflex vs system1 dual imports (>=5 tests)
├── test_tier1_cli.py                 # Tier 1.2: CLI commands (>=5 tests)
├── test_tier1_zero_network.py        # Tier 1.3: Zero-network verification (>=5 tests)
├── test_tier1_latency.py             # Tier 1.4: Sub-2ms local latency assertions (>=5 tests)
├── test_tier1_receipts.py            # Tier 1.5: Cryptographic witness receipts (>=5 tests)
├── test_tier1_ledger.py              # Tier 1.6: SQLite ledger integrity (>=5 tests)
├── test_tier1_conformal.py           # Tier 1.7: Conformal prediction safety gate (>=5 tests)
├── test_tier2_boundaries.py          # Tier 2: Boundary & Corner Cases (empty, extreme, corrupt)
├── test_tier3_combinations.py        # Tier 3: Cross-Feature Combinations (multithread, crypto+gate)
└── test_tier4_real_world.py          # Tier 4: Real-World Scenarios (tool guard, router, cutover)
```

---

## 5. Execution & Verification

### Test Runner Command
To execute the comprehensive E2E test suite:
```bash
python3 -m pytest tests/e2e/ -v
```

To run all repository tests including unit and integration tests:
```bash
python3 -m pytest tests/ -v
```

### Coverage Criteria
- 100% pass rate across all test suites.
- Zero socket egress during marked local execution tests.
- High-precision monotonic latency validation under standard operating environments.
- Cryptographic signature validation conforming to Ed25519 standard.
