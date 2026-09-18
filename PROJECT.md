# Project: Reflex / System 1 Repository Polish & Packaging

## Architecture
Reflex / System 1 is a dual-process cognitive architecture providing ultra-low-latency, non-autoregressive decision making on local metal (System 1 fast reflex) coupled with conformal ambiguity gating and online distillation from strategic deliberative planners (System 2 deliberate governor).

Key Architectural Pillars:
1. **Dual-Process Cognitive Execution**:
   - **System 1 (Reflex Engine)**: Non-autoregressive parallel evaluation on local hardware (< 2ms latency, pure NumPy/MLX, zero token costs, zero external network egress).
   - **Four Tier 1 Levers**:
     - *Lever 1 (Tier 0 Cache)*: Exact hash / L1 reflex cache (< 10 µs execution).
     - *Lever 2 (Sherman-Morrison Distillation)*: Instant closed-form rank-1 covariance update (< 50 µs) transferring System 2 resolutions into System 1 hyperplanes.
     - *Lever 3 (Margin Dominance Gating)*: Early rejection when leading option margin exceeds threshold.
     - *Lever 4 (Telemetry & Signal Fusion)*: Multi-head sensor/context integration.
   - **Conformal Safety Gate**: Finite-sample mathematical uncertainty guard ($1-\alpha$ coverage) detecting out-of-distribution ambiguity and halting for System 2 escalation.
   - **System 2 (Deliberative Governor)**: Slow, analytical reasoning (cloud LLM, heuristic planner, human-in-the-loop) invoked only when System 1 halts or flags ambiguity.
2. **Reference Monitor & Audit Ledger**:
   - Fail-closed tool interception (`ReflexGuardHook`) before side effects take place.
   - Append-only cryptographic SQLite ledger (`ActionLedger`) with SHA-256 hash chaining.
   - Ed25519 digital signatures on all decision witness receipts (`DecisionWitnessReceipt`).
3. **Drop-in Twin Packaging & API Parity**:
   - Complete symmetry between `import reflex` and `import system1`.
   - Identical public interface (92 exports), versioning (`0.1.0`), and all 15 submodules.
   - Drop-in TypeSafe AI SDK compatibility (`patch_typesafe()`, `AsyncTypeSafeClient`).

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | `__all__` Submodule Exports | Define explicit `__all__` in `system1.engine`, `calibration`, `guard`, `ledger`, `receipt` | M1 (DONE) | Survey (E1, E2, E3) |
| F2 | `reflex.core` & `reflex.neural` Packages | Create mirror modules in `src/reflex/` for `core/` and `neural.py` | M1 (DONE) | Survey (E1, E2) |
| F3 | Dynamic Submodule Dispatch in `reflex` | Implement `_MODULE_MAP` and `__getattr__`/`__dir__` in `src/reflex/__init__.py` & `src/system1/__init__.py` | M1 (DONE) | Survey (E2) |
| F4 | `GuardInterceptionResult.allowed` Property | Add `@property def allowed(self) -> bool` to `GuardInterceptionResult` in `src/system1/guard.py` | M1 (DONE) | Survey (E2) |
| F5 | CLI Script Registrations | Add `reflex = "reflex.cli:main"` to `[project.scripts]` in `pyproject.toml` | M1 (DONE) | Survey (E1, E2, E3) |
| F6 | Optional GameBoy Dependencies | Add `gameboy = ["pyboy>=2.0.0"]` to `[project.optional-dependencies]` in `pyproject.toml` | M1 (DONE) | Survey (E3) |
| F7 | Repository Cleanliness & `.gitignore` | Add `.agents/`, `*.s1m`, `._*`, IDE artifacts to `.gitignore` | M1 (DONE) | Survey (E1, E2, E3) |
| F8 | Submodule Parity Automated Tests | Add tests in `tests/test_system1_exports.py` & adversarial suite verifying all 15 submodules in isolated subprocesses | M1 (DONE) | Survey (E1, E2) |
| F9 | Test Badge & Count Harmonization | Update `README.md` badge (366 passed) and text (366+ tests in < 25s) | Final (DONE) | Survey (E1, E2, E3, User) |
| F10 | Dedicated Privacy & Zero Egress Section | Create `## Privacy & Zero Data Egress` in `README.md` matching `#privacy--zero-data-egress` anchor | M2 (DONE) | Survey (E2) |
| F11 | Dual-Process Cognitive Architecture Section | Add prominent Kahneman System 1 / System 2 cognitive architecture section to `README.md` | M2 (DONE) | Survey (E1, E2) |
| F12 | README Quickstart Code Fixes | Fix quickstart code snippets (guard `.allowed` and cache margin threshold) in `README.md` | M2 (DONE) | Survey (E2) |
| F13 | Complete 17-Demo Catalog in README | Document all 17 production demos and benchmarks in `README.md` including Universal Paperclips (`examples/paperclips_typesafe_dropin.py`) | M2 (DONE) | Survey (E1, E2, E3, User) |
| F14 | Sub-2ms vs 20ms Target Clarification | Clarify 20ms frame budget ceiling vs sub-2ms empirical decision latency in README & CLI | M2 (DONE) | Survey (E2) |
| F15 | Empirical Benchmark Verification | Validate all 17 demo and benchmark scripts in `examples/` execute cleanly and reproduce metrics (including `paperclips_typesafe_dropin.py`) | M3 (DONE) | Survey (E3) |
| F16 | Fast Intro Skipping Unit Test | Add dedicated unit test for `fast_skip_intro` with mock memory in `tests/` | M3 (DONE) | Survey (E3) |
| F17 | 100% Automated Test Suite Invariance | Maintain 100% pass rate across all 366+ pytest items with zero regressions | Final (DONE) | ORIGINAL_REQUEST |
| F18 | E2E Requirement Test Suite (Tiers 1-4) | Comprehensive opaque-box test suite verifying all features independently (TEST_READY.md) | E2E (DONE) | ORIGINAL_REQUEST |
| F19 | Adversarial Coverage Hardening (Tier 5) | White-box stress testing and edge-case hardening with Challenger | Final (DONE) | Project Pattern |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Package Integrity & Namespace Parity | F1, F2, F3, F4, F5, F6, F7, F8 | None | DONE |
| M2 | Cohesive Documentation & Architectural Alignment | F9, F10, F11, F12, F13, F14 | M1 | DONE |
| M3 | End-to-End Demo & Benchmark Verification | F15, F16 (All 17 demos verified including `paperclips_typesafe_dropin.py`) | M1 | DONE |
| E2E | E2E Testing Track | F18 (Tiers 1-4 test suite, TEST_INFRA.md, TEST_READY.md) | None | DONE |
| Final | Test Suite Pass & Adversarial Hardening | F17, F19 (100% pytest pass, E2E suite pass, Tier 5 hardening) | M1, M2, M3, E2E | DONE |

---

## Interface Contracts
### `reflex` ↔ `system1`
- Submodule parity: Every module `system1.<name>` has an identical `reflex.<name>`.
- Export symmetry: `reflex.<name>.__all__ == system1.<name>.__all__`.
- Object identity: `getattr(reflex, sym) is getattr(system1, sym)` for all symbols.
- Attribute delegation: Dynamic resolution for all 15 submodules in both namespaces via `_MODULE_MAP` and `_SUBMODULE_NAMES`.

### `GuardInterceptionResult` API Contract
- `interception.allowed`: Boolean property returning `True` iff `interception.outcome == DecisionOutcome.ALLOW`.

### CLI Contract
- Both `system1` and `reflex` console scripts registered in `[project.scripts]` pointing to their respective `main` entrypoints.

---

## Code Layout
```
src/
├── reflex/
│   ├── __init__.py          # Dynamic module dispatch & 92 exports
│   ├── cache.py
│   ├── calibration.py
│   ├── cli.py
│   ├── compat/
│   ├── compiler.py
│   ├── core/                # Core non-autoregressive models & neural projector
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── model.py
│   │   ├── neural.py
│   │   ├── schema.py
│   │   └── telemetry.py
│   ├── embeddings.py
│   ├── engine.py
│   ├── guard.py
│   ├── ledger.py
│   ├── model.py
│   ├── neural.py
│   ├── receipt.py
│   ├── schema.py
│   └── telemetry.py
└── system1/                 # Canonical System 1 implementation (identical structure)
tests/                       # Pytest automated test suite (366+ tests, 366 passed)
├── e2e/                     # Opaque-box requirement test suite (58 tests across Tiers 1-4)
examples/                    # 17 runnable demo and benchmark scripts (including Universal Paperclips showcase)
```
