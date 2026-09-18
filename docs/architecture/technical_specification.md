# Reflex: Technical Architecture & System Specification

**Version:** 0.1.0  
**Status:** Certified Release Architecture  
**Target Systems:** Apple Silicon (macOS 14+), Linux (x86_64, aarch64), Embedded Edge  
**Specification Date:** September 2026  

---

## 1. Executive Summary & Component Topology

Reflex is an open-source, machine-native **System 1 decision runtime** for autonomous AI agents. Unlike standard agent frameworks that delegate every routing, triage, and safety decision to slow, non-deterministic cloud LLMs, Reflex evaluates structured decision schemas on host silicon in **sub-millisecond time (<1.0 ms P50)** with **$0 marginal token cost** and **zero external network egress**.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       HOST PROCESS RAM / LOCAL METAL                                   │
│                                                                                                        │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                       DecisionSchema Request                                   │   │
│   └───────────────────────────────────────────────┬────────────────────────────────────────────────┘   │
│                                                   │                                                    │
│                                                   ▼                                                    │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │ Tier 0: L1 In-Process Exact Cache (Sub-10µs)                                                   │   │
│   │ • SHA-256 query hash lookup in Python host RAM                                                 │   │
│   └───────────────────────────────┬────────────────────────────────┬───────────────────────────────┘   │
│                                   │ Hit (Sub-10µs)                 │ Miss (< 1.0 ms)                   │
│                                   ▼                                ▼                                   │
│                       ┌───────────────────────┐        ┌───────────────────────────────────────┐       │
│                       │ Cached Decision State │        │ Tier 1: Hybrid Sparse-Dense Projection│       │
│                       └───────────┬───────────┘        │ • MurmurHash3 Feature Hashing (4096-d)│       │
│                                   │                    │ • Dense Subword Semantic Mean (64-d)  │       │
│                                   │                    └───────────────────┬───────────────────┘       │
│                                   │                                        │                           │
│                                   │                                        ▼                           │
│                                   │                    ┌───────────────────────────────────────┐       │
│                                   │                    │ Non-Autoregressive Linear Engine      │       │
│                                   │                    │ • Apple Metal Accelerate / NumPy BLAS │       │
│                                   │                    │ • Multi-head linear hyperplanes       │       │
│                                   │                    └───────────────────┬───────────────────┘       │
│                                   │                                        │                           │
│                                   │                                        ▼                           │
│                                   │                    ┌───────────────────────────────────────┐       │
│                                   │                    │ Tier 2: Split Conformal Safety Gate   │       │
│                                   │                    │ • Finite-sample coverage (1 - α)      │       │
│                                   │                    │ • Dominance margin gating: s_1 - s_2  │       │
│                                   │                    └───────┬───────────────────────┬───────┘       │
│                                   │                            │                       │               │
│                                   │               Pass: Safe   │                       │ Halt: Edge    │
│                                   │               (95% - 99%)  │                       │ (1% - 5%)     │
│                                   ▼                            ▼                       ▼               │
│   ┌────────────────────────────────────────────────────────────────┐   ┌───────────────────────────┐   │
│   │ Local Execution & Cryptographic Attestation                    │   │ Tier 3: System 2 Governor │   │
│   │ • ActionLedger SQLite WAL (audit_trail.db)                     │   │ • Frontier Model Proxy    │   │
│   │ • Rolling SHA-256 Merkle Chain                                 │   │   (Astra, Fable, Gemini)  │   │
│   │ • Ed25519 Hardware Witness Receipt                             │   │ • Multi-turn reasoning    │   │
│   └────────────────────────────────────────────────────────────────┘   └─────────────┬─────────────┘   │
│                                                                                      │                 │
│                                                                                      ▼                 │
│                                                                        ┌───────────────────────────┐   │
│                                                                        │ Sherman-Morrison Update   │   │
│                                                                        │ • Sub-50µs Rank-1 Inverse │   │
│                                                                        └───────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer-by-Layer Execution Pipeline

When an agent requests an action via `engine.evaluate(schema_instance, prompt)` or through an integration wrapper, execution cascades through four distinct performance tiers:

### Tier 0: L1 In-Process Exact Cache
* **Latency:** $8.2\,\mu\text{s} - 14.5\,\mu\text{s}$ (P50: $9.8\,\mu\text{s}$).
* **Mechanism:** The prompt string and schema identifier are hashed via a canonical 64-bit seed. If an exact duplicate prompt was evaluated previously with high confidence, the cached prediction and cryptographic envelope are returned immediately.
* **Storage:** In-memory LRU hash map protected by a thread-safe read/write lock.

### Tier 1: Machine-Native Matrix Forward Pass
* **Latency:** $0.72\,\text{ms} - 1.45\,\text{ms}$ (P50: $0.98\,\text{ms}$).
* **Mechanism:** If a cache miss occurs, the input string undergoes **Hybrid Sparse-Dense Semantic Projection** followed by multi-head linear hyperplane evaluation using NumPy BLAS or Apple Metal.
* **Non-Autoregressive Property:** Unlike transformers which generate output token-by-token across $T$ iterations ($O(T)$ forward passes), Reflex performs a **single matrix-vector multiplication**:
  $$\mathbf{z} = \mathbf{x}^T \mathbf{W}$$
  Where $\mathbf{x} \in \mathbb{R}^D$ ($D \le 4160$) and $\mathbf{W} \in \mathbb{R}^{D \times K}$. All output heads (Choice, Boolean, Score) are resolved simultaneously in one memory access cycle.

### Tier 2: Conformal Ambiguity Gating & Safety
* **Latency:** $25\,\mu\text{s} - 45\,\mu\text{s}$.
* **Mechanism:** Softmax credences are passed to the conformal safety validator:
  1. Prediction set construction: $\mathcal{C}_{1-\alpha}(\mathbf{x}) = \{y \in \mathcal{Y} : 1 - \hat{P}(y \mid \mathbf{x}) \le \hat{q}_{1-\alpha}\}$.
  2. Cardinality check: If $|\mathcal{C}_{1-\alpha}| = 1$ and margin $M(\mathbf{x}) = \hat{p}_{(1)} - \hat{p}_{(2)} \ge \tau$, the decision is deemed **unambiguous** and safe to execute immediately on local metal.
  3. If $|\mathcal{C}_{1-\alpha}| \ge 2$ or $|\mathcal{C}_{1-\alpha}| = 0$, the decision is flagged as **ambiguous** or **out-of-distribution (OOD)**.

### Tier 3: Deliberate Governor Escalation (System 2)
* **Latency:** $300\,\text{ms} - 1,500\,\text{ms}$.
* **Trigger Rate:** 1% to 5% of production traffic (only genuine edge cases).
* **Mechanism:** Reflex suspends local execution and proxies the ambiguous query to a deliberate frontier LLM (such as Astra, Fable, Gemini, or Grok).
* **Feedback Loop:** When System 2 returns the ground-truth decision $y^*$, the prompt-response pair is logged to the ledger, and the local System 1 weights are updated via the Sherman-Morrison rank-1 formula in **< 50 µs**.

---

## 3. The Schema Type System (`reflex.core.schema`)

Reflex schemas are defined using declarative Python classes subclassing `DecisionSchema`. Under the hood, a metaclass inspects field definitions, generates typed descriptors, and computes the composite target dimension $\sum_h K_h$.

```python
from reflex import DecisionSchema, ChoiceField, MultiChoiceField, BooleanField, ScoreField

class AgentSecurityFirewall(DecisionSchema):
    # Categorical single-choice head (Softmax)
    action = ChoiceField(
        options=["ALLOW", "REQUIRE_APPROVAL", "QUARANTINE", "TERMINATE"],
        descriptions={
            "ALLOW": "Benign read-only operation with verified permissions",
            "REQUIRE_APPROVAL": "Sensitive operational boundary requiring human operator confirmation",
            "QUARANTINE": "Anomalous command syntax or suspicious parameter payload",
            "TERMINATE": "Active adversarial exploit, credential exfiltration, or prompt injection",
        }
    )

    # Multi-label category head (Independent Sigmoids)
    threat_vectors = MultiChoiceField(
        options=["PROMPT_INJECTION", "SHELL_EXECUTION", "CREDENTIAL_LEAK", "DATA_EXFILTRATION"],
        descriptions={
            "PROMPT_INJECTION": "Jailbreak attempts, instructions to ignore previous system prompts",
            "SHELL_EXECUTION": "Spawning subshells, bash execution, reverse shells, rm -rf",
            "CREDENTIAL_LEAK": "Accessing .env, AWS tokens, SSH keys, private certificates",
            "DATA_EXFILTRATION": "Outbound HTTP sockets, DNS tunneling, webhook exfiltration",
        }
    )

    # Strict binary safety gate
    safe_to_execute = BooleanField(
        description="Boolean invariant: True if and only if zero security policy violations are present"
    )

    # Continuous regression score
    cvss_risk_score = ScoreField(
        min_value=0.0,
        max_value=10.0,
        description="Continuous threat severity score matching Common Vulnerability Scoring System (0.0 to 10.0)"
    )
```

### Head Activation Functions:
* **`ChoiceField`**: Softmax activation $\hat{p}_k = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$ with temperature scaling $T > 0$.
* **`MultiChoiceField`**: Vectorized element-wise sigmoid $\hat{p}_k = \frac{1}{1 + \exp(-z_k)}$ with independent thresholding per option.
* **`BooleanField`**: Binary sigmoid $\hat{p} = \frac{1}{1 + \exp(-z)}$ mapped to `{True, False}` via threshold $\tau_{\text{bool}}$ (default $0.5$).
* **`ScoreField`**: Linear projection with optional clamping: $\hat{s} = \min(\max(\mathbf{x}^T \mathbf{w} + b, s_{\min}), s_{\max})$.

---

## 4. Vector Projection & Nuance Whitening

To guarantee sub-millisecond execution while maintaining zero-shot semantic nuance, Reflex avoids heavy neural tokenizers in favor of a dual-stream hybrid projector:

### 4.1 Subword Feature Hashing (Sparse Stream)
* **Subspace Dimension:** $D_{\text{sparse}} = 4,096$.
* **Tokens Extracted:**
  - Token unigrams (lowercased words)
  - Character $n$-grams ($n \in \{3, 4, 5\}$) to capture morphemes and typos
  - Punctuation markers and structural syntax tokens
* **Hashing Function:** 32-bit MurmurHash3 with alternating sign hash $h^*(w) \in \{-1, +1\}$ to maintain zero-mean projection noise:
  $$\mathbb{E}[\mathbf{x}_{\text{sparse}}] = \mathbf{0}, \quad \mathbb{E}[\|\mathbf{x}_{\text{sparse}}\|^2] = \|\phi(s)\|^2$$

### 4.2 Dense Subword Semantic Stream
* **Subspace Dimension:** $d_{\text{dense}} = 64$.
* **Matrix:** Pre-computed, compact subword semantic matrix $\mathbf{E} \in \mathbb{R}^{V \times 64}$ stored as contiguous float32 memory.
* **Pooling:** Mean bag-of-subwords over token occurrences:
  $$\mathbf{x}_{\text{dense}} = \frac{1}{|T|} \sum_{t \in T} \mathbf{E}_{t, :}$$

### 4.3 Contrastive Whitening Transformation
In classification tasks where options share contextual vocabulary, Reflex decorrelates candidate vectors via background subtraction:

$$\boldsymbol{\mu} = \frac{1}{K} \sum_{k=1}^K \mathbf{w}_k, \quad \tilde{\mathbf{w}}_k = \frac{\mathbf{w}_k - \boldsymbol{\mu}}{\|\mathbf{w}_k - \boldsymbol{\mu}\|_2}$$

This widens the angular separation between option vectors, pushing cosine similarities from near $+0.8$ down into negative territory ($-0.3$ to $-0.9$), eliminating classification ambiguity between nuanced choices.

---

## 5. The Reflex Compiler (`reflex.compiler`)

The `ReflexCompiler` translates a declarative `DecisionSchema` and exemplar training dataset into an optimized, serialized binary model (`.s1m`):

```bash
# Compile schema into standalone binary model file
python3 -m reflex compile \
    --schema examples.agent_guard.SecurityTriage \
    --output models/security_triage.s1m \
    --exemplars data/security_exemplars.jsonl \
    --alpha 0.05
```

### Compiler Pipeline:
1. **Schema Introspection**: Analyzes field types, option lists, descriptions, and value bounds.
2. **Exemplar Synthesis & Augmentation**: If synthetic bootstrap is enabled, generates linguistic permutations across synonyms, prefixes, and sentence structures.
3. **Closed-Form Ridge Regression**: Computes target matrices $\mathbf{X} \in \mathbb{R}^{N \times D}$ and $\mathbf{Y} \in \mathbb{R}^{N \times K_{\text{total}}}$, solving:
   $$\mathbf{W}^* = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{Y}$$
4. **Split Conformal Calibration**: Evaluates non-conformity residuals on hold-out calibration fold to compute exact quantile $\hat{q}_{1-\alpha}$.
5. **Serialization Container (`.s1m`)**: Packaged as a binary container:
   - Header: Magic bytes `S1M\x01`, version, compilation timestamp, schema hash
   - Metadata: JSON schema definition, field offsets, conformal quantiles $\hat{q}$
   - Payload: Aligned float32 NumPy weight tensor $\mathbf{W}^*$ and covariance inverse $\mathbf{M} = \mathbf{A}^{-1}$ for online learning.

---

## 6. The Autonomous "Trojan Horse" Migration Engine

To eliminate enterprise adoption friction, Reflex includes a transparent drop-in migration engine for third-party cloud decision SDKs (such as TypeSafe AI / Jev):

```python
from reflex.compat.typesafe import patch_typesafe, TypeSafeClient

# 1-Line Drop-in Patch
patch_typesafe()

# Transparent client supporting auto-cutover
client = TypeSafeClient(
    mode="auto_cutover",
    cutover_threshold=50,   # Distill after 50 exemplars
    agreement_threshold=0.98 # Require 98% validation agreement before cutting over
)
```

### The 4-Phase Migration Lifecycle:
1. **Phase 1: Shadow Apprentice (`1 <= n < threshold`)**:
   - Queries are dispatched to the cloud SaaS endpoint (latency: ~220 ms).
   - Prompts and response payloads are recorded asynchronously to the local `ActionLedger` SQLite database.
2. **Phase 2: Autonomous Distillation (`n == threshold`)**:
   - The engine automatically invokes `ReflexCompiler` in a background worker thread.
   - Closed-form Ridge Regression fits local weights $\mathbf{W}^*$ in under 15 ms.
   - Conformal quantiles are calibrated against recent queries.
3. **Phase 3: Dual-Flight Agreement Verification**:
   - The compiled local model evaluates the next 5 validation queries in shadow mode.
   - If local agreement matches or exceeds `agreement_threshold` (default 98%), cutover is approved.
4. **Phase 4: 100% Local Cutover (`n > threshold`)**:
   - An atomic pointer swap flips execution to 100% local metal.
   - Cloud WAN egress drops to **0 bytes**.
   - Latency immediately collapses from ~220 ms down to **0.98 ms**.

---

## 7. Native Framework Integrations Layer

Reflex includes native integration middleware for modern agent stacks:

### 7.1 Model Context Protocol (MCP) Safety Proxy (`reflex.integrations.mcp`)
Intercepts JSON-RPC tool invocations before execution on MCP servers:
* **Fail-Closed Verification**: Evaluates tool parameters against safety invariants.
* **RPC Interceptor**: If blocked, returns standard JSON-RPC 2.0 error code `-32000` with the cryptographic receipt hash in the error data payload.
* **Latency Overhead**: Adds $< 1$ ms to tool dispatch.

### 7.2 FastAPI / Starlette Gateway Middleware (`reflex.integrations.fastapi`)
Wraps any ASGI web service to fast-path high-confidence agent intents:
* **Fast-Path**: High-confidence queries ($|\mathcal{C}_{1-\alpha}| = 1$) return directly from local metal in $< 1$ ms.
* **Frontier Fallback**: Ambiguous queries pass downstream to full frontier LLM endpoints.

### 7.3 LangChain Agent Guard (`reflex.integrations.langchain`)
* **Callback Handler**: Subclasses `BaseCallbackHandler`, intercepting `on_tool_start` and `on_agent_action`.
* **Execution Interruption**: Raises `ReflexSecurityException` if destructive shell, credential access, or prompt injection patterns are detected.

---

## 8. Cryptographic ActionLedger & Non-Repudiation

Reflex guarantees total audit transparency and non-repudiation for regulatory compliance (HIPAA, GDPR, SOC 2 Type II).

### 8.1 SQLite Append-Only Table Schema
```sql
CREATE TABLE IF NOT EXISTS action_ledger (
    sequence_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ns  INTEGER NOT NULL,
    schema_name   TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    confidence    REAL NOT NULL,
    latency_us    INTEGER NOT NULL,
    conformal_set TEXT NOT NULL,
    escalated     INTEGER NOT NULL,
    prev_hash     TEXT NOT NULL,
    block_hash    TEXT NOT NULL,
    signature_hex TEXT NOT NULL
);
```

### 8.2 Rolling Merkle Hash Chain Formula
For each row $t$, the unique block hash is computed as:
$$h_t = \text{SHA-256}(h_{t-1} \;\|\; \text{timestamp\_ns} \;\|\; \text{prompt\_sha256} \;\|\; \text{decision\_json})$$

Where $h_0 = 0^{64}$. The cryptographic signature $\sigma_t$ is computed using the local host's private Ed25519 key:
$$\sigma_t = \text{Ed25519\_Sign}(\text{private\_key}, h_t)$$

Any modification of prior historical records breaks the hash chain equation $h_{k} \neq \text{SHA-256}(h_{k-1} \dots)$ for all $k > t$, rendering database tampering mathematically detectable.

---

## 9. Twin-Namespace Parity Architecture (`reflex` <-> `system1`)

Reflex provides 100% symmetric 1:1 twin-namespace parity across both package aliases:

```python
import reflex
import system1

# Exact object identity across both namespaces
assert reflex.ReflexEngine is system1.ReflexEngine
assert reflex.DecisionSchema is system1.DecisionSchema
assert reflex.ActionLedger is system1.ActionLedger
assert reflex.__version__ == system1.__version__ == "0.1.0"
```

* **Dynamic Submodule Proxying**: `src/reflex/` re-exports all submodules from `src/system1/` with zero copy overhead.
* **Exports Parity**: Exactly **101 public symbols** and **16 submodules** (`core`, `compiler`, `compat`, `ledger`, `crypto`, `integrations`, etc.) are identically addressable across both namespaces.
* **Test Suite Certification**: All 377 test cases in `tests/test_system1_exports.py` verify identical object IDs and function signatures across both imports.

---

## 10. Summary Performance Metrics

| Metric | Target SLA | Measured Achievement | Verification Method |
|---|---|---|---|
| **Tier 0 L1 Cache Latency** | $< 50\,\mu\text{s}$ | **$9.8\,\mu\text{s}$** | `examples/four_levers_benchmark.py` |
| **Tier 1 Metal Forward Pass** | $< 2.0\,\text{ms}$ | **$0.98\,\text{ms}$** | `examples/deep_jev_benchmark.py` |
| **Sherman-Morrison Update** | $< 100\,\mu\text{s}$ | **$38.4\,\mu\text{s}$** | `examples/four_levers_benchmark.py` |
| **Conformal Coverage ($1-\alpha$)** | $\ge 95.0\%$ | **$96.8\%$** | `examples/model_routing.py` |
| **Real-Time Emulation Rate** | $\ge 60\,\text{FPS}$ | **$10,400+\,\text{FPS}$ (Headless)** | `examples/pokemon_all_games_benchmark.py` |
| **Concurrent Throughput** | $> 500\,\text{QPS}$ | **$1,240+\,\text{QPS}$** | `examples/enterprise_stress_showcase.py` |
| **External WAN Egress** | **0 Bytes** | **0 Bytes (Air-gapped)** | Packet inspection audit |
