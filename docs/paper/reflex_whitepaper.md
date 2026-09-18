# Reflex: Non-Autoregressive System 1 Decision Runtime with Conformal Ambiguity Gating and Hardware Audit Receipts

**steph4n**  
*Reflex Core Research Team*  
X: [`@steph4n`](https://x.com/steph4n) • September 2026

---

## Abstract

Modern autonomous agent frameworks rely almost universally on autoregressive Large Language Models (LLMs) to make low-level discrete decisions—such as tool selection, action routing, parameter classification, and state triage. This architectural paradigm introduces prohibitive latency (300 ms to 1,500 ms per round-trip), unbounded economic cost ($0.03 to $0.15 per thousand tokens), non-deterministic execution jitter, and severe data privacy vulnerabilities via external Wide Area Network (WAN) egress. 

Inspired by Kahneman’s dual-process cognitive framework, we introduce **Reflex**, an open-source, machine-native **System 1 decision runtime** engineered for sub-millisecond execution directly on host silicon. Reflex implements non-autoregressive decision classification through high-dimensional hybrid sparse-dense vector projection and closed-form Ridge Regression. To guarantee operational safety without human intervention, Reflex integrates **Split Conformal Prediction**, bounding decision error with finite-sample coverage guarantees ($P(Y^* \in \mathcal{C}_{1-\alpha}(X)) \ge 1 - \alpha$) and escalating ambiguous prompts to an external deliberative System 2 governor (e.g., Astra, Fable, Gemini, Grok). Every execution is sealed into an append-only, SHA-256 hash-chained SQLite ledger signed with local Ed25519 cryptographic keys, providing non-repudiable audit receipts with zero network sockets opened. 

In empirical benchmarks, Reflex achieves a **9.8 µs L1 exact-cache hit latency**, a **0.98 ms P50 cold forward-pass latency**, and sustains over **1,000 queries per second (QPS)** across concurrent worker threads on Apple Silicon Metal and Linux BLAS. In a real-world Game Boy emulator testbed running Pokémon Red/Blue at 60 FPS (16.6 ms/frame), Reflex evaluates memory-mapped combat states and issues optimal controller inputs 16 times per hardware frame at $0 marginal cost. Furthermore, via an autonomous "Trojan Horse" migration engine, Reflex demonstrates zero-downtime distillation from SaaS endpoints (such as TypeSafe AI) into local metal, reducing cloud API egress by 98.4%.

---

## 1. Introduction & Problem Formulation

### 1.1 The Autoregressive Bottleneck in Autonomous Agents
The prevailing architectural pattern in agentic artificial intelligence consists of wrapping a foundation LLM in an iterative ReAct (Reasoning + Acting) loop. Under this model, every perception, routing decision, safety filter, and tool invocation requires an autoregressive forward pass through an attention network comprising tens or hundreds of billions of parameters:

$$\tau_{\text{total}} = \tau_{\text{WAN}} + \tau_{\text{prefill}} + \sum_{k=1}^K \tau_{\text{decode}}(t_k)$$

Where $\tau_{\text{WAN}}$ represents network packet serialization and transit (typically 80–350 ms), $\tau_{\text{prefill}}$ denotes prompt key-value matrix computation, and $\tau_{\text{decode}}$ represents step-by-step token generation. Consequently, total decision latency $\tau_{\text{total}}$ rarely drops below 250 ms and frequently exceeds 1,500 ms. 

While acceptable for asynchronous conversational interfaces, this latency profile is fatal for high-frequency, closed-loop cyber-physical and software control environments:
1. **Real-Time Interactive Systems**: Video game emulation, robotics, and industrial automation operate on rigid frame budgets (e.g., 60 Hz = 16.6 ms; 120 Hz = 8.33 ms). Autoregressive calls cause complete frame drops.
2. **High-Throughput Gateways**: API proxies, autonomous firewalls, and customer triage routers processing 5,000 requests per second face unsustainable infrastructural and token costs ($10,000s/month) when proxying micro-decisions to cloud APIs.
3. **Regulatory Privacy & Zero-Egress Invariants**: Modern enterprises bound by HIPAA, GDPR, and SOC 2 Type II regulations cannot transmit Protected Health Information (PHI) or proprietary trade secrets across third-party internet APIs.

### 1.2 The Dual-Process Cognitive Architecture
Cognitive science has long established that biological intelligence does not invoke slow, deliberative reasoning for recurring, pattern-matched stimuli. In Daniel Kahneman’s dual-process model (*Thinking, Fast and Slow*):
* **System 1 (Reflex)**: Operates automatically, fast, and instinctively with near-zero energy consumption and no sense of voluntary control.
* **System 2 (Deliberation)**: Allocates attention to effortful mental operations, including complex computations, novel problem solving, and formal logic.

Reflex implements the **machine-native System 1 layer**. It resolves 95% to 99% of high-frequency agent actions in $<1$ ms on local silicon, halting and escalating to a frontier System 2 model (such as Astra, Fable, Gemini, or Grok) **only when mathematically provable ambiguity is detected**.

```
                           ┌───────────────────────────────┐
                           │    Input Prompt / Action      │
                           └───────────────┬───────────────┘
                                           │
                                           ▼
                           ┌───────────────────────────────┐
                           │   Reflex System 1 Engine      │ ◄─── In-Memory BLAS / Metal
                           │   • Latency: < 1.0 ms P50     │      (Zero Token Cost)
                           │   • Sub-10µs L1 Exact Hit     │
                           └───────────────┬───────────────┘
                                           │ P(y|x)
                                           ▼
                           ┌───────────────────────────────┐
                           │   Conformal Ambiguity Gate    │
                           │   Coverage: P(Y*∈C) ≥ 1 - α   │
                           └───────┬───────────────┬───────┘
                                   │               │
                     Margin Safe   │               │ Ambiguous / Edge Case
                  M(x) ≥ τ (95-99%)│               │ M(x) < τ (1-5%)
                                   ▼               ▼
          ┌───────────────────────────┐    ┌───────────────────────────────┐
          │   Execute on Local Metal  │    │  System 2 Deliberate Governor │
          │   • SQLite ActionLedger   │    │  • Frontier Cloud LLM         │
          │   • Ed25519 Signature     │    │  • Multi-turn Reasoning       │
          └───────────────────────────┘    └───────────────┬───────────────┘
                                                           │
                                                           │ Distillation Feedback
                                                           ▼ (Sherman-Morrison < 50µs)
                                           ┌───────────────────────────────┐
                                           │  Rank-1 Fast Model Adaptation │
                                           └───────────────────────────────┘
```

---

## 2. Mathematical Foundations & Vector Projection

### 2.1 Hybrid Sparse-Dense Semantic Projection
Pure lexical n-gram representations (sparse bag-of-words or hashing trick) suffer from semantic rigidity: synonyms and paraphrases that share no subwords fail to project onto proximate coordinates. Conversely, dense contextual embeddings (e.g., standard 4096-dimensional transformer representations) require deep multi-layer matrix multiplication, violating the sub-millisecond execution budget.

Reflex resolves this trade-off via **Hybrid Sparse-Dense Subword Projection**. Let an input string $s$ be decomposed into token unigrams, character $n$-grams ($n \in \{3, 4, 5\}$), and normalized word stems.

#### Sparse Component
We apply the MurmurHash3 feature hashing trick into a compact sparse subspace $\mathbb{R}^{D_{\text{sparse}}}$ (default $D_{\text{sparse}} = 4096$):

$$\mathbf{x}_{\text{sparse}}[h_i(w)] = \sum_{w \in s} \text{sign}(h_i^*(w)) \cdot \text{IDF}(w)$$

Where $h_i(w) \in \{0, \dots, D_{\text{sparse}}-1\}$ and $\text{sign}(h_i^*(w)) \in \{-1, +1\}$ ensure unbiased expectation $\mathbb{E}[\langle \mathbf{x}, \mathbf{x}' \rangle] = \langle \phi(s), \phi(s') \rangle$.

#### Dense Semantic Component
Concurrently, each subword token $t \in T(s)$ indexes a static, pre-computed dense subword semantic matrix $\mathbf{E} \in \mathbb{R}^{V \times d_{\text{dense}}}$ ($d_{\text{dense}} = 64$):

$$\mathbf{x}_{\text{dense}} = \frac{1}{|T(s)|} \sum_{t \in T(s)} \mathbf{E}_{t, :}$$

#### Normalized Concatenation Fusion
The final representation $\mathbf{x} \in \mathbb{R}^{D}$ ($D = D_{\text{sparse}} + d_{\text{dense}}$) is formed via $L_2$-normalized linear fusion:

$$\mathbf{x} = \alpha \frac{\mathbf{x}_{\text{sparse}}}{\|\mathbf{x}_{\text{sparse}}\|_2} \oplus (1-\alpha) \frac{\mathbf{x}_{\text{dense}}}{\|\mathbf{x}_{\text{dense}}\|_2}$$

Where $\alpha \in [0, 1]$ (empirically tuned to $\alpha = 0.65$) balances lexical precision against semantic generalization.

### 2.2 Contrastive Option Centering & Whitening
In multi-choice decision routing (`ChoiceField`), target options frequently share domain vocabulary (e.g., "approve standard loan" vs. "approve expedited loan"). Unnormalized projection results in highly correlated option prototypes $\mathbf{w}_k$, compressing the angular distance between distinct decisions.

Reflex implements **Contrastive Centering & Whitening**. Given prototype vectors $\{\mathbf{w}_1, \mathbf{w}_2, \dots, \mathbf{w}_K\} \subset \mathbb{R}^D$ representing decision options:
1. Compute the background centroid vector across all candidate choices:
   $$\boldsymbol{\mu} = \frac{1}{K} \sum_{k=1}^K \mathbf{w}_k$$
2. Subtract the shared background centroid to isolate discriminating feature directions:
   $$\mathbf{w}_k' = \mathbf{w}_k - \boldsymbol{\mu}$$
3. Project each option vector onto the unit hypersphere:
   $$\tilde{\mathbf{w}}_k = \frac{\mathbf{w}_k'}{\|\mathbf{w}_k'\|_2}$$

**Theorem 1 (Angular Expansion)**. *Let $\mathbf{w}_1, \mathbf{w}_2$ have cosine similarity $\cos \theta = \langle \mathbf{w}_1, \mathbf{w}_2 \rangle > 0$. If $\mathbf{w}_1, \mathbf{w}_2$ share an identical additive background component $\mathbf{b}$ such that $\mathbf{w}_k = \mathbf{v}_k + \mathbf{b}$ with $\langle \mathbf{v}_1, \mathbf{v}_2 \rangle = 0$ and $\|\mathbf{v}_k\|_2 \ll \|\mathbf{b}\|_2$, then contrastive centering yields $\langle \tilde{\mathbf{w}}_1, \tilde{\mathbf{w}}_2 \rangle = -1$, maximizing decision hyperplane margin.*

---

## 3. Closed-Form Distillation & Fast Online Adaptation

### 3.1 Multi-Head Closed-Form Ridge Regression
When synthesizing a new schema or compiling synthetic exemplars, Reflex does not employ stochastic gradient descent (SGD) or backpropagation. Instead, it solves for the global empirical risk minimizer in closed form in pure NumPy:

$$\mathcal{L}(\mathbf{W}) = \frac{1}{2N} \|\mathbf{X} \mathbf{W} - \mathbf{Y}\|_F^2 + \frac{\lambda}{2} \|\mathbf{W}\|_F^2$$

Setting the gradient $\nabla_{\mathbf{W}} \mathcal{L}(\mathbf{W}) = 0$ yields the closed-form normal equation:

$$\mathbf{W}^* = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}_D)^{-1} \mathbf{X}^T \mathbf{Y}$$

Because $D \le 4160$ and typical exemplar sets comprise $N \in [50, 2000]$, the covariance matrix $\mathbf{A} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})$ is symmetric positive definite (SPD). Reflex computes the Cholesky decomposition $\mathbf{A} = \mathbf{L} \mathbf{L}^T$ in under 12 ms, avoiding iterative training loops entirely.

### 3.2 Sub-50µs Online Calibration via Sherman-Morrison Updates
When System 2 resolves an edge case, or a human supervisor provides an online corrective label $(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})$, the System 1 model must absorb this feedback immediately without triggering a complete $O(D^3)$ re-inversion.

Let $\mathbf{A}_t = \mathbf{X}_t^T \mathbf{X}_t + \lambda \mathbf{I}$ with known inverse $\mathbf{M}_t = \mathbf{A}_t^{-1}$. The arrival of a new observation vector $\mathbf{x}_{t+1} \in \mathbb{R}^D$ constitutes a rank-1 outer product perturbation:

$$\mathbf{A}_{t+1} = \mathbf{A}_t + \mathbf{x}_{t+1} \mathbf{x}_{t+1}^T$$

Applying the **Sherman-Morrison formula**, the exact inverse $\mathbf{M}_{t+1} = \mathbf{A}_{t+1}^{-1}$ is updated in $O(D^2)$ floating-point operations:

$$\mathbf{M}_{t+1} = \mathbf{M}_t - \frac{\mathbf{M}_t \mathbf{x}_{t+1} \mathbf{x}_{t+1}^T \mathbf{M}_t}{1 + \mathbf{x}_{t+1}^T \mathbf{M}_t \mathbf{x}_{t+1}}$$

The optimal weight matrix is then updated instantaneously:

$$\mathbf{W}_{t+1} = \mathbf{W}_t + \mathbf{M}_{t+1} \mathbf{x}_{t+1} (\mathbf{y}_{t+1}^T - \mathbf{x}_{t+1}^T \mathbf{W}_t)$$

In benchmarks on Apple M-series silicon, the entire rank-1 update completes in **38.4 µs**, enabling real-time continuous learning during live request streams.

---

## 4. Conformal Ambiguity Gating & Provable Safety

A major failure mode of small, specialized classifiers is **uncalibrated overconfidence on out-of-distribution (OOD) inputs**. Reflex guarantees safety through **Split Conformal Prediction**.

### 4.1 Calibration Protocol
Given an independent calibration dataset $\mathcal{D}_{\text{cal}} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ of size $n$, we define the non-conformity score $s_i$ as the model’s estimated residual error or negative softmax credence for the true class:

$$s_i = 1 - \hat{P}(Y = y_i \mid \mathbf{x}_i)$$

We sort the non-conformity scores in ascending order $s_{(1)} \le s_{(2)} \le \dots \le s_{(n)}$. For a user-specified significance level $\alpha \in (0, 1)$ (default $\alpha = 0.05$, corresponding to 95% coverage), we define the empirical conformal quantile index:

$$p = \frac{\lceil (n + 1)(1 - \alpha) \rceil}{n}$$

$$\hat{q}_{1-\alpha} = s_{(\lceil (n+1)(1-\alpha) \rceil)}$$

### 4.2 Prediction Set & Ambiguity Gating
For any unseen query $\mathbf{x}_{n+1}$, Reflex constructs the conformal prediction set $\mathcal{C}_{1-\alpha}(\mathbf{x}_{n+1})$ containing all classes whose non-conformity does not exceed the calibrated threshold:

$$\mathcal{C}_{1-\alpha}(\mathbf{x}_{n+1}) = \left\{ y \in \mathcal{Y} : 1 - \hat{P}(Y = y \mid \mathbf{x}_{n+1}) \le \hat{q}_{1-\alpha} \right\}$$

**Theorem 2 (Finite-Sample Coverage Guarantee)**. *Assume exchangeability of the calibration sequence $\mathcal{D}_{\text{cal}}$ and the test point $(\mathbf{x}_{n+1}, Y_{n+1})$. Then, without any distributional or parametric assumptions on $P(X, Y)$, the conformal prediction set satisfies:*

$$\mathbb{P}\left( Y_{n+1} \in \mathcal{C}_{1-\alpha}(\mathbf{x}_{n+1}) \right) \ge 1 - \alpha$$

*Proof*. By exchangeability, the rank of the test non-conformity score $S_{n+1} = 1 - \hat{P}(Y_{n+1} \mid X_{n+1})$ among $\{S_1, \dots, S_n, S_{n+1}\}$ is uniformly distributed over $\{1, 2, \dots, n+1\}$. Therefore:

$$\mathbb{P}(S_{n+1} \le \hat{q}_{1-\alpha}) = \mathbb{P}\left( \text{Rank}(S_{n+1}) \le \lceil (n+1)(1-\alpha) \rceil \right) = \frac{\lceil (n+1)(1-\alpha) \rceil}{n+1} \ge 1 - \alpha$$

This concludes the proof. $\blacksquare$

### 4.3 Fail-Closed Escalation Policy
Reflex applies a strict, fail-closed operational gating rule:
1. **Pass Condition (Fast-Path on Metal)**: If $|\mathcal{C}_{1-\alpha}(\mathbf{x}_{n+1})| = 1$ (the prediction set contains exactly one unambiguous candidate) AND the top-1 margin dominance satisfies $M(\mathbf{x}) = \hat{p}_{(1)} - \hat{p}_{(2)} \ge \tau$, the action executes locally in $<1$ ms.
2. **Halt Condition (Escalate to System 2)**: If $|\mathcal{C}_{1-\alpha}(\mathbf{x}_{n+1})| \ge 2$ (the model is uncertain between two or more plausible options) OR $|\mathcal{C}_{1-\alpha}(\mathbf{x}_{n+1})| = 0$ (the input is completely out of distribution), Reflex halts local execution and forwards the query to the frontier System 2 governor.

---

## 5. Cryptographic Receipts & Zero-Egress Invariants

### 5.1 Append-Only SQLite ActionLedger
Every decision evaluated by Reflex is recorded in an append-only, local SQLite database (`audit_trail.db`) operating in Write-Ahead Logging (`WAL`) mode with `PRAGMA synchronous = NORMAL`.

To guarantee tamper-evidence against database manipulation, records are chained via cryptographic SHA-256 rolling hashes:

$$h_t = \text{SHA-256}\left( h_{t-1} \;\|\; t \;\|\; \text{canonical\_json}(\text{payload}_t) \right)$$

Where $h_0 = 0^{64}$ is the genesis block hash. Any post-hoc row insertion, alteration, or deletion invalidates the subsequent cryptographic hash chain.

### 5.2 Ed25519 Hardware Witness Receipts
Each Reflex deployment maintains an on-device Ed25519 asymmetric keypair stored in `~/.system1/identity/`. For every decision, Reflex constructs a `RunWitnessEnvelope` containing:
* Canonical decision telemetry (action, parameters, confidence, latency)
* Conformal gate status ($|\mathcal{C}_{1-\alpha}|$, $\hat{q}_{1-\alpha}$)
* Rolling SHA-256 ledger hash $h_t$
* Ed25519 digital signature generated via the Edwards-curve Digital Signature Algorithm (RFC 8032):

$$\sigma_t = \text{Sign}_{\text{sk}}\left( h_t \;\|\; \text{timestamp} \right)$$

Third-party auditors can independently verify the authenticity, provenance, and ordering of agent decisions using the public key `identity.pub` without accessing the runtime environment.

### 5.3 Physical Zero-Egress Invariant
Reflex is constructed under a zero-dependency architecture. The core runtime imports exclusively from Python’s standard library and `numpy`. No HTTP clients (e.g., `requests`, `httpx`, `urllib.request`), socket libraries, or background telemetry services are bundled or imported during local evaluation. 

The runtime guarantees that **zero network packets** are transmitted outside the host machine during System 1 inference, satisfying the strict requirements of HIPAA (PHI data boundary) and GDPR (cross-border data sovereignty).

---

## 6. Empirical Evaluation & Benchmarks

### 6.1 Latency Comparison Across Architectures
We benchmarked Reflex on an Apple M3 Max (14-core CPU, 36 GB Unified Memory) running macOS 15 and Linux Ubuntu 24.04 LTS against leading alternative decision paradigms across 10,000 independent trials.

| Runtime Architecture | Hardware Location | Execution Paradigm | P50 Latency | P99 Latency | Relative Speedup | WAN Egress |
|---|---|---|---|---|---|---|
| **Reflex Tier 0 (L1 Cache)** | Host Memory | In-Process Exact Hash | **0.0098 ms (9.8 µs)** | **0.014 ms** | **86,700×** | **0 Bytes** |
| **Reflex System 1 (Metal)** | Host Metal/BLAS | Non-Autoregressive Matrix | **0.98 ms** | **1.34 ms** | **867×** | **0 Bytes** |
| Local 8B LLM (vLLM / Ollama) | Local GPU (RTX 4090) | Autoregressive (KV Cache) | 180.00 ms | 245.00 ms | 4.7× | 0 Bytes |
| Cloud Fast API (Groq / Cerebras)| US-East WAN | Autoregressive Specialized | 220.00 ms | 410.00 ms | 3.9× | Full Payload |
| Frontier Cloud LLM (Astra/Gemini)| Cloud WAN | Autoregressive Frontier | 850.00 ms | 1,480.00 ms | 1.0× (Baseline) | Full Payload |
| ReAct Multi-Turn Cloud Agent | Cloud WAN | Multi-call ReAct Tool Loop | 3,200.00 ms | 6,800.00 ms | 0.26× | Full Payload |

Reflex delivers an **867× latency improvement** over single-call cloud LLMs and an **86,700× improvement** on repeated cache hits.

### 6.2 60 FPS Real-Time Game Boy Control (Pokémon Red/Blue)
To test Reflex under unforgiving real-time constraints, we interfaced the runtime directly with the `PyBoy` Game Boy hardware emulator running *Pokémon Red*. Game Boy hardware ticks at exactly 60.0 Hz (16.6 ms per frame).

```
Hardware Frame Budget:  |================| (16.6 ms)
Cloud LLM Latency:      |======================================================...| (850 ms -> 51 frame drops)
Reflex System 1 Pass:   |=| (0.98 ms -> 16 evaluations per single frame)
```

* **Observation Window**: Reflex extracted battle RAM vectors from `$D057` (Battle Mode), `$D015` (Player HP), `$CFE6` (Enemy HP), and move selection tables.
* **Throughput**: In headless benchmark mode, Reflex sustained **10,400+ FPS**, evaluating 38 µs forward passes per state transition.
* **Combat Decision**: Navigated FIGHT menu $\to$ TACKLE selection, executing attacks and processing HP bar depleting animations at native 60 FPS with zero stutter.

### 6.3 Autonomous "Trojan Horse" Migration Engine
We deployed `TypeSafeClient(mode="auto_cutover")` into an autonomous agent tool pipeline processing live requests:
* **Initial 50 Queries**: Transparently proxied to SaaS API baseline while asynchronously recording inputs and outputs to `ActionLedger`.
* **Cutover Event (Query 51)**: `ReflexCompiler` solved closed-form Ridge Regression in 14.2 ms, calibrated conformal bounds at $\alpha = 0.05$, verified 100% agreement on validation exemplars, and flipped execution to 100% local metal.
* **Post-Cutover Performance**: Latency immediately collapsed from 220 ms to 0.98 ms, while cloud data egress dropped to 0 bytes.

### 6.4 Concurrency & Ledger Integrity
Under a multi-threaded stress test simulating 8 concurrent worker threads executing an enterprise security firewall schema (48 attack vectors including prompt injection, data exfiltration, and unauthorized shell execution):
* Sustained throughput exceeded **1,240 QPS**.
* Zero SQLite lock contention occurred under WAL journal mode.
* Post-run audit verified 100% unbroken SHA-256 chain integrity across all records.

---

## 7. Related Work

1. **Speculative Decoding**: Leviathan et al. (2023) use small draft models to propose candidate tokens verified by target LLMs. While effective for text generation, speculative decoding remains fundamentally autoregressive and cannot break the sub-10ms barrier required for discrete state decisions.
2. **Dual-Process Cognitive AI**: Booch et al. (2021) and Bengio (2017) conceptualized the fusion of System 1 fast heuristics with System 2 deliberate logic. Reflex translates this theoretical framework into an empirical, machine-native software runtime.
3. **Conformal Risk Control**: Angelopoulos & Bates (2021) established distribution-free uncertainty quantification for neural networks. Reflex extends this to discrete agentic routing, providing finite-sample safety bounds without heuristic thresholds.
4. **Hardware-Attested Computing**: Previous verifiable computing frameworks rely on confidential VMs or enclave technologies (e.g., Intel SGX). Reflex introduces cryptographic non-repudiation at the application runtime layer via append-only SQLite hash chains and Ed25519 witness envelopes.

---

## 8. Conclusion

Reflex addresses the fundamental latency, economic, and privacy bottlenecks of contemporary agentic AI. By decoupling high-frequency System 1 reflex actions from deliberate System 2 reasoning, Reflex enables agents to operate at machine-native speed (<1ms P50 latency, 60 FPS hardware budgets) while upholding rigorous mathematical safety guarantees through Split Conformal Prediction and Ed25519 cryptographic audit receipts. Reflex is fully open source under the Apache 2.0 license at `https://github.com/steph4n-gh/reflex`.

---

## References

1. Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
2. Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511.
3. Sherman, J., & Morrison, W. J. (1950). *Adjustment of an Inverse Matrix Corresponding to a Change in One of the Elements*. Annals of Mathematical Statistics, 21(1), 124–127.
4. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer Science & Business Media.
5. Bernstein, D. J., Duif, N., Lange, T., Schwabe, P., & Yang, B. Y. (2012). *High-speed high-security signatures*. Journal of Cryptographic Engineering, 2(2), 77–89.
6. Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Large Language Models via Speculative Decoding*. International Conference on Machine Learning (ICML).
7. Booch, G., Fabiano, F., Horesh, L., et al. (2021). *Thinking Fast and Slow in AI*. Proceedings of the AAAI Conference on Human Computation and Crowdsourcing.
