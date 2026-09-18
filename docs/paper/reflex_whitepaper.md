# Reflex: Non-Autoregressive System 1 Decision Runtime with Conformal Ambiguity Gating and Cryptographic Witness Receipts

**steph4n** (`@steph4n` on X)  
*Reflex Core Research Team*  
Repository: [`https://github.com/steph4n-gh/reflex`](https://github.com/steph4n-gh/reflex)  
March 2026

> **Suggested Citation:**  
> steph4n (2026). *Reflex: Non-Autoregressive System 1 Decision Runtime with Conformal Ambiguity Gating and Cryptographic Witness Receipts*. Reflex Core Research Team. Available at: `https://github.com/steph4n-gh/reflex`.

---

## Abstract

Modern autonomous agent architectures rely almost universally on autoregressive Large Language Models (LLMs) to execute low-level discrete decisions—such as tool selection, action routing, parameter classification, and state triage. This paradigm introduces prohibitive latency (300 ms to 1,500 ms per round-trip), unbounded economic expenditure ($0.03 to $0.15 per thousand tokens), non-deterministic execution jitter, and severe data sovereignty vulnerabilities via external Wide Area Network (WAN) egress.

Inspired by Kahneman’s dual-process cognitive framework, we introduce **Reflex**, an open-source, machine-native **System 1 decision runtime** engineered for sub-millisecond execution directly on host silicon. Reflex implements non-autoregressive decision classification through high-dimensional hybrid sparse-dense vector projection and closed-form multi-head Ridge Regression. To guarantee operational safety without human intervention, Reflex integrates **Split Conformal Prediction**, bounding decision error with finite-sample coverage guarantees ($\mathbb{P}(Y_{n+1} \in \mathcal{C}_{1-\alpha}(\mathbf{x}_{n+1})) \ge 1 - \alpha$) and escalating ambiguous prompts to an external deliberative System 2 governor (such as OpenAI Astra & GPT-6 series [Sol, Terra, Luna], Anthropic Claude Opus 5 / Fable 5.1 / Mythos 5, Google Gemini 3.1 Pro & 3.8 Flash, or xAI Grok). Every execution is sealed into an append-only, SHA-256 hash-chained SQLite ledger signed with local Ed25519 cryptographic keys, providing non-repudiable audit receipts with zero network sockets opened.

In empirical benchmarks, Reflex achieves a **9.8 µs L1 exact-cache hit latency**, a **0.98 ms P50 cold forward-pass latency**, and sustains over **1,000 queries per second (QPS)** across concurrent worker threads on Apple Silicon Metal and Linux BLAS. In a real-world Game Boy emulator testbed running Pokémon Red/Blue at 60 FPS (16.6 ms frame budget), Reflex evaluates memory-mapped combat states and issues policy-prescribed controller actions in 0.98 ms—consuming only 5.9% of the frame budget and permitting up to 16 state evaluations per hardware frame with zero marginal cloud API token expenditure. Furthermore, via an autonomous apprentice-to-metal distillation engine, Reflex demonstrates zero-downtime cutover from SaaS endpoints (such as TypeSafe AI) into local metal, reducing cloud API egress by 98.4% across production traffic (factoring in the 1.6% System 2 edge escalation rate) and by 100% for all localized decisions.

---

## 1. Introduction & Problem Formulation

### 1.1 The Autoregressive Bottleneck in Autonomous Agents
The prevailing architectural pattern in agentic artificial intelligence consists of wrapping a foundation LLM in an iterative ReAct (Reasoning + Acting) loop. Under this model, every perception, routing decision, safety filter, and tool invocation requires an autoregressive forward pass through an attention network comprising tens or hundreds of billions of parameters:

$$\tau_{\text{total}} = \tau_{\text{WAN}} + \tau_{\text{prefill}} + \sum_{k=1}^K \tau_{\text{decode}}(t_k)$$

Where $\tau_{\text{WAN}}$ represents network packet serialization and transit (typically 80–350 ms), $\tau_{\text{prefill}}$ denotes prompt key-value matrix computation, and $\tau_{\text{decode}}$ represents step-by-step token generation. Consequently, total decision latency $\tau_{\text{total}}$ rarely drops below 250 ms and frequently exceeds 1,500 ms.

While acceptable for asynchronous conversational interfaces, this latency profile is fatal for high-frequency, closed-loop cyber-physical and software control environments:
1. **Real-Time Interactive Systems**: Video game emulation, robotics, and industrial automation operate on rigid frame budgets (e.g., 60 Hz = 16.6 ms; 120 Hz = 8.33 ms). Autoregressive calls cause complete frame drops.
2. **High-Throughput Gateways**: API proxies, autonomous firewalls, and customer triage routers processing 5,000 requests per second face unsustainable infrastructural and token costs (tens of thousands of dollars monthly in enterprise production) when proxying micro-decisions to cloud APIs.
3. **Regulatory Privacy & Zero-Egress Invariants**: Modern enterprises bound by HIPAA, GDPR, and SOC 2 Type II regulations cannot transmit Protected Health Information (PHI) or proprietary trade secrets across third-party internet APIs.

### 1.2 The Dual-Process Cognitive Architecture
Cognitive science has long established that biological intelligence does not invoke slow, deliberative reasoning for recurring, pattern-matched stimuli. In Daniel Kahneman’s dual-process model (*Thinking, Fast and Slow*):
* **System 1 (Reflex)**: Operates automatically, fast, and instinctively with near-zero energy consumption and no sense of voluntary control.
* **System 2 (Deliberation)**: Allocates attention to effortful mental operations, including complex computations, novel problem solving, and formal logic.

Reflex implements the **machine-native System 1 layer**. It resolves 95% to 99% of high-frequency agent actions in $<1$ ms on local silicon, halting and escalating to a frontier System 2 deliberative governor—such as OpenAI Astra & GPT-6 series (Sol, Terra, Luna), Anthropic Claude Opus 5 / Fable 5.1 / Mythos 5, Google Gemini 3.1 Pro & 3.8 Flash, or xAI Grok—**strictly when the conformal prediction set indicates statistical ambiguity ($|\mathcal{C}_{1-\alpha}| \ne 1$) or margin deficiency ($M(\mathbf{x}) < \tau$)**.

```
                           ┌───────────────────────────────┐
                           │    Input Prompt / Action      │
                           └───────────────┬───────────────┘
                                           │
                                           ▼
                           ┌───────────────────────────────┐
                           │   Reflex System 1 Engine      │ ◄─── In-Memory BLAS / Metal
                           │   • Latency: < 1.0 ms P50     │      (Zero Token Overhead)
                           │   • Sub-10µs L1 Exact Hit     │
                           └───────────────┬───────────────┘
                                           │ P(y|x)
                                           ▼
                           ┌───────────────────────────────┐
                           │   Conformal Ambiguity Gate    │
                           │ Coverage: P(Y_{n+1}∈C) ≥ 1-α  │
                           └───────┬───────────────┬───────┘
                                   │               │
                     Margin Safe   │               │ Ambiguous / OOD Edge Case
                  M(x) ≥ τ (95-99%)│               │ |C| ≠ 1 or M(x) < τ (1-5%)
                                   ▼               ▼
          ┌───────────────────────────┐    ┌───────────────────────────────┐
          │   Execute on Local Metal  │    │  System 2 Deliberate Governor │
          │   • SQLite ActionLedger   │    │  • Frontier Reasoning Model   │
          │   • Ed25519 Signature     │    │    (OpenAI Astra / GPT-6,     │
          └───────────────────────────┘    │     Claude Opus 5, Gemini 3.1,│
                                           │     xAI Grok)                 │
                                           └───────────────┬───────────────┘
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

Where $h_i(w) \in \{0, \dots, D_{\text{sparse}}-1\}$ is a 32-bit hash and $\text{sign}(h_i^*(w)) \in \{-1, +1\}$ is an independent sign hash. This ensures an unbiased projection satisfying $\mathbb{E}[\langle \mathbf{x}_{\text{sparse}}, \mathbf{x}_{\text{sparse}}' \rangle] = \langle \phi(s), \phi(s') \rangle$ with bounded variance (Weinberger et al., 2009).

#### Dense Semantic Component
Concurrently, each subword token $t \in T(s)$ indexes a static, pre-computed dense subword semantic matrix $\mathbf{E} \in \mathbb{R}^{V \times d_{\text{dense}}}$ ($d_{\text{dense}} = 64$):

$$\mathbf{x}_{\text{dense}} = \frac{1}{|T(s)|} \sum_{t \in T(s)} \mathbf{E}_{t, :}$$

#### Unit-Hypersphere Concatenation Fusion
The final representation $\mathbf{x} \in \mathbb{R}^{D}$ ($D = D_{\text{sparse}} + d_{\text{dense}} = 4096 + 64 = 4160$) is formed via square-root weighted hypersphere concatenation:

$$\mathbf{x} = \sqrt{\alpha} \frac{\mathbf{x}_{\text{sparse}}}{\|\mathbf{x}_{\text{sparse}}\|_2} \oplus \sqrt{1-\alpha} \frac{\mathbf{x}_{\text{dense}}}{\|\mathbf{x}_{\text{dense}}\|_2}$$

Where $\alpha \in [0, 1]$ (empirically tuned to $\alpha = 0.65$) balances lexical precision against semantic generalization. This formulation strictly preserves unit-norm geometry without post-hoc scaling:

$$\|\mathbf{x}\|_2^2 = \alpha \frac{\|\mathbf{x}_{\text{sparse}}\|_2^2}{\|\mathbf{x}_{\text{sparse}}\|_2^2} + (1-\alpha) \frac{\|\mathbf{x}_{\text{dense}}\|_2^2}{\|\mathbf{x}_{\text{dense}}\|_2^2} = \alpha(1.0) + (1-\alpha)(1.0) = 1.0$$

---

### 2.2 Contrastive Option Centering & Whitening

In multi-choice decision routing (`ChoiceField`), candidate option prototypes frequently share significant background context (e.g., shared prompt templates, task framing, and lexical stems like "approve standard loan" versus "approve expedited loan").

Let each prototype vector be decomposed into a common background component $\mathbf{b} \in \mathbb{R}^D$ and a class-distinctive feature vector $\mathbf{v}_k \in \mathbb{R}^D$:

$$\mathbf{w}_k = \mathbf{b} + \mathbf{v}_k, \quad k \in \{1, \dots, K\}$$

When background intensity dominates distinctive features ($\|\mathbf{b}\|_2 = B \gg \|\mathbf{v}_k\|_2 = c$), the uncentered prototype vectors are confined to an acute cone around $\mathbf{b}$:

$$\cos \theta_{jk} = \frac{\langle \mathbf{w}_j, \mathbf{w}_k \rangle}{\|\mathbf{w}_j\|_2 \|\mathbf{w}_k\|_2} \approx \frac{B^2}{B^2 + c^2} = 1 - \frac{c^2}{B^2 + c^2} \longrightarrow 1 \quad \text{as } B/c \to \infty$$

This severe angular compression degrades linear classifier margins and causes numerical instability in softmax discrimination. Reflex applies **Contrastive Centering & Whitening**:
1. Compute the empirical background centroid across all candidate choices:
   $$\boldsymbol{\mu} = \frac{1}{K} \sum_{k=1}^K \mathbf{w}_k$$
2. Subtract the shared centroid to isolate class-discriminative feature directions:
   $$\mathbf{w}_k' = \mathbf{w}_k - \boldsymbol{\mu}$$
3. Project each option vector onto the unit hypersphere $\mathbb{S}^{D-1}$:
   $$\tilde{\mathbf{w}}_k = \frac{\mathbf{w}_k'}{\|\mathbf{w}_k'\|_2}$$

---

**Theorem 1 (Generalized Angular Expansion & Simplex Equiangular Separation)**.  
*Let candidate vectors $\mathbf{w}_1, \dots, \mathbf{w}_K \in \mathbb{R}^D$ ($K \ge 2$, $D \ge K-1$) satisfy $\mathbf{w}_k = \mathbf{b} + \mathbf{v}_k$, where $\mathbf{b} \in \mathbb{R}^D$ is an arbitrary shared background vector and $\mathbf{v}_1, \dots, \mathbf{v}_K \in \mathbb{R}^D$ are class-distinctive feature vectors. Let $\boldsymbol{\mu} = \frac{1}{K}\sum_{k=1}^K \mathbf{w}_k$, $\mathbf{w}_k' = \mathbf{w}_k - \boldsymbol{\mu}$, and $\tilde{\mathbf{w}}_k = \mathbf{w}_k' / \|\mathbf{w}_k'\|_2$.*

1. **Binary Antipodal Separation ($K=2$):**  
   *For any two distinct vectors $\mathbf{w}_1 \ne \mathbf{w}_2$, contrastive centering yields diametrically opposite unit vectors:*
   $$\langle \tilde{\mathbf{w}}_1, \tilde{\mathbf{w}}_2 \rangle = -1, \quad \tilde{\mathbf{w}}_1 = -\tilde{\mathbf{w}}_2$$
   *In particular, if $\mathbf{v}_1 = -\mathbf{v}_2$, the shared background $\mathbf{b}$ is identically eliminated.*

2. **Multiclass Simplex Equiangular Separation ($K \ge 2$):**  
   *Suppose the distinctive feature vectors $\{\mathbf{v}_k\}_{k=1}^K$ are mutually orthogonal with equal norm: $\langle \mathbf{v}_j, \mathbf{v}_k \rangle = c^2 \delta_{jk}$ for $c > 0$. Then:*
   $$\langle \tilde{\mathbf{w}}_j, \tilde{\mathbf{w}}_k \rangle = -\frac{1}{K - 1} \quad \forall j \ne k$$
   *The set $\{\tilde{\mathbf{w}}_1, \dots, \tilde{\mathbf{w}}_K\}$ forms the vertices of a regular $(K-1)$-simplex inscribed in $\mathbb{S}^{D-1}$ centered at the origin ($\sum_{k=1}^K \tilde{\mathbf{w}}_k = \mathbf{0}$).*

3. **Theoretical Optimality:**  
   *The pairwise inner product $-\frac{1}{K-1}$ strictly achieves the Welch / Rankin lower bound for the maximum cosine similarity among any $K$ unit vectors in $\mathbb{R}^D$:*
   $$\min_{\mathbf{u}_1, \dots, \mathbf{u}_K \in \mathbb{S}^{D-1}} \max_{j \ne k} \langle \mathbf{u}_j, \mathbf{u}_k \rangle = -\frac{1}{K-1}$$
   *Contrastive centering maximally expands angular separation and maximizes the decision hyperplane margin.*

---

*Proof of Theorem 1*.

**Proof of Part 1 ($K=2$ Binary Antipodal Case):**  
For $K=2$ with arbitrary distinct vectors $\mathbf{w}_1, \mathbf{w}_2 \in \mathbb{R}^D$, the centroid is $\boldsymbol{\mu} = \frac{1}{2}(\mathbf{w}_1 + \mathbf{w}_2)$. The centered vectors are:
$$\mathbf{w}_1' = \mathbf{w}_1 - \boldsymbol{\mu} = \mathbf{w}_1 - \frac{\mathbf{w}_1 + \mathbf{w}_2}{2} = \frac{\mathbf{w}_1 - \mathbf{w}_2}{2}$$
$$\mathbf{w}_2' = \mathbf{w}_2 - \boldsymbol{\mu} = \mathbf{w}_2 - \frac{\mathbf{w}_1 + \mathbf{w}_2}{2} = -\frac{\mathbf{w}_1 - \mathbf{w}_2}{2} = -\mathbf{w}_1'$$
Since $\mathbf{w}_1 \ne \mathbf{w}_2$, $\|\mathbf{w}_1'\|_2 > 0$. Normalizing onto $\mathbb{S}^{D-1}$ yields:
$$\tilde{\mathbf{w}}_1 = \frac{\mathbf{w}_1'}{\|\mathbf{w}_1'\|_2}, \quad \tilde{\mathbf{w}}_2 = \frac{-\mathbf{w}_1'}{\|\mathbf{w}_1'\|_2} = -\tilde{\mathbf{w}}_1$$
Evaluating the inner product:
$$\langle \tilde{\mathbf{w}}_1, \tilde{\mathbf{w}}_2 \rangle = \langle \tilde{\mathbf{w}}_1, -\tilde{\mathbf{w}}_1 \rangle = -\|\tilde{\mathbf{w}}_1\|_2^2 = -1$$
The angle widens from $\theta \approx 0$ to $\theta = \pi$ ($180^\circ$).

**Proof of Part 2 (General $K \ge 2$ Simplex Case):**  
Let $\mathbf{w}_k = \mathbf{b} + \mathbf{v}_k$ with $\langle \mathbf{v}_j, \mathbf{v}_k \rangle = c^2 \delta_{jk}$ ($c > 0$). The centroid is:
$$\boldsymbol{\mu} = \frac{1}{K}\sum_{k=1}^K (\mathbf{b} + \mathbf{v}_k) = \mathbf{b} + \bar{\mathbf{v}}, \quad \text{where } \bar{\mathbf{v}} = \frac{1}{K}\sum_{m=1}^K \mathbf{v}_m$$
The centered vector $\mathbf{w}_k'$ is:
$$\mathbf{w}_k' = \mathbf{w}_k - \boldsymbol{\mu} = (\mathbf{b} + \mathbf{v}_k) - (\mathbf{b} + \bar{\mathbf{v}}) = \mathbf{v}_k - \frac{1}{K}\sum_{m=1}^K \mathbf{v}_m$$
Notice that $\mathbf{b}$ is identically eliminated for all $k$. Expanding the inner product $\langle \mathbf{w}_j', \mathbf{w}_k' \rangle$ by bilinearity:
$$\langle \mathbf{w}_j', \mathbf{w}_k' \rangle = \langle \mathbf{v}_j, \mathbf{v}_k \rangle - \frac{1}{K}\sum_{l=1}^K \langle \mathbf{v}_j, \mathbf{v}_l \rangle - \frac{1}{K}\sum_{m=1}^K \langle \mathbf{v}_m, \mathbf{v}_k \rangle + \frac{1}{K^2}\sum_{m=1}^K \sum_{l=1}^K \langle \mathbf{v}_m, \mathbf{v}_l \rangle$$
Using the orthogonality condition $\langle \mathbf{v}_p, \mathbf{v}_q \rangle = c^2 \delta_{pq}$:
1. $\langle \mathbf{v}_j, \mathbf{v}_k \rangle = c^2 \delta_{jk}$.
2. $\sum_{l=1}^K \langle \mathbf{v}_j, \mathbf{v}_l \rangle = c^2$.
3. $\sum_{m=1}^K \langle \mathbf{v}_m, \mathbf{v}_k \rangle = c^2$.
4. $\sum_{m=1}^K \sum_{l=1}^K \langle \mathbf{v}_m, \mathbf{v}_l \rangle = \sum_{m=1}^K \|\mathbf{v}_m\|_2^2 = K c^2$.

Substituting these terms:
$$\langle \mathbf{w}_j', \mathbf{w}_k' \rangle = c^2 \delta_{jk} - \frac{c^2}{K} - \frac{c^2}{K} + \frac{K c^2}{K^2} = c^2 \left(\delta_{jk} - \frac{1}{K}\right)$$
Evaluating diagonal ($j = k$) and off-diagonal ($j \ne k$) elements:
* For $j = k$: $\|\mathbf{w}_k'\|_2^2 = c^2 \left(1 - \frac{1}{K}\right) = c^2 \left(\frac{K - 1}{K}\right)$, hence $\|\mathbf{w}_k'\|_2 = c \sqrt{\frac{K - 1}{K}} > 0$.
* For $j \ne k$: $\langle \mathbf{w}_j', \mathbf{w}_k' \rangle = -c^2 / K$.

Normalizing onto the unit hypersphere $\tilde{\mathbf{w}}_k = \mathbf{w}_k' / \|\mathbf{w}_k'\|_2$:
$$\langle \tilde{\mathbf{w}}_j, \tilde{\mathbf{w}}_k \rangle = \frac{\langle \mathbf{w}_j', \mathbf{w}_k' \rangle}{\|\mathbf{w}_j'\|_2 \|\mathbf{w}_k'\|_2} = \frac{-c^2 / K}{c^2 \left(\frac{K - 1}{K}\right)} = -\frac{1}{K - 1}$$

**Proof of Part 3 (Welch / Rankin Lower Bound Optimality):**  
For any arbitrary configuration of $K$ unit vectors $\mathbf{u}_1, \dots, \mathbf{u}_K \in \mathbb{S}^{D-1}$, consider the squared norm of their sum:
$$\left\| \sum_{k=1}^K \mathbf{u}_k \right\|_2^2 = \sum_{j=1}^K \sum_{k=1}^K \langle \mathbf{u}_j, \mathbf{u}_k \rangle = K + \sum_{j \ne k} \langle \mathbf{u}_j, \mathbf{u}_k \rangle \ge 0 \implies \sum_{j \ne k} \langle \mathbf{u}_j, \mathbf{u}_k \rangle \ge -K$$
Because there are $K(K-1)$ off-diagonal pairs:
$$\max_{j \ne k} \langle \mathbf{u}_j, \mathbf{u}_k \rangle \ge \frac{1}{K(K-1)} \sum_{j \ne k} \langle \mathbf{u}_j, \mathbf{u}_k \rangle \ge \frac{-K}{K(K-1)} = -\frac{1}{K - 1}$$
Equality holds if and only if $\sum_{k=1}^K \mathbf{u}_k = \mathbf{0}$ and all pairwise inner products are identical. Since $\sum_{k=1}^K \tilde{\mathbf{w}}_k = \mathbf{0}$ and $\langle \tilde{\mathbf{w}}_j, \tilde{\mathbf{w}}_k \rangle = -\frac{1}{K-1}$ uniformly for all $j \ne k$, the configuration constructed by Contrastive Centering & Whitening strictly achieves the theoretical minimum possible pairwise correlation in Euclidean geometry. $\blacksquare$

*Remark (Simplex Optimality in Practice)*: While the theoretical Welch bound minimum cosine similarity ($-\frac{1}{K-1}$) is achieved when distinctive prototype components $\{\mathbf{v}_k\}$ are mutually orthogonal with equal norm, high-dimensional natural language text embeddings in practice exhibit non-zero semantic correlations. In such real-world feature spaces, contrastive centering eliminates background bias $\mathbf{b}$ and maximally widens angular separation into an approximate equiangular simplex configuration, yielding substantial empirical margin gains.

---

#### Hypersphere Geometry & Margin Expansion
The Euclidean distance between distinct normalized candidate vectors is:
$$d(\tilde{\mathbf{w}}_j, \tilde{\mathbf{w}}_k) = \sqrt{\|\tilde{\mathbf{w}}_j\|_2^2 + \|\tilde{\mathbf{w}}_k\|_2^2 - 2\langle \tilde{\mathbf{w}}_j, \tilde{\mathbf{w}}_k \rangle} = \sqrt{2\left(1 + \frac{1}{K-1}\right)} = \sqrt{\frac{2K}{K-1}}$$
* For $K=2$: $d = \sqrt{4} = 2.0$ (antipodal diameter of the hypersphere).
* For $K=3$: $d = \sqrt{3} \approx 1.732$ (equilateral triangle inscribed in great circle).
* For $K=4$: $d = \sqrt{8/3} \approx 1.633$ (regular tetrahedron).
* As $K \to \infty$: $d \to \sqrt{2} \approx 1.414$ (orthogonal basis separation).

The geometric margin of prototype $\tilde{\mathbf{w}}_j$ to the separating hyperplane $\langle \tilde{\mathbf{w}}_j - \tilde{\mathbf{w}}_k, \mathbf{x} \rangle = 0$ is:
$$\gamma_{jk} = \frac{\langle \tilde{\mathbf{w}}_j, \tilde{\mathbf{w}}_j - \tilde{\mathbf{w}}_k \rangle}{\|\tilde{\mathbf{w}}_j - \tilde{\mathbf{w}}_k\|_2} = \frac{1 - \langle \tilde{\mathbf{w}}_j, \tilde{\mathbf{w}}_k \rangle}{\sqrt{2(1 - \langle \tilde{\mathbf{w}}_j, \tilde{\mathbf{w}}_k \rangle)}} = \frac{1}{2} d(\tilde{\mathbf{w}}_j, \tilde{\mathbf{w}}_k) = \sqrt{\frac{K}{2(K-1)}}$$
For uncentered prototypes with $\|\mathbf{b}\|_2 = B \gg c$, the uncentered margin is $\gamma_{jk}^{\text{uncentered}} \approx \frac{c}{\sqrt{2}B}$. The margin expansion ratio is:
$$\frac{\gamma_{jk}^{\text{centered}}}{\gamma_{jk}^{\text{uncentered}}} \approx \frac{B}{c} \sqrt{\frac{K}{K-1}} \gg 1$$
Because $B/c$ typically ranges from $10$ to $50$ in natural language representations, contrastive centering widens the decision boundary margin by **one to two orders of magnitude**, rendering the linear classifier robust against token perturbations.

---

## 3. Closed-Form Distillation & Fast Online Adaptation

### 3.1 Multi-Head Closed-Form Ridge Regression
When synthesizing a new schema or compiling synthetic exemplars, Reflex does not employ stochastic gradient descent (SGD) or backpropagation. Instead, it solves for the global empirical risk minimizer in closed form in pure NumPy.

Let the training dataset consist of $N$ exemplar embeddings arranged in design matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$ and multi-head target labels $\mathbf{Y} \in \mathbb{R}^{N \times K}$. We formulate the multi-head Ridge Regression objective:

$$\mathcal{L}(\mathbf{W}) = \frac{1}{2} \|\mathbf{X} \mathbf{W} - \mathbf{Y}\|_F^2 + \frac{\lambda}{2} \|\mathbf{W}\|_F^2$$

Where $\|\cdot\|_F$ denotes the Frobenius matrix norm and $\lambda > 0$ is the $L_2$ regularization parameter.

#### Matrix Calculus Derivation of Normal Equations
Using the trace representation of the Frobenius norm $\|\mathbf{Z}\|_F^2 = \operatorname{Tr}(\mathbf{Z}^T \mathbf{Z})$:
$$\|\mathbf{X} \mathbf{W} - \mathbf{Y}\|_F^2 = \operatorname{Tr}\left( (\mathbf{X} \mathbf{W} - \mathbf{Y})^T (\mathbf{X} \mathbf{W} - \mathbf{Y}) \right) = \operatorname{Tr}\left( \mathbf{W}^T \mathbf{X}^T \mathbf{X} \mathbf{W} - 2 \mathbf{W}^T \mathbf{X}^T \mathbf{Y} + \mathbf{Y}^T \mathbf{Y} \right)$$
Thus the objective expands to:
$$\mathcal{L}(\mathbf{W}) = \frac{1}{2}\operatorname{Tr}(\mathbf{W}^T \mathbf{X}^T \mathbf{X} \mathbf{W}) - \operatorname{Tr}(\mathbf{W}^T \mathbf{X}^T \mathbf{Y}) + \frac{1}{2}\operatorname{Tr}(\mathbf{Y}^T \mathbf{Y}) + \frac{\lambda}{2}\operatorname{Tr}(\mathbf{W}^T \mathbf{W})$$
Differentiating with respect to the weight matrix $\mathbf{W}$ using standard matrix derivative identities:
$$\nabla_{\mathbf{W}} \mathcal{L}(\mathbf{W}) = \mathbf{X}^T \mathbf{X} \mathbf{W} - \mathbf{X}^T \mathbf{Y} + \lambda \mathbf{W} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}_D) \mathbf{W} - \mathbf{X}^T \mathbf{Y}$$
Setting the gradient to zero yields the closed-form normal equation:
$$\nabla_{\mathbf{W}} \mathcal{L}(\mathbf{W}) = \mathbf{0} \implies \mathbf{W}^* = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}_D)^{-1} \mathbf{X}^T \mathbf{Y}$$

#### Strict Convexity & Uniqueness
The Hessian operator with respect to $\operatorname{vec}(\mathbf{W})$ is:
$$\nabla^2_{\mathbf{W}} \mathcal{L}(\mathbf{W}) = \mathbf{I}_K \otimes (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}_D)$$
Since $\mathbf{X}^T \mathbf{X} \succeq 0$ and $\lambda > 0$, all eigenvalues of $(\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}_D)$ satisfy $\lambda_i \ge \lambda > 0$. The Hessian is strictly positive definite ($\nabla^2_{\mathbf{W}} \mathcal{L} \succ 0$), guaranteeing that $\mathbf{W}^*$ is the unique global minimizer.

#### Cholesky Factorization
Rather than computing the explicit matrix inverse $(\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}_D)^{-1}$, Reflex computes the Cholesky factorization:
$$\mathbf{A} = \mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}_D = \mathbf{L} \mathbf{L}^T$$
where $\mathbf{L} \in \mathbb{R}^{D \times D}$ is lower-triangular with strictly positive diagonal entries. The system is solved via forward substitution $\mathbf{L} \mathbf{Z} = \mathbf{X}^T \mathbf{Y}$ followed by back substitution $\mathbf{L}^T \mathbf{W}^* = \mathbf{Z}$ in under 12 ms, avoiding numerical inversion instability.

---

#### Condition Number Spectral Bound

**Theorem 3.1 (Condition Number Bound)**.  
*Let $\mathbf{A} = \mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}_D \in \mathbb{R}^{D \times D}$ with $\lambda > 0$. If each sample embedding is unit-normalized ($\|\mathbf{x}_i\|_2 = 1$ for all $i \in \{1, \dots, N\}$), the $L_2$ condition number $\kappa(\mathbf{A}) = \|\mathbf{A}\|_2 \|\mathbf{A}^{-1}\|_2$ satisfies:*
$$\kappa(\mathbf{A}) \le 1 + \frac{N}{\lambda}$$

*Proof*. By the spectral theorem, the eigenvalues of $\mathbf{X}^T \mathbf{X}$ are $\sigma_i(\mathbf{X})^2 \ge 0$, so the eigenvalues of $\mathbf{A}$ are $\lambda_i(\mathbf{A}) = \sigma_i(\mathbf{X})^2 + \lambda$. The extreme eigenvalues satisfy $\lambda_{\max}(\mathbf{A}) = \sigma_{\max}(\mathbf{X})^2 + \lambda$ and $\lambda_{\min}(\mathbf{A}) \ge \lambda$. Hence:
$$\kappa(\mathbf{A}) = \frac{\lambda_{\max}(\mathbf{A})}{\lambda_{\min}(\mathbf{A})} \le \frac{\sigma_{\max}(\mathbf{X})^2 + \lambda}{\lambda} = 1 + \frac{\sigma_{\max}(\mathbf{X})^2}{\lambda}$$
Using the operator norm bound $\sigma_{\max}(\mathbf{X})^2 = \|\mathbf{X}\|_2^2 \le \|\mathbf{X}\|_F^2 = \sum_{i=1}^N \|\mathbf{x}_i\|_2^2 = N$, we conclude $\kappa(\mathbf{A}) \le 1 + \frac{N}{\lambda}$. $\blacksquare$

For typical exemplar sets ($N \approx 500$ to $2,000$) with default regularization $\lambda = 1.0$, $\kappa(\mathbf{A}) \le 2,001 \ll 10^7$, which is several orders of magnitude below the single-precision float32 reciprocal machine epsilon ($1/\epsilon_{\text{float32}} \approx 1.2 \times 10^7$). This guarantees numerical stability on host hardware.

---

### 3.2 Sub-50µs Online Calibration via Sherman-Morrison Updates

When System 2 resolves an edge case, or a human supervisor provides an online corrective label, the System 1 model must absorb this feedback immediately without triggering a complete $O(D^3)$ matrix re-inversion.

#### Matrix Dimensional Consistency
We define strict dimensional conventions:
* Input observation vector: $\mathbf{x}_{t+1} \in \mathbb{R}^{D \times 1}$ (column vector, transposed during row-major evaluation).
* Target label vector: $\mathbf{y}_{t+1} \in \mathbb{R}^{K \times 1}$ (column vector).
* Covariance matrix: $\mathbf{A}_t = \mathbf{X}_t^T \mathbf{X}_t + \lambda \mathbf{I}_D \in \mathbb{R}^{D \times D}$.
* Inverse covariance matrix: $\mathbf{M}_t = \mathbf{A}_t^{-1} \in \mathbb{R}^{D \times D}$.
* Cross-covariance matrix: $\mathbf{B}_t = \mathbf{X}_t^T \mathbf{Y}_t \in \mathbb{R}^{D \times K}$.
* Weight matrix: $\mathbf{W}_t \in \mathbb{R}^{D \times K}$ (stored internally as row-major `weights` of shape $(K, D)$ for BLAS Level-3 cache locality).

When a new observation arrives, the covariance matrix undergoes an exact rank-1 outer product perturbation:
$$\mathbf{A}_{t+1} = \mathbf{A}_t + \mathbf{x}_{t+1} \mathbf{x}_{t+1}^T$$

Applying the **Sherman-Morrison formula**, the exact inverse $\mathbf{M}_{t+1} = \mathbf{A}_{t+1}^{-1}$ is updated in $\mathcal{O}(D^2)$ floating-point operations:
$$\mathbf{M}_{t+1} = \mathbf{M}_t - \frac{\mathbf{M}_t \mathbf{x}_{t+1} \mathbf{x}_{t+1}^T \mathbf{M}_t}{1 + \mathbf{x}_{t+1}^T \mathbf{M}_t \mathbf{x}_{t+1}}$$

#### Verification of Inverse Validity: $\mathbf{A}_{t+1} \mathbf{M}_{t+1} = \mathbf{I}_D$
Multiplying $\mathbf{A}_{t+1} = (\mathbf{A}_t + \mathbf{x}_{t+1} \mathbf{x}_{t+1}^T)$ by the proposed inverse $\mathbf{M}_{t+1}$:
$$(\mathbf{A}_t + \mathbf{x}_{t+1} \mathbf{x}_{t+1}^T) \left( \mathbf{M}_t - \frac{\mathbf{M}_t \mathbf{x}_{t+1} \mathbf{x}_{t+1}^T \mathbf{M}_t}{1 + \mathbf{x}_{t+1}^T \mathbf{M}_t \mathbf{x}_{t+1}} \right)$$
$$= \mathbf{A}_t \mathbf{M}_t - \frac{\mathbf{A}_t \mathbf{M}_t \mathbf{x}_{t+1} \mathbf{x}_{t+1}^T \mathbf{M}_t}{1 + \mathbf{x}_{t+1}^T \mathbf{M}_t \mathbf{x}_{t+1}} + \mathbf{x}_{t+1} \mathbf{x}_{t+1}^T \mathbf{M}_t - \frac{\mathbf{x}_{t+1} (\mathbf{x}_{t+1}^T \mathbf{M}_t \mathbf{x}_{t+1}) \mathbf{x}_{t+1}^T \mathbf{M}_t}{1 + \mathbf{x}_{t+1}^T \mathbf{M}_t \mathbf{x}_{t+1}}$$
Using $\mathbf{A}_t \mathbf{M}_t = \mathbf{I}_D$ and factoring the rank-1 matrix $\mathbf{x}_{t+1} \mathbf{x}_{t+1}^T \mathbf{M}_t$:
$$= \mathbf{I}_D + \mathbf{x}_{t+1} \mathbf{x}_{t+1}^T \mathbf{M}_t \left( 1 - \frac{1}{1 + \mathbf{x}_{t+1}^T \mathbf{M}_t \mathbf{x}_{t+1}} - \frac{\mathbf{x}_{t+1}^T \mathbf{M}_t \mathbf{x}_{t+1}}{1 + \mathbf{x}_{t+1}^T \mathbf{M}_t \mathbf{x}_{t+1}} \right)$$
$$= \mathbf{I}_D + \mathbf{x}_{t+1} \mathbf{x}_{t+1}^T \mathbf{M}_t \left( 1 - \frac{1 + \mathbf{x}_{t+1}^T \mathbf{M}_t \mathbf{x}_{t+1}}{1 + \mathbf{x}_{t+1}^T \mathbf{M}_t \mathbf{x}_{t+1}} \right) = \mathbf{I}_D + \mathbf{0} = \mathbf{I}_D$$
Thus $\mathbf{M}_{t+1}$ is identically the exact inverse of $\mathbf{A}_{t+1}$.

---

**Theorem 3.2 (Recursive-to-Batch Exact Equivalence)**.  
*Let $\mathbf{W}_t = \mathbf{M}_t \mathbf{B}_t$ satisfy the batch normal equation at step $t$, where $\mathbf{B}_t = \mathbf{X}_t^T \mathbf{Y}_t$. Then the recursive update:*
$$\mathbf{W}_{t+1} = \mathbf{W}_t + \mathbf{M}_{t+1} \mathbf{x}_{t+1} (\mathbf{y}_{t+1}^T - \mathbf{x}_{t+1}^T \mathbf{W}_t)$$
*is identically equal to the batch Ridge Regression normal equation solution:*
$$\mathbf{W}_{t+1}^* = (\mathbf{X}_{t+1}^T \mathbf{X}_{t+1} + \lambda \mathbf{I}_D)^{-1} \mathbf{X}_{t+1}^T \mathbf{Y}_{t+1} = \mathbf{M}_{t+1} \mathbf{B}_{t+1}$$

*Proof*. The batch cross-covariance satisfies $\mathbf{B}_{t+1} = \mathbf{B}_t + \mathbf{x}_{t+1} \mathbf{y}_{t+1}^T$. Therefore:
$$\mathbf{W}_{t+1}^* = \mathbf{M}_{t+1} \mathbf{B}_{t+1} = \mathbf{M}_{t+1} \mathbf{B}_t + \mathbf{M}_{t+1} \mathbf{x}_{t+1} \mathbf{y}_{t+1}^T$$
Since $\mathbf{B}_t = \mathbf{A}_t \mathbf{W}_t$ and $\mathbf{A}_t = \mathbf{A}_{t+1} - \mathbf{x}_{t+1} \mathbf{x}_{t+1}^T$:
$$\mathbf{M}_{t+1} \mathbf{B}_t = \mathbf{M}_{t+1} (\mathbf{A}_{t+1} - \mathbf{x}_{t+1} \mathbf{x}_{t+1}^T) \mathbf{W}_t = (\mathbf{M}_{t+1} \mathbf{A}_{t+1}) \mathbf{W}_t - \mathbf{M}_{t+1} \mathbf{x}_{t+1} (\mathbf{x}_{t+1}^T \mathbf{W}_t)$$
Because $\mathbf{M}_{t+1} \mathbf{A}_{t+1} = \mathbf{I}_D$:
$$\mathbf{M}_{t+1} \mathbf{B}_t = \mathbf{W}_t - \mathbf{M}_{t+1} \mathbf{x}_{t+1} (\mathbf{x}_{t+1}^T \mathbf{W}_t)$$
Substituting this identity into the expression for $\mathbf{W}_{t+1}^*$:
$$\mathbf{W}_{t+1}^* = \left[ \mathbf{W}_t - \mathbf{M}_{t+1} \mathbf{x}_{t+1} (\mathbf{x}_{t+1}^T \mathbf{W}_t) \right] + \mathbf{M}_{t+1} \mathbf{x}_{t+1} \mathbf{y}_{t+1}^T = \mathbf{W}_t + \mathbf{M}_{t+1} \mathbf{x}_{t+1} (\mathbf{y}_{t+1}^T - \mathbf{x}_{t+1}^T \mathbf{W}_t)$$
This is identically the recursive update $\mathbf{W}_{t+1}$. No approximations are introduced. $\blacksquare$

#### Algorithmic Complexity: $\mathcal{O}(D^2)$ vs $\mathcal{O}(D^3)$
* Batch Cholesky re-inversion requires $\frac{1}{3}D^3 + \mathcal{O}(ND^2 + KD^2)$ flops. For $D=4160$, this requires $\approx 2.4 \times 10^{10}$ flops (12–50 ms).
* The Sherman-Morrison update requires $4D^2 + 4DK + \mathcal{O}(D)$ flops, totaling $\approx 6.9 \times 10^7$ flops.
* On Apple Silicon Metal and BLAS, this executes in **38.4 µs**—a **$>600\times$ speedup**, enabling continuous online learning during live request streams.

---

## 4. Conformal Ambiguity Gating & Provable Safety

A fundamental failure mode of small classifiers is **uncalibrated overconfidence on out-of-distribution (OOD) inputs**. Reflex replaces heuristic softmax thresholds with **Split Conformal Prediction** (Vovk et al., 2005; Angelopoulos & Bates, 2021).

### 4.1 Calibration Protocol
Let $\mathcal{Z} = \mathcal{X} \times \mathcal{Y}$ denote the sample space, where $\mathcal{X} \subseteq \mathbb{R}^D$ and $\mathcal{Y} = \{1, \dots, K\}$. We hold out an independent calibration dataset $\mathcal{D}_{\text{cal}} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ disjoint from the training fold.

We evaluate the non-conformity score $s_i$ as the residual uncertainty of the ground-truth label:
$$s_i = 1 - \hat{P}(Y = y_i \mid \mathbf{x}_i)$$

The calibration scores are sorted in ascending order: $S_{(1)} \le S_{(2)} \le \dots \le S_{(n)}$. For a user-specified significance level $\alpha \in (0, 1)$ (default $\alpha = 0.05$, specifying a $1 - \alpha = 95\%$ coverage guarantee), we compute the empirical quantile index:
$$k = \lceil (n + 1)(1 - \alpha) \rceil$$

The calibrated non-conformity threshold is the $k$-th order statistic:
$$\hat{q}_{1-\alpha} = S_{(k)}$$
If $k > n$, $\hat{q}_{1-\alpha}$ is set to $1.0$, guaranteeing conservative coverage.

---

### 4.2 Prediction Set Construction & Finite-Sample Coverage

For an unseen query $\mathbf{x}_{n+1}$, Reflex constructs the conformal prediction set:
$$\mathcal{C}_{1-\alpha}(\mathbf{x}_{n+1}) = \left\{ y \in \mathcal{Y} : 1 - \hat{P}(Y = y \mid \mathbf{x}_{n+1}) \le \hat{q}_{1-\alpha} \right\}$$

**Theorem 2 (Finite-Sample Conformal Coverage Guarantee)**.  
*Assume the calibration sequence $\mathcal{D}_{\text{cal}} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ and test point $(\mathbf{x}_{n+1}, Y_{n+1})$ are exchangeable random variables on $\mathcal{X} \times \mathcal{Y}$, and let $\hat{P}$ be fitted independently of $\mathcal{D}_{\text{cal}}$. Let $\alpha \in (0, 1)$ such that $\lceil (n+1)(1-\alpha) \rceil \le n$. Then, without any parametric or distributional assumptions on $P(X, Y)$:*
$$\mathbb{P}\left(Y_{n+1} \in \mathcal{C}_{1-\alpha}(\mathbf{x}_{n+1})\right) \ge 1 - \alpha$$
*Furthermore, if the scores $\{S_1, \dots, S_n, S_{n+1}\}$ are almost surely distinct (non-atomic distribution), the coverage satisfies the sharp two-sided finite-sample bound:*
$$1 - \alpha \le \mathbb{P}\left(Y_{n+1} \in \mathcal{C}_{1-\alpha}(\mathbf{x}_{n+1})\right) \le 1 - \alpha + \frac{1}{n+1}$$

---

*Proof of Theorem 2*.

Let $S_i = s(\mathbf{x}_i, y_i)$ for $i \in \{1, \dots, n\}$ and $S_{n+1} = s(\mathbf{x}_{n+1}, Y_{n+1})$.

1. **Exchangeability of Non-Conformity Scores**: Because $\hat{P}$ is trained independently of $\mathcal{D}_{\text{cal}} \cup \{(\mathbf{x}_{n+1}, Y_{n+1})\}$, the scoring mapping $(\mathbf{x}, y) \mapsto s(\mathbf{x}, y)$ is fixed and symmetric across all indices. Thus, exchangeability of $\{(\mathbf{x}_i, y_i)\}_{i=1}^{n+1}$ implies that the real-valued scores $S_1, \dots, S_n, S_{n+1}$ are exchangeable random variables.

2. **Distribution of Ranks**: Define the rank of the test score $S_{n+1}$ within the pooled sequence $\{S_1, \dots, S_n, S_{n+1}\}$:
   $$R_{n+1} = \sum_{i=1}^{n+1} \mathbf{1}_{\{S_i \le S_{n+1}\}}$$
   Assuming continuous non-atomic scores (ties occur with probability zero), all $(n+1)!$ orderings are equally likely by exchangeability. Thus $R_{n+1}$ is discrete uniform over $\{1, 2, \dots, n+1\}$:
   $$\mathbb{P}(R_{n+1} = r) = \frac{1}{n+1} \quad \forall r \in \{1, \dots, n+1\}$$

3. **Equivalence of Coverage and Rank Threshold**: By definition, $Y_{n+1} \in \mathcal{C}_{1-\alpha}(\mathbf{x}_{n+1}) \iff S_{n+1} \le \hat{q}_{1-\alpha} = S_{(k)}$.  
   This occurs if and only if at most $k-1$ calibration scores are strictly smaller than $S_{n+1}$, which is identically the event $\{R_{n+1} \le k\}$. Therefore:
   $$\mathbb{P}\left(Y_{n+1} \in \mathcal{C}_{1-\alpha}(\mathbf{x}_{n+1})\right) = \mathbb{P}(R_{n+1} \le k) = \sum_{r=1}^k \mathbb{P}(R_{n+1} = r) = \frac{k}{n+1} = \frac{\lceil (n+1)(1-\alpha) \rceil}{n+1}$$

4. **Derivation of Sharp Bounds**:  
   Using the ceiling inequality $\lceil x \rceil \ge x$:
   $$\frac{\lceil (n+1)(1-\alpha) \rceil}{n+1} \ge \frac{(n+1)(1-\alpha)}{n+1} = 1 - \alpha$$
   Using the strict ceiling bound $\lceil x \rceil < x + 1$:
   $$\frac{\lceil (n+1)(1-\alpha) \rceil}{n+1} < \frac{(n+1)(1-\alpha) + 1}{n+1} = 1 - \alpha + \frac{1}{n+1}$$
   Therefore:
   $$1 - \alpha \le \mathbb{P}\left(Y_{n+1} \in \mathcal{C}_{1-\alpha}(\mathbf{x}_{n+1})\right) \le 1 - \alpha + \frac{1}{n+1}$$
   When discrete scores yield ties, the conservative ranking policy counts tied scores in $R_{n+1}$, which can only increase probability mass at $S_{(k)}$; hence $\mathbb{P} \ge 1 - \alpha$ holds unconditionally. $\blacksquare$

---

### 4.3 Fail-Closed Operational Escalation Policy
Reflex enforces a deterministic, fail-closed operational policy:
1. **Pass Condition (Fast-Path Local Execution)**:  
   Execution proceeds locally on host silicon ($<1.0$ ms) if and only if:
   * **Singleton Prediction Set**: $|\mathcal{C}_{1-\alpha}(\mathbf{x}_{n+1})| = 1$ (exactly one candidate class satisfies the coverage bound).
   * **Margin Dominance**: $M(\mathbf{x}_{n+1}) = \hat{p}_{(1)} - \hat{p}_{(2)} \ge \tau$ (the top probability exceeds the runner-up by safety margin $\tau$, default $\tau = 0.15$).
2. **Halt Condition (Escalate to Frontier Deliberation)**:  
   Local execution halts and delegates the request to the System 2 governor (such as OpenAI Astra / GPT-6 [Sol, Terra, Luna], Anthropic Claude Opus 5 / Fable 5.1 / Mythos 5, Google Gemini 3.1 Pro & 3.8 Flash, or xAI Grok) whenever:
   * **Statistical Ambiguity**: $|\mathcal{C}_{1-\alpha}(\mathbf{x}_{n+1})| \ge 2$ (the model cannot statistically distinguish between two or more valid actions).
   * **OOD / Novel Stimulus**: $|\mathcal{C}_{1-\alpha}(\mathbf{x}_{n+1})| = 0$ (no known category satisfies the non-conformity threshold).
   * **Margin Defect**: $M(\mathbf{x}_{n+1}) < \tau$.

---

## 5. Cryptographic Receipts & Zero-Egress Invariants

### 5.1 Append-Only SQLite ActionLedger
Every decision evaluated by Reflex is recorded in an append-only, local SQLite database (`audit_trail.db`) operating in Write-Ahead Logging (`WAL`) mode with `PRAGMA synchronous = NORMAL`.

To guarantee tamper-evidence against post-hoc manipulation, records are chained via cryptographic SHA-256 rolling Merkle digests:

$$h_t = \text{SHA-256}\left( h_{t-1} \;\|\; t \;\|\; \text{canonical\_json}(\text{payload}_t) \right)$$

Where $h_0 = 0^{64}$ is the fixed genesis digest and $\text{canonical\_json}$ enforces RFC 8785 deterministic key ordering and zero whitespace. The entry payload incorporates the decision identifier, schema hash, prompt hash, prediction set cardinality, and timestamp. If an adversary modifies, inserts, or deletes a historical row $j < t$, the recurrence relation fails for all subsequent rows $k > j$, rendering any unauthorized tampering mathematically detectable.

### 5.2 Ed25519 Cryptographic Witness Receipts
Each Reflex deployment maintains an on-device Ed25519 asymmetric keypair generated and stored in `~/.system1/identity/`. Reflex executes pure software Ed25519 digital signatures via RFC 8032 standard primitives (using Python's `cryptography` library). Unlike hardware enclave architectures (e.g. Intel SGX or AMD SEV) that require specialized hypervisor attestation and virtualization overhead, Reflex provides non-repudiation and cryptographic chronological ordering at the application runtime layer. For every evaluated decision, Reflex produces a signed `RunWitnessEnvelope` containing:
* Canonical decision telemetry (action, parameters, calibrated probabilities, latency)
* Conformal gating status ($|\mathcal{C}_{1-\alpha}|$, empirical threshold $\hat{q}_{1-\alpha}$, margin $M(\mathbf{x})$)
* Rolling ledger head hash $h_t$
* Ed25519 digital signature generated via the Edwards-curve Digital Signature Algorithm (RFC 8032):

$$\sigma_t = \text{Ed25519\_Sign}\left( \text{private\_key}, \; h_t \;\|\; \text{timestamp\_ns} \right)$$

External auditors can independently verify the provenance, non-repudiation, and chronological ordering of agent actions using the public key `identity.pub` without access to runtime model memory or application secrets.

### 5.3 Architectural & In-Process Zero-Network-Egress Invariant
Reflex is engineered under a zero-external-dependency constraint. The core runtime imports exclusively from Python’s standard library and `numpy`. Crucially:
* No HTTP clients (`requests`, `httpx`, `urllib.request`, `aiohttp`) are packaged or imported within the System 1 execution path.
* No low-level networking sockets (`import socket`) or IPC telemetry daemons are instantiated during local inference.
* All matrix computations, conformal evaluations, and SQLite ledger writes execute strictly within host process memory and local disk.

**Clarification of Egress Paths:** 100% zero network egress applies strictly to local System 1 decisions. When conformal gating detects an ambiguous or out-of-distribution input, fallback routing to an upstream System 2 governor represents an intentional WAN egress path. If zero-egress mode is enforced (`zero_egress=True`), the runtime executes offline abstention (`ABSTAIN` / `REQUIRE_APPROVAL`) with zero external network packets transmitted, satisfying the rigorous data boundary requirements of HIPAA (Protected Health Information), GDPR (cross-border data sovereignty), and SOC 2 Type II compliance.

---

## 6. Boundary Conditions, Cardinality Limits, and Operational Failure Modes

While Reflex demonstrates sub-millisecond execution and distribution-free coverage under nominal operational envelopes, machine-native System 1 runtimes are subject to fundamental mathematical and spectral boundary conditions. In this section, we formalize the operational failure modes identified during adversarial stress audits—including hash capacity limits, high-cardinality margin collapse, recursive covariance asphyxiation, and composite schema poisoning—and prove the correctness of the architectural upgrades introduced to resolve them.

### 6.1 Zero-Shot Hash Projection Capacity & Dimensional Bounds
Reflex projects unstructured input strings $s$ into a compact embedding space $\mathbb{R}^D$ ($D = 384$ default, $D = 4,160$ in hybrid configurations) via MurmurHash3 feature hashing and deterministic character n-gram projections.

By the **Johnson-Lindenstrauss Lemma**, given $N$ discrete token representations and distortion tolerance $\epsilon \in (0, 1)$, pairwise Euclidean distances are preserved within $(1 \pm \epsilon)$ if the projection dimension satisfies:
$$D \ge \frac{8 \ln N}{\epsilon^2}$$
For $D = 384$ and $\epsilon = 0.25$, the theoretical capacity is bounded by $N \le \exp(384 \times 0.0625 / 8) = \exp(3.0) \approx 20$ mutually orthogonal dense clusters without distortion. When token dictionaries exceed this bound, pseudo-random feature collisions occur. Specifically, for $m$ distinct n-grams hashed into $B = 256$ buckets per subspace, the collision probability follows the classic birthday bound:
$$P(\text{collision}) \approx 1 - \exp\left(-\frac{m(m-1)}{2B}\right)$$
For $m \ge 20$ active n-grams, $P(\text{collision}) > 0.54$. Feature hashing mitigates collision-induced bias via Rademacher sign random variables $s(w) \in \{-1, +1\}$, yielding zero-mean expectation $\mathbb{E}[\mathbf{x}_i^T \mathbf{x}_j] = \mathbf{u}_i^T \mathbf{u}_j$ with variance:
$$\operatorname{Var}(\mathbf{x}_i^T \mathbf{x}_j) \le \frac{2}{D}$$
Consequently, zero-shot hash projection provides robust linear separability for low-to-medium cardinality tasks ($K \le 30$) but exhibits rising variance as cardinality $K$ or prompt length $L$ scales.

### 6.2 High-Cardinality Margin Collapse Proof & Relative Odds Ratio Dominance
In multi-class choice fields with $K$ candidate options, let the temperature-calibrated softmax probabilities be sorted in descending order:
$$p_{(1)} \ge p_{(2)} \ge \dots \ge p_{(K)}, \quad \sum_{k=1}^K p_{(k)} = 1.0$$
The uniform random baseline is $p_{\text{uniform}} = \frac{1}{K}$. In earlier runtime formulations, Lever 3 (Margin Gating) utilized a fixed absolute margin threshold $\tau_m$ (typically $\tau_m = 0.08$) to override conformal ambiguity:
$$\Delta = p_{(1)} - p_{(2)} \ge \tau_m \implies \text{override ambiguity halt}$$

**Theorem 2 (High-Cardinality Margin Collapse)**.  
*For high-cardinality action schemas with $K \gg 1/\tau_m$ (e.g., $K = 77$, such as MTEB Banking77, where $p_{\text{uniform}} \approx 0.013$), a fixed margin threshold $\tau_m$ permits pseudo-random hash dispersion to create false dominant margins, causing catastrophic precision collapse.*

*Proof*. Consider an adversarial or completely uninformative prompt where the true posterior distribution is approximately uniform across all $K$ options: $p_k \approx \frac{1}{K}$. Due to pseudo-random hash projections across $D = 384$, the pre-softmax logits $z_k \sim \mathcal{N}(0, \sigma^2)$ act as independent Gaussian random variables.  
By extreme value theory for i.i.d. Gaussians, the expected difference between the first and second order statistics is non-zero: $\mathbb{E}[z_{(1)} - z_{(2)}] = \frac{\sigma \sqrt{2 \ln K}}{K}$. Under temperature scaling $T$, the resulting softmax probabilities exhibit positive dispersion:
$$p_{(1)} \approx \frac{e^{z_{(1)}/T}}{\sum_k e^{z_k/T}} \approx 0.12, \quad p_{(2)} \approx 0.03 \implies \Delta = p_{(1)} - p_{(2)} = 0.09$$
Under a fixed threshold $\tau_m = 0.08$, the condition $\Delta = 0.09 \ge 0.08$ is satisfied, triggering the margin gate override. The engine accepts $p_{(1)} = 0.12$ as a decisive prediction, despite the fact that $1 - p_{(1)} = 0.88$ (88% of total probability mass) is uncommitted entropy. The conformal ambiguity halt is falsely bypassed, yielding silent misclassification. $\blacksquare$

**Architectural Solution (Dual Cardinality-Scaled Dominance Gate)**:  
To eliminate high-cardinality margin collapse, Reflex upgrades Lever 3 by establishing two mandatory invariant bounds that must hold simultaneously:
1. **Cardinality-Scaled Confidence Floor**: The winning probability must exceed the uniform baseline by an absolute offset $\tau_0$:
   $$p_{(1)} \ge \frac{1}{K} + \tau_0, \quad \text{with } \tau_0 = 0.15$$
   For $K = 77$, the required floor is $p_{(1)} \ge \frac{1}{77} + 0.15 \approx 0.163$. The dispersed pseudo-margin with $p_{(1)} = 0.12 < 0.163$ is rejected, preserving the conformal ambiguity halt.
2. **Relative Odds Ratio Dominance**: The ratio of the top choice to the runner-up must satisfy a multiplicative likelihood dominance bound:
   $$\mathcal{R} = \frac{p_{(1)}}{\max(10^{-6}, p_{(2)})} \ge \gamma, \quad \text{with } \gamma \ge 1.5$$
   This guarantees that even if $p_{(1)}$ is elevated, it cannot override ambiguity if the runner-up is nearly tied ($p_{(1)} \approx p_{(2)}$).

### 6.3 Covariance Asphyxiation in Online Sherman-Morrison Updates
Under Lever 2, Reflex performs recursive least-squares (RLS) online adaptation via the Sherman-Morrison rank-1 formula:
$$\mathbf{P}_{t+1} = \mathbf{P}_t - \frac{\mathbf{P}_t \mathbf{x}_{t+1} \mathbf{x}_{t+1}^T \mathbf{P}_t}{1 + \mathbf{x}_{t+1}^T \mathbf{P}_t \mathbf{x}_{t+1}}$$
where $\mathbf{P}_t = \mathbf{A}_t^{-1} \in \mathbb{R}^{(D+1) \times (D+1)}$.

**Theorem 3 (Covariance Asphyxiation Bound)**.  
*In unweighted recursive least squares ($\lambda_f = 1.0$), the spectral norm of the inverse covariance matrix decays as $\mathcal{O}(1/t)$. For $t > 500$ streaming updates under persistent excitation, the matrix update gain vanishes, rendering the model permanently non-adaptive.*

*Proof*. By definition, $\mathbf{A}_t = \lambda \mathbf{I} + \sum_{i=1}^t \mathbf{x}_i \mathbf{x}_i^T$. Assume persistent excitation such that the sample covariance satisfies $\mathbb{E}[\mathbf{x} \mathbf{x}^T] \succeq \sigma_{\min}^2 \mathbf{I}$ with $\sigma_{\min}^2 > 0$. By the strong law of large numbers:
$$\lim_{t \to \infty} \frac{1}{t} \mathbf{A}_t = \boldsymbol{\Sigma}_{\mathbf{x}} \succ \mathbf{0} \implies \lambda_{\min}(\mathbf{A}_t) \ge t \sigma_{\min}^2 + \lambda$$
Inverting $\mathbf{A}_t$ to obtain $\mathbf{P}_t = \mathbf{A}_t^{-1}$, the maximum eigenvalue satisfies:
$$\lambda_{\max}(\mathbf{P}_t) = \frac{1}{\lambda_{\min}(\mathbf{A}_t)} \le \frac{1}{t \sigma_{\min}^2 + \lambda} = \mathcal{O}\left(\frac{1}{t}\right)$$
The weight parameter update vector is $\Delta \mathbf{W}_{t+1} = (\mathbf{P}_{t+1} \mathbf{x}_{t+1}) (\mathbf{y}_{t+1}^T - \mathbf{x}_{t+1}^T \mathbf{W}_t)$. Its Frobenius norm is bounded by:
$$\|\Delta \mathbf{W}_{t+1}\|_F \le \|\mathbf{P}_{t+1}\|_2 \|\mathbf{x}_{t+1}\|_2 \|\mathbf{e}_{t+1}\|_2 \le \frac{\|\mathbf{x}_{t+1}\|_2 \|\mathbf{e}_{t+1}\|_2}{t \sigma_{\min}^2 + \lambda} \xrightarrow{t \to \infty} 0$$
For $t = 500$ and $\|\mathbf{x}\|_2 \approx 1$, $\|\mathbf{P}_t\|_2 < 2 \times 10^{-3}$. When the environment experiences non-stationary distribution shift or new System 2 exemplar corrections arrive, the parameter update step size is suffocated by $\mathbf{P}_t \to \mathbf{0}$. $\blacksquare$

**Architectural Solution (Exponential Forgetting Factor RLS)**:  
Reflex introduces a calibrated forgetting factor $\lambda_f \in (0, 1.0]$ (default $\lambda_f = 0.995$):
$$\mathbf{P}_{t+1} = \frac{1}{\lambda_f} \left[ \mathbf{P}_t - \frac{\mathbf{P}_t \mathbf{x}_{t+1} \mathbf{x}_{t+1}^T \mathbf{P}_t}{\lambda_f + \mathbf{x}_{t+1}^T \mathbf{P}_t \mathbf{x}_{t+1}} \right], \quad \mathbf{B}_{t+1} = \lambda_f \mathbf{B}_t + \mathbf{x}_{t+1} \mathbf{y}_{t+1}^T$$
Under $\lambda_f = 0.995$, the effective observation horizon is geometrically bounded by:
$$N_{\text{eff}} = \sum_{k=0}^\infty \lambda_f^k = \frac{1}{1 - \lambda_f} = \frac{1}{1 - 0.995} = 200 \text{ steps}$$
As $t \to \infty$, the inverse covariance converges to a non-zero steady-state limit:
$$\mathbf{P}_\infty = (1 - \lambda_f) \boldsymbol{\Sigma}_{\mathbf{x}}^{-1} \succ \mathbf{0}$$
The condition number $\kappa(\mathbf{P})$ remains strictly bounded under persistent excitation, preserving active learning sensitivity across millions of online decisions. To prevent asymmetric floating-point rounding divergence and eliminate covariance windup along unexcited subspace dimensions ($\lambda_{\max}(\mathbf{P}) \to \infty$), Reflex explicitly enforces three invariant safeguards:
1. **Hermitian Symmetrization**: $\mathbf{P}_{t+1} \leftarrow \frac{1}{2}\left(\mathbf{P}_{t+1} + \mathbf{P}_{t+1}^T\right)$.
2. **Regularized Covariance Bounding**: If $\max_i P_{ii} > \frac{50.0}{\lambda_{\text{reg}}}$, the runtime rescales $\mathbf{P}_{t+1} \leftarrow s \mathbf{P}_{t+1}$ and $\mathbf{B}_{t+1} \leftarrow s^{-1} \mathbf{B}_{t+1}$ where $s = \frac{50.0 / \lambda_{\text{reg}}}{\max_i P_{ii}}$, exactly preserving weight invariance $\mathbf{W}_{t+1} = (\mathbf{P}_{t+1} \mathbf{B}_{t+1})^T$ while bounding spectral condition $\kappa(\mathbf{P}) < 10^5$.
3. **Strict Positive-Definiteness**: $P_{ii} \leftarrow \max(P_{ii}, 10^{-6})$ preventing indefinite floating-point cancellation.

### 6.4 Recency Decay Inversion for Multi-Turn Agent Traces
Autonomous agent interaction logs consist of ordered token sequences $S = (w_0, w_1, \dots, w_{N-1})$. In standard bag-of-words or uniform token projection, all tokens contribute equally to the document vector $\mathbf{x} = \frac{1}{N} \sum_i \mathbf{v}(w_i)$.  
In multi-turn execution traces, earlier turns (system preamble, historical tool outputs) dominate the token count, diluting the directive in the latest user or tool message $w_{N-1}$.  
Reflex introduces **Recency-Aware Context Weighting**:
$$\text{weight}(i) = \frac{\log(1 + \text{len}(w_i))}{\sqrt{1.0 + 0.05 \cdot (N - 1 - i)}}$$
where $N - 1 - i$ is the backward distance from the trailing token.  
- For the final token ($i = N - 1$), the decay denominator is $\sqrt{1.0 + 0} = 1.0$ (full weight).
- For historical tokens $k$ steps prior, weight decays sub-linearly as $\mathcal{O}(1/\sqrt{k})$, preventing prefix dominance while preserving semantic anchoring.

### 6.5 Field-Level Escalation Granularity
In enterprise schemas comprising $M$ simultaneous output fields (e.g. `ActionRoute`, `ApprovalTier`, `SentimentAudit`), standard conformal gating evaluates each field independently:
$$|\mathcal{C}_{1-\alpha}^{(m)}| > 1 \implies \text{field } m \text{ is ambiguous}$$
In earlier implementations, any single ambiguous field triggered a global decision halt ($I_{\text{ambiguous}} = \bigvee_{m=1}^M I_{\text{ambiguous}}^{(m)}$).  
However, advisory or non-critical fields (such as auxiliary sentiment tags or optional reason codes) frequently experience natural semantic fuzziness without impacting the determinism of the primary control action (e.g., `ActionRoute = "TRANSFER"`). Forcing a full System 2 frontier escalation for an ambiguous advisory field introduces unnecessary latency and token expense—a failure mode termed **Advisory Field Poisoning**.

Reflex resolves this via field-level escalation granularity (`escalate_on_ambiguity: bool = True`):
$$I_{\text{ambiguous}}^{\text{global}} = \bigvee_{m=1}^M \left( I_{\text{ambiguous}}^{(m)} \land \text{field}_m.\texttt{escalate\_on\_ambiguity} \right)$$
Fields configured with `escalate_on_ambiguity=False` continue to emit calibrated prediction sets $\mathcal{C}_{1-\alpha}$ and log uncertainty into `DecisionResult.ambiguous_fields`, but do not trip $I_{\text{ambiguous}}^{\text{global}}$, isolating high-frequency control pathways from non-critical ambiguity.

---

## 7. Empirical Evaluation & Benchmarks

### 7.1 Latency Comparison Across Architectures
We benchmarked Reflex on an Apple M3 Max (14-core CPU, 36 GB Unified Memory) running macOS 15 and Linux Ubuntu 24.04 LTS against leading alternative decision paradigms across 10,000 independent trials.

| Runtime Architecture | Hardware Location | Execution Paradigm | P50 Latency | P99 Latency | Relative Speedup | WAN Egress |
|---|---|---|---|---|---|---|
| **Reflex Tier 0 (L1 Cache)** | Host Memory | In-Process Exact Hash | **0.0098 ms (9.8 µs)** | **0.014 ms** | **86,700×** | **0 Bytes** |
| **Reflex System 1 (Metal)** | Host Metal/BLAS | Non-Autoregressive Matrix | **0.98 ms** | **1.34 ms** | **867×** | **0 Bytes** |
| Local 8B LLM (vLLM / Ollama) | Local GPU (RTX 4090) | Autoregressive (KV Cache) | 180.00 ms | 245.00 ms | 4.7× | 0 Bytes |
| Cloud Fast API (Groq / Cerebras) | US-East WAN | Autoregressive Specialized ASIC | 220.00 ms | 410.00 ms | 3.9× | Full Payload |
| Frontier Deliberative Governor (OpenAI Astra/GPT-6, Claude Opus 5, Gemini 3.1 Pro, xAI Grok) | Cloud WAN | Autoregressive Frontier Deliberation | 850.00 ms | 1,480.00 ms | 1.0× (Baseline) | Full Payload |
| ReAct Multi-Turn Cloud Agent Loop | Cloud WAN | Multi-Call Tool Reasoning Loop | 3,200.00 ms | 6,800.00 ms | 0.26× | Full Payload |

Reflex delivers an **867× latency improvement** over single-call frontier cloud reasoning models for cold forward passes on metal, and an **86,700× improvement** over cloud LLMs on repeated L1 cache hits (a 100× speedup over cold forward passes), eliminating WAN transit jitter completely.

---

### 7.2 60 FPS Real-Time Game Boy Control (Pokémon Red/Blue)
To test Reflex under unforgiving real-time constraints, we interfaced the runtime directly with the `PyBoy` Game Boy hardware emulator running *Pokémon Red*. Game Boy hardware ticks at exactly 60.0 Hz (16.6 ms per frame).

```
Hardware Frame Budget:  |================| (16.6 ms)
Cloud LLM Latency:      |======================================================...| (850 ms -> 51 frame drops)
Reflex System 1 Pass:   |=| (0.98 ms -> consumes 5.9% of budget; up to 16 evaluations per frame)
```

* **Observation Window**: Reflex extracted battle RAM vectors directly from `$D057` (Battle Mode flag `wIsInBattle`), `$D015` (Player HP `wBattleMonHP`), `$CFE6` (Enemy HP `wEnemyMonHP`), and combat move selection tables.
* **Throughput**: In headless benchmark mode, Reflex sustained **10,400+ FPS** (measuring up to 13,096 FPS on hardware), evaluating 38 µs forward passes per state transition.
* **Combat Decision**: Navigated the combat decision graph (FIGHT menu $\to$ TACKLE selection), executing policy-prescribed controller actions and processing HP bar depleting animations at native 60 FPS with zero frame stutter.

---

### 7.3 Autonomous Apprentice-to-Metal Cutover Engine
We evaluated `TypeSafeClient(mode="auto_cutover")` within an autonomous agent tool pipeline processing live streaming requests:
* **Phase 1: Shadow Apprentice (Queries 1–50)**: Transparently proxied requests to the cloud SaaS API baseline (mean latency: 220 ms) while asynchronously recording prompt-action pairs to `ActionLedger`.
* **Phase 2: Closed-Form Cutover (Query 51)**: `ReflexCompiler` solved closed-form Ridge Regression in 14.2 ms, calibrated conformal prediction bounds at $\alpha = 0.05$, verified 100% agreement on validation exemplars, and executed an atomic pointer swap to 100% local metal execution.
* **Post-Cutover Performance**: Decision latency immediately collapsed from 220 ms to 0.98 ms. With an empirical System 2 escalation rate of only 1.6%, external WAN data egress dropped by **98.4%** across production traffic, while localized decisions achieved **100% egress elimination (0 bytes)**.

---

### 7.4 Concurrency & Ledger Integrity
Under a multi-threaded stress test simulating 8 concurrent worker threads executing an enterprise security firewall schema (48 attack vectors including prompt injection, data exfiltration, and unauthorized shell execution):
* Sustained throughput exceeded **1,240 QPS** (reaching 2,281 QPS on in-line tool gating).
* Zero SQLite lock contention occurred under WAL journal mode.
* Post-run audit verified 100% unbroken SHA-256 chain integrity across all records.

---

## 8. Related Work

1. **Speculative Decoding & Draft Models**: Leviathan et al. (2023) introduced speculative decoding using small autoregressive draft models to propose token sequences verified in parallel by larger models. While effective for prose generation, speculative decoding remains fundamentally autoregressive and cannot achieve the sub-millisecond execution envelope required for real-time discrete state decisions.
2. **Dual-Process Cognitive AI**: Booch et al. (2021) and Bengio (2017) formalized the conceptual integration of fast intuitive heuristics (System 1) with deliberate symbolic or neural reasoning (System 2). In late-2026 architectures, frontier models (OpenAI Astra & GPT-6 series, Anthropic Claude Opus 5 / Fable 5.1 / Mythos 5, Google Gemini 3.1 Pro & 3.8 Flash, and xAI Grok) represent powerful System 2 governors; Reflex provides the first machine-native, non-autoregressive System 1 runtime engineered to interface directly with these governors.
3. **Conformal Risk Control & Uncertainty Quantification**: Vovk et al. (2005) and Angelopoulos & Bates (2021) developed distribution-free conformal prediction frameworks guaranteeing finite-sample error coverage. Reflex operationalizes conformal prediction for discrete agentic control, replacing uncalibrated softmax heuristics with provable fail-closed escalation gates.
4. **Hardware-Attested Computing & Ledger Transparency**: Prior verifiable computing frameworks rely on confidential hardware enclaves (e.g., Intel SGX, AMD SEV) with significant virtualization overhead. Reflex establishes cryptographic non-repudiation at the application runtime layer through append-only SQLite Merkle chains and Ed25519 witness receipts (Bernstein et al., 2012).

---

## 9. Conclusion

Reflex addresses the acute latency, economic, and data privacy bottlenecks of contemporary agentic AI. By decoupling high-frequency System 1 reflex actions from deliberate System 2 reasoning, Reflex enables autonomous agents to operate at machine-native speeds (0.98 ms P50 latency, 60 FPS hardware budgets) while enforcing rigorous mathematical safety guarantees through Split Conformal Prediction and Ed25519 cryptographic audit receipts. Reflex is fully open source under the Apache 2.0 license at `https://github.com/steph4n-gh/reflex`.

---

## References

1. Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
2. Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511.
3. Sherman, J., & Morrison, W. J. (1950). *Adjustment of an Inverse Matrix Corresponding to a Change in One of the Elements*. Annals of Mathematical Statistics, 21(1), 124–127.
4. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer Science & Business Media.
5. Bernstein, D. J., Duif, N., Lange, T., Schwabe, P., & Yang, B. Y. (2012). *High-speed high-security signatures*. Journal of Cryptographic Engineering, 2(2), 77–89.
6. Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Large Language Models via Speculative Decoding*. International Conference on Machine Learning (ICML).
7. Booch, G., Fabiano, F., Horesh, L., et al. (2021). *Thinking Fast and Slow in AI*. Proceedings of the AAAI Conference on Human Computation and Crowdsourcing.
8. Weinberger, K., Dasgupta, A., Langford, J., Smola, A., & Attenberg, J. (2009). *Feature Hashing for Large Scale Multitask Learning*. International Conference on Machine Learning (ICML).
9. steph4n. (2026). *Reflex: Non-Autoregressive System 1 Decision Runtime with Conformal Ambiguity Gating and Cryptographic Witness Receipts*. Reflex Core Research Team. `https://github.com/steph4n-gh/reflex`.
