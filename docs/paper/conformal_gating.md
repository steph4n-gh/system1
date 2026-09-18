# Conformal Ambiguity Gating and Contrastive Simplex Separation for Machine-Native Decision Runtimes

**Research Track: Reliable Machine Learning & Uncertainty Quantification**  
**Working Paper / Workshop Manuscript**

---

## Abstract

Autoregressive transformer architectures incur substantial latency, high computational costs, and inference jitter when deployed for high-frequency discrete decisions such as tool policy enforcement, input classification, and action triage. While non-autoregressive linear and kernel projection methods offer sub-millisecond execution directly on host hardware, uncalibrated linear models are prone to catastrophic misclassification under covariate shift. 

In this work, we formalize a machine-native decision framework combining three theoretical contributions:
1. **Contrastive Whitening & Simplex Separation**: We prove that subtracting candidate centroid prototypes from shared background representations prior to spherical projection achieves the Welch and Rankin lower bound for maximum pairwise cosine similarity ($-\frac{1}{K-1}$ for $K$ classes), maximizing linear decision margins.
2. **Split Conformal Ambiguity Gating**: We establish finite-sample coverage guarantees ($\mathbb{P}(Y_{n+1} \in \mathcal{C}_{1-\alpha}(\mathbf{x}_{n+1})) \ge 1 - \alpha$) over discrete candidate sets, replacing heuristic thresholding with provable fail-closed escalation triggers when the prediction set cardinality $|\mathcal{C}_{1-\alpha}| \ne 1$ or the dominance margin $M(\mathbf{x}) < \tau$.
3. **Sub-50µs Online Rank-1 Distillation**: We demonstrate closed-form parameter adaptation via the Sherman-Morrison formula under an exponential forgetting factor ($\lambda_f$), allowing the runtime to assimilate deliberative feedback instantaneously without backpropagation or catastrophic forgetting.

Empirical evaluation demonstrates sub-millisecond execution ($<0.98$ ms P50 on CPU/Metal) while bounding misclassification risk to specified confidence tolerances $1-\alpha$.

---

## 1. Introduction

High-frequency decision agents—including robotic controllers, microservice API gateways, and autonomous tool firewalls—must evaluate discrete policy actions within strict latency ceilings (often $<10$ ms or bounded hardware frame budgets such as 16.6 ms for 60 Hz loops). Monolithic autoregressive Large Language Models (LLMs) evaluate token sequences sequentially:

$$\tau_{\text{total}} = \tau_{\text{WAN}} + \tau_{\text{prefill}} + \sum_{k=1}^K \tau_{\text{decode}}(t_k)$$

Even with low parameter counts or accelerated hardware, autoregressive generation introduces non-deterministic jitter and network latency when hosted remotely.

Conversely, small non-autoregressive linear models can execute in microseconds. However, deploying classical linear classifiers in autonomous control loops introduces two fatal vulnerabilities:
1. **Severe Angular Compression**: When candidate class descriptors share extensive contextual phrasing, standard embedding projections collapse candidate hyperplanes into an acute cone, degrading discrimination.
2. **Uncalibrated Confidence**: Standard softmax outputs yield overconfident probabilities under distributional drift, offering no formal safety guarantees against false positives.

We address these challenges through an integrated mathematical framework: contrastive simplex separation, split conformal ambiguity gating, and closed-form online adaptation.

---

## 2. Representation Geometry: Contrastive Whitening & Simplex Separation

### 2.1 Hybrid Feature Projection

Let an input string $s$ be mapped into a high-dimensional space via a combination of unbiased feature hashing and dense semantic subword embeddings:

$$\mathbf{x}_{\text{sparse}}[h_i(w)] = \sum_{w \in s} \text{sign}(h_i^*(w)) \cdot \text{IDF}(w), \quad \mathbf{x}_{\text{dense}} = \frac{1}{|T(s)|} \sum_{t \in T(s)} \mathbf{E}_{t, :}$$

The unified input vector $\mathbf{x} \in \mathbb{S}^{D-1}$ is constructed on the unit hypersphere:

$$\mathbf{x} = \sqrt{\alpha} \frac{\mathbf{x}_{\text{sparse}}}{\|\mathbf{x}_{\text{sparse}}\|_2} \oplus \sqrt{1-\alpha} \frac{\mathbf{x}_{\text{dense}}}{\|\mathbf{x}_{\text{dense}}\|_2}$$

This formulation guarantees $\|\mathbf{x}\|_2 = 1.0$ without post-hoc normalization.

### 2.2 Angular Compression & Centroid Removal

Consider $K$ candidate choice prototypes $\mathbf{w}_1, \dots, \mathbf{w}_K \in \mathbb{R}^D$ ($K \ge 2$, $D \ge K-1$). In structured schemas, each prototype is composed of an invariant background vector $\mathbf{b} \in \mathbb{R}^D$ and a class-distinctive feature vector $\mathbf{v}_k \in \mathbb{R}^D$:

$$\mathbf{w}_k = \mathbf{b} + \mathbf{v}_k, \quad k \in \{1, \dots, K\}$$

When the background magnitude dominates the distinctive component ($\|\mathbf{b}\|_2 = B \gg \|\mathbf{v}_k\|_2 = c$), uncentered prototypes exhibit acute pairwise angular compression:

$$\cos \theta_{jk} = \frac{\langle \mathbf{w}_j, \mathbf{w}_k \rangle}{\|\mathbf{w}_j\|_2 \|\mathbf{w}_k\|_2} \approx \frac{B^2}{B^2 + c^2} \longrightarrow 1 \quad \text{as } B/c \to \infty$$

To eliminate this degeneracy, we apply **Contrastive Centering**:
1. Compute the empirical centroid:
   $$\boldsymbol{\mu} = \frac{1}{K} \sum_{k=1}^K \mathbf{w}_k$$
2. Subtract the centroid to isolate distinctive features:
   $$\mathbf{w}_k' = \mathbf{w}_k - \boldsymbol{\mu}$$
3. Project onto the unit hypersphere:
   $$\tilde{\mathbf{w}}_k = \frac{\mathbf{w}_k'}{\|\mathbf{w}_k'\|_2}$$

---

### 2.3 Theoretical Optimality

**Theorem 1 (Simplex Equiangular Separation & Welch Bound Optimality)**.  
*Let candidate vectors $\mathbf{w}_1, \dots, \mathbf{w}_K \in \mathbb{R}^D$ ($K \ge 2$, $D \ge K-1$) satisfy $\mathbf{w}_k = \mathbf{b} + \mathbf{v}_k$, where $\mathbf{b} \in \mathbb{R}^D$ is an arbitrary shared background vector and $\mathbf{v}_1, \dots, \mathbf{v}_K \in \mathbb{R}^D$ are mutually orthogonal with equal norm: $\langle \mathbf{v}_j, \mathbf{v}_k \rangle = c^2 \delta_{jk}$ for $c > 0$.*

*Let $\boldsymbol{\mu} = \frac{1}{K}\sum_{k=1}^K \mathbf{w}_k$, $\mathbf{w}_k' = \mathbf{w}_k - \boldsymbol{\mu}$, and $\tilde{\mathbf{w}}_k = \mathbf{w}_k' / \|\mathbf{w}_k'\|_2$. Then:*

1. **Equiangular Pairwise Separation:**
   $$\langle \tilde{\mathbf{w}}_j, \tilde{\mathbf{w}}_k \rangle = -\frac{1}{K - 1} \quad \forall j \ne k$$

2. **Centroid Annihilation:**
   $$\sum_{k=1}^K \tilde{\mathbf{w}}_k = \mathbf{0}$$

3. **Welch Bound Minimality:**  
   *The inner product $-\frac{1}{K-1}$ strictly attains the Welch and Rankin lower bound for the maximum pairwise cosine similarity of any $K$ unit vectors in $\mathbb{R}^D$:*
   $$\min_{\mathbf{u}_1, \dots, \mathbf{u}_K \in \mathbb{S}^{D-1}} \max_{j \ne k} \langle \mathbf{u}_j, \mathbf{u}_k \rangle = -\frac{1}{K-1}$$

*Proof*.  
The centroid is $\boldsymbol{\mu} = \mathbf{b} + \frac{1}{K}\sum_{m=1}^K \mathbf{v}_m$.  
The centered vector for index $k$ is:
$$\mathbf{w}_k' = (\mathbf{b} + \mathbf{v}_k) - \left(\mathbf{b} + \frac{1}{K}\sum_{m=1}^K \mathbf{v}_m\right) = \mathbf{v}_k - \frac{1}{K}\sum_{m=1}^K \mathbf{v}_m$$
Compute the inner product between centered vectors $j$ and $k$ ($j \ne k$):
$$\langle \mathbf{w}_j', \mathbf{w}_k' \rangle = \left\langle \mathbf{v}_j - \frac{1}{K}\sum_{m=1}^K \mathbf{v}_m, \; \mathbf{v}_k - \frac{1}{K}\sum_{m=1}^K \mathbf{v}_m \right\rangle = -\frac{c^2}{K}$$
Similarly, compute the squared norm for any $k$:
$$\|\mathbf{w}_k'\|_2^2 = c^2 \left(1 - \frac{1}{K}\right) = c^2 \left(\frac{K-1}{K}\right)$$
Therefore, the normalized inner product is:
$$\langle \tilde{\mathbf{w}}_j, \tilde{\mathbf{w}}_k \rangle = \frac{-c^2 / K}{c^2 (K-1) / K} = -\frac{1}{K-1}$$
By Welch (1974), for any $K$ vectors in $\mathbb{R}^D$ with $D \ge K-1$, the maximum inner product satisfies $\max_{j \ne k} \langle \mathbf{u}_j, \mathbf{u}_k \rangle \ge -1/(K-1)$, with equality if and only if the vectors form a regular $(K-1)$-simplex centered at the origin. Thus contrastive centering achieves the theoretical maximum angular separation. $\blacksquare$

---

## 3. Finite-Sample Split Conformal Prediction

To ensure statistical safety in autonomous decision loops, we deploy **Split Conformal Prediction** (Vovk et al., 2005; Angelopoulos & Bates, 2021).

Let $\{(X_i, Y_i)\}_{i=1}^n$ be an exchangeable calibration dataset drawn from distribution $\mathcal{D}$. Given a target error rate $\alpha \in (0, 1)$, we construct a prediction set $\mathcal{C}_{1-\alpha}(X_{n+1})$ satisfying:

$$\mathbb{P}(Y_{n+1} \in \mathcal{C}_{1-\alpha}(X_{n+1})) \ge 1 - \alpha$$

### 3.1 Non-Conformity Scoring

Let $\hat{\pi}_y(\mathbf{x}) = \sigma(\mathbf{W} \mathbf{x})_y$ be the model's calibrated probability for candidate label $y \in \mathcal{Y}$. We define the non-conformity score as the residual deficit:

$$s(\mathbf{x}, y) = 1 - \hat{\pi}_y(\mathbf{x})$$

For each calibration instance $i \in \{1, \dots, n\}$, compute the non-conformity score of the ground-truth label:

$$s_i = 1 - \hat{\pi}_{Y_i}(X_i)$$

The conformal quantile $\hat{q}$ is the $\lceil (n+1)(1-\alpha) \rceil / n$ empirical quantile of $\{s_1, \dots, s_n\}$:

$$\hat{q} = \text{Quantile}\left( \{s_1, \dots, s_n\}, \; \frac{\lceil (n+1)(1-\alpha) \rceil}{n} \right)$$

At test time, the prediction set is constructed as:

$$\mathcal{C}_{1-\alpha}(\mathbf{x}) = \{y \in \mathcal{Y} : \hat{\pi}_y(\mathbf{x}) \ge 1 - \hat{q}\}$$

### 3.2 Dual-Criterion Gating Mechanism

A prediction set $\mathcal{C}_{1-\alpha}(\mathbf{x})$ conveys two distinct failure modes:
1. **Ambiguity / Under-Confidence ($|\mathcal{C}| > 1$):** Multiple competing options satisfy the coverage bound.
2. **Out-of-Distribution Rejection ($|\mathcal{C}| = 0$):** No candidate option meets the threshold $1 - \hat{q}$.

To prevent false executions while minimizing unnecessary escalations, we couple the conformal set size with a **Dominance Margin**:

$$M(\mathbf{x}) = \hat{\pi}_{(1)}(\mathbf{x}) - \hat{\pi}_{(2)}(\mathbf{x})$$

Where $\hat{\pi}_{(1)}$ and $\hat{\pi}_{(2)}$ denote the highest and second-highest class probabilities. The decision is authorized locally if and only if:

$$I_{\text{safe}}(\mathbf{x}) = \left(|\mathcal{C}_{1-\alpha}(\mathbf{x})| == 1\right) \land \left(M(\mathbf{x}) \ge \tau_{\text{margin}}\right)$$

If $I_{\text{safe}}(\mathbf{x}) = 0$, the system halts fail-closed and escalates to a deliberative governor (or human supervisor).

---

## 4. Sub-50µs Online Rank-1 Adaptation via Sherman-Morrison

When an ambiguous decision is escalated and resolved by an external authority (providing ground-truth label $\mathbf{y}^*$), the model must assimilate this information without triggering expensive batch retraining.

### 4.1 Covariance Formulation

In regularized Ridge Regression, the optimal weight matrix satisfies:

$$\mathbf{W}^* = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{Y} = \mathbf{P} \mathbf{B}$$

Where $\mathbf{P} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \in \mathbb{R}^{d \times d}$ is the inverse regularized Gram matrix and $\mathbf{B} = \mathbf{X}^T \mathbf{Y} \in \mathbb{R}^{d \times K}$ is the cross-covariance accumulator.

### 4.2 Rank-1 Update Rule

When receiving a new exemplar $(\mathbf{x}_{\text{aug}}, \mathbf{y}^*)$, where $\mathbf{x}_{\text{aug}} = [\mathbf{x}^T, 1]^T$, we apply an exponential forgetting factor $\lambda_f \in (0, 1]$ (default $\lambda_f = 0.995$) to prevent covariance explosion and bound historical drift:

$$\mathbf{P}_{t+1} = \frac{1}{\lambda_f} \left( \mathbf{P}_t - \frac{\mathbf{P}_t \mathbf{x}_{\text{aug}} \mathbf{x}_{\text{aug}}^T \mathbf{P}_t}{\lambda_f + \mathbf{x}_{\text{aug}}^T \mathbf{P}_t \mathbf{x}_{\text{aug}}} \right)$$

$$\mathbf{B}_{t+1} = \lambda_f \mathbf{B}_t + \mathbf{x}_{\text{aug}} (\mathbf{y}^*)^T$$

$$\mathbf{W}_{t+1} = (\mathbf{P}_{t+1} \mathbf{B}_{t+1})^T$$

Because $\mathbf{x}_{\text{aug}} \in \mathbb{R}^{d}$, the rank-1 update requires only matrix-vector products $\mathbf{u} = \mathbf{P}_t \mathbf{x}_{\text{aug}}$, an inner product $\mathbf{x}_{\text{aug}}^T \mathbf{u}$, and an outer product $\mathbf{u} \mathbf{u}^T$. On standard CPU SIMD / Metal hardware ($d \le 512$), this update completes in **$38.4$ µs** ($<50$ µs SLA), enabling real-time boundary rotation in live interactive environments.

---

## 5. Empirical Results

We evaluated the framework on standard discrete routing and security classification tasks across 10,000 trials on Apple Silicon Metal and Linux BLAS.

### 5.1 Coverage Calibration

| Target Coverage ($1-\alpha$) | Empirical Coverage | Mean Set Size $|\mathcal{C}|$ | Escalation Rate ($|\mathcal{C}| \ne 1$) |
|---|---|---|---|
| 90.0% ($\alpha = 0.10$) | **90.4%** | 1.04 | 3.8% |
| 95.0% ($\alpha = 0.05$) | **95.2%** | 1.09 | 7.1% |
| 99.0% ($\alpha = 0.01$) | **99.1%** | 1.21 | 14.6% |

Empirical coverage strictly satisfies the theoretical lower bound $\ge 1-\alpha$, confirming validity under finite-sample calibration splits.

### 5.2 Latency Comparison

| Step | Execution Mode | Measured P50 | Measured P99 |
|---|---|---|---|
| L1 Exact Hash Hit | In-Memory Hash Table | 0.0098 ms (9.8 µs) | 0.014 ms |
| Forward Pass | BLAS / Metal GEMM | 0.620 ms | 0.980 ms |
| Conformal Gating | Scalar Threshold Check | 0.004 ms | 0.008 ms |
| Online Adaptation | Sherman-Morrison Rank-1 | 0.038 ms (38.4 µs) | 0.048 ms |

---

## 6. Conclusion

By combining contrastive whitening (achieving the Welch optimal simplex separation), finite-sample split conformal prediction sets, and microsecond-grade Sherman-Morrison covariance updates, this framework demonstrates that autonomous agents can resolve high-frequency discrete decisions with mathematical guarantees, zero external WAN egress, and sub-millisecond execution times.

---

## References

1. Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511.
2. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer Science & Business Media.
3. Welch, L. R. (1974). *Lower bounds on the maximum cross correlation of signals*. IEEE Transactions on Information Theory, 20(3), 397–399.
4. Rankin, R. A. (1955). *The closest packing of spherical caps in $n$ dimensions*. Proceedings of the Glasgow Mathematical Association, 2(3), 139–144.
5. Sherman, J., & Morrison, W. J. (1950). *Adjustment of an Inverse Matrix Corresponding to a Change in One of the Elements*. Annals of Mathematical Statistics, 21(1), 124–127.
6. Weinberger, K., Dasgupta, A., Langford, J., Smola, A., & Attenberg, J. (2009). *Feature Hashing for Large Scale Multitask Learning*. International Conference on Machine Learning (ICML).
