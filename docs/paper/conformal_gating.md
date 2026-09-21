# Mathematical notes: uncertainty, prototype geometry and online correction

**System 1 1.0.3 + current source changes · 21 September 2026 · Implementation notes, not new theorems**

This note follows the source checkout, including the
[unreleased optional teaching method](../../CHANGELOG.md#unreleased).

This note describes the mathematics used by the runtime and the assumptions
needed to interpret it. The [whitepaper](system1_whitepaper.md) contains the
product evaluation; the [architecture reference](../architecture/technical_specification.md)
maps behavior to code. Earlier claims of guaranteed safe execution, universal
margin improvement and fixed microsecond update times are withdrawn.

## 1. Split-conformal prediction sets

Fix a categorical scoring model before evaluating the conformal calibration fold.
Let its probabilities be $p_y(x)$ and let $n$ calibration pairs and the next test
pair be exchangeable. Examples-only skills use the LAC nonconformity score

$$s(x,y)=1-p_y(x).$$

Sort calibration scores as $s_{(1)}\le\cdots\le s_{(n)}$ and set

$$k=\lceil(n+1)(1-\alpha)\rceil,\qquad
q=\begin{cases}s_{(k)}&k\le n,\\1&k>n.\end{cases}$$

Because LAC scores lie in $[0,1]$, the second case includes every label. With no
calibration evidence, strict mode also retains all labels. For a calibrated skill,

$$C(x)=\{y:s(x,y)\le q\}.$$

Under the stated assumptions,

$$\Pr\{Y_{n+1}\in C(X_{n+1})\}\ge1-\alpha.$$

For distinct scores, the test score's pooled rank is uniform on $1,\ldots,n+1$.
When $k\le n$, inclusion has probability $k/(n+1)\ge1-\alpha$; when $k>n$ it is
certain. Ties handled by including scores equal to the threshold are conservative.
This is the established split-conformal argument, not a new System 1 result.
See [Angelopoulos and Bates, §2](https://arxiv.org/html/2107.07511v6#S2).

### What the guarantee does not say

Let $A$ be the event that the application accepts a local prediction. Marginal
set coverage does not imply

$$\Pr\{\hat Y=Y\mid A\}\ge1-\alpha.$$

For example, set misses can be concentrated in a small accepted subset while
other inputs receive broad sets. The coverage statement also does not establish
accuracy for each class, simultaneous coverage across fields, adversarial
robustness or safe tool execution. Exchangeability can fail under traffic drift,
related templates or adaptive collection. An empty set requests review; it does
not by itself detect distribution shift.

## 2. Score definitions and release behavior

[ConformalPredictor](../../src/system1/calibration.py) supports LAC and a legacy,
deterministic APS score. APS sorts labels by descending probability and sums mass
through the candidate label. Calibration and prediction use the same score and
ordering. Strict inversion includes labels whose score is at most the stored
quantile; it does not add the next label after crossing that threshold.

New examples-only categorical/Boolean skills use LAC. Legacy and schema-augmented
skills use APS. Format-v2 artifacts record this choice, while v1 files retain
legacy APS semantics. Both score functions lie in $[0,1]$ for valid normalized
probabilities; insufficient evidence conservatively includes all labels.

The current source computes p-value tail counts using left insertion into sorted
calibration scores. Equal scores remain included, preserving the prior inclusive
tail-count definition for both LAC and APS.

The compiler separates temperature fitting from conformal scoring so that the
probability transformation is not fitted on the same labels used to claim set
coverage. Explicit calibration must also be separate from head fitting and final
evaluation. Repeated normalized prompts do not create new independent evidence.

Strict categorical decisions require a singleton set for acceptance. Non-strict
mode can suppress review using margin, a cardinality-dependent confidence floor
and a top-to-runner-up ratio; these are heuristics and do not inherit the strict
coverage interpretation. They do not apply as a universal safety test to all
field types. MultiChoice uses a complete assignment score and continuous scores
use residual intervals. Field-level `escalate_on_ambiguity` determines which
uncertain fields affect the overall review flag.

The published public-data experiments use strict mode and report **acceptance**,
**correctness among accepted decisions** and **raw correctness** separately.
The optional `PromotionPolicy(min_accepted_agreement=.95)` added in 1.0.2 also
tests agreement conditional on an accepted validation group. Its point estimate
and exact lower bound must pass, with no accepted evidence causing deferral.
This adds a third exact bound to the error budget: at attempt `k`, each spends
`(1 - confidence) / (3 * k * (k + 1))`. The default two-bound budget is unchanged.
Teacher agreement is distinct from correctness against independent labels; the
[quality round](../../benchmarks/quality/quality_round/README.md) reports both
qualification and subsequent diagnostic failures.

Default automatic promotion tests 80% agreement and 80% acceptance; that is a
separate policy from alpha 0.05 or a 95% accepted-correctness workload target.
The [SMS observation result](../../benchmarks/quality/workloads/README.md)
promotes under the former and misses the latter.

## 3. Prototype centering: a conditional geometry result

The schema-seeded model can center class prototypes and normalize the residuals.
This is centering, not covariance whitening. It is separate from fitting ridge
heads to representative labeled examples. The following identity describes a
special geometry; it is not a guarantee about arbitrary text features.

Let $w_j=b+v_j$ for $K\ge2$, with mutually orthogonal $v_j$ of equal nonzero norm
$c$ in dimension $D\ge K$. Subtract $\mu=K^{-1}\sum_jw_j$ and write
$w'_j=v_j-K^{-1}\sum_mv_m$. Then

$$\langle w'_j,w'_k\rangle=c^2(\delta_{jk}-1/K),\qquad
\|w'_j\|^2=c^2(K-1)/K.$$

Normalizing therefore gives

$$\langle\tilde w_j,\tilde w_k\rangle=-\frac1{K-1}\quad(j\ne k).$$

This reaches the simplex bound for the maximum signed pairwise inner product:
for any $K$ unit vectors,

$$0\le\left\|\sum_ju_j\right\|^2
=K+\sum_{j\ne k}\langle u_j,u_k\rangle$$

implies $\max_{j\ne k}\langle u_j,u_k\rangle\ge-1/(K-1)$.
The bound is attained by a regular simplex when dimension permits. It is not the
Welch bound on absolute correlation. For two distinct prototypes, centering makes
them antipodal; coincident prototypes have zero residual and cannot be normalized
by this formula. Real correlated prototypes need not become a regular simplex,
and greater separation at prototypes does not prove better test accuracy.

## 4. Ridge fitting and optional online correction

For augmented features $Z=[X,\mathbf1]$, targets $Y$, and the compiler's diagonal
regularizer $R$, the fitted coefficients satisfy

$$\Theta=(Z^TZ+R)^{-1}Z^TY.$$

The implementation computes the inverse with NumPy, retaining it when needed for
online correction; it has a pseudoinverse/least-squares fallback. It does not ship
the Cholesky solver described by older drafts. Bias regularization and sample
weights must be included when comparing against a batch formula.

For ideal arithmetic, an invertible matrix $A$ with inverse $P$, and a new feature
vector $z$, Sherman–Morrison gives

$$P'=(A+zz^T)^{-1}=P-\frac{Pzz^TP}{1+z^TPz},$$

provided the denominator is nonzero. Updating $B'=B+zy^T$ and setting
$\Theta'=P'B'$ yields the unweighted batch solution for the augmented data.
This follows directly by multiplying $A+zz^T$ by the proposed inverse.

The optional runtime update also supports forgetting $f\in(0,1]$ (default 0.995):

$$P'=\frac1f\left(P-\frac{Pzz^TP}{f+z^TPz}\right),\qquad
B'=fB+zy^T.$$

This describes exponentially weighted evidence, not the original unweighted
batch objective when $f<1$. Finite precision and runtime stabilization further
limit exact equivalence. A positive diagonal alone does not prove positive
definiteness; rescaling a matrix alone does not improve its condition number.
No fixed latency, unlimited stability or immunity to forgetting is implied.

Most users can retain corrected examples and recompile. If online correction is
used, changed weights invalidate their old calibration, in memory and saved
artifacts. Strict inference requests review until separate recalibration. A
teacher label is fallible supervision, so one correction should not be described
as making every unseen related case correct.

The source compiler's optional logistic choice fit instead minimizes summed
cross-entropy plus $\lambda\|W\|_F^2/2$, with an unpenalized bias, using SciPy
L-BFGS-B at teaching time. Stored coefficients are scaled by 0.25 to match the
runtime's choice-logit scaling. NumPy inference and separate calibration are
unchanged. The ridge batch-equivalence formulas above do not describe incremental
logistic fitting: retain corrections and recompile to preserve that objective.

## 5. Reading the evidence

The [three-workload report](../../benchmarks/quality/workloads/README.md) records
measured quality, errors, timings, baseline behavior and failed takeovers.
It does not experimentally prove exchangeability or the conditions of the
prototype identity. Deterministic tool permissions and signed audit records are
separate software mechanisms; see [deployment boundaries](../deployment.md).
The [n8n gauntlet](../../benchmarks/quality/n8n_gauntlet/README.md) is a separate
selective-classification experiment with learned review policies and failed
quality gates. Its development scores are not conformal guarantees or independent
confirmation of accepted correctness or unfamiliar-input rejection.
