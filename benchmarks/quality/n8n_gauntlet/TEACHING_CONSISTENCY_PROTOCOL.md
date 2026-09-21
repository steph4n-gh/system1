# Diagnose teaching-label consistency: experiment 2o

Commit this procedure and implementation before running the diagnostic. The
preceding banking ablation leaves fourteen accepted System1 mistakes shared
between original-only and augmented heads. Some development labels appear
inconsistent with their utterances, but our interpretation is not independent
ground truth. MTEB's published cleaning filters do not document human relabeling.
No annotation will be changed or excluded on the basis of model agreement.

## Fixed fitting-fold diagnostic

Audit all 12,019 original CLINC fitting rows (including 80 native OOS rows) and
all 6,026 original banking fitting rows. Reuse `polynomial_review.prepare` for
both the semantic System1 classifier and conventional TF-IDF/logistic method.
The fixed original per-label/hash three folds, .1/.01 numerical regularization,
MiniLM/BGE encoders, original calibration and lexical C=10 remain unchanged.
Generated lessons never fit these temporary fold heads or their prototypes.
The helper also reconstructs already observed parent calibration/development
signals; these do not supply new teaching labels or a selected replacement.

Retain each fitting example's existing label, both out-of-fold suggestions and
their maximum probabilities, reconstructed from the helper's log-odds feature.
Report all disagreements, consensus disagreements and supported/OOS counts.
Use fixed probability bins [0, .5, .9, .95, .99, 1] for descriptive summaries.
Order consensus disagreements by semantic confidence, then lexical confidence,
then normalized group hash; show at most fifty per dataset. This ordering is a
review aid, not a confidence guarantee or automatic label-cleaning rule.

For those examples, show three nearest semantic neighbors from the two fitting
folds that trained their temporary classifiers. Also show the nearest example
with the assigned label and the nearest with the consensus suggestion. Exclude
the entire held-out fold, not just the example itself. Use the same fixed
individual-request encoder features. Similarity is context for inspection;
it does not prove that any annotation is incorrect.

Separately inspect every already-recorded accepted error of each current
System1 development incumbent: CLINC's context-review candidate and banking's
quadratic-review candidate. Show its nearest original fitting examples under
the same feature definition. Retain unfamiliar false accepts as errors. This
is post hoc development diagnosis, not a newly scored or relabeled benchmark.

## Evidence and boundaries

Verify all 147 frozen files and committed diagnostic sources. Run under OS
network denial with no teacher calls or new API cost. Record the exact original
fitting/calibration/development hashes, fold groups, all 18,045 out-of-fold
outcomes, descriptive summaries, neighbor groups and similarities, source
candidate identities, preparation time and environment. No inference-speed
claim is made from diagnostic timings. Output is exclusive; any failure is
retained rather than overwritten.

No threshold is tuned, label changed, row removed, encoder updated or new final
classifier selected. The purpose is to decide whether the next quality change
has evidence behind it and to identify examples needing independent review.
Original tests and human reserves stay unopened and unscored. The full 150+OOS
CLINC and 77-intent banking gauntlet, its failures and original targets remain.
This diagnostic cannot qualify either workload or the n8n recording. PR #3
stays open; no release.

After committing, from the repository root:

```sh
sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python benchmarks/quality/n8n_gauntlet/teaching_consistency.py --output .system1/n8n-gauntlet/teaching-consistency.json
```
