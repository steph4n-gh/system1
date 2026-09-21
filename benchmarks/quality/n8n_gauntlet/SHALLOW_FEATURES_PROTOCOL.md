# Taught nonlinear features: experiment 2l

Commit this procedure, implementation and authored tests before fitting. The
completed fused review loses coverage despite a raw CLINC accuracy gain, and
word fusion hurts banking. The next bounded hypothesis is that a small
supervised nonlinear transform can improve distinctions that the existing
linear head misses. This is an experiment, not a core product change.

## Fixed comparison and teaching

Use the exact original CLINC fitting/calibration/development splits and the
original banking splits plus the earlier 924 supported Gemini fitting lessons.
Those synthetic labels remain unverified. No 2e/2i lesson, new teacher request,
original test, human reserve or new source enters this comparison. Keep all
150 CLINC intents plus native OOS and all 77 banking intents.

The matching semantic controls are the saved experiment-2j controls in
`results/fused-head-development.json`; bind their full report and artifact
identities and verify their original split hashes. They use exactly these
teaching labels, individual-request embeddings, encoders and numerical head
settings. Reuse their recorded outcomes; do not refit or retime unchanged
controls. Retain the stronger incumbent and conventional baseline identities,
including the stronger CLINC baseline's additional 1,696 supported lessons.

Fit one new candidate for each workload:

- Keep the pinned 384-dimensional encoder fixed: MiniLM with one thread for
  CLINC; BGE-small with four threads for banking. All vectors use single-request
  projection; feature caches may accelerate teaching only.
- Teach an `MLPClassifier` on fitting vectors/labels only: one 128-unit ReLU
  hidden layer, Adam, learning rate .001, batch size 128, L2 alpha .01,
  random_state 20260921, shuffle enabled, exactly 100 epochs. Set
  early_stopping=False, tol=0 and n_iter_no_change=101. All other parameters
  remain the installed scikit-learn defaults, recorded in the output. This is
  a fixed update budget, not a convergence claim. Preserve warnings and the
  entire training-loss curve; do not select epochs on calibration/development.
- Export the first layer's weights and bias as float32. Apply ReLU, normalize
  its 128 activations, concatenate with the unchanged unit semantic vector,
  and normalize the combined 512-vector. A zero hidden vector retains the
  semantic vector. This adds one small matrix operation after one encoder pass.
- Discard the MLP's output layer for inference; retain its parameters as
  teaching provenance. Compile the existing System1 logistic numerical head
  on these 512-vectors, with regularization .1 / .01 for CLINC / banking.
  Original calibration data calibrate the resulting head only; strict alpha
  stays .05 / .075. There is no encoder fine-tuning or generation.

The fitting API and numerical parameters follow the
[MLPClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).
This is a proposed task-specific transform, not a claim that generic embedding
whitening helps. Research has [reported classification degradation from whitening](https://arxiv.org/abs/2407.12886);
[LatentGate section 6.2](https://aclanthology.org/2026.acl-industry.153.pdf) also
reports degradation for MiniLM on CLINC. Neither study establishes the outcome
of the supervised transform proposed here.

## Records, advancement and limits

Save the existing-format intent head, transform NPZ and a manifest binding the
encoder, transform, contract/settings and sources. Reconstruct the projector
and saved head; replay the first 16 development requests per workload through
actual text projection. Suggestions, strict eligibility and prediction sets
must match, with maximum confidence difference below 1e-5. Verify the exported
hidden operation against the fitted network on 32 fitting vectors.

Record every development outcome and the same raw accuracy, OOS suggestions,
strict-only metrics and confidence/margin frontiers used in 2j. Frontier targets
remain >=99% accepted supported accuracy and <=1% CLINC unfamiliar acceptance.
All 5,055 new outcomes, loss curves, warnings, fit/preparation times, saved bytes,
source hashes, environments and zero API calls/cost must survive failure.
The run denies OS networking. Outputs are exclusive; do not rerun to rescue
quality. Single-request saved replay verifies serialization, not latency.

Advance a workload to a separately declared full review/runtime comparison only
if it gains at least ten raw supported-correct requests over its matching control
AND strictly improves the better of that control's confidence/margin coverage
frontiers (without strict eligibility). A raw gain alone was insufficient in 2k.
This advancement rule is not a qualification target or candidate adoption.
Any later review cross-fitting must fit the entire nonlinear transform inside
each fitting fold, excluding held-out rows and all generated prompt lineage.

No new dependency, core API or saved-head format is added. No combined adapter
latency is claimed here. The 147-file freeze, all prior failures and both full
workloads remain authoritative. Qualification still requires >=80% supported
coverage, >=99% accepted accuracy, <=1% CLINC unfamiliar false acceptance and
complete p95 <5 ms, then independent full-scope confirmation and the exact saved
skill in the n8n teacher-disconnection recording. Keep PR #3 open; no release.

After committing the exact procedure and implementation:

```sh
sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python benchmarks/quality/n8n_gauntlet/shallow_head_probe.py --folder .system1/n8n-gauntlet/shallow-head-development
```
