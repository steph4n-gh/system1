# Fixed 12-layer encoder: experiment 2m

Commit this procedure, pinned asset manifest and implementation before any
feature extraction, head fitting or model scoring. Experiments 2j–2l did not
improve the selected complete adapter through word or nonlinear feature
additions. This comparison tests a different fixed input representation while
retaining the existing 384-dimensional System1 numerical head and teaching API.

## One new representation, two full workloads

Use [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/blob/a50ef00143b4d5391434df20ae11632588ac25be/README.md),
revision `a50ef00143b4d5391434df20ae11632588ac25be`, Apache-2.0, with the
published ARM64 int8 ONNX file. The upstream card describes a 384-dimensional
sentence encoder with mean pooling and truncation at 256 word pieces. Pin all
six downloaded file hashes and sizes in `minilm12-assets.json`; verify before
loading. The deployed encoder/tokenizer total 34,584,885 bytes, counted separately
from System1. No encoder weights are updated, and the upstream pretraining
corpus has not been independently audited for benchmark overlap.

Use exactly two CPU encoder threads, one inter-op thread, mean pooling,
L2-normalization and 256-token truncation for both workloads. No thread, pooling,
precision, truncation, regularization or model-selection grid. Reuse the existing
encoder operation with a separate pinned settings entry; do not edit the 147
previously frozen files. No new dependency, core API or model format.

Preserve the original all-150-intent CLINC fitting/calibration/development
splits, including native OOS; banking retains all 77 intents and its original
splits plus the earlier 924 supported Gemini fitting lessons. Those synthetic
labels remain unverified. No new source, 2e/2i lesson, teacher call, original
test or reserved human request is used. The matching semantic controls and
conventional baseline outcomes from 2j remain the comparison. Their data labels
are identical; stronger incumbents remain, including the CLINC baseline with
1,696 additional supported lessons.

Extract every fitting/calibration vector using individual-request projection.
Teaching feature caches may be reused only with the exact pinned encoder
identity. Compile one System1 logistic choice head per workload with unchanged
regularization .1 / .01 and original calibration, strict alpha .05 / .075.
Save each head and the manifest binding its encoder and contract/settings.
Do not add the rejected word or nonlinear transform.

## Saved-head measurement and advancement

Load the exact saved head with the real encoder, response caching disabled.
After 100 untimed development warmups, run each entire development cohort once
through actual text encoding and the strict System1 engine, with receipts
disabled and OS networking denied. Time encoding, numerical prediction and
strict decision construction. This is **saved-head timing**, not a complete
adapter: the learned review, request validation and n8n/HTTP transport are
absent. It cannot prove the full latency target.

Retain all 5,055 outcomes, per-request timing, raw accuracy, OOS suggestions,
strict-only metrics, confidence/margin frontiers with and without strict
eligibility, Wilson intervals, source hashes, process import/preflight time,
encoder/head load time, fitting/preparation costs and artifact sizes. Preserve
failures and zero teacher calls/new API cost. Checkpoint completed measurements
before comparing; never remeasure for better numbers. Replay the first sixteen
requests per workload without making a new timing claim; require identical
suggestions, sets and strict eligibility, with confidence difference <1e-5.

Advance a workload to a separately declared complete review/adapter comparison
only if it gains at least ten raw supported-correct requests over its matching
2j semantic control, improves the better confidence/margin frontier coverage
at >=99% accepted supported accuracy / <=1% CLINC OOS false acceptance, has no
saved-head errors or replay mismatches, and saved-head p95 is below 5 ms.
These are development screening criteria, not qualification or adoption.
Any later review must cross-fit original-only examples with this exact encoder
operation and exclude generated prompt lineage from fold models/prototypes.

The full goal remains >=80% supported coverage, >=99% accepted accuracy,
<=1% CLINC unfamiliar false acceptance and complete saved-adapter p95 <5 ms,
independent full-scope confirmation, the exact qualified artifact in n8n and
the qualified teacher-disconnection recording. All failures and both complete
workloads remain required. PR #3 stays open; no release.

Download the pinned subset (the published results never include encoder weights):

```sh
hf download sentence-transformers/all-MiniLM-L12-v2 tokenizer.json tokenizer_config.json config.json README.md 1_Pooling/config.json onnx/model_qint8_arm64.onnx --revision a50ef00143b4d5391434df20ae11632588ac25be --local-dir .system1/n8n-gauntlet/minilm12
```

After committing the exact procedure, from the repository root:

```sh
sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python benchmarks/quality/n8n_gauntlet/minilm12_probe.py --folder .system1/n8n-gauntlet/minilm12-development
```

Outputs are exclusive. Reproduction needs a new directory and must not be
presented as fresh independent evidence on these already-used development rows.
