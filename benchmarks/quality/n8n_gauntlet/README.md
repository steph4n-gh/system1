# n8n decision takeover experiment

**In development; not qualified for takeover.** The goal is to teach a bounded
classification skill, independently qualify it, connect that exact saved skill
to self-hosted n8n, and record the teacher being disconnected. This directory
does not yet contain a passing final evaluation or an integration recording.

The [protocol](PROTOCOL.md) and [source/split manifest](manifest.json) were
committed in `803650c` before any official-test scoring. All 150 CLINC intents
and all 77 BANKING77 intents remain in scope. Required targets are 80% local
acceptance, 99% correctness among accepted requests, at most 1% CLINC out-of-scope
false acceptance, and complete local-adapter p95 below 5 ms. None has been lowered.

## What development has established

The original small lexical classifier does not meet this broader task. Simply
raising its confidence threshold loses most local coverage. Conventional
TF-IDF/logistic regression also misses the targets. A fixed local encoder helps;
cross-entropy fitting is substantially better than ridge on those features.
This motivated the optional Python compiler's `choice_solver="logistic"` method.
It needs SciPy during teaching and retains the existing saved-head format and
NumPy inference. Ridge remains the default. See the
[teaching guide](../../../docs/guides/training_experts.md).

Both **development** candidates now clear the targets through the complete local
adapter, including the saved head, strict System1 review, fixed encoder and an
additional learned/distribution guard:

| Development workload | Supported accepted | Correct among accepted | OOS falsely accepted | Complete p95 |
|---|---:|---:|---:|---:|
| CLINC, fixed MiniLM, one encoder thread | 2,512/2,995 (83.9%) | 2,495/2,512 (99.32%) | 1/100 | 1.84 ms |
| BANKING77, fixed BGE-small, four encoder threads | 1,575/1,960 (80.4%) | 1,560/1,575 (99.05%) | No native OOS cohort | 4.38 ms |

Banking's margin is narrow. Its original reliability check reached 77.6% coverage;
adding 924 generated contrast examples reached 78.9%. Three-fold predictions on
original fitting rows then supplied independent correctness examples for the
review model. Each fold excluded its own rows and all generated examples, whose
prompt lineage could otherwise leak the withheld examples. The review model was
fitted on those correctness examples plus the reserved calibration fold. Its
regularization and strict-review settings were selected on development data.
The one- and two-thread banking runs failed latency (6.23 and 5.36 ms p95); their
results remain published. No encoder weights were updated for these candidates.
The exploratory encoder-adaptation run reaches 1,530/1,960 accepted (78.1%) with
1,515/1,530 correct (99.0%). That adaptation updates pretrained weights; it is
research evidence, not a requirement for ordinary System1 teaching or an adopted
integration path. The main route continues with a fixed encoder and taught head.

These are development results: thresholds/configurations were
selected using development outcomes. They are **not** held-out qualifications,
statistical guarantees or claims that a standard System1 install achieves these
numbers. The official test splits have not been scored. The next step is to freeze
the complete candidate and baseline, then run that independent test once.

The [real n8n development workflow](N8N.md) already uses the exact saved CLINC
candidate. It preserves item identity and explicit review, including when the
local service is stopped. This is integration evidence, not the final qualified
teacher-disconnection recording.

Retained results:

- [Initial feasibility failure](results/initial-feasibility.json), with its
  [original probe](results/initial-feasibility.py).
- [Lexical heads and conventional baselines](results/lexical-development.json).
- Fixed [MiniLM](results/minilm-development.json) and
  [BGE-small](results/bge-small-development.json) encoder probes.
- [Combined lexical/encoder features](results/hybrid-development.json).
- [Distance rejection with the actual System1 logistic compiler](results/gating-development.json).
- [Nonlinear feature probe](results/kernel-development.json),
  [nearby-example agreement](results/agreement-development.json), and
  [calibration-taught reliability](results/reliability-development.json).
- [MPNet-base](results/mpnet-base-development.json): the larger encoder does not
  solve banking and its encoder-only p95 exceeds 5 ms on both workloads.
- Exploratory encoder adaptation on [banking](results/adapted-banking77.json) and
  [CLINC](results/adapted-clinc150.json), including every measured epoch.
- [Teacher connectivity](results/teacher-connectivity.json): one actual Jev and
  one actual Gemini request. These are connectivity checks, **not** teacher
  distillation or measured n8n executions.
- [Generated contrast lessons](results/contrast-lessons.json),
  [complete teacher requests/responses](results/contrast-teacher.json),
  [usage and cost estimate](results/contrast-teacher-summary.json), and
  [matched development comparisons](results/contrast-development.json).
- [Request-length reliability follow-up](results/contrast-length-development.json):
  adding a word-count feature did not close the quality gap.
- [Out-of-fold reliability](results/crossfit-development.json) and its
  [regularization follow-up](results/crossfit-refined-development.json).
- [Linear/prototype combination](results/prototype-development.json): retained
  failure, not adopted by the runtime candidates.
- Complete saved-adapter runs for [CLINC](results/runtime-development.json) and
  banking with [one](results/bank-runtime-development-1.json),
  [two](results/bank-runtime-development-2.json) and
  [four](results/bank-runtime-development-4.json) encoder threads.
- OS network-blocked replay of all [3,095 CLINC](results/runtime-isolation.json)
  and [1,960 banking](results/bank-runtime-isolation.json) development decisions.

Original examples come from the pinned fitting, calibration or development
folds; the additional generated lessons are identified separately. Fitted feature vectors may be cached to avoid repeated preprocessing in
research runs; recorded single-input encoder timings execute the encoder again.
There is no test-response cache. Head-fit timings exclude initial feature
extraction. Exploratory encoder timings exclude the System1 engine and adapter,
and some ran alongside other development processes; they cannot prove the latency
gate. Final timings will follow the protocol on an otherwise idle machine.

## Reproduce development

From a checkout with System1 installed:

```bash
python benchmarks/quality/n8n_gauntlet/prepare.py
python benchmarks/quality/n8n_gauntlet/develop.py
```

The lexical comparison additionally uses scikit-learn and threadpoolctl. For the
optional encoder experiments, install ONNX Runtime and tokenizers in a research
environment, and obtain the pinned files with the Hugging Face `hf` CLI:

```bash
hf download sentence-transformers/all-MiniLM-L6-v2 tokenizer.json tokenizer_config.json config.json README.md 1_Pooling/config.json onnx/model_qint8_arm64.onnx --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --local-dir .system1/n8n-gauntlet/minilm
hf download BAAI/bge-small-en-v1.5 tokenizer.json tokenizer_config.json config.json README.md 1_Pooling/config.json onnx/model.onnx --revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a --local-dir .system1/n8n-gauntlet/bge-small
python benchmarks/quality/n8n_gauntlet/encoder_probe.py minilm
python benchmarks/quality/n8n_gauntlet/encoder_probe.py bge-small
python benchmarks/quality/n8n_gauntlet/hybrid_probe.py
python benchmarks/quality/n8n_gauntlet/gating_probe.py
python benchmarks/quality/n8n_gauntlet/kernel_probe.py
python benchmarks/quality/n8n_gauntlet/agreement_probe.py
python benchmarks/quality/n8n_gauntlet/reliability_probe.py
python benchmarks/quality/n8n_gauntlet/runtime_probe.py
```

`encoder_probe.py --extended` retains a separate report for additional C=100/1000
head fits. All these runners open only the three development folds. `prepare.py`
uses test text solely to exclude normalized duplicates and preserves every
official test row.

MiniLM's ONNX file is 23,026,053 bytes; BGE-small's is 133,093,490 bytes, before
tokenizers, runtime libraries and taught heads. These externally pretrained
encoders are optional research dependencies, not System1's built-in hashed
features. They run locally without text generation. Their pretraining corpora
have not been independently audited for benchmark overlap. Upstream model cards:
[MiniLM (Apache-2.0)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
[BGE-small (MIT)](https://huggingface.co/BAAI/bge-small-en-v1.5).

The adaptation probes additionally used PyTorch 2.14.0 and Transformers 4.57.6,
with the matching pinned MiniLM `model.safetensors` downloaded into the same
directory. Commands are `python benchmarks/quality/n8n_gauntlet/adapt_encoder.py
banking77` and the corresponding `clinc150` run. They use only fitting labels for
gradient updates; calibration remains separate. These slower experiments must
not be described as instant teaching or as proof that language-model fine-tuning
is required. No model files or credentials are committed.

## Recorded contrast teaching

The [contrast-example procedure](TEACHING_ADDENDUM.md) was committed in `679470a`
before its first API call. Gemini generated 924 lessons in 77 requests, using only
fitting examples and category names. No generated lesson overlapped a reserved
normalized group. This is synthetic augmentation, not blind teacher classification
or a passing result. For example, its disposable-card lessons explicitly mention
temporary or single-use numbers, while ordinary virtual-card lessons ask how to
obtain a virtual card. Those labels come from generation instructions, not a human
audit of all generated examples.

Generation took 180.3 seconds and used 25,644 input and 17,510 output tokens. The
published standard-price estimate is $0.05147 using Gemini 2.5 Flash's $0.30 per
million input and $2.50 per million output tokens, checked September 21, 2026.
Actual account tier and billed cost are unavailable. See
[Google's pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-2.5-flash).

```bash
# Replay published teacher responses; no API credential or new calls needed.
python benchmarks/quality/n8n_gauntlet/teach_contrasts.py
python benchmarks/quality/n8n_gauntlet/contrast_probe.py
python benchmarks/quality/n8n_gauntlet/contrast_probe.py --length-feature
python benchmarks/quality/n8n_gauntlet/crossfit_probe.py
python benchmarks/quality/n8n_gauntlet/crossfit_probe.py --refine
python benchmarks/quality/n8n_gauntlet/prototype_probe.py
python benchmarks/quality/n8n_gauntlet/bank_runtime_probe.py --threads 4
```

The first comparison wrote all eight configurations and then aborted during native
library shutdown; its [complete result and failure](results/contrast-development-first.json)
are preserved. Releasing encoder sessions before threadpool teardown produced a
clean run with exactly identical quality results. Offline replay also reproduced
the lesson file byte-for-byte while socket connections were blocked. These checks
do not replace the required final saved-skill isolation and n8n workflow tests.
