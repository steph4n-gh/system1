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

The strongest current **development** result for CLINC combines the fixed MiniLM
encoder, a System1 logistic head and a distance check against the taught intent
distribution: 2,519/2,995 supported requests accepted (84.1%), 2,502/2,519 correct
(99.3%), and 1/100 out-of-scope requests wrongly accepted. Banking still misses:
the fixed BGE-small encoder with a calibration-taught reliability check reaches
1,521/1,960 accepted (77.6%), with 1,506/1,521 correct (99.01%). It still makes
24 errors among 1,568 accepted requests when coverage is raised to 80%.
The exploratory encoder-adaptation run reaches 1,530/1,960 accepted (78.1%) with
1,515/1,530 correct (99.0%). That adaptation updates pretrained weights; it is
research evidence, not a requirement for ordinary System1 teaching or an adopted
integration path. The main route continues with a fixed encoder and taught head.

These are optimistic development frontiers: thresholds/configurations were
selected using development outcomes. They are **not** held-out qualifications,
statistical guarantees or claims that a standard System1 install achieves these
numbers. The official test splits have not been scored. Full inference timing
and transfer still need to pass before selecting a final candidate.

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

All development examples come from the pinned fitting, calibration or development
folds. Fitted feature vectors may be cached to avoid repeated preprocessing in
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

The next declared experiment is [contrast-example teaching](TEACHING_ADDENDUM.md):
a bounded, recorded Gemini run generates clearer lessons from fitting examples
only. Its procedure is committed before calling the API. This is synthetic
augmentation, not blind teacher classification or a passing result.
