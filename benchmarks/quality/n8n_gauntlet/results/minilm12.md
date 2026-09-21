# A larger fixed encoder does not pass the development screen

Experiment 2m was [declared](../MINILM12_PROTOCOL.md) and pushed in
`0d2e4bea291c743ca2c6d4c5585fb7b3b12b07d6` before feature extraction or fitting.
It replaces the input encoder with pinned 12-layer MiniLM while keeping the
same 384-dimensional System1 head, fitting labels, calibration and numerical
settings. **Neither workload passes the fixed advancement rule.** All current
complete-adapter selections and the [full regression failure](latest-regression.md)
remain unchanged.

| Full development workload | Matching control: raw correct | MiniLM12: raw correct | Change | Best confidence/margin coverage: control → new |
|---|---:|---:|---:|---:|
| CLINC150 supported requests | 2,864/2,995 (95.63%) | 2,884/2,995 (96.29%) | +20 | 78.33% → 70.18% |
| BANKING77 | 1,787/1,960 (91.17%) | 1,772/1,960 (90.41%) | −15 | 62.60% → 68.88% |

These optimistic frontiers choose a threshold on development labels at >=99%
accepted supported accuracy and <=1% CLINC unfamiliar false acceptance. Neither
reaches 80%. CLINC improves raw correctness but loses coverage; banking improves
the frontier but loses raw correctness. The prospectively fixed screen requires
both a raw gain of at least ten and a frontier gain, plus saved-head p95 <5 ms.
Neither candidate advances to another review implementation under that rule.

Strict System1 review alone accepts 2,876/2,995 supported CLINC requests with
2,815 correct (97.88%), and falsely accepts 16/100 unfamiliar requests. Banking
accepts 1,872/1,960 with 1,733 correct (92.57%). Those policies fail quality and
are not replacements for the existing learned review guards.

## Timing and teaching

All 5,055 development requests run once through actual text encoding and the
loaded saved head, after 100 untimed warmups per workload. OS networking is
denied; response caching and receipts are disabled. There are no errors, new
teacher calls or new API costs.

| Saved-head measurement | CLINC | Banking |
|---|---:|---:|
| Requests measured | 3,095 | 1,960 |
| p50 / p95 | 1.923 / 2.694 ms | 2.180 / 4.389 ms |
| Encoder load | 166.30 ms | 128.90 ms |
| Saved head / engine load | 253.16 ms | 156.01 ms |
| Numerical head fit | 1.82 s | 1.47 s |
| Fitting / calibration feature preparation | 21.09 / 5.29 s | 16.02 / 4.39 s |

**These are saved-head timings, not complete-adapter latency.** They include
encoding, numerical prediction and strict decision construction, but omit
learned review, request validation and HTTP/n8n transport. They cannot establish
the full sub-5-ms gate or a controlled speed comparison with older candidates.
Process imports and source/network preflight take a separately recorded
1,424.34 ms. All four teaching-feature caches were initially absent; preparation
uses individual-request encoding. Downloads and initial model installation are
not included in teaching times.

CLINC retains its 12,019 fitting rows; banking uses 6,026 original fitting rows
and the earlier 924 synthetic supported lessons. The synthetic labels remain
unverified. No new teaching source, original test or human reserve is scored.
The matching controls and conventional baselines are retained without new fits
or timings, with the stronger CLINC baseline's additional 1,696 lessons disclosed.

## Artifacts and verification

The [full report](minilm12-development.json) retains every outcome and timing,
strict sets, frontiers, Wilson intervals, source identities and environment.
The [asset manifest](../minilm12-assets.json) pins the downloaded subset to
`sentence-transformers/all-MiniLM-L12-v2` revision
`a50ef00143b4d5391434df20ae11632588ac25be`. Hugging Face checksum verification
checked all six downloaded remote files; omitted alternate weights and local
download metadata were reported separately. No encoder weights are committed.
The upstream pretraining corpus has not been independently checked for
benchmark overlap.

The exact heads and manifests are published under `artifacts/minilm12-clinc150`
and `artifacts/minilm12-banking77`, totaling 236,280 and 126,184 bytes. Each needs
the separate 34,584,885-byte encoder/tokenizer. Manifest hashes:

- CLINC: `00a488b7fd4de1a03b367a5347418c885857b990c45a1aef7fc9ac8a6689b89a`.
- Banking: `d16d752db4cbfd1c58e04eaf2cba3ff9847b88ec2db1fa55f155f1a3519f2728`.

The [record audit](minilm12-audit.json) recomputes every outcome's aggregate
metrics and timing, verifies all 147 previously frozen files, fourteen declared
sources, pinned encoder files and both saved heads. It runs no inference.
Sixteen additional text replays per workload reproduce suggestions, prediction
sets, strict eligibility and confidence exactly; no timing is claimed from those
replays. No core API, encoder weights or numerical-head format changed.

This comparison does not resolve quality, supply independent full-scope
confirmation or qualify the n8n recording. PR #3 remains open; no release.
