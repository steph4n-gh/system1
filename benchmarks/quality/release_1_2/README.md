# 1.2.0 research and lifecycle evidence

The bounded research candidates did not earn a new classification or lesson
selection capability. A historical BANKING77 replay successfully exercised
adoption, rejected faulty lessons, restored original labels, reopening and exact
rollback. The new UCI corpus failed development and source-shift gates; it did
not earn adoption or independent qualification.

The complete compact report is [SUMMARY.json](SUMMARY.json). Existing full-scope
assistant/banking benchmark targets are unchanged.

## Prospective protocol and confirmation

[PROTOCOL.md](PROTOCOL.md) was committed at `04adb2d` before scoring. The
larger lifecycle fitting budget was amended prospectively at `f22ad13`; sources,
hashes and group-separated role IDs were committed at `88d68a5`. All model,
projection, source and protocol hashes were frozen at `69ccf18` before the first
confirmation prediction. The 1,986-row Emotion test cohort and 996-row UCI Yelp
cohort were scored once. There was no tuning against confirmation.

An earlier standalone development attempt accidentally left the compiled
model's cache enabled even though the engine cache was disabled. It violated
the uncached protocol and was invalidated before confirmation. The corrected
runner disables both switches; the earlier outputs and failure cases remain
identified as invalid evidence. [Correction record](INVALID_CACHE.md).

## Lesson selection and classification

The public Emotion corpus was new to this project. After specified duplicate
grouping, its source roles contain 15,917 acquisition/calibration rows, 1,987
development rows and 1,986 untouched confirmation rows. Seeds 11, 29 and 47
share identical initial labels within each paired comparison. Acquisition
algorithms use inputs and assessed predictions; a regression check confirms
that changing hidden pool labels cannot change their suggestions.

Each simulated oracle budget includes **all fitting and calibration labels**:
100 = 60+40, 200 = 160+40, 400 = 360+40. No class-balancing search or discarded
inspection receives free labels. Development and confirmation annotations are
separate measurement costs. Automated preparation read the source annotations
to audit conflicting duplicates; 20,000 public annotations were available
before acquisition. No human labeling time was measured. These budgets measure
model-access label allocation within a simulation, not total project annotation
effort or independently reviewed customer labeling.

The following are three-seed means at 400 model-access labels. Correct automatic
yield is correct accepted decisions divided by all evaluated rows, including
reviewed rows in the denominator.

| Candidate | Development raw accuracy | Development correct automatic yield | Confirmation raw accuracy | Confirmation correct automatic yield | Confirmation coverage |
|---|---:|---:|---:|---:|---:|
| TF-IDF ridge, random | 41.74% | 0.94% | 41.04% | 0.99% | 1.48% |
| TF-IDF ridge, uncertainty | 41.12% | 0.96% | 40.38% | 0.79% | 1.04% |
| TF-IDF ridge, uncertainty + diversity | 40.76% | 0.79% | 40.99% | 0.50% | 0.74% |
| Portable TF-IDF logistic, random | 41.45% | 0.02% | 41.62% | 0.08% | 0.12% |
| Frozen MiniLM, random | 50.09% | 0.62% | 52.15% | 0.49% | 0.54% |
| MiniLM + supervised contrastive projection | 53.33% | 0.18% | 52.85% | 0.13% | 0.13% |

Every candidate missed the exploratory .90 accepted accuracy / .20 coverage /
.10 accepted-error upper-bound screen. That screen is already weaker than
deployment qualification. Neither selector earned the required two percentage
point yield gain with consistent seed direction. Contrastive learning modestly
changed raw classifications but reduced useful acceptance relative to its
unchanged frozen encoder control. No candidate qualifies for release.

The conventional scikit-learn TF-IDF/logistic baseline achieved 40.40% mean
development raw accuracy at 400 labels. Its frozen .90 probability acceptance
policy accepted zero rows. Because that policy differs from System1's conformal
policy, its acceptance figures do not establish a matched risk comparison.
Vectorizer/classifier bundle sizes are retained in
[baseline-artifacts.json](baseline-artifacts.json); this supplemental accounting
refits only the already frozen selections and scores no evaluation inputs.

Uncached TF-IDF decisions took approximately .07 ms median; the cached-on-disk
MiniLM encoder required approximately 2.6–2.7 ms for complete uncached
tokenization, encoding, projection and decision on the same machine. Its
90,868,376-byte pretrained weights and Torch/Transformers dependencies are
counted separately from the compiled head. No model download or new product
dependency was introduced. Fitting-time fields for TeachingSession include
session recording and development assessment, not solely a matrix fit. Human
inspection time and installation/startup time were not measured.
The actual runtime/package versions are retained in [environment.json](environment.json).

Full per-seed/budget results: [development](development.json),
[confirmation](confirmation.json), [contrastive development](contrastive-development.json),
[contrastive confirmation](contrastive-confirmation.json).

## New UCI lifecycle and source-shift trial

The fixed Amazon+IMDb development cohort used 1,000 initial fitting labels,
200 calibration labels and 200 recurring development checks, followed by 400
additional reviewed labels. Both candidates had 70.5% raw accuracy; coverage
was 20.0% initially and 21.5% after feedback. Both had seven accepted errors.
The predeclared development gate required .80 raw accuracy, .50 coverage,
at most 20 accepted errors and at most 200 regressions. Neither passed.

The API correctly refused eligibility for independent qualification and
adoption; no qualification request was consumed. An untouched-source
diagnostic was still run under its separate frozen policy:

| Yelp source-shift diagnostic | Result | Frozen requirement |
|---|---:|---:|
| Raw accuracy | 69.28% | Descriptive |
| Correct among accepted | 169 / 194 = 87.11% | Lower bound >=95% |
| Accepted correctness lower bound | 81.57% | >=95% |
| Coverage | 194 / 996 = 19.48% | Lower bound >=80% |
| Coverage lower bound | 17.06% | >=80% |

These stand-alone diagnostic bounds allocate .025 to each tail for joint 95%
confidence, as frozen in this research protocol. They are **not** an issued
product qualification. The product API separately accounts for repeated
qualification attempts. No UCI artifact earned initial approval, so no
adopt/reopen/rollback success is claimed for this corpus.

[Development evidence](lifecycle-development.json),
[frozen diagnostic](lifecycle-confirmation.json).

## Historical lifecycle regression

[HISTORICAL_REPLAY_PROTOCOL.md](HISTORICAL_REPLAY_PROTOCOL.md) was committed at
`0c983a5` before this replay. It uses the unchanged bundled three-intent BANKING77
subset and original annotations, with 287 teaching, 125 calibration and 120
already inspected recurring checks. Its .80 raw accuracy / .80 coverage /
5-error / 10-regression development policy was fixed before assessment.

The original and restored-label candidates each scored 115/120 raw correct,
113/120 accepted and 112/113 correct among accepted. Both earned development
adoption. Intentionally cycling every fitting label produced 4/120 raw correct
and zero accepted; adoption failed and the approved artifact remained intact.
Restoring the original labels created a distinct revision through correction
provenance, without claiming a learning improvement.

All 120 predictions, probabilities, conformal sets and review flags matched
after reopening. Rollback restored the exact first approved bytes, preserved
the restored teaching records, and survived reopening. This demonstrates the
implemented lifecycle on preserved public inputs. It is regression evidence,
with explicit fault injection; the reused cohort cannot provide fresh release
qualification. [Complete replay](historical-replay.json).

## Provenance and retained failures

Emotion's official [dataset card](https://huggingface.co/datasets/dair-ai/emotion/blob/cab853a1dbdf4c42c2b3ef2173804746df8825fe/README.md)
permits educational/research use only. Cite Saravia et al.,
[CARER](https://aclanthology.org/D18-1404/), EMNLP 2018. Its labels are benchmark
annotations and collection documentation is incomplete. Source bytes and
role IDs are pinned in [source-manifest.json](source-manifest.json).

[UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences),
Kotzias (2015), DOI [10.24432/C57604](https://doi.org/10.24432/C57604), is CC BY
4.0 according to UCI. Source bytes and role IDs are pinned in
[lifecycle-source-manifest.json](lifecycle-source-manifest.json). BANKING77
attribution and exact historical source hashes remain in the bundled example.

The [ID-only failure archive](artifacts/evaluation-cases.json.gz) retains wrong
and reviewed outcomes, including the explicitly invalid cache attempt and
complete historical lifecycle comparisons. Its SHA-256 and counts are in
SUMMARY.json. Raw text, corpus archives and trained research weights stay in
ignored local storage; the complete research directory is excluded from
installable distributions apart from designated compact reports.

Both corpora are publicly exposed. Exact/near-duplicate grouping cannot detect
all paraphrases, shared authors or conversations; those metadata are absent.
Pretrained-model exposure cannot be excluded. Fresh to this project does not
mean fresh to a pretrained model or representative customer traffic. These
results do not establish unfamiliar-intent detection, general classification
quality, label efficiency, or production readiness. Teacher calls were zero.
