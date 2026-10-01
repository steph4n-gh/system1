# System1 1.2.0 bounded research protocol

Frozen before evaluation scoring on 2026-10-01. This is a research experiment,
not a deployment qualification or a replacement for any existing benchmark.

## Sources and scope

The six-choice selection experiment uses `dair-ai/emotion`, configuration
`split`, revision `cab853a1dbdf4c42c2b3ef2173804746df8825fe` (16,000 training,
2,000 validation, 2,000 test rows). The official [dataset card](https://huggingface.co/datasets/dair-ai/emotion/blob/cab853a1dbdf4c42c2b3ef2173804746df8825fe/README.md)
permits **educational and research purposes only**. Neither raw data nor trained
weights enter the release or this repository. Saravia et al., [CARER](https://aclanthology.org/D18-1404/),
EMNLP 2018, describes the source. Its labels are supplied benchmark annotations,
not independently reviewed customer judgments. The official card documents
machine-generated annotations and leaves collection details incomplete.

The lifecycle trial uses [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences),
Kotzias (2015), DOI [10.24432/C57604](https://doi.org/10.24432/C57604), licensed
CC BY 4.0 by the official repository. It contains 1,000 sentences each from
Amazon, IMDb, and Yelp. Retained evidence contains IDs and results, not text.

Both corpora are publicly exposed and merely fresh to this project. Neither
establishes performance on private traffic, unfamiliar intents, or arbitrary
distribution shifts. No paid teachers, model downloads, or official-target
changes are permitted. Source URLs, byte sizes, SHA-256 hashes and role IDs are
retained before scoring.

## Grouping and separation

Normalize whitespace and case. Group exact duplicates and near duplicates with
word-trigram Jaccard similarity >= .8, requiring >= 3 shared trigrams, via
connected components. Keep the lexicographically smallest normalized-text hash
per component; discard conflicting-label components. Grouping checks may read
annotations to discard conflicts but do not inspect evaluation errors or fit
any model. No author/conversation IDs or timestamps exist in these extracts;
related-author leakage cannot be ruled out. Short paraphrases can evade the
specified duplicate check.

Emotion: source training is the acquisition pool plus calibration reserve;
source validation is recurring development; source test is untouched
confirmation. Remove any training/validation components connected to test
components and any training components connected to validation components.
Assign roles without reading class labels. Forty randomly ordered training
rows per seed are the separate calibration reserve. The next sixty rows form
the initial teaching batch; acquisition sees remaining inputs only. Every
inspected label is retained; no oracle class balancing. If a class is absent,
record an infeasible candidate rather than inspect extra uncharged labels.

Seeds: 11, 29, 47. Total inspection budgets: 100, 200, 400; each includes the
same 40 calibration labels and therefore 60, 160, 360 fitting labels. The
calibration cost is shown separately and included in the total. Development
and confirmation annotations are separate measurement costs, always reported.

## Fixed candidates

Primary: exact current TeachingSession TF-IDF ridge behavior, max_features=1024,
regularization=.1, strict decisions, alpha=.05, no caching or synthetic
augmentation. Three acquisition paths start from identical per-seed initial
lessons: uniform random, smallest top-two probability margin, and uncertainty
plus diversity. For diversity, reserve 20% of each acquisition batch for a
deterministic random audit sample, take the 5*remaining-count most uncertain
inputs, and select by greedy farthest distance in current normalized TF-IDF
features (start with the most uncertain). Stable input-index tie breaks. At
200 and 400 labels, refit from all acquired lessons; no discarded inspections.
The parent API may use an equivalent bounded cosine diversity implementation;
any difference must be recorded, not silently substituted after scoring.

Conventional comparator: scikit-learn word unigram/bigram sublinear TF-IDF,
max_features=1024 and LogisticRegression(C=1,max_iter=1000), trained on exactly
the same uniform-random labels. Selective acceptance uses a single frozen .9
maximum-probability threshold; report raw accuracy as well as its separate
acceptance policy. A second comparator uses System1's existing logistic solver
with the same portable feature settings and calibration policy; this is a
solver comparison, not a new semantic architecture.

A bounded optional task-specific contrastive candidate is permitted only when
cached local encoder assets and dependencies already exist: fixed frozen
MiniLM-L6 embeddings plus 20 epochs of a 384->64 linear normalized projection,
supervised contrastive loss temperature=.1, Adam lr=.001, batch64, seeds as
above; System1 ridge head fitted afterwards. Limit to 400 random-label budget
and three seeds. Count embedding latency, dependencies and weights. If assets
are absent or import fails, retain a skipped candidate and reason. This tests a
different task and learning objective, but prior encoder-adaptation failures
remain relevant; a win here does not repair the full assistant/banking gates.

## Measures and decisions

For every candidate/budget/seed: raw accuracy, accepted accuracy, accepted
correct yield (accepted_correct / all rows), accepted errors, coverage/review
rate, per-class outcomes, 95% one-sided accepted-error bound, labels inspected,
feature/fitting/selection time, artifact size and uncached end-to-end latency.
Retain every mistaken or reviewed row ID and its predicted/true class.

The exploratory feasibility screen is accepted accuracy >= .90, coverage >=
.20 and a 95% one-sided accepted-error upper bound <= .10. It is deliberately
weaker than deployment requirements and never authorizes adoption.

An acquisition feature earns a superiority claim only if at 400 total labels it
passes the screen, improves mean accepted-correct yield by >= .02 over random,
and improves yield in all three seeds. Confirmation must preserve the mean
gain and screen on the untouched test split. Fix all configurations and retain
model/selection hashes in a committed FREEZE.json before scoring confirmation.
If no candidate wins, keep the negative result and do not claim better label
efficiency. No second search against consumed confirmation.

## Lifecycle trial

UCI Amazon+IMDb provide grouped fit, calibration and recurring development
inputs; untouched Yelp is source-shift qualification. Fixed SHA-based order,
seed20261001: 1,000 fit, 200 calibration, 200 development initially; add 400
remaining training corrections without inspecting development labels for
selection. Preserve all inspected labels and provenance. Base and corrected
models use max_features=1024, regularization=.1 and ridge. Development adoption
policy: raw accuracy >= .80, coverage >= .50, accepted-error count <=20,
regressions <=200; these are explicit rehearsal checks, not deployment gates.
Qualification policy declared before scoring: .95 one-sided lower bound on
accepted accuracy, coverage >= .80, confidence=.95. Only one qualification
batch per exact artifact, retain failures. Do not adopt any failing artifact
under a qualification-required policy. Independently demonstrate reopen,
prediction, exact artifact adoption if eligible, and rollback of an approved
development revision; label a development-only replay honestly. If initial
development checks fail, preserve them and report the unavailable steps rather
than relaxing thresholds or manufacturing successful qualification.

Prospective amendment before any scoring: the 1,000 initial fitting labels and
.50 development coverage requirement replace the original 100/.20 rehearsal
settings. Duplicate removal may leave fewer than 400 correction rows; retain
all available rows and report the exact count. No scores informed this change.
