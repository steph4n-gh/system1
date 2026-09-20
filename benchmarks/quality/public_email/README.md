# A real public email skill, with no API key

System1 learns **spam versus legitimate mail (`ham`)** from the original labels
in the [Apache SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorpus/readme.html).
These are historical real emails, not emails generated for this project. A saved
skill runs locally with the existing TF-IDF projector and ridge classifier.
There are no LLM weights, teacher calls, or new product dependencies.

The representative teaching recipe reaches **603/618 correct accepted decisions
(97.6%)**, accepting **618/632 cases (97.8%)**. Teaching takes **0.94 seconds**,
the saved skill is **78.2 KiB**, and median warm decisions take **0.216 ms** on the
recorded machine. Reloaded answers, probabilities, prediction sets and review
flags match exactly. Fifteen accepted decisions are wrong; review is not a
guarantee of correctness.

This is a separate spam/ham skill. It does **not** qualify the seven-category
[Inbox Zero pilot](../../../examples/inbox_zero/README.md), which retains its
review-only default. It does not send, move or delete email.

## Two experiments, including the failure

| Experiment and classifier | Raw correct | Accepted | Accepted correct | Median decision |
|---|---:|---:|---:|---:|
| Original collections → later collections: System1 | 1,253/1,404 (89.2%) | 1,378/1,404 (98.1%) | 1,239/1,378 (89.9%) | 0.230 ms |
| Same source holdout: TF-IDF/logistic | 1,114/1,404 (79.3%) | 1,017/1,404 (72.4%) | 870/1,017 (85.5%) | 0.257 ms |
| Both collections represented, groups held out: System1 | 609/632 (96.4%) | 618/632 (97.8%) | 603/618 (97.6%) | 0.216 ms |
| Same representative split: TF-IDF/logistic | 613/632 (97.0%) | 632/632 (100%) | 613/632 (97.0%) | 0.241 ms |

The first experiment teaches on the original `easy_ham`, `hard_ham` and `spam`
archives and evaluates on later `easy_ham_2` and corrected `spam_2` additions.
It fails the 95% accepted-correctness target. The original teaching split contains
291 spam and 1,197 ham messages; the later evaluation contains 957 spam and 447 ham.
That is a substantial change in source and class balance. The result demonstrates
that the old lessons and calibration do not transfer reliably to that collection;
it does not isolate class balance as the only cause.

After observing that failure, we froze a **second, retrospective experiment**:
both collections can supply teaching, calibration and evaluation, but related
emails remain grouped and no evaluation group enters that candidate's teaching
or calibration. The settings, features and uncertainty threshold are unchanged.
This measures a more representative teaching recipe on public labels. It is not
a new blinded corpus, and does not erase the source-shift failure. Its split has
1,964 teaching, 661 calibration and 632 evaluation emails.

System1's 95% Wilson interval for accepted correctness in the second experiment
is **96.0%–98.5%**. It meets the 95% accepted correctness / 80% acceptance point
targets. The conventional baseline has slightly better raw correctness; System1
requests review more often and has fewer accepted errors. The small timing
difference on one machine is not evidence of a universal speed advantage.

## Reproduce

Use this checkout containing the TF-IDF feature option. Scikit-learn is used only
for the comparison; it is not required to load or run the resulting System1 skill.

```bash
python -m pip install -e . scikit-learn

# Representative teaching and evaluation, with the already-frozen split:
python benchmarks/quality/evaluate_public_email.py --download --representative

# Reproduce the original source-shift failure separately:
python benchmarks/quality/evaluate_public_email.py --download
```

Downloads require no account or credential. Every source is checked against its
recorded size and SHA-256. After preparation, outbound socket connections are
blocked during teaching, loading and scoring. There are zero teacher calls.
The command prints quality results even when targets fail; exit success means
the experiment completed, not that its quality targets passed.

The representative run writes these local artifacts:

- `.system1/public-email-representative/lessons.json`: real labeled examples in
  separate `teach`, `calibration` and `evaluate` arrays. Only `teach` fits the
  vocabulary and decision weights.
- `.system1/public-email-representative/spam.s1m`: the saved reusable skill.
- `.system1/public-email-representative/results.json`: metrics and every prediction.

Run the saved skill through the ordinary CLI:

```bash
system1 decide "Subject: Project meeting tomorrow. Please bring the revised notes." \
  --model .system1/public-email-representative/spam.s1m --json
```

This command illustrates loading a skill; its invented input is not evaluation
evidence. In the recorded smoke test it suggested `spam` but returned
`is_ambiguous: true`: a legitimate-looking new phrase still needs review. Treat
the category as a suggestion when that flag is true, not an accepted decision.

## Data boundaries and measurement

The five archives contain 6,046 messages. Preparation retains 3,257 groups after
excluding 222 later messages overlapping the original collections, 2,556 other
messages in an already represented group, and 11 messages in conflicting-label
groups. The source publisher's older total predates a corrected spam label.

Connected groups share a message identifier/reference, normalized input/body,
normalized subject with at least three words, or an 80-word body prefix. One
representative per group prevents long threads from dominating the score.
This reduces leakage; it is not an exhaustive semantic near-duplicate detector.
The split manifests contain IDs, group IDs, labels and input hashes, not email
text. Raw mail and generated lessons stay under ignored `.system1/` directories.

Features contain only decoded subject and body, capped at 8,192 characters.
MIME parsing prefers plain text, strips markup/scripts/styles for HTML, and
ignores attachments. Spam labels, archive paths and delivery headers never enter
the model. Vocabulary and IDF are fitted on teaching rows only. Calibration is
separate, with disjoint temperature and conformal subsets. The baseline uses the
same vocabulary, IDF and conformal subset. Both use alpha 0.05 and disable caches.

Timings measure five interleaved passes of individual messages, with warm-up on
teaching data. System1 includes strict review decisions and unsigned receipt
construction, but no signing, persistence, HTTP or mailbox actions. Reports
include p95, first-call and load timings. No live LLM was timed, so these results
do not establish a measured speedup over an LLM. Teaching time covers vocabulary
and weight fitting plus calibration, after download and email preparation.

The [source terms](https://spamassassin.apache.org/old/publiccorpus/readme.html)
offer this corpus for spam-filter evaluation and retain message copyright with
the original senders. The corpus is **not** licensed under this repository's
Apache-2.0 license. We publish download instructions, hashes and derived metrics,
not the messages. Do not inject the corpus into a live email delivery system.

## Frozen evidence

- Source holdout: [protocol](protocol.json), [split manifest](splits.json),
  [complete results](results.json). Protocol frozen in commit `c4af23d` before scoring.
- Representative recipe: [protocol](protocol_representative.json),
  [split manifest](splits_representative.json), [complete results](results_representative.json).
  Protocol frozen in commit `3811ab9` before scoring that recipe.

The practical finding is that System1 can learn a useful, compact email skill
from a public labeled file in under a second. Its quality depends on what the
teaching examples and calibration cover. Historical spam/ham evidence does not
establish modern mailbox performance or seven-category routing quality.
