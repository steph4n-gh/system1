# A local email-category skill for Inbox Zero

This working integration replaces Inbox Zero's **category-classification call**
with a saved System1 skill. Teach from a JSON file, save it once, and serve decisions
locally. No LLM weights, token generation, or teacher API is needed.

**This remains a review-only pilot.** The second candidate uses learned TF-IDF
word features with the existing System1 ridge head. It needs only NumPy at runtime;
its vocabulary and weights are saved together. It adds no LLM or new classifier
framework.

A separate [public real-email experiment](../../benchmarks/quality/public_email/README.md)
now provides a downloadable spam/ham teaching recipe requiring no mailbox or API
credentials. It reaches 97.6% accepted correctness on its representative grouped
split, while preserving an earlier collection-shift failure. Its native labels
do not cover these seven categories, so this pilot's review-only default remains.

| Evaluation | Correct | Accepted | Accepted correct | Median direct latency |
|---|---:|---:|---:|---:|
| Original hashed skill, 35 upstream cases | 28/35 | 5/35 | 5/5 | 2.02 ms |
| TF-IDF skill with broader lessons, same 35 regressions | 32/35 | 24/35 | 24/24 | 0.20 ms |
| TF-IDF skill, 84 newly authored cases | 83/84 | 81/84 | 80/81 | 0.20 ms |
| Same-feature logistic baseline, 84 newly authored cases | 82/84 | 80/84 | 79/80 | 0.20 ms |

The fresh authored cohort meets the 95% accepted accuracy / 80% acceptance point
targets; the upstream regression cohort still misses acceptance. One unsolicited
recruiting pitch was incorrectly accepted as a conversation. Keep that failure
visible. The new examples are synthetic and authored by the same assistant as
the lessons, not independently collected or blinded mailbox evidence. The 95%
Wilson interval for accepted accuracy on the fresh cohort is 93.3%–99.8%.

[The feature/lesson comparison](feature_comparison.json) separates the changes:
TF-IDF with the **original** lessons gets 34/35 upstream classifications right,
but accepts only 7. Broader lessons improve acceptance. The original TF-IDF/logistic
baseline got 35/35 right and accepted 11; these experiments do not establish
System1 as universally more accurate than a conventional classifier. The current
baseline comparison uses the identical capped vocabulary and teaching rows.

The seven categories are **Newsletter, Marketing, Calendar, Receipt, Notification,
OTP, and Conversations**. Arbitrary custom rules, multi-rule requests, cold-email
classification, personalized `about` instructions, and per-sender corrections are
outside this initial skill and require review. A missing or changed rule cannot
silently become a different category. None-of-the-above is not a taught eighth
class; the uncertainty policy can miss unfamiliar content, so representative
out-of-scope examples belong in mailbox validation.

## What runs

```text
Inbox Zero email + standard rule definitions
  → TypeScript System1 provider → authenticated local HTTP endpoint
  → startup-loaded .s1m skill → category or explicit review
  → accepted category follows existing rules; review persists as SKIPPED
```

Review results include a suggested category when one exists in the existing
execution reason. The default service never authorizes automatic actions. Service
failure, unsupported inputs, or uncertainty also leaves the email unprocessed;
there is no silent classifier-to-LLM fallback for this provider. Existing static
rules can still run when classification is unnecessary.

This does **not** remove every LLM from Inbox Zero. Drafting, summarization,
conversation-status resolution (after a `Conversations` match), and calendar
conversation handling can still call one. This patch removes the selected
classification call only. Account opt-in, sensitive-data handling, and trial
usage checks still apply. A classifier opt-out uses Inbox Zero's existing path.

## Reproduce

Use this System1 checkout and the pinned Inbox Zero revision. Its Node/pnpm and
other application prerequisites still apply. No mailbox credentials are needed
for the tests or evaluation.

```bash
# From this System1 checkout:
python -m pip install -e '.[http]' scikit-learn

git clone https://github.com/elie222/inbox-zero.git ../inbox-zero
cd ../inbox-zero
git checkout 2571ce4970aa5d024bb664a6739bd91b1172c367
git apply ../system1/examples/inbox_zero/upstream/system1.patch
pnpm install --frozen-lockfile --ignore-scripts
pnpm --filter inbox-zero-ai exec prisma generate
cd ../system1

node examples/inbox_zero/export_upstream.mjs ../inbox-zero .system1/inbox-zero/sources
python examples/inbox_zero/teach.py
python examples/inbox_zero/evaluate.py --candidate tfidf --inbox-zero ../inbox-zero
```

The commands assume the System1 checkout is named `system1`; adjust paths if yours
has another name. Generated upstream definitions, fixtures, requests, and saved
skills remain under the ignored `.system1/` directory. Do not commit real mailbox
messages. The exporter reads the pinned Git objects rather than executing the
application or evaluating its TypeScript.

## Teach and correct

[lessons_v2.json](lessons_v2.json) is the editable teaching file: `teach` has 252 examples,
and `calibration` has 140 different examples used to decide when to request review.
Each row pairs an email with **the category you want**. That category is its label.
For example:

```json
{
  "id": "my-correction-001",
  "group": "my-original-email-001",
  "email": {
    "from": "billing@example.com",
    "subject": "Your payment receipt",
    "content": "Your payment of $42 was received. Receipt attached.",
    "hasListUnsubscribeHeader": false
  },
  "label": "Receipt"
}
```

Copy the lesson file, add corrections under `teach`, and keep unrelated examples
under `calibration`. Keep variants of the same original email in one `group` and
one split. The script rejects duplicate inputs and groups crossing the splits.
Then teach a separate candidate:

```bash
python examples/inbox_zero/teach.py \
  --lessons .system1/my-lessons.json \
  --output .system1/inbox-zero/candidate.s1m
```

Check it on fresh emails that did not supply its lessons or calibration. Do not
lower uncertainty thresholds to make the acceptance number look better. Once an
evaluation error becomes a lesson, that email is a regression example, not fresh
quality evidence. Runtime Inbox Zero per-sender feedback is deliberately **not**
automatically incorporated by this generic skill: those requests keep requiring
review even after editing the generic teaching file. Supporting such feedback
requires an account-scoped lesson and validation workflow.

## Connect a pilot

Generate a dedicated service token and keep it in your environment or ignored
configuration. The same token goes in both processes; it is not a teacher API key.

```bash
export SYSTEM1_CLASSIFIER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m system1.integrations.inbox_zero \
  --model .system1/inbox-zero/email.s1m
```

In Inbox Zero's local environment:

```dotenv
DEFAULT_CLASSIFIER=system1:email
SYSTEM1_CLASSIFIER_URL=http://127.0.0.1:8781
SYSTEM1_CLASSIFIER_TOKEN=<the-same-generated-service-token>
DEFAULT_CLASSIFIER_ENABLED=false
```

Enable the classifier for the intended account through Inbox Zero's existing
classifier setting. Use unchanged standard rules and single-rule selection for
this pilot; custom instructions, per-sender feedback, or multi-rule questions
require review. Keep `DEFAULT_CLASSIFIER_ENABLED=false` for an opt-in pilot.
The server defaults to loopback, requires bearer authentication, limits requests
to 64 KiB, bounds email fields, and does not log request bodies. In separate
containers use a private service address reachable from the web process; loopback
refers to each container itself. Use appropriate transport protection if traffic
crosses hosts.

Only after your own validation supports automatic handling, add `--enable-actions`
to the service command. That enables accepted choices; uncertain requests still
require review. Restart the service to load a newly checked `.s1m` file.

## Evidence and scope

[protocol_v2.json](protocol_v2.json) fixes the candidate, file digests, and targets;
[results_v2.json](results_v2.json) preserves all predictions, including the accepted
mistake. The vocabulary is fitted only on teaching rows. The 140 separate
calibration rows are split for temperature fitting and conformal scoring. No
candidate parameters or lessons were changed after seeing the new evaluation.

The 35 public upstream examples are now regression tests: they were already seen
in the initial evaluation. The 84 new examples were written and frozen before
running this candidate, but share an author with its lessons. Passing their point
targets is encouraging; it is not enough to enable real mailbox automation.
Combined metrics do not override a failing individual cohort.

The evaluation compares saved/reloaded values, probabilities, and review flags,
then makes 595 requests through the actual TypeScript provider and Python HTTP
service. Outbound Python connections are blocked and adapter fetches are restricted
to the local endpoint. All responses match, with **zero teacher calls**.
HTTP adapter latency was **0.62 ms median / 0.95 ms p95**, including JSON and
validation (17.92 ms first request). Direct classification was 0.20 ms median,
roughly ten times faster than the original 2.02 ms on the same upstream cases.
These timings are this machine/run, not a portable SLA.

Teaching took about 0.125 seconds, loading 3.83 ms, and the saved skill is about
113 KiB. The runtime and dependencies are extra. The vocabulary is part of the
artifact, so its contents reflect the supplied teaching text. TF-IDF weights
match scikit-learn on all 140 calibration messages to numerical tolerance;
scikit-learn is only a comparison dependency.

The original [protocol](protocol.json), [lessons](lessons.json), and
[results](results.json) remain unchanged. Reproduce them with:

```bash
python examples/inbox_zero/evaluate.py --candidate initial --inbox-zero ../inbox-zero
```

To separate the effects of features and teaching data:

```bash
python examples/inbox_zero/compare_features.py
```

That comparison is diagnostic, not another fresh qualification attempt. It
retains all four combinations of original/broader lessons and hash/TF-IDF
features. The remaining step before mailbox automation is evaluation on a
consented, redacted, representative email set with template/thread separation.
The service stays review-only by default while this evidence is missing.

## Upstream patch and license

The separately licensed [Inbox Zero patch](upstream/system1.patch) applies to the
pinned revision. It is a proposed local integration, not merged or endorsed by
Inbox Zero. Its upstream-derived code is supplied under Inbox Zero's
[license and additional terms](upstream/LICENSE), distinct from System1's Apache
license. The exported fixture text retains that same upstream provenance and is
not included in System1's lesson file. See [NOTICE](upstream/NOTICE.md).
