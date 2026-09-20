# A local email-category skill for Inbox Zero

This working integration replaces Inbox Zero's **category-classification call**
with a saved System1 skill. Teach from a JSON file, save it once, and serve decisions
locally. No LLM weights, token generation, or teacher API is needed.

**The first lesson is a review-only pilot, not a qualified automatic replacement.**
On 35 upstream regression examples, System1 got 28 categories right and accepted
5 for automatic handling; all 5 were correct. A plain TF-IDF/logistic classifier
using the same teaching rows got 35 right and accepted 11. Neither met our
95% accepted accuracy / 80% acceptance targets. The baseline is both more accurate
and faster here. This identifies a concrete improvement target for System1's text
representation and decision head; it does not establish which component causes
the gap.

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
python examples/inbox_zero/evaluate.py --inbox-zero ../inbox-zero
```

The commands assume the System1 checkout is named `system1`; adjust paths if yours
has another name. Generated upstream definitions, fixtures, requests, and saved
skills remain under the ignored `.system1/` directory. Do not commit real mailbox
messages. The exporter reads the pinned Git objects rather than executing the
application or evaluating its TypeScript.

## Teach and correct

[lessons.json](lessons.json) is the editable teaching file: `teach` has 140 examples,
and `calibration` has 56 different examples used to decide when to request review.
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

[protocol.json](protocol.json) records the fixed initial candidate and thresholds;
[results.json](results.json) contains every prediction, timing, source digest, and
failure. The 35 examples are public maintainer-authored regression fixtures from
Inbox Zero, not private customer emails. They were not used to fit or calibrate,
but source was reviewed during integration work: this is compatibility evidence,
not a blind benchmark or a production accuracy claim.

The evaluation compares the saved/reloaded skill with the pre-save result, then
sends 175 requests through the actual TypeScript provider and Python HTTP service.
It blocks outbound Python connections and restricts adapter fetches to the local
endpoint. Results: **zero teacher calls**, identical direct/adapter choices and
review decisions, **2.66 ms median / 6.72 ms p95** HTTP adapter latency including
JSON and validation (17.55 ms first request). Direct System1 classification was
2.02 ms median; TF-IDF was 0.22 ms. Timings are this machine/run, not a portable SLA.
Teaching took about 0.36 seconds; the saved skill is about 56 KiB; see the JSON for
exact values and environment. The installed runtime and dependencies are extra.
Only five accepted examples gives a 95% Wilson accuracy interval of roughly
56.6%–100%; observing zero errors in five cases proves little about future mail.

The immediate quality work is to investigate the representation/head gap using
teaching and development data, preserve these upstream cases as regressions, and
then evaluate on a fresh, consented mailbox sample with template/thread separation.
The service and adapter can remain unchanged while the skill improves.

## Upstream patch and license

The separately licensed [Inbox Zero patch](upstream/system1.patch) applies to the
pinned revision. It is a proposed local integration, not merged or endorsed by
Inbox Zero. Its upstream-derived code is supplied under Inbox Zero's
[license and additional terms](upstream/LICENSE), distinct from System1's Apache
license. The exported fixture text retains that same upstream provenance and is
not included in System1's lesson file. See [NOTICE](upstream/NOTICE.md).
