# Teaching close distinctions

This round changes the teaching examples, with no new dependencies, runtime
algorithm, calibration data, or compiler settings. The ordinary example commands
automatically use the additional lessons:

```bash
python examples/support_triage.py
python examples/model_routing.py
python examples/agent_guard.py
```

Support routing already handled all 24 diagnostic contrasts correctly, so its
teaching data stays unchanged. Model routing receives **six lessons** separating
summarizing or translating a supplied proof from checking its correctness.
Operation triage receives **28 lessons** distinguishing local inspection and
editing from external transfer or destructive removal. Variations concerning
the same object share a teaching group.

For example, reading a local telemetry log is `inspect`; uploading it to a remote
server is `restricted`. These are intake classifications. The example executes
none of the described operations, and a label grants no permission.

## Reproduce the comparison

```bash
python benchmarks/quality/evaluate_contrasts.py
```

The runner reads the same JSON teaching files twice: once with all current
lessons, and once excluding rows marked `quality_round` to reproduce 1.0 teaching.
It uses 2048 features, regularization 0.1, no augmentation, unchanged calibration,
alpha 0.05, strict mode, NumPy, and no decision cache. Both variants are saved and
reloaded; answers, probabilities, uncertainty sets, and review flags must agree
before and after reload. Socket connections are blocked throughout.

The [cases](contrast_cases.json) contain 24 diagnostic and 24 confirmation cases
per skill. Diagnostic contrasts and the existing evaluation cases informed lesson
selection. The confirmation cohort was authored after selecting the final lessons
and before evaluating that cohort; teaching-file hashes detect later changes to
the selected lessons.
No diagnostic or confirmation examples enter teaching or calibration. Prompt and
group checks reject accidental reuse across the measured cohorts and teaching.

The [recorded report](results/contrast_round.json) includes all predictions,
reviews, errors, hashes, timings, and descriptive intervals. The command writes
`.system1/contrast-round/report.json` and the six saved skills. It also runs in
pytest. Its regression checks require the original and combined cohorts to meet
95% accepted correctness and 80% acceptance, with no decrease in combined raw
correctness or increase in combined accepted errors against the 1.0 baseline.
Those checks do not certify every individual cohort or production traffic.

## Results

| Skill | Diagnostic correct, before → after | Fresh confirmation correct, before → after | Fresh accepted, before → after | Fresh accepted errors, before → after |
|---|---:|---:|---:|---:|
| Support routing | 24 → 24 / 24 | 22 → 22 / 24 | 19 → 19 | 0 → 0 |
| Model routing | 23 → 24 / 24 | 20 → 22 / 24 | 23 → 23 | 4 → 2 |
| Operation triage | 22 → 24 / 24 | 23 → 24 / 24 | 18 → 23 | 0 → 0 |

All 72 diagnostic cases are now correct. Fresh confirmation improves from 65/72
to 68/72 raw correct. Across the original, diagnostic, and confirmation cohorts
together, raw correctness rises from 458/476 to 464/476; accepted decisions rise
from 438 to 447, and accepted errors decrease from six to four.

The original cohorts retain their raw correctness: support 98/104, routing 95/96,
operations 131/132. Current accepted counts and correctness are 91/92 support,
94/95 routing, and 126/126 operations. In routing, one original JSON-to-TOML error
that previously requested review is now accepted. This regression is retained in
the report; the reduction in errors elsewhere does not erase it.

## Limits and teacher provenance

The fresh routing cohort still contains **two accepted errors**: summarizing a
lemma document and verifying a supplied proof are both routed to `local_small`.
Its accepted correctness is 21/23 (91.3%), below the 95% point target. Fresh support
acceptance is 19/24 (79.2%); both wrong raw predictions request review. The overall
regression checks pass, but these harder individual cohorts are not qualified for
the target on this evidence.

These are small AI-authored examples exercising known policies with new wording.
They share authorship and task concepts, so exact prompt and group separation
does not establish semantic independence. This is a measured improvement in
bounded skills, not proof of broad language understanding or reliable negation.
The fresh failures were retained rather than added to this round's lessons.

All 34 new lessons are locally authored; there are **no new Jev calls or answers**.
The existing Jev cache is untouched. `evaluate_release.py` continues to reproduce
the recorded 1.0 Jev baseline and separately checks the current manually taught
skills. Its output labels those two sources explicitly.
