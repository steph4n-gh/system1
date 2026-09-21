# Recorded boundary teaching: experiment 2e

**This is synthetic teaching data, not a quality result or independent ground
truth.** The [procedure](../BOUNDARY_TEACHING_PROTOCOL.md) was committed in
`28dfe4a` before request preparation. All 227 exact requests were committed in
`459d317` before API calls. A filename preflight rejected the original banking
label `reverted_card_payment?`; `3743414` corrected only that overly strict check
and updated the code hash before any API attempt. All prompts and labels remained
unchanged, as recorded in the [request audit](boundary-request-audit.json).

Each request used only original fitting examples for context. Five target
examples close to neighboring category centroids supplied difficult distinctions,
alongside three examples from each of three nearby categories. The full category
list kept the requested scope explicit. CLINC prompts additionally asked for
related unsupported requests and explanations. No encoder weights were updated,
and no calibration/development/test/reserved text was sent to the teacher.

| Teaching measurement | Observed result |
|---|---:|
| Gemini 2.5 Flash calls / responses | 227 / 227 |
| Input / output / total tokens | 252,684 / 80,778 / 333,462 |
| Retained CLINC supported examples | 1,696 |
| Retained CLINC unsupported examples | 674 |
| Retained banking supported examples | 917 |
| Total retained lessons | 3,287 |
| Rejected normalized duplicates | 637 |
| Malformed responses / missing usage | 0 / 0 |
| Per-call API p50 / p95 | 2,405.9 / 3,153.4 ms |
| Full generation wall time | 516.0 seconds |
| Standard-price estimate / actual bill | $0.2777502 / unavailable |

The price estimate uses Google's [published Gemini 2.5 Flash rates](https://ai.google.dev/gemini-api/docs/pricing#gemini-2.5-flash),
checked September 21, 2026: $0.30 per million text input tokens and $2.50 per
million output tokens. It is not an account billing statement. There were no
retries, model fallbacks, quality-dependent regeneration or additional calls.
Generation latency is not classifier latency and is not a matched Jev comparison.

The [raw requests/responses](boundary-teaching/teacher.json),
[retained lessons](boundary-teaching/lessons.json) and
[usage, pricing and rejection record](boundary-teaching/summary.json) are complete.
The [replay audit](boundary-teacher-audit.json) reproduced every retained lesson
and rejection, checked payload hashes and summed usage without new API calls.
Every rejection was a normalized duplicate against an excluded source or an
earlier proposal in this batch. All original folds, previous generated examples,
Wikipedia teaching rows and reserved human sources participate only in duplicate
exclusion. The human reserves remain unencoded and unscored.

Unsupported explanations are the teacher's rationale, not independent validation
of those labels. A valid JSON response can still contain a bad teaching example.
No examples were repaired, relabeled or filtered based on evaluation performance.
The 637 duplicates also show why generated proposal count must not be reported
as the number of new lessons retained.

The [fixed comparison runner](../boundary_development.py) uses the existing small
intent/review heads and the same frozen encoders, with all generated examples
excluded from cross-validation fold heads and prototypes. It gives conventional
TF-IDF/logistic classification the same new teaching data. A
[standalone baseline format preflight](boundary-baseline-format-preflight.json)
reproduced 32 existing development responses exactly except for the expected new
artifact identity. No model was fitted in that format check.

The ten teacher-boundary tests cover one-attempt accounting, failures,
interruption, credential exclusion, duplicate rejection, invalid responses and
original-label punctuation. CI passed on `3743414` after one unchanged-code retry:
the first Python 3.14 job failed the existing Pokémon mean-latency assertion at
25.26 ms against 25 ms. Its [original failed attempt](https://github.com/steph4n-gh/system1/actions/runs/35640526990/attempts/1)
remains visible. No threshold was changed.

The [observed regression failure](observed-regression.md) remains the latest
larger-cohort quality evidence. This teaching run alone does not improve that
result, supply independent confirmation, or qualify the n8n recording. All scope
and pass criteria are unchanged; PR #3 remains open.
