# Independent confirmation: collection brief

The existing full-scope evaluations fail. Public sources audited so far do not
supply fresh human-labeled English requests covering the whole declared scope.
This brief describes the missing input; no collection, reviewer engagement or
payment has occurred. It does not replace the [original protocol](PROTOCOL.md),
relax its targets, or claim a candidate is qualified.

## Deliverable

Collect **8,580 new requests**, matching the original cohort sizes and supported
taxonomies: 30 for each of the 150 CLINC intents (4,500), 1,000 CLINC unfamiliar
requests, and 40 for each of the 77 BANKING77 intents (3,080). Use the exact labels
from the pinned datasets in [prepare.py](prepare.py). Retain all categories.

A UTF-8 CSV is sufficient:

```csv
request_id,dataset,text,label,author_id,reviewer_id,collection_date
```

Use pseudonymous contributor identifiers. Supply a short provenance note stating
how requests were authored, how labels were checked, what instructions reviewers
received, whether any automation was used, and the permission/license allowing
reproducible public evaluation. Do not collect personal or account information.

## Collection and review

- Authors write fresh single-intent English requests from an agreed task
  description. Do not copy or paraphrase benchmark rows or model mistakes.
  Do not use System1, a competing classifier or an LLM to generate the requests
  or decide their correct labels.
- Cover varied wording, request lengths and contexts for every intent. Do not
  select only canonical, easy examples. The unfamiliar cohort must include both
  clearly unrelated requests and requests near supported services that do not
  express a supported intent.
- A different human reviewer labels each request without seeing the author's
  intended label or any classifier prediction. Resolve disagreements from the
  task definitions before scoring; retain the original annotations and the
  adjudication reason. Ambiguous examples must not be removed because a model
  gets them wrong.
- Pin the instructions, collection procedure and category mapping before
  collection. Reserve the raw data for confirmation: it must not enter teaching,
  calibration, threshold selection or model selection.

## Before scoring

Verify the full counts, label coverage, contributor/provenance records and
normalized duplicate overlap against every fitting, calibration, development,
original-test and generated-teaching source. The existing normalization is in
`prepare.group`. Resolve accidental overlap before freezing, record all removals
and replacements, and publish the final cohort hash. Exact-text exclusion does
not establish semantic independence by itself; the collection record matters.

Freeze the candidate, conventional baseline, all dependencies and confirmation
data in Git before their predictions are inspected. Candidate selection must
use only permitted development evidence. Do not call another run on the original
tests fresh confirmation, or silently repair labels after inspecting predictions.

Run each complete adapter once per request after the declared warmup. Publish
all outcomes, accepted errors, per-intent counts, intervals, latency, artifact
size, teacher usage and costs. Both workloads must reach 80% supported coverage
and 99% accepted correctness; CLINC must also stay at or below 1% unfamiliar false
acceptance. Complete warm single-request adapter p95 must remain below 5 ms.
Passing is not guaranteed by obtaining this dataset.

Only a candidate that passes every gate can replace the development artifact in
the n8n recording. Verify that exact skill identity, run the workflow, disconnect
the live teacher, and record continued local decisions and explicit review with
zero new teacher calls. The existing rehearsal remains unqualified.
