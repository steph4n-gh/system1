# Prospective contrast-example teaching experiment

Declared before this run's first API request. Protocol v1, all official test
rows, and all four pass criteria remain unchanged. Current fixed-encoder banking
development coverage is 77.6% at 99% accepted accuracy; this is a failure.

This experiment tests whether clearer teaching examples improve the same bounded
77-intent skill without updating any encoder weights. Gemini generates examples;
it is not being scored as a blind classifier in this experiment. Target category
names are therefore part of generation prompts. Do not describe these generated
labels as independent human ground truth or as measured teacher accuracy.

## Fixed generation procedure

- Use `gemini-2.5-flash`, temperature 0.7, thinking disabled, JSON output and at
  most 1,536 output tokens per request.
- Make at most one request for each of BANKING77's 77 categories: 77 requests
  total, requesting 12 examples each (at most 924 proposed lessons). No automatic
  retries or quality-dependent regeneration. Preserve failed requests.
- Select the three nearest category centroids using frozen BGE-small embeddings
  of fitting rows only. Include five hash-ordered fitting examples for the target
  and three for each neighboring category. No calibration, development or test
  text goes to the teacher. No examples are handcrafted from development errors.
- Ask for varied, clearly distinguishable requests, including short requests and
  indirect wording. The target label and neighboring examples establish the
  distinction; avoid copying the supplied examples.
- Record exact prompts, raw responses, model version, usage, latency and errors.
  Preserve the original response even if JSON validation fails. Credentials and
  headers are never recorded. Replay uses saved responses without new calls.
- Reject malformed lessons, labels outside the requested target, duplicate
  normalized groups, and any overlap with original fitting, calibration,
  development or test groups. Test data is used only for this duplicate exclusion,
  as allowed in v1; never for prediction or selection. Keep rejection counts.

## Comparison and limits

Fit frozen MiniLM and BGE-small System1 logistic heads on original fitting rows
plus retained generated lessons, with regularization 0.01, 0.1 and 1. Compare
probability and margin rejection and the already attempted calibration-trained
reliability gate (seven numerical features, no class features, C=0.01).
Run the conventional TF-IDF/logistic baseline with the same added lessons, C=1
and C=10. Original calibration/development boundaries stay fixed. Retain every
result, including regressions; generating clearer examples is not assumed to help.

The generated lessons are an additional, explicitly identified source. Their
provenance is the API response, not the original dataset license. Dataset examples
inside prompts retain the BANKING77 attribution. Report token usage and estimated
published-price cost separately from actual billed cost, which is unavailable.
No final candidate or official-test evaluation is authorized by development
success alone: freeze artifacts and settings as required by v1 first.
