# Follow-up after experiment 1

The [first held-out result](results/official-test.md) is a failed qualification,
not a new development baseline to silently overwrite. The original 150-intent
CLINC scope, 77-intent banking scope, 80% coverage, 99% accepted accuracy, 1%
CLINC unfamiliar false acceptance and sub-5-ms p95 remain required. The n8n
workflow and disconnection recording remain unfinished deliverables.

The two concrete deficiencies are unfamiliar-input rejection and banking
distinctions. Keep the encoders fixed and first investigate better teaching of
those decisions. Another encoder, ensemble, larger framework or new product
abstraction is not justified by this result alone.

## Evidence needed before another qualification

The official tests are now observed. A later score on them is follow-up evidence,
not independent confirmation. Do not change their labels, discard ambiguous
rows, reduce the categories or select a few successful workflows as a substitute
for the full gauntlet.

A new experiment must declare its additional teaching examples, calibration and
selection procedure, and a genuinely separate confirmation source covering both
complete supported taxonomies and unfamiliar requests. Freeze the candidate and
all confirmation data before scoring. Teacher-generated examples can help teach;
teacher agreement on synthetic data must not be substituted for independently
established correctness. Independently authored/adjudicated requests may be
needed if a suitable untouched public source cannot be found.

The [source audit](results/confirmation-source-audit.json) compares normalized
groups from two pinned upstream alternatives against every original fold:

- [CLINC OOS-plus](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_oos_plus.json)
  adds 151 distinct unfamiliar groups relative to the prepared original folds.
  Its supported validation/test data and unfamiliar test data add **zero** groups.
- [CLINC Wikipedia augmentation](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/binary_wiki_aug.json)
  adds 14,359 unfamiliar training groups, but no new validation or test groups.
  These are not a fresh full-scope confirmation set or necessarily representative
  of near-boundary user requests.

The audit opened no model and scored no requests. It records URLs, file hashes,
row counts and overlap counts. No new examples have yet been allocated to
teaching, calibration or confirmation. Reserve independent examples before
using any new source to select a policy.

## Next bounded implementation experiment

Teach the review decision with explicitly unsupported requests and difficult
nearby categories, using only declared teaching data. Retain the existing fixed
encoders and saved-head interface. Compare that taught review decision with the
current density/reliability guards on development data, including how much
supported coverage is lost. Keep banking label definitions grounded in the
original fitting examples rather than inferred corrections to test answers.

Before any new run, pin the additional data and a small fixed set of review-policy
settings in a separate experiment protocol. Any new teacher call must retain its
real prompt, response, model, usage and cost; classification prompts must not
contain the expected label. Failures remain in the record. A development gain
alone cannot authorize the qualified demonstration.

The final integration must load the exact newly qualified artifact, run real
n8n requests with item identity preserved, show live-teacher status and call
counts, then disconnect that teacher and continue with local decisions plus
explicit review. The existing local-classifier shutdown test does not prove
teacher disconnection and must not be presented as doing so.
