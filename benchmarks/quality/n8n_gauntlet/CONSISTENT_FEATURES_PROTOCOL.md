# Single-request feature consistency: development experiment 2b

Declare this experiment before fitting. Experiment 2a's saved-adapter run accepted
two unfamiliar CLINC development requests, although its cached-feature selection
predicted one. The [consistency diagnostic](results/feature-consistency.json)
then compared the first 128 development rows one at a time versus in batches of
16 and 32. The quantized MiniLM features changed substantially, with maximum
probability differences above 0.11 and one changed top label. Repeated individual
requests were identical. BGE-small differences were below 0.000001 in probability
and changed no top labels in that sample.

This mismatch is a property of the optional quantized encoder path used by this
research experiment, not evidence that System1's built-in hashed projector or
every encoder has the same behavior. No official test or reserved human OOS
request was used in the diagnostic.

## One fixed comparison

Rebuild the CLINC intent head and review head with **one-request-at-a-time encoder
features throughout teaching, calibration and development selection**, matching
the actual local request path. Keep the frozen MiniLM encoder weights, all source
rows and labels, original group boundaries, logistic regularization .1, conformal
alpha .05, three review-teaching folds and review C=.1 unchanged. Use the selected
experiment-2a review features (seven numerical features plus predicted category)
and the same 2,000 pinned Wikipedia negatives. Do not add a hyperparameter grid.
Banking remains at experiment 2a because this diagnostic did not find comparable
feature drift there.

Use a distinct cache identity for individual projections, never overwrite or
reuse batched feature arrays under that identity. The saved head still serves
the same single-request encoder operation; count its complete latency and bytes.
Record single-request feature extraction time separately from head compilation,
review fitting and warm inference. Neither encoder weights nor a language model
are fine-tuned.

Select the review threshold on the full original development cohort using the
same unchanged quality targets and strict System1 eligibility. Save the new
intent head and review guard, reload them and measure the complete single-request
adapter after 100 development warmups with OS networking denied. Every development
case remains in the denominator. Verify saved decisions agree with selection
decisions; retain a mismatch or failed target rather than hiding it.

Keep experiment 1 and experiment 2a results unchanged. No official test or human
OOS reserve is scored here. This is development evidence only. The original full
150/77-intent goal, independently established confirmation, exact qualified n8n
artifact, live teacher disconnection and recording remain required.
