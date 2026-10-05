# Courier strict-review recovery comparison

Declared 2026-10-05 before the independent confirmation cohort. Evaluate review
handling in the shipped courier simulator with the exact saved skills from
`examples/gaming/skill_playground/results/quality-evidence.zip`, SHA-256
`19d4d89013e4ba5105ff8cb62142480da45467f2623cd5b0cb44d72186e0d6a0`.
No fitting, calibration changes, lessons, downloads or paid experts.

The preliminary map-seed cohort was 98200–98219, paired across ordinary and
corrected purple terrain. It selected immediate fallback: all three hybrid
policies had identical actions, completion, expert calls and safety at every
cap. Those seeds are now regression data. Independently reviewed confirmation
uses 99300–99319, fixed before the first score. Familiar local facts can recur;
these are not held-out language examples or representative customer jobs.

Compare strict stop, immediate fallback, plain bounded waiting, progress-triggered
fallback and direct known rules. Hybrid caps are 0, 1, 8 and 256 attempted expert
callbacks per map. Failures count; exhausted caps dispatch none. Every arm has
the same 180-action world horizon. One runner owns an episode, with one pending
callback and a one-second expert timeout. Callback failure stops it without retry.

The expert is the existing cheap `teacher.decide`, explicitly updated for the
changed purple rule. Independently validate fallback against current observation,
permission and known movement rules. Preserve statistical review flags. A
reviewed local proposal cannot execute without validated fallback. Existing
playground argmax simulation defaults remain unchanged.

Reuse a model response only while the complete observation matches. Neither
waiting arm gets redundant inference. Plain waiting polls up to three times
after the first review. Progress falls back after the second identical reviewed
observation, with the same four-observation absolute bound. Physics does not
advance during waiting. Polls sleep one millisecond for this timing experiment;
report actual scheduling time separately. That chosen delay establishes no paid
expert, external source-fetch or customer queue latency savings.

Interleave hybrid arms in rotating order fixed by seed offset. Record every
action and stop reason, model evaluation, actual native invocation, attempt
reservation, wait poll, per-turn latency, episode elapsed time and unsafe
proposal/dispatch. Unsafe includes known lava/purple danger, walls and locked
gates. Do not repair actions using world truth. Direct rules are a separate cheap
baseline; their native invocations are not billed fallback attempts or subject
to a fictitious zero cap.

Freeze source, artifact, runtime, policy, seeds and budgets before scoring;
refuse existing output directories. Retain failures and paired differences.
Select immediate if progress has no completion, safety or expert-budget benefit.
State clearly if direct rules remain strongest. Report structural duplicates
without seed metadata and prior teaching/development/evaluation overlaps. Do
not infer significance from paired terrain conditions or repeated base layouts.

Independent review must replay physics, reconcile counters and exact paired
action/resource parity, and challenge missing, failed, invalid, unsafe, stale,
denied and late answers. Run the full suite with localhost supported before a
draft PR. No merge, release or deployment is part of this experiment.
