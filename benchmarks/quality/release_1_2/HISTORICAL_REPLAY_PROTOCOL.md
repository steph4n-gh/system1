# Historical banking lifecycle regression replay

Freeze before this replay's first assessment. This is API regression evidence on
an already public, already inspected cohort, **not new quality evidence**.

Input: `examples/teaching/banking_support.json`, SHA-256
`d1bc96779a6ff92ad1bc21cad725b0f1b8bc08db81ee6d129485490165f4d06a`.
It preserves original BANKING77 labels, CC BY 4.0 attribution and source
revision `57ec275d8078af65b7731c2a98be812d844a6d6b`. Use its exact 287 teach,
125 calibrate, 120 recurring development rows and group IDs. Those 120 official
test rows were already evaluated by this project and cannot qualify a release.
No new teacher requests, downloads or generated text.

TeachingSession uses the existing TF-IDF ridge defaults: 1,024 features and
regularization .1. Predeclared development adoption policy: raw accuracy >=.8,
coverage >=.8, accepted-error count <=5, accepted-correct regressions <=10.
Do not relax these limits if the replay fails.

Sequence: assess the original labels and adopt only if passing; deliberately
cycle **all fitting labels** to the next schema choice as an explicit fault
injection; assess, retain the failure and verify the approved artifact remains;
restore every original fitting label with restoration provenance; assess and
adopt only if passing; compare predictions across reopening; rollback to the
exact first approved bytes and verify restored lessons remain. Restoration
changes provenance and artifact identity, not learned quality. Retain every
assessment and explicit fault outcome. If initial or restored assessment fails,
report unavailable steps rather than manufacture an approved revision.
