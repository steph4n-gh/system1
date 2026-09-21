# Teaching consistency reveals annotation concerns and genuine classifier errors

Experiment 2o was [declared](../TEACHING_CONSISTENCY_PROTOCOL.md) and pushed in
`85920952480a8c1de8337040a27376ffed56eaf9` before running. It audits the original
fitting examples using semantic and word-based classifiers, each predicting
examples outside its own fitting folds. **It changes no label, final classifier
or qualification result.** The full [regression failure](latest-regression.md)
remains unchanged.

| Original fitting cohort | Rows | Semantic disagreements with supplied label | Word-based disagreements | Both suggest the same different label |
|---|---:|---:|---:|---:|
| CLINC150, including native OOS | 12,019 | 430 | 967 | 121 |
| BANKING77 | 6,026 | 477 | 978 | 182 |

There are **303 consensus disagreements**, not 303 verified annotation errors.
Both methods use the same original teaching data and can make correlated
mistakes. Maximum class probabilities are model scores, not probabilities that
an annotation is wrong. Of the CLINC consensus disagreements, six concern its
80 original native OOS fitting examples; 115 concern supported intents. Banking
has no native OOS fitting cohort. All rows, fixed probability bins and predictions
are retained in the [full report](teaching-consistency.json).

## What inspection shows

Some fitting labels appear inconsistent with their text and neighboring lessons:

| Original fitting example | Supplied label | Both classifiers suggest |
|---|---|---|
| how is the weather today | `maybe` | `weather` |
| how much is my water bill | `bill_due` | `bill_balance` |
| turn your volume up | `whisper_mode` | `change_volume` |
| please open a call to my mother | `change_ai_name` | `make_call` |
| how many days processing new card? | `contactless_not_working` | `card_arrival` |

For example, the first water-bill utterance's nearest permitted fitting neighbor
is “how much is my water bill for”, labeled `bill_balance`, with cosine 0.9832.
This is evidence for review, not an automatic correction. There are also
counterexamples to trusting agreement: both classifiers interpret “How can I
prove who I am?” as `unable_to_verify_identity`, although the supplied
`verify_my_identity` label reasonably describes a request for instructions.
The latter has high semantic confidence (0.9918) but only 0.5865 lexical confidence.

The source-lineage audit verifies all **28,080 fitting, calibration and development
rows** against the pinned upstream text/label pairs using their original source
and index. There are zero import label mismatches. CLINC's training and validation
sections are selectively decoded from its checksum-verified source; its test
sections are not decoded into rows. Banking uses only the pinned training CSV.
The suspicious assignments were not introduced by our partitioning code.

Separately, the diagnostic inspects all 27 already-recorded accepted CLINC
development errors (26 supported mistakes and one unfamiliar false acceptance)
and all 15 banking errors of the current System1 incumbents. Each includes
nearby original fitting examples. Some have near-equivalent text with different
labels; for example, “My cards were stolen” is labeled `lost_or_stolen_phone`,
while “My card was stolen” is a fitting example labeled `lost_or_stolen_card`.

Other development mistakes reflect genuine distinctions the classifier misses:
“tell me my name” versus “tell me your name”, requesting a new card versus
checking an application, and “you are not wrong about that” versus disagreement.
These need better decisions or review, not a relabeling excuse. This diagnostic
does not establish an irreducible accuracy ceiling or explain away the full
CLINC unfamiliar-input failure.

## Procedure and verification

The semantic classifier uses the same fixed MiniLM/BGE encoders, individual
request features and numerical settings. The conventional method retains
TF-IDF/logistic C=10. All temporary heads use the existing three original folds
and calibration cohort; no generated lesson enters their teaching or prototypes.
The reused helper also reconstructs already-observed parent review signals, but
those signals do not select a new artifact or modify any annotation.

The report prioritizes fifty consensus disagreements per dataset by semantic
confidence, lexical confidence and normalized group hash. It shows the three
nearest semantic neighbors, nearest supplied-label example and nearest
suggested-label example. Every neighbor comes from the two permitted fitting
folds; the entire held-out fold is excluded. The 42 development-error views use
only original fitting neighbors. Similarities do not serve as new labels.

Local preparation takes 23.611 seconds for CLINC and 12.168 seconds for banking,
including the temporary heads, review-signal preparation and neighbor inspection.
Original feature caches were populated; the 27 and 15 development-error vectors
were encoded individually in this run. These are diagnostic preparation times,
not complete-adapter latency or first-install teaching benchmarks. OS networking
was denied; there are zero teacher API calls and zero new API costs.

The [record audit](teaching-consistency-audit.json) checks every one of the 18,045
fitting outcomes, both sets of fifty prioritized examples, all 42 development
errors, source lineage, nine declared sources and all 147 frozen files. It
recomputes neighbor order/similarities from four hash-recorded feature caches,
without fitting or running an encoder. No original test cohort or human reserve
is scored; no label, category, threshold or qualification target changes.

The [seven correction proposals](teaching-correction-proposals.json) preserve
each original fitting example, source index, original label, proposed label and
reason. Six concern CLINC and one banking; none is applied. They were authored
by the coding assistant, whose session cost is not metered by this benchmark;
there is no new teacher API call. These are proposed teaching labels, not
independently verified truth. Uncertain cases are left unchanged.

The next controlled quality step should test these **fitting-only teaching
corrections**, preserving original records. Their effect must be compared with
the incumbents using unchanged calibration and development labels before any
candidate advances. Blindly adopting all 303 consensus suggestions is unjustified.
Fresh full-scope confirmation and the exact qualified n8n recording remain
unfinished. PR #3 stays open; no release.
