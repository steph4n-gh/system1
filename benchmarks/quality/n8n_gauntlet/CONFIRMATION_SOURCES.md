# Further confirmation-source search, 2026-09-21

No full-scope independent confirmation source was identified in this search.
The original targets and all 150 CLINC / 77 banking categories remain required.
This is a source inventory, not a model evaluation. No new source was used for
teaching, threshold selection or scoring.

| Candidate source | Observed provenance | Why it does not yet satisfy the requirement |
|---|---|---|
| [Human CLINC paraphrases](https://github.com/kinit-sk/Crowd-vs-GPT-intent-class/tree/8c62edda5206965a45b77d561eb8a9868552539d/ood_robustness_experiments/challenge_data/clinc150) | Human paraphrases of original seeds. The [paper](https://aclanthology.org/2023.emnlp-main.117/) describes 40 sampled intents before filtering; its released human CSV contains 35 numeric labels. | Partial taxonomy, no banking or unfamiliar cohort, and one unresolved mapping. Potential supplemental evidence only. |
| [MTEB Banking77 v2](https://docs.mteb.org/overview/available_tasks/classification/#banking77classificationv2) | Documentation describes corrections to the original dataset. | Corrected labels on previously used material do not establish independent new requests. Not adopted or scored. |
| [BANKING77-OOS / CLINC-Single-Domain-OOS](https://openreview.net/pdf/23386d9ef406f158d1ac5f38a6fe4c90b6a97aaa.pdf) | The paper describes repartitioning original examples and treating selected supported categories as out of scope. | Reuses original data and changes the supported taxonomy. |
| [RDES challenge subsets](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wang25dp/wang25dp.pdf) | The paper selects challenging examples from the original test splits. | Already observed source tests, not a new confirmation cohort. |
| [PhraseSumm](https://aclanthology.org/2023.findings-ijcnlp.8.pdf), sections 3.1–3.3 | Reuses CLINC150, retains multiword intent labels, and creates unseen-label tests by moving original utterances between splits. | Its additional test split is a repartition of original data with a reduced taxonomy, not freshly authored full-scope requests. |
| [Google Example Extrapolation](https://github.com/google/example_extrapolation#data-preprocessing) | The instructions download original CLINC `data_full.json`, split it and generate synthetic examples with a teacher. | An augmentation recipe, not independently labeled new confirmation data. |
| [Route0x evaluation datasets](https://github.com/PrithivirajDamodaran/Route0x#route0x-evals-a-highlight-reel) | Its README links the CLINC/BANKING77 OOS comparisons to the Salesforce datasets already considered above. It also lists other-domain datasets. | No new full CLINC150/BANKING77 confirmation source established here; unrelated taxonomies cannot replace the declared scope. The project's performance claims were not independently evaluated in this source search. |

The [reproducible human-paraphrase audit](audit_paraphrase_source.py) pins the
repository revision and both CSV hashes. It verifies all original split hashes
before comparing normalized groups. Results are retained in
[human-paraphrase-source-audit.json](results/human-paraphrase-source-audit.json):

- 3,019 rows, 3,016 normalized groups, 39 overlapping original folds and 2,977
  groups absent from those folds. This is lexical group exclusion, not proof of
  independence from the original seed meanings or from all pretraining sources.
- 34 numeric labels map unambiguously through the source's original-training CSV
  to original CLINC labels. Numeric label 15 matches both `how_old_are_you` and
  `where_are_you_from`; that disagreement is retained without relabeling.
- The raw human CSV is reserved locally and remains unencoded and unscored.
  Its labels and individual predictions have not been used to choose a model.

Run `python benchmarks/quality/n8n_gauntlet/audit_paraphrase_source.py` after
preparing the original pinned folds. It downloads only two pinned public CSVs
when absent and never opens a model. Any future use needs an explicit protocol,
resolved provenance/label mapping, and disclosure that these are paraphrases.
It cannot authorize the qualified recording on its own.
