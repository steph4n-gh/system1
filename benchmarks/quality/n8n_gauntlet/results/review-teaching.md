# Taught review: development experiment 2a

**Not qualified.** The [protocol](../REVIEW_TEACHING_PROTOCOL.md) was committed
and pushed in `5ddde63` before the 18 planned review-head fits. The original
intent-head bytes and encoder identities remain unchanged. The comparison used
original fitting/calibration correctness examples, with and without predicted
category information. CLINC additionally compared zero versus 2,000 pinned
Wikipedia negatives. No new live teacher call occurred.

The [selection report](review-teaching-development.json) retains all 12 CLINC and
six banking configurations. Additional negative teaching improved CLINC review
relative to the same review model without it, but that is not a held-out finding.

| Complete development adapter | Supported accepted | Correct among accepted | Unfamiliar falsely accepted | p95 |
|---|---:|---:|---:|---:|
| CLINC | 2,591/2,995 (86.5%) | 2,566/2,591 (99.04%) | **2/100: fails 1% limit** | 2.71 ms |
| Banking | 1,580/1,960 (80.6%) | 1,565/1,580 (99.05%) | No native OOS cohort | 4.24 ms |

The [complete runtime report](review-runtime-development.json) contains every
decision and Wilson intervals. Both adapters ran with OS networking denied,
receipts off and no response cache. Each request executed its encoder. Banking
accepts five more correct development requests than experiment 1, a small
development difference that does not establish improved held-out quality.

CLINC's cached batched-feature selection predicted 2,587 supported acceptances
and one unfamiliar acceptance. Actual single-request inference accepted 2,591
supported requests and two unfamiliar requests. The numerical features therefore
were not interchangeable across these paths. A [128-row-per-workload diagnostic](feature-consistency.json)
confirmed substantial MiniLM batch/single differences; its largest probability
difference exceeded 11 percentage points. Repeated single requests were identical.
BGE-small differences in that sample were under 0.0001 percentage points.

The declared next comparison is [consistent single-request features](../CONSISTENT_FEATURES_PROTOCOL.md)
throughout CLINC teaching and selection. It keeps encoder weights fixed and adds
no new hyperparameter grid. Its outcome is not assumed here. The 151 additional
human OOS examples remain reserved and unscored. Fresh independent supported-input
confirmation covering both full taxonomies remains unresolved; neither the
original failed test nor an easier subset can replace it.

Reproduce development preparation/selection and complete runtime measurement:

```bash
python benchmarks/quality/n8n_gauntlet/prepare_review_teaching.py
python benchmarks/quality/n8n_gauntlet/review_teaching.py
sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  python benchmarks/quality/n8n_gauntlet/review_runtime.py
```

The scripts use the original published intent heads, pinned encoder files and
public splits. Teaching feature caches are distinct from inference: cached
features are never counted as runtime measurements. The runtime materializes
saved candidates under ignored `.system1/n8n-gauntlet/review-*-candidate` folders.
They remain research candidates and are not connected as qualified n8n skills.
