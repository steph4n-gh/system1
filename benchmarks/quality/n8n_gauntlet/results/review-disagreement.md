# Semantic/lexical agreement is not a full-scope fix

This exploratory diagnostic reuses published development predictions. It fits no model, makes no new prediction or teacher call, and opens no original-test or reserved-human data. It asks whether a second classifier's existing suggestion contains a useful simple disagreement signal before adding runtime complexity.

The [replay script](../analyze_review_disagreement.py) aligns normalized groups and labels across the current System1 candidates and two lexical comparisons: the matching original parent intent heads, and the stronger boundary-taught CLINC baseline. The [complete report](review-disagreement.json) pins source-report hashes, candidate identities, error groups, counts and Wilson intervals. The extra CLINC baseline had additional generated lessons; it is not a same-teaching-data comparison.

Two fixed policies are computed from these saved responses, with no threshold search:

- **Agreement veto:** retain a current System1 acceptance only when the lexical suggestion agrees.
- **Strict agreement only:** use agreement plus System1's singleton prediction set and a supported suggestion, bypassing the learned review score.

| Workload / lexical comparison / policy | Supported coverage | Correct among accepted supported | Unfamiliar falsely accepted |
|---|---:|---:|---:|
| clinc150 / matched-parent-labels / current-with-agreement-veto | 2,439/2,995 (81.44%) | 2,424/2,439 (99.38%) | 1/100 (1.00%) |
| clinc150 / matched-parent-labels / strict-agreement-only | 2,631/2,995 (87.85%) | 2,605/2,631 (99.01%) | 6/100 (6.00%) |
| clinc150 / stronger-boundary-taught-baseline / current-with-agreement-veto | 2,455/2,995 (81.97%) | 2,442/2,455 (99.47%) | 1/100 (1.00%) |
| clinc150 / stronger-boundary-taught-baseline / strict-agreement-only | 2,648/2,995 (88.41%) | 2,626/2,648 (99.17%) | 6/100 (6.00%) |
| banking77 / matched-parent-labels / current-with-agreement-veto | 1,495/1,960 (76.28%) | 1,484/1,495 (99.26%) | No native cohort |
| banking77 / matched-parent-labels / strict-agreement-only | 1,693/1,960 (86.38%) | 1,628/1,693 (96.16%) | No native cohort |

For banking, the veto removes four of 15 accepted mistakes but also 82 correct accepted requests. Coverage falls from 1,581/1,960 (80.66%) to 1,495/1,960 (76.28%), below 80%. Bypassing the learned review restores coverage but accepts 65 mistakes, far below the 99% accuracy target. Merely adding a consensus check does not meet the goal.

CLINC's veto removes 10 of 25 accepted supported mistakes with the matching lexical parent, or 12 with the stronger baseline. It also rejects 145 or 127 correct accepted cases. Coverage remains above 80%, but neither version removes the one currently accepted development unfamiliar request. Agreement without the learned review accepts six unfamiliar requests, above the 1% limit.

The models' mistakes are not identical: lexical predictions are correct on 56 CLINC cases (62 with the stronger baseline) and 61 banking cases where System1's raw suggestion is wrong. Those counts establish complementary predictions, not a deployable way to identify the right model per request. No oracle routing, majority-vote accuracy or speculative speedup is claimed.

These are computed routing diagnostics, not new saved-adapter measurements. Combined runtime and latency were not measured; component p95 values cannot be added to claim a joint p95. No ensemble was implemented or promoted. The next bounded review experiment can instead test the input features already computed by each classifier, avoiding a second encoder or teacher. Any such change must be declared before fitting and compared against the stronger retained incumbents. Independent full-scope confirmation remains outstanding.
