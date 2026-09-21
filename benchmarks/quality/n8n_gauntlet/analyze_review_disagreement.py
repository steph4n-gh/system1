"""Exploratory development diagnostic from saved predictions; no fits or timing."""
import json
from pathlib import Path

from audit_observed_regression import digest, metrics

HERE = Path(__file__).resolve().parent


def main():
    paths = {key: HERE / "results" / name for key, name in {
        "polynomial": "polynomial-runtime.json", "boundary": "boundary-runtime.json",
        "clinc": "consistent-review-runtime-development.json"}.items()}
    reports = {key: json.loads(path.read_text()) for key, path in paths.items()}
    report = dict(scope="exploratory development-only disagreement diagnostic; not qualification",
        qualified=False, sources={key: dict(path=str(path.relative_to(HERE)), sha256=digest(path)) for key, path in paths.items()},
        new_model_calls=0, teacher_calls=0, joint_runtime_measured=False, comparisons=[])
    for dataset in ("clinc150", "banking77"):
        semantic = (reports["clinc"]["results"][0] if dataset == "clinc150" else
            next(r for r in reports["polynomial"]["results"] if (r["dataset"], r["kind"], r["condition"]) == (dataset, "system1", "quadratic")))
        lexical = [("matched-parent-labels", next(r for r in reports["polynomial"]["results"]
            if (r["dataset"], r["kind"], r["condition"]) == (dataset, "baseline", "quadratic")))]
        if dataset == "clinc150":
            lexical.append(("stronger-boundary-taught-baseline", next(r for r in reports["boundary"]["results"]
                if (r["dataset"], r["kind"], r["condition"]) == (dataset, "baseline", "positive-and-unsupported"))))
        for name, baseline in lexical:
            outcomes = {policy: [] for policy in ("current-system1", "current-with-agreement-veto", "strict-agreement-only")}
            counts = dict(supported=0, both_correct=0, only_semantic_correct=0, only_lexical_correct=0,
                neither_correct=0, supported_agreements=0, wrong_supported_agreements=0,
                current_accepted_supported_errors=0, accepted_supported_errors_with_disagreement=0,
                current_accepted_correct_supported=0, accepted_correct_supported_with_disagreement=0,
                current_accepted_unfamiliar=0, accepted_unfamiliar_with_disagreement=0)
            for s, b in zip(semantic["outcomes"], baseline["outcomes"], strict=True):
                assert s["group"] == b["group"] and s["truth"] == b["truth"]
                agreement = s["suggestion"] == b["suggestion"]
                supported = s["truth"] != "oos"
                if supported:
                    counts["supported"] += 1
                    sc, bc = s["suggestion"] == s["truth"], b["suggestion"] == s["truth"]
                    counts["both_correct" if sc and bc else "only_semantic_correct" if sc else "only_lexical_correct" if bc else "neither_correct"] += 1
                    counts["supported_agreements"] += int(agreement)
                    counts["wrong_supported_agreements"] += int(agreement and not sc)
                    if not s["needsReview"]:
                        counts["current_accepted_correct_supported" if sc else "current_accepted_supported_errors"] += 1
                        counts["accepted_correct_supported_with_disagreement" if sc else "accepted_supported_errors_with_disagreement"] += int(not agreement)
                elif not s["needsReview"]:
                    counts["current_accepted_unfamiliar"] += 1
                    counts["accepted_unfamiliar_with_disagreement"] += int(not agreement)
                assert isinstance(s["predictionSet"], list)
                accepts = {"current-system1": not s["needsReview"],
                    "current-with-agreement-veto": not s["needsReview"] and agreement,
                    "strict-agreement-only": agreement and s["suggestion"] != "oos" and s["predictionSet"] == [s["suggestion"]]}
                for policy, accept in accepts.items():
                    outcomes[policy].append(dict(group=s["group"], truth=s["truth"], suggestion=s["suggestion"],
                        category=s["suggestion"] if accept else None, needsReview=not accept))
            summaries = {}
            for policy, rows in outcomes.items():
                m = metrics(rows)
                summaries[policy] = dict(metrics=m, development_quality_targets_pass=(m["supported_coverage"]["fraction"] >= .8
                    and m["accepted_accuracy"]["fraction"] >= .99 and (m["oos_false_acceptance"]["fraction"] is None or m["oos_false_acceptance"]["fraction"] <= .01)),
                    accepted_error_groups=[r["group"] for r in rows if not r["needsReview"] and r["category"] != r["truth"]])
            report["comparisons"].append(dict(dataset=dataset, lexical_comparison=name,
                semantic_manifest_sha256=semantic.get("candidate_sha256", semantic.get("manifest_sha256")),
                lexical_manifest_sha256=baseline["candidate_sha256"], counts=counts, policies=summaries))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
