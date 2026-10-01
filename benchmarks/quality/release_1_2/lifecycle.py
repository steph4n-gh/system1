#!/usr/bin/env python3
"""Public, CC BY 4.0 lifecycle trial with a fixed one-shot source-shift gate."""
from __future__ import annotations

import argparse
import json
import time

from scipy.stats import beta
from threadpoolctl import threadpool_limits
from run import ROOT,HERE,CACHE,read,write,sha,outcomes,engine
from system1 import ChoiceField,DecisionSchema,TeachingSession


class Sentiment(DecisionSchema):
    label=ChoiceField(options=["negative","positive"])


FOLDER=CACHE/"sentiment-lifecycle"
POLICY=dict(min_accuracy=.8,min_coverage=.5,max_accepted_errors=20,max_regressions=200)
QUALIFICATION=dict(min_accepted_accuracy=.95,min_coverage=.8,confidence=.95)


def clean_report(report):
    return {k:v for k,v in report.items() if k!="cases"}


def develop():
    if FOLDER.exists():
        raise RuntimeError("Lifecycle development already started; preserve its evidence")
    rows=read(CACHE/"sentiment.json")
    session=TeachingSession(FOLDER,Sentiment,max_features=1024,regularization=.1,choice_solver="ridge")
    records=[dict(input=r["input"],label=r["label"],split=role,source=f"UCI benchmark annotation:{r['origin']}",group=r["id"])
             for role,part in [("teach",rows["fit"]),("calibrate",rows["calibrate"]),("evaluate",rows["develop"])] for r in part]
    session.record_many(records)
    first=session.assess(**POLICY)
    write(FOLDER/"initial-development.json",first)
    initial=None
    if first["passed"]:
        session.adopt()
        initial=sha(session.current_path.read_bytes())
    trial=dict(source="UCI Sentiment Labelled Sentences; benchmark annotations, not customer feedback",
               protocol_sha256=sha((HERE/"PROTOCOL.md").read_bytes()),
               source_manifest_sha256=sha((HERE/"lifecycle-source-manifest.json").read_bytes()),
               initial=clean_report(first),initial_adopted=bool(initial),initial_revision=initial)
    # The entire fixed cohort is inspected, including correct and incorrect
    # predictions. No error-only oracle filtering or free discarded labels.
    current=engine(session.candidate_path)
    before=[]
    for row in rows["corrections"]:
        result=current.decide(row["input"],record_receipt=False)
        before.append(dict(id=row["id"],label=row["label"],predicted=result.values["label"],review=bool(result.is_ambiguous)))
    trial["feedback_count"]=len(before)
    trial["feedback_initial_errors"]=sum(r["label"]!=r["predicted"] for r in before)
    trial["feedback_initial_accepted_errors"]=sum(r["label"]!=r["predicted"] and not r["review"] for r in before)
    write(FOLDER/"feedback-before-labels.json",before)
    session.record_many([dict(input=r["input"],label=r["label"],split="teach",source=f"UCI reviewed feedback:{r['origin']}",group=r["id"])
                         for r in rows["corrections"]])
    corrected=session.assess(**POLICY)
    write(FOLDER/"corrected-development.json",corrected)
    trial["corrected"]=clean_report(corrected)
    trial["corrected_adopted"]=False
    if corrected["passed"]:
        session.adopt()
        trial["corrected_adopted"]=True
        trial["corrected_revision"]=sha(session.current_path.read_bytes())
        prediction=session.predict(rows["unused"][0]["input"])
        reopened=TeachingSession(FOLDER)
        replay=reopened.predict(rows["unused"][0]["input"])
        trial["predict_reopen_equal"]=prediction.values==replay.values and prediction.probabilities==replay.probabilities and prediction.is_ambiguous==replay.is_ambiguous
    trial["history_before_qualification"]=session.history
    # Same reviewed lessons, prospectively stricter deployment requirement.
    # It creates a new, exact assessed artifact; no source-shift scores yet.
    deploy=session.assess(**POLICY,require_qualification=True)
    trial["deployment_assessment"]=clean_report(deploy)
    write(HERE/"lifecycle-development.json",trial)
    freeze=dict(protocol_sha256=trial["protocol_sha256"],source_manifest_sha256=trial["source_manifest_sha256"],
                development_sha256=sha((HERE/"lifecycle-development.json").read_bytes()),
                candidate_sha256=sha(session.candidate_path.read_bytes()),
                session_sha256=sha(session.session_path.read_bytes()),
                development_passed=deploy["passed"],policy=QUALIFICATION,
                source="UCI CC BY 4.0: untouched Yelp source-shift after Amazon+IMDb development",
                initial_revision=initial,qualification_ids_sha256=sha(json.dumps([r["id"] for r in rows["qualify"]])))
    write(HERE/"lifecycle-FREEZE.json",freeze)
    print(json.dumps({k:trial[k] for k in ["initial_adopted","corrected_adopted","feedback_count","feedback_initial_errors"]}))
    print(json.dumps({"initial":first["candidate"],"corrected":corrected["candidate"]}))


def confirm():
    target=HERE/"lifecycle-confirmation.json"
    if target.exists():
        raise RuntimeError("Lifecycle source-shift confirmation already consumed")
    freeze=read(HERE/"lifecycle-FREEZE.json")
    session=TeachingSession(FOLDER)
    assert freeze["candidate_sha256"]==sha(session.candidate_path.read_bytes())
    assert freeze["session_sha256"]==sha(session.session_path.read_bytes())
    assert freeze["protocol_sha256"]==sha((HERE/"PROTOCOL.md").read_bytes())
    assert freeze["source_manifest_sha256"]==sha((HERE/"lifecycle-source-manifest.json").read_bytes())
    rows=read(CACHE/"sentiment.json")["qualify"]
    assert freeze["qualification_ids_sha256"]==sha(json.dumps([r["id"] for r in rows]))
    trial=dict(freeze_sha256=sha((HERE/"lifecycle-FREEZE.json").read_bytes()),
               source=freeze["source"],candidate_sha256=freeze["candidate_sha256"],
               policy=freeze["policy"],api_qualification_ran=False)
    before=sha(session.current_path.read_bytes()) if session.current_path.exists() else None
    if freeze["development_passed"]:
        result=session.qualify([dict(input=r["input"],label=r["label"],group=r["id"]) for r in rows],
                               source=freeze["source"],**freeze["policy"])
        trial["api_qualification_ran"]=True
        trial["qualification"]=result
        try:
            session.adopt()
            trial["qualification_adoption"]="adopted passing artifact"
        except ValueError as error:
            trial["qualification_adoption"]=str(error)
        trial["current_preserved_on_failure"]=result["passed"] or before==(sha(session.current_path.read_bytes()) if session.current_path.exists() else None)
    else:
        # Development failure already prevents deployment. A diagnostic score
        # cannot be represented as a passed API qualification or approval.
        metrics,cases=outcomes(engine(session.candidate_path),rows)
        accepted=metrics["accepted"]
        accepted_correct=metrics["accepted_correct"]
        tail=(1-freeze["policy"]["confidence"])/2
        metrics["accepted_correctness_lower_bound"]=(float(beta.ppf(tail,accepted_correct,accepted-accepted_correct+1))
                                                      if accepted_correct else None)
        metrics["coverage_lower_bound"]=(float(beta.ppf(tail,accepted,len(rows)-accepted+1)) if accepted else 0.)
        metrics["strict_source_shift_gate_pass"]=bool(
            metrics["accepted_correctness_lower_bound"] is not None and
            metrics["accepted_correctness_lower_bound"]>=freeze["policy"]["min_accepted_accuracy"] and
            metrics["coverage_lower_bound"]>=freeze["policy"]["min_coverage"])
        write(FOLDER/"source-shift-diagnostic-cases.json",cases)
        trial["diagnostic_metrics"]=metrics
        trial["qualification_adoption"]="not eligible: frozen development gate failed"
        trial["current_preserved_on_failure"]=before==(sha(session.current_path.read_bytes()) if session.current_path.exists() else None)
    if freeze["initial_revision"]:
        trial["rollback"]=session.rollback(freeze["initial_revision"])
        trial["rollback_exact_bytes"]=sha(session.current_path.read_bytes())==freeze["initial_revision"]
        trial["records_preserved_by_rollback"]=freeze["session_sha256"]==sha(session.session_path.read_bytes())
        reopened=TeachingSession(FOLDER)
        trial["reopen_after_rollback"]=sha(reopened.current_path.read_bytes())==freeze["initial_revision"]
    else:
        trial["rollback"]="unavailable: initial development candidate did not earn approval"
    write(target,trial)
    print(json.dumps(trial))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=["develop","confirm"])
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        {"develop":develop,"confirm":confirm}[args.phase]()
