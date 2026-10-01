#!/usr/bin/env python3
"""API regression replay with explicit faulty labels and original-label repair."""
import json

from threadpoolctl import threadpool_limits
from run import ROOT,HERE,CACHE,read,write,sha
from system1 import ChoiceField,DecisionSchema,TeachingSession

LABELS=["card_arrival","lost_or_stolen_card","cash_withdrawal_charge"]


class Banking(DecisionSchema):
    intent=ChoiceField(options=LABELS)


def clean(report):
    return {k:v for k,v in report.items() if k!="cases"}


def main():
    source=ROOT/"examples/teaching/banking_support.json"
    assert sha(source.read_bytes())=="d1bc96779a6ff92ad1bc21cad725b0f1b8bc08db81ee6d129485490165f4d06a"
    rows=read(source)
    folder=CACHE/"historical-banking"
    if folder.exists():
        raise RuntimeError("Historical replay already started; preserve its assessments")
    session=TeachingSession(folder,Banking,max_features=1024,regularization=.1,choice_solver="ridge")
    session.record_many([dict(input=r["prompt"],label=r["label"],split=role,
                              source="original BANKING77 benchmark annotation",group=r["group"])
                         for role,part in [("teach",rows["teach"]),("calibrate",rows["calibration"]),("evaluate",rows["evaluate"])]
                         for r in part])
    policy=dict(min_accuracy=.8,min_coverage=.8,max_accepted_errors=5,max_regressions=10)
    initial=session.assess(**policy)
    write(folder/"initial.json",initial)
    result=dict(scope="historical recurring development regression; no fresh qualification or improvement claim",
                protocol_sha256=sha((HERE/"HISTORICAL_REPLAY_PROTOCOL.md").read_bytes()),
                source_sha256=sha(source.read_bytes()),policy=policy,initial=clean(initial),
                teacher_calls=0,initial_adopted=False)
    if not initial["passed"]:
        result["unavailable"]="Initial development gate failed; no adoption/rollback earned"
        write(HERE/"historical-replay.json",result)
        print(json.dumps(result))
        return
    session.adopt()
    approved=sha(session.current_path.read_bytes())
    result["initial_adopted"]=True
    result["initial_revision"]=approved
    session.record_many([dict(input=r["prompt"],label=LABELS[(LABELS.index(r["label"])+1)%len(LABELS)],
                              split="teach",source="deliberate cyclic-label fault injection",group=r["group"])
                         for r in rows["teach"]])
    fault=session.assess(**policy)
    write(folder/"fault.json",fault)
    result["fault"]=clean(fault)
    result["approved_preserved_during_fault"]=sha(session.current_path.read_bytes())==approved
    try:
        session.adopt()
        result["fault_adoption"]="unexpectedly adopted"
    except ValueError as error:
        result["fault_adoption"]=str(error)
    session.record_many([dict(input=r["prompt"],label=r["label"],split="teach",
                              source="reviewed restoration of original annotation",group=r["group"])
                         for r in rows["teach"]])
    restored=session.assess(**policy)
    write(folder/"restored.json",restored)
    result["restored"]=clean(restored)
    result["restored_adopted"]=False
    if restored["passed"]:
        session.adopt()
        result["restored_adopted"]=True
        result["restored_revision"]=sha(session.current_path.read_bytes())
        before=[session.predict(r["prompt"]) for r in rows["evaluate"]]
        reopened=TeachingSession(folder)
        after=[reopened.predict(r["prompt"]) for r in rows["evaluate"]]
        result["reopen_predictions_equal"]=all(a.values==b.values and a.probabilities==b.probabilities and
                                               a.conformal_sets==b.conformal_sets and a.is_ambiguous==b.is_ambiguous
                                               for a,b in zip(before,after))
        result["history_before_rollback"]=session.history
        records=sha(session.session_path.read_bytes())
        session.rollback(approved)
        result["rollback_exact_initial_bytes"]=sha(session.current_path.read_bytes())==approved
        result["rollback_preserves_restored_lessons"]=records==sha(session.session_path.read_bytes())
        result["reopen_after_rollback_exact"]=sha(TeachingSession(folder).current_path.read_bytes())==approved
    write(HERE/"historical-replay.json",result)
    print(json.dumps(result))


if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()
