"""Additional variants of existing release contracts. Offline; sentinel effects only.
These fixtures test lifecycle semantics, not model accuracy or security detection rates.
"""
from types import SimpleNamespace
import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from system1 import SystemOneEngine, DecisionSchema, ChoiceField
from system1.guard import ActionProposal, DecisionOutcome, PolicyEngine, PolicyRule, SystemOneGuardHook
from system1.ledger import ActionLedger, LedgerError
from system1.receipt import create_decision_receipt, verify_decision_witness_receipt
from system1.integrations.mcp import SystemOneMCPProxy
import system1.compat.typesafe as compat

class Route(DecisionSchema):
    route = ChoiceField(['north', 'south'])

def test_explicit_embedding_does_not_poison_implicit_embedding():
    e=SystemOneEngine(Route,dimension=16,backend='numpy',strict_mode=True)
    prompt='north native request'; native=e.encode(prompt)
    h=e.model.heads['route'];h.set_weights(np.vstack([3*native,-3*native]),np.zeros(2,dtype=np.float32))
    cp=e.conformal_predictors['route'];cp.is_calibrated=True;cp.calibration_scores=np.array([.99]*100)
    explicit=e.decide(prompt,embedding=-native,strict=True,record_receipt=False)
    assert explicit.values['route']=='south'
    implicit=e.decide(prompt,strict=True,record_receipt=False)
    e.use_cache=False;fresh=e.decide(prompt,strict=True,record_receipt=False)
    assert fresh.values['route']=='north'
    assert implicit.values==fresh.values, {'cached_implicit':implicit.values,'uncached_implicit':fresh.values,'cache_hit':implicit.is_cache_hit}

def test_unlabeled_padding_does_not_manufacture_validation_trials():
    schema=Route();fixture=SimpleNamespace(decide=lambda *a,**k:SimpleNamespace(values={'route':'north'}))
    history=[{'state':'one labeled independent case','answers':{'route':'north'}}]
    history += [{'state':f'unlabeled case {i}','answers':{}} for i in range(99)]
    r=compat.evaluate_promotion_eligibility(fixture,history,schema)
    assert r.total_validation_checks==1
    assert not r.is_eligible, r

def test_correlated_rows_are_not_independent_validation_trials():
    schema=Route();fixture=SimpleNamespace(decide=lambda *a,**k:SimpleNamespace(values={'route':'north'}))
    history=[{'state':f'event version {i}','request_id':f'r{i}','group_id':'one_event','answers':{'route':'north'}} for i in range(100)]
    r=compat.evaluate_promotion_eligibility(fixture,history,schema)
    assert not r.is_eligible, r

def test_new_receipt_must_be_checked_against_supplied_trust_anchor(tmp_path):
    trusted=Ed25519PrivateKey.generate();wrong=Ed25519PrivateKey.generate()
    r=create_decision_receipt(schema_name='test',schema_digest='a'*64,prompt='fixture',values={'ok':True},confidences={'ok':1.0},conformal_sets={'ok':['True']},probabilities={'ok':{'True':1.0}},latency_ms=0.1,is_ambiguous=False,signing_key=wrong)
    assert not verify_decision_witness_receipt(r.to_dict(),public_key=trusted.public_key())
    with ActionLedger(tmp_path/'ledger.db') as l:
        with pytest.raises((LedgerError,ValueError)):
            l.record_decision_receipt(r,trusted_public_key=trusted.public_key())

def test_outcome_identity_matches_signed_authorization(tmp_path):
    key=Ed25519PrivateKey.generate()
    with ActionLedger(tmp_path/'ledger.db') as l:
        g=SystemOneGuardHook(ledger=l,signing_key=key,enforcement_profile=True,auto_calibrate=False,
             policy=PolicyEngine([PolicyRule('allow',outcome=DecisionOutcome.ALLOW,action_pattern='^sentinel$')]))
        p=ActionProposal.create(tenant_id='tenant_A',principal_id='alice',scope='tools',tool='sentinel',arguments={'path':'/fixture'},purpose='local fixture')
        r=g.evaluate_proposal(p).decision_result.receipt
        with pytest.raises((LedgerError,ValueError)):
            l.record_execution_outcome(action_id=p.action_id,receipt_digest=r.compute_digest(),status='SUCCEEDED',tenant_id='tenant_B',principal_id='mallory',trusted_public_key=key.public_key())

def test_control_actual_effect_then_reopen_verifies(tmp_path):
    key=Ed25519PrivateKey.generate();db=tmp_path/'ledger.db';calls=[]
    with ActionLedger(db) as l:
        g=SystemOneGuardHook(ledger=l,signing_key=key,enforcement_profile=True,auto_calibrate=False,
             policy=PolicyEngine([PolicyRule('allow',outcome=DecisionOutcome.ALLOW,action_pattern='^sentinel$')]))
        proxy=SystemOneMCPProxy(guard=g)
        request={'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':'sentinel','arguments':{'path':'/fixture'}}}
        out=proxy.handle_call(request,lambda name,args:calls.append((name,dict(args))) or 'effect complete')
        assert 'result' in out and len(calls)==1
        rows=l.entries();assert len(rows)==2
        assert rows[1]['payload']['status']=='SUCCEEDED'
        assert rows[1]['payload']['prior_receipt_digest']==rows[0]['payload']['receipt_digest']
        assert out['result']['_meta']['reflex']['digest']==rows[0]['payload']['receipt_digest']
    with ActionLedger(db) as l:
        assert l.verify_integrity(trusted_public_key=key.public_key())
