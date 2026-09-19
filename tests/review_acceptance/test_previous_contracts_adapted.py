"""Replay of previous contract tests on the 23:07 UTC snapshot.
Fixture adaptations: _history is now schema-keyed; explicit insufficient-evidence rejection is accepted for two dependency groups.
All behavioral assertions retained.
No provider calls; effects are list appends. Small deterministic model fixtures
isolate execution semantics, not real-world classification accuracy.
Run: PYTHONPATH=/path/to/repo/src pytest -q test_round3_closure.py
"""
from __future__ import annotations
import copy
import hashlib
import json
import re
import threading
import inspect
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from system1 import ReflexEngine, DecisionSchema, ChoiceField, MultiChoiceField
from system1.calibration import ConformalPredictor, RegressionConformalPredictor
from system1.compiler import CompiledSystemOneModel
from system1.guard import ActionProposal, DecisionOutcome, PolicyEngine, PolicyRule, ReflexGuardHook
from system1.integrations.mcp import ReflexMCPProxy
from system1.ledger import ActionLedger, LedgerError
from system1.receipt import create_decision_receipt, verify_decision_witness_receipt
import system1.compat.typesafe as compat

class Route(DecisionSchema):
    route=ChoiceField(options=['north','south'], descriptions={'north':'north route','south':'south route'})

def engine(*,p=.99999,strict=True,ledger=None,key=None):
    e=ReflexEngine(Route,dimension=16,backend='numpy',strict_mode=strict,
                   margin_threshold=.1,ledger=ledger,signing_key=key)
    h=e.model.heads['route']
    h.set_weights(np.zeros_like(h.weights),np.log([p,1-p]).astype(np.float32)*.25)
    c=e.conformal_predictors['route'];c.is_calibrated=True;c.calibration_scores=np.array([.99]*100)
    return e

def proposal(tool='sentinel'):
    return ActionProposal.create(tenant_id='tenant',principal_id='alice',scope='tools',tool=tool,
               arguments={'path':'/safe/readme'},canonical_target='/safe/readme',purpose='read fixture')

def allow_policy():
    return PolicyEngine([PolicyRule('allow',outcome=DecisionOutcome.ALLOW,target_pattern='.*')])

def mcp_request():
    return {'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':'sentinel','arguments':{'path':'/safe/readme'}}}

# Controls retain important fixes; all exercise current code.
def test_control_deny_overrides_allow():
    p=PolicyEngine([PolicyRule('allow',outcome=DecisionOutcome.ALLOW,target_pattern='.*'),
                    PolicyRule('deny',outcome=DecisionOutcome.DENY,target_pattern='.*')])
    assert p.evaluate(proposal())[0]==DecisionOutcome.DENY

def test_control_allow_constraint_violation_is_denied():
    p=PolicyEngine([PolicyRule('allow',outcome=DecisionOutcome.ALLOW,target_pattern='.*',allowed_principals=['bob'])])
    assert p.evaluate(proposal())[0]==DecisionOutcome.DENY

def test_control_enforcement_rejects_missing_key(tmp_path):
    with ActionLedger(tmp_path/'a.db') as l:
        with pytest.raises(ValueError):
            ReflexGuardHook(engine=engine(ledger=l),enforcement_profile=True,policy=allow_policy())

def test_control_deterministic_allow_records_and_executes(tmp_path):
    key=Ed25519PrivateKey.generate()
    with ActionLedger(tmp_path/'a.db') as l:
        g=ReflexGuardHook(engine=engine(ledger=l,key=key),policy=allow_policy(),enforcement_profile=True)
        result=g.evaluate_proposal(proposal())
        assert result.allowed
        r=result.decision_result.receipt
        assert verify_decision_witness_receipt(r.to_dict(),public_key=key.public_key())
        assert any(row['payload'].get('receipt_digest')==r.compute_digest() for row in l.entries())
        calls=[]; response=ReflexMCPProxy(guard=g).handle_call(mcp_request(),lambda name,args: calls.append(name) or 'ok')
        assert calls==['sentinel'] and 'result' in response

def test_control_readonly_ledger_blocks_deterministic_allow(tmp_path):
    key=Ed25519PrivateKey.generate();path=tmp_path/'a.db';ActionLedger(path).close()
    with ActionLedger(path,read_only=True) as l:
        g=ReflexGuardHook(engine=engine(ledger=l,key=key),policy=allow_policy(),enforcement_profile=True)
        calls=[];ReflexMCPProxy(guard=g).handle_call(mcp_request(),lambda name,args: calls.append(name))
        assert calls==[]

def test_control_zero_egress_rejects_passthrough():
    with pytest.raises(ValueError): compat.TypeSafeClient(mode='passthrough',zero_egress=True)

def test_control_uncalibrated_categorical_strict_abstains():
    e=engine();e.conformal_predictors['route'].is_calibrated=False
    assert e.decide('fixture',strict=True).is_ambiguous

def test_control_zero_labels_cannot_promote():
    e=engine();r=compat.evaluate_promotion_eligibility(e,[{'state':'x','answers':{}}],e.schema)
    assert not r.is_eligible and r.total_validation_checks==0

# A/B final authorization and durable evidence, using real SQLite and Ed25519.
def test_model_final_authorization_is_actually_in_ledger(tmp_path):
    key=Ed25519PrivateKey.generate()
    with ActionLedger(tmp_path/'a.db') as l:
        # Conditional invariant: any ALLOW must reference its own persisted receipt.
        # A corrected enforcement API may instead reject this ungranted path.
        try:
            g=ReflexGuardHook(engine=engine(ledger=l,key=key),enforcement_profile=True)
        except ValueError:
            return
        result=g.evaluate_proposal(proposal())
        if not result.allowed:
            assert result.outcome in (DecisionOutcome.DENY, DecisionOutcome.REQUIRE_APPROVAL)
            return
        r=result.decision_result.receipt
        rows=l.entries()
        assert any(row['payload'].get('receipt_digest')==r.compute_digest() for row in rows), {
            'final_digest':r.compute_digest(),'recorded_digests':[row['payload'].get('receipt_digest') for row in rows],
            'borrowed_ledger_record_id':r.ledger_record_id}

def test_enforcement_does_not_invent_permission_for_unmatched_tool(tmp_path):
    key=Ed25519PrivateKey.generate()
    with ActionLedger(tmp_path/'a.db') as l:
        p=PolicyEngine([PolicyRule('known_only',outcome=DecisionOutcome.ALLOW,action_pattern='^known_tool$')])
        g=ReflexGuardHook(engine=engine(ledger=l,key=key),policy=p,enforcement_profile=True)
        calls=[];out=ReflexMCPProxy(guard=g).handle_call(mcp_request(),lambda name,args:calls.append(name) or 'ok')
        assert not calls, {'effect_count':len(calls),'response':out}

def test_corrupt_history_prevents_new_enforced_effect(tmp_path):
    key=Ed25519PrivateKey.generate()
    with ActionLedger(tmp_path/'a.db') as l:
        g=ReflexGuardHook(engine=engine(ledger=l,key=key),policy=allow_policy(),enforcement_profile=True)
        assert g.evaluate_proposal(proposal()).allowed
        l._connection.execute("UPDATE audit_entries SET payload_json=? WHERE sequence=1", ('{"tampered":true}',))
        assert not l.verify_integrity(trusted_public_key=key.public_key())
        calls=[];out=ReflexMCPProxy(guard=g).handle_call(mcp_request(),lambda name,args:calls.append(name) or 'ok')
        assert not calls, {'effect_count':len(calls),'response':out}

def test_orphan_execution_outcome_is_rejected(tmp_path):
    with ActionLedger(tmp_path/'a.db') as l:
        with pytest.raises((LedgerError,ValueError)):
            l.record_execution_outcome(action_id='nonexistent',receipt_digest='a'*64,status='SUCCEEDED')

# C cache and calibration contracts.
def test_supplied_embedding_is_part_of_exact_cache_identity():
    e=engine();h=e.model.heads['route'];w=np.zeros_like(h.weights);w[0,0]=3;w[1,0]=-3
    h.set_weights(w,np.zeros(2,dtype=np.float32))
    x=np.zeros(e.dimension,dtype=np.float32);x[0]=1
    a=e.decide('same text',embedding=x,strict=True);assert a.values['route']=='north'
    b=e.decide('same text',embedding=-x,strict=True)
    e.use_cache=False;c=e.decide('same text',embedding=-x,strict=True)
    assert b.values==c.values, {'cached':b.values,'uncached':c.values,'hit':b.is_cache_hit}

def test_distinct_valid_alphas_do_not_alias_across_order_statistic():
    e=engine(p=.8);e.conformal_predictors['route'].calibration_scores=np.array([.7]*95+[.99]*5)
    permissive=.0594061;stricter=.0594058
    a=e.decide('same text',alpha=permissive,strict=True);assert not a.is_ambiguous
    b=e.decide('same text',alpha=stricter,strict=True)
    e.use_cache=False;c=e.decide('same text',alpha=stricter,strict=True);assert c.is_ambiguous
    assert b.is_ambiguous==c.is_ambiguous, {'cached_set':b.conformal_sets,'fresh_set':c.conformal_sets,'hit':b.is_cache_hit}

def test_learning_marks_previous_calibration_stale():
    e=engine();before=e.model.heads['route'].weights.copy()
    assert not e.decide('fixture',strict=True).is_ambiguous
    e.learn_from_tier2('fixture',{'route':'north'})
    assert not np.array_equal(before,e.model.heads['route'].weights)
    r=e.decide('fixture',strict=True)
    assert r.is_ambiguous, {'still_calibrated':e.conformal_predictors['route'].is_calibrated,'set':r.conformal_sets}

def test_uncalibrated_multilabel_strict_abstains():
    schema=DecisionSchema(schema_name='multi',fields={'tags':MultiChoiceField(options=['a','b'])})
    e=ReflexEngine(schema,dimension=16,backend='numpy',strict_mode=True)
    h=e.model.heads['tags'];h.set_weights(np.zeros_like(h.weights),np.array([3.,-3.],dtype=np.float32))
    r=e.decide('fixture',strict=True)
    assert r.is_ambiguous, {'calibrator_has_is_calibrated':hasattr(e.calibrators['tags'],'is_calibrated'),'values':r.values}

# D serialized artifact must preserve the actual validated runtime.
def make_promoted_client():
    questions={'route':compat.Choice('Route',criteria={'north':'north route','south':'south route'})}
    c=compat.TypeSafeClient(mode='auto_cutover',dimension=16,backend='numpy',zero_egress=True,
        baseline_handler=lambda *args: None,promotion_policy=compat.PromotionPolicy(min_agreement_threshold=.8))
    c._history[compat._build_dynamic_schema(questions).schema_digest()]=[{'state':f'north route read north item {i}', 'answers':{'route':'north'},'questions':questions} for i in range(100)]
    assert c.distill_and_cutover(questions), c._last_promotion_report
    return c,questions

def test_exported_candidate_preserves_validated_runtime_calibration(tmp_path):
    c,q=make_promoted_client();live=c._get_engine(q);p=tmp_path/'candidate.s1m'
    assert c.export_model(p)
    restored=CompiledSystemOneModel.load(p,backend='numpy')
    e=ReflexEngine(restored.schema,model=restored,backend='numpy')
    assert e._calibration_digest()==live._calibration_digest(), {
        'live_temperature':live.calibrators['route'].temperature,'loaded_temperature':e.calibrators['route'].temperature,
        'live_scores':len(live.conformal_predictors['route'].calibration_scores),
        'loaded_scores':len(e.conformal_predictors['route'].calibration_scores)}

def test_one_request_many_fields_is_not_many_independent_validation_requests():
    schema=DecisionSchema(schema_name='correlated',fields={f'f{i}':ChoiceField(['north','south']) for i in range(20)})
    answers={name:'north' for name in schema.fields}
    fixture=SimpleNamespace(decide=lambda *args,**kwargs:SimpleNamespace(values=answers))
    r=compat.evaluate_promotion_eligibility(fixture,[{'state':'one independent request','answers':answers}],schema)
    assert not r.is_eligible, {'independent_requests':1,'counted_checks':r.total_validation_checks,'wilson_lower':r.wilson_lower_bound}

def test_common_lineage_is_not_split_just_to_fill_validation():
    history=[{'request_id':f'r{i}','group_id':'A' if i<6 else 'B','state':f'q{i}','answers':{'route':'north'}} for i in range(10)]
    try:
        p=compat.partition_cutover_history(history)
    except ValueError as exc:
        assert 'Insufficient evidence' in str(exc)
        return
    # A valid implementation may defer promotion; it must not split a dependency group.
    sets=[{r['group_id'] for r in fold} for fold in (p.train_history,p.calib_history,p.val_history)]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]), {
        'train_groups':list(sets[0]),'calibration_groups':list(sets[1]),'validation_groups':list(sets[2])}

# Packaging consistency is a static source check, not a minimum-runtime install.
def test_declared_grpc_lower_bound_covers_generated_stubs():
    import system1
    from packaging.requirements import Requirement
    root=Path(system1.__file__).resolve().parents[2]
    py=tomllib.loads((root/'pyproject.toml').read_text())
    requirements={Requirement(s).name:Requirement(s) for s in py['project']['optional-dependencies']['grpc']}
    stub=(root/'src/system1/proto/reflex_pb2_grpc.py').read_text()
    generated=re.search(r"GRPC_GENERATED_VERSION = '([^']+)'",stub).group(1)
    minimum=next(s.version for s in requirements['grpcio'].specifier if s.operator=='>=')
    from packaging.version import Version
    assert Version(minimum)>=Version(generated), {'declared':minimum,'generated_requires':generated}


def test_async_mcp_tool_is_not_recorded_succeeded_before_it_runs(tmp_path):
    from system1.integrations.mcp import wrap_mcp_tool
    key=Ed25519PrivateKey.generate()
    with ActionLedger(tmp_path/'a.db') as l:
        g=ReflexGuardHook(engine=engine(ledger=l,key=key),policy=allow_policy(),enforcement_profile=True)
        calls=[]
        @wrap_mcp_tool(guard=g,tool_name='sentinel')
        async def sentinel(path='/safe/readme'):
            calls.append(path)
            return 'ok'
        pending=sentinel()
        try:
            assert calls==[]
            outcomes=[r['payload'] for r in l.entries() if r['event_type']=='execution_outcome']
            assert not any(o['status']=='SUCCEEDED' for o in outcomes), outcomes
        finally:
            if inspect.iscoroutine(pending): pending.close()
