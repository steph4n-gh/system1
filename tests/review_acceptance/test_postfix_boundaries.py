"""Post-fix variants of the existing calibration, artifact, promotion and ledger contracts.
Offline. Native NumPy models where inference is relevant; controlled label outputs
where isolating promotion statistics. These are regression fixtures, not accuracy benchmarks.
"""
import hashlib
from types import SimpleNamespace
import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from system1 import DecisionSchema, ScoreField, ReflexEngine
from system1.compiler import CompiledHeadWeights, CompiledSystemOneModel
from system1.ledger import ActionLedger, LedgerError
from system1.receipt import create_decision_receipt
import system1.compat.typesafe as compat
from .test_previous_contracts_adapted import Route, engine, make_promoted_client


def test_nonstrict_update_invalidates_calibration_for_later_strict_call():
    e = engine(strict=False)
    before = e.model.heads['route'].weights.copy()
    assert not e.decide('fixture', strict=True).is_ambiguous
    e.learn_from_tier2('fixture', {'route': 'north'})
    assert not np.array_equal(before, e.model.heads['route'].weights)
    r = e.decide('fixture', strict=True)
    assert r.is_ambiguous, {'still_calibrated': e.conformal_predictors['route'].is_calibrated, 'set': r.conformal_sets}


def test_strict_regression_update_invalidates_previous_interval_calibration():
    schema = DecisionSchema(schema_name='regression', fields={'score': ScoreField(min_value=0, max_value=10)})
    e = ReflexEngine(schema, dimension=16, backend='numpy', strict_mode=True, use_cache=False)
    h = e.model.heads['score']
    h.set_weights(np.zeros_like(h.weights), np.zeros_like(h.biases))
    before = h.weights.copy()
    cp = e.regression_conformal_predictors['score']
    cp.residuals = np.array([.01] * 100)
    cp.is_calibrated = True
    e.learn_from_tier2('fixture', {'score': 6.0})
    assert not np.array_equal(before, h.weights)
    result = e.decide('fixture', strict=True)
    assert result.is_ambiguous, {'still_calibrated': cp.is_calibrated, 'interval': result.conformal_sets}


def test_exported_bytes_match_recorded_promotion_artifact(tmp_path):
    client, questions = make_promoted_client()
    expected = client.last_promotion_report.artifact_digest
    path = tmp_path / 'candidate.s1m'
    assert client.export_model(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, {'promotion_artifact': expected, 'exported_artifact': actual}


def test_serialization_preserves_strict_prediction_set_at_boundary():
    # Construct valid APS calibration rows close to the runtime cutoff. This
    # isolates float64 -> float32 quantization, not real-world model quality.
    e = engine(p=.9)
    e.use_cache = False
    p = e.decide('boundary', strict=True, record_receipt=False).probabilities['route']['north']
    q = p + 1e-7 + 1e-9
    cp = e.conformal_predictors['route']
    cp.calibrate(np.tile([q, 1-q], (100, 1)), ['north'] * 100)
    before = e.decide('boundary', strict=True, record_receipt=False)
    assert before.is_ambiguous
    head = e.model.heads['route']
    ch = CompiledHeadWeights(field_name='route', field_type='choice', weights=head.weights.copy(),
          biases=head.biases.copy(), options=tuple(e.schema.fields['route'].options),
          temperature=e.calibrators['route'].temperature, calibration_scores=tuple(cp.calibration_scores))
    cm = CompiledSystemOneModel(e.schema, heads={'route': ch}, dimension=16, backend='numpy', use_cache=False)
    restored = CompiledSystemOneModel.from_bytes(cm.to_bytes(), backend='numpy')
    restored.use_cache = False
    loaded_engine = ReflexEngine(restored.schema, model=restored, backend='numpy', use_cache=False)
    after = loaded_engine.decide('boundary', strict=True, record_receipt=False)
    assert after.conformal_sets == before.conformal_sets, {
        'live_threshold': q, 'reloaded_threshold': float(loaded_engine.conformal_predictors['route'].calibration_scores[0]),
        'live_set': before.conformal_sets, 'reloaded_set': after.conformal_sets,
        'live_ambiguous': before.is_ambiguous, 'reloaded_ambiguous': after.is_ambiguous}


def prediction_fixture():
    return SimpleNamespace(decide=lambda *a, **kw: SimpleNamespace(values={'route': 'north'}))


def test_shared_lineage_is_not_overridden_by_distinct_group_ids():
    rows = [{'state': f'event-{i}', 'group_id': f'g-{i}', 'lineage_id': 'same-source-event',
             'answers': {'route': 'north'}} for i in range(100)]
    r = compat.evaluate_promotion_eligibility(prediction_fixture(), rows, Route())
    assert not r.is_eligible, r


def test_repeating_one_successful_group_cannot_dominate_group_acceptance():
    rows = [{'state': f'good-{i}', 'group_id': 'one-successful-unit', 'answers': {'route': 'north'}} for i in range(1000)]
    rows += [{'state': f'bad-{i}', 'group_id': f'failed-unit-{i}', 'answers': {'route': 'south'}} for i in range(19)]
    r = compat.evaluate_promotion_eligibility(prediction_fixture(), rows, Route())
    assert not r.is_eligible, {'successful_groups': 1, 'groups': 20, 'report': r}


def test_minimum_validation_samples_uses_scored_independent_units():
    rows = [{'state': f'unit-{i}-row-{j}', 'group_id': f'unit-{i}', 'answers': {'route': 'north'}}
            for i in range(20) for j in range(5)]
    policy = compat.PromotionPolicy(min_validation_samples=100)
    r = compat.evaluate_promotion_eligibility(prediction_fixture(), rows, Route(), policy=policy)
    assert not r.is_eligible, {'independent_units': 20, 'required_units': 100, 'report': r}


def test_generic_action_record_is_not_a_signed_authorization(tmp_path):
    trusted = Ed25519PrivateKey.generate().public_key()
    with ActionLedger(tmp_path/'ledger.db') as ledger:
        ledger.record_action('action', {'note': 'proposal only; no permission or signature'}, action_id='not-authorized')
        with pytest.raises((LedgerError, ValueError)):
            ledger.record_execution_outcome(action_id='not-authorized', receipt_digest='f'*64,
                  status='SUCCEEDED', trusted_public_key=trusted)


def test_control_matching_trusted_incoming_receipt_is_accepted(tmp_path):
    key = Ed25519PrivateKey.generate()
    r = create_decision_receipt(schema_name='test', schema_digest='a'*64, prompt='fixture', values={'ok': True},
        confidences={'ok': 1.0}, conformal_sets={'ok': ['True']}, probabilities={'ok': {'True': 1.0}},
        latency_ms=.1, is_ambiguous=False, signing_key=key)
    with ActionLedger(tmp_path/'ledger.db') as ledger:
        ledger.record_decision_receipt(r, trusted_public_key=key.public_key())
        assert len(ledger.entries()) == 1
        assert ledger.verify_integrity(trusted_public_key=key.public_key())


def test_control_sufficient_scored_independent_groups_can_promote():
    rows = [{'state': f'independent-case-{i}', 'group_id': f'g{i}', 'answers': {'route': 'north'}} for i in range(100)]
    r = compat.evaluate_promotion_eligibility(prediction_fixture(), rows, Route())
    assert r.is_eligible, r
