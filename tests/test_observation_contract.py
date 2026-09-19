"""Regressions for observed teaching, reuse, and the supported Jev SDK contract."""
import asyncio
import json
from pathlib import Path
import runpy
import socket

import numpy as np
import pytest

from system1.calibration import ConformalPredictor
from system1.compat.typesafe import (
    AsyncClient, Choice, Client, Noul, NoulCriteria, PromotionPolicy, Score,
    TypeSafeResponse, _build_dynamic_schema, evaluate_promotion_eligibility,
)
from system1.compiler import CompiledSystemOneModel, SystemOneCompiler
from system1.core.embeddings import HybridProjector
from system1.core.model import DeterministicSemanticProjector
from system1.engine import SystemOneEngine


ROOT = Path(__file__).resolve().parents[1]


def test_observe_disconnect_and_reload(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail('Offline lifecycle contacted the network')
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    monkeypatch.setattr(socket, 'create_connection', no_network)
    demo = runpy.run_path(str(ROOT / 'examples/observe_routing.py'))
    report = demo['run'](tmp_path)
    assert report['promoted']
    assert report['promotion']['wilson_lower_bound'] >= .8
    assert report['promotion']['local_acceptance_rate'] >= .8
    assert report['correct'] == report['local_evaluations'] == 40
    assert report['accepted'] >= 32
    assert report['accepted_errors'] == report['teacher_calls_after_takeover'] == 0
    assert report['reload_identical']
    saved = CompiledSystemOneModel.load(tmp_path / 'routing.s1m')
    assert saved.metadata['sample_counts']['queue']['generated'] == 0
    assert saved.metadata['typesafe_strict_mode'] is True
    assert report == json.loads((tmp_path / 'report.json').read_text())


@pytest.mark.parametrize('projector', [
    DeterministicSemanticProjector(dimension=64),
    HybridProjector(dimension=64, alpha=.35, seed=123),
])
def test_saved_skill_keeps_full_inference_state(tmp_path, projector):
    q = {'route': Choice('Choose a queue', criteria={'billing': 'invoice payment', 'access': 'password login'})}
    schema = _build_dynamic_schema(q)
    fit = {'route': [('invoice refund', 'billing'), ('card payment', 'billing'),
                     ('password reset', 'access'), ('account login', 'access')]}
    calibration = {'route': [(f'payment receipt {i}', 'billing') if i % 2 else
                             (f'login credential {i}', 'access') for i in range(48)]}
    skill = SystemOneCompiler(schema, dimension=64, projector=projector).compile(
        exemplars=fit, calibration_exemplars=calibration, augment=False,
    )
    path = tmp_path / 'skill.s1m'
    skill.save(path)
    before = Client(compiled_model=skill)
    after = Client(model_path=path)
    for state in ['refund credit card', 'my login password', 'unfamiliar printer problem']:
        a = before.systemone(state, q, record_receipt=False)
        b = after.systemone(state, q, record_receipt=False)
        assert a.answers == b.answers
        assert a.is_ambiguous == b.is_ambiguous
    with pytest.raises(ValueError, match='projector'):
        CompiledSystemOneModel.load(path, projector=DeterministicSemanticProjector(dimension=128))
    with pytest.raises(ValueError, match='match these questions'):
        after.systemone('hello', {'route': Choice('DIFFERENT POLICY', criteria=['billing', 'access'])})


def test_sdk_objects_contexts_json_and_ordinal_distribution():
    questions = {
        'route': Choice(instructions={'task': 'route'}, criteria={'billing': ['invoice', 'refund'], 'access': None}),
        'urgent': Noul(instructions=['Is urgent?'], criteria=NoulCriteria(true='urgent', false='routine')),
        'priority': Score(instructions='Priority', criteria=['low', {'level': 'medium'}, 'high']),
    }
    state = {'request': ['invoice', {'reason': 'duplicate'}]}
    with Client(model='local-skill') as client:
        response = client.systemone(state, questions)
        assert response.model == 'local-skill'
        assert response.state == state
        assert response.choices.route.choice in questions['route'].options
        assert 0 <= response.nouls.urgent.noul <= 1
        score = response.scores.priority
        assert set(score.probabilities) == set(score.legend) == {0, 1, 2}
        assert sum(score.probabilities.values()) == pytest.approx(1)
        assert score.score == pytest.approx(sum(k * p for k, p in score.probabilities.items()))
        assert score.legend[1] == {'level': 'medium'}
        assert _build_dynamic_schema(questions).schema_digest() == _build_dynamic_schema(
            {name: question.to_dict() for name, question in questions.items()}).schema_digest()
        # JSON key order must not change local features.
        assert client.systemone(dict(reversed(list(state.items()))), questions).answers == response.answers
    with pytest.raises(RuntimeError, match='closed'):
        client.systemone(state, questions)

    async def check_async():
        async with AsyncClient(model='async-skill') as client:
            result = await client.systemone(state, questions)
            assert result.model == 'async-skill'
        with pytest.raises(RuntimeError, match='closed'):
            await client.systemone(state, questions)
    asyncio.run(check_async())


def test_unsupported_sdk_options_fail_explicitly():
    with pytest.raises(NotImplementedError, match='http_client'):
        Client(http_client=object())
    with Client() as client:
        with pytest.raises(NotImplementedError, match='response_model'):
            client.systemone('hello', {'q': Noul('Is this a greeting?')}, response_model=dict)


def test_repeated_observations_do_not_crash_or_promote():
    def teacher(state, questions):
        return {'answers': {'q': {'type': 'choice', 'choice': 'yes'}}}
    client = Client(mode='auto_cutover', baseline_handler=teacher, cutover_threshold=3)
    for _ in range(8):
        answer = client.systemone('same request', {'q': Choice('Q', criteria=['yes', 'no'])})
        assert answer.choices.q.choice == 'yes'
    assert not client.is_cutover
    assert client.last_promotion_report.rejection_reasons


def test_agreement_without_usable_decisions_does_not_promote():
    q = {'q': Choice('Q', criteria=['yes', 'no'])}
    schema = _build_dynamic_schema(q)
    engine = SystemOneEngine(schema, strict_mode=True)
    history = [{'state': f'yes request {i}', 'answers': {'q': engine.decide(f'yes request {i}').values['q']}}
               for i in range(40)]
    report = evaluate_promotion_eligibility(engine, history, schema, PromotionPolicy(min_local_acceptance=.8))
    assert report.agreement_rate == 1
    assert report.local_acceptance_rate == 0
    assert not report.is_eligible


def test_strict_aps_inverts_scores_and_keeps_ties():
    cp = ConformalPredictor('q', ['A', 'B', 'C'])
    cp.calibrate(np.tile([.8, .15, .05], (40, 1)), ['A'] * 40)
    assert cp.predict_set(np.array([.7, .2, .1]), strict=True).prediction_set == ('A',)
    empty = cp.predict_set(np.array([.9, .06, .04]), strict=True)
    assert empty.is_empty and empty.needs_escalation
    tied = cp.predict_set(np.array([.8, .15, .05]), strict=True)
    assert tied.prediction_set == ('A',)
    # Insufficient calibration evidence must retain the entire feasible domain.
    assert set(cp.predict_set(np.array([1., 0., 0.]), alpha=.001, strict=True).prediction_set) == {'A', 'B', 'C'}


def test_strict_aps_empirical_coverage():
    rng = np.random.default_rng(102)
    probs = rng.dirichlet([2, 2, 2], size=5000)
    labels = [rng.choice(['A', 'B', 'C'], p=row) for row in probs]
    cp = ConformalPredictor('q', ['A', 'B', 'C'])
    cp.calibrate(probs[:1000], labels[:1000])
    covered = sum(label in cp.predict_set(row, alpha=.1, strict=True).prediction_set
                  for row, label in zip(probs[1000:], labels[1000:]))
    assert covered / 4000 >= .88


def test_failed_comparison_never_manufactures_speedup(monkeypatch):
    client = Client(zero_egress=False)
    def unavailable(*args, **kwargs):
        raise OSError('teacher unavailable')
    monkeypatch.setattr(client, 'call_real_api', unavailable)
    with pytest.raises(OSError, match='teacher unavailable'):
        client.compare('hello', {'q': Choice('Q', criteria=['yes', 'no'])})


def test_http_preserves_json_and_ordinal_wire_contract(monkeypatch):
    import io
    import urllib.request
    seen = []
    def urlopen(request, timeout):
        seen.append(json.loads(request.data))
        return io.BytesIO(json.dumps({'answers': {'s': {'type': 'score', 'score': 1.2,
            'probabilities': {'0': .1, '1': .6, '2': .3},
            'legend': {'0': 'low', '1': 'medium', '2': 'high'}}}}).encode())
    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    with Client(mode='passthrough', zero_egress=False, api_key='fake-test-key', model='chosen-model') as client:
        result = client.systemone({'items': [1, 2]}, {'s': Score(criteria=['low', 'medium', 'high'])})
    assert seen == [{'model': 'chosen-model', 'state': {'items': [1, 2]},
                     'questions': {'s': {'type': 'score', 'instructions': '', 'criteria': ['low', 'medium', 'high']}}}]
    assert set(result.scores.s.probabilities) == {0, 1, 2}
    assert result.local_execution is False


def test_connected_lineage_and_normalized_prompts_stay_together():
    from system1.compat.typesafe import partition_cutover_history
    history = [
        {'state': 'Invoice request', 'group_id': 'first', 'answers': {'q': 'yes'}},
        {'state': '  INVOICE  REQUEST ', 'group_id': 'second', 'lineage_id': 'shared', 'answers': {'q': 'no'}},
        {'state': 'A followup', 'group_id': 'third', 'lineage_id': 'shared', 'answers': {'q': 'yes'}},
    ] + [{'state': f'unrelated {i}', 'answers': {'q': 'yes'}} for i in range(7)]
    split = partition_cutover_history(history)
    split.assert_disjoint()
    folds = [split.train_history, split.calib_history, split.val_history]
    assert any(all(row in fold for row in history[:3]) for fold in folds)


def test_comparison_local_side_stays_local_in_passthrough_mode(monkeypatch):
    client = Client(mode='passthrough', zero_egress=False)
    calls = []
    def teacher(*args, **kwargs):
        calls.append(True)
        return TypeSafeResponse({'answers': {'q': {'type': 'noul', 'noul': .5}},
                                 'usage': {'total_tokens': 10}, 'local_execution': False}), 5., 100
    monkeypatch.setattr(client, 'call_real_api', teacher)
    result = client.compare('hello', {'q': Noul('Is this a greeting?')})
    assert result.local_response.local_execution
    assert len(calls) == 1
    assert result.is_live


@pytest.mark.parametrize('script,function', [
    ('deep_jev_benchmark.py', 'run_benchmark'),
    ('killer_use_cases_live_test.py', 'run_use_case_tests'),
])
def test_all_failed_benchmarks_report_no_speedup(script, function, monkeypatch, capsys):
    import urllib.error
    import urllib.request
    def unavailable(*args, **kwargs):
        raise urllib.error.URLError('offline test')
    monkeypatch.setattr(urllib.request, 'urlopen', unavailable)
    demo = runpy.run_path(str(ROOT / 'examples' / script))
    assert demo[function]('fake-test-key') == []
    assert 'no speedup can be reported' in capsys.readouterr().out
