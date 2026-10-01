"""Check explicit corrections, candidate review, cohort isolation and replay."""
from hashlib import sha256
import json
from pathlib import Path
import sys
from threading import Event, Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.teaching_by_doing.app import ROOT, Session, fingerprint, observation
from examples.teaching_by_doing.__main__ import main
from system1 import CompiledSystemOneModel
from system1.compiler import SystemOneCompiler
from system1.core.text import TfidfProjector


def decisions(session):
    return [(r['id'], r['folder'], r['review'], r['possibilities'])
            for r in session.preview()['rows']]


def approved_sample(path):
    session = Session(path)
    session.sample_session()
    report = session.teach()['candidate']
    assert report['passed'] and session.engine is None
    session.adopt()
    return session


def test_three_manual_choices_need_separate_calibration_and_adoption(tmp_path):
    session = Session(tmp_path)
    with pytest.raises(ValueError, match='each folder'):
        session.teach()
    for doc in session.documents.values():
        if doc['purpose'] == 'teach' and doc['folder'] not in {r['folder'] for r in session.records.values()}:
            session.record(doc['id'], doc['folder'])
    with pytest.raises(ValueError, match='calibrate'):
        session.teach()
    assert len(session.records) == 3
    assert session.engine is None
    with pytest.raises(ValueError, match='adopt'):
        session.predict('Invoice', 'Payment is due')


def test_candidate_requires_explicit_adoption_and_survives_restart(tmp_path):
    session = Session(tmp_path)
    session.sample_session()
    report = session.teach()['candidate']
    assert report['passed'] and report['incumbent'] is None
    assert report['counts'] == {'teach': 18, 'calibrate': 42, 'evaluate': 30}
    assert report['candidate']['count'] == 30
    assert not session.snapshot()['taught']
    assert not session.lifecycle.current_path.exists()
    restored = Session(tmp_path)
    assert not restored.snapshot()['candidate']['stale']
    restored.adopt()
    assert restored.snapshot()['candidate']['adopted']
    assert decisions(Session(tmp_path)) == decisions(restored)


def test_correction_keeps_approved_skill_and_stales_pending_candidate(tmp_path):
    session = approved_sample(tmp_path)
    original = decisions(session)
    approved = session.lifecycle.current_path.read_bytes()
    session.teach()
    session.record('teach-01', 'People')
    assert len(session.records) == 60
    assert session.snapshot()['sources'] == {'manual': 1, 'sample': 59}
    assert session.snapshot()['taught']
    assert session.snapshot()['candidate']['stale']
    assert decisions(session) == original
    with pytest.raises(ValueError, match='stale'):
        session.adopt()
    restored = Session(tmp_path)
    assert restored.records['teach-01']['folder'] == 'People'
    restored.sample_session()
    assert restored.records['teach-01']['folder'] == 'People'
    assert decisions(restored) == original
    assert restored.lifecycle.current_path.read_bytes() == approved
    restored.record('teach-01', 'Finance')
    report = restored.teach()['candidate']
    assert report['incumbent'] is not None and report['regressions'] == 0
    restored.adopt()
    assert decisions(restored) == original
    restored.remove('teach-01')
    assert len(Session(tmp_path).records) == 59
    assert decisions(restored) == original


def test_failed_candidate_and_failed_build_preserve_approved_skill(tmp_path, monkeypatch):
    session = approved_sample(tmp_path)
    original = decisions(session)
    approved = session.lifecycle.current_path.read_bytes()
    # Keep all three fitting labels but deliberately reverse the sample policy.
    swap = {'Finance': 'People', 'People': 'Projects', 'Projects': 'Finance'}
    for doc in session.documents.values():
        if doc['purpose'] == 'teach':
            session.record(doc['id'], swap[doc['folder']])
    report = session.teach()['candidate']
    assert not report['passed']
    assert report['candidate']['raw_accuracy'] < report['thresholds']['min_accuracy']
    with pytest.raises(ValueError, match='pass'):
        session.adopt()
    assert session.lifecycle.current_path.read_bytes() == approved
    assert decisions(Session(tmp_path)) == original
    def fail_compile(*args, **kwargs):
        raise ValueError('Compilation failed')
    monkeypatch.setattr(SystemOneCompiler, 'compile', fail_compile)
    with pytest.raises(ValueError, match='Compilation failed'):
        session.teach()
    assert session.lifecycle.current_path.read_bytes() == approved
    assert decisions(session) == original


def test_reserved_samples_cannot_become_lessons_or_calibration(tmp_path):
    session = Session(tmp_path)
    fingerprints = [fingerprint(observation(d)) for d in session.documents.values()]
    assert len(fingerprints) == len(set(fingerprints))
    for document_id in ('evaluate-01', 'challenge-01'):
        doc = session.documents[document_id]
        with pytest.raises(ValueError):
            session.record(document_id, 'Finance')
        with pytest.raises(ValueError, match='reserved'):
            session.record_text(doc['title'].upper(), doc['excerpt'], 'Finance')
    assert not session.records
    session.sample_session()
    session.teach()
    session.adopt()
    digest = session.digest()
    report = session.evaluate()
    assert session.digest() == digest and len(session.records) == 60
    assert session.teaching['fit'] == 18 and session.teaching['calibration'] == 42
    assert report['cases'] == 30 and report['challenge_cases'] == 6
    assert 'Recurring' in report['evidence_scope']
    assert all(r['expected_review'] for r in report['rows'] if r['purpose'] == 'challenge')
    assert all('folder' not in d for d in session.snapshot()['documents'])
    splits = {split: {fingerprint(row['input']) for row in session.lifecycle.records if row['split'] == split}
              for split in ('teach', 'calibrate', 'evaluate')}
    assert not splits['teach'] & splits['calibrate']
    assert not splits['teach'] & splits['evaluate']
    assert not splits['calibrate'] & splits['evaluate']


def test_own_text_needs_explicit_label_and_replaces_corrections(tmp_path):
    session = approved_sample(tmp_path)
    text = {'title': 'September supplier billing', 'excerpt': 'Approve payment for the copper supplier invoice.'}
    before = session.digest()
    session.predict(**text)
    assert session.digest() == before
    with pytest.raises(ValueError, match='folder'):
        session.record_text(**text, folder='')
    session.record_text(**text, folder='People')
    own_id = 'own-' + fingerprint(observation(text))
    assert len(session.records) == 61
    assert session.records[own_id]['folder'] == 'People'
    session.record_text(**text, folder='Finance')
    assert len(session.records) == 61
    session.record(own_id, 'Projects')
    assert session.records[own_id]['folder'] == 'Projects'
    restored = Session(tmp_path)
    assert restored.records[own_id]['source'] == 'manual'
    assert any(row['input'] == observation(text) and row['label'] == 'Projects' and row['split'] == 'teach'
               for row in restored.lifecycle.records)
    restored.remove(own_id)
    assert own_id not in Session(tmp_path).records
    assert all(row['input'] != observation(text) for row in restored.lifecycle.records)


def test_exported_actions_and_lessons_reproduce_the_skill(tmp_path):
    session = approved_sample(tmp_path)
    exported = json.loads(json.dumps(session.export_lessons()))
    fit = exported['exemplars']['folder']
    projector = TfidfProjector.fit([text for text, _ in fit], max_features=1024)
    model = SystemOneCompiler(session.engine.schema, projector=projector, regularization=.1).compile(
        exported['exemplars'], augment=False,
        calibration_exemplars=exported['calibration_exemplars'])
    engine = Session.load_engine(model)
    for row in session.preview()['rows']:
        result = engine.decide(observation(row), record_receipt=False)
        assert result.values['folder'] == row['folder']
        assert result.is_ambiguous == row['review']
        assert result.conformal_sets['folder'] == row['possibilities']
    saved = json.loads((tmp_path/'actions.json').read_text())
    saved['actions'][0]['input'] += ' hidden answer'
    (tmp_path/'actions.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='visible document'):
        Session(tmp_path)


def test_saved_skill_needs_no_catalog_or_teacher(tmp_path, monkeypatch):
    session = approved_sample(tmp_path)
    doc = session.documents['evaluate-01']
    expected = session.predict(doc['title'], doc['excerpt'])
    model = CompiledSystemOneModel.load(session.lifecycle.current_path)
    engine = Session.load_engine(model)
    def forbidden(*args, **kwargs):
        raise AssertionError('Inference attempted to read lessons or teach')
    monkeypatch.setattr(Path, 'read_text', forbidden)
    monkeypatch.setattr(SystemOneCompiler, 'compile', forbidden)
    result = engine.decide(observation(doc), record_receipt=False)
    assert result.values['folder'] == expected['folder']
    assert result.is_ambiguous == expected['review']


def test_legacy_skill_loads_and_historical_evidence_is_not_rewritten(tmp_path):
    with zipfile.ZipFile(ROOT/'results/filing-evidence.zip') as archive:
        for name in ('actions.json', 'filing.s1m', 'report.json'):
            (tmp_path/name).write_bytes(archive.read(name))
    original = (tmp_path/'filing.s1m').read_bytes()
    historical = (tmp_path/'report.json').read_bytes()
    session = Session(tmp_path)
    assert session.lifecycle.current_path.read_bytes() == original
    old = decisions(session)
    session.record('teach-01', 'People')
    assert decisions(Session(tmp_path)) == old
    session.evaluate()
    assert (tmp_path/'report.json').read_bytes() == historical
    assert (tmp_path/'filing.s1m').read_bytes() == original
    assert (tmp_path/'preview-report.json').exists()


def test_legacy_skill_must_use_the_filing_schema(tmp_path):
    from system1 import ChoiceField, DecisionSchema
    class OtherSchema(DecisionSchema):
        folder = ChoiceField(options=['One', 'Two'])
    model = SystemOneCompiler(OtherSchema).compile(
        {'folder': [('first example', 'One'), ('second example', 'Two')]}, augment=False)
    model.save(tmp_path/'filing.s1m')
    with pytest.raises(ValueError, match='filing schema'):
        Session(tmp_path)
    assert not (tmp_path/'current.s1m').exists()


def test_reopen_refuses_to_overwrite_external_lesson_changes(tmp_path):
    session = approved_sample(tmp_path)
    text = observation(session.documents['teach-01'])
    session.lifecycle.record(text, 'People', split='teach', source='human')
    before = session.lifecycle.session_path.read_bytes()
    with pytest.raises(ValueError, match='no records were overwritten'):
        Session(tmp_path)
    assert session.lifecycle.session_path.read_bytes() == before
    session.lifecycle.record('A separate CLI lesson', 'Projects')
    before = session.lifecycle.session_path.read_bytes()
    with pytest.raises(ValueError, match='no records were overwritten'):
        Session(tmp_path)
    assert session.lifecycle.session_path.read_bytes() == before


def test_demo_cli_review_needs_explicit_adopt(tmp_path, monkeypatch, capsys):
    review = tmp_path/'review'
    monkeypatch.setattr(sys, 'argv', ['demo', '--evaluate', '--output-dir', str(review)])
    main()
    assert (review/'assessment.json').exists()
    assert not (review/'current.s1m').exists()
    assert 'approved skill unchanged' in capsys.readouterr().out
    adopted = tmp_path/'adopted'
    monkeypatch.setattr(sys, 'argv', ['demo', '--evaluate', '--adopt', '--output-dir', str(adopted)])
    main()
    assert (adopted/'current.s1m').exists()
    assert (adopted/'preview-report.json').exists()
    before = (adopted/'current.s1m').read_bytes()
    with pytest.raises(SystemExit):
        main()
    assert (adopted/'current.s1m').read_bytes() == before


def test_published_evidence_replays_every_document(tmp_path):
    with zipfile.ZipFile(ROOT/'results/filing-evidence.zip') as archive:
        for name, digest in json.loads(archive.read('manifest.json')).items():
            assert sha256(archive.read(name)).hexdigest() == digest
        report = json.loads(archive.read('report.json'))
        assert report == json.loads((ROOT/'results/report.json').read_text())
        (tmp_path/'filing.s1m').write_bytes(archive.read('filing.s1m'))
        assert sha256((tmp_path/'filing.s1m').read_bytes()).hexdigest() == report['model_sha256']
        model = CompiledSystemOneModel.load(tmp_path/'filing.s1m')
        engine = Session.load_engine(model)
        actions = json.loads(archive.read('actions.json'))['actions']
        assert {a['source'] for a in actions} == {'sample'}
        assert len(actions) == report['teaching']['fit'] + report['teaching']['checks']
        for row in report['rows']:
            result = engine.decide(observation(row), record_receipt=False)
            assert result.values['folder'] == row['folder']
            assert result.is_ambiguous == row['review']
            assert result.conformal_sets['folder'] == row['possibilities']
        regular = [r for r in report['rows'] if r['purpose'] == 'evaluate']
        assert sum(r['folder'] == r['expected'] for r in regular) == report['correct']
        assert sum(not r['review'] for r in regular) == report['accepted']
        assert sum(not r['review'] and r['correct'] for r in regular) == report['accepted_correct']


def test_workspace_solver_reasons_and_retained_revisions(tmp_path):
    session = Session(tmp_path, choice_solver='logistic')
    session.sample_session()
    assert session.teach()['settings']['choice_solver'] == 'logistic'
    session.adopt()
    original = session.lifecycle.current_path.read_bytes()
    first_revision = session.snapshot()['history'][0]['revision']
    prediction = session.predict('xyzzynonexistent', 'blorbunknown splatunknown')
    assert prediction['review'] and 'unknown_vocabulary' in prediction['review_reasons']
    assert prediction['revision'] == first_revision
    session.record_text('Supplier invoice extra lesson', 'Please approve payment for the supplier invoice.', 'Finance')
    assert session.teach()['candidate']['passed']
    session.adopt()
    assert len(session.snapshot()['history']) == 2
    before = session.digest()
    restored = session.rollback(first_revision)
    assert restored['history'][0]['current']
    assert session.lifecycle.current_path.read_bytes() == original
    assert session.digest() == before and restored['candidate']['stale']
    assert Session(tmp_path).lifecycle.current_path.read_bytes() == original
    with pytest.raises(ValueError, match='differs'):
        Session(tmp_path, choice_solver='ridge')


def test_workspace_refuses_authored_qualification_and_keeps_failed_result(tmp_path):
    session = Session(tmp_path)
    session.sample_session()
    state = session.teach(require_qualification=True)
    assert state['candidate']['passed'] and not state['can_adopt']
    with pytest.raises(ValueError, match='qualification'):
        session.adopt()
    for document_id in ('teach-01', 'evaluate-01', 'challenge-01'):
        with pytest.raises(ValueError, match='sample documents'):
            session.qualify([[observation(session.documents[document_id]), 'Finance']], source='Authored fixture')
    with pytest.raises(ValueError, match='disjoint'):
        session.record_text('External lesson', 'Supplier invoice requires payment.', 'Finance')
        session.teach(require_qualification=True)
        session.qualify([['External lesson\nSupplier invoice requires payment.', 'Finance']], source='Repeated lesson')
    before = session.digest()
    state = session.qualify([['xyzzynonexistent blorbunknown', 'Finance']], source='Synthetic failure fixture')
    assert not state['qualification']['passed'] and not state['can_adopt']
    assert session.digest() == before and not session.lifecycle.current_path.exists()
    with pytest.raises(ValueError, match='frozen'):
        session.qualify([['another brandnew strange fixture', 'People']], source='Attempted replacement')
    assert Session(tmp_path).snapshot()['qualification'] == state['qualification']


def test_http_workspace_qualification_adoption_preview_and_rollback(tmp_path, monkeypatch):
    import examples.teaching_by_doing.app as app
    from http.server import HTTPServer

    ready = Event()
    servers = []
    def local_server(address, handler):
        server = HTTPServer(('127.0.0.1', 0), handler)
        servers.append(server)
        ready.set()
        return server
    monkeypatch.setattr(app, 'HTTPServer', local_server)
    worker = Thread(target=app.serve, args=(tmp_path,), daemon=True)
    worker.start()
    assert ready.wait(10), 'Local workspace did not start'
    server = servers[0]
    base = f'http://127.0.0.1:{server.server_port}'
    def request(path, body=None, host='127.0.0.1:8791'):
        headers = {'Host': host}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers['Content-Type'] = 'application/json'
        with urlopen(Request(base+path, data=data, headers=headers), timeout=10) as response:
            return response.read()
    def api(path, body=None):
        return json.loads(request('/api/'+path, body))
    try:
        html = request('/').decode()
        assert 'Fresh qualification' in html and 'Restore selected revision' in html
        with pytest.raises(HTTPError) as rejected:
            request('/api/sample', {}, host='untrusted.example')
        assert rejected.value.code == 403
        state = api('sample', {})
        assert state['settings']['choice_solver'] == 'ridge'
        state = api('teach', {'require_qualification': True})
        assert state['candidate']['passed'] and not state['can_adopt']
        with pytest.raises(HTTPError) as rejected:
            api('adopt', {})
        assert rejected.value.code == 400

        # Synthetic transport fixture only; the runtime cannot establish
        # independence/representativeness from a supplied JSON file.
        catalog = json.loads((ROOT/'documents.json').read_text())['documents']
        cohort = [{'input': observation(doc)+f'\nTransport fixture {index}', 'label': doc['folder'],
                   'group': f"transport-{doc['id']}-{index}"}
                  for doc in catalog if doc['purpose'] == 'evaluate' for index in range(3)]
        state = api('qualify', {'rows': cohort, 'source': 'Synthetic HTTP transport fixture',
                               'min_accepted_accuracy': .9, 'min_coverage': .7, 'confidence': .9})
        assert state['qualification']['passed'] and state['can_adopt']
        actions_before = api('actions')
        with pytest.raises(HTTPError) as rejected:
            title, excerpt = cohort[0]['input'].split('\n', 1)
            api('record-text', {'title': title, 'excerpt': excerpt, 'folder': cohort[0]['label']})
        assert rejected.value.code == 400
        assert api('actions') == actions_before
        state = api('adopt', {})
        assert state['candidate']['adopted'] and not state['can_adopt']
        first_revision = state['history'][0]['revision']
        approved = request('/api/skill')
        assert sha256(approved).hexdigest() == first_revision
        prediction = api('predict', {'title': 'xyzzynonexistent', 'excerpt': 'blorbunknown'})
        assert prediction['review'] and prediction['revision'] == first_revision
        assert 'unknown_vocabulary' in prediction['review_reasons']
        preview = api('preview', {})
        assert preview['total'] == 36 and all(row['revision'] == first_revision for row in preview['rows'])
        api('record-text', {'title': 'Supplier invoice extra lesson',
                            'excerpt': 'Please approve payment for the supplier invoice.', 'folder': 'Finance'})
        assert api('teach', {})['candidate']['passed']
        state = api('adopt', {})
        assert len(state['history']) == 2
        actions = api('actions')
        state = api('rollback', {'revision': first_revision})
        assert state['candidate']['stale'] and not state['can_adopt']
        assert request('/api/skill') == approved and api('actions') == actions
        with pytest.raises(HTTPError) as rejected:
            api('rollback', {'revision': '../current'})
        assert rejected.value.code == 400
        assert request('/api/skill') == approved
    finally:
        server.shutdown()
        server.server_close()
        worker.join(10)
        assert not worker.is_alive()
