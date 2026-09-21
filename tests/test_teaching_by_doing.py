"""Check demonstrated choices, corrections, isolation and portable skill replay."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.teaching_by_doing.app import ROOT, Session, fingerprint, observation
from system1 import CompiledSystemOneModel
from system1.compiler import SystemOneCompiler
from system1.core.text import TfidfProjector


def decisions(session):
    return [(r['id'], r['folder'], r['review'], r['possibilities'])
            for r in session.preview()['rows']]


def test_three_manual_choices_teach_but_do_not_claim_readiness(tmp_path):
    session = Session(tmp_path)
    with pytest.raises(ValueError, match='each folder'):
        session.teach()
    for doc in session.documents.values():
        if doc['purpose'] == 'teach' and doc['folder'] not in {r['folder'] for r in session.records.values()}:
            session.record(doc['id'], doc['folder'])
    session.teach()
    assert session.teaching['fit'] == 3 and session.teaching['checks'] == 0
    assert session.preview()['filed'] == 0
    assert len(session.records) == 3


def test_correction_replaces_choice_invalidates_skill_and_survives_restart(tmp_path):
    session = Session(tmp_path)
    session.sample_session()
    session.teach()
    original = decisions(session)
    session.record('teach-01', 'People')
    assert len(session.records) == 60
    assert session.snapshot()['sources'] == {'manual': 1, 'sample': 59}
    assert not session.snapshot()['taught']
    with pytest.raises(ValueError, match='current choices'):
        session.predict('Invoice', 'Payment is due')
    restored = Session(tmp_path)
    assert restored.engine is None and restored.records['teach-01']['folder'] == 'People'
    restored.sample_session()
    assert restored.records['teach-01']['folder'] == 'People'
    restored.teach()
    assert decisions(restored) != original
    assert decisions(Session(tmp_path)) == decisions(restored)
    restored.record('teach-01', 'Finance')
    restored.teach()
    assert decisions(restored) == original
    restored.remove('teach-01')
    assert len(Session(tmp_path).records) == 59 and restored.engine is None


def test_fresh_batch_cannot_become_lessons_or_calibration(tmp_path):
    session = Session(tmp_path)
    fingerprints = [fingerprint(observation(d)) for d in session.documents.values()]
    assert len(fingerprints) == len(set(fingerprints))
    for document_id in ('evaluate-01', 'challenge-01'):
        with pytest.raises(ValueError):
            session.record(document_id, 'Finance')
    assert not session.records
    session.sample_session()
    session.teach()
    digest = session.digest()
    report = session.evaluate()
    assert session.digest() == digest and len(session.records) == 60
    assert session.teaching['fit'] == 18 and session.teaching['checks'] == 42
    assert report['cases'] == 30 and report['challenge_cases'] == 6
    assert all(r['expected_review'] for r in report['rows'] if r['purpose'] == 'challenge')
    assert all('folder' not in d for d in session.snapshot()['documents'])


def test_exported_actions_and_lessons_reproduce_the_skill(tmp_path):
    session = Session(tmp_path)
    session.sample_session()
    session.teach()
    exported = json.loads(json.dumps(session.export_lessons()))
    fit = exported['exemplars']['folder']
    projector = TfidfProjector.fit([text for text, _ in fit], max_features=1024)
    model = SystemOneCompiler(session.engine.schema, projector=projector, regularization=.1).compile(
        exported['exemplars'], augment=False,
        calibration_exemplars=exported['calibration_exemplars'])
    expected = decisions(session)
    session.engine = Session.load_engine(model)
    assert decisions(session) == expected
    saved = json.loads((tmp_path/'actions.json').read_text())
    saved['actions'][0]['input'] += ' hidden answer'
    (tmp_path/'actions.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='visible document'):
        Session(tmp_path)


def test_saved_skill_needs_no_catalog_or_teacher(tmp_path, monkeypatch):
    session = Session(tmp_path)
    session.sample_session()
    session.teach()
    doc = session.documents['evaluate-01']
    expected = session.predict(doc['title'], doc['excerpt'])
    model = CompiledSystemOneModel.load(tmp_path/'filing.s1m')
    engine = Session.load_engine(model)
    def forbidden(*args, **kwargs):
        raise AssertionError('Inference attempted to read lessons or teach')
    monkeypatch.setattr(Path, 'read_text', forbidden)
    monkeypatch.setattr(SystemOneCompiler, 'compile', forbidden)
    result = engine.decide(observation(doc), record_receipt=False)
    assert result.values['folder'] == expected['folder']
    assert result.is_ambiguous == expected['review']


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
