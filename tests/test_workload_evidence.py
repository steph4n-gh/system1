"""Public workload evidence must preserve labels, separation, and minority-class errors."""
import hashlib
import io
import json
from pathlib import Path
import runpy
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_public_source_preparation_preserves_test_and_groups_duplicates(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.quality import prepare_workloads as prepare

    clinc = {'train': [['Forecast please', 'weather'], ['Set a timer', 'timer']],
             'val': [['Calendar please', 'calendar']],
             'test': [['FORECAST please!', 'weather'], ['Play some music', 'play_music'], ['Unrelated', 'unsupported']]}
    sms = 'ham\tHello friend\nham\tHELLO friend!\nspam\tClaim a reward\nham\tConflicting label\nspam\tConflicting label\n'
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, 'w') as archive:
        archive.writestr('SMSSpamCollection', sms)
    raw = {'clinc.json': json.dumps(clinc).encode(), 'sms.zip': zipped.getvalue()}
    cache = tmp_path / '.system1/workload-sources'
    cache.mkdir(parents=True)
    for name, content in raw.items():
        (cache / name).write_bytes(content)
    monkeypatch.setattr(prepare, 'ROOT', tmp_path)
    monkeypatch.setattr(prepare, 'OUTPUT', tmp_path / 'output')
    monkeypatch.setattr(prepare, 'SOURCES', {name: ('unused', hashlib.sha256(content).hexdigest()) for name, content in raw.items()})
    prepare.prepare()
    assistant = json.loads((tmp_path / 'output/assistant_commands.json').read_text())
    assert [(r['prompt'], r['label']) for r in assistant['evaluate']] == [tuple(r) for r in clinc['test'][:2]]
    assert [r['prompt'] for r in assistant['teach']] == ['Set a timer']
    spam = json.loads((tmp_path / 'output/sms_triage.json').read_text())
    rows = [r for split in ('teach', 'calibration', 'evaluate') for r in spam[split]]
    assert {r['prompt'] for r in rows} == {'Hello friend', 'Claim a reward'}
    assert len({r['group'] for r in rows}) == len(rows)
    assert spam['conflicting_groups_removed'] == 1
    (cache / 'sms.zip').write_bytes(b'changed upstream data')
    with pytest.raises(AssertionError, match='changed'):
        prepare.prepare()


def test_workload_metrics_expose_spam_misses_despite_high_overall_accuracy(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'benchmarks/quality'))
    runner = runpy.run_path(str(ROOT / 'benchmarks/quality/evaluate_workloads.py'))
    rows = [{'label': 'ham', 'prediction': 'ham', 'needs_review': False} for _ in range(99)]
    rows.append({'label': 'spam', 'prediction': 'ham', 'needs_review': False})
    metrics = runner['metrics'](rows, ['ham', 'spam'])
    assert metrics['accepted_accuracy'] == .99
    assert metrics['accepted_errors'] == 1
    assert metrics['per_class']['spam']['raw_recall'] == 0
    assert metrics['per_class']['spam']['accepted_precision'] is None
