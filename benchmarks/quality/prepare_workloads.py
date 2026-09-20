#!/usr/bin/env python3
"""Download pinned public sources and reproduce the fixed workload splits."""
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / '.system1/workloads/data'
SOURCES = {
    'clinc.json': ('https://raw.githubusercontent.com/clinc/oos-eval/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_full.json',
                   '36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0'),
    'sms.zip': ('https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip',
                '1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3'),
}
ASSISTANT = {
    'weather': 'Weather conditions or forecasts.',
    'translate': 'Translate words or phrases between languages.',
    'timer': 'Set, check, or manage a countdown timer.',
    'calendar': 'Check or manage calendar events and appointments.',
    'calculator': 'Perform a numerical calculation.',
    'play_music': 'Play music or request a song.',
}


def group(prompt):
    normalized = ' '.join(re.findall(r'\w+', prompt.casefold()))
    return hashlib.sha256(normalized.encode()).hexdigest()


def row(prompt, label, source_split, index):
    return {'group': group(prompt), 'prompt': prompt, 'label': label,
            'source_split': source_split, 'source_row': index}


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / '.system1/workload-sources'
    cache.mkdir(parents=True, exist_ok=True)
    raw = {}
    for name, (url, digest) in SOURCES.items():
        path = cache / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=30) as response:
                path.write_bytes(response.read())
        raw[name] = path.read_bytes()
        assert hashlib.sha256(raw[name]).hexdigest() == digest, f'Upstream {name} changed'
    clinc = json.loads(raw['clinc.json'])
    assistant = {'name': 'assistant_commands', 'descriptions': ASSISTANT,
                 'provenance': 'CLINC150, six preselected intents with original labels. Official test is untouched; not all 150 intents or an out-of-scope test.'}
    for split, original in [('teach', 'train'), ('calibration', 'val'), ('evaluate', 'test')]:
        assistant[split] = [row(prompt, label, original, i)
                            for i, (prompt, label) in enumerate(clinc[original]) if label in ASSISTANT]
    # Preserve official evaluation rows; remove cross-split duplicates from earlier splits.
    seen = set()
    removed = {}
    for split in ('evaluate', 'calibration', 'teach'):
        before = assistant[split]
        assistant[split] = [r for r in before if r['group'] not in seen]
        removed[split] = len(before) - len(assistant[split])
        seen.update(r['group'] for r in assistant[split])
    assistant['cross_split_rows_removed'] = removed
    with zipfile.ZipFile(io.BytesIO(raw['sms.zip'])) as archive:
        text = archive.read('SMSSpamCollection').decode('utf-8')
    groups = defaultdict(list)
    for i, line in enumerate(text.splitlines()):
        label, prompt = line.split('\t', 1)
        groups[group(prompt)].append(row(prompt, label, 'original', i))
    sms = {'name': 'sms_triage', 'descriptions': {'ham': 'A legitimate personal or service text message.',
                                                'spam': 'An unsolicited promotional, fraudulent, or spam text message.'},
           'provenance': 'UCI SMS Spam Collection, original labels. Fixed normalized-group hash split; historical English SMS, not modern phishing coverage.',
           'teach': [], 'calibration': [], 'evaluate': [],
           'source_rows': sum(len(rows) for rows in groups.values()),
           'normalized_duplicates_removed': sum(len(rows) - 1 for rows in groups.values()),
           'conflicting_groups_removed': 0}
    for digest, rows in sorted(groups.items()):
        if len({r['label'] for r in rows}) != 1:
            sms['conflicting_groups_removed'] += 1
            continue
        bucket = int(digest, 16) % 10
        split = 'teach' if bucket < 6 else 'calibration' if bucket < 8 else 'evaluate'
        sms[split].append(rows[0])
    for data in (assistant, sms):
        path = OUTPUT / f"{data['name']}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
        print(data['name'], {s: len(data[s]) for s in ('teach', 'calibration', 'evaluate')},
              hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    prepare()
