#!/usr/bin/env python3
"""Verify the bundled banking subset against an upstream checkout's banking_data folder."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re


def verify(source_dir):
    path = Path(__file__).resolve().parents[2] / 'examples/teaching/banking_support.json'
    data = json.loads(path.read_text())
    source = {}
    for name, digest in data['source']['sha256'].items():
        raw = (source_dir / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest, f'Unexpected upstream {name}'
        source[name] = list(csv.DictReader(raw.decode().splitlines()))

    def group(prompt):
        normalized = ' '.join(re.findall(r'\w+', prompt.casefold()))
        return hashlib.sha256(normalized.encode()).hexdigest()

    intents = ['card_arrival', 'lost_or_stolen_card', 'cash_withdrawal_charge']
    test_indices = [i for i, row in enumerate(source['test.csv']) if row['category'] in intents]
    assert [r['source_row'] for r in data['evaluate']] == test_indices, 'Official test rows changed'
    held = {group(source['test.csv'][i]['text']) for i in test_indices}
    expected = {'teach': [], 'calibration': []}
    for intent in intents:
        groups = defaultdict(list)
        for i, row in enumerate(source['train.csv']):
            if row['category'] == intent:
                groups[group(row['text'])].append(i)
        for i, key in enumerate(sorted(groups)):
            if key not in held:
                expected['calibration' if i < len(groups) * .3 else 'teach'].append(groups[key][0])
    for split, indices in expected.items():
        assert [r['source_row'] for r in data[split]] == indices, f'{split} partition changed'
    for split in ('teach', 'calibration', 'evaluate'):
        source_split = 'test' if split == 'evaluate' else 'train'
        for row in data[split]:
            original = source[f'{source_split}.csv'][row['source_row']]
            assert row['source_split'] == source_split
            assert row['prompt'] == original['text'] and row['label'] == original['category']
            assert row['group'] == 'banking-' + group(row['prompt'])[:20]
    print('Verified source hashes, original labels, fixed partitions, and complete 120-row test slice.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_dir', type=Path)
    verify(parser.parse_args().source_dir)
