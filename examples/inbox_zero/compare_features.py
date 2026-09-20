#!/usr/bin/env python3
"""Separate feature/lesson effects; these are diagnostics, not fresh qualifications."""
import argparse
import json
from pathlib import Path
import socket
import time
from unittest.mock import patch

from evaluate import summarize
from teach import HERE, no_network, teach
from system1.integrations.inbox_zero import CHOICE_KEY, InboxZeroClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, default=Path('.system1/inbox-zero/sources'))
    parser.add_argument('--output', type=Path, default=Path('.system1/inbox-zero/feature-comparison'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    contract = json.loads((args.sources / 'contract.json').read_text())
    cohorts = {
        'fresh_authored': json.loads((HERE / 'evaluation_v2.json').read_text())['cases'],
        'upstream_regression': json.loads((args.sources / 'upstream-evaluation.json').read_text())['cases'],
    }
    questions = {CHOICE_KEY: {'type': 'choice', 'instructions': contract['question'], 'criteria': contract['criteria']}}
    report = {'purpose': 'Post-evaluation attribution diagnostics. Candidate v2 is fixed; do not select/tune models on these results.', 'teacher_calls': 0, 'candidates': {}}
    with patch.object(socket.socket, 'connect', no_network), patch.object(socket, 'create_connection', no_network):
        for lessons in ('lessons.json', 'lessons_v2.json'):
            for features in ('hash', 'tfidf'):
                name = f'{Path(lessons).stem}-{features}'
                path = args.output / f'{name}.s1m'
                _, teaching = teach(HERE / lessons, args.sources / 'contract.json', path, features=features)
                classifier = InboxZeroClassifier(path, enable_actions=True)
                predictions = []
                for cohort, cases in cohorts.items():
                    for row in cases:
                        start = time.perf_counter()
                        response = classifier.classify({'state': {'email': row['email']}, 'questions': questions})
                        elapsed = (time.perf_counter() - start) * 1000
                        prediction = (response.get('suggestedCategory') if response['needsReview']
                                      else response['answers'][CHOICE_KEY]['choice'])
                        predictions.append({'id': row['id'], 'cohort': cohort, 'acceptable': row['acceptable'],
                                            'prediction': prediction, 'needs_review': response['needsReview'],
                                            'latency_ms': elapsed})
                report['candidates'][name] = {**teaching, 'cohorts': {
                    cohort: summarize([r for r in predictions if r['cohort'] == cohort]) for cohort in cohorts},
                    'predictions': predictions}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value['cohorts'] for key, value in report['candidates'].items()}, indent=2))


if __name__ == '__main__':
    main()
