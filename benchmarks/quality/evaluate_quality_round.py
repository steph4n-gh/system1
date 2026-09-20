#!/usr/bin/env python3
"""Replay the 1.0.2 quality round offline, retaining deferrals and every error."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'examples')]
from _teaching_demo import load_cases
from model_routing import ModelRouterSchema
from evaluate_confidence import evaluate_skill
from evaluate_workloads import metrics
from observe_workloads import load_workload, observe
from system1 import ChoiceField, DecisionSchema, __version__
from system1.compat.typesafe import TypeSafeClient

ROUND = ROOT / 'benchmarks/quality/quality_round'


def evaluate(output):
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = ROUND / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    cases_path = ROUND / 'cases.json'
    assert hashlib.sha256(cases_path.read_bytes()).hexdigest() == protocol['fresh_cases_sha256']
    routing_path = ROUND / 'model_routing_candidate.json'
    assert hashlib.sha256(routing_path.read_bytes()).hexdigest() == protocol['routing_teaching_sha256']
    probes = json.loads(cases_path.read_text())
    report = {'system1': __version__, 'python': platform.python_version(),
              'protocol_commit': 'f4b3e58',
              'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              'provenance': probes['provenance'], 'limits': protocol['limits'],
              'network_calls': 0, 'live_teacher_calls': 0, 'supervised': {}, 'takeover': {}}

    def disconnected(*args, **kwargs):
        raise AssertionError('Quality round must not use a network or live teacher')

    with patch.object(socket.socket, 'connect', disconnected), patch.object(socket, 'create_connection', disconnected):
        for name in ('banking_support', 'assistant_commands', 'sms_triage'):
            observed = observe(name, 'dataset', output / name / 'observed', offline=True, protocol_path=protocol_path)
            if observed['promoted'] and name in probes['workloads']:
                _, _, descriptions = load_workload(name)
                questions = {'intent': {'type': 'choice', 'criteria': descriptions,
                                       'instructions': 'Classify this request into exactly one of the supplied categories.'}}
                with TypeSafeClient(mode='auto_cutover', model_path=output / name / 'observed/skill.s1m') as client:
                    client.call_real_api = disconnected
                    predictions = []
                    for row in probes['workloads'][name]:
                        response = client.systemone(row['prompt'], questions, record_receipt=False)
                        assert response.local_execution
                        predictions.append({**row, 'prediction': response.choices.intent.choice,
                                            'needs_review': response.is_ambiguous or response.get('abstain', False)})
                observed['fresh_authored'] = {'quality': metrics(predictions, descriptions), 'predictions': predictions}
            report['takeover'][name] = observed
        # Explicit teaching remains useful when an observation stream cannot qualify.
        for name in ('banking_support', 'sms_triage'):
            _, data, descriptions = load_workload(name)
            schema = type(f'{name}Schema', (DecisionSchema,), {
                'intent': ChoiceField(options=list(descriptions), descriptions=descriptions)})
            cohorts = {'published_test_regression': data['evaluate'], 'fresh_authored': probes['workloads'][name]}
            _, result = evaluate_skill(schema, data, cohorts, output / f'{name}.s1m')
            for cohort in result['cohorts'].values():
                cohort['quality'] = metrics(cohort['predictions'], descriptions)
            report['supervised'][name] = result
        current = load_cases(routing_path)
        old = load_cases(ROOT / 'benchmarks/quality/results/model_routing_1_0_1.json')
        cohorts = {'primary_regression': current['evaluate'],
                   'prior_hard_regression': json.loads((ROOT / 'benchmarks/quality/confidence_cases.json').read_text())['evaluate'],
                   'fresh_authored': probes['workloads']['model_routing']}
        for variant, data in (('before', old), ('after', current)):
            _, result = evaluate_skill(ModelRouterSchema, data, cohorts, output / f'routing-{variant}.s1m')
            report['supervised'][f'routing_{variant}'] = result
    # Every result is retained. A deferred takeover does not fail this reproducibility run.
    report['routing_candidate_adopted'] = False
    report['routing_candidate_decision'] = (
        'Keep the existing primary lessons: candidate misses 80% fresh acceptance '
        'and introduces an accepted error on the original cohort.'
    )
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / '.system1/quality-102')
    args = parser.parse_args()
    evaluate(args.output_dir)
