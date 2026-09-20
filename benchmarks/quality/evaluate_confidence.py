#!/usr/bin/env python3
"""Check routing calibration and a fixed, externally labeled banking workload offline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'examples')]
from _teaching_demo import load_cases, quality_metrics
from banking_support import BankingSupportSchema
from model_routing import ModelRouterSchema
from system1 import CompiledSystemOneModel, System1Engine, SystemOneCompiler, __version__


def evaluate_skill(schema, data, cohorts, path):
    field = next(iter(schema().fields))
    prompts = {' '.join(r['prompt'].casefold().split())
               for split in ('teach', 'calibration') for r in data[split]}
    groups = {r['group'] for split in ('teach', 'calibration') for r in data[split]}
    for rows in cohorts.values():
        row_prompts = [' '.join(r['prompt'].casefold().split()) for r in rows]
        assert len(set(row_prompts)) == len(rows), 'Repeated evaluation prompt'
        assert not prompts.intersection(row_prompts), 'Evaluation prompt reused'
        assert not groups.intersection(r['group'] for r in rows), 'Evaluation group reused'
        prompts.update(row_prompts)
        groups.update(r['group'] for r in rows)
        for row in rows:
            schema().fields[field].validate_value(row['label'])
    started = time.perf_counter()
    skill = SystemOneCompiler(schema, dimension=2048, regularization=.1, backend='numpy').compile(
        {field: [(r['prompt'], r['label']) for r in data['teach']]}, augment=False,
        calibration_exemplars={field: [(r['prompt'], r['label']) for r in data['calibration']]},
    )
    teaching_ms = (time.perf_counter() - started) * 1000
    skill.save(path)
    restored = CompiledSystemOneModel.load(path)
    original = System1Engine(schema, model=skill, strict_mode=True, use_cache=False)
    reloaded = System1Engine(schema, model=restored, strict_mode=True, use_cache=False)
    result = {'teaching_cases': len(data['teach']), 'calibration_cases': len(data['calibration']),
              'teaching_ms': teaching_ms, 'skill_bytes': path.stat().st_size,
              'reload_identical': True, 'cohorts': {}}
    for name, rows in cohorts.items():
        predictions = []
        for row in rows:
            decision = original.decide(row['prompt'], alpha=.05, record_receipt=False)
            replay = reloaded.decide(row['prompt'], alpha=.05, record_receipt=False)
            assert decision.values == replay.values
            assert decision.probabilities == replay.probabilities
            assert decision.conformal_sets == replay.conformal_sets
            assert decision.is_ambiguous == replay.is_ambiguous
            predictions.append({**row, 'prediction': replay.values[field],
                                'needs_review': replay.is_ambiguous,
                                'prediction_set': replay.conformal_sets[field],
                                'latency_ms': replay.latency_ms})
        quality = quality_metrics(predictions)
        quality['accepted_errors'] = quality['accepted'] - quality['accepted_correct']
        result['cohorts'][name] = {'quality': quality, 'predictions': predictions}
        print(f"{path.stem} {name}: {quality['correct']}/{quality['cases']} correct; "
              f"{quality['accepted']} accepted; {quality['accepted_errors']} accepted errors")
    return skill.heads[field], result


def evaluate(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = ROOT / 'benchmarks/quality/confidence_cases.json'
    cases = json.loads(cases_path.read_text())
    old_cases = json.loads((ROOT / 'benchmarks/quality/contrast_cases.json').read_text())
    routing_path = ROOT / 'benchmarks/quality/results/model_routing_1_0_1.json'
    banking_path = ROOT / 'examples/teaching/banking_support.json'
    assert hashlib.sha256(routing_path.read_bytes()).hexdigest() == cases['teaching_sha256']
    assert hashlib.sha256(banking_path.read_bytes()).hexdigest() == cases['banking_sha256']
    current = load_cases(routing_path)
    baseline = {**current, 'calibration': [r for r in current['calibration']
                                         if r.get('quality_round') != '2026-09-confidence']}
    banking = load_cases(banking_path)
    cohorts = {'development_original': current['evaluate'],
               'development_diagnostic': old_cases['workloads']['model_routing'],
               'development_prior_confirmation': old_cases['confirmation']['model_routing'],
               'fresh_confirmation': cases['evaluate']}
    report = {
        'provenance': cases['provenance'], 'baseline_commit': cases['baseline_commit'],
        'cases_sha256': hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        'teaching_sha256': cases['teaching_sha256'], 'banking_sha256': cases['banking_sha256'],
        'settings': {'dimension': 2048, 'regularization': .1, 'augment': False, 'alpha': .05,
                     'strict': True, 'cache': False, 'receipts': False, 'backend': 'numpy'},
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'numpy': np.__version__, 'system1': __version__},
        'limits': 'Routing development outcomes informed calibration. Fresh routing cases share authorship. '
                  'Banking uses the complete official test slice for three preselected intents, not all 77. '
                  'Point targets are not population accuracy guarantees; all errors are retained.',
        'network_calls': 0, 'teacher_calls': 0,
    }

    def disconnected(*args, **kwargs):
        raise AssertionError('Quality evaluation must not contact a network or teacher')

    with patch.object(socket.socket, 'connect', disconnected), patch.object(socket, 'create_connection', disconnected):
        before_head, before = evaluate_skill(ModelRouterSchema, baseline, cohorts, output_dir / 'routing-before.s1m')
        after_head, after = evaluate_skill(ModelRouterSchema, current, cohorts, output_dir / 'routing-after.s1m')
        assert np.array_equal(before_head.weights, after_head.weights)
        assert np.array_equal(before_head.biases, after_head.biases)
        for name in cohorts:
            assert [r['prediction'] for r in before['cohorts'][name]['predictions']] == [
                r['prediction'] for r in after['cohorts'][name]['predictions']]
        _, banking_result = evaluate_skill(BankingSupportSchema, banking, {'official_test': banking['evaluate']},
                                           output_dir / 'banking-support.s1m')
    report['routing'] = {'weights_unchanged': True, 'raw_predictions_unchanged': True,
                         'before': before, 'after': after}
    report['banking'] = {'source': banking['source'], 'protocol': banking['protocol'], **banking_result}
    report['all_cohorts_meet_targets'] = (
        all(after['cohorts'][name]['quality']['meets_routing_targets'] for name in cohorts)
        and banking_result['cohorts']['official_test']['quality']['meets_routing_targets']
    )
    # The first fresh run missed the 95% point target. Retain that failure above;
    # release regression checks must not turn it into a claim of target attainment.
    report['regression_checks'] = (
        'Each routing cohort retains at least 80% acceptance and no increase in accepted errors; '
        'the original routing and banking workloads meet 95% correctness / 80% acceptance. '
        'These checks permit the explicitly reported fresh-routing target miss.'
    )
    report['regression_checks_passed'] = (
        all(after['cohorts'][name]['quality']['acceptance_rate'] >= .8
            and after['cohorts'][name]['quality']['accepted_errors']
                <= before['cohorts'][name]['quality']['accepted_errors'] for name in cohorts)
        and after['cohorts']['development_original']['quality']['meets_routing_targets']
        and banking_result['cohorts']['official_test']['quality']['meets_routing_targets']
    )
    print(f"All-cohort quality targets: {'PASS' if report['all_cohorts_meet_targets'] else 'NOT MET'}; "
          f"release regression checks: {'PASS' if report['regression_checks_passed'] else 'FAIL'}")
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / '.system1/confidence-round')
    args = parser.parse_args()
    sys.exit(0 if evaluate(args.output_dir)['regression_checks_passed'] else 1)
