#!/usr/bin/env python3
"""Reproduce the frozen contrast round, before the later confidence-calibration round."""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'examples')]
from _teaching_demo import load_cases, quality_metrics
from support_triage import SupportTicketSchema
from model_routing import ModelRouterSchema
from agent_guard import OperationTriageSchema
from system1 import CompiledSystemOneModel, System1Engine, SystemOneCompiler

SCHEMAS = {'support_triage': SupportTicketSchema, 'model_routing': ModelRouterSchema,
           'agent_guard': OperationTriageSchema}


def evaluate(output_dir):
    cases_path = ROOT / 'benchmarks/quality/contrast_cases.json'
    cases = json.loads(cases_path.read_text())
    report = {
        'provenance': cases['provenance'], 'confirmation_provenance': cases['confirmation_provenance'],
        'cases_sha256': hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        'baseline_commit': 'b45b6f9f71df82178bb7b1e35fdf9a6e324b987d',
        'settings': {'dimension': 2048, 'regularization': .1, 'augment': False,
                     'alpha': .05, 'strict': True, 'cache': False, 'backend': 'numpy'},
        'environment': {'python': platform.python_version(), 'platform': platform.platform()},
        'regression_checks': 'Original and combined cohorts must meet 95% accepted correctness and 80% acceptance. Combined raw correctness must not decrease and combined accepted errors must not increase versus 1.0. Individual contrast cohorts are reported separately and may miss these thresholds.',
        'network_calls': 0, 'teacher_calls': 0, 'workloads': {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    def disconnected(*args, **kwargs):
        raise AssertionError('Contrast evaluation must not contact a network or teacher')

    with patch.object(socket.socket, 'connect', disconnected), patch.object(socket, 'create_connection', disconnected):
        for name, schema in SCHEMAS.items():
            path = ROOT / 'examples/teaching' / f'{name}.json'
            if name == 'model_routing':
                path = ROOT / 'benchmarks/quality/results/model_routing_1_0_1.json'
            current = load_cases(path)
            baseline = load_cases(path, include_quality_round=False)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if any(r.get('quality_round') == '2026-09-confidence' for r in current['calibration']):
                # Reconstruct the published contrast snapshot; its recorded hashes
                # and confirmation results must not change as newer lessons evolve.
                current['calibration'] = [r for r in current['calibration']
                                          if r.get('quality_round') != '2026-09-confidence']
                digest = hashlib.sha256((json.dumps(current, indent=2, ensure_ascii=False) + '\n').encode()).hexdigest()
            # These lessons were selected before the confirmation cohort was tested.
            assert digest == cases['teaching_sha256'][name], 'Teaching changed after confirmation was frozen'
            field = next(iter(schema().fields))
            cohorts = {'original': current['evaluate'], 'diagnostic': cases['workloads'][name],
                       'confirmation': cases['confirmation'][name]}
            prompts = {' '.join(r['prompt'].casefold().split()) for split in ('teach', 'calibration') for r in current[split]}
            groups = {r['group'] for split in ('teach', 'calibration') for r in current[split]}
            for rows in cohorts.values():
                row_prompts = [' '.join(r['prompt'].casefold().split()) for r in rows]
                assert len(set(row_prompts)) == len(rows), 'Repeated evaluation prompt'
                assert not prompts.intersection(row_prompts), 'Evaluation prompt reused across cohorts or teaching'
                assert not groups.intersection(r['group'] for r in rows), 'Evaluation family reused across cohorts or teaching'
                for row in rows:
                    schema().fields[field].validate_value(row['label'])
                prompts.update(row_prompts)
                groups.update(r['group'] for r in rows)

            workload = {'teaching_sha256': digest, 'added_lessons': len(current['teach']) - len(baseline['teach']),
                        'calibration_unchanged': current['calibration'] == baseline['calibration'], 'variants': {}}
            for variant, data in [('baseline', baseline), ('current', current)]:
                started = time.perf_counter()
                skill = SystemOneCompiler(schema, dimension=2048, regularization=.1, backend='numpy').compile(
                    {field: [(r['prompt'], r['label']) for r in data['teach']]}, augment=False,
                    calibration_exemplars={field: [(r['prompt'], r['label']) for r in data['calibration']]},
                )
                teaching_ms = (time.perf_counter() - started) * 1000
                skill_path = output_dir / f'{name}-{variant}.s1m'
                skill.save(skill_path)
                restored = CompiledSystemOneModel.load(skill_path)
                before = System1Engine(schema, model=skill, strict_mode=True, use_cache=False)
                after = System1Engine(schema, model=restored, strict_mode=True, use_cache=False)
                results = {'teaching_cases': len(data['teach']), 'calibration_cases': len(data['calibration']),
                           'teaching_ms': teaching_ms, 'skill_bytes': skill_path.stat().st_size,
                           'reload_identical': True, 'cohorts': {}}
                all_rows = []
                for cohort, rows in cohorts.items():
                    predictions = []
                    for row in rows:
                        result = before.decide(row['prompt'], record_receipt=False)
                        replay = after.decide(row['prompt'], record_receipt=False)
                        assert result.values == replay.values
                        assert result.probabilities == replay.probabilities
                        assert result.conformal_sets == replay.conformal_sets
                        assert result.is_ambiguous == replay.is_ambiguous
                        predictions.append({**row, 'prediction': result.values[field],
                                            'needs_review': result.is_ambiguous,
                                            'prediction_set': result.conformal_sets[field],
                                            'latency_ms': result.latency_ms})
                    results['cohorts'][cohort] = {'quality': quality_metrics(predictions), 'predictions': predictions}
                    all_rows.extend(predictions)
                results['combined_quality'] = quality_metrics(all_rows)
                workload['variants'][variant] = results
            report['workloads'][name] = workload
            for cohort in cohorts:
                before = workload['variants']['baseline']['cohorts'][cohort]['quality']
                after = workload['variants']['current']['cohorts'][cohort]['quality']
                print(f"{name} {cohort}: correct {before['correct']} -> {after['correct']}/{after['cases']}; "
                      f"accepted {before['accepted']} -> {after['accepted']}; "
                      f"accepted errors {before['accepted'] - before['accepted_correct']} -> {after['accepted'] - after['accepted_correct']}")
    report['passed'] = all(
        w['calibration_unchanged']
        and w['variants']['current']['cohorts']['original']['quality']['meets_routing_targets']
        and w['variants']['current']['combined_quality']['meets_routing_targets']
        and w['variants']['current']['combined_quality']['correct'] >= w['variants']['baseline']['combined_quality']['correct']
        and (w['variants']['current']['combined_quality']['accepted'] - w['variants']['current']['combined_quality']['accepted_correct'])
            <= (w['variants']['baseline']['combined_quality']['accepted'] - w['variants']['baseline']['combined_quality']['accepted_correct'])
        for w in report['workloads'].values()
    )
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / '.system1/contrast-round')
    args = parser.parse_args()
    sys.exit(0 if evaluate(args.output_dir)['passed'] else 1)
