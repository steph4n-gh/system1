#!/usr/bin/env python3
"""Evaluate three frozen workloads against starter, majority, and classical baselines."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import platform
import socket
import statistics
import sys
import time
from collections import Counter
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'examples')]
from _teaching_demo import quality_metrics
from observe_workloads import load_workload
from system1 import ChoiceField, CompiledSystemOneModel, DecisionSchema, System1Engine, SystemOneCompiler, __version__


def metrics(rows, labels):
    quality = quality_metrics(rows)
    quality['accepted_errors'] = quality['accepted'] - quality['accepted_correct']
    quality['raw_accuracy'] = quality['correct'] / quality['cases']
    quality['per_class'] = {}
    for label in labels:
        actual = sum(r['label'] == label for r in rows)
        predicted = sum(r['prediction'] == label for r in rows)
        correct = sum(r['label'] == label == r['prediction'] for r in rows)
        accepted = [r for r in rows if not r['needs_review']]
        accepted_predictions = sum(r['prediction'] == label for r in accepted)
        accepted_correct = sum(r['label'] == label == r['prediction'] for r in accepted)
        quality['per_class'][label] = {
            'cases': actual, 'raw_recall': correct / actual if actual else None,
            'raw_precision': correct / predicted if predicted else None,
            'accepted_predictions': accepted_predictions,
            'accepted_precision': accepted_correct / accepted_predictions if accepted_predictions else None,
            'correct_accepted_fraction_of_class': accepted_correct / actual if actual else None,
        }
    return quality


def evaluate(output_dir):
    # A comparison-only dependency; the product and its ordinary tests do not need it.
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                              'numpy': np.__version__, 'sklearn': sklearn.__version__, 'system1': __version__},
              'protocol_sha256': hashlib.sha256((ROOT / 'benchmarks/quality/workloads/protocol.json').read_bytes()).hexdigest(),
              'teacher_calls': 0, 'network_calls': 0, 'workloads': {}}

    def disconnected(*args, **kwargs):
        raise AssertionError('Supervised workload evaluation must stay offline')

    with patch.object(socket.socket, 'connect', disconnected), patch.object(socket, 'create_connection', disconnected):
        for name in ('banking_support', 'assistant_commands', 'sms_triage'):
            protocol, data, descriptions = load_workload(name)
            schema = type(f'{name.title().replace("_", "")}Schema', (DecisionSchema,), {
                'intent': ChoiceField(options=list(descriptions), descriptions=descriptions)})
            seen = set()
            for split in ('teach', 'calibration', 'evaluate'):
                groups = {r['group'] for r in data[split]}
                assert not seen.intersection(groups), 'Cross-split group reuse'
                seen.update(groups)
            compiler = SystemOneCompiler(schema, dimension=2048, regularization=.1, backend='numpy')
            start = time.perf_counter()
            calibration = [(r['prompt'], r['label']) for r in data['calibration']]
            skill = compiler.compile({'intent': [(r['prompt'], r['label']) for r in data['teach']]},
                                     augment=False, calibration_exemplars={'intent': calibration})
            teaching_ms = (time.perf_counter() - start) * 1000
            path = output_dir / f'{name}.s1m'
            skill.save(path)
            restored = CompiledSystemOneModel.load(path)
            skill.use_cache = restored.use_cache = False
            before = System1Engine(schema, model=skill, strict_mode=True, use_cache=False)
            after = System1Engine(schema, model=restored, strict_mode=True, use_cache=False)
            starter = System1Engine(schema, dimension=2048, strict_mode=True, use_cache=False)
            predictions = []
            seed_correct = 0
            for row in data['evaluate']:
                original = before.decide(row['prompt'], alpha=.05, record_receipt=False)
                start = time.perf_counter()
                result = after.decide(row['prompt'], alpha=.05, record_receipt=False)
                elapsed = (time.perf_counter() - start) * 1000
                assert original.values == result.values and original.probabilities == result.probabilities
                assert original.conformal_sets == result.conformal_sets and original.is_ambiguous == result.is_ambiguous
                seed_correct += starter.decide(row['prompt'], record_receipt=False).values['intent'] == row['label']
                predictions.append({**row, 'prediction': result.values['intent'], 'needs_review': result.is_ambiguous,
                                    'prediction_set': result.conformal_sets['intent'], 'latency_ms': elapsed})
            majority = Counter(r['label'] for r in data['teach']).most_common(1)[0][0]
            workload = {'dataset_sha256': protocol['workloads'][name]['sha256'],
                        'counts': {s: len(data[s]) for s in ('teach', 'calibration', 'evaluate')},
                        'majority_label': majority,
                        'majority_correct': sum(r['label'] == majority for r in data['evaluate']),
                        'starter_correct': seed_correct,
                        'system1': {'quality': metrics(predictions, descriptions), 'predictions': predictions,
                                    'teaching_ms': teaching_ms, 'skill_bytes': path.stat().st_size,
                                    'latency_median_ms': statistics.median(r['latency_ms'] for r in predictions),
                                    'reload_identical': True}}
            # The classical model uses identical fit rows and the identical conformal half.
            # It does not use the other half for temperature fitting.
            _, conformal = compiler._split_samples(calibration, None, .5)
            start = time.perf_counter()
            classical = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                                      LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs'))
            classical.fit([r['prompt'] for r in data['teach']], [r['label'] for r in data['teach']])
            labels = list(classical.classes_)
            probabilities = classical.predict_proba([prompt for prompt, _ in conformal])
            scores = sorted(1 - probabilities[i, labels.index(label)] for i, (_, label) in enumerate(conformal))
            rank = math.ceil((len(scores) + 1) * .95)
            quantile = scores[rank - 1] if rank <= len(scores) else 1.0
            fit_ms = (time.perf_counter() - start) * 1000
            blob = gzip.compress(pickle.dumps((classical, quantile), protocol=5), mtime=0)
            (output_dir / f'{name}-classical.pkl.gz').write_bytes(blob)
            rows = []
            for row in data['evaluate']:
                start = time.perf_counter()
                probabilities = classical.predict_proba([row['prompt']])[0]
                prediction = labels[int(np.argmax(probabilities))]
                choices = [label for label, p in zip(labels, probabilities) if 1 - p <= quantile]
                elapsed = (time.perf_counter() - start) * 1000
                rows.append({**row, 'prediction': prediction, 'needs_review': len(choices) != 1,
                             'prediction_set': choices, 'latency_ms': elapsed})
            workload['classical'] = {'quality': metrics(rows, descriptions), 'predictions': rows,
                                      'teaching_ms': fit_ms, 'compressed_pickle_bytes': len(blob),
                                      'latency_median_ms': statistics.median(r['latency_ms'] for r in rows)}
            report['workloads'][name] = workload
            for variant in ('system1', 'classical'):
                q = workload[variant]['quality']
                print(f"{name} {variant}: raw {q['correct']}/{q['cases']}; accepted {q['accepted']}; "
                      f"correct among accepted {q['accepted_correct']}/{q['accepted']}; targets={q['meets_routing_targets']}", flush=True)
    report['all_system1_workloads_meet_targets'] = all(w['system1']['quality']['meets_routing_targets'] for w in report['workloads'].values())
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    report = evaluate(ROOT / '.system1/workloads/supervised')
    sys.exit(0 if report['all_system1_workloads_meet_targets'] else 1)
