#!/usr/bin/env python3
"""Observe original dataset labels or real Jev answers; validate and test local takeover."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import time
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'examples')]
from _teaching_demo import quality_metrics
from system1.compat.typesafe import TypeSafeClient, call_real_typesafe_api


def gemini_answer(state, questions):
    criteria = questions['intent']['criteria']
    payload = {
        'systemInstruction': {'parts': [{'text': questions['intent']['instructions'] + '\n' + json.dumps(criteria)}]},
        'contents': [{'role': 'user', 'parts': [{'text': state}]}],
        'generationConfig': {'temperature': 0, 'maxOutputTokens': 256,
                             'thinkingConfig': {'thinkingBudget': 0},
                             'responseMimeType': 'application/json',
                             'responseSchema': {'type': 'OBJECT', 'properties': {
                                 'intent': {'type': 'STRING', 'enum': list(criteria)}}, 'required': ['intent']}},
    }
    request = urllib.request.Request(
        'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json',
                                                  'x-goog-api-key': os.environ['GEMINI_API_KEY']})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = json.load(response)
    text = ''.join(part.get('text', '') for part in raw['candidates'][0]['content']['parts'])
    value = json.loads(text)['intent']
    assert value in criteria
    usage = raw.get('usageMetadata', {})
    normalized = {'model': raw['modelVersion'], 'answers': {'intent': {'type': 'choice', 'choice': value}},
                  'usage': {'input_tokens': usage.get('promptTokenCount', 0),
                            'output_tokens': usage.get('candidatesTokenCount', 0),
                            'total_tokens': usage.get('totalTokenCount', 0)}}
    return normalized, (time.perf_counter() - start) * 1000, 0


def load_workload(name):
    protocol_path = ROOT / 'benchmarks/quality/workloads/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    entry = protocol['workloads'][name]
    path = ROOT / entry['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], 'Frozen data changed'
    data = json.loads(path.read_text())
    descriptions = data.get('descriptions') or {k: v for k, v in data['policy'].items() if k != 'scope'}
    return protocol, data, descriptions


def observe(name, teacher, output_dir):
    protocol, data, descriptions = load_workload(name)
    settings = protocol['takeover']
    stream = sorted(data['teach'] + data['calibration'], key=lambda r:
                    hashlib.sha256(f"{name}:{r['group']}".encode()).hexdigest())[:settings['max_observations']]
    evaluation_groups = {r['group'] for r in data['evaluate']}
    assert not evaluation_groups.intersection(r['group'] for r in stream)
    questions = {'intent': {'type': 'choice', 'criteria': descriptions,
                            'instructions': 'Classify this request into exactly one of the supplied categories.'}}
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / 'teacher_responses.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    original_labels = {r['prompt']: r['label'] for r in stream}
    calls = 0
    observations = []
    used_records = []

    def baseline(state, supplied_questions):
        nonlocal calls
        if teacher == 'dataset':
            response = {'answers': {'intent': {'type': 'choice', 'choice': original_labels[state]}}}
        else:
            digest = hashlib.sha256(json.dumps({'state': state, 'questions': supplied_questions}, sort_keys=True).encode()).hexdigest()
            if digest not in cache:
                if teacher == 'jev':
                    response, latency, _ = call_real_typesafe_api(
                        state, supplied_questions, api_key=os.environ['TYPESAFE_API_KEY'],
                        zero_egress=False, fallback_baseline=False, timeout=30)
                else:
                    response, latency, _ = gemini_answer(state, supplied_questions)
                calls += 1
                # Store only response evidence, never request headers or credentials.
                cache[digest] = {'response': {k: response[k] for k in ('model', 'answers', 'usage')},
                                 'latency_ms': latency, 'recorded_at': time.time()}
                cache_path.write_text(json.dumps(cache, indent=2) + '\n')
            response = cache[digest]['response']
            used_records.append(cache[digest])
        label = response['answers']['intent']['choice']
        assert label in descriptions
        observations.append({'group': next_row['group'], 'label': original_labels[state], 'prediction': label})
        return response

    def disconnected(*args, **kwargs):
        raise AssertionError('Network or teacher called during local evaluation')

    client = TypeSafeClient(mode='auto_cutover', baseline_handler=baseline, backend='numpy',
                           dimension=settings['dimension'], zero_egress=teacher == 'dataset',
                           strict_mode=True, augment=False, fallback_baseline=False, use_cache=False)
    started = time.perf_counter()
    for i, next_row in enumerate(stream, 1):
        client.systemone(next_row['prompt'], questions, group_id=next_row['group'], record_receipt=False)
        if i % 50 == 0:
            print(f'{name} {teacher}: {i} observations; promoted={client.is_cutover}', flush=True)
        if client.is_cutover:
            break
    report = {'workload': name, 'teacher': teacher, 'dataset_sha256': protocol['workloads'][name]['sha256'],
              'protocol_sha256': hashlib.sha256((ROOT / 'benchmarks/quality/workloads/protocol.json').read_bytes()).hexdigest(),
              'promoted': client.is_cutover, 'observations': len(observations), 'new_https_calls': calls,
              'observation_ms': (time.perf_counter() - started) * 1000,
              'observed_teacher_correct': sum(r['prediction'] == r['label'] for r in observations),
              'promotion': asdict(client.last_promotion_report) if client.last_promotion_report else None,
              'teacher_calls_during_test': 0, 'network_calls_during_test': 0,
              'teacher_observations': observations}
    if used_records:
        report['teacher_models'] = sorted({r['response']['model'] for r in used_records})
        report['teacher_latency_median_ms'] = statistics.median(r['latency_ms'] for r in used_records)
        report['teacher_usage'] = {key: sum(r['response']['usage'].get(key, 0) for r in used_records)
                                   for key in ('input_tokens', 'output_tokens', 'total_tokens')}
    if client.is_cutover:
        path = output_dir / 'skill.s1m'
        assert client.export_model(path)
        client.baseline_handler = disconnected
        client.call_real_api = disconnected
        client.zero_egress = True
        predictions = []
        with TypeSafeClient(mode='auto_cutover', model_path=path) as reloaded:
            reloaded.call_real_api = disconnected
            with patch.object(socket.socket, 'connect', disconnected), patch.object(socket, 'create_connection', disconnected):
                for row in data['evaluate']:
                    start = time.perf_counter()
                    result = client.systemone(row['prompt'], questions, record_receipt=False)
                    elapsed = (time.perf_counter() - start) * 1000
                    replay = reloaded.systemone(row['prompt'], questions, record_receipt=False)
                    assert result.answers == replay.answers
                    assert result.is_ambiguous == replay.is_ambiguous
                    assert result.get('abstain', False) == replay.get('abstain', False)
                    assert result.local_execution and replay.local_execution
                    predictions.append({**row, 'prediction': result.choices.intent.choice,
                                        'needs_review': result.is_ambiguous or result.get('abstain', False),
                                        'latency_ms': elapsed})
        quality = quality_metrics(predictions)
        report.update(quality=quality, predictions=predictions, reload_identical=True,
                      skill_bytes=path.stat().st_size,
                      local_latency_median_ms=statistics.median(r['latency_ms'] for r in predictions))
    # No promotion is a result, not a reason to relax the policy or force export.
    report['meets_test_targets'] = report.get('quality', {}).get('meets_routing_targets', False)
    client.close()
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"{name} {teacher}: promoted={report['promoted']}; independent test targets={report['meets_test_targets']}", flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workload', choices=['banking_support', 'assistant_commands', 'sms_triage'])
    parser.add_argument('--teacher', choices=['dataset', 'jev', 'gemini'], default='dataset')
    args = parser.parse_args()
    observe(args.workload, args.teacher, ROOT / '.system1/workloads' / args.workload / args.teacher)
