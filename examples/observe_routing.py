#!/usr/bin/env python3
"""Observe a teacher, validate takeover, disconnect, and reuse one local skill.

The default teacher is an offline rule over synthetic structured support tickets.
This is a reproducible lifecycle demonstration, not a Jev quality benchmark.
Use --teacher jev to collect actual Jev answers (requires TYPESAFE_API_KEY).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from system1.compat.typesafe import Choice, TypeSafeClient

TOPICS = {
    'invoice': 'billing', 'payment': 'billing', 'refund': 'billing', 'subscription': 'billing',
    'password': 'access', 'login': 'access', 'authentication': 'access', 'account': 'access',
}
QUESTIONS = {'queue': Choice(
    'Route a structured request by its topic.',
    criteria={'billing': 'invoice payment refund subscription',
              'access': 'password login authentication account'},
)}


def tickets(seed: int, count: int):
    rng = random.Random(seed)
    for _ in range(count):
        yield {'topic': rng.choice(list(TOPICS)),
               'request': rng.choice(['help', 'question', 'assistance', 'review']),
               'ticket': rng.randrange(1_000_000_000)}


def run(output_dir: Path, teacher: str = 'offline', max_observations: int = 250):
    teacher_calls = 0

    def offline_teacher(state, questions):
        nonlocal teacher_calls
        teacher_calls += 1
        return {'answers': {'queue': {'type': 'choice', 'choice': TOPICS[state['topic']]}},
                'local_execution': False, 'egress_bytes': 0}

    def disconnected(*args, **kwargs):
        raise AssertionError('The teacher was called after takeover')

    if teacher == 'jev' and not os.environ.get('TYPESAFE_API_KEY'):
        raise ValueError('--teacher jev requires TYPESAFE_API_KEY')
    client = TypeSafeClient(
        mode='auto_cutover', backend='numpy', dimension=384,
        zero_egress=teacher == 'offline', fallback_baseline=False,
        baseline_handler=offline_teacher if teacher == 'offline' else None,
    )
    started = time.perf_counter()
    last_attempt_ms = 0.0
    observed = set()
    for state in tickets(7, max_observations):
        observed.add(json.dumps(state, sort_keys=True))
        before = time.perf_counter()
        client.systemone(state, QUESTIONS, record_receipt=False)
        last_attempt_ms = (time.perf_counter() - before) * 1000
        if client.is_cutover:
            break
    observation_ms = (time.perf_counter() - started) * 1000
    report = {
        'teacher': teacher,
        'data': 'Synthetic structured tickets; repeated vocabulary, new ticket IDs. Lifecycle evidence only.',
        'promoted': client.is_cutover,
        'observations': client.teacher_sample_count,
        'observation_ms': observation_ms,
        'final_observation_and_teaching_ms': last_attempt_ms,
        'promotion': asdict(client.last_promotion_report) if client.last_promotion_report else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if client.is_cutover:
        # No teacher or network may be used from here on.
        client.baseline_handler = disconnected
        client.call_real_api = disconnected
        client.zero_egress = True
        skill_path = output_dir / 'routing.s1m'
        assert client.export_model(skill_path)
        restored = TypeSafeClient(mode='auto_cutover', model_path=skill_path)
        restored.call_real_api = disconnected
        outcomes = []
        for state in tickets(108, 40):
            assert json.dumps(state, sort_keys=True) not in observed
            result = client.systemone(state, QUESTIONS, record_receipt=False)
            again = restored.systemone(state, QUESTIONS, record_receipt=False)
            assert result.answers == again.answers
            assert result.is_ambiguous == again.is_ambiguous
            assert result.get('abstain', False) == again.get('abstain', False)
            assert result.local_execution and again.local_execution
            outcomes.append({'topic': state['topic'], 'expected': TOPICS[state['topic']],
                             'prediction': result.choices.queue.choice,
                             'needs_review': result.is_ambiguous or result.get('abstain', False),
                             'latency_ms': result.latency_ms})
        accepted = [row for row in outcomes if not row['needs_review']]
        report.update(
            local_evaluations=len(outcomes),
            correct=sum(row['expected'] == row['prediction'] for row in outcomes),
            accepted=len(accepted),
            accepted_errors=sum(row['expected'] != row['prediction'] for row in accepted),
            local_latency_median_ms=statistics.median(row['latency_ms'] for row in outcomes),
            artifact_bytes=skill_path.stat().st_size,
            reload_identical=True, teacher_calls_after_takeover=0, predictions=outcomes,
        )
        if teacher == 'offline':
            assert teacher_calls == client.teacher_sample_count
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', choices=['offline', 'jev'], default='offline')
    parser.add_argument('--max-observations', type=int, default=250)
    parser.add_argument('--output-dir', type=Path, default=Path('.system1/observe-routing'))
    args = parser.parse_args(argv)
    report = run(args.output_dir, args.teacher, args.max_observations)
    print(f"Teacher: {args.teacher}; synthetic structured-ticket lifecycle demonstration")
    print(f"Observed: {report['observations']}; validated takeover: {report['promoted']}")
    if report['promoted']:
        print(f"Fresh local cases: {report['correct']}/{report['local_evaluations']} correct; "
              f"{report['accepted']} accepted; {report['accepted_errors']} accepted errors")
        print(f"Local median: {report['local_latency_median_ms']:.3f} ms; "
              f"saved skill: {report['artifact_bytes']} bytes; reload identical")
        print('Teacher disconnected: zero further calls')
    else:
        print('More evidence is needed; the teacher remains responsible.')
        print(report['promotion'])
    print(f"Evidence: {args.output_dir / 'report.json'}")
    return report


if __name__ == '__main__':
    main()
