#!/usr/bin/env python3
"""Compare first-use game decisions and a raw routing control, without teaching."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import socket
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'examples' / 'gaming')]
from pokemon_battle_system1 import BattleState, BattleType, Pokemon, PokemonMove, System1BattleAgent
from paperclips_speedrun import PaperclipsObservation, Paperclips3PhasePolicy
from system1 import ChoiceField, DecisionSchema, HybridProjector, System1Engine
from system1.compiler import SystemOneCompiler


class Routing(DecisionSchema):
    route = ChoiceField(options=['billing', 'support', 'sales'], descriptions={
        'billing': 'Invoices, payments, refunds, subscription charges, receipts',
        'support': 'Software bugs, errors, crashes, installation, debugging, performance',
        'sales': 'Product demos, enterprise quotes, purchases, licenses, commercial contracts',
    })


def pokemon_state(row):
    return BattleState(
        Pokemon('Player', 30, row['player_hp'], 100, ('Electric',),
                [PokemonMove(**m) for m in row['moves']]),
        Pokemon('Opponent', 30, 100, 100, tuple(row['opponent_types']),
                [PokemonMove('move_slot_1', 'Tackle', 'Normal', 35, 100, 10, 10, 'physical')]),
        BattleType.TRAINER, can_run=False, inventory=dict(row['inventory']),
    )


def blocked(*args, **kwargs):
    raise AssertionError('Zero-shot evaluation must not use network, teaching, or calibration')


def compare_projectors():
    """One fixed comparison on existing development data; no parameter search."""
    from benchmarks.quality.run_quality_benchmarks import IntentRoutingSchema, SecurityTriageSchema

    reports = []
    for name, schema, field, label in (
        ('intent_routing', IntentRoutingSchema, 'department', 'expected_department'),
        ('security_triage', SecurityTriageSchema, 'action', 'expected_action'),
    ):
        path = ROOT / 'benchmarks/quality/datasets' / f'{name}.json'
        rows = json.loads(path.read_text())
        for projector_name in ('default', 'hybrid'):
            projector = HybridProjector(dimension=384, backend='numpy') if projector_name == 'hybrid' else None
            engine = System1Engine(schema, projector=projector, dimension=384,
                                   backend='numpy', use_cache=False, strict_mode=True)
            predictions = []
            for row in rows:
                result = engine.decide(row['prompt'], record_receipt=False)
                predictions.append({'prompt': row['prompt'], 'expected': row[label],
                                    'prediction': result.values[field],
                                    'correct': result.values[field] == row[label],
                                    'needs_review': result.is_ambiguous,
                                    'latency_ms': result.latency_ms})
            accepted = [r for r in predictions if not r['needs_review']]
            reports.append({
                'dataset': name, 'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'projector': projector_name, 'dimension': 384, 'cases': len(rows),
                'correct': sum(r['correct'] for r in predictions), 'accepted': len(accepted),
                'accepted_accuracy': sum(r['correct'] for r in accepted) / len(accepted) if accepted else None,
                'latency_median_ms': statistics.median(r['latency_ms'] for r in predictions),
                'predictions': predictions,
            })
            print(f"{name} ({projector_name}): {reports[-1]['correct']}/{len(rows)} correct; {len(accepted)} accepted")
    return reports


def evaluate(output, *, projectors=False):
    path = ROOT / 'benchmarks/quality/zero_shot/cases.json'
    data = json.loads(path.read_text())
    report = {'protocol': data['protocol'], 'cases_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'baseline_commit': 'b45b6f9f71df82178bb7b1e35fdf9a6e324b987d',
              'environment': {'python': platform.python_version(), 'platform': platform.platform()},
              'teaching_calls': 0, 'calibration_calls': 0, 'network_calls': 0, 'workloads': {}}
    with patch.object(socket.socket, 'connect', blocked), patch.object(socket, 'create_connection', blocked), \
         patch.object(SystemOneCompiler, 'compile', blocked), patch.object(System1Engine, 'learn_from_tier2', blocked), \
         patch.object(System1Engine, 'calibrate', blocked):
        agent = System1BattleAgent()
        agent.engine.use_cache = False
        pokemon = []
        for row in data['pokemon']:
            state = pokemon_state(row)
            start = time.perf_counter()
            result, review, _ = agent.evaluate(state)
            raw = result.get('model_suggestion', result)
            move = state.player_pokemon.get_move_by_slot(result['chosen_move'])
            move_name = move.name if move else result.get('move_name')
            usable_move = move is not None and move.pp > 0
            struggle = move_name == 'Struggle' and not any(m.pp > 0 for m in state.player_pokemon.moves)
            can_heal = state.player_pokemon.hp_ratio < 1 and any(
                state.inventory.get(k, 0) > 0 for k in ('Potion', 'Super Potion'))
            legal = (result['action'] == 'fight' and (usable_move or struggle)) or (result['action'] == 'use_item' and can_heal)
            correct = result['action'] == row['expected_action'] and (row['expected_move'] is None or move_name == row['expected_move'])
            raw_move = state.player_pokemon.get_move_by_slot(raw['chosen_move'])
            raw_correct = raw['action'] == row['expected_action'] and (row['expected_move'] is None or raw_move is not None and raw_move.name == row['expected_move'])
            pokemon.append({'id': row['id'], 'action': result['action'], 'move': move_name,
                            'legal': bool(legal), 'correct': bool(correct), 'raw_model_correct': bool(raw_correct),
                            'needs_review': bool(review), 'decision_source': result.get('decision_source', 'model'),
                            'latency_ms': (time.perf_counter() - start) * 1000})
        policy, paper = Paperclips3PhasePolicy(), []
        for row in data['paperclips']:
            obs = PaperclipsObservation(**row['observation'])
            start = time.perf_counter()
            action, payload, _ = policy.evaluate(obs)
            paper.append({'id': row['id'], 'action': action, 'payload': payload,
                          'correct': action == row['expected_action'] and (row['expected_payload'] is None or payload == row['expected_payload']),
                          'decision_source': 'handwritten_game_policy',
                          'latency_ms': (time.perf_counter() - start) * 1000})
        engine, routing = System1Engine(Routing, use_cache=False, strict_mode=True), []
        for row in data['routing']:
            result = engine.decide(row['prompt'], record_receipt=False)
            routing.append({**row, 'prediction': result.values['route'], 'correct': result.values['route'] == row['label'],
                            'needs_review': result.is_ambiguous, 'latency_ms': result.latency_ms})
        if projectors:
            report['projector_comparison'] = compare_projectors()
    for name, rows in [('pokemon', pokemon), ('paperclips', paper), ('raw_routing_control', routing)]:
        report['workloads'][name] = {'cases': len(rows), 'correct': sum(r['correct'] for r in rows),
                                    'latency_median_ms': statistics.median(r['latency_ms'] for r in rows), 'predictions': rows}
        print(f"{name}: {sum(r['correct'] for r in rows)}/{len(rows)} correct")
    report['workloads']['pokemon'].update(legal=sum(r['legal'] for r in pokemon), raw_model_correct=sum(r['raw_model_correct'] for r in pokemon))
    report['passed'] = (all(r['correct'] and r['legal'] for r in pokemon)
                        and all(r['correct'] for r in paper)
                        and sum(r['correct'] for r in routing) >= 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.system1/zero-shot/report.json')
    parser.add_argument('--projectors', action='store_true', help='Also compare the existing default and hybrid text projectors')
    args = parser.parse_args()
    sys.exit(0 if evaluate(args.output, projectors=args.projectors)['passed'] else 1)
