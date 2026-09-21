"""A second courier cohort: targeted objective lessons, unchanged review policy.

The original experiment and its evidence remain frozen. This round records both
executing flagged predictions and stopping before any flagged action.
"""
from collections import Counter
import json
from pathlib import Path
import time
import zipfile

from . import teacher
from .experiment import digest, row, run_episode, summarize
from .skills import Network, teach
from .world import World, facts


def objective_lessons(original):
    rows = {r['input']: dict(r) for r in original}
    # Development found missing local key+open+cargo states. These are isolated
    # observations, not trajectories through the withheld full mission.
    for seed in range(600, 700):
        world = World(seed)
        world.has_key = world.gate_open = world.has_supply = True
        for x in range(1, 10):
            world.pos = (x, 1 + seed % 7)
            text = facts(world)
            split = 'calibration' if int(digest(text)[:8], 16) % 4 == 0 else 'teach'
            rows.setdefault(text, row(text, teacher.objective(world), split))
    return list(rows.values())


def stop_on_review(controller):
    def decide(world):
        result = controller(world)
        if result['review']:
            result = {**result, 'proposed_action': result['action'], 'action': None}
        return result
    return decide


def experiment(directory, count=40, seed=6000):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name('results') / 'courier-evidence.zip'
    with zipfile.ZipFile(source) as archive:
        data = json.loads(archive.read('lessons.json'))
        for name in ('objective.s1m', 'terrain.s1m', 'terrain-corrected.s1m', 'movement.s1m'):
            (root / name).write_bytes(archive.read(name))
    original = Network(root)
    original_changed = Network(root, corrected=True)
    revised = objective_lessons(data['objective'])
    started = time.perf_counter()
    updated = teach('objective', revised, root / 'objective.s1m')
    teaching_ms = (time.perf_counter() - started) * 1000
    corrected, changed = Network(root), Network(root, corrected=True)
    assert updated is not original.objective
    results, episodes = {}, {}
    for suite, config in (('unseen', {}), ('changed_terrain', {'purple': True, 'changed': True})):
        old, new = (original, corrected) if suite == 'unseen' else (original_changed, changed)
        results[suite], episodes[suite] = {}, {}
        for name, model in (('before', old), ('after', new)):
            for strict in (False, True):
                key = name + ('_stop_on_review' if strict else '_execute_flagged')
                controller = stop_on_review(model.decide) if strict else model.decide
                rows = [run_episode(controller, s) if not config else run_episode(controller, s, **config)
                        for s in range(seed, seed + count)]
                episodes[suite][key] = rows
                results[suite][key] = summarize(rows)
                print(suite, key, results[suite][key], flush=True)
    old_inputs = {r['input'] for r in data['objective']}
    extra = [r for r in revised if r['input'] not in old_inputs]
    report = {'experiment': 'Courier objective teaching, strict review follow-up',
              'seed_start': seed, 'seed_count': count, 'development_seeds': [5000, 5009],
              'added_labels': dict(Counter(r['split'] for r in extra)), 'teaching_ms': teaching_ms,
              'alpha': 0.05, 'results': results, 'episodes': episodes,
              'lessons_sha256': digest(revised),
              'limitations': ['Synthetic local inventory-state lessons; full mission still withheld.',
                             'Strict episodes stop on every review flag; stopped missions are not successes.',
                             'Empty conformal sets can occur even for correct high-scoring predictions.',
                             'No change to calibration thresholds or core System1 inference.']}
    (root / 'quality-lessons.json').write_text(json.dumps(revised, indent=2) + '\n')
    (root / 'quality-report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('.system1/courier-quality'))
    parser.add_argument('--episodes', type=int, default=40)
    parser.add_argument('--seed', type=int, default=6000)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 200:
        parser.error('episodes must be between 1 and 200')
    experiment(args.output_dir, args.episodes, args.seed)
