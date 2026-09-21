"""Controlled save-state comparisons using the cartridge's real battle engine.

These are edited laboratory starts, not naturally reached campaign encounters.
Only initial HP, inventory, enemy attack/speed and idle timing differ from the
starter battle. No RAM writes, teacher calls or retries occur during play.
"""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import platform
import statistics

from .red import RomBattle, read_state


# Fixed before running the comparison. Include easy, critical, threatened,
# faster/slower opponents, two item kinds and a no-resource control.
CONDITIONS = {
    'healthy': (20, 10, 10, [(0x14, 1)]),
    'critical': (3, 10, 10, [(0x14, 1)]),
    'threat_slow': (6, 28, 8, [(0x14, 2)]),
    'threat_fast': (6, 28, 16, [(0x0b, 1), (0x14, 2)]),
    'super_potion': (7, 32, 16, [(0x14, 1), (0x13, 1)]),
    'no_items': (6, 28, 16, []),
}
TIMINGS = (37, 71, 109)
FIXTURES = {f'{name}-{delay}': {'condition': name, 'idle_frames': delay}
            for name in CONDITIONS for delay in TIMINGS}
CONTROLLERS = {
    'before': ('network', False, False), 'after': ('network', True, False),
    'before_strict': ('network', False, True), 'after_strict': ('network', True, True),
    'rules': ('rules', False, False), 'updated_rules': ('rules', True, False),
    'single': ('single', False, False), 'updated_single': ('single', True, False),
}


def make_checkpoint(battle, base, fixture):
    """Edit a fresh copy of the same menu save; save the complete machine state."""
    from io import BytesIO
    settings = FIXTURES[fixture]
    hp, attack, speed, bag = CONDITIONS[settings['condition']]
    e = battle.emulator
    e.load_state(BytesIO(base))
    m = e.memory
    # Active and party HP must agree: the game's medicine code heals party HP
    # and copies it back to the active battler. This fixture has one party member.
    if m[0xd163] != 1 or m[0xcc2f] != 0:
        raise ValueError('Fixtures require the one-member starter battle')
    for address, value in ((0xd015, hp), (0xd16c, hp), (0xcff6, attack), (0xcffa, speed)):
        m[address] = value >> 8
        m[address+1] = value & 255
    m[0xd31d] = len(bag)
    m[0xd31e:0xd347] = [0] * 41
    for i, (item, count) in enumerate(bag):
        m[0xd31e+2*i], m[0xd31f+2*i] = item, count
    m[0xd31e+len(bag)*2] = 255
    # Let the game redraw the edited HP via a real party-menu round trip.
    battle.press('up'); battle.press('right'); battle.press('a', 30); battle.press('b', 30)
    e.tick(settings['idle_frames'])
    if not battle.at_menu():
        raise RuntimeError('Fixture setup did not return to the battle menu')
    state = read_state(m, battle.rom)
    assert state.player_pokemon.current_hp == hp and state.opponent_pokemon.attack == attack
    assert state.opponent_pokemon.speed == speed
    return battle.save_checkpoint()


def summarize(rows):
    times = [e['latency_ms'] for r in rows for e in r['events']]
    return {'episodes': len(rows), 'wins': sum(r['outcome'] == 'VICTORY' for r in rows),
            'wins_without_review': sum(r['outcome'] == 'VICTORY' and not r['review_steps'] for r in rows),
            'outcomes': dict(Counter(r['outcome'] for r in rows)),
            'turns': sum(r['turns'] for r in rows), 'potions': sum(r['potions'] for r in rows),
            'review_steps': sum(r['review_steps'] for r in rows),
            'model_calls': sum(r['model_calls'] for r in rows), 'teacher_calls': 0,
            'median_decision_ms': statistics.median(times) if times else 0}


def experiment(rom_path, directory):
    """Run every declared fixture once per controller, retaining all outcomes."""
    from importlib.metadata import version
    root = Path(directory)
    output = root / 'rom-suite'
    output.mkdir(exist_ok=True)
    episodes = {name: [] for name in CONTROLLERS}
    with_base = RomBattle(rom_path, root)
    try:
        base = with_base.save_checkpoint()
        (output / 'base.state').write_bytes(base)
        for fixture in FIXTURES:
            checkpoint = make_checkpoint(with_base, base, fixture)
            (output / f'{fixture}.state').write_bytes(checkpoint)
            expected = sha256(checkpoint).hexdigest()
            for name, (mode, corrected, strict) in CONTROLLERS.items():
                battle = RomBattle(rom_path, root, corrected, checkpoint=checkpoint,
                                   mode=mode, strict=strict, fixture=fixture)
                try:
                    assert battle.checkpoint_sha256 == expected
                    while not battle.done:
                        battle.step(screen=False)
                    row = battle.report()
                    episodes[name].append(row)
                    # Persist incrementally; an error is never replaced by a retry.
                    (output / f'{fixture}-{name}.json').write_text(json.dumps(row, indent=2)+'\n')
                    print(fixture, name, row['outcome'], row['turns'], 'turns', row['potions'], 'potions', flush=True)
                finally:
                    battle.close()
    finally:
        with_base.close()
    paired = {key: [a['fixture'] for a, b in zip(episodes['before'], episodes['after'])
                    if (a['outcome'] == 'VICTORY', b['outcome'] == 'VICTORY') == values]
              for key, values in [('improved', (False, True)), ('regressed', (True, False))]}
    report = {'experiment': 'Controlled initial RAM fixtures in the real Red/Blue battle engine',
              'conditions': CONDITIONS, 'fixtures': FIXTURES, 'paired_changes': paired,
              'results': {name: summarize(rows) for name, rows in episodes.items()}, 'episodes': episodes,
              'platform': platform.platform(), 'python': platform.python_version(), 'pyboy': version('pyboy'),
              'limitations': [
                  '18 fixtures are six constructed conditions at three idle timings, not 18 independent campaign encounters.',
                  'Initial HP, bag, enemy attack and speed are edited; the cartridge handles all subsequent battle effects.',
                  'Enemy stats are controlled stress inputs, not a claim of legal level-5 opponent stats.',
                  'Privileged RAM observations include opponent moves; no visual reasoning or navigation learning.',
                  'Identical complete machine states before every controller; action timing can change later RNG draws.',
                  'Existing saved skills are frozen; this set is not used for teaching or calibration.',
                  'Strict controllers stop before flagged actions; other runs execute and count review flags.',
                  'Different game logic and random draws mean simulator traces cannot predict these outcomes.',
              ]}
    (root / 'rom-suite-report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
