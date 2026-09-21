"""ROM-free evidence checks plus opt-in real emulator integration tests."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.gaming.pokemon_teaching import battle as b
from examples.gaming.pokemon_teaching.red import RomBattle
from examples.gaming.pokemon_teaching.rom_experiment import CONTROLLERS, FIXTURES, summarize
from examples.gaming.pokemon_teaching.server import Lab, RomPair
from examples.gaming.pokemon_battle_system1 import BattleState, BattleType, Pokemon, PokemonMove


def restore_observation(data):
    data = deepcopy(data)
    for key in ('player_pokemon', 'opponent_pokemon'):
        value = data[key]
        value['moves'] = [PokemonMove(**move) for move in value['moves']]
        value['types'] = tuple(value['types'])
        data[key] = Pokemon(**value)
    data['battle_type'] = BattleType(data['battle_type'])
    return BattleState(**data)


def test_frozen_rom_evidence_recomputes_predictions_and_totals(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]/'examples/gaming/pokemon_teaching'
    with zipfile.ZipFile(root/'results/rom-evidence.zip') as archive:
        for name, expected in json.loads(archive.read('manifest.json')).items():
            assert sha256(archive.read(name)).hexdigest() == expected
        for name, source in json.loads(archive.read('sources.json')).items():
            path = root/name if (root/name).exists() else root.parent/name
            assert path.read_text() == source
        for name in archive.namelist():
            assert not name.endswith(('.gb', '.gbc', '.state', '.ram', '.png'))
            if name.endswith('.s1m'):
                (tmp_path/name).write_bytes(archive.read(name))
        report = json.loads(archive.read('rom-suite-report.json'))
        assert json.loads((root/'results/rom-summary.json').read_text()) == {
            k:v for k,v in report.items() if k != 'episodes'}
        def no_teaching(*args, **kwargs):
            raise AssertionError('Saved-skill execution called a teacher')
        monkeypatch.setattr(b, 'teach', no_teaching)
        monkeypatch.setattr(b, 'lessons', no_teaching)
        initials, hashes = {}, {}
        for name, rows in report['episodes'].items():
            mode, corrected, strict = CONTROLLERS[name]
            agent = b.Agent(tmp_path, mode=mode, corrected=corrected, stop_on_review=strict)
            assert [r['fixture'] for r in rows] == list(FIXTURES)
            assert summarize(rows) == report['results'][name]
            for row in rows:
                fixture = row['fixture']
                assert initials.setdefault(fixture, row['initial']) == row['initial']
                assert hashes.setdefault(fixture, row['checkpoint_sha256']) == row['checkpoint_sha256']
                for file, expected in row['skill_sha256'].items():
                    assert sha256(archive.read(file)).hexdigest() == expected
                assert row['turns'] == sum(e['executed'] for e in row['events'])
                assert row['potions'] == sum(bool(e.get('item')) and e['executed'] for e in row['events'])
                assert row['review_steps'] == sum(e['review'] for e in row['events'])
                assert row['teacher_calls'] == 0
                previous = None
                for event in row['events']:
                    decision = agent.decide(restore_observation(event['before']))
                    for key in ('action', 'chosen_move', 'review', 'calls'):
                        assert decision[key] == event[key]
                    if previous:
                        assert event['before']['player_pokemon']['current_hp'] == previous['after_hp']
                        assert event['before']['opponent_pokemon']['current_hp'] == previous['after_enemy_hp']
                        assert event['before']['inventory'] == previous['after_inventory']
                    if event['executed']:
                        assert not (strict and event['review'])
                        if event['action'] == 'use_item':
                            item = event['item']
                            assert event['before']['inventory'][item] - event['after_inventory'].get(item, 0) == 1
                        previous = event
                    else:
                        assert event is row['events'][-1]
                        assert row['outcome'] in ('REVIEW', 'UNSUPPORTED_MOVE', 'UNSUPPORTED_ITEM')
                if row['outcome'] == 'VICTORY':
                    assert previous['after_enemy_hp'] == 0 and previous['after_hp'] > 0
                if row['outcome'] == 'DEFEAT':
                    assert previous['after_hp'] == 0
        assert report['paired_changes'] == {
            'improved': ['threat_slow-37', 'threat_slow-71'], 'regressed': ['threat_slow-109']}


def test_rom_review_stops_before_any_controller_input(monkeypatch):
    state = b.encounter(8004, 'survival')
    battle = RomBattle.__new__(RomBattle)
    battle.agent = SimpleNamespace(stop_on_review=True, decide=lambda state: {
        'action':'use_item', 'chosen_move':'move_slot_1', 'review':True, 'calls':3})
    battle.done, battle.events = False, []
    battle.emulator = SimpleNamespace(memory=None)
    battle.rom = b''
    monkeypatch.setattr('examples.gaming.pokemon_teaching.red.read_state', lambda *args:state)
    monkeypatch.setattr(battle, 'snapshot', lambda **kwargs: battle.outcome)
    monkeypatch.setattr(battle, 'press', lambda *args: pytest.fail('Review must precede input'))
    assert battle.step(screen=False) == 'REVIEW'
    assert battle.done and not battle.events[0]['executed']


def test_bad_rom_settings_do_not_destroy_running_pair(tmp_path):
    pair = RomPair(None, tmp_path, None)
    sentinel = object()
    pair.lanes['before'] = sentinel
    for settings in ({'fixture':'unknown'}, {'strict':'no'}, {'run':'shell'}):
        with pytest.raises(ValueError):
            pair.start(settings)
        assert pair.lanes['before'] is sentinel


ROM = os.environ.get('SYSTEM1_TEST_POKEMON_ROM')
requires_rom = pytest.mark.skipif(not ROM, reason='Set SYSTEM1_TEST_POKEMON_ROM for actual emulator checks')


@pytest.fixture
def real_pair(tmp_path):
    pytest.importorskip('pyboy')
    b.prepare(tmp_path)
    lab = Lab(tmp_path)
    pair = RomPair(ROM, tmp_path, lab)
    yield lab, pair
    pair.close()


def finish(pair):
    while not all(battle.done for battle in pair.lanes.values()):
        pair.step()
    return {name:battle.report() for name,battle in pair.lanes.items()}


@requires_rom
def test_actual_potion_correction_preserves_move_and_restores_same_save(real_pair):
    lab, pair = real_pair
    pair.start({'fixture':'threat_slow-37', 'strict':True})
    original = finish(pair)
    assert all(r['outcome'] == 'DEFEAT' for r in original.values())
    move = pair.lanes['current'].agent.move
    file_hash = pair.lanes['current'].skill_hashes['move.s1m']
    checkpoint = pair.checkpoint
    lab.correct(True)
    pair.correct()
    assert pair.lanes['current'].agent.move is move
    assert pair.lanes['current'].skill_hashes['move.s1m'] == file_hash
    assert pair.checkpoint == checkpoint
    corrected = finish(pair)
    assert corrected['before']['outcome'] == 'DEFEAT'
    assert corrected['current']['outcome'] == 'VICTORY'
    assert corrected['current']['potions'] == 1 and corrected['current']['review_steps'] == 0
    assert corrected['current']['checkpoint_sha256'] == original['current']['checkpoint_sha256']
    lab.correct(False)
    pair.correct()
    restored = finish(pair)
    assert restored['current']['outcome'] == 'DEFEAT'
    assert [(e['action'], e['after_hp']) for e in restored['current']['events']] == [
        (e['action'], e['after_hp']) for e in original['current']['events']]


@requires_rom
@pytest.mark.parametrize('fixture', ['threat_fast-71', 'super_potion-71', 'threat_slow-109'])
def test_actual_item_positions_and_retained_regression(real_pair, fixture):
    lab, pair = real_pair
    lab.correct(True)
    pair.start({'fixture':fixture, 'strict':False})
    rows = finish(pair)
    corrected = rows['current']
    items = [e['item'] for e in corrected['events'] if e.get('item')]
    assert items and all(e['executed'] for e in corrected['events'])
    if fixture == 'super_potion-71':
        assert items == ['Super Potion', 'Potion']
    if fixture == 'threat_slow-109':
        assert rows['before']['outcome'] == 'VICTORY' and corrected['outcome'] == 'DEFEAT'
