"""First-use behavior, legal actions, and preservation of taught strategies."""
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.quality.evaluate_zero_shot import evaluate, pokemon_state
from pokemon_battle_system1 import BattleType, System1BattleAgent, battle_choices
from paperclips_speedrun import (
    MockPaperclipsBrowserController, Paperclips3PhasePolicy,
    PaperclipsObservation, PaperclipsSpeedrunRunner, PlaywrightPaperclipsController,
)


@pytest.fixture
def battle():
    cases = json.loads((ROOT / 'benchmarks/quality/zero_shot/cases.json').read_text())
    return pokemon_state(cases['pokemon'][0])


def test_frozen_first_use_cases(tmp_path):
    report = evaluate(tmp_path / 'result.json')
    assert report['passed']
    assert report['workloads']['pokemon']['correct'] == 24
    assert report['workloads']['pokemon']['legal'] == 24
    assert report['workloads']['paperclips']['correct'] == 12
    assert report['workloads']['raw_routing_control']['correct'] >= 8


def test_choices_use_facts_without_consuming_randomness_or_mutating_state(battle):
    # Arbitrary names must not carry the strategy. Stats and accuracy also matter.
    for i, move in enumerate(battle.player_pokemon.moves):
        move.name = f'Unknown {i}'
    battle.player_pokemon.moves[0].accuracy = 0
    before, rng = copy.deepcopy(battle), random.getstate()
    actions, moves = battle_choices(battle)
    assert moves[0].move_type == 'Ice'
    assert actions == ['fight']
    assert battle == before
    assert random.getstate() == rng

    battle.player_pokemon.current_hp = 20
    battle.inventory['Potion'] = 1
    fainted = copy.deepcopy(battle.opponent_pokemon)
    fainted.current_hp = 0
    battle.party = [battle.player_pokemon, fainted]
    battle.can_run = True
    assert battle_choices(battle)[0] == ['fight', 'use_item']
    battle.party.append(copy.deepcopy(battle.opponent_pokemon))
    battle.battle_type = BattleType.WILD
    assert battle_choices(battle)[0] == ['fight', 'use_item', 'switch_pokemon', 'run_away']


def model_engine(action, move):
    return SimpleNamespace(decide=Mock(return_value=SimpleNamespace(
        action=action, chosen_move=move, critical_danger=False, threat_level=2,
        confidences={'action': .99, 'chosen_move': .98}, is_ambiguous=False,
        conformal_sets={'action': [action], 'chosen_move': [move]},
    )))


def test_supplied_model_keeps_its_legal_strategy(battle):
    # A taught strategy may prefer Surf even when immediate damage favors Thunderbolt.
    result, review, _ = System1BattleAgent(engine=model_engine('fight', 'move_slot_2')).evaluate(battle)
    assert result['chosen_move'] == 'move_slot_2'
    assert result['decision_source'] == 'model'
    assert result['confidence'] == .98
    assert not review


@pytest.mark.parametrize('action', ['use_item', 'switch_pokemon', 'run_away'])
def test_illegal_model_action_and_exhausted_move_are_replaced(battle, action):
    battle.inventory = {}
    battle.player_pokemon.moves[0].pp = 0
    result, review, _ = System1BattleAgent(engine=model_engine(action, 'move_slot_1')).evaluate(battle)
    assert result['action'] == 'fight'
    assert result['move_name'] == 'Ice Beam'
    assert result['confidence'] is None
    assert result['conformal_set'] == []
    assert result['model_suggestion']['action'] == action
    assert result['model_suggestion']['confidence'] == .98
    assert review


@pytest.mark.parametrize('cost', [None, float('nan'), float('inf'), -1, 101])
def test_paperclips_does_not_buy_with_unknown_or_unaffordable_cost(cost):
    obs = PaperclipsObservation(phase=2, unused_clips=100, available_matter=100000,
                               harvester_cost=cost)
    assert Paperclips3PhasePolicy().evaluate(obs)[0] == 'wait'


def test_paperclips_uses_affordable_alternative():
    obs = PaperclipsObservation(phase=2, unused_clips=100, available_matter=100000,
                               solar_farm_cost=1000, harvester_cost=200, wire_drone_cost=50)
    action, payload, _ = Paperclips3PhasePolicy().evaluate(obs)
    assert (action, payload) == ('phase2_drone', {'type': 'wire_drone'})


@pytest.mark.parametrize('action,payload,cost_field,count_field', [
    ('phase2_drone', {'type': 'harvester'}, 'harvester_cost', 'harvester_drones'),
    ('phase2_drone', {'type': 'wire_drone'}, 'wire_drone_cost', 'wire_drones'),
    ('phase2_factory', None, 'factory_cost', 'factories'),
    ('phase2_power', {'type': 'solar_farm'}, 'solar_farm_cost', 'solar_farms'),
    ('phase2_power', {'type': 'battery_tower'}, 'battery_tower_cost', 'battery_towers'),
])
def test_mock_purchase_consumes_budget_and_rejects_second_purchase(action, payload, cost_field, count_field):
    controller = MockPaperclipsBrowserController()
    controller.obs = PaperclipsObservation(phase=2, unused_clips=100, **{cost_field: 80})
    controller.execute_action(action, payload)
    count = getattr(controller.obs, count_field)
    assert count > 0
    assert controller.obs.unused_clips == 20
    assert 'unavailable' in controller.execute_action(action, payload)
    assert getattr(controller.obs, count_field) == count
    assert controller.obs.unused_clips == 20


@pytest.mark.asyncio
async def test_failed_live_observation_stops_before_action():
    controller = PlaywrightPaperclipsController()
    controller.is_connected = True
    controller.page = SimpleNamespace(evaluate=AsyncMock(side_effect=ValueError('page failed')))
    controller.execute_action = AsyncMock()
    runner = PaperclipsSpeedrunRunner(mode='mock')
    runner.controller = controller
    with pytest.raises(RuntimeError, match='Could not read'):
        await runner.step()
    controller.execute_action.assert_not_called()
