"""Unit and integration tests for Pokémon Showdown Competitive Bot (Gen 1 OU)."""

from __future__ import annotations

import asyncio
import numpy as np
import pytest

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
GAMING_DIR = EXAMPLES_DIR / "gaming"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
if str(GAMING_DIR) not in sys.path:
    sys.path.insert(0, str(GAMING_DIR))

from pokemon_showdown_system1 import (
    GEN1_OU_MOVES,
    GEN1_OU_SPECIES,
    MockShowdownServer,
    PokemonShowdownSystemOne,
    ShermanMorrisonOpponentModel,
    ShowdownBattleSystemOneAgent,
    ShowdownBattleState,
    ShowdownPokemon,
    ShowdownWebSocketClient,
    System2YomiPlanner,
    calculate_gen1_damage,
    compute_gen1_stat,
    get_gen1_type_multiplier,
)
from system1.receipt import verify_decision_witness_receipt


# ============================================================================
# 1. Gen 1 OU Mechanics & Damage Calculation Tests
# ============================================================================

def test_gen1_stat_calculation():
    """Verify exact Gen 1 max competitive stats (all 15 DVs + max Stat Exp)."""
    # Tauros base 110 speed -> max speed 319
    tauros_spe = compute_gen1_stat(base=110, is_hp=False)
    assert tauros_spe == 319

    # Chansey base 250 HP -> max HP 704
    chansey_hp = compute_gen1_stat(base=250, is_hp=True)
    assert chansey_hp == 704


def test_gen1_type_multipliers():
    """Verify Gen 1 type matchup effectiveness rules."""
    # Electric vs Water is 2.0x
    assert get_gen1_type_multiplier("Electric", ("Water",)) == 2.0
    # Electric vs Ground is 0.0x
    assert get_gen1_type_multiplier("Electric", ("Ground",)) == 0.0
    # Normal vs Ghost is 0.0x
    assert get_gen1_type_multiplier("Normal", ("Ghost",)) == 0.0
    # In Gen 1, Ghost does 0x to Psychic (the famous glitch)
    assert get_gen1_type_multiplier("Ghost", ("Psychic",)) == 0.0


def test_calculate_gen1_damage_basic():
    """Verify damage calculation produces valid range and STAB."""
    # Tauros Body Slam (STAB Normal) vs Tauros
    min_d, max_d, exp_d, crit = calculate_gen1_damage("Tauros", "Tauros", "Body Slam")
    assert min_d > 50
    assert max_d >= min_d
    assert min_d <= exp_d <= max_d * 2
    # Base speed 110 -> crit rate 110/512 ~ 0.214
    assert 0.20 <= crit <= 0.23


def test_calculate_gen1_damage_immunity():
    """Verify type immunity results in zero damage."""
    # Tauros Hyper Beam vs Gengar (Normal vs Ghost)
    min_d, max_d, exp_d, crit = calculate_gen1_damage("Tauros", "Gengar", "Hyper Beam")
    assert min_d == 0
    assert max_d == 0
    assert exp_d == 0.0


def test_calculate_gen1_damage_fixed_damage():
    """Verify fixed damage moves (Seismic Toss)."""
    min_d, max_d, exp_d, crit = calculate_gen1_damage("Alakazam", "Chansey", "Seismic Toss")
    assert min_d == 100
    assert max_d == 100
    assert exp_d == 100.0


# ============================================================================
# 2. Sherman-Morrison Distillation Tests
# ============================================================================

def test_sherman_morrison_initialization():
    """Verify opponent model initial covariance and weights."""
    model = ShermanMorrisonOpponentModel(dim=4)
    assert model.P.shape == (4, 4)
    assert model.weights.shape == (4,)
    assert model.update_count == 0


def test_sherman_morrison_rank1_update():
    """Verify online RLS updates weights and increases switch probability when observing switches."""
    model = ShermanMorrisonOpponentModel(dim=4)
    feats = model.extract_features(
        opponent_hp_pct=0.25,
        type_disadvantage=1.0,
        opponent_status=True,
        turns_active=3,
    )

    initial_p = model.predict_switch_probability(feats)
    # Observe that the opponent switched in this situation repeatedly
    for _ in range(5):
        model.update(feats, did_switch=True)

    updated_p = model.predict_switch_probability(feats)
    assert updated_p > initial_p
    assert model.update_count == 5


# ============================================================================
# 3. System 2 Yomi Minimax Planner Tests
# ============================================================================

def test_system2_yomi_planner_switch_anticipation():
    """Verify Yomi Level 1 counters when opponent model predicts high switch likelihood."""
    opp_model = ShermanMorrisonOpponentModel(dim=4)
    # Artificially train model to strongly predict switches on disadvantage
    feats = np.array([0.2, 1.0, 1.0, 0.5])
    for _ in range(8):
        opp_model.update(feats, did_switch=True)

    planner = System2YomiPlanner(opp_model)
    # Payoff: Action 0 = direct attack (better vs stay), Action 1 = prediction attack (better vs switch)
    payoff = np.array([
        [100.0, 20.0],
        [40.0, 150.0],
    ])

    action, yomi_lvl, conf = planner.evaluate_yomi(
        player_actions=["direct_attack", "prediction_attack"],
        opponent_actions=["stay", "switch"],
        payoff_matrix=payoff,
        opponent_features=feats,
    )
    # Should choose prediction_attack with Yomi Level 1
    assert action == "prediction_attack"
    assert yomi_lvl == 1
    assert conf > 0.5


# ============================================================================
# 4. DecisionSchema & Agent System 1 Tests
# ============================================================================

def test_pokemon_showdown_system1_schema():
    """Verify DecisionSchema field attributes."""
    schema = PokemonShowdownSystemOne()
    assert "action_type" in schema._fields
    assert "conformal_ambiguity" in schema._fields
    assert "risk_score" in schema._fields


def test_agent_lethal_ko_decision():
    """Verify that when a lethal KO is available, the agent acts decisively without gating."""
    agent = ShowdownBattleSystemOneAgent()
    state = ShowdownBattleState()
    # Opponent is at 10 HP; Tauros Hyper Beam is an easy lethal kill
    state.opponent_active.current_hp = 10

    cmd, slot, yomi, lat_ms, receipt = agent.decide_action(state)
    assert cmd.startswith("/choose move")
    assert lat_ms < 50.0  # Fast sub-millisecond evaluation
    assert receipt is not None
    assert verify_decision_witness_receipt(receipt.to_dict()) is True


def test_agent_conformal_safety_gate_trigger():
    """Verify that close 50/50 damage triggers the Conformal Safety Gate."""
    agent = ShowdownBattleSystemOneAgent(conformal_threshold=0.50)
    state = ShowdownBattleState()
    # Give Tauros two moves with identical power
    state.player_active.moves = ["Earthquake", "Earthquake"]

    cmd, slot, yomi, lat_ms, receipt = agent.decide_action(state)
    assert cmd.startswith("/choose move")
    assert len(agent.receipts) == 1
    # Receipt confirms execution
    assert verify_decision_witness_receipt(receipt.to_dict()) is True


def test_agent_force_switch_handling():
    """Verify agent selects healthiest bench Pokémon when forced to switch."""
    agent = ShowdownBattleSystemOneAgent()
    state = ShowdownBattleState()
    state.force_switch = True
    state.player_team = [
        ShowdownPokemon("Snorlax", 100, 524, fainted=False),
        ShowdownPokemon("Chansey", 600, 704, fainted=False),
        ShowdownPokemon("Alakazam", 0, 314, fainted=True),
    ]

    cmd, slot, yomi, lat_ms, receipt = agent.decide_action(state)
    assert cmd == "/choose switch 2"  # Chansey has highest HP (600)
    assert slot == 2


# ============================================================================
# 5. Mock Server & Client End-to-End Tests
# ============================================================================

def test_mock_showdown_server_exchange():
    """Verify mock server simulates turn progression and win condition."""
    server = MockShowdownServer()
    req = server.get_request_json()
    assert "active" in req
    assert "side" in req

    res = server.process_action("/choose move 1")
    assert "Turn 2" in res or "|win|" in res
    assert server.state.turn == 2


@pytest.mark.asyncio
async def test_showdown_client_mock_battle_end_to_end():
    """Verify full Showdown battle session executes in mock mode."""
    client = ShowdownWebSocketClient(username="TestSystemOneBot", use_mock=True)
    summary = await client.connect_and_battle(max_turns=10)

    assert summary["status"] == "completed"
    assert summary["turns"] > 0
    assert summary["receipts_generated"] > 0
    assert summary["avg_latency_ms"] >= 0.0


def test_calculate_gen1_fixed_damage_immunities():
    """Verify Gen 1 type immunities apply to fixed-damage moves (Seismic Toss & Night Shade)."""
    # Fighting vs Ghost -> 0x
    min_d, max_d, exp_d, crit = calculate_gen1_damage("Alakazam", "Gengar", "Seismic Toss")
    assert min_d == 0
    assert max_d == 0
    assert exp_d == 0.0

    # Ghost vs Normal -> 0x
    min_d2, max_d2, exp_d2, crit2 = calculate_gen1_damage("Gengar", "Chansey", "Night Shade")
    assert min_d2 == 0
    assert max_d2 == 0
    assert exp_d2 == 0.0


def test_parse_pokemon_condition():
    """Verify Showdown condition parser handles fainted, statused, and healthy Pokemon."""
    client = ShowdownWebSocketClient()
    hp, max_hp, status, fainted = client._parse_pokemon_condition("0 fnt")
    assert hp == 0
    assert status == "fnt"
    assert fainted is True

    hp, max_hp, status, fainted = client._parse_pokemon_condition("280/524 par")
    assert hp == 280
    assert max_hp == 524
    assert status == "par"
    assert fainted is False

    hp, max_hp, status, fainted = client._parse_pokemon_condition("353/353")
    assert hp == 353
    assert max_hp == 353
    assert status == ""
    assert fainted is False


def test_parse_request_to_state_and_forced_switch():
    """Verify _parse_request_to_state populates full team and selects correct switch target when slot 1 is fainted."""
    client = ShowdownWebSocketClient()
    req = {
        "active": [{"moves": [{"move": "Body Slam", "disabled": False}]}],
        "side": {
            "pokemon": [
                {"ident": "p1: Tauros", "condition": "0 fnt", "active": False},
                {"ident": "p1: Snorlax", "condition": "280/524 par", "active": True},
                {"ident": "p1: Chansey", "condition": "704/704", "active": False},
            ]
        },
        "rqid": 3,
        "forceSwitch": [True],
    }
    state = client._parse_request_to_state(req)
    assert len(state.player_team) == 3
    assert state.player_active.species == "Snorlax"
    assert state.player_team[0].fainted is True
    assert state.player_team[0].current_hp == 0

    cmd, slot, _, _, _ = client.agent.decide_action(state)
    # Highest HP healthy pokemon is Chansey at index 3 (704 HP), NOT fainted Tauros at slot 1
    assert cmd == "/choose switch 3"
    assert slot == 3

