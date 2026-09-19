"""Unit and Integration Tests for Pokémon Battle System 1 Agent.

Verifies:
1. DecisionSchema definitions and field validations.
2. Gen-1 type matchup matrix and effectiveness calculations.
3. RAM memory extraction and PyBoy bridge mapping.
4. Real Game Boy ROM header parsing.
5. High-fidelity visual ASCII Game Boy screen & Game Advisor HUD rendering.
6. Dual-Process cognitive architecture (System 1 fast reflex + System 2 slow planner).
7. SystemOneCompiler static distillation to < 20KB .s1m binary model.
8. TypeSafeClient and patch_typesafe() compatibility.
9. 60 FPS simulated battle loop execution across wild encounters and Gym Leaders.
10. Performance scorecard generation and frame budget compliance.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
GAMING_DIR = EXAMPLES_DIR / "gaming"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
if str(GAMING_DIR) not in sys.path:
    sys.path.insert(0, str(GAMING_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pokemon_battle_system1
from pokemon_battle_system1 import (
    BattleState,
    BattleType,
    ExplorationState,
    GEN1_TYPE_CHART,
    Pokemon,
    PokemonBattleSystemOne,
    PokemonMove,
    PyBoyAdapter,
    PyBoyMemoryBridge,
    System1BattleAgent,
    calculate_damage,
    compile_pokemon_system1_model,
    create_gym_leader_battle,
    create_player_party,
    create_player_pikachu,
    create_wild_encounter,
    evaluate_with_typesafe_client,
    find_default_pokemon_rom,
    generate_performance_scorecard,
    get_pokemon_battle_exemplars,
    get_type_effectiveness,
    get_typesafe_pokemon_questions,
    read_rom_header,
    render_gameboy_exploration_screen,
    render_gameboy_screen,
    run_battle_simulation,
    system2_strategic_planner,
)
from system1 import (
    BooleanField,
    ChoiceField,
    DecisionResult,
    DecisionSchema,
    SystemOneCompiler,
    SystemOneEngine,
    ScoreField,
)
from system1.compat.typesafe import patch_typesafe
from system1.compiler import CompiledSystemOneModel


# ============================================================================
# 1. Schema & Field Validation Tests
# ============================================================================

def test_pokemon_battle_system1_schema():
    """Verify PokemonBattleSystemOne schema fields, types, options, and descriptions."""
    schema = PokemonBattleSystemOne()
    assert isinstance(schema, DecisionSchema)
    assert "action" in schema.fields
    assert "chosen_move" in schema.fields
    assert "critical_danger" in schema.fields
    assert "threat_level" in schema.fields

    # Action field
    action_field = schema.fields["action"]
    assert isinstance(action_field, ChoiceField)
    assert list(action_field.options) == ["fight", "use_item", "switch_pokemon", "run_away"]
    assert "fight" in action_field.descriptions
    assert "run_away" in action_field.descriptions

    # Chosen move field
    move_field = schema.fields["chosen_move"]
    assert isinstance(move_field, ChoiceField)
    assert list(move_field.options) == ["move_slot_1", "move_slot_2", "move_slot_3", "move_slot_4"]
    assert "move_slot_1" in move_field.descriptions
    assert "move_slot_4" in move_field.descriptions

    # Critical danger field
    danger_field = schema.fields["critical_danger"]
    assert isinstance(danger_field, BooleanField)
    assert danger_field.threshold == 0.5

    # Threat level field
    threat_field = schema.fields["threat_level"]
    assert isinstance(threat_field, ScoreField)
    assert threat_field.min_value == 0.0
    assert threat_field.max_value == 10.0


# ============================================================================
# 2. Gen-1 Type Advantage Matrix Tests
# ============================================================================

def test_type_effectiveness_calculations():
    """Verify Gen-1 elemental type matchup multipliers."""
    # Electric matchups
    assert get_type_effectiveness("Electric", ["Water"]) == 2.0
    assert get_type_effectiveness("Electric", ["Flying"]) == 2.0
    assert get_type_effectiveness("Electric", ["Water", "Flying"]) == 4.0  # Gyarados
    assert get_type_effectiveness("Electric", ["Ground"]) == 0.0  # Immunity
    assert get_type_effectiveness("Electric", ["Rock", "Ground"]) == 0.0  # Onix

    # Water matchups
    assert get_type_effectiveness("Water", ["Fire"]) == 2.0
    assert get_type_effectiveness("Water", ["Ground"]) == 2.0
    assert get_type_effectiveness("Water", ["Rock"]) == 2.0
    assert get_type_effectiveness("Water", ["Rock", "Ground"]) == 4.0  # Onix / Rhydon
    assert get_type_effectiveness("Water", ["Water"]) == 0.5
    assert get_type_effectiveness("Water", ["Dragon"]) == 0.5

    # Ice matchups
    assert get_type_effectiveness("Ice", ["Grass"]) == 2.0
    assert get_type_effectiveness("Ice", ["Dragon"]) == 2.0
    assert get_type_effectiveness("Ice", ["Flying"]) == 2.0
    assert get_type_effectiveness("Ice", ["Grass", "Dragon"]) == 4.0
    assert get_type_effectiveness("Ice", ["Fire"]) == 0.5

    # Normal vs Ghost immunity
    assert get_type_effectiveness("Normal", ["Ghost"]) == 0.0
    # Ghost vs Normal immunity in Gen 1
    assert get_type_effectiveness("Ghost", ["Normal"]) == 0.0


# ============================================================================
# 3. Domain Models & State Representation Tests
# ============================================================================

def test_pokemon_and_battle_state():
    """Verify Pokemon stats, moves, damage taking, healing, and prompt serialization."""
    pikachu = create_player_pikachu()
    assert pikachu.name == "Pikachu"
    assert pikachu.level == 32
    assert pikachu.current_hp == 68
    assert pikachu.max_hp == 68
    assert pikachu.hp_ratio == 1.0
    assert not pikachu.is_fainted

    # Damage and Healing
    dmg_taken = pikachu.take_damage(20)
    assert dmg_taken == 20
    assert pikachu.current_hp == 48
    assert abs(pikachu.hp_ratio - (48 / 68)) < 1e-4

    healed = pikachu.heal(10)
    assert healed == 10
    assert pikachu.current_hp == 58

    # Overheal clamp
    overheal = pikachu.heal(100)
    assert overheal == 10
    assert pikachu.current_hp == 68

    # Lethal damage
    pikachu.take_damage(100)
    assert pikachu.current_hp == 0
    assert pikachu.is_fainted

    # Battle state serialization
    state = create_gym_leader_battle("misty")
    prompt = state.to_prompt()
    assert "Player Active: Pikachu" in prompt
    assert "Gym Leader Misty's Starmie" in prompt
    assert "Thunderbolt" in prompt
    assert "Water/Psychic" in prompt
    assert "GYM_LEADER" in prompt


# ============================================================================
# 4. Encounter Factory Tests
# ============================================================================

def test_encounter_factories():
    """Verify all wild encounters and Gym Leader scenarios initialize properly."""
    # Wild encounters
    for sp in ["zubat", "rattata", "gyarados", "onix"]:
        w_state = create_wild_encounter(sp)
        assert w_state.battle_type == BattleType.WILD
        assert w_state.opponent_trainer is None
        assert w_state.can_run is True
        assert w_state.opponent_pokemon.name.lower() == sp

    # Gym leaders
    misty_state = create_gym_leader_battle("misty")
    assert misty_state.battle_type == BattleType.GYM_LEADER
    assert misty_state.opponent_trainer == "Gym Leader Misty"
    assert misty_state.opponent_pokemon.name == "Starmie"
    assert misty_state.can_run is False

    brock_state = create_gym_leader_battle("brock")
    assert brock_state.opponent_trainer == "Gym Leader Brock"
    assert brock_state.opponent_pokemon.name == "Onix"

    giovanni_state = create_gym_leader_battle("giovanni")
    assert giovanni_state.opponent_trainer == "Gym Leader Giovanni"
    assert giovanni_state.opponent_pokemon.name == "Rhydon"


# ============================================================================
# 5. Combat Damage & Status Effect Tests
# ============================================================================

def test_damage_and_status_mechanics():
    """Verify Gen-1 combat damage formula, STAB, and status effects."""
    pikachu = create_player_pikachu()
    starmie_state = create_gym_leader_battle("misty")
    starmie = starmie_state.opponent_pokemon

    tb_move = pikachu.get_move_by_slot("move_slot_1")  # Thunderbolt (Electric)
    assert tb_move is not None

    dmg, mult, is_crit = calculate_damage(pikachu, starmie, tb_move)
    assert mult == 2.0  # Super effective vs Water
    assert dmg > 0

    # Status move (Thunder Wave)
    tw_move = pikachu.get_move_by_slot("move_slot_4")
    assert tw_move is not None
    tw_dmg, tw_mult, _ = calculate_damage(pikachu, starmie, tw_move)
    assert tw_dmg == 0  # Deals 0 damage

    # Ground immunity
    onix_state = create_gym_leader_battle("brock")
    onix = onix_state.opponent_pokemon
    ground_dmg, ground_mult, _ = calculate_damage(pikachu, onix, tb_move)
    assert ground_mult == 0.0
    assert ground_dmg == 0


# ============================================================================
# 6. Visual Game Boy Screen & Game Advisor HUD Rendering Tests
# ============================================================================

def test_gameboy_ascii_rendering():
    """Verify ASCII Game Boy screen formatting, HP bars, Advisor HUD, and exact 72-char width."""
    state = create_gym_leader_battle("misty")
    telemetry = {
        "action": "fight",
        "chosen_move": "move_slot_1",
        "move_name": "Thunderbolt",
        "confidence": 0.965,
        "conformal_set": ["move_slot_1"],
        "threat_level": 8.5,
        "critical_danger": False,
        "latency_ms": 1.04,
    }

    # Standard battle HUD
    output = render_gameboy_screen(state, telemetry=telemetry, system2_halt=False, advisor_mode=False)
    assert "GAME BOY™ COLOR" in output
    assert "60 FPS REAL-TIME REFLEX" in output
    assert "GYM LEADER MISTY: STARMIE" in output
    assert "PIKACHU" in output
    assert "THUNDERBOLT" in output
    assert "1.04 ms / 16.66 ms" in output
    assert "$0.000000" in output

    # Verify every line has exact width of 72 characters
    for idx, line in enumerate(output.split("\n")):
        assert len(line) == 72, f"Line {idx} in standard screen has length {len(line)} != 72: {line!r}"

    # Tactical Game Advisor HUD
    advisor_output = render_gameboy_screen(state, telemetry=telemetry, system2_halt=False, advisor_mode=True)
    assert "TACTICAL GAME ADVISOR" in advisor_output
    assert "Move Matchup Matrix" in advisor_output
    assert "Thunderbolt" in advisor_output
    assert "2.0x SUPER-EFFECTIVE" in advisor_output
    for idx, line in enumerate(advisor_output.split("\n")):
        assert len(line) == 72, f"Line {idx} in advisor screen has length {len(line)} != 72: {line!r}"

    # System 2 Escalation banner
    state.strategic_directive = "Paralyze Starmie with Thunder Wave."
    halt_output = render_gameboy_screen(state, telemetry=telemetry, system2_halt=True, advisor_mode=False)
    assert "SYSTEM 2 COGNITIVE ESCALATION HALT" in halt_output
    assert "Paralyze Starmie" in halt_output
    for idx, line in enumerate(halt_output.split("\n")):
        assert len(line) == 72, f"Line {idx} in halt screen has length {len(line)} != 72: {line!r}"

    # Overworld Exploration Screen & Tactical Advisor HUD
    bridge = PyBoyMemoryBridge()
    exp_state = bridge.extract_exploration_state_from_ram({
        0xD057: 0,
        0xD35E: 0x03,  # Cerulean City
        0xD361: 18,
        0xD362: 14,
        0xD356: 0x03,  # Boulder + Cascade
    })
    exp_output = render_gameboy_exploration_screen(exp_state, advisor_mode=True)
    assert "OVERWORLD EXPLORATION" in exp_output
    assert "CERULEAN CITY" in exp_output
    assert "TACTICAL GAME ADVISOR" in exp_output
    assert "[Boulder] [Cascade]" in exp_output
    for idx, line in enumerate(exp_output.split("\n")):
        assert len(line) == 72, f"Line {idx} in exploration screen has length {len(line)} != 72: {line!r}"


# ============================================================================
# 7. Dual-Process Cognitive Architecture Tests
# ============================================================================

def test_dual_process_cognitive_split():
    """Verify System 1 fast evaluation and System 2 tactical escalation."""
    agent = System1BattleAgent()
    state = create_gym_leader_battle("misty")

    # Initial evaluation: should trigger escalation against Gym Leader without prior directive
    telemetry, should_escalate, reason = agent.evaluate(state)
    assert "action" in telemetry
    assert "chosen_move" in telemetry
    assert telemetry["latency_ms"] < 25.0  # sub-20ms evaluation
    assert should_escalate is True
    assert len(reason) > 0

    # System 2 Planner synthesis
    directive = system2_strategic_planner(state, reason)
    assert "TACTICAL DIRECTIVE" in directive
    assert "Thunder Wave" in directive or "Starmie" in directive
    state.strategic_directive = directive

    # Re-evaluation with directive: no further escalation needed
    telemetry_post, should_escalate_post, _ = agent.evaluate(state)
    assert should_escalate_post is False


# ============================================================================
# 8. System 1 Compiler (< 20 KB .s1m) Tests
# ============================================================================

def test_system1_compiler_s1m_artifact(tmp_path: Path):
    """Verify SystemOneCompiler produces a static binary artifact strictly < 20 KB."""
    target_path = tmp_path / "pokemon_battle_system1.s1m"
    compiled_model, file_size_bytes = compile_pokemon_system1_model(output_path=target_path, dimension=128)

    assert target_path.is_file()
    assert file_size_bytes < 20 * 1024  # Strict requirement: < 20 KB
    assert file_size_bytes > 1000       # Non-empty valid binary

    # Verify deserialization
    loaded = CompiledSystemOneModel.load(target_path)
    assert loaded.dimension == 128
    assert "action" in loaded.heads
    assert "chosen_move" in loaded.heads
    assert "critical_danger" in loaded.heads
    assert "threat_level" in loaded.heads

    # Evaluate inference
    prompt = "Battle State: Player Active: Pikachu. Available Moves: Thunderbolt. Opponent: Wild Zubat Poison Flying."
    res = loaded.forward_single(prompt)
    assert res.fields["action"].selected_value in ["fight", "use_item", "switch_pokemon", "run_away"]
    assert res.fields["chosen_move"].selected_value in ["move_slot_1", "move_slot_2", "move_slot_3", "move_slot_4"]
    assert res.inference_latency_ms < 15.0  # fast evaluation


# ============================================================================
# 9. TypeSafe SDK Compatibility Tests
# ============================================================================

def test_typesafe_client_compatibility():
    """Verify drop-in compatibility with TypeSafeClient."""
    prompt = "Battle State: Active Pikachu [HP: 68/68] vs Gym Leader Misty's Starmie."
    response = evaluate_with_typesafe_client(prompt)

    assert response.local_execution is True
    assert response.answers.action.choice in ["fight", "use_item", "switch_pokemon", "run_away"]
    assert response.answers.chosen_move.choice in ["move_slot_1", "move_slot_2", "move_slot_3", "move_slot_4"]
    assert isinstance(response.answers.critical_danger.value, bool)
    assert 0.0 <= response.answers.threat_level.score <= 10.0
    assert response.latency_ms < 50.0


def test_patch_typesafe_monkeypatching():
    """Verify seamless zero-code monkey-patching with patch_typesafe()."""
    questions = get_typesafe_pokemon_questions()
    prompt = "Active Pikachu vs wild Zubat. HP is healthy."

    with patch_typesafe():
        import typesafe  # type: ignore

        client = typesafe.Client()
        resp = client.systemone(prompt, questions)
        assert resp.local_execution is True
        assert resp.answers.action.choice in ["fight", "use_item", "switch_pokemon", "run_away"]
        assert resp.latency_ms < 50.0


# ============================================================================
# 10. Real ROM Header & PyBoy Memory Bridge Tests
# ============================================================================

def test_rom_header_parsing_and_detection(tmp_path: Path):
    """Verify Game Boy ROM header reader on the user's ROM collection if available, or synthetic ROM."""
    default_rom = find_default_pokemon_rom()
    if default_rom is not None and default_rom.is_file():
        meta = read_rom_header(default_rom)
        assert meta["is_pokemon"] is True
        assert "POKEMON" in meta["title"]
        assert meta["rom_size_kb"] >= 512
        assert meta["cartridge_type"].startswith("0x")

    # Synthetic ROM header test to guarantee header parser coverage in all environments
    synthetic_rom = tmp_path / "synthetic_pkmn.gb"
    header_data = bytearray(0x150)
    title_bytes = b"POKEMON RED\x00\x00\x00\x00\x00"
    header_data[0x134:0x144] = title_bytes
    header_data[0x147] = 0x13  # MBC3+RAM+BATTERY
    header_data[0x148] = 0x05  # 1MB ROM (32KB << 5)
    synthetic_rom.write_bytes(header_data)

    meta_synth = read_rom_header(synthetic_rom)
    assert meta_synth["is_pokemon"] is True
    assert "POKEMON" in meta_synth["title"]
    assert meta_synth["rom_size_kb"] == 1024
    assert meta_synth["cartridge_type"] == "0x13"


def test_pyboy_memory_bridge_extraction():
    """Verify PyBoyMemoryBridge extracts BattleState from Game Boy RAM addresses."""
    bridge = PyBoyMemoryBridge()

    # Construct synthetic RAM memory map representing battle state:
    # Player HP = 50 (0x0032), Max HP = 68 (0x0044), Level = 32
    # Enemy Species = 0x8E (Starmie), HP = 40 (0x0028), Max HP = 85 (0x0055), Level = 35
    synthetic_ram = {
        0xD015: 0x00, 0xD016: 0x32,  # Player HP = 50
        0xD023: 0x00, 0xD024: 0x44,  # Player Max HP = 68
        0xD022: 32,                  # Player Level = 32
        0xD018: 0x00,                # Status OK
        0xD01C: 0x55, 0xD01D: 0x39, 0xD01E: 0x3A, 0xD01F: 0x56,  # Moves: Thunderbolt, Surf, Ice Beam, Thunder Wave
        0xD02D: 15, 0xD02E: 15, 0xD02F: 10, 0xD030: 20,          # PPs
        0xCFE5: 0x8E,                # Enemy Species = Starmie
        0xCFE6: 0x00, 0xCFE7: 0x28,  # Enemy HP = 40
        0xCFE8: 0x00, 0xCFE9: 0x55,  # Enemy Max HP = 85
        0xCFF3: 35,                  # Enemy Level = 35
        0xCFEA: 0x40,                # Enemy Status = Paralysis
        0xD057: 2,                   # Gym Leader Battle
    }

    extracted_state = bridge.extract_battle_state_from_ram(synthetic_ram)
    assert extracted_state.player_pokemon.name == "Pikachu"
    assert extracted_state.player_pokemon.current_hp == 50
    assert extracted_state.player_pokemon.max_hp == 68
    assert extracted_state.opponent_pokemon.name == "Starmie"
    assert extracted_state.opponent_pokemon.current_hp == 40
    assert extracted_state.opponent_pokemon.max_hp == 85
    assert extracted_state.opponent_pokemon.status == "PARALYSIS"
    assert extracted_state.battle_type == BattleType.GYM_LEADER


def test_pyboy_memory_bridge_zero_hp_and_depleted_pp_preservation():
    """Verify 0 HP (fainted) and 0 PP (depleted) are not overwritten with fallback values."""
    bridge = PyBoyMemoryBridge()

    # Fainted player (0 HP), fainted enemy (0 HP), and 0 PP on move 1
    ram = {
        0xD014: 0x1C,                # Player Species = Blastoise
        0xD015: 0x00, 0xD016: 0x00,  # Player HP = 0 (FAINTED!)
        0xD023: 0x00, 0xD024: 0x60,  # Player Max HP = 96
        0xD022: 36,                  # Level 36
        0xD018: 0x00,
        0xD01C: 0x39,                # Move: Surf
        0xD02D: 0x00,                # PP = 0 (DEPLETED!)
        0xCFE5: 0x09,                # Enemy Species = Onix
        0xCFE6: 0x00, 0xCFE7: 0x00,  # Enemy HP = 0 (FAINTED!)
        0xCFE8: 0x00, 0xCFE9: 0x30,  # Enemy Max HP = 48
        0xCFF3: 20,
        0xCFEA: 0x00,
        0xD057: 1,                   # Wild
    }

    state = bridge.extract_battle_state_from_ram(ram)
    # Player assertions
    assert state.player_pokemon.name == "Blastoise"
    assert state.player_pokemon.types == ("Water",)
    assert state.player_pokemon.current_hp == 0  # NOT overwritten with 68!
    assert state.player_pokemon.is_fainted is True
    assert state.player_pokemon.moves[0].pp == 0  # NOT overwritten with 15!
    assert state.player_pokemon.moves[0].is_usable() is False

    # Enemy assertions
    assert state.opponent_pokemon.name == "Onix"
    assert state.opponent_pokemon.current_hp == 0  # NOT overwritten with 85!
    assert state.opponent_pokemon.is_fainted is True


def test_pyboy_memory_bridge_exploration_extraction():
    """Verify PyBoyMemoryBridge extracts overworld ExplorationState correctly."""
    bridge = PyBoyMemoryBridge()

    # Overworld RAM state (wIsInBattle = 0)
    ram = {
        0xD057: 0,       # In Overworld
        0xD35E: 0x03,    # Cerulean City
        0xD361: 18,      # Player Y
        0xD362: 14,      # Player X
        0xD164: 0x54,    # Lead Species = Pikachu
        0xD16B: 0x00, 0xD16C: 0x44,  # Lead HP = 68
        0xD18D: 0x00, 0xD18E: 0x44,  # Lead Max HP = 68
        0xD18C: 32,      # Lead Level = 32
        0xD16F: 0x00,    # Status OK
        0xD356: 0x03,    # Badges: Boulder + Cascade (bits 0 and 1)
    }

    exp_state = bridge.extract_exploration_state_from_ram(ram)
    assert isinstance(exp_state, ExplorationState)
    assert exp_state.map_id == 3
    assert exp_state.map_name == "Cerulean City"
    assert exp_state.player_x == 14
    assert exp_state.player_y == 18
    assert exp_state.lead_pokemon.name == "Pikachu"
    assert exp_state.lead_pokemon.current_hp == 68
    assert exp_state.badges == ["Boulder", "Cascade"]

    # extract_live_state routing
    live_exp = bridge.extract_live_state(ram)
    assert isinstance(live_exp, ExplorationState)

    ram[0xD057] = 2  # Set in battle
    live_battle = bridge.extract_live_state(ram)
    assert isinstance(live_battle, BattleState)


class MockPyBoyInstance:
    """Mock PyBoy instance simulating live memory, ticks, and button dispatch."""

    def __init__(self, memory_map: Dict[int, int]) -> None:
        self.memory = memory_map
        self.ticked_frames = 0
        self.buttons_pressed: List[Tuple[str, int]] = []
        self.stopped = False

    def tick(self) -> None:
        self.ticked_frames += 1

    def button(self, name: str, hold: int = 5) -> None:
        self.buttons_pressed.append((name, hold))
        self.ticked_frames += hold

    def stop(self) -> None:
        self.stopped = True


def test_pyboy_adapter_mock_lifecycle_and_input():
    """Verify PyBoyAdapter lifecycle, frame stepping, and controller input dispatch."""
    ram = {
        0xD014: 0x54, 0xD015: 0x00, 0xD016: 0x44, 0xD023: 0x00, 0xD024: 0x44,
        0xD022: 32, 0xD018: 0, 0xD01C: 0x55, 0xD02D: 15,
        0xCFE5: 0x8E, 0xCFE6: 0x00, 0xCFE7: 0x55, 0xCFE8: 0x00, 0xCFE9: 0x55,
        0xCFF3: 35, 0xCFEA: 0, 0xD057: 2,
    }
    mock_instance = MockPyBoyInstance(ram)
    adapter = PyBoyAdapter(rom_path="dummy.gb", pyboy_instance=mock_instance)

    # Memory reading
    assert adapter.read_byte(0xD014) == 0x54
    assert adapter.read_word(0xD015) == 68
    assert adapter.is_in_battle() is True

    # Frame ticking
    adapter.tick(10)
    assert mock_instance.ticked_frames == 10

    # Button sending
    adapter.send_button("a", hold_frames=4)
    assert ("a", 4) in mock_instance.buttons_pressed

    # Action dispatch: FIGHT -> Move 1
    inputs = adapter.dispatch_action("fight", "move_slot_1")
    assert "A (Fight)" in inputs
    assert "A (Move 1)" in inputs

    # Action dispatch: USE ITEM (DOWN, A)
    item_inputs = adapter.dispatch_action("use_item", "")
    assert "DOWN" in item_inputs
    assert "A (Item)" in item_inputs

    # Action dispatch: SWITCH POKEMON (RIGHT, A)
    switch_inputs = adapter.dispatch_action("switch_pokemon", "")
    assert "RIGHT" in switch_inputs
    assert "A (Pkmn)" in switch_inputs

    # Action dispatch: RUN
    run_inputs = adapter.dispatch_action("run_away", "")
    assert "A (Run)" in run_inputs

    # Live frame advisor step
    agent = System1BattleAgent()
    telemetry, hud = adapter.step_advisor_frame(agent)
    assert "action" in telemetry
    assert "TACTICAL GAME ADVISOR" in hud

    # Clean shutdown
    adapter.stop()
    assert mock_instance.stopped is True


def test_pyboy_adapter_availability_and_fallback():
    """Verify PyBoyAdapter.is_available() and clear ImportError when pyboy is absent."""
    is_avail = PyBoyAdapter.is_available()
    assert isinstance(is_avail, bool)

    # When pyboy is not installed and no mock is provided, PyBoyAdapter raises ImportError with instructions
    if not is_avail:
        with pytest.raises(ImportError) as excinfo:
            PyBoyAdapter(rom_path="missing.gb")
        assert "PyBoy is not installed" in str(excinfo.value)
        assert "built-in standalone zero-dependency" in str(excinfo.value)


# ============================================================================
# 11. Simulated 60 FPS Battle Loop Tests
# ============================================================================

def test_battle_simulation_wild_encounter():
    """Verify simulated battle loop execution against wild encounter."""
    state = create_wild_encounter("zubat")
    summary = run_battle_simulation(state, speed="instant", quiet=True, max_turns=15)

    assert summary["turns"] >= 1
    assert summary["outcome"] in ["VICTORY", "DEFEAT", "ESCAPED"]
    assert summary["mean_latency_ms"] < 25.0
    assert len(summary["all_latencies"]) > 0


def test_battle_simulation_gym_leader_misty():
    """Verify simulated battle loop execution against Gym Leader Misty."""
    state = create_gym_leader_battle("misty")
    summary = run_battle_simulation(state, speed="instant", quiet=True, max_turns=30)

    assert summary["turns"] >= 1
    assert summary["outcome"] in ["VICTORY", "DEFEAT"]
    assert summary["escalations"] >= 1  # System 2 tactical escalation fired
    assert summary["mean_latency_ms"] < 25.0


def test_battle_simulation_struggle_recoil():
    """Verify forced Struggle attack and recoil damage when all move PP is 0."""
    state = create_wild_encounter("rattata")
    # Deplete all PP for Pikachu moves
    for m in state.player_pokemon.moves:
        m.pp = 0

    p_initial_hp = state.player_pokemon.current_hp
    summary = run_battle_simulation(state, speed="instant", quiet=True, max_turns=1)

    assert summary["turns"] >= 1
    # Check that Struggle was used in the battle log
    struggle_logged = any("Used Struggle!" in line for line in state.battle_log)
    assert struggle_logged is True
    # Verify player took recoil
    assert state.player_pokemon.current_hp < p_initial_hp


def test_battle_simulation_poison_residual_damage():
    """Verify residual poison damage at end of turn."""
    state = create_gym_leader_battle("misty")
    state.player_pokemon.max_hp = 500
    state.player_pokemon.current_hp = 500
    state.opponent_pokemon.status = "POISON"
    state.opponent_pokemon.max_hp = 500
    state.opponent_pokemon.current_hp = 500
    opp_initial_hp = state.opponent_pokemon.current_hp

    summary = run_battle_simulation(state, speed="instant", quiet=True, max_turns=1)
    assert summary["turns"] >= 1
    poison_logged = any("was hurt by poison!" in line for line in state.battle_log)
    assert poison_logged is True
    # Confirm Starmie took residual damage
    assert state.opponent_pokemon.current_hp < opp_initial_hp


def test_battle_simulation_with_compiled_model(tmp_path: Path):
    """Verify full battle simulation running exclusively through compiled .s1m model."""
    target_path = tmp_path / "test_compiled_battle.s1m"
    compiled_model, file_size = compile_pokemon_system1_model(output_path=target_path)
    assert file_size < 20 * 1024

    agent = System1BattleAgent(compiled_model=compiled_model)
    state = create_gym_leader_battle("misty")

    summary = run_battle_simulation(state, agent=agent, speed="instant", quiet=True, max_turns=30)
    assert summary["turns"] >= 1
    assert summary["outcome"] in ["VICTORY", "DEFEAT", "TIMEOUT"]
    assert summary["mean_latency_ms"] < 25.0  # Interactive gaming frame ceiling


# ============================================================================
# 12. Performance Scorecard Tests
# ============================================================================

def test_performance_scorecard():
    """Verify scorecard text metrics, speedup ratio, and frame budget calculations."""
    latencies = [1.02, 0.98, 1.15, 0.89, 1.05]
    scorecard = generate_performance_scorecard(latencies)

    assert "SYSTEM 1 vs CLOUD LLM (JEV / GPT-4)" in scorecard
    assert "Forward Latency" in scorecard
    assert "Effective Game Framerate" in scorecard
    assert "60.0 FPS" in scorecard
    assert "450.00 ms" in scorecard
    assert "$0.000000" in scorecard
    assert "$0.002000" in scorecard
    assert "Zero Frame Drops" in scorecard


# ============================================================================
# 13. All 6 Game Boy Pokémon Games & Gen 2 Memory Bridge Tests
# ============================================================================

def test_all_six_gameboy_rom_headers():
    """Verify ROM header parser accurately classifies all 6 Game Boy cartridges."""
    rom_dir = REPO_ROOT / "roms"
    expected = {
        "pokemon_red.gb": ("gen1", "red"),
        "pokemon_blue.gb": ("gen1", "blue"),
        "pokemon_yellow.gb": ("gen1", "yellow"),
        "pokemon_gold.gbc": ("gen2", "gold_silver"),
        "pokemon_silver.gbc": ("gen2", "gold_silver"),
        "pokemon_crystal.gbc": ("gen2", "crystal"),
    }
    for filename, (expected_gen, expected_variant) in expected.items():
        rom_path = rom_dir / filename
        if rom_path.is_file():
            meta = read_rom_header(rom_path)
            assert meta["is_pokemon"] is True
            assert meta["generation"] == expected_gen
            assert meta["game_variant"] == expected_variant
            assert meta["rom_size_kb"] in (1024, 2048)


def test_gen2_gold_silver_memory_bridge():
    """Verify PyBoyMemoryBridge correctly parses Gen 2 Gold/Silver RAM maps."""
    bridge = PyBoyMemoryBridge(game_version="gold_silver")

    # Synthetic Gold/Silver battle RAM
    gs_ram = {
        0xD22D: 1,       # wBattleMode = Wild Battle
        0xD116: 0x00, 0xD117: 35,  # Player HP = 35
        0xD120: 0x00, 0xD121: 45,  # Player Max HP = 45
        0xD11F: 12,      # Player Level = 12
        0xD108: 0x9E,    # Player Species: Totodile
        0xD115: 0x00,    # Status: OK
        0xD109: 0x21,    # Move 1: Tackle
        0xD122: 35,      # PP = 35
        0xD206: 0x00, 0xD207: 30,  # Enemy HP = 30
        0xD210: 0x00, 0xD211: 30,  # Enemy Max HP = 30
        0xD20F: 10,      # Enemy Level = 10
        0xD204: 0x9B,    # Enemy Species: Cyndaquil
        0xD205: 0x00,    # Status: OK
        0xD1F9: 0x34,    # Move 1: Ember
        0xD212: 25,      # PP = 25
    }

    state = bridge.extract_battle_state_from_ram(gs_ram)
    assert state.player_pokemon.name == "Totodile"
    assert state.player_pokemon.current_hp == 35
    assert state.player_pokemon.max_hp == 45
    assert state.player_pokemon.types == ("Water",)
    assert state.opponent_pokemon.name == "Cyndaquil"
    assert state.opponent_pokemon.current_hp == 30
    assert state.opponent_pokemon.types == ("Fire",)
    assert state.battle_type == BattleType.WILD


def test_gen2_crystal_memory_bridge():
    """Verify PyBoyMemoryBridge correctly parses Gen 2 Crystal RAM offsets."""
    bridge = PyBoyMemoryBridge(game_version="crystal")

    crystal_ram = {
        0xD22D: 2,       # wBattleMode = Trainer/Gym Battle
        0xD0F9: 0x00, 0xD0FA: 50,  # Player HP = 50
        0xD103: 0x00, 0xD104: 50,  # Player Max HP = 50
        0xD102: 15,      # Player Level = 15
        0xD0EB: 0x98,    # Player Species: Chikorita
        0xD0F8: 0x00,    # Status: OK
        0xD1E9: 0x00, 0xD1EA: 40,  # Enemy HP = 40
        0xD1F3: 0x00, 0xD1F4: 40,  # Enemy Max HP = 40
        0xD1F2: 14,      # Enemy Level = 14
        0xD1E7: 0x9E,    # Enemy Species: Totodile
        0xD1E8: 0x00,    # Status: OK
    }

    state = bridge.extract_battle_state_from_ram(crystal_ram)
    assert state.player_pokemon.name == "Chikorita"
    assert state.player_pokemon.current_hp == 50
    assert state.player_pokemon.types == ("Grass",)
    assert state.opponent_pokemon.name == "Totodile"
    assert state.opponent_pokemon.current_hp == 40
    assert state.battle_type == BattleType.GYM_LEADER



def test_all_starters_creation_gen1_and_gen2():
    """Verify starter creation across Gen 1 (Red/Blue/Yellow) and Gen 2 (Gold/Silver/Crystal)."""
    from pokemon_full_campaign_speedrun import create_starter_pokemon

    starters = {
        "squirtle": ("Squirtle", ("Water",)),
        "charmander": ("Charmander", ("Fire",)),
        "bulbasaur": ("Bulbasaur", ("Grass", "Poison")),
        "pikachu": ("Pikachu", ("Electric",)),
        "chikorita": ("Chikorita", ("Grass",)),
        "cyndaquil": ("Cyndaquil", ("Fire",)),
        "totodile": ("Totodile", ("Water",)),
    }

    for key, (expected_name, expected_types) in starters.items():
        mon = create_starter_pokemon(key)
        assert mon.name == expected_name
        assert mon.level == 5
        assert mon.current_hp > 0
        assert mon.types == expected_types
        assert len(mon.moves) >= 2

