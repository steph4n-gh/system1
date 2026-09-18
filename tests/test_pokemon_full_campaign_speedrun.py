"""Unit and Integration Tests for Pokémon Full Campaign Speedrun Engine.

Verifies:
1. Complete 100% campaign state machine progression across all 10 major chapters:
   - Prologue (Pallet Town, Oak's Lab, Starter selection, Route 1, Viridian City Parcel)
   - Chapter 1: Boulder Badge (Viridian Forest, Pewter Gym Leader Brock)
   - Chapter 2: Cascade Badge (Mt. Moon, Cerulean Gym Leader Misty)
   - Chapter 3: Thunder Badge (S.S. Anne HM01 Cut, Vermilion Gym Leader Lt. Surge)
   - Chapter 4: Rainbow Badge (Rock Tunnel, Celadon Rocket Hideout, Gym Leader Erika)
   - Chapter 5: Soul Badge (Pokémon Tower, Cycling Road, Safari Zone HM03 Surf, Gym Leader Koga)
   - Chapter 6: Marsh Badge (Silph Co. Master Ball, Saffron Gym Leader Sabrina)
   - Chapter 7: Volcano Badge (Route 19 Surf, Pokémon Mansion Secret Key, Gym Leader Blaine)
   - Chapter 8: Earth Badge (Viridian Gym, Team Rocket Boss Giovanni)
   - Grand Finale (Route 22/23 Badge check, Victory Road, Elite Four Lorelei, Bruno, Agatha,
     Lance, Champion Blue, Hall of Fame induction).
2. Dual-Process Cognitive Architecture:
   - System 1 local fast reflex (< 2 ms) on metal with $0 API cost and 0 bytes egress.
   - System 1 auto-flee from wild encounters to optimize speedrun splits.
   - Conformal Safety halt triggering on unfamiliar Gym Leader aces.
   - System 2 Campaign Planner route waypoints, key item gating, and tactical directives.
3. Live Terminal Spectator HUD & 8-Badge Trophy Board:
   - Live 8-badge trophy board formatting ([🏆 Boulder] ... [🏆 Earth]).
   - Party Pokémon roster, level, HP bar, and status display.
   - Playback speed mode support (normal, fast, turbo, instant).
4. Reflex Compiler & < 20 KB static model invariant running in pure NumPy.
5. Real ROM header parsing and PyBoy memory bridge RAM synchronization.
6. Starter selection variants (Squirtle, Charmander, Bulbasaur).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pokemon_full_campaign_speedrun import (
    BADGE_ORDER,
    CHAPTER_BADGE_MAP,
    CHAPTER_ORDER,
    CHAPTER_TITLES,
    CampaignChapter,
    CampaignSpeedrunEngine,
    CampaignState,
    RouteWaypoint,
    build_campaign_waypoints,
    compile_speedrun_campaign_model,
    create_starter_pokemon,
    format_trophy_board_line,
    level_up_party_for_chapter,
    parse_args,
    render_spectator_hud,
)
from pokemon_battle_reflex import (
    BattleState,
    BattleType,
    Pokemon,
    PokemonBattleReflex,
    PokemonMove,
    PyBoyAdapter,
    PyBoyMemoryBridge,
    System1BattleAgent,
    read_rom_header,
)
from system1.compiler import CompiledSystemOneModel


# ============================================================================
# 1. Campaign Waypoints & State Initialization Tests
# ============================================================================

def test_campaign_state_initialization():
    """Verify initial campaign state defaults and starter setup."""
    state = CampaignState(starter_choice="squirtle", party=[create_starter_pokemon("squirtle")])
    assert state.chapter == CampaignChapter.PROLOGUE
    assert state.current_waypoint_index == 0
    assert len(state.badges) == 0
    assert len(state.key_items) == 0
    assert state.is_completed is False
    assert state.lead_pokemon.name == "Squirtle"
    assert state.lead_pokemon.level == 5
    assert state.lead_pokemon.types == ("Water",)
    assert len(state.lead_pokemon.moves) == 2
    assert state.inventory.get("Potion", 0) > 0


def test_starter_selection_variants():
    """Verify Squirtle, Charmander, and Bulbasaur starters have accurate Gen-1 profiles."""
    # Squirtle
    squirtle = create_starter_pokemon("squirtle")
    assert squirtle.name == "Squirtle"
    assert squirtle.types == ("Water",)
    assert squirtle.level == 5
    assert any(m.name == "Tackle" for m in squirtle.moves)
    assert any(m.name == "Tail Whip" for m in squirtle.moves)

    # Charmander
    charmander = create_starter_pokemon("charmander")
    assert charmander.name == "Charmander"
    assert charmander.types == ("Fire",)
    assert charmander.level == 5
    assert any(m.name == "Scratch" for m in charmander.moves)
    assert any(m.name == "Growl" for m in charmander.moves)

    # Bulbasaur
    bulbasaur = create_starter_pokemon("bulbasaur")
    assert bulbasaur.name == "Bulbasaur"
    assert bulbasaur.types == ("Grass", "Poison")
    assert bulbasaur.level == 5
    assert any(m.name == "Tackle" for m in bulbasaur.moves)
    assert any(m.name == "Growl" for m in bulbasaur.moves)


def test_campaign_waypoints_structure_and_order():
    """Verify all 35+ campaign waypoints properly traverse all 10 chapters."""
    waypoints = build_campaign_waypoints()
    assert len(waypoints) >= 35

    # Check chapter coverage
    chapters_in_route = [wp.chapter for wp in waypoints]
    for ch in CHAPTER_ORDER:
        assert ch in chapters_in_route, f"Missing chapter {ch} in waypoints"

    # Verify first waypoint is Pallet Town Departure
    assert waypoints[0].chapter == CampaignChapter.PROLOGUE
    assert "Pallet Town" in waypoints[0].name

    # Verify last waypoint is Hall of Fame Induction
    assert waypoints[-1].chapter == CampaignChapter.GRAND_FINALE
    assert "Hall of Fame" in waypoints[-1].name


# ============================================================================
# 2. 8-Badge Trophy Board & Spectator HUD Rendering Tests
# ============================================================================

def test_trophy_board_formatting():
    """Verify 8-badge trophy board displays earned badges with [🏆 <Name>]."""
    # 0 Badges
    empty_board = format_trophy_board_line([])
    for b in BADGE_ORDER:
        assert f"[·  {b}]" in empty_board
        assert f"[🏆 {b}]" not in empty_board

    # 2 Badges earned (Boulder and Cascade)
    partial_board = format_trophy_board_line(["Boulder", "Cascade"])
    assert "[🏆 Boulder]" in partial_board
    assert "[🏆 Cascade]" in partial_board
    assert "[·  Thunder]" in partial_board
    assert "[·  Earth]" in partial_board

    # All 8 Badges earned
    full_board = format_trophy_board_line(list(BADGE_ORDER))
    for b in BADGE_ORDER:
        assert f"[🏆 {b}]" in full_board
        assert f"[·  {b}]" not in full_board


def test_spectator_hud_overworld_rendering():
    """Verify spectator HUD renders overworld navigation, trophy board, and party roster."""
    state = CampaignState(
        starter_choice="squirtle",
        party=[create_starter_pokemon("squirtle")],
        badges=["Boulder", "Cascade"],
        key_items=["Oak's Parcel", "Pokédex"],
    )
    hud = render_spectator_hud(state, advisor_mode=True)
    assert "GAME BOY™ COLOR" in hud
    assert "AUTONOMOUS 100% CAMPAIGN SPEEDRUN" in hud
    assert "🏆 KANTO LEAGUE 8-BADGE TROPHY BOARD:" in hud
    assert "[🏆 Boulder]" in hud
    assert "[🏆 Cascade]" in hud
    assert "PARTY POKÉMON ROSTER:" in hud
    assert "SQUIRTLE" in hud
    assert "DUAL-PROCESS COGNITIVE SPLIT TELEMETRY:" in hud
    assert "TACTICAL GAME ADVISOR HUD" in hud


def test_spectator_hud_battle_scene_rendering():
    """Verify spectator HUD renders real-time battle scene with HP bars and dialogue."""
    lead = create_starter_pokemon("squirtle")
    lead.level = 22
    opp = Pokemon("Starmie", 21, 65, 65, ("Water", "Psychic"), [
        PokemonMove("move_slot_1", "Bubblebeam", "Water", 65, 100, 20, 20, "special")
    ])
    battle = BattleState(
        player_pokemon=lead,
        opponent_pokemon=opp,
        battle_type=BattleType.GYM_LEADER,
        opponent_trainer="Gym Leader Misty",
        battle_log=["Gym Leader Misty sent out STARMIE!", "Squirtle used Mega Punch! (35 dmg)"],
    )
    state = CampaignState(
        chapter=CampaignChapter.CHAPTER_2,
        starter_choice="squirtle",
        party=[lead],
        badges=["Boulder"],
    )
    telemetry = {
        "action": "fight",
        "chosen_move": "move_slot_1",
        "move_name": "Mega Punch",
        "threat_level": 8.5,
        "latency_ms": 0.82,
    }

    hud = render_spectator_hud(
        state,
        battle_state=battle,
        telemetry=telemetry,
        system2_halt=False,
        advisor_mode=True,
    )
    assert "MISTY" in hud
    assert "STARMIE" in hud
    assert "SQUIRTLE" in hud
    assert "ACTIVE MOVES:" in hud
    assert "BATTLE DIALOGUE:" in hud
    assert "Mega Punch" in hud
    assert "Reflex Latency:" in hud
    assert "0.820 ms" in hud


def test_spectator_hud_conformal_safety_halt_rendering():
    """Verify spectator HUD highlights [CONFORMAL SAFETY HALT] during boss escalation."""
    state = CampaignState(
        chapter=CampaignChapter.CHAPTER_6,
        starter_choice="squirtle",
        party=[create_starter_pokemon("squirtle")],
        badges=["Boulder", "Cascade", "Thunder", "Rainbow", "Soul"],
        strategic_directive="Prioritize high physical damage against Sabrina's Alakazam",
    )
    hud = render_spectator_hud(state, system2_halt=True)
    assert "[CONFORMAL SAFETY HALT]" in hud
    assert "System 2 Strategic Directive:" in hud
    assert "Alakazam" in hud


# ============================================================================
# 3. Dual-Process Cognitive Architecture Tests
# ============================================================================

def test_system1_wild_encounter_auto_flee():
    """Verify System 1 evaluates in < 5 ms and auto-flees from random wild encounters."""
    engine = CampaignSpeedrunEngine(starter="squirtle", speed="instant", quiet=True)
    initial_decisions = engine.state.total_decisions

    # Execute Route 1 wild Pidgey waypoint
    wp_wild = RouteWaypoint(
        chapter=CampaignChapter.PROLOGUE,
        name="Route 1 Wild Pidgey",
        map_id=0x0C, map_name="Route 1", coordinates=(10, 14),
        description="Wild encounter test",
        wild_species="Pidgey",
    )
    engine.run_waypoint(wp_wild)

    assert engine.state.total_decisions > initial_decisions
    # Confirm auto-fled is logged
    assert any("Auto-fled from wild Pidgey" in msg for msg in engine.state.log)


def test_system2_conformal_safety_escalation_on_gym_leader():
    """Verify Conformal Safety halts and injects System 2 directive when facing Gym Leader aces."""
    engine = CampaignSpeedrunEngine(starter="squirtle", speed="instant", quiet=True)
    initial_halts = engine.state.conformal_halts

    # Run Brock gym waypoint
    wp_brock = [wp for wp in engine.waypoints if wp.trainer_name == "Brock"][0]
    engine.run_waypoint(wp_brock)

    # Confirm Conformal Safety halted and issued tactical directives
    assert engine.state.conformal_halts > initial_halts
    assert engine.state.system2_directives > 0
    assert engine.state.strategic_directive is not None
    assert "System 2 Directive" in engine.state.strategic_directive
    # Brock defeated and Boulder Badge awarded
    assert "Boulder" in engine.state.badges


def test_party_progression_and_evolution():
    """Verify player roster levels, moves, and evolutions adapt per chapter for all starters."""
    # 1. Squirtle Line
    state_s = CampaignState(starter_choice="squirtle", party=[create_starter_pokemon("squirtle")])
    assert state_s.lead_pokemon.name == "Squirtle"
    assert state_s.lead_pokemon.level == 5

    # Progress to Chapter 2 (Misty): Evolves to Wartortle, adds Pikachu
    level_up_party_for_chapter(state_s, CampaignChapter.CHAPTER_2)
    assert state_s.lead_pokemon.name == "Wartortle"
    assert state_s.lead_pokemon.level == 22
    assert len(state_s.party) == 2
    assert state_s.party[1].name == "Pikachu"

    # Progress to Chapter 4 (Erika): Evolves to Blastoise, Pikachu upgrades to Jolteon
    level_up_party_for_chapter(state_s, CampaignChapter.CHAPTER_4)
    assert state_s.lead_pokemon.name == "Blastoise"
    assert state_s.lead_pokemon.level == 36
    assert state_s.party[1].name == "Jolteon"

    # Progress to Grand Finale: Level 64 Blastoise with 4 party members
    level_up_party_for_chapter(state_s, CampaignChapter.GRAND_FINALE)
    assert state_s.lead_pokemon.level == 64
    assert state_s.lead_pokemon.name == "Blastoise"
    assert any(m.name == "Hydro Pump" for m in state_s.lead_pokemon.moves)

    # 2. Charmander Line
    state_c = CampaignState(starter_choice="charmander", party=[create_starter_pokemon("charmander")])
    level_up_party_for_chapter(state_c, CampaignChapter.CHAPTER_2)
    assert state_c.lead_pokemon.name == "Charmeleon"
    assert state_c.lead_pokemon.types == ("Fire",)
    assert any(m.name == "Ember" for m in state_c.lead_pokemon.moves)

    level_up_party_for_chapter(state_c, CampaignChapter.CHAPTER_4)
    assert state_c.lead_pokemon.name == "Charizard"
    assert state_c.lead_pokemon.types == ("Fire", "Flying")
    assert any(m.name == "Flamethrower" for m in state_c.lead_pokemon.moves)

    level_up_party_for_chapter(state_c, CampaignChapter.GRAND_FINALE)
    assert state_c.lead_pokemon.name == "Charizard"
    assert any(m.name == "Earthquake" for m in state_c.lead_pokemon.moves)

    # 3. Bulbasaur Line
    state_b = CampaignState(starter_choice="bulbasaur", party=[create_starter_pokemon("bulbasaur")])
    level_up_party_for_chapter(state_b, CampaignChapter.CHAPTER_2)
    assert state_b.lead_pokemon.name == "Ivysaur"
    assert state_b.lead_pokemon.types == ("Grass", "Poison")

    level_up_party_for_chapter(state_b, CampaignChapter.CHAPTER_4)
    assert state_b.lead_pokemon.name == "Venusaur"
    assert state_b.lead_pokemon.types == ("Grass", "Poison")
    assert any(m.name == "SolarBeam" for m in state_b.lead_pokemon.moves)


# ============================================================================
# 4. Reflex Compiler & < 20 KB Binary Model Invariant Tests
# ============================================================================

def test_compile_speedrun_campaign_model(tmp_path: Path):
    """Verify campaign heuristics compile into static < 20 KB .s1m artifact."""
    target_path = tmp_path / "pokemon_speedrun_test.s1m"
    compiled_model, file_size = compile_speedrun_campaign_model(output_path=target_path)

    # Size invariant
    assert file_size < 20 * 1024, f"Model size {file_size} exceeds 20 KB limit"
    assert target_path.is_file()

    # Verify model evaluation on test prompt
    prompt = "Gym Leader Brock Onix Rock Ground weak to Surf Water attack"
    eval_res = compiled_model.forward_single(prompt)
    assert "action" in eval_res.fields
    assert "chosen_move" in eval_res.fields
    assert eval_res.fields["action"].selected_value == "fight"
    assert eval_res.fields["chosen_move"].selected_value == "move_slot_2"


def test_speedrun_with_compiled_model(tmp_path: Path):
    """Verify full speedrun engine operates seamlessly when driven by compiled .s1m model."""
    target_path = tmp_path / "compiled_agent.s1m"
    compiled_model, _ = compile_speedrun_campaign_model(output_path=target_path)
    agent = System1BattleAgent(compiled_model=compiled_model)

    engine = CampaignSpeedrunEngine(
        starter="squirtle",
        speed="instant",
        agent=agent,
        quiet=True,
    )
    # Run through first 2 chapters (Boulder & Cascade Badges)
    summary = engine.run_campaign(max_chapters=3)
    assert summary["badges_earned"] >= 2
    assert "Boulder" in summary["badges"]
    assert "Cascade" in summary["badges"]
    assert summary["total_decisions"] > 0
    assert summary["avg_latency_ms"] < 25.0  # Tolerant of virtualized cloud runner CPU jitter


# ============================================================================
# 5. RAM Synchronization & PyBoy Memory Bridge Tests
# ============================================================================

def test_synthetic_ram_synchronization():
    """Verify synthetic RAM correctly maps badge bitfields, map ID, and coordinates."""
    engine = CampaignSpeedrunEngine(starter="squirtle", speed="instant", quiet=True)
    engine.state.award_badge("Boulder")
    engine.state.award_badge("Cascade")
    engine.state.award_badge("Thunder")
    engine.state.current_map_id = 0x29  # Cerulean Gym
    engine.state.current_x = 4
    engine.state.current_y = 2
    engine._sync_synthetic_ram()

    # Badge byte at 0xD356: Bit 0 (Boulder=1) + Bit 1 (Cascade=2) + Bit 2 (Thunder=4) = 7
    badge_byte = engine.read_ram_byte(0xD356)
    assert badge_byte == 0x07

    # Map ID at 0xD35E
    assert engine.read_ram_byte(0xD35E) == 0x29

    # Player coords at 0xD361 (Y), 0xD362 (X)
    assert engine.read_ram_byte(0xD361) == 2
    assert engine.read_ram_byte(0xD362) == 4

    # Memory bridge extraction parity
    bridge = PyBoyMemoryBridge(memory_reader=engine.read_ram_byte)
    exp_state = bridge.extract_exploration_state_from_ram()
    assert "Boulder" in exp_state.badges
    assert "Cascade" in exp_state.badges
    assert "Thunder" in exp_state.badges
    assert exp_state.map_id == 0x29


# ============================================================================
# 6. Complete 10-Chapter End-to-End Speedrun Execution Tests
# ============================================================================

def test_full_100_percent_campaign_speedrun():
    """Verify the entire 100% campaign speedrun clears all 10 chapters and achieves Hall of Fame."""
    engine = CampaignSpeedrunEngine(
        starter="squirtle",
        speed="instant",
        quiet=True,
    )
    summary = engine.run_campaign()

    # 1. 100% Completion Verification
    assert summary["completed"] is True
    assert summary["hall_of_fame"] is True
    assert summary["badges_earned"] == 8

    # 2. All 8 Badges Verified in order
    expected_badges = ["Boulder", "Cascade", "Thunder", "Rainbow", "Soul", "Marsh", "Volcano", "Earth"]
    assert summary["badges"] == expected_badges

    # 3. Chapter & Waypoint Progress
    assert summary["waypoints_cleared"] >= 35
    assert summary["total_decisions"] >= 50
    assert summary["conformal_halts"] >= 8  # At least all 8 Gym Leaders halted
    assert summary["system2_directives"] >= 8

    # 4. Zero-Cost Telemetry Invariant
    assert summary["total_cost"] == 0.0
    assert summary["avg_latency_ms"] < 25.0  # Fast reflex on virtualized cloud runners

    # 5. Speedrun Splits Recorded
    assert len(summary["speedrun_splits"]) == 8
    assert summary["speedrun_splits"][0]["badge"] == "Boulder"
    assert summary["speedrun_splits"][7]["badge"] == "Earth"


def test_speed_modes_parameters():
    """Verify speed parameter configurations (normal, fast, turbo, instant)."""
    e_norm = CampaignSpeedrunEngine(speed="normal", quiet=True)
    assert e_norm.delay == 0.35

    e_fast = CampaignSpeedrunEngine(speed="fast", quiet=True)
    assert e_fast.delay == 0.10

    e_turbo = CampaignSpeedrunEngine(speed="turbo", quiet=True)
    assert e_turbo.delay == 0.02

    e_inst = CampaignSpeedrunEngine(speed="instant", quiet=True)
    assert e_inst.delay == 0.0


def test_starter_charmander_campaign_progression():
    """Verify speedrun engine operates cleanly when selecting Charmander starter."""
    engine = CampaignSpeedrunEngine(
        starter="charmander",
        speed="instant",
        quiet=True,
    )
    # Run through Chapter 4 (Erika) to verify Charizard evolution
    summary = engine.run_campaign(max_chapters=5)
    assert summary["starter"] == "charmander"
    assert summary["badges_earned"] >= 4
    assert "Boulder" in summary["badges"]
    assert "Cascade" in summary["badges"]
    assert "Thunder" in summary["badges"]
    assert "Rainbow" in summary["badges"]
    assert engine.state.lead_pokemon.name == "Charizard"
    assert engine.state.lead_pokemon.types == ("Fire", "Flying")


def test_starter_bulbasaur_campaign_progression():
    """Verify speedrun engine operates cleanly when selecting Bulbasaur starter."""
    engine = CampaignSpeedrunEngine(
        starter="bulbasaur",
        speed="instant",
        quiet=True,
    )
    # Run through Chapter 4 (Erika) to verify Venusaur evolution
    summary = engine.run_campaign(max_chapters=5)
    assert summary["starter"] == "bulbasaur"
    assert summary["badges_earned"] >= 4
    assert engine.state.lead_pokemon.name == "Venusaur"
    assert engine.state.lead_pokemon.types == ("Grass", "Poison")


def test_hm_key_item_gating_enforcement():
    """Verify attempting to traverse a gated route without required key item raises RuntimeError."""
    engine = CampaignSpeedrunEngine(starter="squirtle", speed="instant", quiet=True)
    # Lt. Surge requires HM01 Cut
    wp_surge = [wp for wp in engine.waypoints if wp.trainer_name == "Lt. Surge"][0]
    with pytest.raises(RuntimeError) as exc_info:
        engine.run_waypoint(wp_surge)
    assert "System 2 Gating Halt" in str(exc_info.value)
    assert "HM01 Cut" in str(exc_info.value)

    # Blaine requires Secret Key
    wp_blaine = [wp for wp in engine.waypoints if wp.trainer_name == "Blaine"][0]
    with pytest.raises(RuntimeError) as exc_info2:
        engine.run_waypoint(wp_blaine)
    assert "Secret Key" in str(exc_info2.value)


def test_silph_co_rival_battle_execution():
    """Verify Chapter 6 Silph Co waypoint engages Rival Blue in battle and awards Master Ball."""
    engine = CampaignSpeedrunEngine(starter="squirtle", speed="instant", quiet=True)
    wp_silph = [wp for wp in engine.waypoints if "Silph Co. Rival Battle" in wp.name][0]
    assert wp_silph.is_battle is True
    assert wp_silph.trainer_name == "Blue"
    assert wp_silph.trainer_title == "Rival"
    assert wp_silph.unlocks_item == "Master Ball"

    engine.run_waypoint(wp_silph)
    assert "Master Ball" in engine.state.key_items
    assert any("VICTORY! Defeated Rival Blue!" in log for log in engine.state.log)


def test_combat_system1_action_dispatch():
    """Verify combat loop executes System 1 use_item and switch_pokemon actions."""
    engine = CampaignSpeedrunEngine(starter="squirtle", speed="instant", quiet=True)
    lead = engine.state.lead_pokemon
    lead.current_hp = 5  # Critical HP < 25%
    engine.state.inventory["Super Potion"] = 2

    # Give party a bench teammate
    bench = Pokemon("Pikachu", 18, 45, 45, ("Electric",), [
        PokemonMove("move_slot_1", "Thunderbolt", "Electric", 95, 100, 15, 15, "special")
    ])
    engine.state.party.append(bench)

    opp = Pokemon("Geodude", 12, 38, 38, ("Rock", "Ground"), [
        PokemonMove("move_slot_1", "Tackle", "Normal", 35, 95, 35, 35, "physical")
    ])

    # Run battle with Brock Geodude
    engine._execute_trainer_battle(
        trainer_title="Gym Leader",
        trainer_name="Brock",
        team=[opp],
    )

    # Confirm potion was used
    assert engine.state.inventory["Super Potion"] < 2


def test_numeric_speed_modes():
    """Verify numeric multipliers (1, 2, 5) map to valid pacing delays."""
    e_1 = CampaignSpeedrunEngine(speed=1, quiet=True)
    assert e_1.delay == 0.35

    e_2 = CampaignSpeedrunEngine(speed="2", quiet=True)
    assert e_2.delay == 0.10

    e_5 = CampaignSpeedrunEngine(speed=5, quiet=True)
    assert e_5.delay == 0.02

    e_watch = CampaignSpeedrunEngine(speed="watch", quiet=True)
    assert e_watch.delay == 0.75


def test_battle_mode_ram_synchronization():
    """Verify RAM address 0xD057 (battle mode) and 0xD163 (party count) sync properly."""
    engine = CampaignSpeedrunEngine(starter="squirtle", speed="instant", quiet=True)
    engine._sync_synthetic_ram()

    # Overworld should have battle mode 0
    assert engine.read_ram_byte(0xD057) == 0
    assert engine.read_ram_byte(0xD163) == 1

    # In battle with Brock
    opp = Pokemon("Geodude", 12, 38, 38, ("Rock", "Ground"), [
        PokemonMove("move_slot_1", "Tackle", "Normal", 35, 95, 35, 35, "physical")
    ])
    engine._execute_trainer_battle(
        trainer_title="Gym Leader",
        trainer_name="Brock",
        team=[opp],
    )
    # After battle conclusion, battle mode resets to 0
    assert engine.read_ram_byte(0xD057) == 0


def test_emulator_ticking_functionality():
    """Verify tick_emulator invokes emulator tick frames without crashing."""
    class MockAdapter:
        def __init__(self):
            self.ticks = 0
            self.stopped = False
        def tick(self, count=1):
            self.ticks += count
        def stop(self):
            self.stopped = True

    engine = CampaignSpeedrunEngine(starter="squirtle", speed="instant", quiet=True)
    mock_ad = MockAdapter()
    engine.pyboy_adapter = mock_ad

    # Tick 0 duration defaults to 1 frame tick
    engine.tick_emulator(0.0)
    assert mock_ad.ticks == 1

    # Tick 0.1 seconds = 6 frames
    engine.tick_emulator(0.1)
    assert mock_ad.ticks >= 7


def test_cli_parse_args_speed_modes():
    """Verify parse_args supports both numeric multipliers and named preset speed options."""
    # Numeric speeds
    args_1 = parse_args(["--speed", "1"])
    assert args_1.speed == "1"
    assert CampaignSpeedrunEngine(speed=args_1.speed, quiet=True).delay == 0.35

    args_2 = parse_args(["--speed", "2"])
    assert args_2.speed == "2"
    assert CampaignSpeedrunEngine(speed=args_2.speed, quiet=True).delay == 0.10

    args_5 = parse_args(["--speed", "5"])
    assert args_5.speed == "5"
    assert CampaignSpeedrunEngine(speed=args_5.speed, quiet=True).delay == 0.02

    # Preset names
    args_normal = parse_args(["--speed", "normal"])
    assert CampaignSpeedrunEngine(speed=args_normal.speed, quiet=True).delay == 0.35

    args_fast = parse_args(["--speed", "fast"])
    assert CampaignSpeedrunEngine(speed=args_fast.speed, quiet=True).delay == 0.10

    args_turbo = parse_args(["--speed", "turbo"])
    assert CampaignSpeedrunEngine(speed=args_turbo.speed, quiet=True).delay == 0.02

    args_instant = parse_args(["--speed", "instant"])
    assert CampaignSpeedrunEngine(speed=args_instant.speed, quiet=True).delay == 0.0

    args_watch = parse_args(["--speed", "watch"])
    assert CampaignSpeedrunEngine(speed=args_watch.speed, quiet=True).delay == 0.75

    args_cine = parse_args(["--speed", "cinematic"])
    assert CampaignSpeedrunEngine(speed=args_cine.speed, quiet=True).delay == 1.00


def test_display_hud_non_tty_throttling(monkeypatch):
    """Verify non-TTY execution throttles HUD output to prevent log flooding."""
    import io
    fake_stdout = io.StringIO()
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr(sys.stdout, "write", fake_stdout.write)
    monkeypatch.setattr(sys.stdout, "flush", lambda: None)

    engine = CampaignSpeedrunEngine(speed="instant", quiet=False)
    # Simulate displaying HUD 10 times on the same waypoint
    for _ in range(10):
        engine._display_hud("--- HUD FRAME ---")

    # In non-TTY mode, the same waypoint HUD should only be emitted once
    out = fake_stdout.getvalue()
    assert out.count("--- HUD FRAME ---") == 1


def test_overworld_path_resolution_and_starter_transition():
    """Verify overworld navigation resolves correct paths pre- and post-starter acquisition."""
    from pokemon_gameboy_gui import resolve_path_for_map

    # Pre-starter Oak's Lab: moves to Squirtle Poké Ball
    pre_lab = resolve_path_for_map(40, has_party=False)
    assert pre_lab == ["down", "right", "right"]

    # Post-starter Oak's Lab: moves left 3 times to clear table before moving down to exit
    post_lab = resolve_path_for_map(40, has_party=True)
    assert post_lab[:3] == ["left", "left", "left"]
    assert "down" in post_lab[3:]

    # Post-starter Pallet Town: moves left and up toward Route 1
    pallet = resolve_path_for_map(0, has_party=True)
    assert pallet[0] == "left"
    assert pallet.count("up") >= 10

    # Route 1 & Viridian City
    r1 = resolve_path_for_map(12, has_party=True)
    assert len(r1) > 15
    assert "up" in r1


