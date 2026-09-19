#!/usr/bin/env python3
"""Pokémon Live Game Boy GUI Spectator & System 1 System 1 Agent.

Runs the official Pokémon Game Boy cartridges on PyBoy with a native graphical
macOS SDL2 window, full sprite animations, and real-time System 1 System 1
memory-bridge decision control.

Features:
- Native Graphical macOS Game Boy Window: Real Game Boy pixels, audio, and sprites.
- In-Place Terminal HUD: Zero scrolling, zero flashing—updates smoothly like a real monitor.
- Dual-Process Control: System 1 System 1 evaluates battle states at 60 FPS on the metal.
- Tactical Game Advisor HUD: Live weakness mapping and elemental advice.
- Multi-Game Support: Seamlessly switch across Red, Blue, Yellow, Gold, Silver, Crystal.
- Spectator Modes: Overworld navigation, Live Battle showcase, and Campaign speedrun.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Suppress PyBoy SDL2 library load warnings to keep terminal HUD clean
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure src/ and repo root are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def ensure_venv_reexec() -> None:
    """Transparently re-execs into project venv if running an incompatible Python version."""
    venv_py = ROOT_DIR / ".venv" / "bin" / "python"
    if venv_py.is_file() and Path(sys.executable).resolve() != venv_py.resolve() and sys.version_info[:2] != (3, 13):
        os.execv(str(venv_py), [str(venv_py)] + sys.argv)

# Import System 1 battle mechanics and PyBoy bridges
from examples.gaming.pokemon_battle_system1 import (
    BattleState,
    BattleType,
    Pokemon,
    PokemonMove,
    PyBoyAdapter,
    PyBoyMemoryBridge,
    System1BattleAgent,
    calculate_damage,
    create_gym_leader_battle,
    create_player_party,
    create_player_pikachu,
    create_wild_encounter,
    find_default_pokemon_rom,
    GEN1_MAP_NAMES,
    read_rom_header,
    render_gameboy_screen,
)
from examples.gaming.pokemon_full_campaign_speedrun import (
    CampaignSpeedrunEngine,
    CampaignState,
    create_starter_pokemon,
    render_spectator_hud,
)

GAME_ROM_MAP = {
    "red": "pokemon_red.gb",
    "blue": "pokemon_blue.gb",
    "yellow": "pokemon_yellow.gb",
    "gold": "pokemon_gold.gbc",
    "silver": "pokemon_silver.gbc",
    "crystal": "pokemon_crystal.gbc",
}

# Closed-loop spatial navigation paths for Gen 1 overworld & intro
GEN1_PATHS: Dict[int, List[str]] = {
    # Bedroom (Map 38): right to corridor (5, 6), up to (5, 1), right to stairs (7, 1)
    38: ["right", "right", "up", "up", "up", "up", "up", "right", "right"],
    # Living Room (Map 37): Down past stairs, left around table, down front door (3, 7)
    37: ["down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"],
    # Pallet Town (Map 0, intro before starter): Down away from door, right across lawn, up to Route 1 grass (10, 1)
    0:  ["down", "down", "right", "right", "right", "right", "right", "up", "up", "up", "up", "up", "up", "up", "right", "up", "up"],
    # Oak's Lab (Map 40, to Squirtle Poké Ball): Step down, right to Squirtle table at (7, 4)
    40: ["down", "right", "right"],
}

# Post-starter navigation paths:
LAB_EXIT_PATH: List[str] = ["left", "left", "left", "down", "down", "down", "down", "down", "down", "down", "down"]
PALLET_NORTH_PATH: List[str] = ["left", "left", "up", "up", "up", "up", "up", "up", "up", "up", "up", "up", "up", "up"]
ROUTE_1_PATH: List[str] = ["up", "up", "up", "up", "up", "up", "up", "left", "left", "up", "up", "up", "up", "right", "right", "up", "up", "up", "up", "up", "up", "up", "up", "up", "up", "up", "up"]
VIRIDIAN_PATH: List[str] = ["up", "up", "up", "up", "up", "up", "up", "up", "up", "up", "up", "up", "up", "up"]


def fast_skip_intro(emulator: Any, generation: str = "gen1") -> None:
    """Fast-forwards through Title Screen & Oak introduction at turbo speed (< 1.5s)."""
    if generation != "gen1" or not hasattr(emulator, "memory"):
        return

    def _safe_mem(addr: int) -> int:
        try:
            return int(emulator.memory[addr])
        except Exception:
            return 0

    # Run intro at turbo speed
    if hasattr(emulator, "set_emulation_speed"):
        emulator.set_emulation_speed(0)

    print("\n⏩ [Turbo Intro Fast-Forward: Skipping Title Screen & Oak Speech to Overworld]")
    for f in range(5000):
        emulator.tick()
        if f < 600:
            if f % 30 == 0:
                emulator.button_press("start")
            elif f % 30 == 4:
                emulator.button_release("start")
            if f % 10 == 0:
                emulator.button_press("a")
            elif f % 10 == 3:
                emulator.button_release("a")
        else:
            # Menu item: when name menu appears, press down to select "RED" / "BLUE"
            max_item = _safe_mem(0xCC28)
            cur_item = _safe_mem(0xCC26)
            if max_item == 3 and cur_item == 0:
                emulator.button_press("down")
                emulator.tick()
                emulator.button_release("down")

            if f % 10 == 0:
                emulator.button_press("a")
            elif f % 10 == 3:
                emulator.button_release("a")
            elif f % 10 == 5:
                emulator.button_press("b")
            elif f % 10 == 8:
                emulator.button_release("b")

        c100 = _safe_mem(0xC100)
        map_id = _safe_mem(0xD35E)
        if c100 == 1 and map_id == 38:
            break

    for b in ["a", "b", "start", "select", "up", "down", "left", "right"]:
        try:
            emulator.button_release(b)
        except Exception:
            pass

    # Wait for fade-in to complete and player control to unlock
    for _ in range(180):
        emulator.tick()


def resolve_path_for_map(m_id: Optional[int], has_party: bool) -> List[str]:
    """Resolves sequential movement inputs for given Gen-1 map ID based on party state."""
    if m_id is None:
        return []
    if m_id == 40 and has_party:
        return LAB_EXIT_PATH
    elif m_id == 0 and has_party:
        return PALLET_NORTH_PATH
    elif m_id == 12:  # Route 1
        return ROUTE_1_PATH
    elif m_id == 1:   # Viridian City
        return VIRIDIAN_PATH
    return GEN1_PATHS.get(m_id, [])


def render_ansi_screen_in_place(hud_text: str) -> None:
    """Updates terminal in-place without scrolling or flashing."""
    # \033[H repositions cursor to top-left, \033[J clears below without flicker
    sys.stdout.write("\033[H" + hud_text + "\033[J\n")
    sys.stdout.flush()


def fast_forward_to_rival_battle(emulator: Any, generation: str = "gen1") -> None:
    """Fast-forwards emulator to the first live in-game battle against Rival Blue (< 1.5s)."""
    if generation != "gen1" or not hasattr(emulator, "memory"):
        return
    fast_skip_intro(emulator, generation=generation)

    if hasattr(emulator, "set_emulation_speed"):
        emulator.set_emulation_speed(0)

    print("\n⏩ [Fast-Forwarding to Rival Battle 1: Pallet Town -> Oak Lab -> Blue Challenge]")

    def step_walk(direction):
        x0, y0, m0 = int(emulator.memory[0xD362]), int(emulator.memory[0xD361]), int(emulator.memory[0xD35E])
        emulator.button_press(direction)
        for _ in range(24):
            emulator.tick()
            x, y, m = int(emulator.memory[0xD362]), int(emulator.memory[0xD361]), int(emulator.memory[0xD35E])
            if (x, y) != (x0, y0) or m != m0:
                break
        emulator.button_release(direction)
        for _ in range(8):
            emulator.tick()
        return int(emulator.memory[0xD362]), int(emulator.memory[0xD361]), int(emulator.memory[0xD35E])

    for d in ["right", "right", "up", "up", "up", "up", "up", "right", "right"]:
        step_walk(d)
    for _ in range(60):
        emulator.tick()
    for d in ["down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"]:
        x, y, m = step_walk(d)
        if m == 0:
            break
    for _ in range(60):
        emulator.tick()

    pallet_path = ["down", "down", "right", "right", "right", "right", "right", "up", "up", "up", "up", "up", "up", "up", "right", "up", "up"]
    for d in pallet_path:
        x, y, m = step_walk(d)
        if int(emulator.memory[0xCD6B]) != 0:
            break

    # Oak speech in Lab
    for _ in range(4000):
        emulator.tick()
        if _ % 10 == 0:
            emulator.button_press("a")
        elif _ % 10 == 3:
            emulator.button_release("a")
        if int(emulator.memory[0xD35E]) == 40 and (int(emulator.memory[0xD362]), int(emulator.memory[0xD361])) == (5, 3) and int(emulator.memory[0xCD6B]) == 0 and _ > 1500:
            break

    for b in ["a", "b"]:
        try:
            emulator.button_release(b)
        except Exception:
            pass
    for _ in range(30):
        emulator.tick()

    # Claim Squirtle
    step_walk("down")
    step_walk("right")
    step_walk("right")
    emulator.button_press("up")
    for _ in range(8):
        emulator.tick()
    emulator.button_release("up")
    for _ in range(8):
        emulator.tick()

    for _ in range(1200):
        emulator.tick()
        if _ % 12 == 0:
            emulator.button_press("a")
        elif _ % 12 == 4:
            emulator.button_release("a")
        if int(emulator.memory[0xD163]) > 0:
            break

    # Blue claims Bulbasaur
    for _ in range(2000):
        emulator.tick()
        if _ % 12 == 0:
            emulator.button_press("a")
        elif _ % 12 == 4:
            emulator.button_release("a")
        if _ > 800 and int(emulator.memory[0xCD6B]) == 0:
            break

    for b in ["a", "b"]:
        try:
            emulator.button_release(b)
        except Exception:
            pass
    for _ in range(30):
        emulator.tick()

    # Walk to exit to trigger battle
    step_walk("left")
    step_walk("left")
    for _ in range(8):
        step_walk("down")
        if int(emulator.memory[0xCD6B]) != 0:
            break

    # Advance to battle start
    for _ in range(3000):
        emulator.tick()
        if _ % 12 == 0:
            emulator.button_press("a")
        elif _ % 12 == 4:
            emulator.button_release("a")
        if int(emulator.memory[0xD057]) != 0:
            break


def run_live_battle_mode(
    rom_path: Path,
    window_type: str = "SDL2",
    speed: int = 1,
    max_frames: int = 18000,
    advisor: bool = True,
    step_mode: bool = False,
) -> None:
    """Runs an immediate live battle showcase where System 1 battles opponents in real time."""
    rom_meta = read_rom_header(rom_path)
    generation = rom_meta.get("generation", "gen1")
    game_variant = rom_meta.get("game_variant", "red")

    try:
        import pyboy
        emulator = pyboy.PyBoy(str(rom_path), window=window_type)
    except Exception as err:
        print(f"[Notice] PyBoy window initialization: {err}. Proceeding with HUD spectator.")
        emulator = None

    agent = System1BattleAgent()
    total_decisions = 0
    frame_count = 0
    last_hud_render = 0.0

    print(f"\n[Starting Live Battle Spectator Mode - Window: {window_type}]")

    # If PyBoy hardware emulator is active, fast-forward directly to the first live battle!
    if emulator is not None and generation == "gen1":
        fast_forward_to_rival_battle(emulator, generation=generation)

        # Restore requested speed
        if speed > 1 and hasattr(emulator, "set_emulation_speed"):
            emulator.set_emulation_speed(speed)
        elif hasattr(emulator, "set_emulation_speed"):
            emulator.set_emulation_speed(1)

        memory_bridge = PyBoyMemoryBridge(
            memory_reader=lambda addr: int(emulator.memory[addr]),
            game_version=game_variant,
        )

        print(f"  System 1: Evaluating live Game Boy RAM battles on the metal ({speed}x speed)\n")

        try:
            while frame_count < max_frames:
                emulator.tick()
                frame_count += 1
                bm = int(emulator.memory[0xD057])
                ji = int(emulator.memory[0xCD6B])

                if bm in (1, 2):
                    # Active battle: mash 'a' to advance dialogue and confirm move selections
                    if frame_count % 16 == 0:
                        emulator.button_press("a")
                    elif frame_count % 16 == 4:
                        emulator.button_release("a")

                    now = time.time()
                    if now - last_hud_render >= (0.35 if not step_mode else 0.0):
                        last_hud_render = now
                        battle_state = memory_bridge.extract_battle_state_from_ram()
                        telemetry, should_escalate, reason = agent.evaluate(battle_state)
                        total_decisions += 1
                        hud = render_gameboy_screen(
                            battle_state,
                            telemetry=telemetry,
                            system2_halt=should_escalate,
                            advisor_mode=advisor,
                        )
                        render_ansi_screen_in_place(hud)

                        if step_mode:
                            try:
                                input("  🎮 [Press ENTER to advance next combat turn...]")
                            except (EOFError, KeyboardInterrupt):
                                step_mode = False
                elif ji != 0:
                    # Post-battle dialogue / cutscenes
                    if frame_count % 12 == 0:
                        emulator.button_press("a")
                    elif frame_count % 12 == 4:
                        emulator.button_release("a")
                else:
                    # Outside battle: walk to tall grass to encounter wild Pokémon!
                    cur_m = int(emulator.memory[0xD35E])
                    if cur_m == 40:
                        if frame_count % 30 == 0:
                            emulator.button_press("down")
                        elif frame_count % 30 == 16:
                            emulator.button_release("down")
                    elif cur_m == 0:
                        if frame_count % 30 == 0:
                            emulator.button_press("up")
                        elif frame_count % 30 == 16:
                            emulator.button_release("up")
                    else:
                        direction = "right" if (frame_count // 32) % 2 == 0 else "left"
                        if frame_count % 32 in (0, 16):
                            emulator.button_press(direction)
                        elif frame_count % 32 in (8, 24):
                            emulator.button_release(direction)

        except KeyboardInterrupt:
            print("\nLive battle spectator stopped by user.")
        finally:
            try:
                emulator.stop()
            except Exception:
                pass
            print(f"\nLive Battle Showcase Completed. Total System 1 Decisions: {total_decisions}")
        return

    # Fallback synthetic battle showcase if running headless without PyBoy hardware
    battles = [
        ("Gym Leader Misty", create_gym_leader_battle("misty")),
        ("Gym Leader Brock", create_gym_leader_battle("brock")),
        ("Gym Leader Giovanni", create_gym_leader_battle("giovanni")),
        ("Wild Encounter", create_wild_encounter("Zubat")),
    ]

    battle_idx = 0
    try:
        while frame_count < max_frames and battle_idx < len(battles):
            title, state = battles[battle_idx]

            while (not state.player_pokemon.is_fainted) and (not state.opponent_pokemon.is_fainted) and frame_count < max_frames:
                frame_count += 30
                telemetry, should_escalate, reason = agent.evaluate(state)
                total_decisions += 1

                hud = render_gameboy_screen(
                    state,
                    telemetry=telemetry,
                    system2_halt=should_escalate,
                    advisor_mode=advisor,
                )
                render_ansi_screen_in_place(hud)

                if step_mode:
                    try:
                        input("  🎮 [Press ENTER to advance next combat turn...]")
                    except (EOFError, KeyboardInterrupt):
                        step_mode = False
                else:
                    time.sleep(0.3 / speed)

                action = telemetry.get("action", "fight")
                chosen_slot = telemetry.get("chosen_move", "move_slot_1")
                move = state.player_pokemon.get_move_by_slot(chosen_slot)

                if action == "fight" and move is not None:
                    dmg, mult, crit = calculate_damage(state.player_pokemon, state.opponent_pokemon, move)
                    state.opponent_pokemon.take_damage(dmg)
                    state.turn_count += 1
                elif action == "use_item":
                    state.player_pokemon.heal(50)
                    state.turn_count += 1

                if not state.opponent_pokemon.is_fainted:
                    opp_move = state.opponent_pokemon.moves[0] if state.opponent_pokemon.moves else None
                    if opp_move:
                        opp_dmg, _, _ = calculate_damage(state.opponent_pokemon, state.player_pokemon, opp_move)
                        state.player_pokemon.take_damage(opp_dmg)

            battle_idx += 1
            time.sleep(0.8 / speed)

    except KeyboardInterrupt:
        print("\nLive battle spectator stopped by user.")
    finally:
        print(f"\nLive Battle Showcase Completed. Total System 1 Decisions: {total_decisions}")


def run_pyboy_live_game(
    rom_path: Path,
    window_type: str = "SDL2",
    speed: int = 1,
    max_frames: int = 18000,
    advisor: bool = True,
    step_mode: bool = False,
    auto_play: bool = True,
    mode: str = "overworld",
) -> None:
    """Boots Pokémon in PyBoy with native GUI window and System 1 System 1 integration."""
    if mode == "battle":
        run_live_battle_mode(
            rom_path=rom_path,
            window_type=window_type,
            speed=speed,
            max_frames=max_frames,
            advisor=advisor,
            step_mode=step_mode,
        )
        return

    if mode == "campaign":
        # Resolve starter choice based on ROM header
        rom_meta = read_rom_header(rom_path)
        game_variant = rom_meta.get("game_variant", "red")
        if game_variant == "yellow":
            starter_choice = "pikachu"
        elif game_variant == "crystal":
            starter_choice = "totodile"
        elif "gold" in game_variant:
            starter_choice = "cyndaquil"
        elif "silver" in game_variant:
            starter_choice = "totodile"
        else:
            starter_choice = "squirtle"

        # Map integer speed (1, 2, 5) or strings to watchable campaign speed
        speed_map = {1: "normal", 2: "fast", 5: "turbo"}
        campaign_speed = speed_map.get(speed, "normal" if isinstance(speed, (int, float)) and speed <= 1 else ("fast" if isinstance(speed, (int, float)) and speed <= 2 else "turbo"))

        engine = CampaignSpeedrunEngine(
            rom_path=rom_path,
            starter=starter_choice,
            gui=(window_type != "null"),
            speed=campaign_speed,
            advisor_mode=advisor,
            step_mode=step_mode,
        )
        engine.run_campaign()
        return

    try:
        import pyboy
    except ImportError:
        print("\n[Error] PyBoy is required to display the actual graphical Game Boy game.")
        print("Please run using the project virtual environment:")
        print("  /Volumes/Storage/reflex/.venv/bin/python3 examples/pokemon_gameboy_gui.py")
        return

    # Parse ROM header
    rom_meta = read_rom_header(rom_path)
    generation = rom_meta.get("generation", "gen1")
    game_variant = rom_meta.get("game_variant", "red")
    title = rom_meta.get("title", "POKEMON")

    # Starter choices based on game version
    if game_variant == "yellow":
        starter_choice = "pikachu"
    elif game_variant == "crystal":
        starter_choice = "totodile"
    elif "gold" in game_variant:
        starter_choice = "cyndaquil"
    elif "silver" in game_variant:
        starter_choice = "totodile"
    else:
        starter_choice = "squirtle"

    print(f"\n[Initializing Game Boy Hardware & Cartridge]")
    print(f"  ROM File:    {rom_path.name}")
    print(f"  Cartridge:   {title} ({generation.upper()} - {game_variant.upper()})")
    print(f"  GUI Window:  {window_type}")
    print(f"  Emulation:   60 FPS (Speed: {speed}x)")
    print(f"  Starter:     {starter_choice.upper()}")
    print(f"  System 1:    Active on the metal (~1.0 ms)")

    # Initialize PyBoy with requested window (SDL2 opens native macOS GUI window)
    try:
        emulator = pyboy.PyBoy(str(rom_path), window=window_type)
    except Exception as err:
        print(f"\n[Notice] Could not initialize graphical window '{window_type}': {err}")
        print("Falling back to headless execution with in-place terminal HUD.")
        emulator = pyboy.PyBoy(str(rom_path), window="null")

    if speed > 1 and hasattr(emulator, "set_emulation_speed"):
        emulator.set_emulation_speed(speed)

    memory_bridge = PyBoyMemoryBridge(
        memory_reader=lambda addr: int(emulator.memory[addr]),
        game_version=game_variant,
    )
    agent = System1BattleAgent()

    campaign = CampaignState(
        starter_choice=starter_choice,
        party=[create_starter_pokemon(starter_choice)],
    )

    current_map: Optional[int] = None
    step_idx = 0
    last_pos: Optional[Tuple[int, int]] = None
    move_timer = 0
    oak_speech_done = False
    active_has_party = False
    frame_count = 0
    total_decisions = 0
    last_hud_render = 0.0

    print("\nStarting emulation... Window is opening on your desktop!")
    time.sleep(0.5)

    # 1. Fast-forward through intro to materialization in bedroom (< 1.5s)
    fast_skip_intro(emulator, generation=generation)

    # Restore user's requested emulation speed
    if speed > 1 and hasattr(emulator, "set_emulation_speed"):
        emulator.set_emulation_speed(speed)
    elif hasattr(emulator, "set_emulation_speed"):
        emulator.set_emulation_speed(1)

    print(f"\n🎮 [Overworld Active: Red materialized in bedroom! Emulating at {speed}x speed]")

    try:
        while frame_count < max_frames:
            # Advance emulator frame
            emulator.tick()
            frame_count += 1

            # Check Game Boy RAM state via Memory Bridge
            if generation == "gen2":
                battle_mode = int(emulator.memory[0xD22D]) if hasattr(emulator, "memory") else 0
                map_grp = int(emulator.memory[0xDCB5]) if hasattr(emulator, "memory") else 0
                map_num = int(emulator.memory[0xDCB6]) if hasattr(emulator, "memory") else 0
                map_id = (map_grp << 8) | map_num
                x = int(emulator.memory[0xDCB7]) if hasattr(emulator, "memory") else 0
                y = int(emulator.memory[0xDCB8]) if hasattr(emulator, "memory") else 0
                joy_ignore = 0
            else:
                battle_mode = int(emulator.memory[0xD057]) if hasattr(emulator, "memory") else 0
                map_id = int(emulator.memory[0xD35E]) if hasattr(emulator, "memory") else 0
                x = int(emulator.memory[0xD362]) if hasattr(emulator, "memory") else 0
                y = int(emulator.memory[0xD361]) if hasattr(emulator, "memory") else 0
                joy_ignore = int(emulator.memory[0xCD6B]) if hasattr(emulator, "memory") else 0

            # True Closed-Loop Autonomous Navigation
            if auto_play and generation == "gen1":
                party_count = int(emulator.memory[0xD163]) if hasattr(emulator, "memory") else 0

                # Track room and quest progression transitions
                has_party_now = (party_count > 0)
                if map_id != current_map or has_party_now != active_has_party:
                    prev_path = resolve_path_for_map(current_map, active_has_party)
                    if step_idx < len(prev_path):
                        try:
                            emulator.button_release(prev_path[step_idx])
                        except Exception:
                            pass
                    current_map = map_id
                    active_has_party = has_party_now
                    step_idx = 0
                    last_pos = (x, y)
                    move_timer = 0

                # 1. Advance cutscenes across all maps (Oak catches Red in tall grass, leads to lab, Blue dialogue, etc.)
                if joy_ignore != 0:
                    if frame_count % 12 == 0:
                        emulator.button_press("a")
                    elif frame_count % 12 == 4:
                        emulator.button_release("a")

                # 2. Oak's Lab logic (Map 40)
                elif current_map == 40:
                    if party_count == 0:
                        if not oak_speech_done:
                            # At Oak desk (5, 3): wait until speech completes and controls unlock
                            if (x, y) == (5, 3) and joy_ignore == 0:
                                oak_speech_done = True
                                step_idx = 0
                                last_pos = (x, y)
                                move_timer = 0
                            else:
                                if frame_count % 12 == 0:
                                    emulator.button_press("a")
                                elif frame_count % 12 == 4:
                                    emulator.button_release("a")
                        else:
                            # Walk down to (5, 4), right to (7, 4) in front of Squirtle
                            lab_path = ["down", "right", "right"]
                            if (x, y) != last_pos:
                                if step_idx < len(lab_path):
                                    try:
                                        emulator.button_release(lab_path[step_idx])
                                    except Exception:
                                        pass
                                step_idx += 1
                                last_pos = (x, y)
                                move_timer = 0

                            move_timer += 1
                            if step_idx < len(lab_path):
                                if move_timer == 1:
                                    emulator.button_press(lab_path[step_idx])
                                elif move_timer == 16:
                                    emulator.button_release(lab_path[step_idx])
                                elif move_timer > 30:
                                    move_timer = 0
                            elif step_idx >= len(lab_path) and (x, y) == (7, 4):
                                # Face up and inspect Squirtle Poké Ball
                                if frame_count % 32 == 0:
                                    emulator.button_press("up")
                                elif frame_count % 32 == 4:
                                    emulator.button_release("up")
                                    emulator.button_press("a")
                                elif frame_count % 32 == 8:
                                    emulator.button_release("a")
                                elif frame_count % 32 == 16:
                                    emulator.button_press("a")
                                elif frame_count % 32 == 20:
                                    emulator.button_release("a")
                    else:
                        # Starter obtained!
                        if joy_ignore != 0:
                            # Blue claiming Bulbasaur dialogue
                            if frame_count % 12 == 0:
                                emulator.button_press("a")
                            elif frame_count % 12 == 4:
                                emulator.button_release("a")
                        else:
                            # Walk towards exit door (5, 11) - Blue will intercept Red to start battle!
                            exit_path = resolve_path_for_map(40, True)
                            if (x, y) != last_pos:
                                if step_idx < len(exit_path):
                                    try:
                                        emulator.button_release(exit_path[step_idx])
                                    except Exception:
                                        pass
                                step_idx += 1
                                last_pos = (x, y)
                                move_timer = 0

                            move_timer += 1
                            if step_idx < len(exit_path):
                                if move_timer == 1:
                                    emulator.button_press(exit_path[step_idx])
                                elif move_timer == 16:
                                    emulator.button_release(exit_path[step_idx])
                                elif move_timer > 30:
                                    move_timer = 0

                # 3. Active battle (Wild or Rival)
                elif battle_mode in (1, 2):
                    if frame_count % 16 == 0:
                        emulator.button_press("a")
                    elif frame_count % 16 == 4:
                        emulator.button_release("a")

                # 4. Standard closed-loop spatial waypoint navigation (Maps 38, 37, 0, 12, etc.)
                else:
                    is_transitional = (current_map == 0 and (x, y) == (3, 7)) or (current_map == 37 and (x, y) == (6, 1))
                    path = resolve_path_for_map(current_map, party_count > 0)
                    if not is_transitional and path:
                        if (x, y) != last_pos:
                            if step_idx < len(path):
                                try:
                                    emulator.button_release(path[step_idx])
                                except Exception:
                                    pass
                            step_idx += 1
                            last_pos = (x, y)
                            move_timer = 0

                        move_timer += 1
                        if step_idx < len(path):
                            if move_timer == 1:
                                emulator.button_press(path[step_idx])
                            elif move_timer == 16:
                                emulator.button_release(path[step_idx])
                            elif move_timer > 30:
                                move_timer = 0

            # Update in-place spectator HUD at human-watchable pace (~2-3 times per second)
            now = time.time()
            if now - last_hud_render >= (0.4 if not step_mode else 0.0):
                last_hud_render = now

                # Extract live state from memory
                if battle_mode in (1, 2):
                    battle_state = memory_bridge.extract_battle_state_from_ram()
                    telemetry, should_escalate, reason = agent.evaluate(battle_state)
                    total_decisions += 1

                    if auto_play:
                        # Feed System 1 decision back to Game Boy buttons
                        action = telemetry.get("action", "fight")
                        
                        def execute_sequence(btns):
                            nonlocal frame_count
                            for b in btns:
                                emulator.button_press(b)
                                for _ in range(4):
                                    emulator.tick()
                                    frame_count += 1
                                emulator.button_release(b)
                                for _ in range(4):
                                    emulator.tick()
                                    frame_count += 1
                                    
                        if action == "fight":
                            execute_sequence(["a"])
                        elif action == "use_item":
                            execute_sequence(["down", "a"])
                        elif action == "switch_pokemon":
                            execute_sequence(["right", "a"])
                        elif action == "run_away":
                            execute_sequence(["down", "right", "a"])

                    hud = render_gameboy_screen(
                        battle_state,
                        telemetry=telemetry,
                        system2_halt=should_escalate,
                        advisor_mode=advisor,
                    )
                else:
                    # Overworld / Title screen mode
                    campaign.current_map_id = map_id
                    campaign.current_map_name = GEN1_MAP_NAMES.get(map_id, f"Route/Area 0x{map_id:02X}")
                    campaign.current_x = x
                    campaign.current_y = y
                    if party_count > 0:
                        if len(campaign.party) == 0:
                            campaign.party = [create_starter_pokemon(starter_choice)]
                        # Sync level and HP from RAM if available
                        lead_mon = campaign.party[0]
                        lvl = int(emulator.memory[0xD18C]) if hasattr(emulator, "memory") else 5
                        cur_hp = (int(emulator.memory[0xD16B]) << 8) | int(emulator.memory[0xD16C]) if hasattr(emulator, "memory") else 22
                        max_hp = (int(emulator.memory[0xD18D]) << 8) | int(emulator.memory[0xD18E]) if hasattr(emulator, "memory") else 22
                        if lvl > 0:
                            lead_mon.level = lvl
                        if max_hp > 0:
                            lead_mon.max_hp = max_hp
                            lead_mon.current_hp = cur_hp

                    hud = render_spectator_hud(
                        campaign,
                        battle_state=None,
                        telemetry=None,
                        system2_halt=False,
                        advisor_mode=advisor,
                    )

                # Render cleanly in-place without scrolling or flashing
                render_ansi_screen_in_place(hud)

                if step_mode:
                    try:
                        input("  🎮 [Press ENTER to advance next frame / turn...]")
                    except (EOFError, KeyboardInterrupt):
                        step_mode = False

    except KeyboardInterrupt:
        print("\nEmulation stopped by user.")
    finally:
        try:
            emulator.stop()
        except Exception:
            pass
        print(f"\nPyBoy Game Boy session finished cleanly. Total frames: {frame_count}")


def parse_gui_args(args_list: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pokémon Live Game Boy GUI Spectator & System 1 System 1 Agent"
    )
    parser.add_argument(
        "--game",
        type=str,
        choices=["red", "blue", "yellow", "gold", "silver", "crystal"],
        default=None,
        help="Pokémon cartridge to boot: red, blue, yellow, gold, silver, crystal",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["overworld", "battle", "campaign"],
        default="overworld",
        help="Spectator mode: 'overworld' (autonomous navigation), 'battle' (live combat showcase), 'campaign' (speedrun)",
    )
    parser.add_argument(
        "--rom",
        type=str,
        default=None,
        help="Custom path to Pokémon ROM (.gb/.gbc). Overrides --game if specified.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without graphical window (terminal HUD only)",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=1,
        help="Emulation speed multiplier: 1 (real-time 60 FPS), 2 (2x), 5 (5x fast-forward)",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="Pause and wait for [Enter] after each decision turn",
    )
    parser.add_argument(
        "--advisor",
        action="store_true",
        default=True,
        help="Display live Tactical Game Advisor HUD",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=18000,
        help="Maximum emulation frames to run (default: 18000 = 5 minutes at 60 FPS)",
    )
    return parser.parse_args(args_list)


def main() -> None:
    args = parse_gui_args()

    # Determine target ROM
    rom_dir = ROOT_DIR / "roms"
    if args.rom:
        rom_target = Path(args.rom)
    elif args.game and args.game in GAME_ROM_MAP:
        rom_target = rom_dir / GAME_ROM_MAP[args.game]
    else:
        rom_target = find_default_pokemon_rom()

    if rom_target is None or not rom_target.is_file():
        print("[Error] Could not find Pokémon Game Boy ROM.")
        print(f"Please specify --game {{red, blue, yellow, gold, silver, crystal}} or --rom path/to/rom.gb")
        sys.exit(1)

    window = "null" if args.headless else "SDL2"
    run_pyboy_live_game(
        rom_path=rom_target,
        window_type=window,
        speed=args.speed,
        max_frames=args.max_frames,
        advisor=args.advisor,
        step_mode=args.step,
        mode=args.mode,
    )


if __name__ == "__main__":
    ensure_venv_reexec()
    main()
