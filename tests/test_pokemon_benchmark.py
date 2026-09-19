"""Unit and Integration Tests for Pokémon Multi-Game Headless Benchmark & GUI Runner.

Verifies:
1. Extended ROM header parsing, platform identification, and 8-bit checksum verification.
2. Latency statistics calculation (mean, median, p95, p99, min, max, QPS throughput).
3. Memory Bridge integrity verification across Gen 1 and Gen 2 cartridges.
4. System 1 System 1 inference evaluation and sub-5ms metal verification.
5. Multi-game benchmark end-to-end execution.
6. ANSI comparison table formatting and JSON serialization.
7. CLI argument parsing for both benchmark and live GUI spectator scripts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
try:
    import pytest
except ImportError:
    pytest = None


REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
ROMS_DIR = REPO_ROOT / "roms"
ROMS_PRESENT = (ROMS_DIR / "pokemon_red.gb").is_file()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from examples.gaming.pokemon_all_games_benchmark import (
    ALL_POKEMON_GAMES,
    CartridgeMetadata,
    LatencyMetrics,
    MemoryBridgeIntegrity,
    SystemOnePerformance,
    SingleGameBenchmarkResult,
    compute_latency_stats,
    evaluate_memory_bridge_integrity,
    evaluate_system1_performance,
    format_ansi_comparison_table,
    parse_benchmark_args,
    parse_extended_rom_header,
    results_to_json_dict,
    run_multi_game_benchmark,
    run_single_game_benchmark,
)
from examples.gaming.pokemon_battle_system1 import (
    BattleState,
    BattleType,
    Pokemon,
    PokemonMove,
    System1BattleAgent,
    create_gym_leader_battle,
    create_wild_encounter,
    read_rom_header,
)
from examples.gaming.pokemon_gameboy_gui import GAME_ROM_MAP, fast_skip_intro, parse_gui_args


# ============================================================================
# 1. ROM Header Parsing & Checksum Tests
# ============================================================================

def test_all_six_roms_exist():
    """Verify all 6 Game Boy cartridges are present in the roms/ directory."""
    if not ROMS_PRESENT:
        pytest.skip("Game Boy ROMs not present in environment (ROM dumps are gitignored)")
    for key, filename in ALL_POKEMON_GAMES.items():
        rom_path = ROMS_DIR / filename
        assert rom_path.is_file(), f"Missing cartridge: {rom_path}"
        assert rom_path.stat().st_size in (1024 * 1024, 2 * 1024 * 1024)


def test_parse_extended_rom_header():
    """Verify extended header parsing and 8-bit checksums across all 6 cartridges."""
    if not ROMS_PRESENT:
        pytest.skip("Game Boy ROMs not present in environment (ROM dumps are gitignored)")
    expected_meta = {
        "red": {"gen": "GEN1", "platform": "DMG/SGB", "rom_kb": 1024, "checksum": "0x20"},
        "blue": {"gen": "GEN1", "platform": "DMG/SGB", "rom_kb": 1024, "checksum": "0xD3"},
        "yellow": {"gen": "GEN1", "platform": "CGB/DMG", "rom_kb": 1024, "checksum": "0x97"},
        "gold": {"gen": "GEN2", "platform": "CGB/DMG", "rom_kb": 2048, "checksum": "0x4B"},
        "silver": {"gen": "GEN2", "platform": "CGB/DMG", "rom_kb": 2048, "checksum": "0x2A"},
        "crystal": {"gen": "GEN2", "platform": "CGB-Only", "rom_kb": 2048, "checksum": "0x26"},
    }

    for game_key, filename in ALL_POKEMON_GAMES.items():
        rom_path = ROMS_DIR / filename
        meta = parse_extended_rom_header(rom_path, game_key)
        exp = expected_meta[game_key]

        assert meta.game_key == game_key
        assert meta.generation == exp["gen"]
        assert meta.platform == exp["platform"]
        assert meta.rom_size_kb == exp["rom_kb"]
        assert meta.checksum_valid is True
        assert meta.checksum_hex.upper() == exp["checksum"].upper()
        assert "POKEMON" in meta.title or "PM_" in meta.title


def test_synthetic_header_checksum_validation(tmp_path: Path):
    """Verify checksum validator rejects corrupt headers."""
    dummy_rom = tmp_path / "corrupt_rom.gb"
    raw = bytearray(0x200)
    raw[0x134:0x143] = b"TEST ROM\x00"
    raw[0x147] = 0x00  # ROM ONLY
    raw[0x148] = 0x00  # 32KB
    raw[0x14D] = 0xAA  # Invalid checksum

    dummy_rom.write_bytes(raw)
    meta = parse_extended_rom_header(dummy_rom, "test")
    assert meta.checksum_valid is False


# ============================================================================
# 2. Latency Statistical Metric Computation Tests
# ============================================================================

def test_compute_latency_stats_basic():
    """Verify mean, median, min, max, percentiles, and QPS calculation."""
    # 10 measurements: 10, 20, 30, ..., 100 microseconds
    times = [float(i * 10) for i in range(1, 11)]
    stats = compute_latency_stats(times)

    assert stats.mean_us == 55.0
    assert stats.min_us == 10.0
    assert stats.max_us == 100.0
    assert stats.median_us == 60.0
    assert stats.p95_us == 100.0
    assert abs(stats.throughput_qps - (1_000_000.0 / 55.0)) < 5.0



def test_compute_latency_stats_empty():
    """Verify empty input returns clean zeroed metrics."""
    stats = compute_latency_stats([])
    assert stats.mean_us == 0
    assert stats.throughput_qps == 0


def test_compute_latency_stats_single():
    """Verify single-element input behavior."""
    stats = compute_latency_stats([42.5])
    assert stats.mean_us == 42.5
    assert stats.median_us == 42.5
    assert stats.min_us == 42.5
    assert stats.max_us == 42.5


# ============================================================================
# 3. RAM Memory Bridge Integrity Tests
# ============================================================================

def test_memory_bridge_integrity_across_all_games():
    """Verify PyBoyMemoryBridge extraction produces 10/10 passed checks on all cartridges."""
    if not ROMS_PRESENT:
        pytest.skip("Game Boy ROM dumps are gitignored and not present in CI environment")
    for game_key, filename in ALL_POKEMON_GAMES.items():
        rom_path = ROMS_DIR / filename
        meta = read_rom_header(rom_path)
        integrity = evaluate_memory_bridge_integrity(rom_path, meta["game_variant"])

        assert integrity.passed is True
        assert integrity.passed_checks == integrity.total_checks == 10
        assert integrity.player_hp > 0
        assert integrity.player_max_hp > 0
        assert integrity.player_hp <= integrity.player_max_hp
        assert 1 <= integrity.player_level <= 100
        assert integrity.move_count == 4
        assert len(integrity.pp_values) == 4
        assert all(pp >= 0 for pp in integrity.pp_values)
        assert integrity.player_status == "OK"
        assert integrity.player_species != ""
        assert integrity.opponent_species != ""
        assert len(integrity.issues) == 0


# ============================================================================
# 4. System 1 System 1 Inference Performance Tests
# ============================================================================

def test_system1_inference_performance():
    """Verify System 1 decision evaluation, sub-5ms verification, and conformal safety."""
    agent = System1BattleAgent()
    state = create_gym_leader_battle("misty")

    perf = evaluate_system1_performance(state, agent=agent, num_decisions=25)

    assert perf.total_decisions == 25
    assert perf.neural_forward.mean_us > 0
    assert perf.neural_forward.mean_us < 1000.0  # Sub-1ms metal verification!
    assert perf.sub_1ms_verified is True
    assert perf.neural_forward.throughput_qps > 1000.0
    assert perf.e2e_pipeline.mean_us > 0
    assert perf.conformal_avg_set_size >= 1.0
    assert 0.0 <= perf.conformal_ambiguity_rate <= 100.0


# ============================================================================
# 5. Single Game & Multi-Game Benchmark Execution Tests
# ============================================================================

def test_run_single_game_benchmark():
    """Verify single game benchmark returns valid SingleGameBenchmarkResult."""
    if not ROMS_PRESENT:
        pytest.skip("Game Boy ROM dumps are gitignored and not present in CI environment")
    agent = System1BattleAgent()
    res = run_single_game_benchmark(
        game_key="red",
        rom_filename="pokemon_red.gb",
        rom_dir=ROMS_DIR,
        frame_count=30,
        num_decisions=10,
        agent=agent,
    )

    assert isinstance(res, SingleGameBenchmarkResult)
    assert res.game_key == "red"
    if res.emulator_available:
        assert res.headless_fps > 0
    else:
        assert res.headless_fps >= 0.0
    assert res.memory_integrity.passed is True
    assert res.system1_performance.sub_1ms_verified is True


def test_run_multi_game_benchmark_subset():
    """Verify multi-game benchmark orchestrates a targeted subset of cartridges."""
    if not ROMS_PRESENT:
        pytest.skip("Game Boy ROM dumps are gitignored and not present in CI environment")
    results = run_multi_game_benchmark(
        games=["red", "gold"],
        rom_dir=ROMS_DIR,
        frames=30,
        decisions=10,
        quiet=True,
    )

    assert len(results) == 2
    keys = [r.game_key for r in results]
    assert keys == ["red", "gold"]
    for r in results:
        if r.emulator_available:
            assert r.headless_fps > 0
        else:
            assert r.headless_fps >= 0.0
        assert r.memory_integrity.passed is True



# ============================================================================
# 6. Table Formatting & JSON Export Tests
# ============================================================================

def test_format_ansi_comparison_table():
    """Verify ANSI table renders cleanly with headers, rows, and summary statistics."""
    if not ROMS_PRESENT:
        pytest.skip("Game Boy ROM dumps are gitignored and not present in CI environment")
    results = run_multi_game_benchmark(
        games=["red", "crystal"],
        rom_dir=ROMS_DIR,
        frames=20,
        decisions=5,
        quiet=True,
    )

    table = format_ansi_comparison_table(results)
    assert isinstance(table, str)
    assert "POKÉMON MULTI-CARTRIDGE HEADLESS BENCHMARK" in table
    assert "Red" in table
    assert "Crystal" in table
    assert "PASS (10/10)" in table
    assert "SUB-1MS VERIFIED" in table or "EXCEEDS 1MS" in table


def test_results_to_json_dict():
    """Verify JSON dictionary structure and serialization."""
    if not ROMS_PRESENT:
        pytest.skip("Game Boy ROM dumps are gitignored and not present in CI environment")
    results = run_multi_game_benchmark(
        games=["blue"],
        rom_dir=ROMS_DIR,
        frames=20,
        decisions=5,
        quiet=True,
    )

    json_dict = results_to_json_dict(results)
    assert "timestamp" in json_dict
    assert json_dict["total_games_tested"] == 1
    assert "summary" in json_dict
    assert json_dict["summary"]["all_memory_bridges_passed"] is True
    assert json_dict["summary"]["sub_1ms_verified"] is True
    assert len(json_dict["games"]) == 1

    # Verify JSON string dumps without TypeError
    dumped = json.dumps(json_dict)
    assert len(dumped) > 100


# ============================================================================
# 7. CLI Argument Parsing Tests
# ============================================================================

def test_parse_benchmark_args():
    """Verify CLI flag parsing for pokemon_all_games_benchmark.py."""
    args = parse_benchmark_args(["--games", "red", "yellow", "--frames", "50", "--decisions", "25", "--json"])
    assert args.games == ["red", "yellow"]
    assert args.frames == 50
    assert args.decisions == 25
    assert args.json is True


def test_parse_benchmark_args_defaults():
    """Verify default values for benchmark CLI flags."""
    args = parse_benchmark_args([])
    assert args.games == ["all"]
    assert args.frames == 300
    assert args.decisions == 100
    assert args.json is False
    assert args.quiet is False


def test_parse_gui_args():
    """Verify CLI flag parsing for pokemon_gameboy_gui.py."""
    args = parse_gui_args([
        "--game", "crystal",
        "--mode", "battle",
        "--headless",
        "--speed", "5",
        "--step",
        "--max-frames", "500",
    ])
    assert args.game == "crystal"
    assert args.mode == "battle"
    assert args.headless is True
    assert args.speed == 5
    assert args.step is True
    assert args.max_frames == 500
    assert args.advisor is True


def test_format_ansi_dynamic_metrics():
    """Verify ANSI table footer computes dynamic ms latency and integrity without hardcoding."""
    cart = CartridgeMetadata(
        game_key="red",
        filename="pokemon_red.gb",
        title="POKEMON RED",
        generation="GEN1",
        game_variant="red",
        platform="DMG/SGB",
        cartridge_type_hex="0x13",
        cartridge_type_desc="MBC3+RAM+BATTERY",
        rom_size_kb=1024,
        ram_size_desc="32 KB",
        checksum_valid=True,
        checksum_hex="0x20",
    )
    integrity = MemoryBridgeIntegrity(
        passed=True,
        total_checks=10,
        passed_checks=10,
        player_species="Squirtle",
        player_hp=22,
        player_max_hp=22,
        player_level=5,
        move_count=4,
        moves=["Tackle", "Tail Whip", "Water Gun", "Bubble"],
        pp_values=[35, 30, 25, 30],
        player_status="OK",
        opponent_species="Bulbasaur",
        opponent_hp=21,
        opponent_max_hp=21,
        map_id=0,
        map_name="Pallet Town",
        player_x=5,
        player_y=5,
        battle_mode=0,
    )
    perf = SystemOnePerformance(
        neural_forward=compute_latency_stats([50.0]),
        e2e_pipeline=compute_latency_stats([120.0]),
        sub_1ms_verified=True,
        total_decisions=1,
        conformal_avg_set_size=2.0,
        conformal_ambiguity_rate=0.0,
        escalation_rate=0.0,
        escalation_count=0,
    )
    res = SingleGameBenchmarkResult(
        game_key="red",
        cartridge=cart,
        headless_fps=5000.0,
        frame_count=100,
        frame_time_sec=0.02,
        memory_integrity=integrity,
        system1_performance=perf,
        emulator_available=True,
    )

    table = format_ansi_comparison_table([res])
    assert "SUB-1MS VERIFIED" in table or "EXCEEDS 1MS" in table
    assert "~0.05 ms" in table
    assert "1/1 PASS" in table
    assert "100% integrity across cartridges" in table


def test_gameboy_gui_game_rom_mapping():
    """Verify GAME_ROM_MAP maps all 6 game keys to valid files."""
    if not ROMS_PRESENT:
        pytest.skip("Game Boy ROM dumps are gitignored and not present in CI environment")
    for key, filename in GAME_ROM_MAP.items():
        rom_path = ROMS_DIR / filename
        assert rom_path.is_file(), f"File mapped to {key} does not exist: {rom_path}"


# ============================================================================
# 8. Fast Intro Skipping Tests (Feature F16)
# ============================================================================

def test_fast_skip_intro_non_gen1_early_return():
    """Verify fast_skip_intro returns immediately without emulator interaction if generation != 'gen1'."""
    class MockEmulator:
        def __init__(self):
            self.memory: Dict[int, int] = {0xC100: 0}
            self.actions: List[str] = []

        def tick(self) -> None:
            self.actions.append("tick")

        def set_emulation_speed(self, speed: int) -> None:
            self.actions.append(f"speed_{speed}")

        def button_press(self, btn: str) -> None:
            self.actions.append(f"press_{btn}")

        def button_release(self, btn: str) -> None:
            self.actions.append(f"release_{btn}")

    mock = MockEmulator()
    fast_skip_intro(mock, generation="gen2")
    assert len(mock.actions) == 0

    fast_skip_intro(mock, generation="gold")
    assert len(mock.actions) == 0


def test_fast_skip_intro_no_memory_early_return():
    """Verify fast_skip_intro returns immediately if emulator does not have a memory attribute."""
    class MockNoMemoryEmulator:
        def __init__(self):
            self.actions: List[str] = []

        def tick(self) -> None:
            self.actions.append("tick")

        def set_emulation_speed(self, speed: int) -> None:
            self.actions.append(f"speed_{speed}")

        def button_press(self, btn: str) -> None:
            self.actions.append(f"press_{btn}")

        def button_release(self, btn: str) -> None:
            self.actions.append(f"release_{btn}")

    mock = MockNoMemoryEmulator()
    assert not hasattr(mock, "memory")
    fast_skip_intro(mock, generation="gen1")
    assert len(mock.actions) == 0


def test_fast_skip_intro_mock_emulator_success():
    """Verify fast_skip_intro completes turbo sequence, releases buttons, and handles fade-in."""
    class MockSuccessEmulator:
        def __init__(self, break_frame: int = 650):
            self.memory: Dict[int, int] = {
                0xCC28: 0,
                0xCC26: 0,
                0xC100: 0,
                0xD35E: 0,
            }
            self.break_frame = break_frame
            self.speeds: List[int] = []
            self.pressed_buttons: List[str] = []
            self.released_buttons: List[str] = []
            self.frame_count: int = 0
            self.fade_ticks: int = 0
            self.loop_terminated: bool = False

        def set_emulation_speed(self, speed: int) -> None:
            self.speeds.append(speed)

        def button_press(self, btn: str) -> None:
            self.pressed_buttons.append(btn)

        def button_release(self, btn: str) -> None:
            self.released_buttons.append(btn)

        def tick(self) -> None:
            if not self.loop_terminated:
                self.frame_count += 1
                # When name menu appears after frame 600, set max_item=3 and cur_item=0
                if self.frame_count == 610:
                    self.memory[0xCC28] = 3
                    self.memory[0xCC26] = 0
                elif self.frame_count > 610:
                    self.memory[0xCC26] = 1

                # Condition match: player enters Pallet Town bedroom / overworld
                if self.frame_count >= self.break_frame:
                    self.memory[0xC100] = 1
                    self.memory[0xD35E] = 38
                    self.loop_terminated = True
            else:
                self.fade_ticks += 1

    mock = MockSuccessEmulator(break_frame=650)
    fast_skip_intro(mock, generation="gen1")

    # 1. Verify emulation speed was set to 0 (turbo speed)
    assert 0 in mock.speeds

    # 2. Verify button presses and releases for start, a, b, and down
    for btn in ("start", "a", "b", "down"):
        assert btn in mock.pressed_buttons, f"Button '{btn}' was not pressed"
        assert btn in mock.released_buttons, f"Button '{btn}' was not released"

    # 3. Verify loop terminated on condition match before reaching 5000 frames
    assert mock.frame_count < 5000
    assert mock.memory[0xC100] == 1
    assert mock.memory[0xD35E] == 38

    # 4. Verify all buttons were released after loop termination
    expected_all_buttons = {"a", "b", "start", "select", "up", "down", "left", "right"}
    final_cleanup_releases = set(mock.released_buttons[-8:])
    assert final_cleanup_releases == expected_all_buttons

    # 5. Verify 180 fade-in ticks were completed
    assert mock.fade_ticks == 180


def test_fast_skip_intro_max_frames_bound():
    """Verify fast_skip_intro safely terminates after 5000 frames if condition is never met."""
    class MockNeverMatchEmulator:
        def __init__(self):
            self.memory: Dict[int, int] = {
                0xCC28: 0,
                0xCC26: 0,
                0xC100: 0,
                0xD35E: 0,
            }
            self.speeds: List[int] = []
            self.pressed_buttons: List[str] = []
            self.released_buttons: List[str] = []
            self.ticks: int = 0

        def set_emulation_speed(self, speed: int) -> None:
            self.speeds.append(speed)

        def button_press(self, btn: str) -> None:
            self.pressed_buttons.append(btn)

        def button_release(self, btn: str) -> None:
            self.released_buttons.append(btn)

        def tick(self) -> None:
            self.ticks += 1

    mock = MockNeverMatchEmulator()
    fast_skip_intro(mock, generation="gen1")

    # Loop ran full 5000 frames plus 180 fade-in ticks = 5180 ticks
    assert mock.ticks == 5180
    assert 0 in mock.speeds
    expected_all_buttons = {"a", "b", "start", "select", "up", "down", "left", "right"}
    assert set(mock.released_buttons[-8:]) == expected_all_buttons


def test_fast_skip_intro_sparse_dict_key_error():
    """Verify fast_skip_intro handles sparse dictionary memory raising KeyError."""
    class MockSparseDictEmulator:
        def __init__(self):
            # Sparse dictionary: missing 0xCC28, 0xCC26, 0xC100, 0xD35E
            self.memory: Dict[int, int] = {0x1234: 0}
            self.speeds: List[int] = []
            self.pressed_buttons: List[str] = []
            self.released_buttons: List[str] = []
            self.ticks: int = 0

        def set_emulation_speed(self, speed: int) -> None:
            self.speeds.append(speed)

        def button_press(self, btn: str) -> None:
            self.pressed_buttons.append(btn)

        def button_release(self, btn: str) -> None:
            self.released_buttons.append(btn)

        def tick(self) -> None:
            self.ticks += 1

    mock = MockSparseDictEmulator()
    fast_skip_intro(mock, generation="gen1")

    # Terminates within bounds (5000 frames + 180 fade ticks = 5180 ticks)
    assert mock.ticks == 5180
    assert 0 in mock.speeds
    expected_all_buttons = {"a", "b", "start", "select", "up", "down", "left", "right"}
    assert set(mock.released_buttons[-8:]) == expected_all_buttons


def test_fast_skip_intro_undersized_buffer_index_error():
    """Verify fast_skip_intro handles undersized buffer/list memory raising IndexError."""
    class MockUndersizedBufferEmulator:
        def __init__(self):
            # Short list: len=100, addresses like 0xC100 (49408) will raise IndexError
            self.memory: List[int] = [0] * 100
            self.speeds: List[int] = []
            self.pressed_buttons: List[str] = []
            self.released_buttons: List[str] = []
            self.ticks: int = 0

        def set_emulation_speed(self, speed: int) -> None:
            self.speeds.append(speed)

        def button_press(self, btn: str) -> None:
            self.pressed_buttons.append(btn)

        def button_release(self, btn: str) -> None:
            self.released_buttons.append(btn)

        def tick(self) -> None:
            self.ticks += 1

    mock = MockUndersizedBufferEmulator()
    fast_skip_intro(mock, generation="gen1")

    # Terminates within bounds (5000 frames + 180 fade ticks = 5180 ticks)
    assert mock.ticks == 5180
    assert 0 in mock.speeds
    expected_all_buttons = {"a", "b", "start", "select", "up", "down", "left", "right"}
    assert set(mock.released_buttons[-8:]) == expected_all_buttons


def test_fast_skip_intro_corrupted_memory_values():
    """Verify fast_skip_intro handles corrupted memory values (strings, None, object())."""
    class MockCorruptedValuesEmulator:
        def __init__(self):
            self.memory: Dict[int, Any] = {
                0xCC28: object(),
                0xCC26: "xyz",
                0xC100: "corrupted",
                0xD35E: None,
            }
            self.speeds: List[int] = []
            self.pressed_buttons: List[str] = []
            self.released_buttons: List[str] = []
            self.ticks: int = 0

        def set_emulation_speed(self, speed: int) -> None:
            self.speeds.append(speed)

        def button_press(self, btn: str) -> None:
            self.pressed_buttons.append(btn)

        def button_release(self, btn: str) -> None:
            self.released_buttons.append(btn)

        def tick(self) -> None:
            self.ticks += 1

    mock = MockCorruptedValuesEmulator()
    fast_skip_intro(mock, generation="gen1")

    # Terminates within bounds (5000 frames + 180 fade ticks = 5180 ticks)
    assert mock.ticks == 5180
    assert 0 in mock.speeds
    expected_all_buttons = {"a", "b", "start", "select", "up", "down", "left", "right"}
    assert set(mock.released_buttons[-8:]) == expected_all_buttons


def test_fast_skip_intro_sparse_to_valid_transition():
    """Verify fast_skip_intro handles sparse/corrupted memory transitioning to valid state."""
    class MockTransitionEmulator:
        def __init__(self, break_frame: int = 650):
            # Initially empty / sparse dictionary
            self.memory: Dict[int, Any] = {0xCC28: "invalid"}
            self.break_frame = break_frame
            self.speeds: List[int] = []
            self.pressed_buttons: List[str] = []
            self.released_buttons: List[str] = []
            self.frame_count: int = 0
            self.fade_ticks: int = 0
            self.loop_terminated: bool = False

        def set_emulation_speed(self, speed: int) -> None:
            self.speeds.append(speed)

        def button_press(self, btn: str) -> None:
            self.pressed_buttons.append(btn)

        def button_release(self, btn: str) -> None:
            self.released_buttons.append(btn)

        def tick(self) -> None:
            if not self.loop_terminated:
                self.frame_count += 1
                if self.frame_count >= self.break_frame:
                    self.memory[0xC100] = 1
                    self.memory[0xD35E] = 38
                    self.loop_terminated = True
            else:
                self.fade_ticks += 1

    mock = MockTransitionEmulator(break_frame=650)
    fast_skip_intro(mock, generation="gen1")

    # Terminates early when condition is met despite initial sparse/invalid memory
    assert mock.frame_count < 5000
    assert mock.fade_ticks == 180
    expected_all_buttons = {"a", "b", "start", "select", "up", "down", "left", "right"}
    assert set(mock.released_buttons[-8:]) == expected_all_buttons


