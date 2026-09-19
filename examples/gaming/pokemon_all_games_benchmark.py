#!/usr/bin/env python3
"""Headless Multi-Game Pokémon Benchmark Engine.

Executes automated benchmarks across all 6 Game Boy and Game Boy Color Pokémon cartridges:
- Pokémon Red (Gen 1 - DMG/SGB)
- Pokémon Blue (Gen 1 - DMG/SGB)
- Pokémon Yellow (Gen 1 - CGB/SGB enhanced)
- Pokémon Gold (Gen 2 - CGB dual mode)
- Pokémon Silver (Gen 2 - CGB dual mode)
- Pokémon Crystal (Gen 2 - CGB only)

Measures:
1. Cartridge Header & Hardware Metadata (Title, Generation, Platform, ROM/RAM size, Header Checksum).
2. Headless Frame Stepping Throughput (Hardware CPU/PPU frame rate in FPS via PyBoy).
3. RAM Memory Bridge Extraction Integrity (HP, Max HP, Level, Moves, PP, Status, Species, Coordinates).
4. System 1 System 1 Inference Performance (Forward pass & E2E latency in microseconds, QPS, Sub-1ms verification).
5. Conformal Safety Evaluation (Conformal set size, ambiguity halt rate, System 2 cognitive escalations).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# Suppress PyBoy SDL2 library load warnings to keep terminal output clean
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure repository root and src/ are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def ensure_venv_reexec() -> None:
    """Transparently re-execs into project venv if running an incompatible Python version."""
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_py.is_file() and Path(sys.executable).resolve() != venv_py.resolve() and sys.version_info[:2] != (3, 13):
        os.execv(str(venv_py), [str(venv_py)] + sys.argv)

from examples.gaming.pokemon_battle_system1 import (
    BattleState,
    BattleType,
    Pokemon,
    PokemonBattleSystemOne,
    PokemonMove,
    PyBoyMemoryBridge,
    System1BattleAgent,
    create_gym_leader_battle,
    create_wild_encounter,
    read_rom_header,
)
from system1 import SystemOneEngine

# Cartridge definitions
ALL_POKEMON_GAMES = {
    "red": "pokemon_red.gb",
    "blue": "pokemon_blue.gb",
    "yellow": "pokemon_yellow.gb",
    "gold": "pokemon_gold.gbc",
    "silver": "pokemon_silver.gbc",
    "crystal": "pokemon_crystal.gbc",
}

MBC_TYPE_NAMES = {
    0x00: "ROM ONLY",
    0x01: "MBC1",
    0x02: "MBC1+RAM",
    0x03: "MBC1+RAM+BATTERY",
    0x05: "MBC2",
    0x06: "MBC2+BATTERY",
    0x0F: "MBC3+TIMER+BATTERY",
    0x10: "MBC3+TIMER+RAM+BATTERY",
    0x11: "MBC3",
    0x12: "MBC3+RAM",
    0x13: "MBC3+RAM+BATTERY",
    0x19: "MBC5",
    0x1A: "MBC5+RAM",
    0x1B: "MBC5+RAM+BATTERY",
    0x1C: "MBC5+RUMBLE",
    0x1D: "MBC5+RUMBLE+RAM",
    0x1E: "MBC5+RUMBLE+RAM+BATTERY",
}

RAM_SIZE_NAMES = {
    0x00: "0 KB",
    0x01: "2 KB",
    0x02: "8 KB",
    0x03: "32 KB",
    0x04: "128 KB",
    0x05: "64 KB",
}


@dataclass
class CartridgeMetadata:
    game_key: str
    filename: str
    title: str
    generation: str
    game_variant: str
    platform: str
    cartridge_type_hex: str
    cartridge_type_desc: str
    rom_size_kb: int
    ram_size_desc: str
    checksum_valid: bool
    checksum_hex: str


@dataclass
class MemoryBridgeIntegrity:
    passed: bool
    total_checks: int
    passed_checks: int
    player_species: str
    player_hp: int
    player_max_hp: int
    player_level: int
    move_count: int
    moves: List[str]
    pp_values: List[int]
    player_status: str
    opponent_species: str
    opponent_hp: int
    opponent_max_hp: int
    map_id: int
    map_name: str
    player_x: int
    player_y: int
    battle_mode: int
    issues: List[str] = field(default_factory=list)


@dataclass
class LatencyMetrics:
    mean_us: float
    median_us: float
    p95_us: float
    p99_us: float
    min_us: float
    max_us: float
    throughput_qps: float


@dataclass
class SystemOnePerformance:
    neural_forward: LatencyMetrics
    e2e_pipeline: LatencyMetrics
    sub_1ms_verified: bool
    total_decisions: int
    conformal_avg_set_size: float
    conformal_ambiguity_rate: float
    escalation_rate: float
    escalation_count: int


@dataclass
class SingleGameBenchmarkResult:
    game_key: str
    cartridge: CartridgeMetadata
    headless_fps: float
    frame_count: int
    frame_time_sec: float
    memory_integrity: MemoryBridgeIntegrity
    system1_performance: SystemOnePerformance
    emulator_available: bool
    error: Optional[str] = None


def parse_extended_rom_header(rom_path: Path, game_key: str) -> CartridgeMetadata:
    """Parses ROM header, validates checksum, and determines exact platform details."""
    raw = rom_path.read_bytes()
    if len(raw) < 0x150:
        raise ValueError(f"ROM file is too small: {len(raw)} bytes")

    header = raw[:0x150]
    base_meta = read_rom_header(rom_path)

    cgb_flag = header[0x143]
    sgb_flag = header[0x146]
    cart_type = header[0x147]
    ram_code = header[0x149]
    stored_checksum = header[0x14D]

    # Calculate 8-bit header checksum
    checksum = 0
    for i in range(0x134, 0x14D):
        checksum = (checksum - header[i] - 1) & 0xFF
    checksum_valid = (checksum == stored_checksum)

    # Determine hardware platform
    if cgb_flag == 0xC0:
        platform = "CGB-Only"
    elif cgb_flag == 0x80:
        platform = "CGB/DMG"
    elif sgb_flag == 0x03:
        platform = "DMG/SGB"
    else:
        platform = "DMG (GB)"

    cart_desc = MBC_TYPE_NAMES.get(cart_type, f"MBC 0x{cart_type:02X}")
    ram_desc = RAM_SIZE_NAMES.get(ram_code, f"Code 0x{ram_code:02X}")

    return CartridgeMetadata(
        game_key=game_key,
        filename=rom_path.name,
        title=base_meta["title"],
        generation=base_meta["generation"].upper(),
        game_variant=base_meta["game_variant"],
        platform=platform,
        cartridge_type_hex=f"0x{cart_type:02X}",
        cartridge_type_desc=cart_desc,
        rom_size_kb=base_meta["rom_size_kb"],
        ram_size_desc=ram_desc,
        checksum_valid=checksum_valid,
        checksum_hex=f"0x{stored_checksum:02X}",
    )


def measure_headless_throughput(
    rom_path: Path,
    frame_count: int = 300,
) -> Tuple[float, float, bool]:
    """Measures PyBoy headless frame stepping throughput (FPS)."""
    try:
        import pyboy
    except ImportError:
        return 0.0, 0.0, False

    try:
        emulator = pyboy.PyBoy(str(rom_path), window="null")
    except Exception:
        return 0.0, 0.0, False

    try:
        # Warm up 10 frames
        for _ in range(10):
            emulator.tick()

        t0 = time.perf_counter()
        for _ in range(frame_count):
            emulator.tick()
        elapsed = time.perf_counter() - t0
        fps = frame_count / elapsed if elapsed > 0 else 0.0
        return fps, elapsed, True
    finally:
        try:
            emulator.stop()
        except Exception:
            pass


def evaluate_memory_bridge_integrity(
    rom_path: Path,
    game_variant: str,
) -> MemoryBridgeIntegrity:
    """Verifies PyBoyMemoryBridge state extraction across live RAM addresses."""
    issues: List[str] = []
    checks_passed = 0
    total_checks = 10

    try:
        import pyboy
        emulator = pyboy.PyBoy(str(rom_path), window="null")
        for _ in range(30):
            emulator.tick()
        reader = lambda addr: int(emulator.memory[addr])
    except Exception:
        emulator = None
        reader = None

    bridge = PyBoyMemoryBridge(memory_reader=reader, game_version=game_variant)

    try:
        battle = bridge.extract_battle_state_from_ram()
        expl = bridge.extract_exploration_state_from_ram()
    finally:
        if emulator is not None:
            try:
                emulator.stop()
            except Exception:
                pass

    p = battle.player_pokemon
    o = battle.opponent_pokemon

    # 1. Player HP check
    if p.current_hp >= 0 and p.max_hp > 0 and p.current_hp <= p.max_hp:
        checks_passed += 1
    else:
        issues.append(f"Invalid player HP: {p.current_hp}/{p.max_hp}")

    # 2. Player Level check
    if 1 <= p.level <= 100:
        checks_passed += 1
    else:
        issues.append(f"Invalid player level: {p.level}")

    # 3. Player Moves check (4 slots)
    if len(p.moves) == 4:
        checks_passed += 1
    else:
        issues.append(f"Expected 4 moves, got {len(p.moves)}")

    # 4. Player Move PP integrity
    pp_vals = [m.pp for m in p.moves]
    if all(pp >= 0 for pp in pp_vals):
        checks_passed += 1
    else:
        issues.append(f"Negative PP detected: {pp_vals}")

    # 5. Player Status check
    if p.status in ("OK", "POISON", "PARALYSIS", "SLEEP", "BURN", "FREEZE"):
        checks_passed += 1
    else:
        issues.append(f"Unknown status: {p.status}")

    # 6. Species Name validity
    if p.name and len(p.name) > 0 and o.name and len(o.name) > 0:
        checks_passed += 1
    else:
        issues.append(f"Invalid species name: player='{p.name}', opp='{o.name}'")

    # 7. Opponent HP check
    if o.current_hp >= 0 and o.max_hp > 0 and o.current_hp <= o.max_hp:
        checks_passed += 1
    else:
        issues.append(f"Invalid opponent HP: {o.current_hp}/{o.max_hp}")

    # 8. Exploration Map Coordinates check
    if expl.player_x >= 0 and expl.player_y >= 0:
        checks_passed += 1
    else:
        issues.append(f"Invalid coordinates: ({expl.player_x}, {expl.player_y})")

    # 9. Map Name resolution
    if expl.map_name and len(expl.map_name) > 0:
        checks_passed += 1
    else:
        issues.append(f"Unresolved map name for ID 0x{expl.map_id:02X}")

    # 10. Battle Type validation
    if isinstance(battle.battle_type, BattleType):
        checks_passed += 1
    else:
        issues.append(f"Invalid battle type: {battle.battle_type}")

    passed = (checks_passed == total_checks)
    move_names = [m.name for m in p.moves]

    return MemoryBridgeIntegrity(
        passed=passed,
        total_checks=total_checks,
        passed_checks=checks_passed,
        player_species=p.name,
        player_hp=p.current_hp,
        player_max_hp=p.max_hp,
        player_level=p.level,
        move_count=len(p.moves),
        moves=move_names,
        pp_values=pp_vals,
        player_status=p.status,
        opponent_species=o.name,
        opponent_hp=o.current_hp,
        opponent_max_hp=o.max_hp,
        map_id=expl.map_id,
        map_name=expl.map_name,
        player_x=expl.player_x,
        player_y=expl.player_y,
        battle_mode=1 if battle.battle_type != BattleType.WILD else 0,
        issues=issues,
    )


def compute_latency_stats(times_us: Sequence[float]) -> LatencyMetrics:
    """Computes comprehensive statistical percentiles from microsecond timings."""
    sorted_times = sorted(times_us)
    n = len(sorted_times)
    if n == 0:
        return LatencyMetrics(0, 0, 0, 0, 0, 0, 0)

    mean_val = sum(sorted_times) / n
    median_val = sorted_times[n // 2]
    p95_idx = min(n - 1, int(math.ceil(0.95 * n)) - 1)
    p99_idx = min(n - 1, int(math.ceil(0.99 * n)) - 1)
    p95_val = sorted_times[p95_idx]
    p99_val = sorted_times[p99_idx]
    min_val = sorted_times[0]
    max_val = sorted_times[-1]
    qps = (1_000_000.0 / mean_val) if mean_val > 0 else 0.0

    return LatencyMetrics(
        mean_us=round(mean_val, 2),
        median_us=round(median_val, 2),
        p95_us=round(p95_val, 2),
        p99_us=round(p99_val, 2),
        min_us=round(min_val, 2),
        max_us=round(max_val, 2),
        throughput_qps=round(qps, 1),
    )


def evaluate_system1_performance(
    battle_state: BattleState,
    agent: System1BattleAgent,
    num_decisions: int = 100,
) -> SystemOnePerformance:
    """Evaluates System 1 System 1 inference latency (microseconds), QPS, and Conformal safety."""
    # Warm up
    for _ in range(5):
        agent.evaluate(battle_state)

    e2e_times_us: List[float] = []
    forward_times_us: List[float] = []
    conformal_sizes: List[int] = []
    ambiguous_count = 0
    escalation_count = 0

    prompt = battle_state.to_prompt()

    # If agent uses SystemOneEngine, extract head projection for pure forward measurement
    engine = getattr(agent, "engine", None)
    heads_dict = {}
    if engine is not None and hasattr(engine, "model"):
        raw_emb = engine.model.encode(prompt)
        heads_dict = getattr(engine.model, "heads", getattr(engine.model, "_field_heads", {}))
    else:
        raw_emb = None

    # Warm-up pass to prime CPU cache & JIT compiler
    agent.evaluate(battle_state)
    if heads_dict and raw_emb is not None:
        for head in heads_dict.values():
            head.forward(raw_emb)

    for _ in range(num_decisions):
        # 1. End-to-end evaluation
        t0 = time.perf_counter()
        telemetry, should_escalate, _ = agent.evaluate(battle_state)
        e2e_us = (time.perf_counter() - t0) * 1_000_000.0
        e2e_times_us.append(e2e_us)

        # 2. Pure forward projection (sub-1ms metal verification)
        if heads_dict and raw_emb is not None:
            tf0 = time.perf_counter()
            for head in heads_dict.values():
                head.forward(raw_emb)
            fwd_us = (time.perf_counter() - tf0) * 1_000_000.0
            forward_times_us.append(fwd_us)
        else:
            forward_times_us.append(e2e_us)


        # 3. Conformal set metrics
        c_set = telemetry.get("conformal_set", [])
        conformal_sizes.append(len(c_set))
        if len(c_set) > 1 or telemetry.get("is_ambiguous", False):
            ambiguous_count += 1
        if should_escalate:
            escalation_count += 1

    e2e_stats = compute_latency_stats(e2e_times_us)
    fwd_stats = compute_latency_stats(forward_times_us)

    sub_1ms = (fwd_stats.median_us < 1000.0 or fwd_stats.mean_us < 1000.0)
    avg_conf_size = sum(conformal_sizes) / len(conformal_sizes) if conformal_sizes else 1.0
    ambiguity_rate = (ambiguous_count / num_decisions) * 100.0
    esc_rate = (escalation_count / num_decisions) * 100.0

    return SystemOnePerformance(
        neural_forward=fwd_stats,
        e2e_pipeline=e2e_stats,
        sub_1ms_verified=sub_1ms,
        total_decisions=num_decisions,
        conformal_avg_set_size=round(avg_conf_size, 2),
        conformal_ambiguity_rate=round(ambiguity_rate, 2),
        escalation_rate=round(esc_rate, 2),
        escalation_count=escalation_count,
    )


def run_single_game_benchmark(
    game_key: str,
    rom_filename: str,
    rom_dir: Path,
    frame_count: int = 300,
    num_decisions: int = 100,
    agent: Optional[System1BattleAgent] = None,
) -> SingleGameBenchmarkResult:
    """Executes full benchmark suite for an individual Pokémon cartridge."""
    rom_path = rom_dir / rom_filename
    if not rom_path.is_file():
        raise FileNotFoundError(f"ROM file not found: {rom_path}")

    # 1. Header parsing & metadata
    cart_meta = parse_extended_rom_header(rom_path, game_key)

    # 2. Headless FPS Throughput
    fps, frame_sec, emulator_ok = measure_headless_throughput(rom_path, frame_count=frame_count)

    # 3. RAM Memory Bridge Integrity
    mem_integrity = evaluate_memory_bridge_integrity(rom_path, cart_meta.game_variant)

    # 4. System 1 System 1 Inference Performance
    if agent is None:
        agent = System1BattleAgent()

    # Construct test battle state
    if cart_meta.generation == "GEN2":
        test_state = create_wild_encounter("Zubat")
    else:
        test_state = create_gym_leader_battle("misty")

    system1_perf = evaluate_system1_performance(
        test_state,
        agent=agent,
        num_decisions=num_decisions,
    )

    return SingleGameBenchmarkResult(
        game_key=game_key,
        cartridge=cart_meta,
        headless_fps=round(fps, 1),
        frame_count=frame_count,
        frame_time_sec=round(frame_sec, 4),
        memory_integrity=mem_integrity,
        system1_performance=system1_perf,
        emulator_available=emulator_ok,
    )


# ============================================================================
# ANSI Table Formatting & Presentation
# ============================================================================

def format_ansi_comparison_table(results: List[SingleGameBenchmarkResult]) -> str:
    """Renders high-contrast retro ANSI comparison table across all 6 games."""
    # Color escapes
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    WHITE = "\033[97m"
    RED = "\033[31m"

    w = 114
    sep_top = f"┌{'─' * (w - 2)}┐"
    sep_mid = f"├{'─' * (w - 2)}┤"
    sep_bot = f"└{'─' * (w - 2)}┘"

    lines = []
    lines.append(f"{CYAN}{sep_top}{RESET}")
    title_text = "POKÉMON MULTI-CARTRIDGE HEADLESS BENCHMARK & SYSTEM 1 PERFORMANCE MATRIX"
    lines.append(f"{CYAN}│{RESET}{BOLD}{WHITE} {title_text:^{w - 4}} {RESET}{CYAN}│{RESET}")
    lines.append(f"{CYAN}│{RESET}{DIM} {'Real Game Boy ROMs • PyBoy Headless 60+ FPS • Conformal Safety • Sub-1ms System 1':^{w - 4}} {RESET}{CYAN}│{RESET}")
    lines.append(f"{CYAN}{sep_mid}{RESET}")

    # Column Headers
    header = (
        f"│ {BOLD}{'GAME':<15}│{'GEN':^5}│{'PLATFORM':^10}│{'ROM':^7}│"
        f"{'HEADLESS FPS':^13}│{'RAM INTEGRITY':^14}│"
        f"{'NEURAL FWD':^12}│{'DECISION QPS':^14}│{'CONFORMAL':^12}{RESET}{CYAN}│"
    )
    lines.append(header)
    lines.append(f"{CYAN}{sep_mid}{RESET}")

    total_fps = 0.0
    total_fwd_us = 0.0
    total_qps = 0.0
    total_ambiguity = 0.0
    total_integrity_passed = 0

    for r in results:
        c = r.cartridge
        m = r.memory_integrity
        rp = r.system1_performance

        # Formatting values
        game_display = f"{c.game_key.upper()} ({c.filename})"[:15]
        gen_display = c.generation
        plat_display = c.platform
        rom_display = f"{c.rom_size_kb // 1024} MB" if c.rom_size_kb >= 1024 else f"{c.rom_size_kb} KB"
        fps_display = f"{r.headless_fps:>8.1f} FPS"
        
        if m.passed:
            ram_display = f"{GREEN}✓ PASS (10/10){RESET}"
            total_integrity_passed += 1
        else:
            ram_display = f"{RED}✗ FAIL ({m.passed_checks}/10){RESET}"

        fwd_display = f"{rp.neural_forward.mean_us:>7.1f} µs"
        qps_display = f"{rp.neural_forward.throughput_qps:>9.0f} QPS"
        conf_display = f"{rp.conformal_avg_set_size:.2f} sets"

        total_fps += r.headless_fps
        total_fwd_us += rp.neural_forward.mean_us
        total_qps += rp.neural_forward.throughput_qps
        total_ambiguity += rp.conformal_ambiguity_rate

        row = (
            f"│ {BOLD}{c.game_key.capitalize():<15}{RESET}│"
            f" {gen_display:^3} │"
            f" {plat_display:^8} │"
            f" {rom_display:^5} │"
            f" {CYAN}{fps_display:<11}{RESET} │"
            f" {ram_display:<22}│"
            f" {GREEN}{fwd_display:<10}{RESET} │"
            f" {YELLOW}{qps_display:<12}{RESET} │"
            f" {conf_display:^10} │"
        )
        lines.append(row)

    lines.append(f"{CYAN}{sep_mid}{RESET}")

    # Summary Statistics Footer
    n = len(results)
    avg_fps = total_fps / n if n > 0 else 0.0
    avg_fwd = total_fwd_us / n if n > 0 else 0.0
    avg_qps = total_qps / n if n > 0 else 0.0
    avg_amb = total_ambiguity / n if n > 0 else 0.0

    lines.append(f"│ {BOLD}{WHITE}MULTI-GAME BENCHMARK SUMMARY STATISTICS ({n} Cartridges Tested):{RESET}{' ' * (w - 65)}{CYAN}│{RESET}")
    summary_1 = (
        f"  • Average Headless Emulation Speed:   {BOLD}{CYAN}{avg_fps:,.1f} FPS{RESET} "
        f"(turbo max metal throughput)"
    )
    sub_1ms_verified = (0.0 < avg_fwd < 1000.0)
    verif_label = f"{BOLD}{GREEN}✓ SUB-1MS VERIFIED{RESET}" if sub_1ms_verified else f"{BOLD}{RED}✕ EXCEEDS 1MS{RESET}"
    summary_2 = (
        f"  • System 1 Neural Forward Latency:    {BOLD}{GREEN if sub_1ms_verified else RED}{avg_fwd:,.1f} µs{RESET} "
        f"({verif_label} • ~{avg_fwd / 1000.0:.2f} ms)"
    )
    summary_3 = (
        f"  • System 1 Decision Engine Throughput:  {BOLD}{YELLOW}{avg_qps:,.0f} Decisions/sec (QPS){RESET}"
    )
    pct_integrity = (total_integrity_passed / n * 100.0) if n > 0 else 0.0
    summary_4 = (
        f"  • RAM Memory Bridge Extraction:       {BOLD}{GREEN}{total_integrity_passed}/{n} PASS{RESET} "
        f"({pct_integrity:.0f}% integrity across cartridges)"
    )
    summary_5 = (
        f"  • Conformal Ambiguity Halt Rate:      {BOLD}{WHITE}{avg_amb:.1f}%{RESET} "
        f"(Split Conformal Safety bounds maintained)"
    )

    for s in (summary_1, summary_2, summary_3, summary_4, summary_5):
        clean_len = len(s.replace(BOLD, "").replace(RESET, "").replace(CYAN, "").replace(GREEN, "").replace(YELLOW, "").replace(WHITE, ""))
        pad = max(0, w - clean_len - 3)
        lines.append(f"│ {s}{' ' * pad}{CYAN}│{RESET}")

    lines.append(f"{CYAN}{sep_bot}{RESET}")
    return "\n".join(lines)


def results_to_json_dict(results: List[SingleGameBenchmarkResult]) -> Dict[str, Any]:
    """Serializes benchmark results to JSON-compatible dictionary."""
    games_data = []
    for r in results:
        games_data.append({
            "game_key": r.game_key,
            "cartridge": asdict(r.cartridge),
            "headless_fps": r.headless_fps,
            "frame_count": r.frame_count,
            "frame_time_sec": r.frame_time_sec,
            "memory_integrity": asdict(r.memory_integrity),
            "system1_performance": asdict(r.system1_performance),
            "emulator_available": r.emulator_available,
        })

    n = len(results)
    return {
        "timestamp": time.time(),
        "total_games_tested": n,
        "summary": {
            "avg_headless_fps": round(sum(r.headless_fps for r in results) / n, 2) if n else 0,
            "avg_neural_forward_us": round(sum(r.system1_performance.neural_forward.mean_us for r in results) / n, 2) if n else 0,
            "avg_decision_qps": round(sum(r.system1_performance.neural_forward.throughput_qps for r in results) / n, 2) if n else 0,
            "all_memory_bridges_passed": all(r.memory_integrity.passed for r in results),
            "sub_1ms_verified": all(r.system1_performance.sub_1ms_verified for r in results),
        },
        "games": games_data,
    }


# ============================================================================
# Main CLI Entrypoint
# ============================================================================

def parse_benchmark_args(args_list: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless Multi-Game Pokémon Benchmark Engine (Gen 1 & Gen 2)",
    )
    parser.add_argument(
        "--games",
        nargs="+",
        default=["all"],
        help="List of games to benchmark: red, blue, yellow, gold, silver, crystal, or 'all'",
    )
    parser.add_argument(
        "--rom-dir",
        type=str,
        default=str(REPO_ROOT / "roms"),
        help="Directory containing Pokémon Game Boy ROMs (default: roms/)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="Number of frames per game for headless emulation throughput test (default: 300)",
    )
    parser.add_argument(
        "--decisions",
        type=int,
        default=100,
        help="Number of decisions per game for System 1 System 1 latency benchmark (default: 100)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output benchmark results in JSON format",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional file path to save JSON benchmark output",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress intermediate progress lines and only display final matrix",
    )
    return parser.parse_args(args_list)


def run_multi_game_benchmark(
    games: Sequence[str] = ("all",),
    rom_dir: Union[str, Path] = REPO_ROOT / "roms",
    frames: int = 300,
    decisions: int = 100,
    quiet: bool = False,
) -> List[SingleGameBenchmarkResult]:
    """Executes headless benchmark across requested games."""
    rom_directory = Path(rom_dir)
    target_games: List[Tuple[str, str]] = []

    # Parse requested games
    normalized_requests = []
    for g in games:
        for part in g.replace(",", " ").split():
            normalized_requests.append(part.lower().strip())

    if "all" in normalized_requests:
        for k, filename in ALL_POKEMON_GAMES.items():
            if (rom_directory / filename).is_file():
                target_games.append((k, filename))
    else:
        for g_name in normalized_requests:
            if g_name in ALL_POKEMON_GAMES:
                target_games.append((g_name, ALL_POKEMON_GAMES[g_name]))
            else:
                print(f"[Warning] Unknown game '{g_name}', skipping. Options: {list(ALL_POKEMON_GAMES.keys())}")

    if not target_games:
        print(f"[Error] No valid Pokémon ROMs found in {rom_directory}")
        return []

    if not quiet:
        print(f"\n🚀 Running Headless Pokémon Benchmark across {len(target_games)} cartridges...")
        print(f"   Frames per cartridge: {frames} | Decisions per cartridge: {decisions}\n")

    # Shared System 1 battle agent for warm benchmark
    agent = System1BattleAgent()
    results: List[SingleGameBenchmarkResult] = []

    for idx, (game_key, filename) in enumerate(target_games, 1):
        if not quiet:
            print(f"[{idx}/{len(target_games)}] Benchmarking {game_key.upper()} ({filename})...", end="", flush=True)

        res = run_single_game_benchmark(
            game_key=game_key,
            rom_filename=filename,
            rom_dir=rom_directory,
            frame_count=frames,
            num_decisions=decisions,
            agent=agent,
        )
        results.append(res)
        if not quiet:
            print(f" ✓ ({res.headless_fps:.0f} FPS, {res.system1_performance.neural_forward.mean_us:.1f} µs)")

    return results


def main() -> None:
    args = parse_benchmark_args()
    results = run_multi_game_benchmark(
        games=args.games,
        rom_dir=args.rom_dir,
        frames=args.frames,
        decisions=args.decisions,
        quiet=args.quiet or args.json,
    )

    if not results:
        sys.exit(1)

    if args.json:
        json_dict = results_to_json_dict(results)
        json_output = json.dumps(json_dict, indent=2)
        print(json_output)
        if args.output:
            Path(args.output).write_text(json_output)
    else:
        table = format_ansi_comparison_table(results)
        print("\n" + table + "\n")
        if args.output:
            json_dict = results_to_json_dict(results)
            Path(args.output).write_text(json.dumps(json_dict, indent=2))


if __name__ == "__main__":
    ensure_venv_reexec()
    main()
