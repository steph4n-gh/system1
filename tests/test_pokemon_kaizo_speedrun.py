"""Unit and integration tests for Game Boy Pokémon Speedrun / Kaizo Zero-Wipe Engine."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
GAMING_DIR = EXAMPLES_DIR / "gaming"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
if str(GAMING_DIR) not in sys.path:
    sys.path.insert(0, str(GAMING_DIR))

from pokemon_kaizo_speedrun import (
    KaizoZeroWipeGate,
    SpeedrunMilestone,
    SpeedrunPartyMember,
    UncappedPyBoyTurboRunner,
    WorstCaseDamageAssessment,
)


# ============================================================================
# 1. Zero-Wipe Conformal Gate Threat Assessment Tests
# ============================================================================

def test_conformal_gate_safe_attack():
    """Verify that when incoming damage is low, the gate recommends ATTACK with low risk."""
    gate = KaizoZeroWipeGate(alpha=0.01)
    threat = gate.assess_threat(
        player_hp=100,
        player_max_hp=100,
        player_def=80,
        player_spc=80,
        player_types=("Water",),
        enemy_atk=30,
        enemy_spc=30,
        enemy_spe=40,
        enemy_move_power=40,
        enemy_move_type="Normal",
        enemy_move_category="physical",
    )
    assert threat.conformal_risk_score < 50.0
    assert threat.recommended_action == "ATTACK"
    assert not threat.is_fatal_regular
    assert not threat.is_fatal_critical


def test_conformal_gate_lethal_regular_hit():
    """Verify that when max damage roll is lethal, the gate trips with risk 1.0 and HEAL."""
    gate = KaizoZeroWipeGate(alpha=0.01)
    threat = gate.assess_threat(
        player_hp=15,  # Very low HP
        player_max_hp=100,
        player_def=50,
        player_spc=50,
        player_types=("Water",),
        enemy_atk=80,
        enemy_spc=80,
        enemy_spe=70,
        enemy_move_power=95,
        enemy_move_type="Electric",  # Super-effective
        enemy_move_category="special",
    )
    assert threat.conformal_risk_score == 1.0
    assert threat.is_fatal_regular is True
    assert threat.recommended_action == "HEAL"


def test_conformal_gate_critical_hit_detection():
    """Verify that when regular damage is survivable but critical hit is lethal, the gate trips."""
    gate = KaizoZeroWipeGate(alpha=0.01)
    # Starmie / Jolteon high speed with Thunderbolt
    threat = gate.assess_threat(
        player_hp=65,  # Survives regular hit (~50 dmg) but dies to crit (~95 dmg)
        player_max_hp=120,
        player_def=75,
        player_spc=75,
        player_types=("Water",),
        enemy_atk=90,
        enemy_spc=110,
        enemy_spe=130,  # 130/512 ~ 25.4% crit chance!
        enemy_move_power=95,
        enemy_move_type="Electric",
        enemy_move_category="special",
    )
    assert threat.crit_probability >= 0.20
    assert threat.is_fatal_critical is True
    assert threat.conformal_risk_score >= 0.80
    assert threat.recommended_action in ("HEAL", "X_ITEM")


def test_conformal_gate_kaizo_enhancement():
    """Verify Kaizo mode doubles effective enemy critical hit probability."""
    gate = KaizoZeroWipeGate(alpha=0.01)
    reg_threat = gate.assess_threat(
        player_hp=80, player_max_hp=100, player_def=70, player_spc=70,
        player_types=("Water",), enemy_atk=60, enemy_spc=60, enemy_spe=60,
        enemy_move_power=80, enemy_move_type="Normal", is_kaizo=False,
    )
    kaizo_threat = gate.assess_threat(
        player_hp=80, player_max_hp=100, player_def=70, player_spc=70,
        player_types=("Water",), enemy_atk=60, enemy_spc=60, enemy_spe=60,
        enemy_move_power=80, enemy_move_type="Normal", is_kaizo=True,
    )
    assert kaizo_threat.crit_probability == reg_threat.crit_probability * 2.0


# ============================================================================
# 2. Speedrun Route & Evolution Progression Tests
# ============================================================================

def test_speedrun_route_milestones_count():
    """Verify all 16 major speedrun milestones exist in route."""
    runner = UncappedPyBoyTurboRunner(use_pyboy=False)
    assert len(runner.milestones) == 16
    assert runner.milestones[0] == SpeedrunMilestone.PALLET_TOWN
    assert runner.milestones[1] == SpeedrunMilestone.BROCK_BOULDER
    assert runner.milestones[3] == SpeedrunMilestone.MISTY_CASCADE
    assert runner.milestones[-1] == SpeedrunMilestone.CHAMPION_BLUE


def test_carry_evolution_progression():
    """Verify Squirtle evolves into Wartortle and Blastoise at correct milestones."""
    runner = UncappedPyBoyTurboRunner(use_pyboy=False)
    assert runner.carry.species == "Squirtle"

    # Step through Brock (Boulder) and Misty (Cascade)
    runner.step_route_milestone()  # Pallet Town
    runner.step_route_milestone()  # Brock
    runner.step_route_milestone()  # Mt. Moon
    runner.step_route_milestone()  # Misty (Wartortle evolution)

    assert runner.carry.species == "Wartortle"
    assert "Cascade" in runner.badges_collected

    # Step through Erika (Rainbow) -> Blastoise evolution
    runner.step_route_milestone()  # S.S. Anne Cut
    runner.step_route_milestone()  # Lt. Surge Thunder
    runner.step_route_milestone()  # Erika Rainbow

    assert runner.carry.species == "Blastoise"
    assert "Rainbow" in runner.badges_collected
    assert "Surf" in runner.carry.moves
    assert "Ice Beam" in runner.carry.moves


# ============================================================================
# 3. Turbo Runner & Zero-Wipe Certification Tests
# ============================================================================

def test_uncapped_turbo_runner_pure_numpy_mode():
    """Verify full campaign runs to completion with 0 wipes in pure NumPy mode."""
    runner = UncappedPyBoyTurboRunner(use_pyboy=False)
    report = runner.run_campaign()

    assert report["milestones_completed"] == 16
    assert report["wipes"] == 0  # 0% wipe rate guaranteed!
    assert len(report["badges"]) == 8  # All 8 gym badges
    assert report["interventions"] > 0  # Conformal gate intervened on high-threat turns
    assert report["carry"]["species"] == "Blastoise"
    assert report["fps"] > 1000.0


def test_uncapped_pyboy_turbo_runner_real_rom():
    """Verify uncapped headless PyBoy runner with real Pokémon Red ROM achieves > 3,000 FPS."""
    rom_path = REPO_ROOT / "roms" / "pokemon_red.gb"
    if not rom_path.is_file():
        pytest.skip("Pokémon Red ROM not found on disk")

    pytest.importorskip("pyboy", reason="PyBoy emulator not installed")
    runner = UncappedPyBoyTurboRunner(rom_path=rom_path, use_pyboy=True)
    assert runner.pyboy is not None

    report = runner.run_campaign(max_milestones=5)
    assert report["milestones_completed"] == 5
    assert report["wipes"] == 0
    # On Apple Silicon M-series, uncapped PyBoy headless reaches 5,000 to 13,000+ FPS!
    assert report["fps"] >= 3000.0


def test_conformal_gate_type_immunity_zero_damage():
    """Verify that type immunity evaluates to strictly 0 damage and recommended ATTACK."""
    gate = KaizoZeroWipeGate(alpha=0.01)
    # Ground type against Electric move
    threat = gate.assess_threat(
        player_hp=1,
        player_max_hp=100,
        player_def=50,
        player_spc=50,
        player_types=("Ground",),
        enemy_atk=100,
        enemy_spc=100,
        enemy_spe=100,
        enemy_move_power=100,
        enemy_move_type="Electric",
    )
    assert threat.min_damage == 0
    assert threat.max_damage_regular == 0
    assert threat.max_damage_critical == 0
    assert threat.is_fatal_regular is False
    assert threat.is_fatal_critical is False
    assert threat.conformal_risk_score == 0.0
    assert threat.recommended_action == "ATTACK"


def test_conformal_gate_enemy_level_scaling():
    """Verify assess_threat correctly factors in enemy level."""
    gate = KaizoZeroWipeGate(alpha=0.01)
    low_lvl = gate.assess_threat(
        player_hp=100, player_max_hp=100, player_def=50, player_spc=50,
        player_types=("Water",), enemy_atk=50, enemy_spc=50, enemy_spe=50,
        enemy_move_power=80, enemy_move_type="Normal", enemy_level=10,
    )
    high_lvl = gate.assess_threat(
        player_hp=100, player_max_hp=100, player_def=50, player_spc=50,
        player_types=("Water",), enemy_atk=50, enemy_spc=50, enemy_spe=50,
        enemy_move_power=80, enemy_move_type="Normal", enemy_level=60,
    )
    assert high_lvl.max_damage_regular > low_lvl.max_damage_regular * 3


def test_milestone_starter_pick_no_spurious_healing():
    """Verify Pallet Town starter selection does not trigger emergency healing."""
    runner = UncappedPyBoyTurboRunner(use_pyboy=False)
    milestone, desc, lat = runner.step_route_milestone()
    assert milestone == SpeedrunMilestone.PALLET_TOWN
    assert "Preemptive Healing" not in desc
    assert runner.conformal_interventions == 0

