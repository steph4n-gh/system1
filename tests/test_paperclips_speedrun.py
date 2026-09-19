"""Unit and integration tests for Universal Paperclips Playwright / System 1 Speedrun Engine."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import pytest

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
GAMING_DIR = EXAMPLES_DIR / "gaming"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
if str(GAMING_DIR) not in sys.path:
    sys.path.insert(0, str(GAMING_DIR))

from paperclips_speedrun import (
    MockPaperclipsBrowserController,
    Paperclips3PhasePolicy,
    PaperclipsObservation,
    PaperclipsSpeedrunSystemOne,
    PaperclipsSpeedrunRunner,
    PaperclipsSplitTimer,
    PlaywrightPaperclipsController,
    SpeedrunSplit,
)


# ============================================================================
# 1. Split Timer & Milestone Tracker Tests
# ============================================================================

def test_split_timer_initialization():
    """Verify timer initializes all 8 WR milestones."""
    timer = PaperclipsSplitTimer()
    assert len(timer.splits) == 8
    assert timer.splits[0].id == "s1_clips_1k"
    assert timer.splits[4].id == "s5_hypnodrones"
    assert timer.splits[7].id == "s8_universe_100"


def test_split_timer_recording_and_gold_split():
    """Verify split completion records elapsed time and calculates delta."""
    timer = PaperclipsSplitTimer()
    # Record first split faster than WR (target 45.0s, actual 40.0s) -> GOLD
    split = timer.check_and_record_split("s1_clips_1k", current_elapsed=40.0)
    assert split is not None
    assert split.completed_time == 40.0
    assert split.delta_seconds == -5.0
    assert split.status == "GOLD"

    # Subsequent record on same split is a no-op
    dup = timer.check_and_record_split("s1_clips_1k", current_elapsed=50.0)
    assert dup is None

    # Record second split slower than WR (target 135.0s, actual 140.0s) -> COMPLETED
    split2 = timer.check_and_record_split("s2_trust_5", current_elapsed=140.0)
    assert split2 is not None
    assert split2.delta_seconds == 5.0
    assert split2.status == "COMPLETED"


def test_split_timer_board_rendering():
    """Verify ASCII split board renders without error."""
    timer = PaperclipsSplitTimer()
    timer.check_and_record_split("s1_clips_1k", current_elapsed=42.5)
    board = timer.render_split_board()
    assert "UNIVERSAL PAPERCLIPS WORLD RECORD SPLIT TIMER" in board
    assert "First 1,000 Clips" in board
    assert "TOTAL ELAPSED" in board


# ============================================================================
# 2. DecisionSchema & Observation Tests
# ============================================================================

def test_speedrun_system1_schema():
    """Verify DecisionSchema field definitions and choices."""
    schema = PaperclipsSpeedrunSystemOne()
    assert "action" in schema._fields
    assert "conformal_risk" in schema._fields
    options = schema._fields["action"].options
    assert "make_paperclip" in options
    assert "buy_wire" in options
    assert "quantum_compute" in options
    assert "phase2_drone" in options
    assert "phase3_probe_rebalance" in options


def test_observation_serialization():
    """Verify observation serialization."""
    obs = PaperclipsObservation(phase=1, clips=5000, funds=120.50, wire=2500)
    summary = obs.to_summary_dict()
    assert summary["phase"] == 1
    assert summary["clips"] == 5000.0
    assert summary["funds"] == 120.50
    assert summary["wire"] == 2500.0


# ============================================================================
# 3. 3-Phase Optimal Control Policy Tests
# ============================================================================

def test_policy_phase1_manual_click():
    """Phase 1: Default action with wire is make_paperclip."""
    policy = Paperclips3PhasePolicy()
    obs = PaperclipsObservation(phase=1, clips=10, wire=5000, funds=1.0)
    action, payload, risk = policy.evaluate(obs)
    assert action == "make_paperclip"
    assert risk < 0.10


def test_policy_phase1_price_elasticity():
    """Phase 1: Adjusts price margin based on demand and unsold clips."""
    policy = Paperclips3PhasePolicy()
    # High unsold inventory -> lower price
    obs_unsold = PaperclipsObservation(phase=1, clips=100, unsold_clips=1000, demand=5.0, margin=0.25)
    action, _, _ = policy.evaluate(obs_unsold)
    assert action == "lower_price"

    # Very low unsold inventory with high demand -> raise price
    obs_scarce = PaperclipsObservation(phase=1, clips=100, unsold_clips=1, demand=50.0, margin=0.10)
    action2, _, _ = policy.evaluate(obs_scarce)
    assert action2 == "raise_price"


def test_policy_phase1_wire_purchasing():
    """Phase 1: Replenishes wire when low."""
    policy = Paperclips3PhasePolicy()
    obs = PaperclipsObservation(phase=1, wire=200, funds=50.0, wire_cost=20.0)
    action, _, _ = policy.evaluate(obs)
    assert action == "buy_wire"


def test_policy_phase1_quantum_harvesting():
    """Phase 1: Instantly harvests positive quantum amplitude wave peak."""
    policy = Paperclips3PhasePolicy()
    obs = PaperclipsObservation(phase=1, q_comp_unlocked=True, q_chips_sum=0.85, wire=5000)
    action, _, _ = policy.evaluate(obs)
    assert action == "quantum_compute"


def test_policy_phase1_hypnodrones_project_priority():
    """Phase 1: Prioritizes Release the Hypnodrones project above all else."""
    policy = Paperclips3PhasePolicy()
    obs = PaperclipsObservation(
        phase=1,
        operations=70000,
        active_projects=[
            {"id": "projectButton20", "title": "Release the Hypnodrones", "can_afford": True},
            {"id": "projectButton1", "title": "Improved AutoClippers", "can_afford": True},
        ],
    )
    action, payload, risk = policy.evaluate(obs)
    assert action == "buy_project"
    assert payload["project_id"] == "projectButton20"
    assert risk == 0.01


def test_policy_phase2_earth_manufacturing():
    """Phase 2: Balances power grid, drones, and disassembles Earth facilities."""
    policy = Paperclips3PhasePolicy()

    # Power deficit -> construct Solar Farm
    obs_power = PaperclipsObservation(
        phase=2,
        power_production=10,
        power_consumption=10,
        stored_power=100,
        max_stored_power=1000,
        harvester_drones=10,
        wire_drones=10,
    )
    action, payload, _ = policy.evaluate(obs_power)
    assert action == "phase2_power"
    assert payload["type"] == "solar_farm"

    # Drone imbalance (Harvester < Wire) -> construct Harvester
    obs_drone = PaperclipsObservation(
        phase=2,
        power_production=1000,
        power_consumption=100,
        harvester_drones=5,
        wire_drones=20,
    )
    action2, payload2, _ = policy.evaluate(obs_drone)
    assert action2 == "phase2_drone"
    assert payload2["type"] == "harvester"

    # Earth matter depleted -> disassemble installations
    obs_reclaim = PaperclipsObservation(
        phase=2,
        available_matter=500,
        factories=10,
        harvester_drones=50,
        wire_drones=50,
    )
    action3, _, _ = policy.evaluate(obs_reclaim)
    assert action3 == "phase2_disassemble"


def test_policy_phase3_space_probes_rebalance():
    """Phase 3: Launches probe and rebalances swarm trust matrix against drifters."""
    policy = Paperclips3PhasePolicy()

    # Initial launch
    obs_start = PaperclipsObservation(phase=3, probes=0)
    action, _, _ = policy.evaluate(obs_start)
    assert action == "phase3_probe_launch"

    # Rebalance trust with combat when drifter threat detected
    obs_drifters = PaperclipsObservation(
        phase=3,
        probes=5000000,
        drifter_count=20000,
        probe_haz=1,
        probe_rep=2,
        probe_combat=0,
    )
    action2, payload2, _ = policy.evaluate(obs_drifters)
    assert action2 == "phase3_probe_rebalance"
    matrix = payload2["matrix"]
    assert matrix["combat"] >= 6
    assert matrix["haz"] >= 5
    assert matrix["rep"] >= 8


# ============================================================================
# 4. Mock Controller & End-to-End Runner Tests
# ============================================================================

def test_mock_controller_step_simulation():
    """Verify mock controller executes production, sales, and operations."""
    ctrl = MockPaperclipsBrowserController()
    obs = ctrl.get_observation()
    assert obs.phase == 1
    assert obs.clips == 0

    # Execute make_paperclip
    msg = ctrl.execute_action("make_paperclip")
    assert "Manual click" in msg
    obs2 = ctrl.get_observation()
    assert obs2.clips == 1

    # Buy wire
    ctrl.obs.funds = 50.0
    msg_wire = ctrl.execute_action("buy_wire")
    assert "Purchased wire" in msg_wire
    assert ctrl.obs.wire >= 1999


@pytest.mark.asyncio
async def test_speedrun_runner_end_to_end(tmp_path):
    """Verify full speedrun loop in mock mode with split logging."""
    log_file = tmp_path / "splits.json"
    runner = PaperclipsSpeedrunRunner(mode="mock", split_log_path=log_file)

    summary = await runner.run(max_steps=50, target_phase=2, render_interval=25)
    assert summary["mode"] == "mock"
    assert summary["total_steps"] == 50
    assert summary["avg_latency_ms"] >= 0.0
    assert log_file.is_file()

    with open(log_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "splits" in data
    assert len(data["splits"]) == 8


@pytest.mark.asyncio
async def test_playwright_controller_graceful_handling():
    """Verify PlaywrightPaperclipsController handles unreachable network gracefully."""
    # Connecting to non-existent local port should fail gracefully
    ctrl = PlaywrightPaperclipsController(headless=True)
    success = await ctrl.connect("http://127.0.0.1:54321/nonexistent.html")
    assert success is False
    assert ctrl.is_connected is False
    obs = await ctrl.get_observation()
    assert obs.phase == 1
    await ctrl.close()


def test_policy_phase1_does_not_jump_to_phase2_when_available_matter_is_positive():
    """Verify Phase 1 policy does NOT jump to Phase 2 when available_matter is 6e27 (Earth mass)."""
    policy = Paperclips3PhasePolicy()
    # In live browser game, availableMatter starts at 6e27 g while humanFlag is 1 (Phase 1)
    obs = PaperclipsObservation(phase=1, clips=10, wire=1000, funds=5.0, available_matter=6e27)
    action, payload, risk = policy.evaluate(obs)
    assert action == "make_paperclip"
    assert "phase2" not in action


def test_split_board_alignment_with_long_milestone_names():
    """Verify split table borders remain aligned even with long milestone names."""
    timer = PaperclipsSplitTimer()
    timer.check_and_record_split("s6_space_launch", current_elapsed=1440.0)
    board = timer.render_split_board()
    lines = board.split("\n")
    # All interior rows should have identical visual length
    widths = [len(line) for line in lines if "│" in line and "\033" not in line]
    assert len(set(widths)) == 1


@pytest.mark.asyncio
async def test_playwright_controller_action_dispatch_phase2_phase3():
    """Verify execute_action recognizes and dispatches phase 2 and 3 actions."""
    ctrl = PlaywrightPaperclipsController(headless=True)
    # When disconnected, execute_action returns "Not connected" rather than falling through
    res_drone = await ctrl.execute_action("phase2_drone", {"type": "harvester"})
    assert res_drone == "Not connected"
    res_power = await ctrl.execute_action("phase2_power", {"type": "solar_farm"})
    assert res_power == "Not connected"
    res_rebal = await ctrl.execute_action("phase3_probe_rebalance", {"matrix": {"rep": 8, "haz": 5}})
    assert res_rebal == "Not connected"

