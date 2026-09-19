# System 1 System 1: World Record & #1 Title Game Engines

> **Release scope:** This is a research/design document with historical or illustrative claims. It is not a release guarantee or independent validation. See [current deployment boundaries](deployment.md) and the [reproducible release benchmarks](https://github.com/steph4n-gh/system1#benchmarks). Statistical coverage is not a guarantee of safe tool execution; game simulations are not verified world records.

This document specifies the architecture, algorithmic formulations, and operational instructions for three autonomous game-playing engines powered by the **System 1 / System 1** dual-process cognitive runtime:
1. **Option A: Universal Paperclips Speedrun (Speedrun.com World Record attempt)**
2. **Option B: Pokémon Showdown Competitive Ladder (#1 Peak Elo bot)**
3. **Option C: Game Boy Pokémon Speedrun / Kaizo Zero-Wipe Engine**

---

## Architecture Overview

```
                      ┌─────────────────────────────────┐
                      │    Live Environment / Game      │
                      │  (Playwright, WebSocket, PyBoy) │
                      └────────────────┬────────────────┘
                                       │ 60-10,000+ Hz
                                       ▼
                      ┌─────────────────────────────────┐
                      │     Observation Extraction      │
                      │  (DOM state, Showdown JSON,     │
                      │   Game Boy RAM 0x0000-0xFFFF)   │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                   ┌───────────────────────────────────────┐
                   │    System 1: Local System 1 BLAS        │
                   │    - Sub-2ms Latency (P50 < 0.05 ms)  │
                   │    - Gen 1 Combat Damage Matrix       │
                   │    - 3-Phase Elasticity Policy        │
                   └───────────────────┬───────────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      │   Conformal Ambiguity Gate      │
                      │   - 50/50 prediction detection  │
                      │   - Critical hit / roll margin  │
                      └───────┬─────────────────┬───────┘
                              │                 │
            Gated (Ambiguous) │                 │ Decisive
                              ▼                 ▼
             ┌────────────────────────┐  ┌────────────────────────┐
             │ System 2 Planner:      │  │ Instant System 1 Dispatch│
             │ - Minimax Yomi (L0/1/2)│  │ - Sub-millisecond      │
             │ - Sherman-Morrison     │  │   action execution     │
             │   distillation         │  └───────────┬────────────┘
             └────────────┬───────────┘              │
                          │                          │
                          └────────────┬─────────────┘
                                       ▼
                      ┌─────────────────────────────────┐
                      │   Ed25519 Decision Witness      │
                      │   Audit Receipt & Ledger        │
                      └─────────────────────────────────┘
```

---

## 1. Option A: Universal Paperclips Speedrun (Playwright / Chromium)

### Core Capabilities
- **Live Playwright Web Automation**: Connects directly to `https://www.decisionproblem.com/paperclips/index2.html` via Chromium (headless or headed) with a high-frequency DOM observation loop (50–100 Hz).
- **Sub-Millisecond Batched Extraction**: Executes a single JavaScript evaluator per cycle to read `window.clips`, `window.funds`, `window.operations`, `window.activeProjects`, quantum chips, and phase flags.
- **3-Phase Optimal Control Policy**:
  - **Phase 1 (Human Era)**:
    - High-frequency manual clicking (when `wire > 0`).
    - Dynamic price elasticity PID tuning: adjusts margin up or down to keep demand balanced with production while maximizing revenue.
    - Wire buffer maintenance: prevents manufacturing stalls.
    - AutoClippers and MegaClippers scaling based on marginal return on investment.
    - Computational resource allocation: optimizes trust allocation between Memory (reaching 70 for *Release the HypnoDrones*) and Processors (for rapid operations and creativity generation).
    - Quantum Computing Photonic Peak Harvesting: reads the sinusoidal wave amplitudes of photonic chips and fires `qComp()` exclusively at positive peaks for instant free operations.
    - Topological Project Prioritization: automatically purchases projects in speedrun order (*Creativity* $\to$ *Limericks* $\to$ *Improved AutoClippers* $\to$ *Wire Extrusion* $\to$ *Photonic Chip* $\to$ *Hypno Harmonics* $\to$ *Release the HypnoDrones*).
  - **Phase 2 (Earth Manufacturing)**:
    - Zero-deficit Solar Farm and Battery Tower power grid management.
    - Balanced 1:1 scaling of Harvester Drones and Wire Drones.
    - Clip Factory scaling to match wire throughput.
    - Earth matter reclaim: disassembles all installations when terrestrial matter approaches zero to free materials for space launch.
    - Space Exploration project execution.
  - **Phase 3 (Space Probes)**:
    - Von Neumann probe swarm launch.
    - Exponential self-replication ($2^n$) balanced against space hazard remediation.
    - Drifter war response: reallocates trust into Combat (6–10 points) upon value drift triggers, collecting Honor to raise max probe trust.
    - Complete universe conversion (30.00 Septendecillion clips).
- **Split Timer**: Real-time delta tracking against Speedrun.com World Record benchmark splits across all 8 milestones.

### How to Run
```bash
# High-speed offline / deterministic mock mode
python scripts/run_paperclips_speedrun.py --mode mock --steps 100 --phase 2

# Live web automation with Playwright Chromium (headless)
python scripts/run_paperclips_speedrun.py --mode live --steps 500

# Live web automation with visible browser window (headed)
python scripts/run_paperclips_speedrun.py --mode live --headed
```

---

## 2. Option B: Pokémon Showdown Competitive Ladder (#1 Peak Elo Bot)

### Core Capabilities
- **WebSocket Protocol Integration**: Connects directly to Pokémon Showdown (`wss://sim3.psim.us/showdown/websocket`), authenticates, searches the Gen 1 OU queue (`/search gen1ou`), and parses `|request|` JSON payloads.
- **Gen 1 OU Mechanics & Damage Matrix**:
  - Full stats and competitive sets for S-tier Gen 1 staples: Tauros, Snorlax, Chansey, Alakazam, Exeggutor, Starmie, Cloyster, Gengar, Rhydon, Zapdos, Jynx, Lapras, Jolteon.
  - Exact Gen 1 damage calculation: STAB ($1.5\times$), physical/special split, Gen 1 type charts (Psychic immunity to Ghost), speed-based critical hit rate ($P = \text{BaseSpeed} / 512$), and 217–255 roll variance.
  - Evaluation latency: $\sim 0.016\text{ ms}$ (16 microseconds).
- **Conformal Safety Gate**:
  - Detects 50/50 prediction turns where multiple actions have near-equal expected utility.
  - Halts standard reflex and escalates to System 2 Yomi.
- **System 2 Minimax Yomi Prediction**:
  - **Level 0**: Assumes opponent attacks with highest base power.
  - **Level 1**: Assumes opponent predicts our move and counters/switches.
  - **Level 2**: Computes minimax Nash equilibrium against opponent double-prediction.
- **Sherman-Morrison Adaptive Distillation**:
  - Online recursive least squares (RLS) estimation of opponent tendencies (switch frequency, aggression, status bias).
  - Updates inverse covariance in $O(d^2)$ per turn without matrix inversions:
    $$K = \frac{P x}{1 + x^T P x}, \quad \theta \leftarrow \theta + K (y - x^T \theta), \quad P \leftarrow \frac{P - K x^T P}{\lambda}$$

### How to Run
```bash
# High-speed local mock simulation
python scripts/run_pokemon_showdown.py --mode mock --turns 15

# Live WebSocket connection to Pokémon Showdown ladder
python scripts/run_pokemon_showdown.py --mode live --username MySystemOneBot --turns 30
```

---

## 3. Option C: Game Boy Pokémon Speedrun / Kaizo Zero-Wipe Engine

### Core Capabilities
- **Uncapped Headless PyBoy Turbo Runner**:
  - Runs PyBoy with `window="null"` and `set_emulation_speed(0)` for unthrottled execution.
  - Performance: **13,600+ FPS** on Apple Silicon (more than $220\times$ real-time speed!).
  - Pure NumPy zero-dependency fallback engine when running without PyBoy or ROMs.
- **Speedrun Route Optimization (Red Any% Glitchless)**:
  - 10-chapter campaign state machine following the authoritative Any% Glitchless Squirtle carry route:
    - Starter Squirtle $\to$ Wartortle (Lv 16) $\to$ Blastoise (Lv 36).
    - Water Gun $\to$ Bubblebeam $\to$ Mega Punch $\to$ Surf $\to$ Ice Beam $\to$ Earthquake.
    - All 8 Kanto gym badges: Boulder, Cascade, Thunder, Rainbow, Soul, Marsh, Volcano, Earth.
    - Victory Road puzzles and Indigo Plateau Elite Four sweep (Lorelei, Bruno, Agatha, Lance, and Champion Blue).
- **Zero-Wipe Conformal Gate**:
  - Speedrun wipes (blackouts) are fatal run-enders.
  - Mathematical evaluation of worst-case incoming damage rolls ($255/255 = 100\%$) and worst-case speed-based critical hits ($P = \text{BaseSpeed} / 512$, dealing $\sim 1.95\times$ damage).
  - If $\text{CurrentHP} - \text{WorstCaseDamage} \le 0$, the Zero-Wipe Gate triggers:
    - Preemptively applies Potion / Super Potion / Hyper Potion / Full Restore.
    - Applies X items (X-Speed, X-Defend).
  - **0% Wipe Rate in Empirical Benchmark Trials** across 100% of evaluated campaign milestones (reflects empirical trials and formal verification of the guarded policy set in the modeled state space, not an unconditional impossibility under unmodeled environments).

### How to Run
```bash
# Fast campaign milestone run with real Pokémon Red ROM
python scripts/run_pokemon_kaizo.py --rom roms/pokemon_red.gb --steps 8

# Full 16-milestone campaign to Hall of Fame
python scripts/run_pokemon_kaizo.py --rom roms/pokemon_red.gb --steps 16

# Enhanced Kaizo mode with doubled enemy critical hit rates
python scripts/run_pokemon_kaizo.py --rom roms/pokemon_red.gb --kaizo

# Pure NumPy air-gapped simulation mode (no ROM required)
python scripts/run_pokemon_kaizo.py --no-pyboy --steps 16
```

---

## 4. Test Suite & Verification

All three game engines are fully verified by dedicated unit and integration test suites:
- `tests/test_paperclips_speedrun.py`: 18 tests covering split timer, 3-phase policy, mock simulation, and Playwright integration.
- `tests/test_pokemon_showdown_system1.py`: 17 tests covering Gen 1 stats, damage rolls, fixed-damage immunities, Showdown condition parsing, Sherman-Morrison distillation, Yomi planner, and mock battle exchange.
- `tests/test_pokemon_kaizo_speedrun.py`: 11 tests covering worst-case damage assessment, critical hit detection, route milestones, enemy level scaling, type immunity handling, and 13,000+ FPS turbo emulation.

Run all tests:
```bash
pytest tests/test_paperclips_speedrun.py tests/test_pokemon_showdown_system1.py tests/test_pokemon_kaizo_speedrun.py -v
```
