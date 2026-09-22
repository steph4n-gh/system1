# Gaming demonstrations and their limits

Current scope for **System 1 1.1.0**. The legacy filename is retained for links;
this project has no independently verified speedrun world record or #1 ladder
result. These examples explore game rules, local decisions, observation loops
and optional emulator/browser integrations.

## First-use behavior

The [frozen first-use evaluation](../benchmarks/quality/zero_shot/README.md)
checks 24 Pokémon and 12 Paperclips immediate decisions with teaching and network
access disabled. Explicit game rules improve the former from 6/24 to 24/24 and
the latter from 8/12 to 12/12. The raw Pokémon model suggestion remains 6/24;
the raw routing control remains 8/12. These are small authored regression cases,
not evidence of learned long-term strategy or game completion.

Pokémon ranks currently legal moves by immediate damage and heals at low HP when
an item exists. A supplied engine or taught skill retains its legal strategy;
invalid suggestions receive a legal fallback and a review signal. Paperclips
uses observed wire-unit prices and affordable purchases. Neither policy attaches
invented model confidence to a rule decision.

## Run the offline demonstrations

From an installed repository checkout:

```bash
python benchmarks/quality/evaluate_zero_shot.py
python scripts/run_paperclips_speedrun.py --mode mock --steps 100 --phase 2
python scripts/run_pokemon_showdown.py --mode mock --turns 15
python scripts/run_pokemon_kaizo.py --no-pyboy --steps 16
```

| Demonstration | What is implemented | What the result does not establish |
|---|---|---|
| Paperclips | Handwritten three-phase policy, accelerated mock economy, optional browser state/action loop, illustrative split targets | Learned control, real-game economic optimality or a speed record |
| Showdown | Damage heuristics, limited minimax fallback, receipts, mock and WebSocket transport | Full battle-state fidelity, competitive ladder strength or fitted conformal coverage |
| Kaizo | Scripted campaign milestones, damage-risk interventions and optional emulator ticking | Autonomous campaign navigation, zero wipe probability or a conformal safety theorem |
| Model-backed battle/campaign prototypes | Supplied model/skill integration, legal-action handling and real adapter promotion state | Complete model-controlled gameplay or promotion after a fixed turn count |

The Showdown opponent-model update helper exists, but the supplied loops do not
call it. Kaizo's historical `conformal_*` identifiers are compatibility names for
a damage heuristic, not fitted prediction sets. Paperclips split targets are
illustrative constants without verified record provenance.

## Optional external environments

Live Paperclips uses Playwright and a browser installation, separate from the
base package:

```bash
python -m pip install playwright
python -m playwright install chromium
python scripts/run_paperclips_speedrun.py --mode live --headed --steps 500
```

This contacts the game website and performs game actions. Failed observations
stop the run rather than substituting a fabricated state. Live behavior and
timing depend on the website; the release quality checks use the offline mock.

The Showdown live entry point requires `websockets` and connects to a public
service. Its CLI exposes `--mode live` and `--username`; use it only in an
appropriate testing context. No live ladder result is recorded here.

Game Boy paths require `python -m pip install 'system1[gameboy]'` and a legally
obtained compatible ROM supplied by the user. For the optional Kaizo integration:

```bash
python scripts/run_pokemon_kaizo.py --rom roms/pokemon_red.gb --steps 16
```

Attaching PyBoy or ticking emulator frames does not turn scripted milestone
progression into a measured ROM speedrun. The complete [example catalog](../examples/README.md)
identifies the individual files and control paths.

## Verification

```bash
python -m pytest tests/test_zero_shot_games.py tests/test_paperclips_speedrun.py tests/test_pokemon_showdown_system1.py tests/test_pokemon_kaizo_speedrun.py -q
```

These checks cover the modeled rules and integration contracts. Optional emulator
checks depend on installed extras and assets; skipped tests are not live gameplay
validation. Report policy accuracy, legal actions, model review flags, emulator
throughput and actual game completion separately.
