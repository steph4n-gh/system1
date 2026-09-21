# Example catalog

Run examples from a repository checkout with System 1 installed. The primary
workflow is **teach or observe → validate → run locally → save and reuse**.
The [teaching guide](../docs/guides/training_experts.md) and
[teacher adapter](../docs/typesafe.md) describe the supported core.

## Start with a bounded skill

| Example | What it demonstrates |
|---|---|
| [teach_skill.py](teach_skill.py) | Eight editable lessons; custom lesson file, message and saved path; [beginner walkthrough](../docs/guides/first_skill.md) |
| [support_triage.py](support_triage.py) | Taught department routing, calibration, saved/reloaded evaluation |
| [model_routing.py](model_routing.py) | A stated routing policy; does not call downstream models |
| [agent_guard.py](agent_guard.py) | Operation classification plus a separate explicit permission/audit example |
| [banking_support.py](banking_support.py) | Three BANKING77 intents using original public labels |
| [observe_routing.py](observe_routing.py) | Default-gate takeover from an offline rule or optional actual Jev, then saved reuse |
| [_teaching_demo.py](_teaching_demo.py) | Shared measurement helper, not a standalone task |

```bash
python examples/support_triage.py
python examples/observe_routing.py
python examples/banking_support.py
```

[Teaching data/results](teaching/README.md) report accepted accuracy and review
rates separately. The [public-workload runner](../benchmarks/quality/workloads/README.md)
adds six assistant intents, SMS and actual Jev/Gemini takeover with offline replay.
No live credentials are needed for the commands above.

## API and experimental demonstrations

| Example | Scope and evidence boundary |
|---|---|
| [core_standalone_evaluator.py](core_standalone_evaluator.py) | NumPy-only core and typed outputs; sample predictions are not a quality benchmark |
| [train_expert.py](train_expert.py) | Legacy name for synthetic/supplied examples, correction and composition; these are optional teaching mechanisms |
| [four_levers_benchmark.py](four_levers_benchmark.py) | Cache, online correction, margins and numerical telemetry; repeated scripted feedback is not generalization evidence |
| [typesafe_sdk_dropin_showcase.py](typesafe_sdk_dropin_showcase.py) | Local SDK calls, patching, async access and optional comparison; only the documented SDK surface is supported |
| [auto_cutover_showcase.py](auto_cutover_showcase.py) | Legacy augmented/relaxed-promotion simulation; clinical/financial schemas are unvalidated illustrations |
| [autonomous_agent_firewall_showcase.py](autonomous_agent_firewall_showcase.py) | Multi-field observation, concurrency and audit; default lookup teacher and WAN timings are simulated, with relaxed promotion settings |
| [enterprise_stress_showcase.py](enterprise_stress_showcase.py) | Alternate entry point for the same firewall demonstration |
| [jev_comparison_demos.py](jev_comparison_demos.py) | Local multi-field schemas; does not establish general Jev parity |
| [deep_jev_benchmark.py](deep_jev_benchmark.py) | Optional HTTP comparison harness; successful calls and failed requests are distinguished, no observed-skill qualification |
| [killer_use_cases_live_test.py](killer_use_cases_live_test.py) | A few typed-schema scenarios with optional live comparisons, not validated domain specialists |
| [paperclips_typesafe_dropin.py](paperclips_typesafe_dropin.py) | Structured game-state adapter, actual returned distributions and optional observation; no verified real-game completion |

Cloud modes require explicit configuration and may make billable calls. Prefer
`observe_routing.py` and the workload runner for normal promotion evidence. A
statistical action label in any example does not itself grant a tool permission.

## Gaming and emulation

| Example | Actual control path |
|---|---|
| [gaming/pokemon_battle_system1.py](gaming/pokemon_battle_system1.py) | Default immediate-damage/healing rules; supplied engines or saved skills can control strategy, with legal-action checks |
| [gaming/pokemon_full_campaign_speedrun.py](gaming/pokemon_full_campaign_speedrun.py) | Scripted route/level progression, battle integration and real adapter promotion state |
| [gaming/pokemon_gameboy_gui.py](gaming/pokemon_gameboy_gui.py) | Optional RAM advisor, navigation scripts and partial action-to-button mapping |
| [gaming/pokemon_all_games_benchmark.py](gaming/pokemon_all_games_benchmark.py) | Optional cartridge/emulator throughput and decision measurements, not game completion |
| [gaming/pokemon_kaizo_speedrun.py](gaming/pokemon_kaizo_speedrun.py) | Scripted milestones and damage-risk heuristic; no fitted conformal calibration or taught decision model |
| [gaming/pokemon_showdown_system1.py](gaming/pokemon_showdown_system1.py) | Damage heuristic and minimax fallback, mock/WebSocket transport; opponent-update helper is not called by the supplied loops |
| [gaming/paperclips_speedrun.py](gaming/paperclips_speedrun.py) | Handwritten three-phase policy with accelerated mock and optional Playwright controller |
| [gaming/teach_paperclips_wire.py](gaming/teach_paperclips_wire.py) | Teach, save and check one numeric restocking skill; [live demonstration and limitations](gaming/PAPERCLIPS_TEACHING.md) |
| [gaming/snake_arena/](gaming/snake_arena/README.md) | Three-player GUI: taught System1 skill vs real Laya-MLX and Jev; timed/equal-move runs, explicit planner hints and optional shield; [recordings and results](gaming/snake_arena/RESULTS.md) |
| [gaming/pokemon_teaching/](gaming/pokemon_teaching/README.md) | Two taught battle skills, live healing correction, measured regressions and optional real Red/Blue ROM check |
| [gaming/skill_playground/](gaming/skill_playground/README.md) | Three taught skills, live terrain correction, held-out mission composition and measured single-model/rules baselines |
| [gaming/__init__.py](gaming/__init__.py) | Package marker |

See [game commands and limits](../docs/SPEEDRUN_SHOWDOWN_WORLD_RECORDS.md).
The [frozen first-use cases](../benchmarks/quality/zero_shot/README.md) distinguish
rule-based improvements from raw model quality. ROMs are not included; live
browser play, public ladder results and complete model-controlled campaigns have
not been established by these evaluations.
