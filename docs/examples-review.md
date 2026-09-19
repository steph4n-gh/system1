# Complete example review — 19 September 2026

> Historical review of 0.2.2 and earlier behavior. For resolved findings, current results, and supported versus experimental scope, see the [1.0 release evidence](releases/1.0.md). The measurements below are retained as review history.

**Release follow-up:** the ordered corrections below have now been implemented.
The new [flagship](../examples/observe_routing.py) adds one entry point (25 Python
files total). It passes the normal gates using observed labels, disconnects its
offline rule teacher, and checks complete reload equivalence. The
[adapter contract](typesafe.md) describes supported SDK calls and release boundaries.
The original review findings and measurements below remain as the pre-change
record. The two enterprise showcases explicitly retain their legacy augmentation
and relaxed gates; they are simulation examples, not production quality evidence.

Takeover/reload regressions now cover the complete projector and calibration
state. Paperclips uses explicit live-mode configuration and actual probabilities;
campaign cutover uses the adapter's real promotion state. Compiled battle skills
use the engine's saved uncertainty state. Failed HTTP comparisons are excluded,
and scripted gaming and repeated-stream benchmarks state their evidence limits.


System 1's central product journey is **observe a teacher → teach a small skill →
validate → take over locally → save and reuse the skill**. A teacher can be Jev,
another API through a callback, labeled historical data, or an individual
correction. The repository already contains much of this journey, plus typed
outputs, numeric telemetry, online updates, reusable skills, uncertainty checks,
and signed audit records. Evaluating only the starter classifier misses that
value. The three new manual teaching examples demonstrate one entry point into
this product; they should not become its entire story.

The next work should finish and demonstrate this existing journey. It does not
require a new training platform, a language-model fine-tuning pipeline, or another
abstraction layer.

## Scope and evidence

Reviewed all **24 Python files** under `examples/` (22 entry points and two support
modules), including their schemas, data sources, teaching paths, decision/action
loops, reporting, and relevant implementation and tests. Also reviewed the three
teaching datasets, their three recorded results, and their documentation.

Focused tests: **193 passed, 10 skipped in 19.61 seconds** on the review machine.
They cover automatic cutover and its invariants, the two enterprise showcases,
online learning and telemetry, compiler workflows, core isolation, primary
examples, and the gaming implementations. Optional skips include missing MLX and
ROM-dependent paths. Passing these tests establishes their tested behavior, not
all the marketing claims printed by the examples.

Seven separate entry points completed with socket connections and HTTP requests
blocked: `core_standalone_evaluator`, `train_expert`, `four_levers_benchmark`,
`jev_comparison_demos`, and the mock Paperclips speedrun, mock Showdown, and Kaizo
examples. Local logs and additional probes are in the ignored
`.system1/example-review/` directory. Live Jev comparisons, browser gameplay,
public Showdown matches, and ROM-dependent GUI behavior were not validated in
this pass.

## What every example demonstrates

Paths in this table are relative to `examples/`.

| Example | Capability demonstrated | Evidence boundary / next action |
|---|---|---|
| `teach_skill.py` | Eight labeled examples, compile, save, reload, local decision. | Smallest manual-teaching introduction; explicitly requests review with this little evidence. |
| `support_triage.py` | Teaches department routing from explicit examples. | 21/24 correct on the recorded authored evaluation; one response accepted without review. Keep as a manual-teaching companion. |
| `model_routing.py` | Teaches a stated request-to-execution-tier policy. | 22/24 correct on authored evaluation; all request review. Classifies requests without invoking downstream models. |
| `agent_guard.py` | Teaches operation triage; separately demonstrates deterministic permission checks and signed execution outcomes. | 24/24 triage labels correct on authored evaluation, all request review. Actual tool permission comes from the explicit policy. |
| `_teaching_demo.py` | Shared teach/save/reload/evaluate/report workflow for those three examples. | Reports accuracy and review rate separately and checks split overlap. Support module. |
| `core_standalone_evaluator.py` | NumPy-only core, several typed outputs, single and batch inference. | Isolation and execution work; its sample predictions do not establish application quality. |
| `train_expert.py` | Synthetic and supplied teaching data, portable skills, individual feedback, router-to-specialist composition. | Offline entry point passes. In the observed feedback example, severity changes from routine to critical after one correction. Some printed timing/size success claims are unconditional. |
| `four_levers_benchmark.py` | Exact/similarity cache, online correction, margin gating, text plus numeric telemetry. | Offline run changes the same prompt from allow to block under different telemetry, and updates an edge case after feedback. Its repeated-stream escalation result uses a fixed teacher answer and is not a quality benchmark. |
| `typesafe_sdk_dropin_showcase.py` | Import replacement, monkey patching, async calls, local receipts, optional comparisons. | Demonstrates the adapter's own interface; does not establish compatibility with the official SDK's complete contract. |
| `auto_cutover_showcase.py` | Observation, compilation, validation, per-schema promotion, local execution, ledger and export, across three domain schemas. | Uses a relaxed demo promotion policy and permits simulated teacher fallback. Financial and clinical schemas are illustrations, not validated domain solutions. |
| `autonomous_agent_firewall_showcase.py` | Five simultaneous output fields, teaching from observed answers, cutover, uncertainty, concurrent inference, and audit integrity. | Strong existing lifecycle example. Its default teacher is a labeled lookup table with simulated WAN metrics. See the measured replay below. |
| `enterprise_stress_showcase.py` | Alternate entry point for the firewall example. | Reuses that implementation; not a separate capability or independent benchmark. |
| `jev_comparison_demos.py` | Multi-field routing, support triage, proposed-action classification, and multi-label moderation. | Runs locally. Its comparison table contains fixed claims, and the proposed-action example does not invoke a hardware enforcement boundary. |
| `deep_jev_benchmark.py` | HTTP comparison harness for routing, action triage, and moderation. | No observed-teacher teaching step. Failed HTTP requests still enter latency/speedup summaries; correct this before treating results as comparative evidence. |
| `killer_use_cases_live_test.py` | Typed schemas for smart-home intent, financial alerts, code triage, passage relevance, and insurance triage. | Breadth of integration examples, not five validated specialist skills. One sample per scenario and latency alone cannot establish quality; failed HTTP requests still get comparison timing. |
| `paperclips_typesafe_dropin.py` | Structured game state serialized as text, dynamic action choices, trajectory logs, adapter comparison and cutover modes. | Current cutover mode fails at client construction because its network/teacher configuration conflicts with `zero_egress=True`. Logs also reconstruct distributions from confidence instead of retaining the returned distribution. |
| `gaming/paperclips_speedrun.py` | Observe/action loop with mock and Playwright browser controllers. | The runner executes a handwritten three-phase policy; it does not call the System 1 model or compiler. Its speed is not evidence of learned control or a verified world record. |
| `gaming/pokemon_battle_system1.py` | Model-backed battle decisions, compiled skills, tactical escalation, emulator memory bridge and action adapter. | Real runtime integration exists. The strategic planner is scripted; the compiled branch builds heuristic uncertainty sets from confidence rather than using saved conformal calibration. |
| `gaming/pokemon_full_campaign_speedrun.py` | Campaign orchestration, battle-model calls, skill loading, synthetic state, optional emulator connection. | Route progression and leveling are scripted. Its separate cutover flag counts turns without teaching from observed answers or checking promotion. Replace that duplicate path with the real adapter before claiming observation-based takeover. |
| `gaming/pokemon_gameboy_gui.py` | Live RAM observation, visual advisor, navigation scripts and limited action-to-button mapping. | Useful integration prototype. Live-battle mode advances with scripted button presses; another mode dispatches broad predicted actions. This is not evidence of a complete model-controlled campaign. |
| `gaming/pokemon_all_games_benchmark.py` | Cartridge metadata, emulator throughput, RAM extraction, decision latency and escalation measurements. | Measures integration/performance, not game completion or strategic quality. ROM-dependent checks were skipped here. |
| `gaming/pokemon_kaizo_speedrun.py` | Handwritten damage-risk checks and a scripted route with optional emulator ticking. | Does not teach or invoke a decision model. The damage heuristic is called conformal in the example but has no fitted conformal calibration; milestones are assigned by code. |
| `gaming/pokemon_showdown_system1.py` | Damage-based action policy, a small online opponent model, minimax fallback, receipts, mock and WebSocket transport. | The main policy uses damage rules rather than the System 1 decision model. The opponent-update method exists but neither supplied battle loop calls it. Live parsing also leaves opponent state at defaults. No ladder result is established. |
| `gaming/__init__.py` | Package marker. | Support module, no executable capability. |

## Direct replay of the existing observation example

Replayed the 48 authored firewall cases in their existing interleaved order,
using `make_firewall_baseline_handler`, `HybridProjector(dimension=384)`, and the
example's promotion policy. No live teacher was used and no dataset labels were
changed.

| Measurement | Result |
|---|---:|
| Teacher responses before promotion | 20 |
| Subsequent local responses | 28 |
| Correct local action labels | 25/28 (89.3%) |
| Local responses marked `abstain` | 26/28 |
| Responses accepted without abstention | 2/28, both action labels correct |

This demonstrates actual learning and local execution on subsequent cases. It
also shows why **local execution rate, action-label accuracy, and autonomous
completion rate must be reported separately**. The five-field schema can abstain
because another field is uncertain even when its action label is correct.
The current cutover compiler also adds schema-derived synthetic examples; the
observed teacher responses are not its only source of labels. The manual
`augment=False` teaching path is available if the intended demonstration is
learning exclusively from observed answers.

The example disables the statistical-bound requirement. Keeping the same
20-observation trigger but using the production promotion policy resulted in no
promotion during those 48 cases. That is not the same as testing the constructor's
default 50-observation trigger. A small turn count is a demo setting, not a
general guarantee of sufficient evidence.

Exporting the promoted firewall skill and reopening it with the compatibility
client, with the same dimension and projector, preserved all 28 action labels
but changed three action prediction sets. The largest action-confidence change
was approximately 0.563. The adapter's `_get_engine` copies weights into a new
engine without restoring the full saved calibration. This undermines the promise
that a validated skill can be saved and reused with the same behavior.

A separate campaign probe reached `has_cutover=True` after two decisions with
the teacher client disabled and zero cloud observations. That confirms the
campaign's counter-based switch is distinct from the real teaching lifecycle.

## Ordered next steps

1. **Repair the existing observation-to-reuse journey.** Preserve the evaluated
   skill's projector, weights, calibration, schema identity, and uncertainty
   behavior when promoting, exporting, and reloading it through the adapter.
   Add a regression that checks predictions, distributions, and review decisions
   before and after reload. Check that promotion measures useful local completion
   as well as label agreement. Keep the teacher active when the evidence is
   insufficient; do not lower the default standard just to show a quick cutover.

2. **Make the Jev integration match the claimed developer experience.** The
   [official Python quickstart](https://github.com/typesafe-ai/typesafe-sdk-python)
   uses a client context manager, object-valued `state`, and the response's
   `.choices` accessor; the local adapter currently fails those calls. Score also
   lacks the official per-level probabilities and legend. Fix and test the
   supported contract, and state any remaining differences precisely. Another API
   can already supply answers through the existing teacher callback; there is no
   need to build a provider framework for the demonstration.

3. **Make one short observation example the flagship.** Use a bounded workflow
   such as support routing: let a teacher answer, show the accumulating evidence,
   attempt validated promotion, stop teacher calls, evaluate fresh cases locally,
   then save and reopen the skill. Report observation count, teaching time,
   teacher-call count, held-out agreement, accepted fraction, errors, artifact
   size, and local latency. A deterministic offline teacher makes the tutorial
   easy to run; label it clearly. A separate real-Jev run is needed for any Jev
   parity claim. Keep the three manual teaching examples as useful companions.

4. **Bring the other examples into agreement with their implementations.** Fix
   the Paperclips cutover configuration. Reuse the adapter in the campaign
   example instead of maintaining a second counter-based cutover. Label scripted
   policies and simulated teachers explicitly; retain actual response
   distributions. Calculate success messages from results, exclude failed API
   calls from performance comparisons, and avoid treating game simulations as
   evidence of live completion or competitive records. Preserve the useful demos
   without presenting them as independent quality benchmarks.

These are focused release corrections to existing behavior and examples. They
take priority over expanding the architecture or adding more workloads. After
they pass targeted checks, run the full release checks and remote CI before
tagging the prepared version.
