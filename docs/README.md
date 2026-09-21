# System 1 documentation

Current stable runtime: **1.0.3**. Reviewed 20 September 2026.
System 1 is not an LLM. Start with one journey: **choose a decision → show
examples → check answers → save and reuse → improve**. A person or a running
teacher can supply the examples.
GitHub can contain documentation and evaluation tools added after the published
package; [CHANGELOG](../CHANGELOG.md) identifies those unreleased source changes.

## Use the product

| Start here | Purpose |
|---|---|
| [README quickstart](../README.md#quickstart) | Install System 1 and run a complete taught skill |
| [Your first skill](guides/first_skill.md) | What an example means, an editable lesson file, corrections and readiness |
| [Teaching guide](guides/training_experts.md) | Label formats, separate evidence, compilation and corrections |
| [Teacher observation](typesafe.md) | Callback contract, actual Jev/Gemini evidence, promotion and reuse |
| [Pokémon teaching lab](../examples/gaming/pokemon_teaching/README.md) | Correct one battle skill, compare equal-budget baselines, and test saved skills in a local ROM |
| [Compositional teaching playground](../examples/gaming/skill_playground/README.md) | Live correction of one skill in a small agent, with held-out missions and baseline evidence |
| [Example catalog](../examples/README.md) | Every example's role and supported or experimental scope |
| [Inbox Zero pilot](../examples/inbox_zero/README.md) | Teach email categories, serve locally, and inspect the initial quality limits |
| [Public real-email recipe](../benchmarks/quality/public_email/README.md) | Download labeled mail, teach spam/ham locally, and reproduce both successful and failed evaluations |
| [Application integrations](integrations.md) | Connect a checked skill to LangChain, MCP, ASGI, gRPC or monitoring |
| [Tool permissions](guides/policy_guard.md) | Add an explicit permission rule when an answer would trigger an action |
| [Deployment boundaries](deployment.md) | Permissions, identity, audit, network paths and service limits |
| [Observability](observability/README.md) | Optional metrics/tracing and dashboard |
| [Container templates](../deploy/README.md) | Experimental local Docker/Kubernetes starting points |

## Understand the implementation and evidence

| Document | Scope |
|---|---|
| [Architecture reference](architecture/technical_specification.md) | Actual components, defaults, code links and lifecycle |
| [Technical brief](paper/system1_technical_brief.md) | Short product and evidence overview |
| [Whitepaper](paper/system1_whitepaper.md) | Current design, measurements and limitations |
| [Mathematical notes](paper/conformal_gating.md) | Established results and their assumptions |
| [Public workloads](../benchmarks/quality/workloads/README.md) | Banking, assistant, SMS, classical baseline and real teachers |
| [1.0.2 quality round](../benchmarks/quality/quality_round/README.md) | Stricter takeover qualification, SMS improvement and fresh diagnostic failures |
| [Teaching examples](../examples/teaching/README.md) | Authored lessons, current counts and historical development |
| [Benchmark index](../benchmarks/README.md) and [quality suite](../benchmarks/quality/README.md) | Reproduction commands and distinct evaluation targets |
| [Game examples](SPEEDRUN_SHOWDOWN_WORLD_RECORDS.md) | Scripted policies, model hooks and emulator limits |
| [First-use game results](../benchmarks/quality/zero_shot/README.md) | Frozen immediate decisions; no campaign/record claim |

The papers are maintainer-authored documentation, not peer-reviewed publications.
Quality claims must identify the task, evidence split and acceptance rule. Local
computation, acceptance without review and correctness are separate measurements.
Do not compare unlike timing cohorts as a measured speedup.

## Release and review history

[1.0.3 security release](releases/1.0.3.md) · [1.0.2 release report](releases/1.0.2.md) · [1.0.1 release report](releases/1.0.1.md) · [1.0 release and migration](releases/1.0.md) ·
[completed stabilization checklist](stable-release-checklist.md) ·
[historical launch review](launch-review.md) · [historical example review](examples-review.md).

Historical reports and raw response/result files are retained as evidence of
what ran then. They are not rewritten to match later improvements. Current guides
link to the appropriate snapshot and disclose failed targets. See
[contributing](../CONTRIBUTING.md) for the documentation check and release procedure.
