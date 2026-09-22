# System 1 documentation

Current stable runtime: **1.0.3**. Source documentation reviewed 22 September 2026.
System 1 is not an LLM. Start with one journey: **choose a decision → show
examples → check a candidate → adopt → correct and compare**. A person or a running
teacher can supply the examples.
GitHub can contain documentation and evaluation tools added after the published
package; [CHANGELOG](../CHANGELOG.md) identifies those unreleased source changes.

## Use the product

| Start here | Purpose |
|---|---|
| [README quickstart](../README.md#quickstart) | Install System 1 and run a complete taught skill |
| [Your first skill](guides/first_skill.md) | What an example means, an editable lesson file, corrections and readiness |
| [Correct, compare, and adopt](guides/correcting_skills.md) | Retain lessons, inspect before/after answers, and explicitly adopt a passing candidate; source-only single-choice text workflow |
| [Teach by doing](../examples/teaching_by_doing/README.md) | Use the same candidate/adoption flow by filing fictional documents |
| [Teaching guide](guides/training_experts.md) | Label formats, separate evidence, compilation and corrections |
| [Teacher observation](typesafe.md) | Callback contract, actual Jev/Gemini evidence, promotion and reuse |
| [Example catalog](../examples/README.md) | Every example's role and supported or experimental scope |
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
| [Workload and evidence index](../benchmarks/quality/README.md) | One overview of task scope, held-out or reused evidence, baselines, successes and failed targets |
| [Real-document workflow test](../benchmarks/quality/document_workflow/README.md) | BBC topic routing: fewer accepted mistakes with more review; every candidate rejected by default adoption gates |
| [Teaching examples](../examples/teaching/README.md) | Authored lessons, current counts and historical development |
| [Benchmark index](../benchmarks/README.md) | Separate quality, throughput and historical runners |
| [Research and integration examples](../examples/README.md) | Inbox Zero, n8n, games and composition labs, with each experiment's limits |

The quality index links the public-data, teacher, email, BBC workflow, n8n and
game reports. Research remains available without being a prerequisite for teaching
one useful skill.

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
