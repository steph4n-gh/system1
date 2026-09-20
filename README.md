<p align="center">
  <img src="https://raw.githubusercontent.com/steph4n-gh/system1/main/assets/system1-logo.jpg" alt="System 1 logo" width="128" />
</p>

# System 1: Local Decision Runtime for AI Agents

**System 1 is not an LLM.** Teach a bounded decision skill from examples or by observing a teacher, validate it, then run it locally and save it for reuse. The built-in runtime uses small numerical decision heads: no language-model weights to download, no token generation, and no GPU required. Optional **MLX support** runs decision-head matrix operations on Apple Silicon.

A teacher can be a person, a JSON file, a rule, Jev, Gemini or another API. An LLM can supply the lessons; the resulting local skill is not an LLM. System 1 chooses among outputs you define. It does not generate prose or inherit a teacher's general knowledge.

[![CI](https://github.com/steph4n-gh/system1/actions/workflows/ci.yml/badge.svg)](https://github.com/steph4n-gh/system1/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/system1.svg)](https://pypi.org/project/system1/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

**Version 1.0.2:** explicit quality requirements for accepted decisions, fuller SMS observation, and an honest teaching-quality evaluation. The core remains teaching bounded decisions, validated local takeover, portable skills, and explicit tool policies. See the [results and remaining limits](https://github.com/steph4n-gh/system1/blob/main/docs/releases/1.0.2.md) and [1.0 migration guide](https://github.com/steph4n-gh/system1/blob/main/docs/releases/1.0.md).

[Quickstart](#quickstart) · [Policy guard](#policy-guard) · [Integrations](#integrations) · [Benchmarks](#benchmarks) · [All documentation](https://github.com/steph4n-gh/system1/blob/main/docs/README.md) · [Limits and deployment](https://github.com/steph4n-gh/system1/blob/main/docs/deployment.md) · [Contributing](https://github.com/steph4n-gh/system1/blob/main/CONTRIBUTING.md)

## What it does

- **Observe → teach → validate → run locally:** keep the teacher answering until the observed skill passes held-out checks, then use the same call site locally. Save and reload the validated skill with its uncertainty behavior intact.
- **Structured classification:** choice, boolean, multi-choice, and score fields using local NumPy projections, with optional MLX acceleration.
- **Uncertainty handling:** calibration and conformal prediction sets for routing uncertain decisions to application-defined review or fallback paths.
- **Deterministic permissions:** `PolicyEngine` evaluates explicit rules; `SystemOneGuard(enforcement_profile=True)` requires a matching permission grant, signing key, and durable ledger before returning `ALLOW`.
- **Audit evidence:** Ed25519 software signatures (RFC 8032) on decision receipts and a SHA-256 hash-chained SQLite ledger. Applications remain responsible for authenticating callers and enforcing decisions at the tool boundary.
- **Local execution:** the core decision path requires no network or cloud API. Optional fallback clients, telemetry exporters, and application tools can use the network.

## Quickstart

The project is packaged on PyPI as `system1`. Use Python 3.11 or newer:

```bash
python -m pip install system1
```

To try the code in this repository:

```bash
git clone https://github.com/steph4n-gh/system1.git
cd system1
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Start with a complete taught skill:

```bash
python examples/support_triage.py
system1 decide "Please correct the invoice address" --model .system1/examples/support_triage/skill.s1m --json
```

The example teaches support routing, calibrates uncertainty, saves the skill,
reloads it, and checks separate evaluation cases. It runs entirely locally.
For the smallest self-contained Python API example, see [teach one skill](https://github.com/steph4n-gh/system1/blob/main/examples/teach_skill.py).
A predicted value is not permission to execute a tool.

## Observe a teacher, then take over

```bash
python examples/observe_routing.py
```

This example observes answers, teaches one routing skill, passes the normal promotion gates, disconnects the teacher, evaluates fresh requests locally, then saves and reopens the skill. Both the offline run and the **live Jev run** promoted after **357 observations**, answered **40/40 fresh synthetic tickets correctly**, accepted all 40, and saved a **5 KB** skill. Reloading preserved answers, probabilities, and review decisions. Live observation took about two minutes; local median decisions took about **0.4 ms** on the review machine.

The default teacher is a rule over a small structured-ticket vocabulary. `--teacher jev` uses actual Jev responses with your API key; another API can use the existing teacher callback. These tickets demonstrate the lifecycle, not broad language understanding. See the [adapter contract](https://github.com/steph4n-gh/system1/blob/main/docs/typesafe.md), [live result](https://github.com/steph4n-gh/system1/blob/main/benchmarks/quality/results/jev_live_cutover.json), and [three-skill Jev comparison](https://github.com/steph4n-gh/system1/blob/main/docs/releases/1.0.md).

Promotion depends on evidence, not a fixed turn count. Insufficient evidence keeps the teacher active, and uncertain local responses still request review.

On a separate public six-intent assistant workload, actual **Jev and Gemini 2.5 Flash**
each enabled takeover after **358 observations**. Each resulting skill accepted
**171/180** unseen official test requests locally, with **167/171 correct (97.7%)**,
and made no further teacher calls. The [three-workload comparison](https://github.com/steph4n-gh/system1/blob/main/benchmarks/quality/workloads/README.md)
includes a classical baseline and unsuccessful takeover results. Its recorded
teacher responses can be replayed without credentials.

For a stronger requirement, use `PromotionPolicy(min_accepted_agreement=.95)`.
The [1.0.2 quality round](https://github.com/steph4n-gh/system1/blob/main/benchmarks/quality/quality_round/README.md) waited for
2,961 original-label SMS observations and reached **954/980 correct accepted
decisions (97.3%)** on the earlier 1,001-case test. Banking and assistant remained
deferred under this stricter policy. Fresh authored SMS probes reached only
**27/33 correct accepted decisions (81.8%)**. The stronger gate helps qualify a
specific workload; representative data and independent evaluation still matter.

## Teach a skill

Show System 1 an input and the answer you want: “Refund my payment” → `billing`. That answer is what “label” means. Save the resulting `.s1m` skill and reuse it locally. A person or System 2 can supply the examples; teaching does not require an LLM. The existing compiler fits a small decision head using NumPy.

**New to teaching?** The [first-skill walkthrough](https://github.com/steph4n-gh/system1/blob/main/docs/guides/first_skill.md) explains the editable teaching file, correcting a mistake, review requests, and checking whether a skill is ready.

```bash
python examples/support_triage.py
python examples/model_routing.py
python examples/agent_guard.py
system1 decide "Please correct the invoice address" --model .system1/examples/support_triage/skill.s1m --json
```

Each command teaches from explicit examples, uses separate calibration cases, then saves, reloads, and evaluates the skill. On the review machine, teaching plus calibration took about **0.2 seconds**, saved skills were **25–32 KiB**, and uncached decisions took about **0.5 ms**. Preparing and reviewing labeled examples takes additional work.

| Taught skill | Accepted locally | Correct among accepted |
|---|---:|---:|
| Support triage | 92/104 (88.5%) | 91/92 (98.9%) |
| Model routing | 94/96 (97.9%) | 94/94 (100%) |
| Operation triage | 126/132 (95.5%) | 126/126 (100%) |

These are current results on the primary authored demonstration cohorts; the support result contains one accepted error. A harder fresh routing cohort reaches only 41/44 correct among accepted answers (93.2%). The recorded **actual Jev** comparison reproduces the original 1.0 lessons separately. See [data and confidence intervals](https://github.com/steph4n-gh/system1/blob/main/examples/teaching/README.md) and [1.0.1 results and remaining errors](https://github.com/steph4n-gh/system1/blob/main/docs/releases/1.0.1.md). These results do not establish general Jev parity or security detection capability.

For an externally labeled example, teach three banking-support intents from the bundled BANKING77 data:

```bash
python examples/banking_support.py
system1 decide "My card was stolen" --model .system1/examples/banking_support/skill.s1m --json
```

On the complete official test slice for those three intents, the saved skill accepts 118/120 queries, with 115/118 correct (97.5%). Teaching and calibration take about 0.2 seconds locally. This measures three intents, not the full 77-intent benchmark. See [source, license, and split details](https://github.com/steph4n-gh/system1/blob/main/examples/teaching/BANKING77.md).

For the smallest API example, see [teach one skill](https://github.com/steph4n-gh/system1/blob/main/examples/teach_skill.py). The [teaching guide](https://github.com/steph4n-gh/system1/blob/main/docs/guides/training_experts.md) covers your own data and evaluation. In Python, use `compile(examples, augment=False)` and `System1Engine(..., strict_mode=True)` for this workflow.

## Policy guard

This executable example permits one configuration lookup for one application-supplied principal, records the actual lookup result, and verifies the ledger. Other keys or tools have no grant. It persists a local demonstration key across runs.

```python
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from system1 import (
    ActionLedger, ActionProposal, DecisionOutcome, PolicyEngine, PolicyRule,
    RiskLevel, SystemOneGuard, load_private_key, save_keypair,
)

key_path = Path(".system1/demo-identity/identity.key")
if not key_path.exists():
    save_keypair(Ed25519PrivateKey.generate(), key_path.parent)
signing_key = load_private_key(key_path)

policy = PolicyEngine(rules=[PolicyRule(
    rule_id="read_service_name",
    tools=["read_config"],
    allowed_principals=["agent-worker"],
    allowed_tenants=["demo"],
    allowed_scopes=["config:read"],
    argument_limits={"key": ["service_name"]},
    outcome=DecisionOutcome.ALLOW,
    risk=RiskLevel.READ_ONLY,
)])

with ActionLedger(".system1/demo-audit.sqlite", require_durable=True) as ledger:
    guard = SystemOneGuard(
        policy=policy, ledger=ledger, signing_key=signing_key,
        enforcement_profile=True,
    )
    proposal = ActionProposal.create(
        tenant_id="demo", principal_id="agent-worker", scope="config:read",
        tool="read_config", arguments={"key": "service_name"},
        canonical_target="config:service_name", purpose="Inspect service name",
    )
    auth = guard.evaluate_proposal(proposal)
    if auth.outcome != DecisionOutcome.ALLOW:
        raise PermissionError(auth.reason)

    # Execute exactly the authorized operation and arguments.
    result = {"service_name": "system1-demo"}[proposal.arguments["key"]]
    ledger.record_execution_outcome(
        action_id=proposal.action_id,
        receipt_digest=auth.receipt.compute_digest(),
        status="SUCCEEDED", result_payload={"value": result},
        tenant_id=proposal.tenant_id, principal_id=proposal.principal_id,
        scope=proposal.scope, trusted_public_key=signing_key.public_key(),
    )
    assert ledger.verify_integrity(trusted_public_key=signing_key.public_key())
    print(result)
```

The application must derive identity from its authenticated session, constrain tool arguments and targets, and keep agent code from bypassing the guard. Policy correctness is the operator's responsibility. The default guard without `enforcement_profile=True` can use statistical classification to allow actions. See [deployment boundaries](https://github.com/steph4n-gh/system1/blob/main/docs/deployment.md) before granting consequential permissions.

## Integrations

### LangChain

Install `python -m pip install 'system1[langchain]'`. Attach a configured guard to actual tool invocations:

```python
from system1.integrations import SystemOneGuardCallbackHandler

# guard is your configured SystemOneGuard, with a live ledger and signing key.
handler = SystemOneGuardCallbackHandler(
    guard=guard, tenant_id="demo", principal_id="agent-worker",
)
result = your_tool.invoke(tool_arguments, config={"callbacks": [handler]})
```

The callback uses scope `langchain:tools:exec` and the serialized tool input in its proposal. Match policy rules to that contract. `SystemOneGuardBlockedException` propagates to the caller on a deny or approval requirement. `wrap_langchain_tool` also supports guarding Python callables directly.

### MCP

`System1MCPProxy` intercepts JSON-RPC tool calls. Supply your configured guard and authenticated identity, then use `handle_call(request, executor)` or `async_handle_call(request, executor)` to connect it to your application's executor. See [integration tests](https://github.com/steph4n-gh/system1/blob/main/tests/test_integrations.py) for the dispatch contract. Diagnostic mode does not produce trusted signed enforcement evidence by default.

### FastAPI / ASGI

`add_system1_gateway(app, schema=YourSchema)` can respond to confident classification requests locally and pass uncertain requests to the downstream application. Install your ASGI framework separately. Configure authentication and request-size limits **outside** this middleware: local responses bypass downstream endpoint dependencies. This is a classification gateway, not a tool authorization boundary. See [deployment guidance](https://github.com/steph4n-gh/system1/blob/main/docs/deployment.md#asgi-gateway).

### gRPC

```bash
python -m pip install 'system1[grpc]'
system1 serve --grpc --host 127.0.0.1 --port 50051
```

The bundled [protobuf contract](https://github.com/steph4n-gh/system1/blob/main/src/system1/proto/system1.proto) supports clients generated for other languages. The CLI server is a local diagnostic service with insecure gRPC transport. Configure signing, ledger, and policy through the Python `serve(...)` API for a custom deployment. The repository includes experimental [container and Kubernetes templates](https://github.com/steph4n-gh/system1/blob/main/deploy/README.md), but does not supply a published image or an authenticated service.

### Observability

Install `system1[observability]` for Prometheus or `system1[otel]` for OpenTelemetry. See the [observability guide](https://github.com/steph4n-gh/system1/blob/main/docs/observability/README.md) and [Grafana dashboard](https://github.com/steph4n-gh/system1/blob/main/docs/observability/grafana-dashboard.json). Starting a metrics server or configuring an exporter changes the application's network behavior.

## Benchmarks

Check current manual teaching and the recorded 1.0 Jev baseline from the repository root:

```bash
python benchmarks/quality/evaluate_release.py
```

This command uses saved real responses and blocks network access during local evaluation. It exits unsuccessfully if any primary skill misses 95% accepted accuracy or 80% acceptance. These are observed demonstration targets, not population guarantees.

The [public-data workload evaluation](https://github.com/steph4n-gh/system1/blob/main/benchmarks/quality/workloads/README.md)
measures three bounded tasks on **1,301 unseen cases**, with settings and splits
fixed before the new evaluations:

| Explicitly taught skill | Accepted locally | Correct among accepted | Median local decision |
|---|---:|---:|---:|
| Banking support, three intents | 118/120 (98.3%) | 115/118 (97.5%) | 0.378 ms |
| Assistant commands, six intents | 173/180 (96.1%) | 172/173 (99.4%) | 0.344 ms |
| SMS spam triage | 980/1,001 (97.9%) | 957/980 (97.7%) | 0.444 ms |

Teaching plus calibration took 0.23–1.77 seconds after labeled data existed;
saved skills were 20.4–47.0 KiB. These timings exclude data preparation, startup,
receipts and downstream actions. A TF-IDF/logistic-regression baseline matched raw
correctness on banking and assistant, scored lower on SMS, and ran faster locally
on all three. The report retains every error and the unsuccessful banking/SMS
automatic-takeover experiments. Explicit teaching and automatic observation use
different evidence partitions and settings.

The [historical seed and teaching comparisons](https://github.com/steph4n-gh/system1/blob/main/benchmarks/quality/README.md#historical-seed-benchmarks)
remain available. In the recorded 0.2.2 seed run, security-triage and intent-routing
raw accuracy were only 50% and 51%; raw predictions are not safe tool permissions.
Use task-specific teaching and evaluation before relying on a skill.

Conformal coverage applies to prediction sets under exchangeability and appropriate held-out calibration. It does not guarantee that a singleton prediction is safe, control the error rate conditional on local acceptance, or imply a particular local-retention percentage. Distribution shift and model updates require reevaluation. See [limits](https://github.com/steph4n-gh/system1/blob/main/docs/deployment.md#statistical-limits) and the [conformal prediction introduction](https://arxiv.org/abs/2107.07511).

## Examples and research

- [Support triage](https://github.com/steph4n-gh/system1/blob/main/examples/support_triage.py), [model routing](https://github.com/steph4n-gh/system1/blob/main/examples/model_routing.py), and [agent guard](https://github.com/steph4n-gh/system1/blob/main/examples/agent_guard.py) teach and check one skill each. Their [datasets and measured results](https://github.com/steph4n-gh/system1/blob/main/examples/teaching/README.md) are included, along with a [comparison of teaching close distinctions](https://github.com/steph4n-gh/system1/blob/main/benchmarks/quality/contrast_round.md).
- [Teach one skill](https://github.com/steph4n-gh/system1/blob/main/examples/teach_skill.py), [advanced expert examples](https://github.com/steph4n-gh/system1/blob/main/examples/train_expert.py), and [observed teaching and takeover](https://github.com/steph4n-gh/system1/blob/main/examples/observe_routing.py). Cloud modes require explicit configuration.
- [Gaming examples](https://github.com/steph4n-gh/system1/blob/main/examples/gaming/) explore simulated environments and optional local emulation; they are not independently verified world records. ROMs are not included. The [first-use evaluation](https://github.com/steph4n-gh/system1/blob/main/benchmarks/quality/zero_shot/README.md) measures game rules and legal actions separately from raw classifier quality.
- [Whitepaper](https://github.com/steph4n-gh/system1/blob/main/docs/paper/system1_whitepaper.md), [technical brief](https://github.com/steph4n-gh/system1/blob/main/docs/paper/system1_technical_brief.md), and [mathematical notes](https://github.com/steph4n-gh/system1/blob/main/docs/paper/conformal_gating.md) describe the current implementation and evidence. They are maintainer-authored documents, not peer-reviewed publications. The [example catalog](https://github.com/steph4n-gh/system1/blob/main/examples/README.md) separates taught skills from experimental and scripted demonstrations.

Both `import system1` and the legacy `import reflex` expose the same API. The `reflex` namespace can conflict with the separate Reflex web-framework package, so use separate environments when needed. The [TypeSafe adapter](https://github.com/steph4n-gh/system1/blob/main/docs/typesafe.md) supports the basic sync/async decision API, structured state, typed response accessors, and ordinal score distributions. Its documented contract does not include the entire SDK transport/Pydantic surface.

## CLI

```bash
system1 decide "How do I reset my password?" --schema triage --json
system1 decide "Read documentation" --schema guard --sign --ledger audit.sqlite --json
system1 verify-receipt receipt.json --public-key identity.pub
system1 calibrate --dataset data.json --schema triage --bins 10
system1 compile --schema triage --output triage.s1m --json
```

Use `system1 --help` or `system1 <command> --help` for options. Receipt verification requires a trusted public key supplied independently of the receipt.

## Development

```bash
python -m pip install -e '.[dev,langchain]'
python -m pytest tests/ -q
```

Or use the committed lockfile with `uv sync --locked --extra dev --extra langchain` and `uv run --no-sync pytest -q`. CI tests Linux and macOS, checks built distributions, and exercises real LangChain dispatch. Optional MLX and emulator tests require their extras and suitable hardware/assets. See [contributing](https://github.com/steph4n-gh/system1/blob/main/CONTRIBUTING.md) and [security reporting](https://github.com/steph4n-gh/system1/blob/main/SECURITY.md).

## License

[Apache License 2.0](https://github.com/steph4n-gh/system1/blob/main/LICENSE).
