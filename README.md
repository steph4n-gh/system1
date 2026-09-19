<p align="center">
  <img src="https://raw.githubusercontent.com/steph4n-gh/system1/main/assets/system1-logo.jpg" alt="System 1 logo" width="128" />
</p>

# System 1: Local Decision Runtime for AI Agents

Teach a repeatable decision skill from examples or by observing a teacher such as Jev. Validate it, take over locally, and save the skill for reuse.

[![CI](https://github.com/steph4n-gh/system1/actions/workflows/ci.yml/badge.svg)](https://github.com/steph4n-gh/system1/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/system1.svg)](https://pypi.org/project/system1/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

**Status: beta.** System 1 provides a small local classifier and a separate deterministic policy guard. The included seed models need evaluation and calibration on your workload. They are not a general-purpose detector of malicious actions or prompt injection.

[Quickstart](#quickstart) · [Policy guard](#policy-guard) · [Integrations](#integrations) · [Benchmarks](#benchmarks) · [Limits and deployment](docs/deployment.md) · [Contributing](CONTRIBUTING.md)

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

Define the outputs you need, then evaluate a prompt:

```python
from system1 import ChoiceField, DecisionSchema, System1Engine

class SupportRoute(DecisionSchema):
    team = ChoiceField(
        options=["billing", "technical_support"],
        descriptions={
            "billing": "Invoices, payments, subscription plans, and refunds",
            "technical_support": "Software installation, errors, and debugging",
        },
    )

engine = System1Engine(SupportRoute)
decision = engine.decide("I need a refund for my subscription")
print(decision.values)
print(f"Latency: {decision.latency_ms:.2f} ms")
print(f"Needs review: {decision.is_ambiguous}")
print(f"Prediction sets: {decision.conformal_sets}")
```

This starts with schema-derived seed prototypes. Outputs and latency depend on the schema, data, and hardware. A predicted value is not permission to execute a tool.

## Observe a teacher, then take over

```bash
python examples/observe_routing.py
```

This short offline example shows the complete journey: observe answers, teach one routing skill, pass the normal promotion gates, disconnect the teacher, evaluate fresh requests locally, then save and reopen the skill. In the recorded run it needed **190 observations**, answered **40/40 fresh synthetic tickets correctly**, accepted all 40, and saved a **5 KB** skill. Reloading preserved answers, probabilities, and review decisions. Local median latency was about **0.45 ms** on the review machine.

The default teacher is a rule over a small structured-ticket vocabulary, clearly labeled as a simulation. It demonstrates the lifecycle, not general language understanding or Jev quality parity. `--teacher jev` observes actual Jev responses with your API key; another API can use the existing teacher callback. See the [adapter guide and compatibility contract](docs/typesafe.md) and [recorded evidence](examples/teaching/results/observed_routing.json).

Promotion depends on evidence, not a fixed turn count. Insufficient evidence keeps the teacher active, and uncertain local responses still request review.

## Teach a skill

Give System 1 labeled examples of one task, save the resulting `.s1m` skill, and reuse it locally. A person or System 2 can supply the examples; teaching does not require an LLM. The existing compiler fits a small decision head using NumPy.

```bash
python examples/support_triage.py
python examples/model_routing.py
python examples/agent_guard.py
system1 decide "Please correct the invoice address" --model .system1/examples/support_triage/skill.s1m --json
```

Each primary example supplies labeled teaching cases, separate calibration cases, and 24 unseen evaluation cases. One command teaches, saves, reloads, and reports the results. On the review machine, teaching plus calibration took 127–159 ms, saved skills were 25–32 KiB, and uncached decisions took about 0.5 ms. Preparing good labeled examples takes additional work.

The authored demonstration cases reached 87.5% support-routing accuracy, 91.7% model-routing accuracy, and 100% operation-triage accuracy. Strict uncertainty checks requested review on 23/24 support cases, all 24 routing cases, and 22/24 operation cases. The three accepted responses were correct. See the [complete results, data, and limits](examples/teaching/README.md); these are demonstrations, not production quality guarantees.

For the smallest API example, see [teach one skill](examples/teach_skill.py). The [teaching guide](docs/guides/training_experts.md) covers your own data and evaluation. In Python, use `compile(examples, augment=False)` and `System1Engine(..., strict_mode=True)` for this workflow.

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

The application must derive identity from its authenticated session, constrain tool arguments and targets, and keep agent code from bypassing the guard. Policy correctness is the operator's responsibility. The default guard without `enforcement_profile=True` can use statistical classification to allow actions. See [deployment boundaries](docs/deployment.md) before granting consequential permissions.

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

`System1MCPProxy` intercepts JSON-RPC tool calls. Supply your configured guard and authenticated identity, then use `handle_call(request, executor)` or `async_handle_call(request, executor)` to connect it to your application's executor. See [integration tests](tests/test_integrations.py) for the dispatch contract. Diagnostic mode does not produce trusted signed enforcement evidence by default.

### FastAPI / ASGI

`add_system1_gateway(app, schema=YourSchema)` can respond to confident classification requests locally and pass uncertain requests to the downstream application. Install your ASGI framework separately. Configure authentication and request-size limits **outside** this middleware: local responses bypass downstream endpoint dependencies. This is a classification gateway, not a tool authorization boundary. See [deployment guidance](docs/deployment.md#asgi-gateway).

### gRPC

```bash
python -m pip install 'system1[grpc]'
system1 serve --grpc --host 127.0.0.1 --port 50051
```

The bundled [protobuf contract](src/system1/proto/system1.proto) supports clients generated for other languages. The CLI server is a local diagnostic service with insecure gRPC transport. Configure signing, ledger, and policy through the Python `serve(...)` API for a custom deployment. The repository does not supply a published Docker image or Kubernetes manifests.

### Observability

Install `system1[observability]` for Prometheus or `system1[otel]` for OpenTelemetry. See the [observability guide](docs/observability/README.md) and [Grafana dashboard](docs/observability/grafana-dashboard.json). Starting a metrics server or configuring an exporter changes the application's network behavior.

## Benchmarks

Reproduce the included seed-model benchmarks from the repository root:

```bash
python benchmarks/quality/run_quality_benchmarks.py
system1 bench --schema triage --iterations 200 --json
```

The 100-example datasets are small, repository-authored evaluations, not independent security certifications. The launch-review run measured:

| Task | Overall quality | Additional metric | Median latency |
|---|---|---|---|
| Security triage | 50% accuracy; 0.486 macro-F1 | BLOCK recall: 0.371 | 0.410 ms |
| Intent routing | 51% accuracy; 0.495 macro-F1 | Billing F1: 0.653 | 0.488 ms |
| Threat scoring, 0–10 | MAE: 3.068; RMSE: 3.480 | Pearson r: 0.309 | 0.469 ms |

See the [recorded results and environment](benchmarks/quality/results/launch_review.json) and [methodology](benchmarks/quality/README.md). These are raw classification results, not the accuracy of authorized tool actions. Latency is workload- and hardware-dependent; these measurements do not establish durable end-to-end authorization latency or a service-level guarantee. No cloud providers were measured in this run.

Teaching from the existing examples improves raw accuracy in a separate five-fold development check: intent routing reaches 68% and security triage 71% with 2048 features. Both still require review on every case under strict uncertainty gating; their calibration sets are too small. This is evidence that examples help, not release acceptance evidence. See the [teaching comparison](benchmarks/quality/README.md#teaching-comparison) for default-dimension results, protocol, and limitations.

Conformal coverage applies to prediction sets under exchangeability and appropriate held-out calibration. It does not guarantee that a singleton prediction is safe, control the error rate conditional on local acceptance, or imply a particular local-retention percentage. Distribution shift and model updates require reevaluation. See [limits](docs/deployment.md#statistical-limits) and the [conformal prediction introduction](https://arxiv.org/abs/2107.07511).

## Examples and research

- [Support triage](examples/support_triage.py), [model routing](examples/model_routing.py), and [agent guard](examples/agent_guard.py) teach and check one skill each. Their [datasets and measured results](examples/teaching/README.md) are included.
- [Teach one skill](examples/teach_skill.py), [advanced expert examples](examples/train_expert.py), and [observed teaching and takeover](examples/observe_routing.py). Cloud modes require explicit configuration.
- [Gaming examples](examples/gaming/) explore simulated environments and optional local emulation; they are not independently verified world records. ROMs are not included.
- [Research manuscripts](docs/paper/) describe the design and earlier experiments. Their historical timing and quality claims are not release acceptance criteria; use the reproducible measurements above.

Both `import system1` and the legacy `import reflex` expose the same API. The `reflex` namespace can conflict with the separate Reflex web-framework package, so use separate environments when needed. The [TypeSafe adapter](docs/typesafe.md) supports the basic sync/async decision API, structured state, typed response accessors, and ordinal score distributions. Its documented contract does not include the entire SDK transport/Pydantic surface.

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

Or use the committed lockfile with `uv sync --locked --extra dev --extra langchain` and `uv run --no-sync pytest -q`. CI tests Linux and macOS, checks built distributions, and exercises real LangChain dispatch. Optional MLX and emulator tests require their extras and suitable hardware/assets. See [contributing](CONTRIBUTING.md) and [security reporting](SECURITY.md).

## License

[Apache License 2.0](LICENSE).
