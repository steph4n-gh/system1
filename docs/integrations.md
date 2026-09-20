# Connect a checked skill to your application

Start with a [taught, checked skill](guides/first_skill.md) and honor its review
requests. The integrations below connect local decisions or explicit permissions
to an existing application. For tool execution, first configure the
[policy guard](guides/policy_guard.md).

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

### Inbox Zero email categories

The [Inbox Zero pilot](../examples/inbox_zero/README.md) supplies JSON teaching
files, an authenticated local endpoint, and a provider patch for its seven
standard email categories. Install the optional server with `system1[http]`.
The current candidate uses portable TF-IDF features and the existing ridge head,
with no new runtime dependency. It is roughly ten times faster than the initial
hashed email skill. The fresh authored cohort meets point quality targets, while
the upstream regression cohort still misses acceptance; independent evidence for
all seven categories is also missing. The separate
[public spam/ham experiment](../benchmarks/quality/public_email/README.md) offers a
real-email teaching recipe without credentials, but has a different label contract.
The Inbox Zero provider therefore starts in review-only mode and leaves
uncertain or failed classifications unprocessed without an LLM fallback.
Other Inbox Zero LLM features remain outside this classification integration.
