# Observability

Optional Prometheus metrics and OpenTelemetry spans describe decisions and review
signals. They do not measure whether an external teacher or human completed a
review. Starting a metrics server or configuring a remote exporter changes the
application's network behavior.

## Install

```bash
python -m pip install 'system1[observability,otel]'
```

Use just `system1[observability]` or `system1[otel]` if only one is needed. The
base package does not require these dependencies.

## Prometheus

This self-contained example instruments a diagnostic schema; without teaching
and calibration it should request review. Substitute your validated engine for
application use.

```python
from system1 import ChoiceField, DecisionSchema, System1Engine
from system1.integrations.observability import SystemOneMetricsExporter

class Route(DecisionSchema):
    queue = ChoiceField(options=["billing", "support"])

engine = System1Engine(Route, strict_mode=True)
metrics = SystemOneMetricsExporter()
metrics.instrument(engine)
result = engine.decide("Please correct my invoice")
print("Needs review:", result.is_ambiguous)
```

To expose `/metrics`, call `metrics.start_server(port=9090)` and keep your
application running. It uses `prometheus_client.start_http_server`'s default
listen address; restrict network access through your deployment boundary. It
provides no authentication. Example scrape configuration:

```yaml
scrape_configs:
  - job_name: system1
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:9090']
```

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `system1_decisions_total` | Counter | `schema`, `outcome`, `cache_hit` | Instrumented decision calls; outcome is the first returned field's value |
| `system1_decision_latency_seconds` | Histogram | `schema` | The result's reported decision latency, converted from milliseconds |
| `system1_escalations_total` | Counter | `schema`, `reason` | Decisions flagged for review, not completed external escalations |
| `system1_cache_hit_ratio` | Gauge | None | Cumulative hits divided by decisions since exporter creation |
| `system1_conformal_set_size` | Histogram | `schema` | Set sizes recorded across fields |
| `system1_ledger_entries_total` | Gauge | None | Last observed ledger sequence/depth |

Latency buckets are 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1 and 1.0 seconds.
The reason labels `ambiguous`, `low_margin` and `ood` come from an internal
heuristic; `ood` is not an independently calibrated distribution-shift detector.
Do not interpret this histogram as full request, durable authorization or tool
execution latency. Limit schema/outcome cardinality in your application.

For manual recording instead of wrapping `decide()`:

```python
metrics.record_decision(
    result=result,
    schema_name=result.schema_name,
    cache_hit=result.is_cache_hit,
    escalated=result.is_ambiguous,
)
```

Do not also manually record the same call after instrumentation. The wrapper
updates the ledger gauge on a best-effort basis if the engine has a ledger;
otherwise call `metrics.update_ledger_gauge(ledger.audit_head()[0])` yourself.
A ledger-read failure can leave the previous gauge value in place.

## OpenTelemetry

A self-contained console example:

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from system1 import ChoiceField, DecisionSchema, System1Engine
from system1.integrations.otel import SystemOneOTelInstrumentor

class Route(DecisionSchema):
    queue = ChoiceField(options=["billing", "support"])

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
engine = System1Engine(Route, strict_mode=True)
SystemOneOTelInstrumentor(tracer_provider=provider).instrument(engine)
engine.decide("Please correct my invoice")
provider.shutdown()
```

The legacy `reflex.*` telemetry names are retained for compatibility:
`reflex.schema`, `reflex.latency_ms`, `reflex.latency_seconds`, `reflex.outcome`,
`reflex.cache_hit`, `reflex.is_ambiguous` and `reflex.conformal_set_size_max`.
The span name is `reflex.decide.<schema>`. An ambiguity flag represents a review
request, not proof that another service was called.

Remote OTLP export additionally requires a separately installed exporter, for
example `python -m pip install opentelemetry-exporter-otlp-proto-grpc`. That package
is not part of `system1[otel]`. Configure its collector, transport and access
controls in your application. Metrics and OTel can wrap the same engine; attach
each once to avoid duplicate instrumentation.

## Dashboard

Import [grafana-dashboard.json](grafana-dashboard.json) and select your Prometheus
data source. It includes latency quantiles, decisions/second, review-signal rates,
cumulative cache hit ratio, conformal-set buckets and ledger depth. Its legacy
`reflex-decision-engine` UID is retained so an import can update an existing dashboard.

[Implementation](../../src/system1/integrations/observability.py) ·
[Tracing implementation](../../src/system1/integrations/otel.py) ·
[Deployment boundaries](../deployment.md).
