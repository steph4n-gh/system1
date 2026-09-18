# Reflex Observability Guide

Production monitoring for the Reflex decision engine via **Prometheus** metrics and **OpenTelemetry** distributed tracing.

## Installation

```bash
# Prometheus metrics endpoint
pip install system1[observability]

# OpenTelemetry tracing
pip install system1[otel]

# Both
pip install system1[observability,otel]
```

---

## Prometheus Metrics

### Quick Start

```python
from system1.engine import ReflexEngine
from system1.integrations.observability import ReflexMetricsExporter

# 1. Create your engine
engine = ReflexEngine(MySchema)

# 2. Attach metrics exporter (wraps engine.decide automatically)
metrics = ReflexMetricsExporter()
metrics.instrument(engine)

# 3. Start the /metrics HTTP server (default port 9090)
metrics.start_server(port=9090)

# 4. Use the engine as normal — all decisions are tracked
result = engine.decide("Is this action safe?")
```

### Available Metrics

| Metric | Type | Labels | Description |
|---|---|---|---|
| `reflex_decisions_total` | Counter | `schema`, `outcome`, `cache_hit` | Total decisions evaluated |
| `reflex_decision_latency_seconds` | Histogram | `schema` | Evaluation latency (sub-ms buckets) |
| `reflex_escalations_total` | Counter | `schema`, `reason` | Escalated decisions count |
| `reflex_cache_hit_ratio` | Gauge | — | Rolling cache hit ratio (0–1) |
| `reflex_conformal_set_size` | Histogram | `schema` | Conformal prediction set sizes |
| `reflex_ledger_entries_total` | Gauge | — | Current ActionLedger depth |

### Histogram Buckets

The latency histogram uses sub-millisecond buckets tuned for Reflex's performance profile:

```
0.0001s, 0.0005s, 0.001s, 0.005s, 0.01s, 0.05s, 0.1s, 1.0s
```

### Prometheus Scrape Config

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'reflex'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:9090']
```

### Manual Recording

If you prefer not to use `instrument()`, record decisions manually:

```python
metrics = ReflexMetricsExporter()
result = engine.decide("route this query")

metrics.record_decision(
    result=result,
    schema_name="RoutingSchema",
    cache_hit=result.is_cache_hit,
    escalated=result.is_ambiguous,
)
```

### Ledger Gauge

The ledger entries gauge is updated automatically when using `instrument()`, or manually:

```python
metrics.update_ledger_gauge(ledger.audit_head()[0])
```

---

## OpenTelemetry Tracing

### Quick Start

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

from system1.engine import ReflexEngine
from system1.integrations.otel import ReflexOTelInstrumentor

# 1. Configure OTel (example: console exporter)
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

# 2. Instrument the engine
engine = ReflexEngine(MySchema)
instrumentor = ReflexOTelInstrumentor(tracer_provider=provider)
instrumentor.instrument(engine)

# 3. Decisions now emit spans automatically
result = engine.decide("Classify this input")
```

### Span Attributes

Each decision span includes:

| Attribute | Type | Description |
|---|---|---|
| `reflex.schema` | string | Schema name |
| `reflex.latency_ms` | float | Decision latency in milliseconds |
| `reflex.latency_seconds` | float | Decision latency in seconds |
| `reflex.outcome` | string | Top-level choice value |
| `reflex.cache_hit` | bool | Whether the result was a cache hit |
| `reflex.is_ambiguous` | bool | Whether the decision was escalated |
| `reflex.conformal_set_size_max` | int | Largest conformal prediction set |

### Exporting to Jaeger / OTLP

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(
    SimpleSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317"))
)
```

---

## Combining Both

Both instrumentors can wrap the same engine. They compose cleanly:

```python
metrics = ReflexMetricsExporter()
otel = ReflexOTelInstrumentor(tracer_provider=provider)

# Order doesn't matter
metrics.instrument(engine)
otel.instrument(engine)

metrics.start_server(port=9090)
```

---

## Grafana Dashboard

Import the included dashboard template:

1. Open Grafana → Dashboards → Import
2. Upload `grafana-dashboard.json` from this directory
3. Select your Prometheus data source

The dashboard includes panels for:
- **Decision latency histogram** — P50/P90/P99 latency over time
- **Decisions per second** — Rate of evaluated decisions
- **Escalation rate** — Escalations by reason (ambiguous, low_margin, ood)
- **Cache hit ratio** — Rolling gauge
- **Conformal set size distribution** — Histogram quantiles

---

## Architecture

```
┌──────────────────┐     ┌─────────────────────┐
│  ReflexEngine    │────▶│ ReflexMetricsExporter│────▶ /metrics :9090
│  .decide()       │     │  (Prometheus)        │       ↓
│                  │     └─────────────────────┘   Prometheus
│                  │     ┌─────────────────────┐       ↓
│                  │────▶│ ReflexOTelInstrumentor│    Grafana
│                  │     │  (OpenTelemetry)     │────▶ Jaeger / OTLP
└──────────────────┘     └─────────────────────┘
```
