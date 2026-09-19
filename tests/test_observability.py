"""Tests for System 1 Prometheus & OpenTelemetry observability integrations.

Uses mocking to avoid hard dependencies on prometheus_client / opentelemetry
packages in CI. When the real packages *are* installed, the tests exercise the
concrete classes directly and verify actual metric values via ``generate_latest``.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Lightweight stubs for DecisionResult (avoids importing the full engine
# which pulls in numpy/cryptography and the rest of the runtime).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _StubReceipt:
    decision_id: str = "stub-id"
    schema_name: str = "stub"
    schema_digest: str = "digest"
    prompt: str = ""
    prompt_digest: str = ""
    values: Dict[str, Any] = field(default_factory=dict)
    confidences: Dict[str, float] = field(default_factory=dict)
    conformal_sets: Dict[str, List[str]] = field(default_factory=dict)
    probabilities: Dict[str, Dict[str, float]] = field(default_factory=dict)
    latency_ms: float = 0.0
    is_ambiguous: bool = False
    timestamp: str = ""
    truth_ledger_head: str = ""
    ledger_record_id: Optional[str] = None
    signer_public_key: Optional[str] = None
    envelope: Optional[str] = None


@dataclass(frozen=True)
class _StubDecisionResult:
    """Minimal stand-in for ``DecisionResult`` that satisfies the metrics API."""
    schema_name: str = "TestSchema"
    schema_digest: str = "abc123"
    prompt: str = "test prompt"
    values: Dict[str, Any] = field(default_factory=lambda: {"action": "execute"})
    confidences: Dict[str, float] = field(default_factory=lambda: {"action": 0.95})
    conformal_sets: Dict[str, List[str]] = field(default_factory=lambda: {"action": ["execute"]})
    probabilities: Dict[str, Dict[str, float]] = field(default_factory=dict)
    is_ambiguous: bool = False
    latency_ms: float = 0.42
    alpha: float = 0.05
    receipt: Any = field(default_factory=_StubReceipt)
    margins: Dict[str, float] = field(default_factory=dict)
    margin_thresholds: Dict[str, float] = field(default_factory=dict)
    margin_gate_active: Dict[str, bool] = field(default_factory=dict)
    odds_ratios: Dict[str, float] = field(default_factory=dict)
    relative_odds_ratio_thresholds: Dict[str, float] = field(default_factory=dict)
    confidence_floors: Dict[str, float] = field(default_factory=dict)
    ambiguous_fields: List[str] = field(default_factory=list)
    escalated_fields: List[str] = field(default_factory=list)
    telemetry: Optional[Any] = None
    is_cache_hit: bool = False
    embedding: Optional[Any] = None


# ---------------------------------------------------------------------------
# Detect whether real prometheus_client is available
# ---------------------------------------------------------------------------
_HAS_PROMETHEUS = False
try:
    import prometheus_client as _pc
    _HAS_PROMETHEUS = True
except ImportError:
    pass


# ========================================================================
# Prometheus / Observability Tests
# ========================================================================

class TestSystemOneMetricsExporter:
    """Tests for :class:`SystemOneMetricsExporter`."""

    # Helper — import with real or mocked prometheus_client
    @staticmethod
    def _import_exporter():
        """Import SystemOneMetricsExporter, works whether prometheus_client is
        installed or not (installs a lightweight mock shim if missing)."""
        try:
            import prometheus_client  # noqa: F401
        except ImportError:
            # Build a minimal shim so the module can load
            shim = types.ModuleType("prometheus_client")

            class _FakeMetric:
                def __init__(self, *a, **kw):
                    self._labelnames = kw.get("labelnames", [])
                    self._value = 0.0
                    self._children: Dict[tuple, "_FakeMetric"] = {}

                def labels(self, **kw):
                    key = tuple(sorted(kw.items()))
                    if key not in self._children:
                        self._children[key] = _FakeMetric()
                    return self._children[key]

                def inc(self, amount=1):
                    self._value += amount

                def observe(self, value):
                    # Accumulate count like a real histogram
                    self._value += 1

                def set(self, value):
                    self._value = value

            class _Registry:
                pass

            shim.Counter = _FakeMetric
            shim.Gauge = _FakeMetric
            shim.Histogram = _FakeMetric
            shim.CollectorRegistry = _Registry
            shim.start_http_server = lambda port, registry=None: None
            sys.modules["prometheus_client"] = shim

        from system1.integrations.observability import SystemOneMetricsExporter
        return SystemOneMetricsExporter

    @staticmethod
    def _get_metric_value(exporter, metric_name, labels=None):
        """Get the actual metric value from the exporter's registry.

        When prometheus_client is installed, uses generate_latest to parse
        the /metrics output. Falls back to reading _value on mock objects.
        """
        if _HAS_PROMETHEUS:
            from prometheus_client import generate_latest
            output = generate_latest(exporter.registry).decode()
            # Parse the specific metric line
            for line in output.split('\n'):
                if line.startswith('#'):
                    continue
                # Match by metric name (counters get _total suffix, etc.)
                if metric_name in line:
                    if labels:
                        # Check all labels are present
                        all_match = all(
                            f'{k}="{v}"' in line
                            for k, v in labels.items()
                        )
                        if all_match:
                            # Extract value: last token
                            return float(line.strip().split()[-1])
                    elif '{' not in line:
                        # No-label metric, exact name match
                        parts = line.strip().split()
                        if parts[0] == metric_name:
                            return float(parts[-1])
            return None
        else:
            # Mock fallback: navigate the fake metric
            metric = getattr(exporter, {
                'system1_decisions_total': 'decisions_total',
                'system1_escalations_total': 'escalations_total',
                'system1_decision_latency_seconds': 'decision_latency_seconds',
                'system1_conformal_set_size': 'conformal_set_size',
                'system1_cache_hit_ratio': 'cache_hit_ratio',
                'system1_ledger_entries_total': 'ledger_entries_total',
            }.get(metric_name, metric_name))
            if labels:
                return metric.labels(**labels)._value
            return metric._value

    def test_instantiation(self):
        Cls = self._import_exporter()
        exporter = Cls()
        assert exporter is not None
        assert hasattr(exporter, "decisions_total")
        assert hasattr(exporter, "decision_latency_seconds")
        assert hasattr(exporter, "escalations_total")
        assert hasattr(exporter, "cache_hit_ratio")
        assert hasattr(exporter, "conformal_set_size")
        assert hasattr(exporter, "ledger_entries_total")

    def test_record_decision_basic(self):
        Cls = self._import_exporter()
        exporter = Cls()
        result = _StubDecisionResult()
        # Should not raise
        exporter.record_decision(
            result=result,
            schema_name="TestSchema",
            cache_hit=False,
            escalated=False,
        )

    def test_record_decision_increments_counter(self):
        """Verify the decisions_total Counter is actually incremented."""
        Cls = self._import_exporter()
        exporter = Cls()
        result = _StubDecisionResult()
        exporter.record_decision(
            result=result,
            schema_name="TestSchema",
            cache_hit=False,
            escalated=False,
        )
        val = self._get_metric_value(
            exporter,
            'system1_decisions_total',
            labels={'schema': 'TestSchema', 'outcome': 'execute', 'cache_hit': 'false'},
        )
        assert val == 1.0

        # Record a second decision
        exporter.record_decision(
            result=result,
            schema_name="TestSchema",
            cache_hit=False,
            escalated=False,
        )
        val2 = self._get_metric_value(
            exporter,
            'system1_decisions_total',
            labels={'schema': 'TestSchema', 'outcome': 'execute', 'cache_hit': 'false'},
        )
        assert val2 == 2.0

    def test_record_decision_updates_cache_ratio(self):
        Cls = self._import_exporter()
        exporter = Cls()

        # Record 2 misses and 1 hit → ratio should be 1/3
        for _ in range(2):
            exporter.record_decision(
                result=_StubDecisionResult(),
                schema_name="S",
                cache_hit=False,
                escalated=False,
            )

        exporter.record_decision(
            result=_StubDecisionResult(is_cache_hit=True),
            schema_name="S",
            cache_hit=True,
            escalated=False,
        )

        assert exporter._total_decisions == 3
        assert exporter._cache_hits == 1
        # Verify the Gauge value itself
        expected_ratio = 1.0 / 3.0
        gauge_val = self._get_metric_value(exporter, 'system1_cache_hit_ratio')
        assert gauge_val is not None
        assert abs(gauge_val - expected_ratio) < 1e-9

    def test_record_escalation_ambiguous(self):
        Cls = self._import_exporter()
        exporter = Cls()
        result = _StubDecisionResult(
            is_ambiguous=True,
            ambiguous_fields=["action"],
            escalated_fields=["action"],
        )
        exporter.record_decision(
            result=result,
            schema_name="TestSchema",
            cache_hit=False,
            escalated=True,
        )
        # Check escalations_total was incremented with reason=ambiguous
        val = self._get_metric_value(
            exporter,
            'system1_escalations_total',
            labels={'schema': 'TestSchema', 'reason': 'ambiguous'},
        )
        assert val == 1.0

    def test_record_escalation_low_margin(self):
        Cls = self._import_exporter()
        exporter = Cls()
        result = _StubDecisionResult(
            is_ambiguous=True,
            margin_gate_active={"action": True},
        )
        exporter.record_decision(
            result=result,
            schema_name="TestSchema",
            cache_hit=False,
            escalated=True,
        )
        # Check escalations_total was incremented with reason=low_margin
        val = self._get_metric_value(
            exporter,
            'system1_escalations_total',
            labels={'schema': 'TestSchema', 'reason': 'low_margin'},
        )
        assert val == 1.0

    def test_record_decision_with_conformal_sets(self):
        Cls = self._import_exporter()
        exporter = Cls()
        result = _StubDecisionResult(
            conformal_sets={"action": ["execute", "escalate"], "safety": ["safe"]},
        )
        # Should observe set sizes 2 and 1
        exporter.record_decision(
            result=result,
            schema_name="TestSchema",
            cache_hit=False,
            escalated=False,
        )
        assert exporter._total_decisions == 1
        # If real prometheus, verify histogram count
        if _HAS_PROMETHEUS:
            val = self._get_metric_value(
                exporter,
                'system1_conformal_set_size_count',
                labels={'schema': 'TestSchema'},
            )
            # Two conformal sets were observed (sizes 2 and 1)
            assert val == 2.0

    def test_ledger_gauge_update(self):
        Cls = self._import_exporter()
        exporter = Cls()
        exporter.update_ledger_gauge(42)
        val = self._get_metric_value(exporter, 'system1_ledger_entries_total')
        assert val == 42.0

    def test_record_decision_empty_values(self):
        """Edge case: result with no values should still record without error."""
        Cls = self._import_exporter()
        exporter = Cls()
        result = _StubDecisionResult(values={}, conformal_sets={})
        exporter.record_decision(
            result=result,
            schema_name="EmptySchema",
            cache_hit=False,
            escalated=False,
        )
        assert exporter._total_decisions == 1
        # Outcome should be empty string
        val = self._get_metric_value(
            exporter,
            'system1_decisions_total',
            labels={'schema': 'EmptySchema', 'outcome': '', 'cache_hit': 'false'},
        )
        assert val == 1.0

    def test_multiple_schemas(self):
        """Metrics should track different schemas independently via labels."""
        Cls = self._import_exporter()
        exporter = Cls()
        for name in ("SchemaA", "SchemaB", "SchemaC"):
            exporter.record_decision(
                result=_StubDecisionResult(schema_name=name),
                schema_name=name,
                cache_hit=False,
                escalated=False,
            )
        assert exporter._total_decisions == 3
        # Each schema should have exactly 1 decision
        for name in ("SchemaA", "SchemaB", "SchemaC"):
            val = self._get_metric_value(
                exporter,
                'system1_decisions_total',
                labels={'schema': name, 'outcome': 'execute', 'cache_hit': 'false'},
            )
            assert val == 1.0

    def test_classify_escalation_reason(self):
        Cls = self._import_exporter()
        exporter = Cls()

        # ambiguous with no margin gate
        r1 = _StubDecisionResult(is_ambiguous=True, margin_gate_active={})
        assert exporter._classify_escalation_reason(r1) == "ambiguous"

        # low margin
        r2 = _StubDecisionResult(is_ambiguous=True, margin_gate_active={"x": True})
        assert exporter._classify_escalation_reason(r2) == "low_margin"

        # ood (not ambiguous)
        r3 = _StubDecisionResult(is_ambiguous=False)
        assert exporter._classify_escalation_reason(r3) == "ood"

    def test_instrument_wraps_decide(self):
        """Test that instrument() monkey-patches engine.decide to auto-record metrics."""
        Cls = self._import_exporter()
        exporter = Cls()

        class _FakeSchema:
            schema_name = "FakeSchema"

        class _FakeEngine:
            schema = _FakeSchema()
            ledger = None

            def decide(self, prompt, **kw):
                return _StubDecisionResult(prompt=prompt, schema_name="FakeSchema")

        engine = _FakeEngine()
        original_decide = engine.decide
        exporter.instrument(engine)

        # decide should now be wrapped
        assert engine.decide is not original_decide

        # Calling decide should still return a result AND record metrics
        result = engine.decide("hello")
        assert result.prompt == "hello"
        assert exporter._total_decisions == 1

        # Verify the counter was incremented
        val = self._get_metric_value(
            exporter,
            'system1_decisions_total',
            labels={'schema': 'FakeSchema', 'outcome': 'execute', 'cache_hit': 'false'},
        )
        assert val == 1.0

    def test_latency_histogram_observation(self):
        """Verify the latency histogram records the correct value."""
        Cls = self._import_exporter()
        exporter = Cls()
        result = _StubDecisionResult(latency_ms=5.0)  # 5ms = 0.005s
        exporter.record_decision(
            result=result,
            schema_name="LatencyTest",
            cache_hit=False,
            escalated=False,
        )
        if _HAS_PROMETHEUS:
            # The 0.005s bucket should have count 1
            val = self._get_metric_value(
                exporter,
                'system1_decision_latency_seconds_bucket',
                labels={'schema': 'LatencyTest', 'le': '0.005'},
            )
            assert val == 1.0
            # The 0.001s bucket should have count 0 (5ms > 1ms)
            val2 = self._get_metric_value(
                exporter,
                'system1_decision_latency_seconds_bucket',
                labels={'schema': 'LatencyTest', 'le': '0.001'},
            )
            assert val2 == 0.0
            # Sum should be 0.005
            sum_val = self._get_metric_value(
                exporter,
                'system1_decision_latency_seconds_sum',
                labels={'schema': 'LatencyTest'},
            )
            assert abs(sum_val - 0.005) < 1e-9


# ========================================================================
# OpenTelemetry Tests
# ========================================================================

class TestSystemOneOTelInstrumentor:
    """Tests for :class:`SystemOneOTelInstrumentor`."""

    @staticmethod
    def _import_instrumentor():
        """Import SystemOneOTelInstrumentor, with a mock shim if opentelemetry is
        not installed."""
        try:
            import opentelemetry  # noqa: F401
        except ImportError:
            # Build a minimal opentelemetry shim
            otel = types.ModuleType("opentelemetry")
            otel_trace = types.ModuleType("opentelemetry.trace")

            class _FakeSpan:
                def set_attribute(self, key, value):
                    pass

                def set_status(self, status):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    pass

            class _FakeTracer:
                def start_as_current_span(self, name, kind=None):
                    return _FakeSpan()

            class _FakeStatus:
                def __init__(self, code, description=None):
                    pass

            class _FakeStatusCode:
                OK = 0
                ERROR = 1

            class _SpanKind:
                INTERNAL = 0

            def get_tracer(name, tracer_provider=None):
                return _FakeTracer()

            otel_trace.get_tracer = get_tracer
            otel_trace.Tracer = _FakeTracer
            otel_trace.TracerProvider = type(None)
            otel_trace.SpanKind = _SpanKind
            otel_trace.Status = _FakeStatus
            otel_trace.StatusCode = _FakeStatusCode
            otel.trace = otel_trace

            sys.modules["opentelemetry"] = otel
            sys.modules["opentelemetry.trace"] = otel_trace

        from system1.integrations.otel import SystemOneOTelInstrumentor
        return SystemOneOTelInstrumentor

    def test_instantiation(self):
        Cls = self._import_instrumentor()
        instrumentor = Cls()
        assert instrumentor is not None
        assert hasattr(instrumentor, "tracer")

    def test_instrument_wraps_decide(self):
        Cls = self._import_instrumentor()
        instrumentor = Cls()

        # Build a fake engine-like object
        class _FakeSchema:
            schema_name = "FakeSchema"

        class _FakeEngine:
            schema = _FakeSchema()

            def decide(self, prompt, **kw):
                return _StubDecisionResult(prompt=prompt)

        engine = _FakeEngine()
        original_decide = engine.decide
        instrumentor.instrument(engine)

        # decide should now be wrapped
        assert engine.decide is not original_decide

        # Calling decide should still return a result
        result = engine.decide("hello")
        assert result.prompt == "hello"
        assert result.schema_name == "TestSchema"

    def test_instrument_with_custom_tracer_name(self):
        Cls = self._import_instrumentor()
        instrumentor = Cls(tracer_name="custom.reflex")
        assert instrumentor is not None

    def test_span_attributes_recorded(self):
        """Verify that span attributes are recorded during instrumented decide()."""
        Cls = self._import_instrumentor()

        _HAS_OTEL_SDK = False
        try:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
            _HAS_OTEL_SDK = True
        except (ImportError, AttributeError):
            pass

        if _HAS_OTEL_SDK:

            class _MemoryExporter(SpanExporter):
                """Minimal in-memory span exporter compatible across SDK versions."""
                def __init__(self):
                    self._spans = []
                def export(self, spans):
                    self._spans.extend(spans)
                    return SpanExportResult.SUCCESS
                def shutdown(self):
                    pass
                def get_finished_spans(self):
                    return list(self._spans)

            memory_exporter = _MemoryExporter()
            provider = TracerProvider()
            provider.add_span_processor(SimpleSpanProcessor(memory_exporter))

            instrumentor = Cls(tracer_provider=provider)
        else:
            instrumentor = Cls()

        class _FakeSchema:
            schema_name = "SpanTestSchema"

        class _FakeEngine:
            schema = _FakeSchema()

            def decide(self, prompt, **kw):
                return _StubDecisionResult(
                    prompt=prompt,
                    schema_name="SpanTestSchema",
                    latency_ms=1.5,
                    is_cache_hit=True,
                )

        engine = _FakeEngine()
        instrumentor.instrument(engine)
        engine.decide("test prompt")

        if _HAS_OTEL_SDK:
            spans = memory_exporter.get_finished_spans()
            assert len(spans) == 1
            span = spans[0]
            attrs = dict(span.attributes)
            assert attrs["reflex.schema"] == "SpanTestSchema"
            assert attrs["reflex.latency_ms"] == 1.5
            assert abs(attrs["reflex.latency_seconds"] - 0.0015) < 1e-9
            assert attrs["reflex.cache_hit"] is True
            assert attrs["reflex.outcome"] == "execute"
