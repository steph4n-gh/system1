"""System 1 OpenTelemetry Tracing Integration.

Creates OpenTelemetry spans for System 1 decision evaluations, recording
latency, outcome, schema name, and cache-hit status as span attributes.

Requires the optional ``opentelemetry-api`` and ``opentelemetry-sdk`` packages::

    pip install system1[otel]
"""

from __future__ import annotations

import functools
from typing import Any, Optional, TYPE_CHECKING

from opentelemetry import trace

if TYPE_CHECKING:
    from system1.engine import DecisionResult, SystemOneEngine


_TRACER_NAME = "reflex.decision_engine"


class SystemOneOTelInstrumentor:
    """OpenTelemetry instrumentor for :class:`SystemOneEngine`.

    Usage::

        from system1.engine import SystemOneEngine
        from system1.integrations.otel import SystemOneOTelInstrumentor

        engine = SystemOneEngine(MySchema)
        instrumentor = SystemOneOTelInstrumentor()
        instrumentor.instrument(engine)
    """

    def __init__(
        self,
        *,
        tracer_provider: Optional[trace.TracerProvider] = None,
        tracer_name: str = _TRACER_NAME,
    ) -> None:
        if tracer_provider is not None:
            self._tracer = trace.get_tracer(tracer_name, tracer_provider=tracer_provider)
        else:
            self._tracer = trace.get_tracer(tracer_name)

    # ------------------------------------------------------------------
    # Instrumentation
    # ------------------------------------------------------------------

    def instrument(self, engine: "SystemOneEngine") -> "SystemOneEngine":
        """Wrap *engine.decide* to emit an OpenTelemetry span per decision.

        Returns the same engine instance (mutated) for chaining convenience.
        """
        original_decide = engine.decide
        tracer = self._tracer

        @functools.wraps(original_decide)
        def _traced_decide(*args: Any, **kwargs: Any) -> "DecisionResult":
            schema_name = engine.schema.schema_name
            with tracer.start_as_current_span(
                f"reflex.decide.{schema_name}",
                kind=trace.SpanKind.INTERNAL,
            ) as span:
                span.set_attribute("reflex.schema", schema_name)

                result = original_decide(*args, **kwargs)

                # Record result attributes
                span.set_attribute(
                    "reflex.latency_ms", result.latency_ms
                )
                span.set_attribute(
                    "reflex.latency_seconds", result.latency_ms / 1000.0
                )
                span.set_attribute("reflex.cache_hit", result.is_cache_hit)
                span.set_attribute("reflex.is_ambiguous", result.is_ambiguous)

                # Outcome — first value in the decision result
                if result.values:
                    first_value = next(iter(result.values.values()), "")
                    span.set_attribute("reflex.outcome", str(first_value))

                # Conformal set sizes
                if result.conformal_sets:
                    max_set_size = max(
                        len(v) for v in result.conformal_sets.values()
                    )
                    span.set_attribute(
                        "reflex.conformal_set_size_max", max_set_size
                    )

                if result.is_ambiguous:
                    span.set_status(
                        trace.Status(
                            trace.StatusCode.OK,
                            description="Decision escalated (ambiguous)",
                        )
                    )
                else:
                    span.set_status(trace.Status(trace.StatusCode.OK))

                return result

        engine.decide = _traced_decide  # type: ignore[method-assign]
        return engine

    @property
    def tracer(self) -> trace.Tracer:
        """The underlying :class:`opentelemetry.trace.Tracer`."""
        return self._tracer


__all__ = ["SystemOneOTelInstrumentor"]
