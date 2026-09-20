"""System 1 Prometheus / OpenMetrics Observability Integration.

Exposes a Prometheus-compatible ``/metrics`` HTTP endpoint for the System 1
decision engine, tracking decision counts, latency histograms, escalation
rates, cache-hit ratios, conformal-set sizes, and ledger depth.

Requires the optional ``prometheus_client`` package::

    python -m pip install 'system1[observability]'
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Callable, Optional, TYPE_CHECKING

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

if TYPE_CHECKING:
    from system1.engine import DecisionResult, SystemOneEngine


# Sub-millisecond histogram buckets suitable for the System 1 engine
_LATENCY_BUCKETS = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0)

# Default conformal-set-size buckets (1 to 20)
_SET_SIZE_BUCKETS = tuple(float(i) for i in range(1, 21))


class SystemOneMetricsExporter:
    """Prometheus metrics exporter for :class:`SystemOneEngine`.

    Usage::

        from system1.engine import SystemOneEngine
        from system1.integrations.observability import SystemOneMetricsExporter

        engine = SystemOneEngine(MySchema)
        metrics = SystemOneMetricsExporter()
        metrics.instrument(engine)      # auto-records on every decide()
        metrics.start_server(port=9090) # serves /metrics
    """

    def __init__(self, *, registry: Optional[CollectorRegistry] = None) -> None:
        self._registry = registry or CollectorRegistry()
        self._server_started = False

        # --- Counters ---
        self.decisions_total = Counter(
            "system1_decisions_total",
            "Total number of System 1 decisions evaluated.",
            labelnames=["schema", "outcome", "cache_hit"],
            registry=self._registry,
        )

        self.escalations_total = Counter(
            "system1_escalations_total",
            "Total number of escalated decisions.",
            labelnames=["schema", "reason"],
            registry=self._registry,
        )

        # --- Histograms ---
        self.decision_latency_seconds = Histogram(
            "system1_decision_latency_seconds",
            "Decision evaluation latency in seconds.",
            labelnames=["schema"],
            buckets=_LATENCY_BUCKETS,
            registry=self._registry,
        )

        self.conformal_set_size = Histogram(
            "system1_conformal_set_size",
            "Size of conformal prediction sets per decision.",
            labelnames=["schema"],
            buckets=_SET_SIZE_BUCKETS,
            registry=self._registry,
        )

        # --- Gauges ---
        self.cache_hit_ratio = Gauge(
            "system1_cache_hit_ratio",
            "Cumulative ratio of cache hits to total decisions since exporter creation.",
            registry=self._registry,
        )

        self.ledger_entries_total = Gauge(
            "system1_ledger_entries_total",
            "Current number of entries in the ActionLedger.",
            registry=self._registry,
        )

        # Internal bookkeeping for cache-hit ratio
        self._total_decisions = 0
        self._cache_hits = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_decision(
        self,
        result: "DecisionResult",
        schema_name: str,
        cache_hit: bool,
        escalated: bool,
    ) -> None:
        """Record metrics for a single decision evaluation.

        Parameters
        ----------
        result:
            The :class:`DecisionResult` returned by :meth:`SystemOneEngine.decide`.
        schema_name:
            Name of the decision schema used.
        cache_hit:
            Whether the result came from the Tier 0 semantic cache.
        escalated:
            Whether the decision was escalated (ambiguous / low margin / OOD).
        """
        # Determine "outcome" — the value of the first ChoiceField (top-level)
        outcome = ""
        if result.values:
            first_value = next(iter(result.values.values()), "")
            outcome = str(first_value)

        cache_hit_label = "true" if cache_hit else "false"

        self.decisions_total.labels(
            schema=schema_name,
            outcome=outcome,
            cache_hit=cache_hit_label,
        ).inc()

        # Latency: result stores ms, Prometheus convention is seconds
        latency_s = result.latency_ms / 1000.0
        self.decision_latency_seconds.labels(schema=schema_name).observe(latency_s)

        # Conformal set sizes
        if result.conformal_sets:
            for _field_name, cset in result.conformal_sets.items():
                self.conformal_set_size.labels(schema=schema_name).observe(
                    float(len(cset))
                )

        # Escalation
        if escalated:
            reason = self._classify_escalation_reason(result)
            self.escalations_total.labels(schema=schema_name, reason=reason).inc()

        # Cache-hit ratio since exporter creation
        with self._lock:
            self._total_decisions += 1
            if cache_hit:
                self._cache_hits += 1
            ratio = self._cache_hits / self._total_decisions
            self.cache_hit_ratio.set(ratio)

    def update_ledger_gauge(self, entry_count: int) -> None:
        """Manually update the ledger entries gauge.

        Parameters
        ----------
        entry_count:
            The current number of entries in the :class:`ActionLedger`.
        """
        self.ledger_entries_total.set(entry_count)

    # ------------------------------------------------------------------
    # Instrumentation (monkey-patch wrapper)
    # ------------------------------------------------------------------

    def instrument(self, engine: "SystemOneEngine") -> "SystemOneEngine":
        """Wrap *engine.decide* so that every call auto-records metrics.

        Returns the same engine instance (mutated) for chaining convenience.
        """
        original_decide = engine.decide

        @functools.wraps(original_decide)
        def _instrumented_decide(*args: Any, **kwargs: Any) -> "DecisionResult":
            result = original_decide(*args, **kwargs)
            self.record_decision(
                result=result,
                schema_name=result.schema_name,
                cache_hit=result.is_cache_hit,
                escalated=result.is_ambiguous,
            )
            # Best-effort ledger gauge update
            if engine.ledger is not None:
                try:
                    seq, _ = engine.ledger.audit_head()
                    self.update_ledger_gauge(seq)
                except Exception:
                    pass
            return result

        engine.decide = _instrumented_decide  # type: ignore[method-assign]
        return engine

    # ------------------------------------------------------------------
    # HTTP server
    # ------------------------------------------------------------------

    def start_server(self, port: int = 9090) -> None:
        """Start a background HTTP server exposing ``/metrics``.

        Parameters
        ----------
        port:
            TCP port to listen on (default 9090).
        """
        if self._server_started:
            return
        start_http_server(port, registry=self._registry)
        self._server_started = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_escalation_reason(result: "DecisionResult") -> str:
        """Heuristically classify why a decision was escalated."""
        if result.is_ambiguous:
            # Check if it's a margin-based escalation
            for _field, active in result.margin_gate_active.items():
                if active:
                    return "low_margin"
            return "ambiguous"
        return "ood"

    @property
    def registry(self) -> CollectorRegistry:
        """The :class:`CollectorRegistry` backing this exporter."""
        return self._registry


__all__ = ["SystemOneMetricsExporter"]
