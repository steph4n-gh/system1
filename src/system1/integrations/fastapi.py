"""Reflex FastAPI / Starlette Gateway Middleware.

Sub-millisecond on-device ASGI middleware for local fast-path routing
and fail-closed conformal escalation to frontier governors (Astra, Fable, Gemini, Grok).
"""

from __future__ import annotations

import json
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from system1.core import DecisionSchema
from system1.engine import DecisionResult, ReflexEngine
from system1.receipt import DecisionWitnessReceipt, create_decision_receipt


class ReflexGatewayMiddleware:
    """ASGI Middleware providing sub-millisecond local routing and frontier escalation.
    
    Compatible with FastAPI, Starlette, Litestar, and any ASGI 3.0 compliant framework.
    """

    def __init__(
        self,
        app: Any,
        schema: Union[DecisionSchema, Type[DecisionSchema]],
        *,
        engine: Optional[ReflexEngine] = None,
        route_paths: Sequence[str] = ("/v1/chat/completions", "/v1/decide", "/api/route"),
        fastpath_threshold: Optional[float] = 0.85,
        alpha: float = 0.05,
        require_singleton_conformal: bool = True,
        is_fastpath_fn: Optional[Callable[[DecisionResult, Dict[str, Any]], bool]] = None,
        extract_prompt_fn: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
        fastpath_response_fn: Optional[Callable[[DecisionResult, Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        self.app = app
        self.schema = schema
        self.engine = engine or ReflexEngine(schema)
        self.route_paths = tuple(route_paths)
        self.fastpath_threshold = fastpath_threshold
        self.alpha = alpha
        self.require_singleton_conformal = require_singleton_conformal
        self.is_fastpath_fn = is_fastpath_fn
        self.extract_prompt_fn = extract_prompt_fn or self._default_extract_prompt
        self.fastpath_response_fn = fastpath_response_fn or self._default_fastpath_response

    @staticmethod
    def _default_extract_prompt(body_json: Dict[str, Any]) -> Optional[str]:
        """Extracts text prompt from typical LLM / Gateway request formats."""
        if "prompt" in body_json and isinstance(body_json["prompt"], str):
            return body_json["prompt"]
        if "input" in body_json and isinstance(body_json["input"], str):
            return body_json["input"]
        if "messages" in body_json and isinstance(body_json["messages"], list):
            # Extract last user message
            for msg in reversed(body_json["messages"]):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        # Multi-part content
                        text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                        return " ".join(text_parts)
        return None

    @staticmethod
    def _default_fastpath_response(decision: DecisionResult, body_json: Dict[str, Any]) -> Dict[str, Any]:
        """Constructs an OpenAI-compatible / Reflex response payload."""
        first_choice = next(iter(decision.values.values()), None) if decision.values else None
        return {
            "id": f"reflex_{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "reflex-system1-metal",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(decision.values),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "reflex": {
                "decision": decision.values,
                "confidences": decision.confidences,
                "latency_ms": decision.latency_ms,
                "schema_digest": decision.schema_digest,
                "execution": "local_metal_fastpath",
            },
        }

    async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        """ASGI 3.0 entry point."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET")

        # Fast-path evaluation applies only to POST requests matching route_paths or explicit header
        headers = dict(scope.get("headers", []))
        explicit_route = headers.get(b"x-reflex-route", b"").lower() == b"true"
        matches_path = any(path.startswith(rp) for rp in self.route_paths)

        if method != "POST" or (not matches_path and not explicit_route):
            await self.app(scope, receive, send)
            return

        # Receive the request body
        body_chunks = []
        more_body = True
        while more_body:
            message = await receive()
            body_chunks.append(message.get("body", b""))
            more_body = message.get("more_body", False)

        body_bytes = b"".join(body_chunks)

        # Re-inject body so downstream can read if escalated
        async def replay_receive() -> Dict[str, Any]:
            return {"type": "http.request", "body": body_bytes, "more_body": False}

        try:
            body_json = json.loads(body_bytes) if body_bytes else {}
        except Exception:
            # If payload is not valid JSON, escalate downstream
            await self.app(scope, replay_receive, send)
            return

        prompt = self.extract_prompt_fn(body_json)
        if not prompt:
            await self.app(scope, replay_receive, send)
            return

        # Fast forward pass on local metal
        t0 = time.perf_counter()
        decision: DecisionResult = self.engine.decide(
            prompt,
            alpha=self.alpha,
            record_receipt=True,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Conformal & confidence gating
        if self.is_fastpath_fn is not None:
            is_confident = self.is_fastpath_fn(decision, body_json)
        else:
            is_confident = True
            if self.fastpath_threshold is not None:
                for k, conf in decision.confidences.items():
                    if conf < self.fastpath_threshold:
                        is_confident = False
                        break

            if self.require_singleton_conformal and is_confident:
                for k, cset in decision.conformal_sets.items():
                    if len(cset) != 1:
                        is_confident = False
                        break

        # If confident and unambiguous, resolve immediately via fast-path
        if is_confident:
            resp_data = self.fastpath_response_fn(decision, body_json)
            resp_bytes = json.dumps(resp_data).encode("utf-8")

            res_headers = [
                (b"content-type", b"application/json"),
                (b"x-reflex-status", b"fastpath"),
                (b"x-reflex-latency-ms", f"{latency_ms:.3f}".encode("utf-8")),
                (b"x-reflex-receipt-digest", str(decision.schema_digest or "").encode("utf-8")),
            ]

            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": res_headers,
            })
            await send({
                "type": "http.response.body",
                "body": resp_bytes,
                "more_body": False,
            })
            return

        # Otherwise escalate upstream to the deliberate governor
        scope["state"] = scope.get("state", {})
        scope["state"]["reflex_escalation"] = {
            "latency_ms": latency_ms,
            "confidences": decision.confidences,
            "conformal_sets": {k: list(v) for k, v in decision.conformal_sets.items()},
            "digest": decision.schema_digest,
        }

        # Intercept send to inject escalation header
        async def send_with_escalation_header(message: Dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                hdr = list(message.get("headers", []))
                hdr.append((b"x-reflex-status", b"escalated"))
                hdr.append((b"x-reflex-latency-ms", f"{latency_ms:.3f}".encode("utf-8")))
                message["headers"] = hdr
            await send(message)

        await self.app(scope, replay_receive, send_with_escalation_header)


def add_reflex_gateway(
    app: Any,
    schema: Union[DecisionSchema, Type[DecisionSchema]],
    *,
    engine: Optional[ReflexEngine] = None,
    **kwargs: Any,
) -> Any:
    """Convenience helper to add ReflexGatewayMiddleware to a FastAPI or Starlette application."""
    if hasattr(app, "add_middleware"):
        app.add_middleware(ReflexGatewayMiddleware, schema=schema, engine=engine, **kwargs)
        return app
    return ReflexGatewayMiddleware(app, schema=schema, engine=engine, **kwargs)


__all__ = [
    "ReflexGatewayMiddleware",
    "add_reflex_gateway",
]
