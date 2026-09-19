"""ASGI protocol and audit evidence regression tests."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from system1 import ChoiceField, DecisionSchema, System1Engine
from system1.integrations import SystemOneGatewayMiddleware, SystemOneMCPBlockedError
from system1.guard import DecisionOutcome, GuardInterceptionResult


class Route(DecisionSchema):
    route = ChoiceField(options=["local", "review"])


SCOPE = {"type": "http", "method": "POST", "path": "/v1/decide", "headers": []}


@pytest.mark.parametrize("body", [b"null", b"42", b"true", b'"prompt"', b"[]", b"{invalid"])
async def test_non_object_json_passes_through_and_replays_once(body):
    messages = iter([
        {"type": "http.request", "body": body[:2], "more_body": True},
        {"type": "http.request", "body": body[2:], "more_body": False},
        {"type": "http.disconnect"},
    ])

    async def receive():
        return next(messages)

    async def downstream(scope, receive, send):
        assert await receive() == {"type": "http.request", "body": body, "more_body": False}
        assert await receive() == {"type": "http.disconnect"}

    engine = Mock()
    middleware = SystemOneGatewayMiddleware(downstream, Route, engine=engine)
    await middleware(dict(SCOPE), receive, None)
    engine.decide.assert_not_called()


async def test_disconnect_during_body_does_not_evaluate_partial_request():
    messages = iter([
        {"type": "http.request", "body": b'{"prompt": "partial"}', "more_body": True},
        {"type": "http.disconnect"},
    ])

    async def receive():
        return next(messages)

    engine, downstream = Mock(), Mock()
    middleware = SystemOneGatewayMiddleware(downstream, Route, engine=engine)
    await middleware(dict(SCOPE), receive, None)
    engine.decide.assert_not_called()
    downstream.assert_not_called()


def test_multipart_prompt_ignores_non_text_values():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": None}, {"type": "text", "text": 42},
        {"type": "text", "text": "hello"},
    ]}]}
    assert SystemOneGatewayMiddleware._default_extract_prompt(body) == "hello"


async def test_structured_escalation_is_respected_with_singleton_set():
    engine = Mock()
    engine.decide.return_value = SimpleNamespace(
        confidences={"route": 0.99}, conformal_sets={"route": ["local"]},
        is_ambiguous=True, escalated_fields=["route"], schema_digest="schema",
    )
    downstream_called = False

    async def downstream(scope, receive, send):
        nonlocal downstream_called
        downstream_called = True
        assert scope["state"]["system1_escalation"]["confidences"] == {"route": 0.99}

    async def receive():
        return {"type": "http.request", "body": b'{"prompt":"hello"}', "more_body": False}

    await SystemOneGatewayMiddleware(downstream, Route, engine=engine)(dict(SCOPE), receive, None)
    assert downstream_called


async def test_gateway_and_mcp_publish_receipt_digest():
    decision = System1Engine(Route).decide("local lookup")
    engine = Mock()
    engine.decide.return_value = decision
    sent = []

    async def receive():
        return {"type": "http.request", "body": json.dumps({"prompt": "local lookup"}).encode()}

    async def send(message):
        sent.append(message)

    middleware = SystemOneGatewayMiddleware(
        Mock(), Route, engine=engine, is_fastpath_fn=lambda *_: True,
    )
    await middleware(dict(SCOPE), receive, send)
    digest = decision.receipt.compute_digest()
    assert dict(sent[0]["headers"])[b"x-reflex-receipt-digest"] == digest.encode()
    assert digest != decision.schema_digest

    error = SystemOneMCPBlockedError(GuardInterceptionResult(
        DecisionOutcome.REQUIRE_APPROVAL, "review", decision_result=decision,
    ))
    assert error.to_jsonrpc_error("request")["error"]["data"]["receipt_digest"] == digest
