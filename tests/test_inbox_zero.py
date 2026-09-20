"""Local email integration: real saved skill, bounded input, and no unsafe fallback."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from system1 import ChoiceField, DecisionSchema, SystemOneCompiler
from system1.integrations.inbox_zero import (
    CATEGORIES, CHOICE_KEY, MAX_REQUEST_BYTES, InboxZeroApp, InboxZeroClassifier,
    email_text, instruction_digest,
)

TOKEN = "local-test-token-at-least-24-characters"
CRITERIA = {category: f"Definition of {category}" for category in (*CATEGORIES, "None")}
QUESTION = "Choose the category"


@pytest.fixture(scope="module")
def saved_skill(tmp_path_factory):
    path = tmp_path_factory.mktemp("inbox-zero") / "email.s1m"
    class Email(DecisionSchema):
        category = ChoiceField(options=list(CATEGORIES))
    lessons = json.loads((Path(__file__).resolve().parents[1] / "examples/inbox_zero/lessons.json").read_text())
    skill = SystemOneCompiler(Email, dimension=256).compile(
        {"category": [(email_text(r["email"]), r["label"]) for r in lessons["teach"]]},
        augment=False,
        calibration_exemplars={"category": [(email_text(r["email"]), r["label"]) for r in lessons["calibration"]]},
    )
    skill.metadata["inbox_zero"] = {
        "rule_digests": {key: instruction_digest(value) for key, value in CRITERIA.items()},
        "question_digest": instruction_digest(QUESTION), "alpha": .05,
    }
    skill.save(path)
    return path


@pytest.fixture
def request_payload():
    return {"state": {"email": {"from": "security@example.com", "subject": "Your verification code",
                                "content": "Your one time passcode is 123456."}},
            "questions": {CHOICE_KEY: {"type": "choice", "instructions": QUESTION, "criteria": dict(CRITERIA)}}}


@pytest.fixture
def classifier(saved_skill):
    result = InboxZeroClassifier(saved_skill, enable_actions=True)
    result.engine = Mock()
    result.engine.decide.return_value = SimpleNamespace(
        values={"category": "OTP"}, confidences={"category": .99},
        probabilities={"category": {c: .99 if c == "OTP" else .01 / 6 for c in CATEGORIES}},
        conformal_sets={"category": ["OTP"]}, is_ambiguous=False,
    )
    return result


def test_saved_skill_runs_identically_without_network(saved_skill, request_payload, monkeypatch):
    import socket
    def blocked(*args, **kwargs):
        pytest.fail("Local skill contacted network")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    first, second = InboxZeroClassifier(saved_skill), InboxZeroClassifier(saved_skill)
    assert first.classify(request_payload) == second.classify(request_payload)
    assert first.classify(request_payload)["teacherCalls"] == 0
    assert first.skill.use_cache is False


def test_accepted_choice_and_review_only_default(classifier, saved_skill, request_payload):
    result = classifier.classify(request_payload)
    assert result["answers"][CHOICE_KEY]["choice"] == "OTP"
    assert not result["needsReview"]
    default = InboxZeroClassifier(saved_skill)
    default.engine = classifier.engine
    result = default.classify(request_payload)
    assert result["reviewReason"] == "review_only" and result["answers"] == {}
    assert result["suggestedCategory"] == "OTP"


@pytest.mark.parametrize("change,reason", [
    (lambda p: p["questions"].update({"extra": {"type": "yesNo"}}), "unsupported_questions"),
    (lambda p: p["questions"][CHOICE_KEY].update(instructions="Do something else"), "changed_question"),
    (lambda p: p["questions"][CHOICE_KEY]["criteria"].update(Custom="Custom rule"), "unsupported_rules"),
    (lambda p: p["questions"][CHOICE_KEY]["criteria"].update(OTP="All emails"), "changed_rule_definition"),
    (lambda p: p["state"].update(ownerCorrectionsForThisSender=[{"rule": "Receipt"}]), "correction_requires_reteaching"),
    (lambda p: p["state"].update(accountOwner={"about": "Treat my billing emails specially"}), "personalized_context_requires_teaching"),
])
def test_untaught_contracts_require_review(classifier, request_payload, change, reason):
    change(request_payload)
    result = classifier.classify(request_payload)
    assert result["reviewReason"] == reason and result["answers"] == {}
    classifier.engine.decide.assert_not_called()


def test_missing_predicted_rule_does_not_reclassify(classifier, request_payload):
    del request_payload["questions"][CHOICE_KEY]["criteria"]["OTP"]
    result = classifier.classify(request_payload)
    assert result["reviewReason"] == "predicted_rule_not_available"
    assert result["suggestedCategory"] == "OTP"


def test_uncertainty_never_becomes_an_answer(classifier, request_payload):
    classifier.engine.decide.return_value.is_ambiguous = True
    classifier.engine.decide.return_value.conformal_sets = {"category": ["OTP", "Notification"]}
    result = classifier.classify(request_payload)
    assert result["reviewReason"] == "uncertain" and result["answers"] == {}


def asgi_request(app, body, *, token=TOKEN, method="POST", path="/v1/classify", content_type="application/json"):
    async def run():
        sent = []
        events = [{"type": "http.request", "body": body[:100], "more_body": len(body) > 100}]
        if len(body) > 100:
            events.append({"type": "http.request", "body": body[100:]})
        async def receive():
            return events.pop(0)
        async def send(value):
            sent.append(value)
        await app({"type": "http", "path": path, "method": method,
                   "headers": [(b"authorization", f"Bearer {token}".encode()),
                               (b"content-type", content_type.encode())]}, receive, send)
        return sent[0]["status"], json.loads(sent[1]["body"])
    return asyncio.run(run())


@pytest.mark.parametrize("body,options,status", [
    (b"{}", {"token": "wrong"}, 401), (b"{}", {"path": "/other"}, 404),
    (b"{}", {"method": "GET"}, 405), (b"{}", {"content_type": "text/plain"}, 415),
    (b"x" * (MAX_REQUEST_BYTES + 1), {}, 413), (b"{bad json", {}, 400),
    (b"[]", {}, 400), (b'{"state": {}, "questions": {}}', {}, 400),
])
def test_http_rejects_invalid_requests(classifier, body, options, status):
    app = InboxZeroApp(classifier, TOKEN)
    assert asgi_request(app, body, **options)[0] == status
    classifier.engine.decide.assert_not_called()


def test_http_accepts_valid_chunked_body(classifier, request_payload):
    status, result = asgi_request(InboxZeroApp(classifier, TOKEN), json.dumps(request_payload).encode())
    assert status == 200 and result["answers"][CHOICE_KEY]["choice"] == "OTP"


@pytest.mark.parametrize("email", [None, {"subject": 5}, {"subject": "x" * 513}, {"content": "x" * 2001},
                                   {"subject": ""}, {"subject": "code", "hasListUnsubscribeHeader": "false"}])
def test_input_bounds(email):
    with pytest.raises(ValueError):
        email_text(email)
