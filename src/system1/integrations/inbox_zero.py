"""A saved, bounded email-category skill for Inbox Zero's classifier boundary.

The service never contacts a teacher. The caller must handle ``needs_review``
before using an answer. User-defined rules need a separately taught skill.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from system1.compiler import CompiledSystemOneModel
from system1.engine import System1Engine

CATEGORIES = ("Newsletter", "Marketing", "Calendar", "Receipt", "Notification", "OTP", "Conversations")
CHOICE_KEY = "__rule_choice__"
MAX_REQUEST_BYTES = 64 * 1024
MAX_CONTENT_CHARS = 2000


def instruction_digest(value: str) -> str:
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()


def email_text(email: Mapping[str, Any]) -> str:
    """Use exactly the same representation during teaching and serving."""
    if not isinstance(email, Mapping):
        raise ValueError("email must be an object")
    parts = []
    for key, limit in (("from", 512), ("subject", 512), ("content", MAX_CONTENT_CHARS)):
        value = email.get(key, "")
        if not isinstance(value, str) or len(value) > limit:
            raise ValueError(f"email.{key} must be a string of at most {limit} characters")
        parts.append(f"{key}: {value}")
    if not email.get("subject", "").strip() and not email.get("content", "").strip():
        raise ValueError("email needs a subject or content")
    unsubscribe = email.get("hasListUnsubscribeHeader", False)
    if not isinstance(unsubscribe, bool):
        raise ValueError("hasListUnsubscribeHeader must be a boolean")
    parts.append(f"list_unsubscribe: {'present' if unsubscribe else 'absent'}")
    return "\n".join(parts)


class InboxZeroClassifier:
    """Classify with one startup-loaded skill and a fixed set of rule definitions."""

    def __init__(self, model_path: str | Path, *, enable_actions: bool = False) -> None:
        self.enable_actions = enable_actions
        self.skill = CompiledSystemOneModel.load(model_path)
        if set(self.skill.schema.fields) != {"category"}:
            raise ValueError("Expected one category field")
        if tuple(self.skill.schema.fields["category"].options) != CATEGORIES:
            raise ValueError("Expected the seven Inbox Zero email categories")
        metadata = self.skill.metadata.get("inbox_zero", {})
        self.rule_digests = metadata.get("rule_digests", {})
        self.question_digest = metadata.get("question_digest")
        if not isinstance(self.question_digest, str) or len(self.question_digest) != 64:
            raise ValueError("The skill must include the bound classification question")
        if set(self.rule_digests) != set(CATEGORIES) | {"None"} or any(
            not isinstance(value, str) or len(value) != 64 for value in self.rule_digests.values()
        ):
            raise ValueError("The skill must include the bound Inbox Zero rule definitions")
        self.alpha = metadata.get("alpha", 0.05)
        if not isinstance(self.alpha, (int, float)) or not 0 < self.alpha < 1:
            raise ValueError("Invalid saved review policy")
        self.skill.use_cache = False
        self.engine = System1Engine(self.skill.schema, model=self.skill, strict_mode=True, use_cache=False)
        self.model_id = f"system1-email-{hashlib.sha256(Path(model_path).read_bytes()).hexdigest()[:16]}"

    def classify(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Request must be an object")
        state = payload.get("state")
        questions = payload.get("questions")
        if not isinstance(state, dict) or not isinstance(questions, dict):
            raise ValueError("state and questions must be objects")
        text = email_text(state.get("email"))
        owner = state.get("accountOwner")
        if owner is not None and not isinstance(owner, dict):
            raise ValueError("accountOwner must be an object")
        if owner and owner.get("about"):
            return self._review("personalized_context_requires_teaching")
        if set(questions) != {CHOICE_KEY}:
            return self._review("unsupported_questions")
        question = questions[CHOICE_KEY]
        if not isinstance(question, dict) or question.get("type") != "choice":
            raise ValueError("Expected a choice question")
        instructions = question.get("instructions")
        if not isinstance(instructions, str) or instruction_digest(instructions) != self.question_digest:
            return self._review("changed_question")
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("criteria must be a nonempty object")
        if any(key not in self.rule_digests for key in criteria):
            return self._review("unsupported_rules")
        for key, description in criteria.items():
            if not isinstance(description, str):
                raise ValueError("Rule descriptions must be strings")
            if instruction_digest(description) != self.rule_digests[key]:
                return self._review("changed_rule_definition")
        # Corrections are lessons, not arbitrary new instructions at inference.
        # Until incorporated into a newly checked skill, leave the email for review.
        if state.get("ownerCorrectionsForThisSender"):
            return self._review("correction_requires_reteaching")
        decision = self.engine.decide(text, alpha=self.alpha, record_receipt=False)
        category = decision.values["category"]
        if decision.is_ambiguous:
            return self._review("uncertain", decision)
        if category not in criteria:
            return self._review("predicted_rule_not_available", decision)
        if not self.enable_actions:
            return self._review("review_only", decision)
        return {
            "model": self.model_id,
            "inputTokens": 0,
            "needsReview": False,
            "answers": {
                CHOICE_KEY: {
                    "type": "choice", "choice": category,
                    "confidence": decision.confidences["category"],
                    "probabilities": decision.probabilities["category"],
                }
            },
            "predictionSet": decision.conformal_sets["category"],
            "teacherCalls": 0,
        }

    def _review(self, reason: str, decision: Any = None) -> dict[str, Any]:
        return {
            "model": self.model_id, "inputTokens": 0, "answers": {},
            "needsReview": True, "reviewReason": reason, "teacherCalls": 0,
            "suggestedCategory": decision.values["category"] if decision else None,
            "predictionSet": decision.conformal_sets["category"] if decision else [],
        }


class InboxZeroApp:
    """Small ASGI endpoint; run it locally with an existing ASGI server."""

    def __init__(self, classifier: InboxZeroClassifier, token: str) -> None:
        if not token or len(token) < 24:
            raise ValueError("Use a service token with at least 24 characters")
        self.classifier = classifier
        self.authorization = f"Bearer {token}".encode("utf-8")

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif event["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        headers = dict(scope.get("headers", []))
        if not hmac.compare_digest(headers.get(b"authorization", b""), self.authorization):
            await self._respond(send, 401, {"error": "unauthorized"})
            return
        if scope.get("path") != "/v1/classify":
            await self._respond(send, 404, {"error": "not_found"})
            return
        if scope.get("method") != "POST":
            await self._respond(send, 405, {"error": "method_not_allowed"})
            return
        if headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower() != b"application/json":
            await self._respond(send, 415, {"error": "expected_json"})
            return
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            body.extend(event.get("body", b""))
            if len(body) > MAX_REQUEST_BYTES:
                await self._respond(send, 413, {"error": "request_too_large"})
                return
            if not event.get("more_body", False):
                break
        try:
            result = self.classifier.classify(json.loads(body))
        except (ValueError, TypeError, RecursionError):
            await self._respond(send, 400, {"error": "invalid_request"})
            return
        await self._respond(send, 200, result)

    @staticmethod
    async def _respond(send: Any, status: int, value: dict) -> None:
        body = json.dumps(value, allow_nan=False).encode("utf-8")
        await send({"type": "http.response.start", "status": status, "headers": [
            (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
            (b"cache-control", b"no-store"),
        ]})
        await send({"type": "http.response.body", "body": body})


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Serve a saved Inbox Zero email skill without a teacher")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8781)
    parser.add_argument("--enable-actions", action="store_true",
                        help="Return accepted choices after validating the skill on your own emails")
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError:
        parser.error("Install the optional server: pip install 'system1[http]'")
    token = os.environ.get("SYSTEM1_CLASSIFIER_TOKEN", "")
    app = InboxZeroApp(InboxZeroClassifier(args.model, enable_actions=args.enable_actions), token)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False, limit_concurrency=32,
                timeout_keep_alive=5, h11_max_incomplete_event_size=MAX_REQUEST_BYTES)


if __name__ == "__main__":
    main()
