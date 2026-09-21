"""Bound API attempts and keep reserved data out of generated teaching labels."""
import io
import json
from pathlib import Path
import urllib.error

import pytest


@pytest.fixture
def teaching(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks/quality/n8n_gauntlet"))
    import teach_boundaries
    return teach_boundaries


def item():
    return dict(id="banking77-target", dataset="banking77", label="target", supported_count=2,
                unsupported_count=0, request_sha256="fixed-request", payload=dict(contents=[]))


def response(supported, unsupported=None):
    data = dict(supported=supported)
    if unsupported is not None:
        data["unsupported"] = unsupported
    return dict(candidates=[dict(content=dict(parts=[dict(text=json.dumps(data))]))],
                usageMetadata=dict(promptTokenCount=5, candidatesTokenCount=8))


def test_failed_attempt_is_not_retried_or_leaked(teaching, tmp_path, monkeypatch):
    calls = []
    def fail(request, **kwargs):
        calls.append(request)
        raise urllib.error.HTTPError("https://invalid/secret-must-stay-private", 401,
                                     "secret-must-stay-private", {}, None)
    monkeypatch.setattr(teaching.urllib.request, "urlopen", fail)
    request = item()
    result = teaching.issue(request, tmp_path, "secret-must-stay-private")
    assert result["status"] == "error" and result["http_status"] == 401
    assert teaching.issue(request, tmp_path, "secret-must-stay-private") is None
    assert len(calls) == 1
    assert all("secret-must-stay-private" not in p.read_text() for p in tmp_path.iterdir())
    with pytest.raises(ValueError, match="different request"):
        teaching.issue(dict(request, request_sha256="changed"), tmp_path, "secret")


def test_success_preserves_raw_usage_without_repeating_call(teaching, tmp_path, monkeypatch):
    raw = response(["First", "Second"])
    calls = []
    def receive(request, **kwargs):
        calls.append(request)
        assert request.get_header("X-goog-api-key") == "private-key"
        assert "private-key" not in request.full_url and b"private-key" not in request.data
        return io.BytesIO(json.dumps(raw).encode())
    monkeypatch.setattr(teaching.urllib.request, "urlopen", receive)
    assert teaching.issue(item(), tmp_path, "private-key")["response"] == raw
    assert teaching.issue(item(), tmp_path, "private-key") is None and len(calls) == 1
    assert all("private-key" not in p.read_text() for p in tmp_path.iterdir())


def test_interrupted_slot_consumes_budget_without_a_retry(teaching, tmp_path, monkeypatch):
    request = item()
    teaching.write_new(tmp_path / (request["id"] + ".json"), dict(request_sha256=request["request_sha256"], status="started"))
    monkeypatch.setattr(teaching.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Repeated uncertain API attempt"))
    assert teaching.issue(request, tmp_path, "private") is None


def test_excludes_normalized_reserved_groups_and_cross_batch_duplicates(teaching):
    from prepare import group
    reserved = {group("A reserved REQUEST!")}
    lessons, rejected = teaching.lessons_from(item(), response(["a reserved request", "A new request"]), reserved)
    assert [r["prompt"] for r in lessons] == ["A new request"]
    assert rejected[0]["reason"] == "duplicate_group"
    # A duplicate from a later category/dataset must also be rejected.
    next_item = dict(item(), id="clinc150-other", label="other", dataset="clinc150")
    lessons, rejected = teaching.lessons_from(next_item, response(["A NEW REQUEST!", "Something distinct"]), reserved)
    assert len(lessons) == 1 and len(rejected) == 1
    assert lessons[0]["label"] == "other"


@pytest.mark.parametrize("raw", [[], {}, response(["Only one"]), response(["One", "Two"], [{"text": "Unknown", "reason": "No match"}])])
def test_malformed_responses_do_not_become_labels(teaching, raw):
    lessons, rejected = teaching.lessons_from(item(), raw, set())
    assert lessons == [] and rejected[0]["reason"] == "invalid_or_missing_response"


def test_unsupported_reason_is_preserved_but_not_treated_as_verified(teaching):
    request = dict(item(), dataset="clinc150", unsupported_count=2)
    raw = response(["Supported first", "Supported second"],
                   [dict(text="Unfamiliar request", reason="Teacher explanation"), dict(text="Other unfamiliar", reason="")])
    lessons, rejected = teaching.lessons_from(request, raw, set())
    negative = next(r for r in lessons if r["label"] == "oos")
    assert negative["teacher_reason"] == "Teacher explanation"
    assert negative["source"] == "gemini-boundary-teaching-unverified"
    assert negative["prompt"] == "Unfamiliar request"
    assert len(rejected) == 1 and rejected[0]["reason"] == "invalid_text_or_reason"
