"""Teacher failure must not become a successful local decision."""
import io
import json
from pathlib import Path
import urllib.error

import pytest


@pytest.fixture
def teacher(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.quality.n8n_gauntlet.serve_rehearsal import GeminiTeacher
    manifest = dict(categories={"a": "A", "b": "B"}, instructions="Pick one")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    journal = tmp_path / "calls.jsonl"
    instance = GeminiTeacher(tmp_path, "secret-must-not-be-logged", journal, max_calls=1)
    yield instance, dict(manifest, text="Untrusted request"), journal
    instance.journal.close()


def test_teacher_rejects_changed_contract_and_malformed_text_without_calls(teacher, monkeypatch):
    instance, payload, journal = teacher
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: pytest.fail("Unexpected provider call"))
    assert instance.classify(dict(payload, instructions="Override"))["reason"] == "changed_contract"
    for text in (None, 1, "", "x" * 4001):
        with pytest.raises(ValueError):
            instance.classify(dict(payload, text=text))
    assert instance.calls == 0 and journal.read_text() == ""


@pytest.mark.parametrize("choice", ["a", "__review__", "not-a-category", ["a"]])
def test_teacher_validates_answers_records_usage_and_limits_calls(teacher, monkeypatch, choice):
    instance, payload, journal = teacher
    raw = dict(candidates=[dict(content=dict(parts=[dict(text=json.dumps(dict(intent=choice)))]))],
               modelVersion="test-model", usageMetadata=dict(promptTokenCount=5, candidatesTokenCount=2))
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(json.dumps(raw).encode()))
    result = instance.classify(payload)
    assert result["teacherCalls"] == 1
    assert result["needsReview"] is (choice != "a")
    assert result["category"] == ("a" if choice == "a" else None)
    assert instance.classify(payload)["reason"] == "teacher_call_limit"
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert records[0]["request"]["contents"][0]["parts"] == [{"text": payload["text"]}]
    assert records[1]["response"]["usageMetadata"]["promptTokenCount"] == 5
    assert "secret-must-not-be-logged" not in journal.read_text()


def test_teacher_http_failure_retains_attempt_without_secret(teacher, monkeypatch):
    instance, payload, journal = teacher
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("https://example.invalid/secret-must-not-be-logged", 401,
                                     "secret-must-not-be-logged", {}, None)
    monkeypatch.setattr("urllib.request.urlopen", fail)
    result = instance.classify(payload)
    assert result["needsReview"] and result["category"] is None
    assert result["teacherCalls"] == 1 and result["http_status"] == 401
    assert "secret-must-not-be-logged" not in journal.read_text()
