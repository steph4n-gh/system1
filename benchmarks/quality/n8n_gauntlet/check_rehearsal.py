"""Check real n8n routing before/after stopping the separate teacher process.

These fixed development examples exercise wiring, never qualify the skill.
"""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
URL = "http://127.0.0.1:5681/webhook/system1-disconnection-rehearsal"


def ledger(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    return dict(sha256=hashlib.sha256(raw).hexdigest(),
                attempted_provider_calls=sum(row["event"] == "attempt" for row in rows),
                provider_responses=sum(row["event"] == "response" for row in rows))


def run(phase, journal, output):
    if output.exists():
        raise ValueError("Use a new output path; retain previous runs")
    token = os.environ["SYSTEM1_CLASSIFIER_TOKEN"]
    fixtures = json.loads((HERE / "rehearsal-probes.json").read_text())
    rows = fixtures["requests"]
    report = dict(scope="UNQUALIFIED development integration; not quality confirmation or teacher accuracy",
                  phase=phase, candidate=fixtures["candidate"], n8n_version="2.39.10",
                  journal_before=ledger(journal), scenarios=[])
    if phase != "connected":
        port = 8794 if phase == "disconnected" else 8793
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                raise AssertionError("Expected service is still listening")
        except OSError as exc:
            assert exc.errno == errno.ECONNREFUSED
            report["stopped_service_control"] = dict(port=port, errno=exc.errno)
    scenarios = {"mixed": rows, "all_local": rows[:2]}
    if phase == "disconnected":
        scenarios["all_review"] = rows[2:]
    if phase == "local-down":
        scenarios = {"local_service_unavailable": rows[:2]}
    for name, inputs in scenarios.items():
        payload = dict(requests=[{key: row[key] for key in ("id", "text")} for row in inputs])
        req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + token})
        started = time.perf_counter()
        with urllib.request.urlopen(req, timeout=70) as response:
            outputs = json.load(response)
        report["scenarios"].append(dict(name=name, inputs=payload, outputs=outputs,
                                       round_trip_ms=(time.perf_counter() - started) * 1000))
        # Save real responses before assertions, including a failed run's evidence.
        output.write_text(json.dumps(report, indent=2) + "\n")
        assert isinstance(outputs, list) and len(outputs) == len(inputs)
        assert sorted(row["id"] for row in inputs) == sorted(row["id"] for row in outputs)
        by_id = {row["id"]: row for row in inputs}
        for value in outputs:
            original = by_id[value["id"]]
            assert value["text"] == original["text"]
            assert value["qualification"] == "development_only" and "error" not in value
            if phase == "local-down" or value["id"] == "malformed":
                assert value["route"] == "review" and value["category"] is None
                assert value["needsReview"] and value["teacherCalls"] == value["teacherRequests"] == 0
            elif not original["local_review"]:
                assert value["route"] == "local" and not value["needsReview"]
                assert value["category"] == original["recorded_category"]
                assert value["candidate"] == fixtures["candidate"]
                assert value["teacherCalls"] == value["teacherRequests"] == 0
            elif phase == "connected":
                assert value["candidate"] == fixtures["candidate"]
                assert value["teacherCalls"] == value["teacherRequests"] == 1
                assert value["teacherStatus"] == "responded"
                assert value["reason"] in ("teacher_answer", "teacher_review")
                assert isinstance(value["model"], str)
                # A teacher may abstain. Correctness is not inferred from its answer.
                assert value["route"] in ("teacher", "review")
            else:
                assert value["route"] == "review" and value["needsReview"] and value["category"] is None
                assert value["teacherStatus"] == "unavailable" and value["teacherRequests"] == 1
                assert value["teacherCalls"] is None  # No response: count unknown at the caller.
    req = urllib.request.Request(URL, data=b'{"requests":[]}',
        headers={"Content-Type": "application/json", "Authorization": "Bearer invalid"})
    try:
        urllib.request.urlopen(req, timeout=5)
    except urllib.error.HTTPError as exc:
        report["unauthorized_status"] = exc.code
        assert exc.code in (401, 403)
    else:
        raise AssertionError("Invalid credential accepted")
    report["journal_after"] = ledger(journal)
    new_calls = report["journal_after"]["attempted_provider_calls"] - report["journal_before"]["attempted_provider_calls"]
    assert new_calls == (2 if phase == "connected" else 0)
    if phase == "connected":
        assert report["journal_after"]["provider_responses"] - report["journal_before"]["provider_responses"] == 2
    if phase != "connected":
        assert report["journal_before"] == report["journal_after"]
    report["checks_passed"] = True
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(phase=phase, decisions=sum(len(s["outputs"]) for s in report["scenarios"]),
                         new_provider_attempts=new_calls, checks_passed=True)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("connected", "disconnected", "local-down"))
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.phase, args.journal, args.output)
