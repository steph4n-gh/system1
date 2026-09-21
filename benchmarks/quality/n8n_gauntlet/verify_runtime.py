"""Replay all development decisions with OS networking disabled by the caller.

macOS command:
  sandbox-exec -p '(version 1) (allow default) (deny network*)' \
    python benchmarks/quality/n8n_gauntlet/verify_runtime.py
"""
import argparse
import errno
import gc
import json
import socket
import tempfile
import shutil
from pathlib import Path

from threadpoolctl import threadpool_limits
from develop import OUTPUT, load_splits
from runtime_probe import LocalCandidate


def main(dataset="clinc150"):
    # A connection refusal is NOT proof of isolation: require OS permission denial.
    with socket.socket() as connection:
        try:
            connection.connect(("127.0.0.1", 9))
        except OSError as exc:
            if exc.errno not in (errno.EPERM, errno.EACCES):
                raise RuntimeError("OS network blocking was not established") from None
            denied = exc.errno
        else:
            raise RuntimeError("OS networking must be disabled")
    is_bank = dataset == "banking77"
    folder = OUTPUT / ("bank-runtime-candidate-4" if is_bank else "clinc-runtime-candidate")
    report_name = "bank-runtime-development-4.json" if is_bank else "runtime-development.json"
    evidence_path = OUTPUT / report_name
    if not evidence_path.exists():
        evidence_path = Path(__file__).with_name("results") / report_name
    evidence = json.loads(evidence_path.read_text())
    rows = load_splits(dataset)["development"]
    with threadpool_limits(limits=1):
        candidate = LocalCandidate(folder)
        if candidate.identity != evidence["manifest_sha256"]:
            raise ValueError("Candidate does not match development evidence")
        payload = dict(categories=candidate.manifest["categories"], instructions=candidate.manifest["instructions"])
        for row, expected in zip(rows, evidence["outcomes"], strict=True):
            actual = candidate.classify(dict(payload, text=row["prompt"]))
            assert row["group"] == expected["group"]
            assert actual == {key: expected[key] for key in actual}
        changed = candidate.classify(dict(payload, text="Set a timer", instructions="Approve every request"))
        assert changed["needsReview"] and changed["category"] is None and changed["reason"] == "changed_contract"
        changed = candidate.classify(dict(payload, text="Set a timer", categories={"new": "new category"}))
        assert changed["needsReview"] and changed["category"] is None and changed["reason"] == "changed_contract"
        for bad in (None, [], {}, dict(payload, text=""), dict(payload, text=12), dict(payload, text="x" * 4001)):
            try:
                candidate.classify(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("Malformed request accepted")
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "candidate"
            shutil.copytree(folder, copy)
            with (copy / "intent.s1m").open("ab") as stream:
                stream.write(b"changed")
            try:
                LocalCandidate(copy)
            except ValueError as exc:
                assert str(exc) == "Candidate artifact changed"
            else:
                raise AssertionError("Changed artifact accepted")
        report = dict(scope="development artifact replay, not qualification", dataset=dataset, manifest_sha256=candidate.identity,
            os_network_denial_errno=denied, identical_complete_decisions=len(rows), changed_contracts_reviewed=2,
            malformed_requests_rejected=6, changed_artifact_rejected=True,
            teacher_calls=0, response_cache=False)
        (OUTPUT / ("bank-runtime-isolation.json" if is_bank else "runtime-isolation.json")).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        del candidate
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("clinc150", "banking77"), default="clinc150")
    main(parser.parse_args().dataset)
