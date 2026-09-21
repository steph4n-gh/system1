"""Replay development responses with fitting libraries blocked; no timing claim."""
import errno
import argparse
import gc
import importlib.abc
import json
from pathlib import Path
import socket
import sys


class BlockFittingLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"scipy", "sklearn"}:
            raise ImportError("Fitting library blocked: " + fullname)


sys.meta_path.insert(0, BlockFittingLibraries())

from threadpoolctl import threadpool_limits

from audit_observed_regression import digest, same
from develop import load_splits
from polynomial_runtime import PolynomialCandidate

HERE = Path(__file__).resolve().parent


def main(*, report_name="polynomial-runtime.json", artifact_prefix="polynomial-", candidate_class=PolynomialCandidate,
         scope="saved System1 development artifacts after numerical import repair; not qualification or new timing"):
    with socket.socket() as connection:
        try:
            connection.connect(("127.0.0.1", 9))
        except OSError as exc:
            assert exc.errno in (errno.EPERM, errno.EACCES), "Run with OS networking denied"
            denied = exc.errno
        else:
            raise AssertionError("Networking is enabled")
    report_path = HERE / "results" / report_name
    report = json.loads(report_path.read_text())
    results = []
    with threadpool_limits(limits=1):
        for measured in report["results"]:
            if measured["kind"] != "system1":
                continue
            folder = HERE / "artifacts" / (artifact_prefix + measured["folder"])
            candidate = candidate_class(folder)
            assert candidate.identity == measured["candidate_sha256"]
            rows = load_splits(measured["dataset"])["development"]
            payload = {key: candidate.manifest[key] for key in ("categories", "instructions")}
            # The pinned development rows are already sorted by normalized group.
            for row, recorded in zip(rows[:100], measured["outcomes"][:100], strict=True):
                actual = candidate.classify(dict(payload, text=row["prompt"]))
                expected = {k: v for k, v in recorded.items() if k not in ("group", "truth", "latency_ms")}
                same(actual, expected)
            for changed in (dict(payload, categories=[]), dict(payload, instructions="Changed decision")):
                response = candidate.classify(dict(changed, text=rows[0]["prompt"]))
                assert response == dict(candidate=candidate.identity, needsReview=True, category=None,
                                        reason="changed_contract", teacherCalls=0)
            for malformed in (None, [], dict(payload), dict(payload, text=12), dict(payload, text=" "), dict(payload, text="x" * 4001)):
                try:
                    candidate.classify(malformed)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Malformed request accepted")
            results.append(dict(dataset=measured["dataset"], condition=measured["condition"],
                manifest_sha256=candidate.identity, matching_recorded_responses=100, score_tolerance=1e-12,
                changed_contracts_reviewed=2, malformed_requests_rejected=6, teacher_calls=0))
            del candidate
            gc.collect()
    assert not any(name.split(".")[0] in {"sklearn", "scipy"} for name in sys.modules)
    print(json.dumps(dict(scope=scope, qualified=False,
        runtime_report_sha256=digest(report_path), os_network_denial_errno=denied,
        blocked_imports=["scipy", "sklearn"], results=results), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", action="store_true")
    if parser.parse_args().context:
        from context_runtime import ContextCandidate
        main(report_name="context-runtime.json", artifact_prefix="", candidate_class=ContextCandidate,
             scope="saved input-context System1 development artifacts; not qualification or new timing")
    else:
        main()
