"""Separate fresh-process load probe after report recovery; no decisions or fit."""
import time
STARTED = time.perf_counter()

import json
import platform
from pathlib import Path

from threadpoolctl import threadpool_limits
from boundary_development import BoundaryBaseline
from review_runtime import ReviewCandidate
from evaluate import deny_network_control, digest

HERE = Path(__file__).resolve().parent
denied = deny_network_control()
import_ms = (time.perf_counter() - STARTED) * 1000
source = json.loads((HERE / "results/direct-oos-runtime.json").read_text())
results = []
with threadpool_limits(limits=1):
    for row in source["results"]:
        folder = HERE.parents[2] / ".system1/n8n-gauntlet/direct-oos-development" / row["folder"]
        started = time.perf_counter()
        candidate = (ReviewCandidate if row["kind"] == "system1" else BoundaryBaseline)(folder)
        elapsed = (time.perf_counter() - started) * 1000
        assert candidate.identity == row["candidate_sha256"]
        results.append(dict(kind=row["kind"], condition=row["condition"], candidate_sha256=candidate.identity, load_ms=elapsed))
        del candidate
print(json.dumps(dict(scope="separate fresh-process startup probe; not original lost timings or quality measurement",
    qualified=False, python=platform.python_version(), platform=platform.platform(), os_network_denial_errno=denied,
    teacher_calls=0, decision_requests=0, import_and_network_preflight_ms=import_ms,
    source_runtime_sha256=digest(HERE / "results/direct-oos-runtime.json"), results=results), indent=2))
