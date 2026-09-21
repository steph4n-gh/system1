"""Audit the latest saved choices on observed tests; never mark them qualified."""
import time
STARTED = time.perf_counter()

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
from threadpoolctl import threadpool_limits

from audit_observed_regression import same
from boundary_development import BoundaryBaseline
from context_runtime import ContextCandidate
from develop import ROOT
from evaluate import deny_network_control, digest, gates, quality, split_rows
from observed_regression import frozen as previous_frozen
from polynomial_review import PolynomialBaseline
from polynomial_runtime import PolynomialCandidate

HERE = Path(__file__).resolve().parent
FREEZE = HERE / "latest-regression-freeze.json"
CANDIDATES = dict(clinc150=dict(system1="clinc150-system1-context", baseline="clinc150-baseline-positive-and-unsupported"),
                  banking77=dict(system1="polynomial-banking77-system1-quadratic", baseline="polynomial-banking77-baseline-quadratic"))
LOADERS = dict(clinc150=dict(system1=ContextCandidate, baseline=BoundaryBaseline),
               banking77=dict(system1=PolynomialCandidate, baseline=PolynomialBaseline))
REFERENCES = dict(clinc150=dict(system1="context-runtime.json", baseline="boundary-runtime.json"),
                  banking77=dict(system1="polynomial-runtime.json", baseline="polynomial-runtime.json"))


def freeze():
    previous, _ = previous_frozen(False)
    files = dict(previous["files"])
    extra = [HERE / name for name in ("LATEST_REGRESSION_PROTOCOL.md", "latest_regression.py", "audit_observed_regression.py",
        "context_runtime.py", "context_scores.py", "polynomial_runtime.py", "polynomial_scores.py", "polynomial_review.py",
        "boundary_development.py", "teach_boundaries.py", "observed-regression-freeze.json", "results/observed-regression.json",
        "results/context-runtime.json", "results/polynomial-runtime.json", "results/boundary-runtime.json")]
    selected = json.loads((HERE / "results/context-runtime.json").read_text())["comparisons"]
    for dataset, methods in CANDIDATES.items():
        for kind, name in methods.items():
            folder = HERE / "artifacts" / name
            expected = next(r["selected_development_candidate"]["candidate_sha256"] for r in selected
                            if (r["dataset"], r["kind"]) == (dataset, kind))
            assert digest(folder / "manifest.json") == expected
            extra.extend(path for path in folder.iterdir() if path.is_file())
            manifest = json.loads((folder / "manifest.json").read_text())
            if "parent_manifest_sha256" in manifest:
                parent = ROOT / manifest["base_artifact"]
                assert digest(parent / "manifest.json") == manifest["parent_manifest_sha256"]
                extra.extend(path for path in parent.iterdir() if path.is_file())
    for path in extra:
        files[str(path.relative_to(ROOT))] = digest(path)
    spec = dict(scope="latest choices, observed-test regression only", qualified=False,
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        prior_freeze_sha256=digest(HERE / "observed-regression-freeze.json"), criteria=previous["criteria"],
        candidates=CANDIDATES, references=REFERENCES, files=files)
    with FREEZE.open("x") as stream:
        stream.write(json.dumps(spec, indent=2) + "\n")
    print(f"Bound {len(files)} files; preflight then commit before original-test access")


def frozen(require_commit):
    previous, _ = previous_frozen(False)
    spec = json.loads(FREEZE.read_text())
    assert spec["qualified"] is False and spec["criteria"] == previous["criteria"]
    assert spec["candidates"] == CANDIDATES and spec["references"] == REFERENCES
    assert spec["prior_freeze_sha256"] == digest(HERE / "observed-regression-freeze.json")
    for name, expected in spec["files"].items():
        assert digest(ROOT / name) == expected, name
    revision = None
    if require_commit:
        name = str(FREEZE.relative_to(ROOT))
        revision = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", name], cwd=ROOT, text=True).strip()
        assert revision and subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT) == FREEZE.read_bytes(), "Commit the exact freeze first"
        for path, expected in spec["files"].items():
            raw = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)
            assert hashlib.sha256(raw).hexdigest() == expected, path
    return spec, revision


def evaluate(dataset, kind, development, rows, journal, preflight):
    folder = HERE / "artifacts" / CANDIDATES[dataset][kind]
    begin = time.perf_counter()
    candidate = LOADERS[dataset][kind](folder)
    load_ms = (time.perf_counter() - begin) * 1000
    payload = {key: candidate.manifest[key] for key in ("categories", "instructions")}
    references = None
    if preflight:
        report = json.loads((HERE / "results" / REFERENCES[dataset][kind]).read_text())
        references = next(r["outcomes"] for r in report["results"] if r["candidate_sha256"] == candidate.identity)
    for row in development[:100]:
        candidate.classify(dict(payload, text=row["prompt"]))
    outcomes = []
    for index, row in enumerate(rows):
        begin = time.perf_counter()
        try:
            value = candidate.classify(dict(payload, text=row["prompt"]))
        except Exception as exc:
            value = dict(candidate=candidate.identity, needsReview=True, category=None, suggestion=None, teacherCalls=0, error=type(exc).__name__)
        elapsed = (time.perf_counter() - begin) * 1000
        outcome = dict(index=index, group=row["group"], text=row["prompt"], truth=row["label"], latency_ms=elapsed, **value)
        journal.write(json.dumps(dict(dataset=dataset, kind=kind, **outcome)) + "\n")
        journal.flush()
        if references is not None:
            assert references[index]["group"] == row["group"]
            same(value, {k: v for k, v in references[index].items() if k not in ("group", "truth", "latency_ms")})
        outcomes.append(outcome)
    measured = quality(outcomes)
    times = [r["latency_ms"] for r in outcomes]
    latency = dict(requests=len(rows), warmups=100, p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)), max_ms=max(times))
    errors = sum("error" in r for r in outcomes)
    passed = gates(measured, latency["p95_ms"], errors)
    required = [folder] + ([candidate.parent_folder] if hasattr(candidate, "parent_folder") else [])
    result = dict(dataset=dataset, kind=kind, folder=folder.name, candidate_sha256=candidate.identity, manifest=candidate.manifest,
        metrics=measured, latency=latency, load_ms=load_ms, point_gates=passed, all_point_targets_pass=all(passed.values()),
        qualified=False, teacher_calls=sum(r["teacherCalls"] for r in outcomes), errors=errors, response_cache=False, receipts=False,
        artifact_bytes=sum(p.stat().st_size for directory in required for p in directory.iterdir() if p.is_file()),
        external_encoder_bytes=sum(candidate.manifest.get("encoder", {}).get(key, 0) for key in ("model_bytes", "tokenizer_bytes")),
        per_intent={label: quality([r for r in outcomes if r["truth"] == label]) for label in sorted({r["truth"] for r in outcomes})},
        accepted_errors=[r for r in outcomes if not r["needsReview"] and r["category"] != r["truth"]], outcomes=outcomes)
    print(json.dumps({key: result[key] for key in ("dataset", "kind", "metrics", "latency", "point_gates")}), flush=True)
    del candidate
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "preflight", "observed"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
        return
    if args.output is None:
        parser.error("--output is required")
    preflight = args.action == "preflight"
    spec, revision = frozen(not preflight)
    denied = deny_network_control()
    machine = dict(platform=platform.platform(), python=platform.python_version())
    if platform.system() == "Darwin":
        machine["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    report = dict(scope="development-only preflight" if preflight else "experiment 2h observed original tests; not qualification",
        qualified=False, independent_confirmation=False, freeze_commit=revision, freeze_sha256=digest(FREEZE), criteria=spec["criteria"],
        environment=machine, packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "onnxruntime", "tokenizers", "threadpoolctl")},
        process_import_and_freeze_preflight_ms=(time.perf_counter() - STARTED) * 1000, os_network_denial_errno=denied,
        teacher_calls=0, new_api_cost=0, all_system1_point_targets_pass=False, results=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream, args.output.with_suffix(".jsonl").open("x") as journal, threadpool_limits(limits=1):
        journal.write(json.dumps(dict(event="started", scope=report["scope"], freeze_commit=revision)) + "\n")
        journal.flush()
        for dataset in ("clinc150", "banking77"):
            development = split_rows(dataset, "development")
            rows = development[:16] if preflight else split_rows(dataset, "test")
            for kind in ("system1", "baseline"):
                report["results"].append(evaluate(dataset, kind, development, rows, journal, preflight))
        report["all_system1_point_targets_pass"] = all(r["all_point_targets_pass"] for r in report["results"] if r["kind"] == "system1")
        stream.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
