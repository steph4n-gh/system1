"""Frozen observed-test audit. Passing point gates never qualifies these candidates."""
import time
STARTED = time.perf_counter()

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
from threadpoolctl import threadpool_limits

from baseline_review import ReviewBaseline
from develop import ROOT
from evaluate import deny_network_control, digest, frozen_state, gates, quality, split_rows
from review_runtime import ReviewCandidate

HERE = Path(__file__).resolve().parent
FREEZE = HERE / "observed-regression-freeze.json"
CANDIDATES = dict(
    clinc150=dict(system1="clinc-review-consistent", baseline="clinc150-baseline-review"),
    banking77=dict(system1="banking-review", baseline="banking77-baseline-review"))


def freeze():
    original, _ = frozen_state(False)
    files = dict(original["files"])
    extra = [HERE / name for name in ("OBSERVED_REGRESSION_PROTOCOL.md", "observed_regression.py",
             "review_runtime.py", "baseline_review.py", "BASELINE_REVIEW_PROTOCOL.md",
             "REVIEW_TEACHING_PROTOCOL.md", "CONSISTENT_FEATURES_PROTOCOL.md", "results/official-test.json")]
    for methods in CANDIDATES.values():
        for name in methods.values():
            extra.extend(path for path in (HERE / "artifacts" / name).iterdir() if path.is_file())
    for path in extra:
        files[str(path.relative_to(ROOT))] = digest(path)
    result = dict(scope="observed-test regression only; never independent qualification", qualified=False,
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        original_freeze_sha256=digest(HERE / "freeze.json"), criteria=original["criteria"],
        candidates=CANDIDATES, files=files)
    with FREEZE.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(f"Created observed-test freeze binding {len(files)} files; commit after preflight")


def frozen(official):
    original, _ = frozen_state(False)
    spec = json.loads(FREEZE.read_text())
    assert spec["qualified"] is False and spec["criteria"] == original["criteria"]
    if spec["original_freeze_sha256"] != digest(HERE / "freeze.json"):
        raise ValueError("Original freeze changed")
    for name, expected in spec["files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen file changed: {name}")
    revision = None
    if official:
        relative = str(FREEZE.relative_to(ROOT))
        revision = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", relative], cwd=ROOT, text=True).strip()
        if not revision or subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=ROOT) != FREEZE.read_bytes():
            raise ValueError("Commit the exact freeze before observing these tests again")
        for name, expected in spec["files"].items():
            import hashlib
            raw = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"Source or artifact absent from freeze commit: {name}")
    return spec, revision


def reference_outcomes(dataset, kind):
    if kind == "baseline":
        report = "baseline-review-runtime.json"
    else:
        report = "consistent-review-runtime-development.json" if dataset == "clinc150" else "review-runtime-development.json"
    source = json.loads((HERE / "results" / report).read_text())
    return next(row["outcomes"] for row in source["results"] if row["dataset"] == dataset)


def evaluate(kind, dataset, folder, development, rows, journal, preflight):
    started = time.perf_counter()
    candidate = (ReviewCandidate if kind == "system1" else ReviewBaseline)(folder)
    load_ms = (time.perf_counter() - started) * 1000
    payload = {key: candidate.manifest[key] for key in ("categories", "instructions")}
    for row in development[:100]:
        candidate.classify(dict(payload, text=row["prompt"]))
    outcomes, timings = [], []
    references = reference_outcomes(dataset, kind) if preflight else None
    for index, row in enumerate(rows):
        started = time.perf_counter()
        try:
            value = candidate.classify(dict(payload, text=row["prompt"]))
        except Exception as exc:
            value = dict(needsReview=True, category=None, teacherCalls=0, error=type(exc).__name__)
        elapsed = (time.perf_counter() - started) * 1000
        outcome = dict(index=index, group=row["group"], text=row["prompt"], truth=row["label"], latency_ms=elapsed, **value)
        journal.write(json.dumps(dict(dataset=dataset, kind=kind, **outcome)) + "\n")
        journal.flush()
        if references is not None:
            assert row["group"] == references[index]["group"]
            assert all(value[key] == references[index][key] for key in value)
        outcomes.append(outcome)
        timings.append(elapsed)
    metrics = quality(outcomes)
    errors = sum("error" in row for row in outcomes)
    latency = dict(requests=len(rows), warmups=100, p50_ms=float(np.median(timings)),
                   p95_ms=float(np.percentile(timings, 95)), max_ms=float(max(timings)))
    passed = gates(metrics, latency["p95_ms"], errors)
    own_bytes = sum(path.stat().st_size for path in folder.iterdir() if path.is_file())
    base_bytes = sum(path.stat().st_size for path in candidate.base_folder.iterdir() if path.is_file()) if kind == "baseline" else 0
    result = dict(dataset=dataset, kind=kind, candidate_sha256=candidate.identity, manifest=candidate.manifest,
        load_ms=load_ms, metrics=metrics, latency=latency, point_gates=passed,
        all_point_targets_pass=all(passed.values()), qualified=False,
        errors=errors, teacher_calls=sum(row["teacherCalls"] for row in outcomes), response_cache=False, receipts=False,
        artifact_bytes=own_bytes + base_bytes, external_encoder_bytes=sum(candidate.manifest.get("encoder", {}).get(key, 0)
            for key in ("model_bytes", "tokenizer_bytes")),
        per_intent={label: quality([row for row in outcomes if row["truth"] == label]) for label in sorted({row["truth"] for row in outcomes})},
        accepted_errors=[row for row in outcomes if not row["needsReview"] and row["category"] != row["truth"]], outcomes=outcomes)
    print(json.dumps({key: result[key] for key in ("dataset", "kind", "metrics", "latency", "point_gates", "qualified")}), flush=True)
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
    report = dict(scope="development evaluator preflight" if preflight else "previously observed original tests; regression only",
        qualified=False, independent_confirmation=False, all_system1_point_targets_pass=False,
        freeze_commit=revision, freeze_sha256=digest(FREEZE), criteria=spec["criteria"],
        environment=machine, packages={name: importlib.metadata.version(name) for name in
            ("numpy", "scipy", "scikit-learn", "onnxruntime", "tokenizers", "threadpoolctl")},
        process_import_preflight_ms=(time.perf_counter() - STARTED) * 1000, os_network_denial_errno=denied, results=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream, args.output.with_suffix(".jsonl").open("x") as journal, threadpool_limits(limits=1):
        journal.write(json.dumps(dict(event="started", scope=report["scope"], qualified=False, freeze_commit=revision)) + "\n")
        journal.flush()
        for dataset in ("clinc150", "banking77"):
            development = split_rows(dataset, "development")
            rows = development[:16] if preflight else split_rows(dataset, "test")
            for kind in ("system1", "baseline"):
                folder = HERE / "artifacts" / spec["candidates"][dataset][kind]
                report["results"].append(evaluate(kind, dataset, folder, development, rows, journal, preflight))
        report["all_system1_point_targets_pass"] = all(row["all_point_targets_pass"] for row in report["results"] if row["kind"] == "system1")
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), qualified=False,
                         all_system1_point_targets_pass=report["all_system1_point_targets_pass"])), flush=True)


if __name__ == "__main__":
    main()
