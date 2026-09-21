"""Evaluate frozen artifacts once; no fitting or policy selection occurs here.

Use --preflight on development inputs before committing the freeze. --official
requires every frozen source/artifact to match committed Git objects and OS
network denial. Reports/journals are created exclusively, never overwritten.
Reproduction on the same tests is a replication, not fresh confirmation.
"""
import time
STARTED = time.perf_counter()

import argparse
import errno
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = ROOT / ".system1/n8n-gauntlet"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def proportion(numerator, denominator):
    if not denominator:
        return dict(numerator=numerator, denominator=0, fraction=None, wilson95=None)
    p = numerator / denominator
    z = 1.959963984540054
    divisor = 1 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / divisor
    half = z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator ** 2)) / divisor
    return dict(numerator=numerator, denominator=denominator, fraction=p,
                wilson95=[max(0., center - half), min(1., center + half)])


def quality(outcomes):
    supported = [r for r in outcomes if r["truth"] != "oos"]
    accepted = [r for r in supported if not r["needsReview"]]
    unfamiliar = [r for r in outcomes if r["truth"] == "oos"]
    return dict(supported_coverage=proportion(len(accepted), len(supported)),
                accepted_accuracy=proportion(sum(r["category"] == r["truth"] for r in accepted), len(accepted)),
                raw_supported_accuracy=proportion(sum(r.get("suggestion") == r["truth"] for r in supported), len(supported)),
                oos_false_acceptance=proportion(sum(not r["needsReview"] for r in unfamiliar), len(unfamiliar)))


def gates(metrics, p95_ms, errors):
    coverage = metrics["supported_coverage"]["fraction"]
    accuracy = metrics["accepted_accuracy"]["fraction"]
    unfamiliar = metrics["oos_false_acceptance"]["fraction"]
    return dict(coverage_at_least_80=coverage is not None and coverage >= .8,
                accepted_accuracy_at_least_99=accuracy is not None and accuracy >= .99,
                oos_false_acceptance_at_most_1=unfamiliar is None or unfamiliar <= .01,
                adapter_p95_below_5_ms=p95_ms < 5, no_runtime_errors=errors == 0)


def deny_network_control():
    with socket.socket() as connection:
        try:
            connection.connect(("127.0.0.1", 9))
        except OSError as exc:
            if exc.errno not in (errno.EPERM, errno.EACCES):
                raise RuntimeError("Require OS network denial, not a closed port") from None
            return exc.errno
        raise RuntimeError("OS network access is enabled")


def frozen_state(official):
    path = HERE / "freeze.json"
    freeze = json.loads(path.read_text())
    for name, expected in freeze["files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen file changed: {name}")
    if official:
        # The immutable Git commit containing this exact freeze proves that the
        # evaluation recipe existed before test inspection. Later report commits
        # may be present when somebody reproduces the experiment.
        relative = str(path.relative_to(ROOT))
        commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", relative], cwd=ROOT, text=True).strip()
        if not commit:
            raise ValueError("Freeze must be committed before evaluation")
        committed = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        if committed != path.read_bytes():
            raise ValueError("Freeze differs from committed version")
        for name, expected in freeze["files"].items():
            raw = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"Frozen file not in freeze commit: {name}")
    else:
        commit = None
    return freeze, commit


def split_rows(dataset, split):
    manifest = json.loads((HERE / "manifest.json").read_text())
    raw = (OUTPUT / f"{dataset}-{split}.json").read_bytes()
    expected = manifest["datasets"][dataset]["splits"][split]
    if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError(f"Changed {dataset} {split}")
    rows = json.loads(raw)
    if len(rows) != expected["rows"]:
        raise ValueError("Changed split size")
    return rows


def evaluate_one(kind, dataset, folder, development, rows, journal):
    import numpy as np
    from runtime_probe import LocalCandidate
    from baseline import Baseline

    started = time.perf_counter()
    candidate = (LocalCandidate if kind == "system1" else Baseline)(folder)
    load_ms = (time.perf_counter() - started) * 1000
    payload = {key: candidate.manifest[key] for key in ("categories", "instructions")}
    for row in development[:100]:
        candidate.classify(dict(payload, text=row["prompt"]))
    outcomes, timings = [], []
    for index, row in enumerate(rows):
        started = time.perf_counter()
        try:
            result = candidate.classify(dict(payload, text=row["prompt"]))
        except Exception as exc:
            result = dict(needsReview=True, category=None, teacherCalls=0,
                          error=type(exc).__name__, detail=str(exc))
        elapsed = (time.perf_counter() - started) * 1000
        timings.append(elapsed)
        outcome = dict(index=index, group=row["group"], text=row["prompt"], truth=row["label"], latency_ms=elapsed, **result)
        outcomes.append(outcome)
        journal.write(json.dumps(dict(dataset=dataset, kind=kind, **outcome)) + "\n")
        journal.flush()
    metrics = quality(outcomes)
    latency = dict(requests=len(timings), warmups=100, p50_ms=float(np.median(timings)),
                   p95_ms=float(np.percentile(timings, 95)), max_ms=float(max(timings)))
    errors = sum("error" in row for row in outcomes)
    passed = gates(metrics, latency["p95_ms"], errors)
    result = dict(dataset=dataset, kind=kind, candidate_sha256=candidate.identity, manifest=candidate.manifest,
                  load_ms=load_ms, latency=latency, metrics=metrics, gates=passed, all_targets_pass=all(passed.values()),
                  errors=errors, teacher_calls=sum(row["teacherCalls"] for row in outcomes), response_cache=False,
                  receipts=False, artifact_bytes=sum(path.stat().st_size for path in folder.iterdir() if path.is_file()),
                  external_encoder_bytes=sum(candidate.manifest.get("encoder", {}).get(key, 0) for key in ("model_bytes", "tokenizer_bytes")),
                  per_intent={label: quality([row for row in outcomes if row["truth"] == label])
                              for label in sorted({r["truth"] for r in outcomes})},
                  accepted_errors=[r for r in outcomes if not r["needsReview"] and r["category"] != r["truth"]],
                  outcomes=outcomes)
    print(json.dumps({key: result[key] for key in ("dataset", "kind", "latency", "metrics", "gates", "all_targets_pass")}), flush=True)
    del candidate
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="Only development inputs; never opens tests")
    mode.add_argument("--official", action="store_true", help="Evaluate committed frozen artifacts on all official tests")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze, commit = frozen_state(args.official)
    denied = deny_network_control()
    # Imports complete before timing individual loads/decisions.
    import numpy
    import baseline
    import runtime_probe
    from threadpoolctl import threadpool_limits
    startup_ms = (time.perf_counter() - STARTED) * 1000
    packages = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "scikit-learn", "onnxruntime", "tokenizers", "threadpoolctl")}
    machine = dict(platform=platform.platform(), python=platform.python_version(), machine=platform.machine())
    if sys.platform == "darwin":
        machine["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    report = dict(scope="official test evaluation" if args.official else "development-only evaluator preflight",
                  freeze_commit=commit, freeze_sha256=digest(HERE / "freeze.json"),
                  environment=machine, packages=packages, process_import_preflight_ms=startup_ms,
                  os_network_denial_errno=denied, criteria=freeze["criteria"],
                  teaching=freeze["teaching"], results=[], all_system1_targets_pass=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    journal_path = args.output.with_suffix(".jsonl")
    # Exclusive create BEFORE any test read; interrupted runs keep their journal.
    with args.output.open("x") as report_stream, journal_path.open("x") as journal, threadpool_limits(limits=1):
        journal.write(json.dumps(dict(event="started", scope=report["scope"], freeze_commit=commit)) + "\n")
        journal.flush()
        for dataset in ("clinc150", "banking77"):
            development = split_rows(dataset, "development")
            rows = split_rows(dataset, "test") if args.official else development
            for kind in ("system1", "baseline"):
                folder = ROOT / freeze["candidates"][dataset][kind]
                report["results"].append(evaluate_one(kind, dataset, folder, development, rows, journal))
        report["all_system1_targets_pass"] = all(r["all_targets_pass"] for r in report["results"] if r["kind"] == "system1")
        report_stream.write(json.dumps(report, indent=2) + "\n")
        report_stream.flush()
    print(json.dumps(dict(report=str(args.output), all_system1_targets_pass=report["all_system1_targets_pass"])), flush=True)


if __name__ == "__main__":
    main()
