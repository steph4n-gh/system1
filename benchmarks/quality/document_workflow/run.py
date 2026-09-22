#!/usr/bin/env python3
"""One frozen, offline correction-workflow experiment on real BBC articles."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from system1 import ChoiceField, DecisionSchema, TeachingSession, __version__
from system1.compiler import CompiledSystemOneModel
from system1.engine import SystemOneEngine

HERE = Path(__file__).resolve().parent
URL = "https://raw.githubusercontent.com/codehax41/BBC-Text-Classification/9a64d059d04fe11489692d7214ef4561e30c4673/bbc-text.csv"
SOURCE_SHA = "fdaee0f7451cd8db2709d00e992886fe1c387ee332b8c7b7c1554ed3d3e3382e"
LABELS = ["business", "entertainment", "politics", "sport", "tech"]
SEED = 20260922


class NewsRoute(DecisionSchema):
    folder = ChoiceField(options=LABELS)


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_sha(path):
    return sha(path.read_bytes()) if path.exists() else None


def prepare(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        source = list(csv.DictReader(stream))
    assert len(source) == 2225 and sorted({r["category"] for r in source}) == LABELS
    rows, clipped = [], 0
    for row in source:
        normalized = " ".join(row["text"].split())
        clipped += len(normalized) > 8192
        text = normalized[:8192]
        rows.append({"id": sha(text.casefold()), "input": text, "label": row["category"]})
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        parents[find(a)] = find(b)

    exact, inverted, sizes = {}, defaultdict(list), []
    near_edges = 0
    for i, row in enumerate(rows):
        if row["id"] in exact:
            union(i, exact[row["id"]])
        exact[row["id"]] = i
        tokens = re.findall(r"\w+", row["input"].casefold())
        shingles = set(zip(*(tokens[offset:] for offset in range(5))))
        assert shingles, "Corpus document has fewer than five words"
        sizes.append(len(shingles))
        shared = Counter(itertools.chain.from_iterable(inverted[s] for s in shingles))
        for j, overlap in shared.items():
            if overlap / min(sizes[i], sizes[j]) >= .8 or overlap / (sizes[i] + sizes[j] - overlap) >= .5:
                union(i, j)
                near_edges += 1
        for shingle in shingles:
            inverted[shingle].append(i)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    retained, conflicts = [], []
    manifest_groups = []
    for group in groups.values():
        group = sorted(group, key=lambda r: r["id"])
        conflict = len({r["label"] for r in group}) != 1
        manifest_groups.append({"representative": None if conflict else group[0]["id"],
                                "members": [r["id"] for r in group]})
        (conflicts if conflict else retained).append(group if conflict else group[0])
    splits = {name: [] for name in ("teach", "calibrate", "evaluate", "holdout", "feedback")}
    for label in LABELS:
        ordered = sorted((r for r in retained if r["label"] == label),
                         key=lambda r: sha(f"{SEED}:split:{r['id']}"))
        assert len(ordered) > 180, (label, len(ordered))
        start = 0
        for name, count in (("teach", 20), ("calibrate", 40), ("evaluate", 40), ("holdout", 80)):
            splits[name].extend(ordered[start:start + count])
            start += count
        splits["feedback"].extend(ordered[start:])
    all_ids = [r["id"] for rows_ in splits.values() for r in rows_]
    assert len(all_ids) == len(set(all_ids)) == len(retained)
    summary = {"source_count": len(source), "clipped_documents": clipped,
               "exact_duplicate_rows": len(rows) - len(exact), "near_duplicate_edges": near_edges,
               "components": len(groups), "conflicting_components": len(conflicts),
               "retained": len(retained), "discarded": len(rows) - len(retained),
               "counts": {name: len(rows_) for name, rows_ in splits.items()}}
    manifest = {"preparation": summary, "groups": sorted(manifest_groups, key=lambda g: g["members"]),
                "splits": {name: [{"id": r["id"], "label": r["label"]} for r in rows_]
                           for name, rows_ in splits.items()}}
    return splits, manifest


def engine(path):
    model = CompiledSystemOneModel.load(path)
    model.use_cache = False
    return SystemOneEngine(model.schema, model=model, strict_mode=True, use_cache=False)


def outcome(model, row):
    start = time.perf_counter_ns()
    result = model.decide(row["input"], alpha=.05, record_receipt=False)
    elapsed = (time.perf_counter_ns() - start) / 1e6
    return {"id": row["id"], "label": row["label"], "predicted": result.values["folder"],
            "correct": result.values["folder"] == row["label"], "review": bool(result.is_ambiguous),
            "prediction_set": result.conformal_sets["folder"], "latency_ms": elapsed}


def wilson(correct, count):
    if not count:
        return None
    p, z = correct / count, 1.959963984540054
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count)) / denominator
    margin = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return [center - margin, center + margin]


def metrics(cases):
    count = len(cases)
    correct = sum(r["correct"] for r in cases)
    accepted = [r for r in cases if not r["review"]]
    ac = sum(r["correct"] for r in accepted)
    return {"count": count, "raw_correct": correct, "raw_accuracy": correct/count,
            "raw_accuracy_wilson95": wilson(correct, count), "accepted": len(accepted),
            "accepted_correct": ac, "accepted_errors": len(accepted)-ac,
            "accepted_accuracy": ac/len(accepted) if accepted else None,
            "accepted_accuracy_wilson95": wilson(ac, len(accepted)),
            "coverage": len(accepted)/count, "review": count-len(accepted),
            "latency_ms_p50": float(np.median([r["latency_ms"] for r in cases])),
            "latency_ms_p95": float(np.percentile([r["latency_ms"] for r in cases], 95))}


def compare(before, after):
    assert [r["id"] for r in before] == [r["id"] for r in after]
    result = {}
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(before), size=(5000, len(before)))
    for name, predicate in (("raw", lambda r: r["correct"]),
                            ("accepted_correct", lambda r: r["correct"] and not r["review"])):
        left = np.array([predicate(r) for r in before], dtype=int)
        right = np.array([predicate(r) for r in after], dtype=int)
        delta = right-left
        result[name] = {"gained": int(np.sum(delta == 1)), "lost": int(np.sum(delta == -1)),
                        "delta_fraction": float(delta.mean()),
                        "paired_bootstrap95": np.quantile(delta[samples].mean(axis=1), [.025, .975]).tolist()}
    return result


def assess_adopt(session):
    before = file_sha(session.current_path)
    start = time.perf_counter()
    report = session.assess()
    assessment_seconds = time.perf_counter()-start
    assert file_sha(session.current_path) == before, "Assessment replaced incumbent"
    start = time.perf_counter()
    if report["passed"]:
        session.adopt()
        assert file_sha(session.current_path) == file_sha(session.candidate_path)
    else:
        try:
            session.adopt()
        except ValueError as error:
            assert "did not pass" in str(error)
        else:
            raise AssertionError("Failed candidate was adopted")
        assert file_sha(session.current_path) == before
    elapsed = time.perf_counter()-start
    reopened = TeachingSession(session.directory)
    assert reopened.snapshot()["current"] == file_sha(session.current_path)
    parity_count = 0
    if session.engine is not None:
        for row in session.records[:10]:
            first, second = session.predict(row["input"]), reopened.predict(row["input"])
            assert (first.values, first.conformal_sets, first.probabilities, first.is_ambiguous) == (
                second.values, second.conformal_sets, second.probabilities, second.is_ambiguous)
            parity_count += 1
    public = {key: value for key, value in report.items() if key != "cases"}
    return {"assessment": public, "adopted": report["passed"],
            "current_sha256": file_sha(session.current_path),
            "candidate_sha256": file_sha(session.candidate_path),
            "artifact_bytes": session.candidate_path.stat().st_size,
            "assessment_seconds": assessment_seconds, "adoption_check_seconds": elapsed,
            "incumbent_preserved_during_assessment": True,
            "reopened_prediction_parity": True if parity_count else None,
            "reopened_prediction_checks": parity_count}


def run(output, splits, provenance):
    initial = TeachingSession(output / "initial", NewsRoute)
    initial.record_many([{ "input": r["input"], "label": r["label"], "split": split,
                          "source": "published BBC category"}
                         for split in ("teach", "calibrate", "evaluate") for r in splits[split]])
    stages = {"initial": assess_adopt(initial)}
    print("Initial adoption checks:", json.dumps(stages["initial"]["assessment"]["candidate"]), flush=True)
    base_model = engine(initial.candidate_path)
    pool = sorted(splits["feedback"], key=lambda r: sha(f"{SEED}:feedback:{r['id']}"))[:400]
    assert len(pool) == 400
    predictions = [outcome(base_model, row) for row in pool]
    selected = [(row, result) for row, result in zip(pool, predictions)
                if not result["correct"] or result["review"]][:100]
    assert selected, "No corrections/reviews available under this frozen protocol"
    feedback = {"labels_inspected": len(pool), "lessons_added": len(selected),
                "wrong_predictions_corrected": sum(not result["correct"] for _, result in selected),
                "reviewed_cases_taught": sum(result["review"] for _, result in selected),
                "targeted_ids": [r["id"] for r, _ in selected],
                "ordinary_ids": [r["id"] for r in pool[:len(selected)]],
                "initial_feedback_outcomes": predictions}
    for name, rows in (("corrected", [r for r, _ in selected]), ("ordinary", pool[:len(selected)])):
        shutil.copytree(initial.directory, output / name)
        session = TeachingSession(output / name)
        before = file_sha(session.current_path)
        session.record_many([{"input": r["input"], "label": r["label"], "split": "teach",
                              "source": "published category supplied as simulated human feedback"} for r in rows])
        assert file_sha(session.current_path) == before
        try:
            session.adopt()
        except ValueError as error:
            assert "stale" in str(error)
        else:
            raise AssertionError("An edited session adopted a stale candidate")
        # Repeating an identical correction must not create a duplicate lesson.
        record = next(r for r in session.records if r["input"] == rows[0]["input"])
        prior_state = session.session_path.read_bytes()
        session.record_many([record])
        assert session.session_path.read_bytes() == prior_state
        stages[name] = {**assess_adopt(session), "stale_adoption_blocked": True,
                        "incumbent_preserved_during_edits": True, "repeated_correction_deduplicated": True}
        print(name, "adoption:", stages[name]["adopted"], stages[name]["assessment"]["reasons"], flush=True)
    # No holdout predictions have been made before this durable freeze record.
    frozen = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "provenance": provenance, "stages": stages,
              "feedback": {k: v for k, v in feedback.items() if k != "initial_feedback_outcomes"}}
    save(output / "frozen-before-holdout.json", frozen)
    holdout_ids = {r["id"] for r in splits["holdout"]}
    for name in stages:
        session = TeachingSession(output / name)
        assert not holdout_ids.intersection(sha(r["input"].casefold()) for r in session.records)
        assert file_sha(session.candidate_path) == stages[name]["candidate_sha256"]
    cases, holdout = {}, {}
    # First and only scoring pass, after every development choice is fixed.
    models = {name: engine(output / name / "candidate.s1m") for name in stages}
    for name in stages:
        cases[name] = []
    for row in splits["holdout"]:
        for name, model in models.items():
            cases[name].append(outcome(model, row))
    for name, rows in cases.items():
        summary = metrics(rows)
        bound = summary["accepted_accuracy_wilson95"]
        holdout[name] = {**summary,
                         "per_class": {label: metrics([r for r in rows if r["label"] == label]) for label in LABELS},
                         "point_target_met": bool(summary["accepted_accuracy"] is not None and
                                                  summary["accepted_accuracy"] >= .95 and summary["coverage"] >= .8),
                         "accepted_accuracy_bound_and_coverage_met": bool(bound and bound[0] >= .95 and summary["coverage"] >= .8)}
    report = {"provenance": provenance, "workflow": stages, "feedback": feedback,
              "holdout": holdout, "paired": {
                  "corrected_vs_initial": compare(cases["initial"], cases["corrected"]),
                  "ordinary_vs_initial": compare(cases["initial"], cases["ordinary"]),
                  "corrected_vs_ordinary": compare(cases["ordinary"], cases["corrected"])},
              "holdout_cases": cases,
              "actual_serving_artifacts": {name: stage["current_sha256"] for name, stage in stages.items()},
              "network_blocked_during_workflow": True, "teacher_api_calls": 0,
              "frozen_before_holdout_sha256": file_sha(output / "frozen-before-holdout.json")}
    save(output / "report.json", report)
    print(json.dumps({"holdout": {name: {k: v for k, v in row.items() if k != "per_class"}
                                    for name, row in holdout.items()}, "paired": report["paired"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".system1/document-workflow")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise SystemExit("This output already contains a run; use a new output directory for reproduction")
    data = output / "bbc-text.csv"
    if not data.exists() and args.download:
        with urlopen(URL, timeout=60) as response:
            data.write_bytes(response.read())
    if not data.exists() or file_sha(data) != SOURCE_SHA:
        raise SystemExit("Missing or changed dataset; use --download with an empty output directory")
    splits, manifest = prepare(data)
    save(output / "manifest.json", manifest)
    source_hashes = {str(p.relative_to(ROOT)): file_sha(p) for p in sorted((ROOT / "src/system1").rglob("*.py"))}
    provenance = {"prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                  "source_url": URL, "source_sha256": SOURCE_SHA,
                  "protocol_sha256": file_sha(HERE / "PROTOCOL.md"), "runner_sha256": file_sha(Path(__file__)),
                  "manifest_sha256": file_sha(output / "manifest.json"), "seed": SEED,
                  "preparation": manifest["preparation"], "system1_version": __version__,
                  "unreleased_worktree": True, "runtime_source_hashes": source_hashes,
                  "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}
    save(output / "prepared-before-predictions.json", provenance)
    print("Preparation:", json.dumps(manifest["preparation"]), flush=True)
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Network is disabled during evaluation")):
        run(output, splits, provenance)


if __name__ == "__main__":
    main()
