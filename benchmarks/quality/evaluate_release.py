#!/usr/bin/env python3
"""Check current manual skills alongside the recorded 1.0 Jev baseline.

Uses the recorded Jev responses by default. --refresh-teacher makes paid HTTP
requests for missing responses using TYPESAFE_API_KEY. No key enters the report.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]
from _teaching_demo import load_cases, quality_metrics, run_example
from support_triage import SupportTicketSchema
from model_routing import ModelRouterSchema
from agent_guard import OperationTriageSchema
from system1.compat.typesafe import Client, call_real_typesafe_api
from system1.compiler import SystemOneCompiler

SCHEMAS = {"support_triage": SupportTicketSchema, "model_routing": ModelRouterSchema,
           "agent_guard": OperationTriageSchema}


def observation_key(prompt, questions):
    return hashlib.sha256(json.dumps({"state": prompt, "questions": questions}, sort_keys=True).encode()).hexdigest()


def questions_for(schema, data):
    name, definition = next(iter(schema().fields.items()))
    return {name: {"type": "choice", "criteria": definition.descriptions,
                   "instructions": json.dumps(data["policy"])}}


def refresh_teacher(cache_path, records, missing):
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise ValueError("--refresh-teacher requires TYPESAFE_API_KEY")
    def fetch(item):
        digest, prompt, questions = item
        response, latency, _ = call_real_typesafe_api(
            prompt, questions, api_key=key, zero_egress=False, fallback_baseline=False, timeout=30,
        )
        return digest, {"response": {k: response[k] for k in ("model", "answers", "usage")},
                        "latency_ms": latency, "recorded_at": time.time()}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for count, (digest, record) in enumerate(pool.map(fetch, missing), 1):
            records[digest] = record
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"source": "Actual successful Jev HTTPS responses on authored demonstration inputs.",
                                             "records": records}, indent=2) + "\n")
            if count % 25 == 0:
                print(f"Recorded {count}/{len(missing)} missing Jev responses", flush=True)


def evaluate(output_dir, cache_path, refresh=False):
    datasets = {name: load_cases(ROOT / "examples" / "teaching" / f"{name}.json", include_quality_round=False) for name in SCHEMAS}
    records = json.loads(cache_path.read_text())["records"] if cache_path.exists() else {}
    missing = []
    for name, schema in SCHEMAS.items():
        questions = questions_for(schema, datasets[name])
        for split in ("teach", "calibration", "evaluate"):
            for row in datasets[name][split]:
                digest = observation_key(row["prompt"], questions)
                if digest not in records:
                    missing.append((digest, row["prompt"], questions))
    if missing:
        if not refresh:
            raise ValueError(f"Missing {len(missing)} recorded Jev responses; use --refresh-teacher to collect them")
        refresh_teacher(cache_path, records, missing)

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"data": "Authored demonstration cases; current manual teaching is checked alongside the recorded 1.0 Jev baseline. Not customer data or an independent benchmark.",
              "manual_teaching_source": "Current example JSON files, including locally authored quality-round lessons. Their results need not match the recorded Jev baseline.",
              "teacher_source": "Recorded successful live Jev HTTP responses for the 1.0 baseline. New quality-round lessons are locally authored and excluded from this teacher comparison.",
              "targets": {"accepted_accuracy": .95, "local_acceptance": .8},
              "workloads": {}, "teacher_cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest()}
    for name, schema in SCHEMAS.items():
        data = datasets[name]
        questions = questions_for(schema, data)
        field = next(iter(questions))
        selected = {row["prompt"]: records[observation_key(row["prompt"], questions)]
                    for split in ("teach", "calibration", "evaluate") for row in data[split]}
        def teacher_label(row):
            return selected[row["prompt"]]["response"]["answers"][field]["value"]
        examples = {field: [(row["prompt"], teacher_label(row)) for row in data["teach"]]}
        calibration = {field: [(row["prompt"], teacher_label(row)) for row in data["calibration"]]}
        started = time.perf_counter()
        skill = SystemOneCompiler(schema=questions, dimension=2048, regularization=.1).compile(
            examples, calibration_exemplars=calibration, augment=False,
        )
        teaching_ms = (time.perf_counter() - started) * 1000
        skill.metadata.update(typesafe_strict_mode=True, teacher_cache_sha256=report["teacher_cache_sha256"])
        skill_path = output_dir / f"{name}-jev.s1m"
        skill.save(skill_path)
        before, after = Client(compiled_model=skill), Client(model_path=skill_path)
        rows = []
        def disconnected(*args, **kwargs):
            raise AssertionError("Local evaluation contacted a network or teacher")
        before.call_real_api = after.call_real_api = disconnected
        with patch.object(socket.socket, "connect", disconnected), patch.object(socket, "create_connection", disconnected):
            for row in data["evaluate"]:
                response = before.systemone(row["prompt"], questions, record_receipt=False)
                reloaded = after.systemone(row["prompt"], questions, record_receipt=False)
                assert response.answers == reloaded.answers
                assert response.is_ambiguous == reloaded.is_ambiguous
                assert response.get("abstain", False) == reloaded.get("abstain", False)
                assert response.local_execution and reloaded.local_execution
                rows.append({**row, "prediction": response.answers[field].value,
                             "teacher_prediction": teacher_label(row),
                             "needs_review": response.is_ambiguous or response.get("abstain", False),
                             "local_latency_ms": response.latency_ms,
                             "teacher_latency_ms": selected[row["prompt"]]["latency_ms"]})
            with contextlib.redirect_stdout(io.StringIO()):
                manual = run_example(schema, name, ["--output-dir", str(output_dir / "manual")])
        metrics = quality_metrics(rows)
        agreement_rows = [{**row, "label": row["teacher_prediction"]} for row in rows]
        teacher_agreement = quality_metrics(agreement_rows)
        workload = {"manual_teaching": manual["quality"], "observed_teaching": metrics,
                    "local_agreement_with_jev": teacher_agreement,
                    "teacher_expected_label_accuracy": sum(row["teacher_prediction"] == row["label"] for row in rows) / len(rows),
                    "teaching_ms": teaching_ms, "skill_bytes": skill_path.stat().st_size,
                    "local_latency_median_ms": statistics.median(row["local_latency_ms"] for row in rows),
                    "jev_latency_median_ms": statistics.median(row["teacher_latency_ms"] for row in rows),
                    "teacher_models": sorted({r["response"]["model"] for r in selected.values()}),
                    "successful_teacher_requests": len(selected),
                    "teacher_usage": {key: sum(r["response"]["usage"].get(key, 0) for r in selected.values())
                                      for key in ("input_tokens", "output_tokens", "total_tokens")},
                    "observed_teacher_label_disagreements": sum(teacher_label(row) != row["label"] for split in ("teach", "calibration") for row in data[split]),
                    "teacher_calls_during_local_evaluation": 0, "reload_identical": True,
                    "evaluation_cohorts": {cohort: quality_metrics([row for row in rows if row.get("evaluation_cohort") == cohort])
                                           for cohort in sorted({row["evaluation_cohort"] for row in rows})},
                    "predictions": rows}
        report["workloads"][name] = workload
        manual_metrics = manual["quality"]
        print(f"{name} current manual: accepted {manual_metrics['accepted']}/{manual_metrics['cases']}; correctness {manual_metrics['accepted_accuracy']:.1%}")
        print(f"{name} recorded Jev 1.0: accepted {metrics['accepted']}/{metrics['cases']}; correctness {metrics['accepted_accuracy']:.1%}; Jev agreement {teacher_agreement['accepted_accuracy']:.1%}")
    report["passed"] = all(w["manual_teaching"]["meets_routing_targets"] and w["observed_teaching"]["meets_routing_targets"]
                           and w["local_agreement_with_jev"]["meets_routing_targets"] for w in report["workloads"].values())
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".system1" / "release-quality")
    parser.add_argument("--teacher-cache", type=Path, default=ROOT / "benchmarks" / "quality" / "results" / "jev_observations.json")
    parser.add_argument("--refresh-teacher", action="store_true")
    args = parser.parse_args()
    report = evaluate(args.output_dir, args.teacher_cache, args.refresh_teacher)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
