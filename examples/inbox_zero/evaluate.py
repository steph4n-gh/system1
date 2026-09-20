#!/usr/bin/env python3
"""Freeze a skill, compare a simple baseline, and exercise Inbox Zero's actual adapter."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import secrets
import shutil
import socket
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
from teach import HERE, load_lessons, no_network, teach
from system1 import System1Engine, SystemOneCompiler, __version__
from system1.compat.typesafe import compute_wilson_score_lower
from system1.integrations.inbox_zero import CATEGORIES, CHOICE_KEY, InboxZeroClassifier, email_text


def summarize(rows):
    accepted = [r for r in rows if not r["needs_review"]]
    correct = sum(r["prediction"] in r["acceptable"] for r in rows)
    accepted_correct = sum(r["prediction"] in r["acceptable"] for r in accepted)
    accuracy = accepted_correct / len(accepted) if accepted else None
    coverage = len(accepted) / len(rows)
    def interval(successes, total):
        return [compute_wilson_score_lower(successes, total),
                1 - compute_wilson_score_lower(total - successes, total)] if total else None
    return {"cases": len(rows), "raw_correct": correct, "raw_accuracy": correct / len(rows),
            "accepted": len(accepted), "accepted_correct": accepted_correct,
            "accepted_errors": len(accepted) - accepted_correct, "accepted_accuracy": accuracy,
            "reviewed": len(rows) - len(accepted), "acceptance_rate": coverage,
            "accepted_accuracy_interval_95": interval(accepted_correct, len(accepted)),
            "meets_targets": accuracy is not None and accuracy >= .95 and coverage >= .8,
            "latency_median_ms": statistics.median(r["latency_ms"] for r in rows),
            "latency_p95_ms": float(np.quantile([r["latency_ms"] for r in rows], .95))}


def evaluate(args):
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = load_lessons(args.lessons)
    contract = json.loads(args.contract.read_text())
    evaluation = json.loads(args.evaluation.read_text())
    protocol_path = HERE / ("protocol_v2.json" if args.candidate == "tfidf" else "protocol.json")
    protocol = json.loads(protocol_path.read_text())
    if evaluation["upstream"]["commit"] != "2571ce4970aa5d024bb664a6739bd91b1172c367":
        raise ValueError("Evaluation must use the pinned upstream revision")
    cases = [{**row, "cohort": "upstream_regression"} for row in evaluation["cases"]]
    if args.candidate == "tfidf":
        fresh_path = HERE / "evaluation_v2.json"
        for path in (args.lessons, fresh_path):
            if hashlib.sha256(path.read_bytes()).hexdigest() != protocol["sha256"].get(path.name):
                raise ValueError("The frozen candidate dataset changed")
        fresh = json.loads(fresh_path.read_text())
        cases = [{**row, "cohort": "fresh_authored"} for row in fresh["cases"]] + cases
    seen = {" ".join(email_text(r["email"]).casefold().split()) for split in ("teach", "calibration") for r in data[split]}
    if any(" ".join(email_text(r["email"]).casefold().split()) in seen for r in cases):
        raise ValueError("Evaluation overlaps teaching/calibration")
    if len(cases) != (119 if args.candidate == "tfidf" else 35) or any(not set(r["acceptable"]) <= set(CATEGORIES) for r in cases):
        raise ValueError("Unexpected upstream evaluation")
    model_path = output / "email.s1m"
    questions = {CHOICE_KEY: {"type": "choice", "instructions": contract["question"], "criteria": contract["criteria"]}}
    requests = [{"id": row["id"], "payload": {"state": {"email": row["email"]}, "questions": questions}} for row in cases]
    with patch.object(socket.socket, "connect", no_network), patch.object(socket, "create_connection", no_network):
        skill, teaching = teach(args.lessons, args.contract, model_path,
                               features="tfidf" if args.candidate == "tfidf" else "hash")
        skill.use_cache = False
        before = System1Engine(skill.schema, model=skill, strict_mode=True, use_cache=False)
        start = time.perf_counter()
        restored = InboxZeroClassifier(model_path, enable_actions=True)
        load_ms = (time.perf_counter() - start) * 1000
        rows = []
        for row, request in zip(cases, requests):
            original = before.decide(email_text(row["email"]), alpha=.05, record_receipt=False)
            decision = restored.engine.decide(email_text(row["email"]), alpha=.05, record_receipt=False)
            assert original.values == decision.values and original.probabilities == decision.probabilities
            assert original.conformal_sets == decision.conformal_sets and original.is_ambiguous == decision.is_ambiguous
            start = time.perf_counter()
            result = restored.classify(request["payload"])
            elapsed = (time.perf_counter() - start) * 1000
            rows.append({"id": row["id"], "cohort": row["cohort"], "acceptable": row["acceptable"], "prediction": decision.values["category"],
                         "needs_review": result["needsReview"], "prediction_set": result["predictionSet"], "latency_ms": elapsed})
        compiler = SystemOneCompiler(skill.schema, dimension=2048, regularization=.1, backend="numpy")
        _, conformal = compiler._split_samples([(email_text(r["email"]), r["label"]) for r in data["calibration"]], None, .5)
        # Supply the exact vocabulary for v2; fit IDF on the same teaching inputs.
        vocabulary = skill.projector.to_config()["terms"] if args.candidate == "tfidf" else None
        baseline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, vocabulary=vocabulary),
                                 LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"))
        start = time.perf_counter()
        baseline.fit([email_text(r["email"]) for r in data["teach"]], [r["label"] for r in data["teach"]])
        if args.candidate == "tfidf":
            probes = [email_text(row["email"]) for row in data["calibration"]]
            np.testing.assert_allclose(skill.projector.project_batch(probes),
                                       baseline[0].transform(probes).toarray(), atol=1e-6, rtol=1e-6)
        labels = list(baseline.classes_)
        ps = baseline.predict_proba([text for text, _ in conformal])
        scores = sorted(1 - ps[i, labels.index(label)] for i, (_, label) in enumerate(conformal))
        rank = math.ceil((len(scores) + 1) * .95)
        quantile = scores[rank - 1] if rank <= len(scores) else 1.0
        baseline_teaching_ms = (time.perf_counter() - start) * 1000
        baseline_rows = []
        for row in cases:
            start = time.perf_counter()
            probabilities = baseline.predict_proba([email_text(row["email"])])[0]
            prediction = labels[int(np.argmax(probabilities))]
            prediction_set = [label for label, p in zip(labels, probabilities) if 1 - p <= quantile]
            baseline_rows.append({"id": row["id"], "cohort": row["cohort"], "acceptable": row["acceptable"], "prediction": prediction,
                                  "prediction_set": prediction_set, "needs_review": len(prediction_set) != 1,
                                  "latency_ms": (time.perf_counter() - start) * 1000})
    # Only this loopback endpoint can be fetched by the actual TypeScript provider.
    # The Python service also has outbound socket connections disabled.
    input_path, wire_path = output / "requests.json", output / "adapter-results.json"
    input_path.write_text(json.dumps(requests))
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = {**os.environ, "SYSTEM1_CLASSIFIER_TOKEN": secrets.token_urlsafe(32)}
    runner = "from unittest.mock import patch; import socket; from system1.integrations.inbox_zero import main; " \
             "patch.object(socket.socket, 'connect', side_effect=AssertionError('Outbound network blocked')).start(); main()"
    with (output / "service.log").open("w") as log:
        server = subprocess.Popen([sys.executable, "-c", runner, "--model", str(model_path), "--port", str(port), "--enable-actions"], env=env, stdout=log, stderr=log)
        try:
            for _ in range(100):
                if server.poll() is not None:
                    raise RuntimeError("Local service stopped; inspect service.log")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.1):
                        break
                except OSError:
                    time.sleep(.05)
            else:
                raise TimeoutError("Local service did not start")
            subprocess.run([shutil.which("pnpm"), "exec", "tsx", str(HERE / "adapter_check.mts"),
                            str(args.inbox_zero.resolve()), str(input_path), str(wire_path), f"http://127.0.0.1:{port}"],
                           cwd=args.inbox_zero, env=env, check=True, timeout=120, capture_output=True, text=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    wire = json.loads(wire_path.read_text())
    by_id = {row["id"]: row for row in rows}
    for response in wire["rows"]:
        expected, result = by_id[response["id"]], response["result"]
        assert result["needsReview"] == expected["needs_review"]
        assert result["teacherCalls"] == result["inputTokens"] == 0
        if not result["needsReview"]:
            assert result["answers"][CHOICE_KEY]["choice"] == expected["prediction"]
    timings = [r["latency_ms"] for r in wire["rows"]]
    report = {"protocol": protocol, "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              "lessons_sha256": hashlib.sha256(args.lessons.read_bytes()).hexdigest(), "upstream": evaluation["upstream"],
              "environment": {"system1": __version__, "python": platform.python_version(), "platform": platform.platform(),
                              "numpy": np.__version__, "sklearn": sklearn.__version__},
              "teacher_calls": 0, "external_network_calls": 0,
              "system1": {**teaching, "load_ms": load_ms, "reload_identical": True, "quality": summarize(rows), "predictions": rows},
              "tfidf_logistic": {"teaching_ms": baseline_teaching_ms, "quality": summarize(baseline_rows), "predictions": baseline_rows},
              "adapter_http": {"requests": wire["local_calls"], "responses_match_direct": True,
                               "first_request_ms": timings[0], "median_ms": statistics.median(timings),
                               "p95_ms": float(np.quantile(timings, .95)), "external_fetches": wire["external_fetches"]}}
    for key in ("system1", "tfidf_logistic"):
        report[key]["cohorts"] = {cohort: summarize([row for row in report[key]["predictions"] if row["cohort"] == cohort])
                                  for cohort in sorted({row["cohort"] for row in cases})}
    report["fresh_targets_met"] = (report["system1"]["cohorts"].get("fresh_authored", {}).get("meets_targets", False))
    report["all_cohort_targets_met"] = all(q["meets_targets"] for q in report["system1"]["cohorts"].values())
    report["rollout_ready"] = False
    report["rollout_limit"] = "Synthetic/maintainer-authored evidence only; validate on independent representative mailbox data before automation. Inspect each cohort, not just combined metrics."
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(output / "report.json"), "system1": report["system1"]["quality"],
                      "cohorts": report["system1"]["cohorts"],
                      "baseline": report["tfidf_logistic"]["cohorts"], "adapter_http": report["adapter_http"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("initial", "tfidf"), default="initial")
    parser.add_argument("--lessons", type=Path)
    parser.add_argument("--contract", type=Path, default=Path(".system1/inbox-zero/sources/contract.json"))
    parser.add_argument("--evaluation", type=Path, default=Path(".system1/inbox-zero/sources/upstream-evaluation.json"))
    parser.add_argument("--inbox-zero", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.lessons = args.lessons or HERE / ("lessons_v2.json" if args.candidate == "tfidf" else "lessons.json")
    args.output = args.output or Path(".system1/inbox-zero") / ("evaluation-v2" if args.candidate == "tfidf" else "evaluation")
    evaluate(args)
