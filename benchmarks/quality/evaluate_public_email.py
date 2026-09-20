#!/usr/bin/env python3
"""Download pinned public mail, teach one local skill, and score later collections."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from email import policy
from email.parser import BytesParser
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import platform
import re
import socket
import statistics
import sys
import tarfile
import time
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]
from _teaching_demo import quality_metrics
from system1 import (ChoiceField, CompiledSystemOneModel, DecisionSchema,
                     System1Engine, SystemOneCompiler, TfidfProjector, __version__)

HERE = Path(__file__).resolve().parent / "public_email"


class SpamDecision(DecisionSchema):
    category = ChoiceField(options=["ham", "spam"])


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1
        self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)
        self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def canonical(value):
    return " ".join(value.casefold().split())


def parse_message(raw):
    message = BytesParser(policy=policy.default).parsebytes(raw)
    part = message.get_body(preferencelist=("plain", "html"))
    body = ""
    if part is not None:
        payload = part.get_payload(decode=True)
        if payload is not None:
            try:
                body = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
            except LookupError:
                body = payload.decode("utf-8", errors="replace")
        if part.get_content_type() == "text/html":
            parser = VisibleText()
            parser.feed(body)
            body = " ".join("".join(parser.parts).split())
    subject = str(message.get("Subject", ""))
    text = f"Subject: {subject}\n\n{body}"[:8192]
    keys = {"input:" + canonical(text)}
    if canonical(body):
        keys.add("body:" + canonical(body))
    # Conservative thread/template grouping uses metadata only to prevent leakage.
    subject_key = re.sub(r"\b(?:re|fwd?)\s*:\s*", "", subject.casefold())
    subject_key = re.sub(r"\d+", "#", subject_key)
    subject_words = re.findall(r"\w+", subject_key)
    if len(subject_words) >= 3:
        keys.add("subject:" + " ".join(subject_words))
    body_words = re.findall(r"\w+", re.sub(r"\d+", "#", body.casefold()))
    if len(body_words) >= 80:
        keys.add("prefix:" + " ".join(body_words[:80]))
    for header in ("Message-ID", "References", "In-Reply-To"):
        for value in message.get_all(header, []):
            keys.update("message:" + item.casefold() for item in re.findall(r"<([^<>]+)>", str(value)))
    return text, {digest(key) for key in keys}


def split_groups(rows):
    """Keep connected threads/templates in one split and score one case per group."""
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners = {}
    for i, row in enumerate(rows):
        for key in sorted(row["keys"]):
            if key in owners:
                parents[find(i)] = find(owners[key])
            else:
                owners[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    splits = {name: [] for name in ("teach", "calibration", "evaluate")}
    excluded = Counter()
    for group in groups.values():
        if len({row["label"] for row in group}) != 1:
            excluded["conflicting_label_messages"] += len(group)
            continue
        originals = [row for row in group if row["collection"] == "original"]
        pool = originals or group
        excluded["later_messages_overlapping_original"] += len(group) - len(originals) if originals else 0
        excluded["additional_messages_in_same_group"] += len(pool) - 1
        row = min(pool, key=lambda r: r["id"])
        group_id = digest("\n".join(sorted(r["id"] for r in pool)))
        split = "evaluate" if not originals else (
            "calibration" if int(digest("system1-public-email-v1:" + group_id), 16) % 5 == 0 else "teach")
        splits[split].append({k: row[k] for k in ("id", "label", "text")} | {"group": group_id})
    for rows_in_split in splits.values():
        rows_in_split.sort(key=lambda row: row["id"])
    return splits, dict(excluded)


def representative_split(splits):
    """A separate, retrospective experiment with both collections in each split."""
    mixed = {name: [] for name in ("teach", "calibration", "evaluate")}
    for rows in splits.values():
        for row in rows:
            bucket = int(digest("system1-public-email-representative-v1:" + row["group"]), 16) % 5
            split = "evaluate" if bucket == 0 else "calibration" if bucket == 1 else "teach"
            mixed[split].append(row)
    return {name: sorted(rows, key=lambda row: row["id"]) for name, rows in mixed.items()}


def prepare(protocol, directory, download=False):
    source_dir = directory / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    rows, counts = [], Counter()
    for source in protocol["sources"]:
        path = source_dir / source["file"]
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"{path}: rerun with --download")
            with urlopen(source["url"], timeout=60) as response:
                raw = response.read(source["bytes"] + 1)
            if len(raw) != source["bytes"] or digest(raw) != source["sha256"]:
                raise ValueError(f"Download checksum mismatch: {path.name}")
            path.write_bytes(raw)
        raw = path.read_bytes()
        if len(raw) != source["bytes"] or digest(raw) != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path.name}")
        if not path.name.endswith(".tar.bz2"):
            continue
        # Read archive members in memory; never extract paths or execute mail content.
        with tarfile.open(path, "r:bz2") as archive:
            for member in sorted(archive.getmembers(), key=lambda m: m.name):
                if not member.isfile() or not re.fullmatch(r"\d+\.[0-9a-f]+", Path(member.name).name):
                    continue
                if member.size > 2 * 1024 * 1024:
                    raise ValueError("Unexpected oversized corpus message")
                with archive.extractfile(member) as stream:
                    text, keys = parse_message(stream.read())
                rows.append({"id": path.name + ":" + member.name, "text": text, "keys": keys,
                             "label": source["label"], "collection": source["collection"]})
                counts[path.name] += 1
    splits, excluded = split_groups(rows)
    if protocol.get("split_mode") == "representative":
        splits = representative_split(splits)
    public = {"source_counts": dict(counts), "excluded": excluded,
              "counts": {name: dict(Counter(r["label"] for r in items)) for name, items in splits.items()},
              "splits": {name: [{k: row[k] for k in ("id", "group", "label")} |
                                {"text_sha256": digest(row["text"])} for row in items]
                         for name, items in splits.items()}}
    encoded = (json.dumps(public, indent=2) + "\n").encode()
    (directory / "splits.json").write_bytes(encoded)
    if protocol["prepared_sha256"] and digest(encoded) != protocol["prepared_sha256"]:
        raise ValueError("Prepared split changed from the frozen protocol")
    (directory / "lessons.json").write_text(json.dumps(splits, indent=2) + "\n")
    return splits, public, digest(encoded)


def summarize(rows):
    result = quality_metrics(rows)
    result["accepted_errors"] = result["accepted"] - result["accepted_correct"]
    result["raw_accuracy"] = result["correct"] / len(rows)
    result["per_class"] = {}
    for label in ("ham", "spam"):
        actual = [r for r in rows if r["label"] == label]
        accepted = [r for r in rows if r["prediction"] == label and not r["needs_review"]]
        result["per_class"][label] = {
            "cases": len(actual), "raw_correct": sum(r["prediction"] == label for r in actual),
            "accepted_predictions": len(accepted), "accepted_correct": sum(r["label"] == label for r in accepted)}
    return result


def timing(values):
    return {"median_ms": statistics.median(values), "p95_ms": float(np.quantile(values, .95)),
            "calls": len(values)}


def no_network(*args, **kwargs):
    raise AssertionError("Teaching and decisions must remain offline")


def evaluate(data, protocol, directory):
    if not protocol["prepared_sha256"]:
        raise ValueError("Freeze prepared_sha256 in the protocol before evaluation")
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    config = protocol["candidate"]
    texts = [row["text"] for row in data["teach"]]
    labels = [row["label"] for row in data["teach"]]
    calibration = [(row["text"], row["label"]) for row in data["calibration"]]
    with patch.object(socket.socket, "connect", no_network), patch.object(socket, "create_connection", no_network):
        start = time.perf_counter()
        projector = TfidfProjector.fit(texts, max_features=config["max_features"])
        compiler = SystemOneCompiler(SpamDecision, projector=projector,
                                     regularization=config["regularization"], backend=config["backend"])
        skill = compiler.compile({"category": list(zip(texts, labels))}, augment=False,
                                 calibration_exemplars={"category": calibration})
        teaching_ms = (time.perf_counter() - start) * 1000
        model_path = directory / "spam.s1m"
        skill.save(model_path)
        start = time.perf_counter()
        restored = CompiledSystemOneModel.load(model_path)
        load_ms = (time.perf_counter() - start) * 1000
        engines = []
        for model in (skill, restored):
            model.use_cache = False
            engines.append(System1Engine(SpamDecision, model=model, strict_mode=True, use_cache=False))
        before, engine = engines
        baseline = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, vocabulary=projector.to_config()["terms"]),
            LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"))
        start = time.perf_counter()
        baseline.fit(texts, labels)
        # Mirror the compiler's actual temperature/conformal split.
        _, conformal = compiler._split_samples(calibration, None, .5)
        probabilities = baseline.predict_proba([text for text, _ in conformal])
        classes = list(baseline.classes_)
        scores = sorted(1 - probabilities[i, classes.index(label)] for i, (_, label) in enumerate(conformal))
        rank = math.ceil((len(scores) + 1) * (1 - config["alpha"]))
        quantile = scores[rank - 1] if rank <= len(scores) else 1.0
        baseline_ms = (time.perf_counter() - start) * 1000
        np.testing.assert_allclose(projector.project_batch([t for t, _ in calibration]),
                                   baseline[0].transform([t for t, _ in calibration]).toarray(), atol=1e-6, rtol=1e-6)

        def decide(text):
            return engine.decide(text, alpha=config["alpha"], record_receipt=False)

        def logistic(text):
            vector = baseline[0].transform([text])
            probabilities = baseline[1].predict_proba(vector)[0]
            prediction_set = [label for label, p in zip(classes, probabilities) if 1 - p <= quantile]
            # Match the product's zero-known-feature review behavior.
            review = len(prediction_set) != 1 or vector.nnz == 0
            return classes[int(np.argmax(probabilities))], prediction_set, review

        start = time.perf_counter()
        decide(data["evaluate"][0]["text"])
        cold_ms = (time.perf_counter() - start) * 1000
        system_rows, baseline_rows = [], []
        for row in data["evaluate"]:
            result = decide(row["text"])
            original = before.decide(row["text"], alpha=config["alpha"], record_receipt=False)
            assert (result.values, result.probabilities, result.conformal_sets, result.is_ambiguous) == (
                original.values, original.probabilities, original.conformal_sets, original.is_ambiguous)
            common = {k: row[k] for k in ("id", "group", "label")}
            system_rows.append(common | {"prediction": result.values["category"],
                               "prediction_set": result.conformal_sets["category"], "needs_review": result.is_ambiguous})
            prediction, prediction_set, review = logistic(row["text"])
            baseline_rows.append(common | {"prediction": prediction, "prediction_set": prediction_set, "needs_review": review})
        for text in texts[:10]:
            decide(text)
            logistic(text)
        timings = {"system1": [], "tfidf_logistic": []}
        paths = [("system1", decide), ("tfidf_logistic", logistic)]
        for repeat in range(5):
            for row in data["evaluate"]:
                for name, function in paths[::1 if repeat % 2 == 0 else -1]:
                    start = time.perf_counter()
                    function(row["text"])
                    timings[name].append((time.perf_counter() - start) * 1000)
    report = {
        "protocol_sha256": digest(json.dumps(protocol, indent=2) + "\n"),
        "prepared_sha256": protocol["prepared_sha256"],
        "environment": {"system1": __version__, "python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "sklearn": sklearn.__version__},
        "teacher_calls": 0, "outbound_connections_during_teaching_and_evaluation": 0,
        "system1": {"quality": summarize(system_rows), "teaching_ms": teaching_ms,
                    "load_ms": load_ms, "first_decision_ms": cold_ms, "skill_bytes": model_path.stat().st_size,
                    "reload_identical": True, "timing": timing(timings["system1"]), "predictions": system_rows},
        "tfidf_logistic": {"quality": summarize(baseline_rows), "teaching_ms": baseline_ms,
                           "timing": timing(timings["tfidf_logistic"]), "predictions": baseline_rows},
        "inbox_zero_seven_category_rollout_ready": False, "limitations": protocol["limitations"]}
    (directory / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {key: value for key, value in report[name].items() if key != "predictions"}
                      for name in ("system1", "tfidf_logistic")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download missing checksum-pinned public archives")
    parser.add_argument("--prepare-only", action="store_true", help="Inspect split counts without teaching or scoring")
    parser.add_argument("--representative", action="store_true", help="Run the separate retrospective mixed-collection experiment")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.output = args.output or ROOT / ".system1" / ("public-email-representative" if args.representative else "public-email")
    protocol = json.loads((HERE / ("protocol_representative.json" if args.representative else "protocol.json")).read_text())
    data, public, prepared_hash = prepare(protocol, args.output, args.download)
    print(json.dumps({"counts": public["counts"], "excluded": public["excluded"], "prepared_sha256": prepared_hash}, indent=2))
    if not args.prepare_only:
        evaluate(data, protocol, args.output)


if __name__ == "__main__":
    main()
