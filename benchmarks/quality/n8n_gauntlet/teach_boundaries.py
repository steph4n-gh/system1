"""Prepare, record and replay the bounded experiment-2e teacher requests.

No inference evaluation here. Request geometry uses fitting rows only; other
sources contribute normalized hashes solely when rejecting duplicate lessons.
"""
import argparse
import csv
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = ROOT / ".system1/n8n-gauntlet"
PROTOCOL = HERE / "BOUNDARY_TEACHING_PROTOCOL.md"
REQUESTS = HERE / "boundary-teaching-requests.json"
RECORDS = OUTPUT / "boundary-teacher"
MODEL = "gemini-2.5-flash"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_new(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def committed(paths):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for path in paths:
        raw = subprocess.check_output(["git", "show", f"{revision}:{path.relative_to(ROOT)}"], cwd=ROOT)
        if raw != path.read_bytes():
            raise ValueError("Commit the exact procedure/requests before continuing")
    return revision


def prepare():
    import numpy as np
    from threadpoolctl import threadpool_limits
    from encoder_probe import Encoder
    from evaluate import deny_network_control
    from observed_regression import frozen
    from review_teaching import features

    revision = committed([PROTOCOL, Path(__file__).resolve()])
    frozen(False)
    denied = deny_network_control()
    manifest = json.loads((HERE / "manifest.json").read_text())
    report = dict(scope="experiment 2e generation requests; fitting context only", source_revision=revision,
        model=MODEL, max_calls=227, protocol_sha256=digest(PROTOCOL), code_sha256=digest(Path(__file__)),
        os_network_denial_errno=denied, datasets={}, requests=[])
    with threadpool_limits(limits=1):
        for dataset, name in (("clinc150", "minilm"), ("banking77", "bge-small")):
            source = OUTPUT / f"{dataset}-fit.json"
            expected = manifest["datasets"][dataset]["splits"]["fit"]
            if digest(source) != expected["sha256"]:
                raise ValueError("Fitting source changed")
            rows = json.loads(source.read_text())
            encoder = Encoder(name)
            preparation = []
            x = features(rows, encoder, single=True, evidence=preparation)
            labels = sorted({row["label"] for row in rows} - {"oos"})
            indices = {label: [i for i, row in enumerate(rows) if row["label"] == label] for label in labels}
            centers = np.stack([x[indices[label]].mean(axis=0) for label in labels])
            centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-9)
            similarity = centers @ centers.T
            np.fill_diagonal(similarity, -np.inf)
            oos = sorted([r for r in rows if r["label"] == "oos"], key=lambda r: r["group"])
            report["datasets"][dataset] = dict(fit_sha256=digest(source), fit_rows=len(rows), encoder=encoder.identity,
                preparation=preparation, supported_labels=labels)
            for position, label in enumerate(labels):
                nearby = sorted(range(len(labels)), key=lambda i: (-float(similarity[position, i]), labels[i]))[:3]
                ids = indices[label]
                margins = x[ids] @ centers[position] - (x[ids] @ centers[nearby].T).max(axis=1)
                chosen = sorted(range(len(ids)), key=lambda i: (float(margins[i]), rows[ids[i]]["group"]))[:5]
                target = [rows[ids[i]] for i in chosen]
                neighbors = {labels[i]: sorted([rows[j] for j in indices[labels[i]]], key=lambda r: r["group"])[:3] for i in nearby}
                context = dict(target=label, target_examples=[r["prompt"] for r in target],
                    neighboring_categories={k: [r["prompt"] for r in v] for k, v in neighbors.items()},
                    all_supported_categories=labels)
                negative_count = 8 if oos else 0
                negative_context = [oos[(position * 8 + i) % len(oos)] for i in range(8)] if oos else []
                if negative_context:
                    context["unsupported_fitting_examples"] = [r["prompt"] for r in negative_context]
                prompt = ("Create teaching examples for a bounded single-intent classifier, not answers to a test. "
                    "Write exactly 12 new, diverse user requests clearly belonging to the target category, "
                    "with details that distinguish it from the neighboring categories. Include short, indirect, "
                    "and everyday wording; do not copy examples or include category names. Do not put multiple "
                    "requests into one example. Return them as a supported array of strings. ")
                props = dict(supported=dict(type="ARRAY", items=dict(type="STRING")))
                required = ["supported"]
                if negative_count:
                    prompt += ("Also write exactly eight related requests that fit NONE of all_supported_categories. "
                        "Being outside just the target is insufficient. A request belonging to any other listed "
                        "category must NOT be marked unsupported. Use the unsupported fitting examples to understand "
                        "the boundary. Return an unsupported array of objects with text and a brief reason why no "
                        "supported category applies. Do not put the explanation in the request text. ")
                    props["unsupported"] = dict(type="ARRAY", items=dict(type="OBJECT", properties=dict(
                        text=dict(type="STRING"), reason=dict(type="STRING")), required=["text", "reason"]))
                    required.append("unsupported")
                payload = dict(contents=[dict(role="user", parts=[dict(text=prompt + "\n" + json.dumps(context, ensure_ascii=False))])],
                    generationConfig=dict(temperature=.7, maxOutputTokens=4096, thinkingConfig=dict(thinkingBudget=0),
                        responseMimeType="application/json", responseSchema=dict(type="OBJECT", properties=props, required=required)))
                report["requests"].append(dict(id=f"{dataset}-{label}", dataset=dataset, label=label,
                    supported_count=12, unsupported_count=negative_count,
                    context_groups=[r["group"] for r in target + [r for part in neighbors.values() for r in part] + negative_context],
                    request_sha256=payload_digest(payload), payload=payload))
            del encoder
            gc.collect()
    assert len(report["requests"]) == 227
    write_new(REQUESTS, report)
    print(f"Prepared {len(report['requests'])} requests; commit them before live generation", flush=True)


def checked_requests():
    committed([PROTOCOL, Path(__file__).resolve(), REQUESTS])
    report = json.loads(REQUESTS.read_text())
    if report["protocol_sha256"] != digest(PROTOCOL) or report["code_sha256"] != digest(Path(__file__)):
        raise ValueError("Generation procedure changed")
    if len(report["requests"]) != report["max_calls"] or report["max_calls"] != 227:
        raise ValueError("Declared call budget changed")
    ids = [r["id"] for r in report["requests"]]
    # BANKING77's original taxonomy includes `reverted_card_payment?`.
    if len(set(ids)) != 227 or any(not i.replace("_", "").replace("-", "").replace("?", "").isalnum() for i in ids):
        raise ValueError("Invalid request identities")
    for item in report["requests"]:
        if item["request_sha256"] != payload_digest(item["payload"]):
            raise ValueError("Request payload changed")
    return report


def issue(item, folder, key):
    """One durable attempt per immutable request, including transport failures."""
    attempt = folder / f"{item['id']}.json"
    result = folder / f"{item['id']}.result.json"
    if attempt.exists():
        if json.loads(attempt.read_text())["request_sha256"] != item["request_sha256"]:
            raise ValueError("Existing attempt has a different request")
        return None
    record = dict(id=item["id"], requested_model=MODEL, request_sha256=item["request_sha256"],
                  started_at=time.time(), status="started")
    write_new(attempt, record)
    started = time.perf_counter()
    request = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        data=json.dumps(item["payload"]).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            record["response"] = json.load(response)
        record["status"] = "received"
    except Exception as exc:
        record["status"] = "error"
        record["error_type"] = type(exc).__name__
        if isinstance(exc, urllib.error.HTTPError):
            record["http_status"] = exc.code
    record["latency_ms"] = (time.perf_counter() - started) * 1000
    write_new(result, record)
    return record


def live(limit):
    report = checked_requests()
    key = os.environ.get("GEMINI_API_KEY")
    if not key and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                key = line.partition("=")[2].strip().strip("\"'")
    if not key:
        raise ValueError("GEMINI_API_KEY is unavailable")
    RECORDS.mkdir(parents=True, exist_ok=True)
    issued = 0
    for item in report["requests"]:
        if issued >= limit:
            break
        record = issue(item, RECORDS, key)
        if record is None:
            continue
        issued += 1
        print(f"{item['id']}: {record['status']} ({issued} new calls)", flush=True)
        if record.get("http_status") in (401, 403, 429):
            raise RuntimeError("Teacher authorization/rate limit failure; attempt retained without retry")


def duplicate_groups():
    from prepare import group
    manifest = json.loads((HERE / "manifest.json").read_text())
    reserved, sources = set(), {}
    for dataset in ("clinc150", "banking77"):
        for split, expected in manifest["datasets"][dataset]["splits"].items():
            path = OUTPUT / f"{dataset}-{split}.json"
            if digest(path) != expected["sha256"]:
                raise ValueError("Original split changed")
            reserved.update(r["group"] for r in json.loads(path.read_text()))
            sources[path.name] = digest(path)
    path = HERE / "results/contrast-lessons.json"
    reserved.update(r["group"] for r in json.loads(path.read_text())["lessons"])
    sources[path.name] = digest(path)
    extra = json.loads((HERE / "review-data-manifest.json").read_text())
    for name, expected in extra["files"].items():
        path = OUTPUT / f"review-{name}.json"
        if digest(path) != expected["sha256"]:
            raise ValueError("Review teaching/reserve source changed")
        reserved.update(r["group"] for r in json.loads(path.read_text()))
        sources[path.name] = digest(path)
    path = OUTPUT / "clinc-human-paraphrases-reserve.csv"
    expected = json.loads((HERE / "results/human-paraphrase-source-audit.json").read_text())["source"]["sha256"]
    if digest(path) != expected:
        raise ValueError("Reserved paraphrases changed")
    reserved.update(group(r["text"]) for r in csv.DictReader(io.StringIO(path.read_text())))
    sources[path.name] = digest(path)
    return reserved, sources


def lessons_from(item, response, reserved):
    from prepare import group
    try:
        parts = response["candidates"][0]["content"]["parts"]
        parsed = json.loads("".join(p.get("text", "") for p in parts if not p.get("thought")))
        supported = parsed["supported"]
        unsupported = parsed.get("unsupported", [])
        if not isinstance(supported, list) or not isinstance(unsupported, list):
            raise ValueError("Expected arrays")
        if len(supported) != item["supported_count"] or len(unsupported) != item["unsupported_count"]:
            raise ValueError("Unexpected proposal count")
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return [], [dict(id=item["id"], reason="invalid_or_missing_response")]
    proposed = [(item["label"], text, None) for text in supported]
    proposed.extend(("oos", r.get("text") if isinstance(r, dict) else None,
                     r.get("reason") if isinstance(r, dict) else None) for r in unsupported)
    lessons, rejected = [], []
    for index, (label, text, reason) in enumerate(proposed):
        if not isinstance(text, str) or not text.strip() or len(text) > 2000 or (
                label == "oos" and (not isinstance(reason, str) or not reason.strip() or len(reason) > 2000)):
            rejected.append(dict(id=item["id"], index=index, reason="invalid_text_or_reason"))
            continue
        key = group(text)
        if key in reserved:
            rejected.append(dict(id=item["id"], index=index, group=key, reason="duplicate_group"))
            continue
        reserved.add(key)
        row = dict(prompt=text.strip(), label=label, group=key, dataset=item["dataset"],
            source="gemini-boundary-teaching-unverified", request_id=item["id"], request_sha256=item["request_sha256"])
        if reason is not None:
            row["teacher_reason"] = reason
        lessons.append(row)
    return lessons, rejected


def collect(folder):
    report = checked_requests()
    folder.mkdir(parents=True, exist_ok=False)
    reserved, sources = duplicate_groups()
    records, lessons, rejected, usage = [], [], [], {}
    for item in report["requests"]:
        attempt = RECORDS / f"{item['id']}.json"
        if not attempt.exists():
            continue
        path = RECORDS / f"{item['id']}.result.json"
        record = json.loads((path if path.exists() else attempt).read_text())
        if record["request_sha256"] != item["request_sha256"]:
            raise ValueError("Teacher evidence does not match frozen request")
        response = record.get("response", {})
        records.append(dict(record, dataset=item["dataset"], label=item["label"], payload=item["payload"]))
        metadata = response.get("usageMetadata", {}) if isinstance(response, dict) else {}
        for key, value in metadata.items():
            if type(value) is int:
                usage[key] = usage.get(key, 0) + value
        retained, failures = lessons_from(item, response, reserved)
        lessons.extend(retained)
        rejected.extend(failures)
    summary = dict(scope="experiment 2e unverified synthetic teaching, not independent quality evidence",
        request_manifest_sha256=digest(REQUESTS), protocol_sha256=digest(PROTOCOL),
        planned_calls=227, calls=len(records), responses=sum(r["status"] == "received" for r in records),
        unfinished_or_unattempted_slots=227 - sum(r["status"] != "started" for r in records),
        usage=usage, calls_without_usage=sum(not isinstance(r.get("response"), dict) or
            not r["response"].get("usageMetadata") for r in records),
        labels_independently_verified=False, excluded_source_hashes=sources,
        lessons={dataset: {kind: sum(r["dataset"] == dataset and (r["label"] == "oos") == (kind == "unsupported") for r in lessons)
                          for kind in ("supported", "unsupported")} for dataset in ("clinc150", "banking77")},
        rejections=rejected, api_latency_sum_ms=sum(r.get("latency_ms", 0) for r in records),
        published_price_estimate_for_recorded_usage_usd=(usage.get("promptTokenCount", 0) * .3 +
            (usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)) * 2.5) / 1e6,
        actual_billed_cost_usd=None, price_source=dict(url="https://ai.google.dev/gemini-api/docs/pricing#gemini-2.5-flash",
            checked="2026-09-21", input_usd_per_million=.3, output_usd_per_million=2.5,
            account_tier_and_bill="unavailable; missing usage is not zero cost"))
    write_new(folder / "teacher.json", records)
    write_new(folder / "lessons.json", dict(scope=summary["scope"], lessons=lessons))
    write_new(folder / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rejections", "excluded_source_hashes")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "live", "collect"))
    parser.add_argument("--limit", type=int, default=227)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.limit <= 227:
        parser.error("--limit must be between 1 and 227")
    if args.action == "prepare":
        prepare()
    elif args.action == "live":
        live(args.limit)
    elif args.output is None:
        parser.error("collect requires a fresh --output directory")
    else:
        collect(args.output)
