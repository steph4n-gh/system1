"""Bounded, recorded API lesson generation; see TEACHING_ADDENDUM.md."""
import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np
from threadpoolctl import threadpool_limits

from develop import OUTPUT, ROOT, load_splits
from encoder_probe import Encoder
from prepare import group

MODEL = "gemini-2.5-flash"
RECORDS = OUTPUT / "contrast-teacher"


def requests():
    data = load_splits("banking77")
    encoder = Encoder("bge-small")
    fit = data["fit"]
    key = hashlib.sha256((encoder.projector_digest() + json.dumps(fit, sort_keys=True)).encode()).hexdigest()
    x = np.load(OUTPUT / f"features-{key}.npy", allow_pickle=False)
    labels = sorted({r["label"] for r in fit})
    examples = {label: sorted([r for r in fit if r["label"] == label], key=lambda r: r["group"]) for label in labels}
    centers = np.stack([x[[r["label"] == label for r in fit]].mean(axis=0) for label in labels])
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    similarities = centers @ centers.T
    np.fill_diagonal(similarities, -np.inf)
    for i, label in enumerate(labels):
        nearby = [labels[j] for j in np.argsort(-similarities[i])[:3]]
        context = dict(target=label, target_examples=[r["prompt"] for r in examples[label][:5]],
                       neighboring_categories={other: [r["prompt"] for r in examples[other][:3]] for other in nearby})
        prompt = ("Write exactly 12 new, diverse bank-customer requests clearly belonging to the target category. "
                  "Use the neighboring categories to avoid ambiguous wording. Include short everyday requests, "
                  "indirect wording, and varied sentence lengths. Do not copy examples or mention category names. "
                  "Return JSON with an examples array of strings. These are teaching examples, not answers to a test.\n"
                  + json.dumps(context, ensure_ascii=False))
        payload = dict(contents=[dict(role="user", parts=[dict(text=prompt)])],
            generationConfig=dict(temperature=.7, maxOutputTokens=1536, thinkingConfig=dict(thinkingBudget=0),
                responseMimeType="application/json", responseSchema=dict(type="OBJECT",
                    properties=dict(examples=dict(type="ARRAY", items=dict(type="STRING"))), required=["examples"])))
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        yield label, payload, digest


def main(live=False):
    RECORDS.mkdir(parents=True, exist_ok=True)
    if live and not os.environ.get("GEMINI_API_KEY"):
        # Parse just this credential; never execute the file or print values.
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                os.environ["GEMINI_API_KEY"] = line.partition("=")[2].strip().strip("\"'")
    with threadpool_limits(limits=1):
        for label, payload, digest in requests():
            path = RECORDS / f"{label}.json"
            if path.exists():
                if json.loads(path.read_text())["request_sha256"] != digest:
                    raise ValueError("Existing teacher request differs; do not overwrite its evidence")
                continue
            if not live:
                raise ValueError("Missing teacher evidence; use --live for the declared bounded run")
            record = dict(label=label, requested_model=MODEL, request_sha256=digest, payload=payload,
                          started_at=time.time(), status="started")
            path.write_text(json.dumps(record, indent=2) + "\n")
            start = time.perf_counter()
            request = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
                data=json.dumps(payload).encode(), headers={"Content-Type": "application/json",
                    "x-goog-api-key": os.environ["GEMINI_API_KEY"]})
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    record["response"] = json.load(response)
                record["status"] = "received"
            except Exception as exc:
                # Exception messages can contain transport secrets. Record type/status only.
                record["status"] = "error"
                record["error_type"] = type(exc).__name__
                if isinstance(exc, urllib.error.HTTPError):
                    record["http_status"] = exc.code
            record["latency_ms"] = (time.perf_counter() - start) * 1000
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
            print(f"{label}: {record['status']}", flush=True)
            if record.get("http_status") in (401, 403, 429):
                raise RuntimeError("Teacher authorization or rate limit failure; request evidence retained")

    # Test rows participate ONLY through their normalized-group hashes.
    manifest = json.loads((ROOT / "benchmarks/quality/n8n_gauntlet/manifest.json").read_text())
    reserved = set()
    for split in ("fit", "calibration", "development", "test"):
        raw = (OUTPUT / f"banking77-{split}.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["datasets"]["banking77"]["splits"][split]["sha256"]:
            raise ValueError("Split hash changed")
        reserved.update(r["group"] for r in json.loads(raw))
    lessons, rejected, usage = [], [], {}
    records = [json.loads(p.read_text()) for p in sorted(RECORDS.glob("*.json"))]
    for record in records:
        response = record.get("response", {})
        for key, value in response.get("usageMetadata", {}).items():
            if isinstance(value, int):
                usage[key] = usage.get(key, 0) + value
        try:
            parts = response["candidates"][0]["content"]["parts"]
            texts = json.loads("".join(p.get("text", "") for p in parts if not p.get("thought")))["examples"]
            if not isinstance(texts, list) or len(texts) != 12:
                raise ValueError("Expected twelve examples")
        except (KeyError, IndexError, TypeError, ValueError):
            rejected.append(dict(label=record["label"], reason="invalid_or_missing_response"))
            continue
        for text in texts:
            if not isinstance(text, str) or not text.strip() or len(text) > 2000:
                rejected.append(dict(label=record["label"], reason="invalid_text"))
                continue
            digest = group(text)
            if digest in reserved:
                rejected.append(dict(label=record["label"], reason="duplicate_group", group=digest))
                continue
            reserved.add(digest)
            lessons.append(dict(prompt=text.strip(), label=record["label"], group=digest,
                                source="gemini-generated-contrast", request_sha256=record["request_sha256"]))
    result = dict(scope="synthetic teaching; no blind teacher accuracy measured", calls=len(records),
                  responses=sum(r["status"] == "received" for r in records), usage=usage,
                  lessons=lessons, rejected=rejected, actual_billed_cost_usd=None)
    (OUTPUT / "contrast-lessons.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("lessons", "rejected")}, indent=2))
    print(f"Retained {len(lessons)} lessons; {len(rejected)} rejected entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    main(parser.parse_args().live)
