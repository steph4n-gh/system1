"""Pinned sources and group-disjoint folds; no model evaluation here."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / ".system1/n8n-gauntlet"
SOURCES = {
    "clinc.json": (
        "https://raw.githubusercontent.com/clinc/oos-eval/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_full.json",
        "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0",
    ),
    "banking-train.csv": (
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/train.csv",
        "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    ),
    "banking-test.csv": (
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/test.csv",
        "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    ),
}


def group(text):
    return hashlib.sha256(" ".join(re.findall(r"\w+", text.casefold())).encode()).hexdigest()


def rows(pairs, source):
    return [dict(prompt=text, label=label, group=group(text), source=source, index=i)
            for i, (text, label) in enumerate(pairs)]


def unique_earlier(items, reserved):
    groups = defaultdict(list)
    for item in items:
        groups[item["group"]].append(item)
    kept = []
    removed = dict(reserved_overlap=0, conflicting_label=0, duplicate=0)
    for digest, members in sorted(groups.items()):
        if digest in reserved:
            removed["reserved_overlap"] += len(members)
        elif len({r["label"] for r in members}) != 1:
            removed["conflicting_label"] += len(members)
        else:
            kept.append(members[0])
            removed["duplicate"] += len(members) - 1
    return kept, removed


def partition(training, test, development=None):
    """Reserve tests first; labels never select which official tests to retain."""
    reserved = {r["group"] for r in test}
    removed = {}
    if development is not None:
        development, removed["development"] = unique_earlier(development, reserved)
        reserved.update(r["group"] for r in development)
    training, removed["training"] = unique_earlier(training, reserved)
    by_label = defaultdict(list)
    for item in training:
        by_label[item["label"]].append(item)
    result = dict(fit=[], calibration=[], development=development or [], test=test)
    for label, members in sorted(by_label.items()):
        members.sort(key=lambda r: r["group"])
        check_count = 20 if development is not None else len(members) // 5
        cut = check_count if development is not None else 2 * check_count
        if len(members) <= cut:
            raise ValueError(f"Insufficient teaching rows for {label}")
        result["calibration"].extend(members[:check_count])
        if development is None:
            result["development"].extend(members[check_count:cut])
        result["fit"].extend(members[cut:])
    seen = set()
    for name, members in result.items():
        groups = {r["group"] for r in members}
        if seen & groups:
            raise ValueError(f"Group leakage into {name}")
        seen.update(groups)
    return {"splits": result, "removed": removed, "labels": sorted(by_label)}


def prepare():
    cache = ROOT / ".system1/workload-sources"
    cache.mkdir(parents=True, exist_ok=True)
    raw = {}
    for name, (url, digest) in SOURCES.items():
        path = cache / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Upstream source changed: {name}")
            path.write_bytes(data)
        raw[name] = path.read_bytes()
        if hashlib.sha256(raw[name]).hexdigest() != digest:
            raise ValueError(f"Source checksum mismatch: {name}")
    clinc = json.loads(raw["clinc.json"])
    clinc_rows = {key: rows(pairs, key) for key, pairs in clinc.items()}
    banking = {}
    for split in ("train", "test"):
        reader = csv.DictReader(io.StringIO(raw[f"banking-{split}.csv"].decode()))
        banking[split] = rows([(r["text"], r["category"]) for r in reader], split)
    datasets = {
        "clinc150": partition(clinc_rows["train"] + clinc_rows["oos_train"],
                              clinc_rows["test"] + clinc_rows["oos_test"],
                              clinc_rows["val"] + clinc_rows["oos_val"]),
        "banking77": partition(banking["train"], banking["test"]),
    }
    if len(datasets["clinc150"]["splits"]["test"]) != 5500 or len(banking["test"]) != 3080:
        raise ValueError("Official test counts changed")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {"sources": {name: {"url": url, "sha256": sha} for name, (url, sha) in SOURCES.items()},
                "datasets": {}}
    for name, dataset in datasets.items():
        paths = {}
        for split, members in dataset["splits"].items():
            path = OUTPUT / f"{name}-{split}.json"
            path.write_text(json.dumps(members, indent=2, ensure_ascii=False) + "\n")
            paths[split] = {"rows": len(members), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest["datasets"][name] = {"splits": paths, "labels": dataset["labels"], "removed": dataset["removed"]}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({name: {s: d["rows"] for s, d in info["splits"].items()}
                      for name, info in manifest["datasets"].items()}, indent=2))
    return manifest


if __name__ == "__main__":
    prepare()
